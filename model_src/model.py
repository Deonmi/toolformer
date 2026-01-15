import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from einops import rearrange, repeat
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional, Union, Dict, Any, Callable
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM
from torch.utils.tensorboard import SummaryWriter

import pandas as pd
from pathlib import Path
import json
import os
import numpy as np
import argparse

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, max_seq_len: int, device: torch.device) -> torch.Tensor:
        t = torch.arange(max_seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[None, :, :], emb.sin()[None, :, :]

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_emb(q: torch.Tensor, k: torch.Tensor, 
                    cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class SwiGLU(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(dim, dim * 2)
        self.w2 = nn.Linear(dim, dim)
        self.w3 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.w1(x).chunk(2, dim=-1)
        x = F.silu(x) * gate
        return self.dropout(self.w2(x))

class GroupedQueryAttention(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int = 128, 
                 num_kv_groups: Optional[int] = None, dropout: float = 0.):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.num_heads = heads
        self.num_kv_groups = num_kv_groups or heads
        self.num_kv_heads = heads // self.num_kv_groups
        
        inner_dim = head_dim * heads
        self.q_proj = nn.Linear(dim, inner_dim, bias=False)
        self.k_proj = nn.Linear(dim, head_dim * self.num_kv_groups, bias=False)
        self.v_proj = nn.Linear(dim, head_dim * self.num_kv_groups, bias=False)
        self.o_proj = nn.Linear(inner_dim, dim, bias=False)
        
        self.scale = head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, 
                cos: torch.Tensor, sin: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, n, _ = x.shape
        
        q = self.q_proj(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, n, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, n, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # RoPE
        q, k = apply_rotary_emb(q, k, cos, sin)
        
        # 重用 KV heads
        k = repeat(k, 'b kh n d -> b (kh rh) n d', rh=self.num_heads//self.num_kv_heads)
        v = repeat(v, 'b kh n d -> b (kh rh) n d', rh=self.num_heads//self.num_kv_heads)
        
        # 注意力计算
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask[:, None, :, :], float('-inf'))
            
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        out = attn @ v
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.o_proj(out)

class AdvancedTransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 8, ff_mult: int = 8/3, 
                 num_kv_groups: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        self.attn_norm = nn.RMSNorm(dim)
        self.ffn_norm = nn.RMSNorm(dim)
        
        self.attn = GroupedQueryAttention(
            dim, heads, dim//heads, num_kv_groups, dropout
        )
        self.ffn = nn.Sequential(
            SwiGLU(dim, dropout),
            nn.Linear(dim//2, dim) 
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-Attention + Residual
        normed_x = self.attn_norm(x)
        attn_out = self.attn(normed_x, cos, sin, attn_mask)
        x = x + self.dropout(attn_out)
        
        # FeedForward + Residual
        normed_x = self.ffn_norm(x)
        ffn_out = self.ffn(normed_x)
        x = x + self.dropout(ffn_out)
        
        return x

class UnifiedToolformer(nn.Module):
    """统一 Toolformer，支持自实现和预训练模型"""
    
    MODEL_REGISTRY = {}
    
    def __init__(self, 
                 model_type: str = "scratch",  # "scratch" | "llama" | "gpt2"
                 config: Optional[Any] = None,
                 pretrained_path: Optional[str] = None,
                 tokenizer: Optional[Any] = None):
        super().__init__()
        
        self.model_type = model_type
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad_token_id

        if model_type == "scratch":
            self._build_from_scratch(config)
        elif model_type in self.MODEL_REGISTRY:
            self.model = self.MODEL_REGISTRY[model_type](config, pretrained_path)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
    
    def _build_from_scratch(self, config):
        self.config = config
        
        self.token_emb = nn.Embedding(config.vocab_size, config.dim)
        self.rope_emb = RotaryEmbedding(
            config.dim//config.num_heads, 
            config.max_seq_len, 
            config.rope_theta
        )
        
        self.layers = nn.ModuleList([
            AdvancedTransformerBlock(
                config.dim, config.num_heads, 
                num_kv_groups=config.num_kv_groups
            ) for _ in range(config.num_layers)
        ])
        
        self.norm = nn.RMSNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    @classmethod
    def register_model(cls, name: str, model_cls: Callable):
        cls.MODEL_REGISTRY[name] = model_cls
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        if self.model_type == "scratch":
            return self._forward_scratch(input_ids, attention_mask)
        else:
            return self.model(input_ids, attention_mask=attention_mask, **kwargs)
    
    def _forward_scratch(self, input_ids: torch.Tensor, 
                        attention_mask: Optional[torch.Tensor] = None):
        b, n = input_ids.shape
        device = input_ids.device
        
        x = self.token_emb(input_ids)
        
        cos, sin = self.rope_emb(n, device)
        
        if attention_mask is None:
            mask = torch.full((n, n), float('-inf'), device=device)
            mask = torch.triu(mask, diagonal=1)
        else:
            mask = torch.full((n, n), 0.0, device=device)
            mask.masked_fill_(~attention_mask.bool(), float('-inf'))
            mask = torch.triu(mask, diagonal=1)
        
        for layer in self.layers:
            x = layer(x, cos[:n], sin[:n], mask)
        
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return {'logits': logits}

    def set_special_tokens(
        self,
        api_start_token: str = "<API>",
        api_end_token: str = "</API>",
        api_result_token: str = "￫", 
    ):
        self.api_start_token = api_start_token
        self.api_end_token = api_end_token
        self.api_result_token = api_result_token

        self.api_start_id = self.tokenizer.convert_tokens_to_ids(api_start_token)
        self.api_end_id = self.tokenizer.convert_tokens_to_ids(api_end_token)
        self.api_result_id = self.tokenizer.convert_tokens_to_ids(api_result_token)

    @torch.no_grad()
    def generate_with_tools(
        self,
        prompt: str,
        api_router: callable,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop_ids: Optional[list] = None,
        device: Optional[torch.device] = None,
    ) -> str:
        self.eval()
        device = device or next(self.parameters()).device

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(device)
        generated = input_ids.clone()  # (1, L)

        for _ in range(max_new_tokens):
            outputs = self.forward(generated)
            logits = outputs["logits"][:, -1, :] 
            logits = logits / max(temperature, 1e-6)

            probs = F.softmax(logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            cutoff = cumulative_probs > top_p
            cutoff[..., 1:] = cutoff[..., :-1].clone()
            cutoff[..., 0] = False
            sorted_probs[cutoff] = 0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            next_token = sorted_indices[0, torch.multinomial(sorted_probs[0], num_samples=1)]

            generated = torch.cat([generated, next_token.view(1, 1)], dim=-1)

            if next_token.item() == self.api_result_id:
                text_so_far = self.tokenizer.decode(generated[0], skip_special_tokens=False)
                api_name, api_input = self._extract_last_api_call(text_so_far)

                if api_name is not None:
                    api_result = api_router(api_name, api_input)  # str

                    insert_text = f" {api_result} {self.api_end_token}"
                    insert_ids = self.tokenizer.encode(insert_text, add_special_tokens=False)
                    insert_ids = torch.tensor(insert_ids, dtype=torch.long, device=device).unsqueeze(0)

                    generated = torch.cat([generated, insert_ids], dim=-1)

            if stop_ids is not None and next_token.item() in stop_ids:
                break

        return self.tokenizer.decode(generated[0], skip_special_tokens=True)

    def _extract_last_api_call(self, text: str):
        try:
            start_idx = text.rfind(self.api_start_token)
            if start_idx == -1:
                return None, None

            sub = text[start_idx + len(self.api_start_token):]

            sub = sub.strip()
            lparen = sub.find("(")
            rparen = sub.rfind(")")
            if lparen == -1 or rparen == -1 or rparen <= lparen:
                return None, None

            api_name = sub[:lparen].strip()
            api_input = sub[lparen + 1:rparen].strip()

            if (api_input.startswith('"') and api_input.endswith('"')) or \
               (api_input.startswith("'") and api_input.endswith("'")):
                api_input = api_input[1:-1]

            return api_name, api_input
        except Exception:
            return None, None
