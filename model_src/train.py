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

class ToolformerTrainer:
    def __init__(self, model: UnifiedToolformer, config):
        self.model = model
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=config.lr,
            weight_decay=config.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=1, 
            gamma=0.95
        )
       # TensorBoard
        self.writer = SummaryWriter(log_dir=config.log_dir)
        print(f'📊 TensorBoard: tensorboard --logdir={config.log_dir}')
        
        self.global_step = 0
        self.train_losses = []
        self.val_losses = []
        self.lrs = []
        
        self.save_dir = config.save_dir
        os.makedirs(self.save_dir, exist_ok=True)
    
    def _collate_fn(self, batch):
        """padding"""
        tokens = [item for item in batch]
        lengths = [len(t) for t in tokens]
        max_len = max(lengths)
        
        padded = torch.full((len(tokens), max_len), self.model.pad_id, dtype=torch.long)
        for i, t in enumerate(tokens):
            padded[i, :len(t)] = t
        return padded
    
    def _train_epoch(self, train_loader):
        self.model.train()
        epoch_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Train")):
            batch = batch.to(next(self.model.parameters()).device)
            inp, labels = batch[:, :-1], batch[:, 1:]
            
            outputs = self.model(inp)
            logits = rearrange(outputs['logits'], 'b n c -> b c n')
            loss = F.cross_entropy(logits, labels, ignore_index=self.model.pad_id)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            if self.global_step % 50 == 0:
                self.writer.add_scalar('Loss/Train', loss.item(), self.global_step)
                self.writer.add_scalar('LR', self.scheduler.get_last_lr()[0], self.global_step)
        
        return epoch_loss / num_batches
    
    @torch.no_grad()
    def _validate_epoch(self, val_loader):
        self.model.eval()
        val_loss = 0
        num_batches = 0
        
        for batch in tqdm(val_loader, desc="Val"):
            batch = batch.to(next(self.model.parameters()).device)
            inp, labels = batch[:, :-1], batch[:, 1:]
            
            outputs = self.model(inp)
            logits = rearrange(outputs['logits'], 'b n c -> b c n')
            loss = F.cross_entropy(logits, labels, ignore_index=self.model.pad_id)
            
            val_loss += loss.item()
            num_batches += 1
        
        return val_loss / num_batches
    
    def _log_metrics(self, epoch, train_loss, val_loss):
        self.writer.add_scalars('Loss', {
            'Train': train_loss,
            'Validation': val_loss
        }, epoch)
        
        self.writer.add_scalar('Epoch_Loss/Train', train_loss, epoch)
        self.writer.add_scalar('Epoch_Loss/Val', val_loss, epoch)
        self.writer.add_scalar('Learning_Rate', self.scheduler.get_last_lr()[0], epoch)
        
        # LR 调度器状态
        self.writer.add_scalar('Scheduler/Last_LR', self.scheduler.get_last_lr()[0], epoch)
    
    def _save_model_and_logs(self):
        save_path = Path(self.save_dir) / "final_model"
        save_path.mkdir(exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'epoch': self.model.finetune_config['epochs']
        }
        torch.save(checkpoint, save_path / "toolformer_final.pth")
        
        hparams = {
            'batch_size': self.model.finetune_config['batch_size'],
            'lr': self.model.finetune_config['lr'],
            'epochs': self.model.finetune_config['epochs'],
            'step_size': 1,
            'gamma': 0.95
        }
        self.writer.add_hparams(hparams, {
            'final/train_loss': self._train_epoch.__closure__[0].cell_contents if hasattr(self, '_train_epoch') else 0,
            'final/val_loss': self._validate_epoch.__closure__[0].cell_contents if hasattr(self, '_validate_epoch') else 0
        })
        

    def _train_loop(self, train_loader, val_loader):
        self.model.train()
        for epoch in range(self.model.finetune_config['epochs']):
            train_loss = self._train_epoch(train_loader)
            val_loss = self._validate_epoch(val_loader)
            self._log_metrics(epoch, train_loss, val_loss)
            self.scheduler.step()
            
            print(f"\nEpoch {epoch+1}/{self.model.finetune_config['epochs']}")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"LR: {self.scheduler.get_last_lr()[0]:.2e}")

    def train(self, tsv_path: str, tokenizer: any, 
                         val_split: float = 0.1,
                         max_length: int = 1024, max_samples: int = None):
        full_dataset = TSVFinetuneDataset(tsv_path, tokenizer, max_length, max_samples)
        train_size = int(len(full_dataset) * (1 - val_split))
        val_size = len(full_dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=self.model.finetune_config['batch_size'],
                                shuffle=True, num_workers=4, pin_memory=True,
                                collate_fn=self._collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=self.model.finetune_config['batch_size'],
                              shuffle=False, num_workers=4, pin_memory=True,
                              collate_fn=self._collate_fn)
        
        self._train_loop(train_loader, val_loader)
        
        self.writer.close()
        self._save_model_and_logs()

class TSVFinetuneDataset(Dataset):
    """
    Load the Toolformer enhanced dataset from the TSV file
    Format: original_text, enhanced_text
    Train using the enhanced_text column
    """
    def __init__(
        self, 
        tsv_path: str,
        tokenizer: any,
        max_length: int = 1024,
        max_samples: Optional[int] = None
    ):
        print(f"Loading TSV dataset: {tsv_path}")
        
        self.df = pd.read_csv(tsv_path, sep='\t', 
                            names=['original_text', 'enhanced_text'],
                            dtype={'original_text': str, 'enhanced_text': str})
        
        if max_samples:
            self.df = self.df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        
        print(f"Dataset statistics: {len(self.df)} samples")
        print(f"Maximum length limit: {max_length}")

        self.tokenizer = tokenizer
        self.max_length = max_length
        
        print("Tokenization...")
        self.encoded_data = self._preprocess()
    
    def _preprocess(self):
        """batch tokenization enhanced_text"""
        encoded = []
        
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Encoding"):
            text = row['enhanced_text']
            
            # Tokenize (supports special tokens <API>/[, </API>/], ->, etc.)
            tokens = self.tokenizer.encode(
                text,
                max_length=self.max_length,
                truncation=True,
                add_special_tokens=False
            )
            
            encoded.append(tokens)
        
        return encoded
    
    def __len__(self):
        return len(self.encoded_data)
    
    def __getitem__(self, idx):
        tokens = self.encoded_data[idx]
        return torch.tensor(tokens, dtype=torch.long)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv_path", type=str, default="toolformer_data.tsv")
    parser.add_argument("--model_type", type=str, default="llama")
    parser.add_argument("--pretrained_path", type=str, default="huggyllama/llama-7b")
    # val_split
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    # lr
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    # log_dir
    parser.add_argument("--log_dir", type=str, default="log")
    # save_dir
    parser.add_argument("--save_dir", type=str, default="model")

    # scrath config
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--rope_theta", type=float, default=500000.0)
    parser.add_argument("--num_kv_groups", type=int, default=2)

    args = parser.parse_args()
    MODEL_TYPE = args.model_type

    if MODEL_TYPE == "scratch":
        model = UnifiedToolformer("scratch", args)
        # GPT2 tokenizer (scratch)
        tokenizer = LlamaTokenizer.from_pretrained(args.pretrained_path)
        tokenizer.pad_token = tokenizer.eos_token
    else:
        def llama_adapter(config, pretrained_path: str):
            model = AutoModelForCausalLM.from_pretrained(pretrained_path)
            model.config.rope_theta = config.rope_theta
            return model

        UnifiedToolformer.register_model(MODEL_TYPE, llama_adapter)
        model = UnifiedToolformer(MODEL_TYPE, args)
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_path)
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"The parameter volume of mdoel: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    
    # === train ===
    trainer = ToolformerTrainer(model, args)
    trainer.train(
        tsv_path=args.tsv_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
        val_split=args.val_split
    )

if __name__ == "__main__":
    main()