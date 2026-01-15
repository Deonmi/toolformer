
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Callable, Dict, Any
import re
from dataclasses import dataclass

from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM
from model import UnifiedToolformer


@dataclass
class ScratchModelConfig:
    dim: int
    num_layers: int
    num_heads: int
    vocab_size: int
    max_seq_len: int
    num_kv_groups: int

def load_toolformer_model(model_type: str, checkpoint_path: str):
    """
    Load the trained Toolformer model
    Args:
        checkpoint_path: Path to the .pth file saved during training
        config: inference configuration
    Returns:
        model: Loaded UnifiedToolformer
        tokenizer: The corresponding tokenizer
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    
    if model_type == "scratch":
        model_config = ScratchModelConfig(
            dim=512, num_layers=6, num_heads=8, vocab_size=50257,
            max_seq_len=1024, num_kv_groups=2
        )
        model = UnifiedToolformer("scratch", model_config, tokenizer=tokenizer)
    else:
        model_config = checkpoint['model_config']
        model = UnifiedToolformer(model_type, model_config, tokenizer=tokenizer)

    model.set_special_tokens(
        api_start_token="[",
        api_end_token="]",
        api_result_token="→",
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model

def api_router(api_name: str, api_input: str) -> str:
    if api_name.lower() == "qa":
        return f"[ANSWER_TO: {api_input}]"
    elif api_name.lower() == "calculator":
        try:
            return str(eval(api_input))
        except Exception:
            return "[CALC_ERROR]"
    else:
        return "[UNKNOWN_API]"

def demo_api_router(api_name: str, api_input: str) -> str:
    if api_name.lower() == "qa":
       # Simulate QA system
        answers = {
            "What is BlackInAmerica.com?": "An online community for African Americans",
            "Where is Alcorn State University located?": "Lorman, Mississippi",
            "What is the capital of France?": "Paris"
        }
        return answers.get(api_input, f"[QA: {api_input}]")
    
    elif api_name.lower() == "calculator":
        try:
            result = eval(api_input, {"__builtins__": {}}, {})
            return f"{result:.2f}"
        except:
            return "[CALC_ERROR]"
    
    elif api_name.lower() == "calendar":
        return "Today is Thursday, January 08, 2026"
    
    else:
        return f"[UNKNOWN_API: {api_name}]"

def toolformer_inference(
    model: UnifiedToolformer,
    tokenizer: any,
    prompt: str,
    api_router: Callable = demo_api_router,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    stop_strings: Optional[list] = None
):
    """
    Complete reasoning process of Toolformer
    Args:
        model: Loaded UnifiedToolformer
        tokenizer: tokenizer
        prompt: input prompt
        api_router: API routing function
        max_new_tokens: Maximum generated length
        temperature, top_p: sampling parameters
        stop_strings: List of stop strings
    """
    with torch.no_grad():
        output = model.generate_with_tools(
            prompt=prompt,
            api_router=api_router,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_ids=[tokenizer.eos_token_id],
        )
    
    print(f"Toolformer Output:\n{output}")
    return output

def main():
    CHECKPOINT_PATH = "toolformer_checkpoints/final_model/toolformer_final.pth"
    model = load_toolformer_model(CHECKPOINT_PATH)
    
    test_cases = [
        "The capital of France is", # <API>QA(\"What is the capital of France?\") ￫
        "What is 123 + 456 - 78 * 2?", # <API>Calculator(\"123 + 456 - 78 * 2\") ￫
        "today is", # <API>Calendar() ￫
    ]
    
    for i, prompt in enumerate(test_cases, 1):
        print(f"\n{'='*100}")
        print(f"case {i}/{len(test_cases)}")
        toolformer_inference(
            model=model,
            tokenizer=model.tokenizer,
            prompt=prompt,
            api_router=demo_api_router,
            max_new_tokens=128,
            temperature=0.1
        )
    
    print("\n Toolformer Done！")

if __name__ == "__main__":
    main()
