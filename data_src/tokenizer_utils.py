# tokenizer_utils.py

from typing import List
from transformers import AutoTokenizer

def simple_tokenize(text: str) -> List[str]:
    return text.split()

def simple_detokenize(tokens: List[str]) -> str:
    return " ".join(tokens)

# Load the tokenizer of the large model for encoding and decoding
# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("gpt2")

def tokenize(tokenizer: AutoTokenizer, text: str) -> List[int]:
    return tokenizer.encode(text)
    
def detokenize(tokenizer: AutoTokenizer, tokens: List[str]) -> str:
    return tokenizer.decode(tokens)
