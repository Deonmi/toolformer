# llm_client.py

from typing import List, Dict, Any
from config import MODEL_NAME, TOP_LOGPROBS
import openai
from openai import OpenAI

# openai.base_url = "https://api.openai.com/v1"
# openai.api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(
    # api_key=OPENAI_API_KEY
    # base_url="https://api.openai.com/v1"
)
def chat_with_logprobs(prompt: str) -> Dict[str, Any]:
    """
    Unify the encapsulation of the chat.completions.create call and enable logprobs.
    Return the original completion object (dict / pydantic object).
    """
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=64,
        n=1,
        stream=False,
        timeout=60,
        logprobs=True,
        top_logprobs=TOP_LOGPROBS
    )
    return resp


def chat_text_only(prompt: str, temperature: float = 0.7, max_tokens: int = 512) -> str:
    """
    A regular dialogue call that does not require logprobs, used to generate annotated text with [QA(...)].
    """
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        n=1,
        stream=False,
        timeout=60,
    )
    return resp.choices[0].message.content.strip()
