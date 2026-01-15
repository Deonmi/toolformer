# sampling.py

from dataclasses import dataclass
from typing import List, Dict, Tuple
import re

from config import TAU_S, TOP_K_POS, MAX_CALLS_PER_POS, API_START_TOKEN
from llm_client import chat_with_logprobs, chat_text_only
from tokenizer_utils import simple_tokenize, simple_detokenize
from prompts import (
    QA_PROMPT_PREFIX, 
    CALCULATOR_PROMPT_PREFIX, 
    WIKISEARCH_PROMPT_PREFIX, 
    MT_PROMPT_PREFIX, 
    CALENDAR_PROMPT_PREFIX
)

def get_token_prob_from_logprobs(choice) -> float:
    """
    Read the probability of API_START_TOKEN at the first generated token position from choice.logprobs.
    Assuming the API return structure is roughly as follows:
    choice.logprobs.content[0].top_logprobs -> list[{token, logprob}, ...]
    """
    content_logprobs = choice.logprobs.content
    if not content_logprobs:
        return 0.0

    first_pos = content_logprobs[0]
    # first_pos.top_logprobs: list of objects with .token, .logprob
    for cand in first_pos.top_logprobs:
        if cand.token == API_START_TOKEN:
            import math
            return math.exp(cand.logprob)  # log probability -> probability

    # If not in top_logprobs, assume probability is very low
    return 0.0


def build_next_token_prompt(task_prefix: str, prefix_text: str) -> str:
    """
    Given P(x) + prefix, ask "what is the next token?" for reading logprobs.
    """
    return task_prefix.format(input_text=prefix_text)


def compute_position_prob(
    task_prefix: str,
    x_tokens: List[str],
    pos: int,
) -> float:
    """
    In the corresponding paper, p_i = p_M(<API> | P(x), x_1..x_{i-1}) is given. [file:1]
    Here, we use "chat+logprobs" as an approximation: we ask the model to predict the next token and read the probability of the API_START_TOKEN.
    """
    prefix = simple_detokenize(x_tokens[:pos-1])
    prompt = build_next_token_prompt(task_prefix, prefix)
    resp = chat_with_logprobs(prompt)
    choice = resp.choices[0]
    p_i = get_token_prob_from_logprobs(choice)
    return p_i


def sample_candidate_positions_for_text(
    x_text: str,
    task_prefix: str = QA_PROMPT_PREFIX,
    tau_s: float = TAU_S,
    k: int = TOP_K_POS,
) -> List[int]:
    """
    Input a text x, and output the set of positions I (1-based).
    """
    x_tokens = simple_tokenize(x_text)
    n = len(x_tokens)
    scored: List[Tuple[float, int]] = []

    for i in range(1, n + 1):
        p_i = compute_position_prob(task_prefix, x_tokens, i)
        if p_i > tau_s:
            scored.append((p_i, i))

    scored.sort(key=lambda x: x[0], reverse=True)
    I = [i for (p, i) in scored[:k]]
    return I

API_CALL_PATTERN = re.compile(r'\[(QA|Calculator|WikiSearch|MT|Calendar)\((.*?)\)\]')

@dataclass
class SampledCall:
    api_name: str
    api_input: str
    position: int  # 1-based index in x_tokens


def build_annotation_prompt_for_position(
    task_prefix: str,
    x_before: str,
    x_after: str,
) -> str:
    """
    At the specified position i, split the text into two parts A and B, and let the model insert an API call at the end of A.
    """
    return f"""{task_prefix}
You are now given a text split into two parts. Insert at most ONE API call
at the end of Part A, following the examples above.

Part A:
{x_before}

Part B:
{x_after}

Output the combined text with exactly one API call inserted at the boundary if useful.
"""


def extract_api_calls(text: str) -> List[SampledCall]:
    calls = []
    for m in API_CALL_PATTERN.finditer(text):
        api_name = m.group(1)
        api_input = m.group(2)
        calls.append(SampledCall(api_name=api_name, api_input=api_input, position=-1))
    return calls


def sample_api_calls_at_positions(
    x_text: str,
    positions: List[int],
    task_prefix: str = QA_PROMPT_PREFIX,
    m: int = MAX_CALLS_PER_POS,
) -> Dict[int, List[SampledCall]]:
    """
    For each position i ∈ positions, call the chat interface once to generate text containing API calls, and parse out the calls.
    """
    x_tokens = simple_tokenize(x_text)
    n = len(x_tokens)
    result: Dict[int, List[SampledCall]] = {}

    for i in positions:
        x_before = simple_detokenize(x_tokens[:i-1])
        x_after = simple_detokenize(x_tokens[i-1:])

        prompt = build_annotation_prompt_for_position(task_prefix, x_before, x_after)
        output = chat_text_only(prompt, temperature=0.7, max_tokens=256)

        calls = extract_api_calls(output)
        if not calls:
            continue

        # Take at most m elements and record their positions i
        picked = []
        for c in calls[:m]:
            c.position = i
            picked.append(c)
        result[i] = picked

    return result


def sampling_phase_for_text(
    x_text: str,
    task_prefix: str = QA_PROMPT_PREFIX,
    tau_s: float = TAU_S,
    k: int = TOP_K_POS,
    m: int = MAX_CALLS_PER_POS,
) -> Tuple[List[int], Dict[int, List[SampledCall]]]:
    """
    Complete sampling stage:
      1) Calculate p_i and select I
      2) Sample up to m API calls at each i
    """
    I = sample_candidate_positions_for_text(
        x_text=x_text,
        task_prefix=task_prefix,
        tau_s=tau_s,
        k=k,
    )

    calls_by_pos = sample_api_calls_at_positions(
        x_text=x_text,
        positions=I,
        task_prefix=task_prefix,
        m=m,
    )

    return I, calls_by_pos
