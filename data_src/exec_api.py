import argparse
from llm_client import chat_text_only
from dataclasses import dataclass
from typing import Callable, Dict

@dataclass
class APICall:
    api_name: str   # "QA", "Calculator", "WikiSearch", "MT", "Calendar"
    api_input: str  # For QA, it refers to natural language questions


# ====== QA tool implementation: call chat_text_only ======

def execute_qa(question: str) -> str:
    """
    Use a large model to answer a simple factual question, as an implementation of a QA API.
    Question: For example, 'Where was Joe Biden born?''
    The return value is a single text sequence, which is directly written back to the Toolformer dataset as r_i.
    """
    # A short system command can be added here to remind the model to only output the answer itself
    prompt = (
        "Answer the following question briefly and factually. "
        "Reply with the answer only, without extra explanation.\n\n"
        f"Question: {question}"
    )
    answer = chat_text_only(prompt, temperature=0.0, max_tokens=64)
    return answer.strip()

def execute_calculator(expr: str) -> str:
    try:
        value = eval(expr, {"__builtins__": {}}, {})
        return str(value)
    except Exception as e:
        return f"Calculator error: {e}"

def execute_wikisearch(term: str) -> str:
    return f"Wikipedia snippet about {term}"

def execute_mt(text: str) -> str:
    return f"English translation of: {text}"

def execute_calendar(_: str = "") -> str:
    from datetime import datetime
    today = datetime.now().strftime("%A, %B %d, %Y")
    return f"Today is {today}."


API_EXECUTORS: Dict[str, Callable[[str], str]] = {
    "QA": execute_qa,
    "Calculator": execute_calculator,
    "WikiSearch": execute_wikisearch,
    "MT": execute_mt,
    "Calendar": execute_calendar,
}


def execute_api_call(call: APICall) -> str:
    if call.api_name not in API_EXECUTORS:
        return f"Unknown API: {call.api_name}"
    func = API_EXECUTORS[call.api_name]
    return func(call.api_input)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file_path', type=str, action='store_true', help='input file path')
    args = parser.parse_args()
    file_path = args.file_path

    # qa_call = APICall(api_name="QA", api_input="Where was Joe Biden born?")
    # r = execute_api_call(qa_call)
    # print(f"[QA(\"{qa_call.api_input}\")] -> {r}")

    # Read the pandas file and add a column of API call results
    df = pd.read_csv(file_path, sep='\t')
    result = []
    for i, row in df.iterrows():
        api_name = row['api_name']
        api_input = row['api_input']
        result.append(execute_api_call(api_name, api_input))

    df["api_result"] = result
    df.to_csv(file_path, sep='\t', index=False)

