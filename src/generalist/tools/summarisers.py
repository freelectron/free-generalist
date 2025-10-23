from dataclasses import dataclass
import json
import regex as re

from .data_model import ShortAnswer
from ..models.core import llm


def construct_short_answer(task: str, context: str) -> ShortAnswer:
    """Summarizes a context and provides a structured short answer to the task.

    This function sends the task and context to an LLM, which synthesizes 
    the information and return a shorten result.

    Note:
        This function requires a configured LLM client, represented here as `llm`.

    Args:
        task: The specific question or instruction to be performed.
        context: A string containing the text/information to be analyzed.

    Returns:
        A ShortAnswer dataclass instance containing the answer and clarification.
    """
    prompt = f"""
    You are presented with a list of expert answers from different sources that you need to summarize.

    LIST:
    {context}

    Based **ONLY** on that list and without any additional assumptions from your side, perform the task specified (or answer the question).

    TASK:
    {task}

    Your answer should be in a valid JSON format like so:
    {{
        "answer": "<a single number, word, or phrase which is the answer to the question>",
        "clarification": "<a very short mention of what the answer is based on, **always relate it back to the question**>"
    }}

    Rules:
        - If the text contains the complete answer → put the exact answer in "answer".
        - If the text contains no relevant information → put "answer": "not found".
        - If the text contains some but not all information → put "answer": "not found".
        - The "clarification" must mention the relevant part of the text and explain briefly the reasoning based on the task/question.
    """

    llm_response = llm.complete(prompt)
    response_text = llm_response.text
    response_text = response_text.strip()

    json_match = re.search(r"```json\n(.*?)```", response_text, re.DOTALL)
    code_string = json_match.group(1) if json_match else ""
    if len(code_string) > 1: 
        response_text = code_string

    data = json.loads(response_text)
    return ShortAnswer(
        answer=data.get("answer", "did-not-parse"),
        clarification=data.get("clarification", "did-not-parse.")
    )