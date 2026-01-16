import json
import regex as re

from .data_model import ShortAnswer
from ..models.core import llm
from ..utils import current_function
from clog import get_logger


logger = get_logger(__name__)


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
    You are presented with a list of expert information from oen or several sources that you need to summarize.

    RESOURCES:
    {context}

    Based **ONLY** on that resource list and without any additional assumptions from your side, perform the task specified (or answer the question).

    TASK:
    {task}

    Your answer should be in a valid JSON format like so:
    {{
        "answered": <write only "true" or "false", put "true" if the answers for the TASK are found in resources, else put "false">  
        "answer": "<a single number, word, or phrase which is the answer to the question>",
        "clarification": "<a very short mention of what the answer is based on, **always relate it back to the question**>"
    }}

    Rules:
        - If the text contains the complete answer → put the exact answer in the "answer" field & "true" in "answered" field.  
        - If the text contains no relevant information → put "not found" in the "answer" field & "true" in "answered" field.
        - If the text contains some but not all information → put "answer": "not found".
        - The "clarification" must mention the relevant part of the text and explain briefly the reasoning based on the task/question.
    
    Example 1:
     Resources:
        The population of Paris in 2023 was estimated to be 2.16 million people according to the National Institute of Statistics.;
        London's population exceeds 9 million residents as of the latest census;
        Berlin has approximately 3.8 million inhabitants within city limits.;
     Task: What is the population of Paris?
     Output:
     {{
        "answered": "true",
        "answer": "2.16 million",
        "clarification": "The answer is based on resource 1 which states Paris had an estimated population of 2.16 million in 2023, directly answering the question about Paris's population."
    }}
    
    Example 2:
     Resources:
        "Mount Everest's summit reaches high above the sea level (2020 date)."
        "The Himalayan peak known as Everest stands at 29,031.7 feet tall."
        "Everest's height was recalculated in 2020 to be 8,848.86 meters according to joint Chinese-Nepalese survey."
     Task: How tall is Mount Everest in meters?
     Output:
     {{
        "answered": "false",
        "answer": "not found",
        "clarification": "The exact answer to the task is not provided."
    }} 
    """

    llm_response = llm.complete(prompt)
    response_text = llm_response.text
    response_text = response_text.strip()

    json_match = re.search(r"```json\n(.*?)```", response_text, re.DOTALL)
    code_string = json_match.group(1) if json_match else ""
    if len(code_string) > 1: 
        response_text = code_string

    logger.info(f"- {current_function()} -- Short answer:\n{response_text}.")
    data = json.loads(response_text)
    return ShortAnswer(
            answered=True if data.get("answered") in ["True", "true", "yes", "1"] else False,
            answer=data.get("answer", "did-not-parse"),
            clarification=data.get("clarification", "did-not-parse.")
        )