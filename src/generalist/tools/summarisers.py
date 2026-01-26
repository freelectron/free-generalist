import json
import regex as re

from .data_model import ShortAnswer, AgentRunSummary
from ..models.core import llm
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
    You are presented with a list of information from one or several sources that you need to summarize.

    RESOURCES:
    {context}

    Based **ONLY** on that resource list and without any additional assumptions from your side, perform the task specified (or answer the question).

    TASK:
    {task}

    Your answer should be in a valid JSON format like so:
    {{
        "answered": <write only "true" or "false", put "true" if the answers for the TASK is clearly stated in the resources, else put "false">  
        "answer": "<a single number, word, or phrase which is the answer to the question>",
        "clarification": "<a very short mention of what the answer is based on, **always relate it back to the question**>"
    }}

    Rules:
        - If the text contains the complete answer → put the exact answer in the "answer" field & "true" in the "answered" field.  
        - If the text contains no relevant information → put "not found" in the "answer" field & "false" in the "answered" field.
        - If the text contains some but not all information → put "answer": "not found" & "false" in the "answered" field.
        - The "clarification" must mention the relevant part of the text and explain briefly the reasoning why the question has been answered or not.
    
    Example 1:
     Resources:
        The population of Paris in 2023 was estimated to be 2.16 million people according to the National Institute of Statistics;
        London's population exceeds 9 million residents as of the latest census;
        Berlin has approximately 3.8 million inhabitants within city limits;
     Task: What is the population of Paris?
     Output:
     {{
        "answered": "true",
        "answer": "2.16 million",
        "clarification": "Resource 1 states that Paris had an estimated population of 2.16 million in 2023."
    }}
    
    Example 2:
     Resources:
        "Mount Everest's summit reaches high above the sea level (2020 date)."
        "The Himalayan peak known is surrounded by beautiful scenery, but it is extremely hard to reach alone."
        "Everest's height was recalculated in 2020 by joint Chinese-Nepalese survey."
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

    logger.info(f"Short answer:\n{response_text}.")
    data = json.loads(response_text)
    return ShortAnswer(
            answered=True if data.get("answered") in ["True", "true", "yes", "1"] else False,
            answer=data.get("answer", "did-not-parse"),
            clarification=data.get("clarification", "did-not-parse.")
        )


def construct_task_completion(task: str, context: str, agent_capability: str) -> AgentRunSummary:
    """
    Evaluates whether a task has been accomplished based on provided context.

    The task does not require a final answer. It is considered completed
    if the main steps or intent appear to be fulfilled based solely on
    the given resources.
    """

    prompt = f"""
    You are an agent that can {agent_capability}!
    You are presented with a list of information describing work, actions,
    or outcomes related to the task.

    RESOURCES:
    {context}

    Based **ONLY** on the resources above and without any additional assumptions,
    determine whether the task has been accomplished. 

    TASK:
    {task}
    
    Your response MUST be valid JSON in the following format:
    {{
        "completed": <write only "true" or "false">,
        "summary": "<a short phrase describing what was achieved, or why it was not completed.'>"
    }}
    
    Example 1:
     Task: Calculate what the average prices was at the end of the day for the following file trades.csv.  
     Agent: code_writing_execution
     Resources:
        [
            ContentResource(
                provided_by="write_code"), 
                content="import pandas as pd; # read file ... ",
                link="",
                metainfo={{}}, 
            ), 
        ]
     Output:
        {{
            "completed": "false",
            "summary": "the code to complete the task is written but it is not executed."
        }}
    Explanation: code_writing_execution can not only write code but also execute it, and the task asks for concrete numerical output.
    
    Example 2:
     Task: Search information about the dotcom economical crisis and find the dates. 
     Agent: deep_web_search 
     Resources:
        [
            ContentResource(
                provided_by="deep_web_search"), 
                content="downloaded content for 2001 economic crisis to /var/tmp/h3893298432/deiowur329",
                link="/var/tmp/h3893298432/deiowur329",
                metainfo={{}}, 
            ), 
        ]
     Output:
        {{
            "completed": "true",
            "summary": "the information has been found and stored locally for further processing"
        }}
    Explanation: deep_web_search can download resources and it has done so.
    
    Rules:
        - If the main steps or intent of the task are clearly completed → "completed": true
        - If the task appears partially done but core objectives are met → "completed": true
        - If key steps are missing or the task intent is not fulfilled → "completed": false
    
    **IMPORTANT**: A concrete answer is not required as long as the main steps or intent are completed and the task does not ask for it explicitly.
    """

    llm_response = llm.complete(prompt)
    response_text = llm_response.text.strip()

    json_match = re.search(r"```json\n(.*?)```", response_text, re.DOTALL)
    code_string = json_match.group(1) if json_match else ""
    if len(code_string) > 1:
        response_text = code_string

    logger.info(f"Task completion:\n{response_text}.")

    data = json.loads(response_text)

    return AgentRunSummary(
        completed=True if data.get("completed") in ["True", "true", "yes", "1"] else False,
        summary=data.get("summary", "did-not-parse"),
    )

def task_completed():
    """
    This tool signals that the task has been completed.
    """
    return "end"