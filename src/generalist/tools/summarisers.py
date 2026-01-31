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
    You are presented with a list of information from one or several sources. 
    Resources to use:
    {context}

    Based **ONLY** on that resource list and without any additional assumptions from your side, evaluate whether the task specified was performed. 
    TASK:
    {task}

    Your answer should be in a valid JSON format like so:
    {{
        "answered": <write only "true" or "false", put "true" if the answers for the TASK is clearly stated in the resources, else put "false">  
        "answer": "<a single number, word, or phrase which is the answer to the question>",
        "clarification": "<a very short mention of what the answer is based on, **always relate it back to the question**>"
    }}
    
    **IMPORTANT**: if only partial answer, do not mark the task as answered : true ! 
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