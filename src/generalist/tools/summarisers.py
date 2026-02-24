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
    You are presented with a list of information from one or several sources: {context}

    Based **ONLY** on that information and without any additional assumptions from your side, evaluate whether the task specified was performed. 
    TASK: {task}

    Your answer should be in a valid JSON format like so:
    {{
        "answered": <write only "true" or "false", put "true" if the answers for the TASK is clearly stated in the resources, else put "false">  
        "answer": "<a single number, word, or phrase which is the answer to the question>",
        "clarification": "<a very short mention of what is stated in the context/sources>"
    }}
    
    **IMPORTANT**: regardless whether the task has been completed, `clarification` field should include the main findings. 
    **IMPORTANT**: if only partial answer, do not mark the task as answered:true !!! 
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
    # FIXME: either make parsing more robust or do manually
    if isinstance(data["answered"], bool):
        data["answered"] = str(data["answered"])

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
    You are an agent that can ONLY {agent_capability}. Thus your capabilities are: {agent_capability}. 
    You are presented with a list of information describing work, actions, or outcomes of the previous steps:
    {context}

    Based **ONLY** on the resources above and without any additional assumptions, determine whether the agent has accomplished its task: {task}
    And whether it should proceed to the next step.
    
    Your response MUST be valid JSON in the following format:
    {{
        "done": <write only "true" or "false">,
        "summary": "<a short phrase describing what was achieved, and if agent can do something else with its available capabilities.>"
    }}
    Explanation:
    {{
        "done": <whether the agent has done everything it could based on its capabilities>,
        "summary": "<a short phrase describing what was achieved and how the task was answered, and if agent can do something else with its available capabilities.>"
    }}
    """

    llm_response = llm.complete(prompt)
    response_text = llm_response.text.strip()

    json_match = re.search(r"```json\n(.*?)```", response_text, re.DOTALL)
    code_string = json_match.group(1) if json_match else ""
    if len(code_string) > 1:
        response_text = code_string

    logger.info(f"Task completion:\n{response_text}.")

    data = json.loads(response_text)
    # FIXME: either make parsing more robust or do manually
    if isinstance(data["done"], bool):
        data["done"] = str(data["done"])

    return AgentRunSummary(
        completed=True if data.get("done") in ["True", "true", "yes", "1"] else False,
        summary=data.get("summary", "did-not-parse"),
    )

def summarise_findings(task: str, context: str) -> ShortAnswer:
    """
    Evaluates whether the task has been accomplished based on provided context.

    The task does not require a final answer. It is considered completed
    if the main steps or intent appear to be fulfilled based solely on
    the given resources.
    """

    prompt = f"""
    You are presented with a list of information describing work, actions, or outcomes of previous steps:
    {context}

    Based **ONLY** on the resources above and without any additional assumptions, provide a summary of information. 
     
    Extract (copy-paste) the information that might be needed for the task: {task}.
    """

    llm_response = llm.complete(prompt)
    response_text = llm_response.text.strip()

    logger.info(f"From `summarise_findings`:\n{response_text}.")


    return ShortAnswer(
        answered=False,
        answer="this is just a summary of previous steps.",
        clarification=response_text,
    )


def task_completed():
    """
    This tool signals that the task has been completed.
    """
    return "end"