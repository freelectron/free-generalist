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
You are tasked with determining whether the following TASK can be answered using the provided information.

TASK: {task}

INFORMATION PROVIDED:
{context}

INSTRUCTIONS:
1. Review the information provided above carefully
2. Determine if the TASK can be fully answered using ONLY the information given
3. Do NOT use external knowledge or make assumptions beyond what is explicitly stated

OUTPUT FORMAT (valid JSON):
{{
    "answer": "<provide the direct answer as a concise word, number, or short phrase, IF the TASK is completely answered in the information, otherwise use leave blank>",
    "clarification": "<briefly explain what the information contains and how it relates to the task>"
}}

IMPORTANT RULES:
- Provide "answer" ONLY when the task has a complete answer in the provided information
- If the answer is partial or incomplete, set "answered" to "false"
- ALWAYS fill the "clarification" field with the main findings from the information, regardless of whether the task was answered
- Base your response strictly on the provided information without adding external knowledge
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

    answer = data.get("answer", None)
    if not answer:
        answered = False
    elif answer in ["", "None", "blank"]:
        answered = False
    else:
        answered = True

    return ShortAnswer(
        answered=answered,
        answer=answer,
        clarification=data.get("clarification", None)
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
