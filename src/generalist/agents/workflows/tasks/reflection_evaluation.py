import json
import regex as re

from generalist.tools.data_model import AgentRunSummary
from generalist.models.core import MLFlowLLMWrapper
from clog import get_logger


logger = get_logger(__name__)


def evaluate_task_completion(task: str, context: str, agent_capability: str, llm: MLFlowLLMWrapper) -> AgentRunSummary:
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
    ```json
    {{
        "done": <write only "true" or "false">,
        "summary": "<a short phrase describing what was achieved, and if agent can do something else with its available capabilities.>"
    }}
    ```
    
    Explanation:
    {{
        "done": <whether the agent has done everything it could based on its capabilities>,
        "summary": "<a short phrase describing what was achieved and how the task was answered, and if agent can do something else with its available capabilities.>"
    }}
    """

    llm_response = llm.complete(prompt)
    response_text = llm_response.text.strip()

    json_match = re.search(r"json.*?(\{.*\})", response_text, re.DOTALL | re.IGNORECASE)
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
        # FIXME: summary is not being used anywhere, at least log it? 
        summary=data.get("summary", "did-not-parse"),
    )
