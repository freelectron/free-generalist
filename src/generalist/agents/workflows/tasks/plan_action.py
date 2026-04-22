from generalist.models.core import MLFlowLLMWrapper
from generalist.tools.base import BaseTool
from clog import get_logger


logger = get_logger(__name__)


def plan_next_action(
    task: str,
    context: str,
    agent_capability: str,
    tools: list[BaseTool] | None,
    previous_reflection: str | None,
    llm: MLFlowLLMWrapper,
) -> str:
    tools_str = "\n".join([f"- {tool.name}: {tool.description}" for tool in tools])

    prompt = f"""
    Role: {agent_capability}

    Task: {task}

    Context from previous steps:
    {context}

    {f"Previous reflection: {previous_reflection}" if previous_reflection else ""}

    Available tools:
    {tools_str}

    Based on the task and context, reason about:
    1. What information do you have so far?
    2. What is still missing to complete the task?
    3. Which tool should you use next and why?

    Provide a brief plan for the next action (2-3 sentences).
    """

    response = llm.complete(prompt)
    plan = response.text.strip()
    logger.info(f"Plan: {plan}")

    return plan
