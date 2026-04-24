from generalist.models.core import MLFlowLLMWrapper
from generalist.tools import BaseTool
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
    
    **IMPORTANT**
    Based on the task and available context, produce a plan only — DO NOT EXECUTE ANYTHING!
    Identify which tool to use next and why, given what is already known.
    The plan MUST BE SELF-CONTAINED: include all key details (e.g. file paths, parameters, values from context) needed to execute the next step without referring back to prior context.
    Be concise (2-3 sentences).
    """

    response = llm.complete(prompt)
    plan = response.text.strip()
    logger.info(f"Plan: {plan}")

    return plan
