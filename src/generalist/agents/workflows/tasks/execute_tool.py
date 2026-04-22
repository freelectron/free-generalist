from generalist.models.core import MLFlowLLMWrapper, LLMResponse
from generalist.tools.base import BaseTool
from clog import get_logger


logger = get_logger(__name__)


def call_tool(
    task: str,
    context: str,
    plan: str | None,
    tools: list[BaseTool] | None,
    llm: MLFlowLLMWrapper,
) -> LLMResponse:
    prompt = f"""
    Task: {task}

    Context from previous steps:
    {context}

    Plan: {plan}

    Based on the plan above, you MUST call exactly ONE of the available tools now.
    Choose the most appropriate tool to make progress on the task.
    """

    response = llm.predict_and_call(prompt=prompt, tools=tools)
    logger.info(f"Tool called: {response.tool_call.tool_name if response.tool_call else 'none'}")

    return response
