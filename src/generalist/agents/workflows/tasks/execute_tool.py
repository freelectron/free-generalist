from generalist.dialer.core import MLFlowLLMWrapper, LLMResponse
from generalist.tools import BaseTool
from clog import get_logger


logger = get_logger(__name__)


def call_tool(
    task: str,
    plan: str | None,
    tools: list[BaseTool] | None,
    llm: MLFlowLLMWrapper,
) -> LLMResponse:
    # Context from previous steps is intentionally NOT injected here: plan_next_action is
    # required to emit a self-contained plan (file paths, params, values distilled from the
    # full context), so the executor only needs task + plan + tool schemas to pick a tool.
    if not tools:
        raise ValueError("call_tool requires at least one tool, got none.")

    prompt = f"""
    Task: {task}

    Plan: {plan or ""}

    Pick exactly ONE tool from the list above that best advances the plan.
    """
    response = llm.complete_and_call(prompt=prompt, tools=tools)
    logger.info(f"Tool called: {response.tool_call.tool_name if response.tool_call else 'none'}")

    return response
