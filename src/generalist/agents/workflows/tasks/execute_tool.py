from generalist.models.core import MLFlowLLMWrapper, LLMResponse
from generalist.prompt_modifiers.ollama_tool_call import tool_to_llm_schema, add_tool_directive
from generalist.tools import BaseTool
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

    Based on the plan above, you MUST call exactly ONE of the available tools in THIS PROMPT.
    Tools:
        {[tool_to_llm_schema(tool) for tool in tools] if tools else None}
    Choose the most appropriate tool to make progress on the task.
    YOUR OUTPUT MUST BE A JSON TOOL CALL!
    """
    prompt_formatted = add_tool_directive(prompt)

    response = llm.predict_and_call(prompt=prompt_formatted, tools=tools)
    logger.info(f"Tool called: {response.tool_call.tool_name if response.tool_call else 'none'}")

    return response
