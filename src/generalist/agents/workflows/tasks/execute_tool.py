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
    # TODO: so the dilemma here is whether to add context like so
    #  Context from previous steps:
    #  {context}
    prompt = f"""
    Task: {task}

    Plan: {plan}

    **IMPORTANT: Your ONLY output must be a single JSON tool call — no explanation, no prose, nothing else.**

    Available tools:
        {[tool_to_llm_schema(tool) for tool in tools] if tools else None}

    Required output format:
    ```json
    {{
        "function": {{
            "name": "<tool_name>",
            "arguments": {{
                "<param>": "<value>"
            }}
        }}
    }}
    ```

    Pick exactly ONE tool from the list above that best advances the plan. Output only the JSON.
    """
    prompt_formatted = add_tool_directive(prompt)

    response = llm.predict_and_call(prompt=prompt_formatted, tools=tools)
    logger.info(f"Tool called: {response.tool_call.tool_name if response.tool_call else 'none'}")

    return response
