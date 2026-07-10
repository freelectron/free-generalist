import json
from typing import Callable, get_origin, Union, get_args, get_type_hints
import inspect


def python_type_to_json_schema(py_type):
    origin = get_origin(py_type)

    if origin is Union:
        args = [arg for arg in get_args(py_type) if arg is not type(None)]
        if len(args) == 1:
            return python_type_to_json_schema(args[0])

    base = origin or py_type

    if base is str:
        return {"type": "string"}
    elif base is int:
        return {"type": "integer"}
    elif base is float:
        return {"type": "number"}
    elif base is bool:
        return {"type": "boolean"}

    elif base is list:
        args = get_args(py_type)
        if args:
            return {
                "type": "array",
                "items": python_type_to_json_schema(args[0])
            }
        return {"type": "array"}

    elif base is dict:
        return {"type": "object"}

    return {"type": "string"}

def tool_to_llm_schema(tool) -> dict:
    """
    Ollama style function calling.
    """
    sig = inspect.signature(tool.run)
    type_hints = get_type_hints(tool.run)

    properties = {}
    required = []

    for name, param in sig.parameters.items():
        if name == "self":
            continue

        param_type = type_hints.get(name, str)
        properties[name] = python_type_to_json_schema(param_type)

        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": inspect.getdoc(tool.run) or tool.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }

def add_tool_directive(prompt: str, tools: list, extra_schemas: list[dict] | None = None):
    tool_schemas = [tool_to_llm_schema(tool) for tool in tools]
    if extra_schemas:
        # Pre-built OpenAI-format schemas (e.g. MCP tools) merged verbatim.
        tool_schemas = tool_schemas + list(extra_schemas)

    prompt_delta = f"""
    Available client side tools:
        {json.dumps(tool_schemas, indent=2)}
        
    IMPORTANT:
    - Use ONLY the client-side tools listed in the prompt above. Do not invent or reference any other tools.
    - Do not modify or expand any file paths provided in the prompt; pass them through verbatim as tool arguments.
    - When asked for a PLAN or REFLECTION, respond as requested and do NOT emit a tool call.
    - Otherwise, your ENTIRE response must be a single JSON tool call wrapped in a ```json``` fenced block. No prose, no explanation, no additional text.

    Required output format (substitute <tool_name> and arguments with values from an available tool):
    ```json
    {{
        "function": {{
            "name": "<tool_name>",
            "arguments": {{
                "<param_1>": "<value_1>",
                "<param_2>": "<value_2>"
            }}
        }}
    }}
    ```

    Example (assuming the available tool `get_weather` accepts `city` and `units`):
    ```json
    {{
        "function": {{
            "name": "get_weather",
            "arguments": {{
                "city": "Amsterdam",
                "units": "celsius"
            }}
        }}
    }}
    ```
    """

    return "You should also take into account:\n" + prompt + "\n" + prompt_delta