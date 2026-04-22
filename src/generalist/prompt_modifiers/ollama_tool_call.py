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

def add_tool_directive(prompt: str):
    """
    Instruction to the llm to ouput tools in a specific format.

    Example:
    ```json
    {
        "function": {
          "name": "get_weather",
          "arguments": {
            "city": "Amsterdam",
            "units": "celsius"
          }
        }
    }
    ```
    """
    prompt_delta = """
    If you need to use a tool, output only a single json like so
     
    Given: Available Tools [get_weather]
    ''```json
    {
        "function": {
          "name": "get_weather",
          "arguments": {
            "city": "Amsterdam",
            "units": "celsius"
          }
        }
    }
    ```''
    
    Only output a single json in the exact format:
    ''```json
    {
        "function": {
            "name": "<tool_name>",
            "arguments": 
                {
                    <arguments also in json format that given in ur prompt>
                }
    }
    ```''
    Note: do not modify or expand the (file)paths you are given. 
    """

    return prompt + "\n" + prompt_delta