import inspect
from abc import ABC
from typing import Callable, get_origin, Union, get_args, get_type_hints

import ollama
import mlflow

from browser import ChromeBrowser
from browser.llm_browser import LLMBrowser
from clog import get_logger
from generalist.prompt_modifiers.ollama_tool_call import add_tool_directive
from generalist.prompt_modifiers.utils import parse_out_tool_call


logger = get_logger(__name__)
REQUEST_TIMEOUT = 180
LOCAL_OLLAMA_QWEN_MODEL_NAME = "qwen2.5:14b"


class LLMToolCall:
    def __init__(self, name: str, output: str | None):
        self.tool_name = name
        self.tool_output = output

class LLMResponse:
    def __init__(self, text: str, tool_call: LLMToolCall | None = None):
        self.text = text
        self.response = text
        self.tool_call = tool_call


class LLMBase(ABC):
    """
    Base class for interacting with LLM API's.
    """
    # TODO: replace in the children
    model: str  = "placeholder"

    def complete(self, prompt: str, *args, **kwargs) -> LLMResponse:
        """
        Just answer the prompt
        """
        raise NotImplementedError

    def predict_and_call(self, prompt: str, tools: list[Callable], *args, **kwargs) -> LLMResponse:
        """
        First predicts if we need to use a tool from `tools` based on the `prompt`.
        If yes, calls the tool and returns the result.
        """
        raise NotImplementedError


class LLMOpenClaw(LLMBase):
    def __init__(self, browser: ChromeBrowser):
        self.llm = LLMBrowser(browser)

    def complete(self, prompt: str, *args, **kwargs):
        answer = self.llm.call(prompt)
        return LLMResponse(answer)

    def complete_with_tools(self, prompt: str):
        prompt_modified = add_tool_directive(prompt)
        answer = self.complete(prompt_modified)
        tool_call = parse_out_tool_call(answer.text)

        return answer.text, tool_call


class LLMBrowserWithTools(LLMBase):
    @classmethod
    def python_type_to_json_schema(cls, py_type):
        origin = get_origin(py_type)

        if origin is Union:
            args = [arg for arg in get_args(py_type) if arg is not type(None)]
            if len(args) == 1:
                return cls.python_type_to_json_schema(args[0])

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
                    "items": cls.python_type_to_json_schema(args[0])
                }
            return {"type": "array"}

        elif base is dict:
            return {"type": "object"}

        return {"type": "string"}

    @classmethod
    def tool_to_llm_schema(cls, tool) -> dict:
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
            properties[name] = cls.python_type_to_json_schema(param_type)

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

    def __init__(self, browser: ChromeBrowser):
        self.llm = LLMBrowser(browser)

    def complete(self, prompt: str, *args, **kwargs) -> LLMResponse:
        answer =  self.llm.call(message=prompt)

        return LLMResponse(answer)

    def predict_and_call(self, prompt: str, tools: list, *args, **kwargs) -> LLMResponse:
        tool_descriptions = [self.tool_to_llm_schema(tool) for tool in tools]
        prompt += f"\nAvailable Tools:{tool_descriptions}"
        prompt_formatted = add_tool_directive(prompt)
        answer = self.complete(prompt=prompt_formatted)

        tool_call = parse_out_tool_call(answer.text)
        if tool_call:
            available_tools = {tool.name: tool for tool in tools}
            tool_name = tool_call["function"]["name"]
            tool_kwargs = tool_call["function"]["arguments"]
            tool = available_tools.get(tool_name)
            res_tool = tool.run(**tool_kwargs)
            answer.tool_call = LLMToolCall(tool_name, res_tool)

        return answer

class LLMOllama(LLMBase):
    def __init__(self, model:str, request_timeout):
        self.model = model
        self._timeout = request_timeout

    def complete(self, prompt: str, **kwargs):
        result = ollama.chat(model=self.model, messages=[{"role": "user", "content": prompt}], **kwargs)
        return LLMResponse(result.message.content)

    def predict_and_call(self, prompt: str, tools: list, **kwargs):
        tool_schemas = [LLMBrowserWithTools.tool_to_llm_schema(tool) for tool in tools]
        result = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            tools=tool_schemas,
            **kwargs,
        )
        if len(result.message.tool_calls) > 1:
            raise ValueError(f"More than 1 tool identified by LLM: {result.message}")
        elif len(result.message.tool_calls) == 1:
            available_tools = {tool.name: tool for tool in tools}
            tool_name = result.message.tool_calls[0].function.name
            tool = available_tools.get(tool_name)
            res_tool = tool.run(**result.message.tool_calls[0].function.arguments)
            tool_call = LLMToolCall(tool_name, res_tool)
            return LLMResponse(result.message.content, tool_call)
        else:
            return LLMResponse(result.message.content)


# Note: only needed to get traces and logs
class MLFlowLLMWrapper:
    """
    Generic class to wrap calls to llm with MLFlow logging.
    Use this class for debugging LLM calls, monkeypatch the original
    """
    def __init__(self, llm_instance: LLMBase):
        self.llm = llm_instance

    def complete(self, prompt, **kwargs) -> LLMResponse:
        # Get caller function name and module
        caller_frame = inspect.currentframe().f_back
        caller_function = caller_frame.f_code.co_name
        caller_module = caller_frame.f_globals.get('__name__', 'unknown')

        with mlflow.start_run(nested=True, run_name=f"{self.llm.model}_{caller_function}"):
            mlflow.log_param("caller", f"{caller_module}.{caller_function}")
            mlflow.log_param("llm_name", self.llm.model)

            raw_response = self.llm.complete(prompt, **kwargs)

            mlflow.log_metric("prompt_length", len(prompt))
            mlflow.log_metric("response_length", len(raw_response.text))

            mlflow.log_text(prompt, f"prompt_{caller_function}.txt")
            mlflow.log_text(raw_response.text, f"response_{caller_function}.txt")
            
            return raw_response

    def predict_and_call(self, prompt, tools, **kwargs) -> LLMResponse:
        # Get caller function name and module
        caller_frame = inspect.currentframe().f_back
        caller_function = caller_frame.f_code.co_name
        caller_module = caller_frame.f_globals.get('__name__', 'unknown')

        with mlflow.start_run(nested=True, run_name=f"{self.llm.model}_{caller_function}"):
            mlflow.log_param("caller", f"{caller_module}.{caller_function}")
            mlflow.log_param("llm_name", self.llm.model)

            raw_response = self.llm.predict_and_call(prompt=prompt, tools=tools, **kwargs)
            # TODO: think about data handling here,
            #  overwriting is loosing info about the original json for the tool call
            if raw_response.tool_call:
                raw_response.text = raw_response.tool_call.tool_output
                raw_response.response = raw_response.text

            mlflow.log_metric("prompt_length", len(prompt))
            mlflow.log_metric("response_length", len(raw_response.text))

            mlflow.log_text(prompt, f"prompt_{caller_function}.txt")
            mlflow.log_text(raw_response.text, f"response_{caller_function}.txt")

            return raw_response
