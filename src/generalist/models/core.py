import inspect
from abc import ABC
from typing import Callable, get_origin, Union, get_args, get_type_hints

import ollama
import mlflow

from browser import ChromeBrowser
from browser.llm_browser import LLMBrowser
from clog import get_logger
from generalist.prompt_modifiers.ollama_tool_call import add_tool_directive, tool_to_llm_schema
from generalist.prompt_modifiers.utils import parse_out_tool_call


logger = get_logger(__name__)
REQUEST_TIMEOUT = 180
LOCAL_OLLAMA_QWEN_MODEL_NAME = "qwen2.5:14b"


class LLMToolCall:
    def __init__(self, name: str, output: str | None):
        self.tool_name = name
        self.tool_output = output

    def __str__(self):
        return f"ToolCall({self.tool_name}): {self.tool_output}"

class LLMResponse:
    def __init__(self, text: str, tool_call: LLMToolCall | None = None):
        self.text = text
        self.tool_call = tool_call

    def __str__(self):
        return f"LLMResponse({self.text}) with {str(self.tool_call)}"


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
    def __init__(self, browser: ChromeBrowser):
        self.llm = LLMBrowser(browser)

    def complete(self, prompt: str, *args, **kwargs) -> LLMResponse:
        answer =  self.llm.call(message=prompt)

        return LLMResponse(answer)

    def predict_and_call(self, prompt: str, tools: list, *args, **kwargs) -> LLMResponse:
        answer = self.complete(prompt=prompt)

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

    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        result = ollama.chat(model=self.model, messages=[{"role": "user", "content": prompt}], **kwargs)
        return LLMResponse(result.message.content)

    def predict_and_call(self, prompt: str, tools: list, **kwargs) -> LLMResponse:
        tool_schemas = [tool_to_llm_schema(tool) for tool in tools]
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
            mlflow.log_metric("response_length", len(str(raw_response.text)))

            mlflow.log_text(prompt, f"prompt_{caller_function}.txt")
            mlflow.log_text(str(raw_response.text), f"response_{caller_function}.txt")
            
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

            mlflow.log_metric("prompt_length", len(prompt))
            mlflow.log_metric("response_length", len(str(raw_response.text)))

            mlflow.log_text(prompt, f"prompt_{caller_function}.txt")
            mlflow.log_text(str(raw_response.text), f"response_{caller_function}.txt")

            return raw_response
