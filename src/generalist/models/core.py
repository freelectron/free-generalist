import inspect
from typing import Callable

import ollama
import mlflow

from clog import get_logger


logger = get_logger(__name__)
REQUEST_TIMEOUT = 180
LOCAL_MODEL_NAME = "qwen2.5:14b"


class LLMToolCall:
    def __init__(self, name: str, output: str | None):
        self.tool_name = name
        self.tool_output = output

class LLMResponse:
    def __init__(self, text: str, tool_call: LLMToolCall | None = None):
        self.text = text
        self.response = text
        self.tool_call = tool_call

class LLM:
    """
    Unified class for interacting with LLM API's.
    """
    def __init__(self, model:str, request_timeout):
        self.model = model
        self._timeout = request_timeout

    def complete(self, prompt: str, **kwargs):
        result = ollama.chat(model=self.model, messages=[{"role": "user", "content": prompt}], **kwargs)
        return LLMResponse(result.message.content)

    def predict_and_call(self, user_msg: str, tools: list[Callable], **kwargs):
        result = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": user_msg}],
            tools=tools,
            **kwargs,
        )
        if len(result.message.tool_calls) > 1:
            raise ValueError(f"More than 1 tool identified by LLM: {result.message}")
        elif len(result.message.tool_calls) == 1:
            available_functions = {fn.__name__: fn for fn in tools}
            tool_name = result.message.tool_calls[0].function.name
            fn = available_functions.get(tool_name, None )
            res_tool = fn(**result.message.tool_calls[0].function.arguments)
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
    def __init__(self, llm_instance):
        self.llm = llm_instance

    def complete(self, prompt, **kwargs):
        # Get caller function name and module
        caller_frame = inspect.currentframe().f_back
        caller_function = caller_frame.f_code.co_name
        caller_module = caller_frame.f_globals.get('__name__', 'unknown')

        with mlflow.start_run(nested=True, run_name=f"{self.llm.model}_{caller_function}"):
            mlflow.log_param("caller", f"{caller_module}.{caller_function}")
            mlflow.log_param("llm_name", self.llm.model)

            response = self.llm.complete(prompt, **kwargs)

            mlflow.log_metric("prompt_length", len(prompt))
            mlflow.log_metric("response_length", len(response.text))

            mlflow.log_text(prompt, f"prompt_{caller_function}.txt")
            mlflow.log_text(response.text, f"response_{caller_function}.txt")
            
            return response

    def predict_and_call(self, user_msg, tools, **kwargs):
        # Get caller function name and module
        caller_frame = inspect.currentframe().f_back
        caller_function = caller_frame.f_code.co_name
        caller_module = caller_frame.f_globals.get('__name__', 'unknown')

        with mlflow.start_run(nested=True, run_name=f"{self.llm.model}_{caller_function}"):
            mlflow.log_param("caller", f"{caller_module}.{caller_function}")
            mlflow.log_param("llm_name", self.llm.model)

            response = self.llm.predict_and_call(user_msg=user_msg, tools=tools, **kwargs)

            mlflow.log_metric("prompt_length", len(user_msg))
            mlflow.log_metric("response_length", len(response.response))

            mlflow.log_text(user_msg, f"prompt_{caller_function}.txt")
            mlflow.log_text(response.response, f"response_{caller_function}.txt")

            return response


local_llm_with_mlflow = MLFlowLLMWrapper(
    LLM(
        model=LOCAL_MODEL_NAME,
        request_timeout=REQUEST_TIMEOUT,
    )
)