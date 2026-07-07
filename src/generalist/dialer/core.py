import inspect
import json
import os
from abc import ABC
from typing import Callable, get_origin, Union, get_args, get_type_hints

import ollama
import litellm
import mlflow
import requests

from browser import ChromeBrowser
from browser.llm_browser import LLMBrowser
from clog import get_logger
from generalist.prompt_modifiers.ollama_tool_call import add_tool_directive, tool_to_llm_schema
from generalist.prompt_modifiers.utils import parse_out_tool_call
from mcp_internal.client.session import MCPConnection

logger = get_logger(__name__)
REQUEST_TIMEOUT = 180
LOCAL_OLLAMA_QWEN_MODEL_NAME = "qwen2.5:14b"
ZAI_DEFAULT_MODEL = "glm-5.2"
# GLM Coding Plan uses a dedicated endpoint, separate from the pay-as-you-go one
# (https://api.z.ai/api/paas/v4). Pointing litellm at this base is what bills against
# your coding-plan quota instead of the pay-as-you-go balance.
ZAI_CODING_PLAN_API_BASE = "https://api.z.ai/api/coding/paas/v4"


class LLMToolCall:
    def __init__(self, name: str, output: str | None):
        self.tool_name = name
        self.tool_output = output

    def __str__(self):
        return f"ToolCall({self.tool_name}): {self.tool_output}"

class LLMResponse:
    def __init__(self, text: str | None, tool_call: LLMToolCall | None = None):
        self.text = text
        self.tool_call = tool_call

    def __str__(self):
        return f"LLMResponse({self.text}) with {str(self.tool_call)}"


class LLMToolsExecutor(ABC):
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

    def complete_and_call(self, prompt: str, tools: list[Callable], *args, **kwargs) -> LLMResponse:
        """
        First predicts if we need to use a tool from `tools` based on the `prompt`.
        If yes, calls the tool and returns the result.
        """
        raise NotImplementedError


class LLMBrowserServer:
    """ Parse out the tool call and return it separately without executing. """
    def __init__(self, browser: ChromeBrowser):
        self.llm = LLMBrowser(browser)

    def complete(self, prompt: str):
        answer = self.llm.call(prompt)

        return answer


class LLMBrowserDialer(LLMToolsExecutor):
    """ Also executes tools that are returned by an LLM. """
    def __init__(self, host: str, port: int, auth_token: str):
        self._api_base = f"http://{host}:{port}"
        self._auth_token = auth_token

    def complete(self, prompt: str, *args, **kwargs) -> LLMResponse:
        resp = requests.post(
            f"{self._api_base}/api/chat",
            json={"model": "web", "messages": [{"role": "user", "content": prompt}], "stream": False},
            headers={"Authorization": f"Bearer {self._auth_token}"},
        )
        resp.raise_for_status()
        return LLMResponse(json.loads(resp.json())["message"]["content"])

    def complete_and_call(self, prompt: str, tools: list, *args, **kwargs) -> LLMResponse:
        prompt_formatted = add_tool_directive(prompt, tools)
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


class LLMZaiDialer(LLMToolsExecutor):
    """ Connects to ZAI (Zhipu AI) GLM models via litellm and executes returned tool calls.

    Targets the GLM Coding Plan endpoint by default (ZAI_CODING_PLAN_API_BASE), which
    bills against the coding-plan quota. For pay-as-you-go balance instead, pass
    api_base="https://api.z.ai/api/paas/v4".

    Args:
        model: bare ZAI model name, e.g. "glm-5.2" (see ZAI_DEFAULT_MODEL).
        api_key: ZAI API key (defaults to ZAI_API_KEY env var).
        request_timeout: per-request timeout in seconds.
        api_base: OpenAI-compatible ZAI base URL.
    """
    def __init__(
        self,
        model: str = ZAI_DEFAULT_MODEL,
        api_key: str = None,
        request_timeout: int = REQUEST_TIMEOUT,
        api_base: str = ZAI_CODING_PLAN_API_BASE,
    ):
        self.model = model
        self._api_key = api_key or os.getenv("ZAI_API_KEY")
        self._timeout = request_timeout
        self._api_base = api_base

        # TODO: separate this into a method, make optional and fail-safe
        # Must match mcp.settings.streamable_http_path on the server (FastMCP default is /mcp)
        self._mcp_uri = "http://localhost:9000/mcp"
        self._mcp_session = MCPConnection(self._mcp_uri)
        self.mcp_tools = self._mcp_session.list_tools()

    def _completion(self, prompt: str, **kwargs):
        # Use litellm's openai/ prefix with an explicit api_base so the request hits the
        # coding-plan endpoint directly, instead of litellm's zai/ handler (standard endpoint).
        # Returns the raw litellm response so callers can read native message.tool_calls.
        return litellm.completion(
            model=f"openai/{self.model}",
            api_base=self._api_base,
            api_key=self._api_key,
            messages=[{"role": "user", "content": prompt}],
            timeout=self._timeout,
            **kwargs,
        )

    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        result = self._completion(prompt, **kwargs)
        return LLMResponse(result.choices[0].message.content)

    def complete_and_call(self, prompt: str, tools: list, **kwargs) -> LLMResponse:
        tool_schemas = [tool_to_llm_schema(tool) for tool in tools]
        result = self._completion(prompt, tools=tool_schemas+self.mcp_tools, **kwargs)

        message = result.choices[0].message
        tool_calls = getattr(message, "tool_calls", None) or []

        if len(tool_calls) > 1:
            raise ValueError(f"More than 1 tool identified by LLM: {message}")
        elif len(tool_calls) == 1:
            available_tools = {tool.name: tool for tool in tools}
            tc = tool_calls[0]
            tool_name = tc.function.name
            if tool_name in [ t["function"]["name"] for t in self.mcp_tools]:
                tool_res = self._mcp_session.call_tool(tc)
            else:
                tool = available_tools.get(tool_name)
                tool_args = json.loads(tc.function.arguments)
                tool_res = tool.run(**tool_args)

            tool_call = LLMToolCall(tool_name, tool_res)
            return LLMResponse(message.content, tool_call)
        else:
            return LLMResponse(message.content)


# Note: only needed to get traces and logs
class MLFlowLLMWrapper:
    """
    Generic class to wrap calls to llm with MLFlow logging.
    Use this class for debugging LLM calls, monkeypatch the original
    """
    def __init__(self, llm_instance: LLMToolsExecutor):
        self.llm = llm_instance

    def complete(self, prompt: str, **kwargs) -> LLMResponse:
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

    def complete_and_call(self, prompt:str, tools:list, **kwargs) -> LLMResponse:
        # Get caller function name and module
        caller_frame = inspect.currentframe().f_back
        caller_function = caller_frame.f_code.co_name
        caller_module = caller_frame.f_globals.get('__name__', 'unknown')

        with mlflow.start_run(nested=True, run_name=f"{self.llm.model}_{caller_function}"):
            mlflow.log_param("caller", f"{caller_module}.{caller_function}")
            mlflow.log_param("llm_name", self.llm.model)

            raw_response = self.llm.complete_and_call(prompt=prompt, tools=tools, **kwargs)

            mlflow.log_metric("prompt_length", len(prompt))
            mlflow.log_metric("response_length", len(str(raw_response.text)))

            mlflow.log_text(prompt, f"prompt_{caller_function}.txt")
            mlflow.log_text(str(raw_response.text), f"response_{caller_function}.txt")

            return raw_response


if __name__ == "__main__":
    # from generalist.tools.data_model import BaseTool
    # dialer = LLMBrowserDialer(host="localhost", port=8000, auth_token="0000")
    # prompt = "What was the capital of Prussia? Available tools:  1) 'get_capital' - get capital of the country, args: 'country' = specify the country."
    # class BT(BaseTool):
    #     name: "get_capital"
    #     description: "Dummy"
    #
    #     def run(self):
    #         print("OK")
    # print(dialer.complete_and_call(prompt, tools=[BT]))

    litellm._turn_on_debug()
    dialer = LLMZaiDialer()
    prompt = "do 5 + 2317"
    tools = []
    print(dialer.complete_and_call(prompt=prompt, tools=tools))