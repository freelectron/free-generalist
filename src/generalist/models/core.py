from llama_index.llms.ollama import Ollama
import mlflow


REQUEST_TIMEOUT = 180
MODEL_NAME = "qwen2.5:14b"


llm = Ollama(
    model=MODEL_NAME, 
    request_timeout=REQUEST_TIMEOUT
)


class MLFlowLLMWrapper:
    """
    Generic class to wrap calls to llm with MLFlow logging.
    Use this class for debugging LLM calls, monkeypatch the original
    """
    def __init__(self, llm_instance):
        self.llm = llm_instance

    def complete(self, prompt, **kwargs):
        import inspect

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
    

