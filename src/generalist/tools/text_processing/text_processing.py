import os

from langchain_text_splitters import CharacterTextSplitter

from generalist.models.core import MLFlowLLMWrapper
from generalist.tools import BaseTool
from generalist.tools.text_processing.utils import parse_config
from clog import get_logger


logger = get_logger(__name__)

DEFAULT_CHUNK_SIZE = 40000
DEFAULT_CHUNK_OVERLAP = 500


def _process_chunk(task: str, text: str, llm: MLFlowLLMWrapper) -> str:
    prompt = f"""
    Perform the instruction/task in the user's question.
    Use only the information provided in the context.

    TASK:
    {task}

    CONTEXT:
    {text}

    If the text does not contain the relevant info, just output 1-2 short sentence what it contains.  
    """
    return llm.complete(prompt).text


class ProcessTextFileTool(BaseTool):
    name = "process_text_file"
    description = ("Reads a text file and performs a processing task on its contents using an LLM, chunk by chunk. "
                   "Writing the result as in database")

    def __init__(self, llm: MLFlowLLMWrapper):
        self.llm = llm

    def run(self, file_path: str, task: str) -> str:
        """
        Reads a file and performs a task on its text content using an LLM.

        Args:
            file_path: Path to the text file to process.
            task: Instruction to perform on the file content (e.g. "Summarise this text", "Extract all dates").

        Returns:
            str: Concatenated results from all chunks where the information was found.
        """
        try:
            with open(os.path.expanduser(file_path), "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return f"Error reading file: {e}"

        conf = parse_config(tool_function="process_text", param="mode")
        conf_local = conf.get("local", {}) if conf else {}
        chunk_size = conf_local.get("chunk_size", DEFAULT_CHUNK_SIZE)
        chunk_overlap = conf_local.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP)

        splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator=" ")
        chunks = splitter.split_text(text)

        responses = []
        for chunk in chunks:
            if chunk:
                result = _process_chunk(task, chunk, self.llm)
                if "NOT FOUND" not in result:
                    responses.append(result)

        return "\n\n".join(responses) if responses else "NOT FOUND"