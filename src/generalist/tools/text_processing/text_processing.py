from typing import Literal

from langchain_text_splitters import CharacterTextSplitter

from browser import CHATGPT_SESSION, DEEPSEEK_SESSION
from generalist.models.core import llm
from clog import get_logger
from generalist.tools.text_processing.utils import parse_config


logger = get_logger(__name__)


def _process_chunk_local(task: str, text: str) -> str:
    """Performs a task on a single block of text using an LLM.

    This function is a general-purpose processor that asks an LLM to execute
    an instruction based only on the provided context.

    Note:
        This function requires a configured LLM client, represented here as `llm`.

    Args:
        task: The instruction to be performed (e.g., "Summarise this text").
        text: The context text for the LLM to work with.

    Returns:
        The raw string response from the LLM.
    """
    prompt = f"""
    Perform the instruction/task in the user's question.
    Use only the information provided in the context.

    TASK:
    {task}

    CONTEXT:
    {text}

    **IMPORTANT**: If the text does not include the SPECIFIC information required for the task, output "NOT FOUND".
    Otherwise, provide the direct answer.
    """
    llm_result = llm.complete(prompt)
    return llm_result.text


def _process_chunk_remote(task: str, text: str) -> str:
    prompt = f"""
    Perform the instruction/task in the user's question.
    Use only the information provided in the context.

    TASK:
    {task}

    CONTEXT:
    {text}

    **IMPORTANT**: If the text does not include the SPECIFIC information required for the task, output "NOT FOUND".
    Otherwise, provide the direct answer.
    """

    answer = CHATGPT_SESSION.send_message(prompt)
    # answer =  DEEPSEEK_SESSION.send_message(prompt)



    return answer


def process_text(task: str, text: str, mode: Literal["local", "remote"] = "local") -> list[str]:
    """
    Splits a large text into chunks and processes each chunk with an LLM to perform the mentioned task.

    This is useful for analysing documents that are too large to fit into a
    single LLM context window. Each chunk is processed independently.

    Args:
        task: The task to perform on each chunk of text.
        text: The entire body of text to be processed.
        mode: Whether to use local or remote llm for text processing

    Returns:
        A list of string responses, with one response for each processed chunk.
    """
    conf_proces_text = parse_config(tool_function="process_text", param="mode")
    conf_proces_text_mode = conf_proces_text.get(mode, None)
    default_chunk_size = 4000
    default_chunk_overlap = 500
    chunk_size = conf_proces_text_mode.get("chunk_size", default_chunk_size) if conf_proces_text_mode else default_chunk_size
    chunk_overlap = conf_proces_text_mode.get("chunk_overlap", default_chunk_overlap) if conf_proces_text_mode else default_chunk_overlap

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator=" "  
    )
    chunks = text_splitter.split_text(text)

    processor = _process_chunk_remote if mode == "remote" else _process_chunk_local

    responses = []
    for chunk in chunks:
        chunk_response = processor(task, chunk)
        if "NOT FOUND" not in chunk_response:
            responses.append(chunk_response)

    return responses
