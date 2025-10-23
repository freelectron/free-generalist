
from langchain_text_splitters import CharacterTextSplitter

from ..models.core import llm
from .. import logging
from ..utils import current_function


logger = logging.getLogger(__name__)


def task_with_text_llm(task: str, text: str) -> str:
    """Performs a task on a single block of text using an LLM.

    This function is a general-purpose processor that asks an LLM to execute
    an instruction based only on the provided context.

    Note:
        This function requires a configured LLM client, represented here as `llm`.

    Args:
        task: The instruction to be performed (e.g., "Summarize this text").
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


def text_process_llm(task: str, text: str, chunk_size: int = 10000, chunk_overlap: int = 500) -> list[str]:
    """Splits a large text into chunks and processes each chunk with an LLM.

    This is useful for analyzing documents that are too large to fit into a
    single LLM context window. Each chunk is processed independently.

    Args:
        task: The task to perform on each chunk of text.
        text: The entire body of text to be processed.
        chunk_size: The maximum number of characters in each chunk.
        chunk_overlap: The number of characters to overlap between consecutive chunks.

    Returns:
        A list of string responses, with one response for each processed chunk.
    """
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator=" "  
    )
    chunks = text_splitter.split_text(text)

    responses = []
    for chunk in chunks:
        chunk_response = task_with_text_llm(task, chunk)
        if "NOT FOUND" not in chunk_response:
            responses.append(chunk_response)

    return responses