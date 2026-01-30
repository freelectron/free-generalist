import json

from langchain_text_splitters import CharacterTextSplitter

from .data_model import ContentResource
from ..models.core import llm
from clog import get_logger


logger = get_logger(__name__)


def parse_resource(task: str):
    """Detect a file or online resource link in a task using the LLM.

    Sends a prompt to the configured LLM asking it to identify whether the
    provided `task` contains a link to a file or an online resource. If the
    LLM identifies a resource, it should return a JSON object with the
    following keys: ``provided_by``, ``content``, and ``link``.

    The function returns a ``ContentResource`` built from the parsed JSON when
    a resource is found. If the LLM responds with the string ``NOT FOUND``,
    the function returns ``None``.

    Args:
        task: The user instruction or task string to inspect for a resource link.

    Returns:
        A ``ContentResource`` populated from the LLM output if a resource was
        identified; otherwise ``None``.

    Raises:
        json.JSONDecodeError: If the LLM indicates a resource but returns
            invalid JSON.
        KeyError: If the parsed JSON is missing required keys.
    """
    prompt = f"""
    You are given a task identify if it contains a link to a file or a resources online.
    If it does, produce a json with the following structure: 
    
    {{
     "provided_by": "user_task",
     "content": << a few words description of what could be in the file link/ online resource >>
     "link": <<identified link>>
    }}
    
    For example: 
    In:
    Task = "Search the following file ./user/pdev/audio_transcript.txt and see what the sentiment of this text is."
    Out:
    {{
     "provided_by": "user_task",
     "content": "user given audio transcript provided by the user"
     "link": "./user/pdev/audio_transcript.txt"
    }}
    In:
    Task = "See what the evaluation of AMD mentioned by this https://finance.yahoo.com/news/ais-valuation-problem.html"
    Out:
    {{
     "provided_by": "user_task",
     "content": "user given link to a financial article about AMD"
     "link": "https://finance.yahoo.com/news/ais-valuation-problem.html"
    }}
    In:
    Task = "Go online to wikipidia and search history of Prussia"
    Out: NOT FOUND
    
    IF no link is given, return a single string "NOT FOUND" 
    YOUR CURRENT TASK IS: {task}
    Start now:
    """

    response = llm.complete(prompt)

    if "NOT FOUND" not in response.text:
        json_content = response.text
        logger.info(f"JSON to parse to determine resources: {json_content}")
        loaded_dict = json.loads(json_content)

        return ContentResource(
            provided_by=loaded_dict["provided_by"],
            content=loaded_dict["content"],
            link=loaded_dict["link"],
            metadata={},
        )

    return None

def _process_chunk(task: str, text: str) -> str:
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


def process_text(task: str, text: str, chunk_size: int = 4000, chunk_overlap: int = 500) -> list[str]:
    """Splits a large text into chunks and processes each chunk with an LLM to perform the mentioned task..

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
        chunk_response = _process_chunk(task, chunk)
        if "NOT FOUND" not in chunk_response:
            responses.append(chunk_response)

    return responses