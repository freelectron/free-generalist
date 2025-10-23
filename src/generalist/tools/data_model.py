from dataclasses import dataclass
from typing import Optional, Any


@dataclass
class Task:
    question: str
    objective: str
    plan: list[str]


@dataclass
class ContentResource:
    """A unified dataclass for handling any type of resource that is being used for a task (web document, pdf file ).

    Attributes:
        provided_by: who supplied the resource (user or an agent/tool call) <=> as_answer_to
        content: The main text content of the web page. Can be None if not yet (down)loaded.
        link: The unique URL or file path for the resource.
        metadata: A dictionary containing additional information, such as search result data.
    """
    provided_by: str 
    content: Optional[str]
    link: str
    metadata: dict[str, Any]


@dataclass
class WebSearchResult: 
    link: str 
    metadata: dict


@dataclass
class ShortAnswer:
    """A dataclass to hold a structured answer from the LLM.

    Attributes:
        answer: The direct answer to the task, or "not found" if unavailable.
        clarification: A brief explanation of the context or reason for the answer.
    """
    answer: str = "not found"
    clarification: str = "No information processed yet."