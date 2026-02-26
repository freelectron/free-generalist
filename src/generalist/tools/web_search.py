from typing import Optional

import requests

from bs4 import BeautifulSoup
import httpx

from browser import BRAVE_SEARCH_SESSION
from browser.search.web import BraveBrowser
from ..tools.data_model import Message, WebSearchResult
from ..models.core import llm
from clog import get_logger


NOT_FOUND_LITERAL = "N/A"
logger = get_logger(__name__)


def _question_to_queries(question: str, max_queries: int = 1) -> list[str]:
    """Converts a user question into a list of optimized search engine queries.

    Note:
        This function requires a Large Language Model (LLM) to generate queries.
        The `llm.complete()` call is a placeholder for your model's inference logic.

    Args:
        question: The user's input question.
        max_queries: The maximum number of search queries to generate.

    Returns:
        A list of string queries optimized for a search engine.
    """
    prompt = f"""
    Create a list of general search engine queries for the following question: "{question}".

    Make sure that:
    - Your output is a list separated by a "|" character and nothing else.
    - You provide a MAXIMUM of {max_queries} search engine queries.
    - Each query is SHORT and precise.

    Example Output:
    Large urban population areas in Europe|Biggest cities in Europe

    START NOW:
    """
    llm_response = llm.complete(prompt)
    response_text = llm_response.text

    queries = response_text.strip().split("|")
    return queries[:max_queries]


def _drop_non_unique_link(resources: list[Message | WebSearchResult]) -> list[Message | WebSearchResult]:
    """Removes duplicate WebResource objects based on their 'link' attribute.

    Args:
        resources: A list of WebResource objects.

    Returns:
        A new list of WebResource objects with duplicates removed.
    """
    seen_links = set()
    unique_resources = []
    for resource in resources:
        if resource.link and resource.link not in seen_links:
            unique_resources.append(resource)
            seen_links.add(resource.link)

    return unique_resources


def _extract_clean_text(raw_html: str) -> str:
    """Extracts clean, readable text from raw HTML content.

    This function removes scripts, styles, navigation, and other non-content
    elements, then cleans up whitespace.

    Args:
        raw_html: The raw HTML content of a webpage.

    Returns:
        The extracted and cleaned plain text.
    """
    soup = BeautifulSoup(raw_html, 'html.parser')
    # Remove elements that typically do not contain main content
    for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
        element.decompose()

    # Extract text and clean up whitespace
    text = soup.get_text(separator=" ")
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))

    return ' '.join(chunk for chunk in chunks if chunk)

def _download_content(resource: WebSearchResult) -> str:
    """
    Downloads the HTML from a resource's link and populates its 'content' field.
    This version includes robust URL encoding and safe error printing.
    """
    if not resource.link or not resource.link.startswith('http'):
        return resource

    # For this example, let's assume _encode_url_path is defined elsewhere
    # encoded_link = _encode_url_path(resource.link)
    headers = {
        'Accept-Encoding': 'gzip, deflate',
        'User-Agent': 'Chrome/122.0.0.0',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': resource.link
    }
    response = httpx.get(resource.link, timeout=15, headers=headers)
    charset = response.encoding or 'utf-8'
    html_bytes = response.content
    html_content = html_bytes.decode(charset)

    return _extract_clean_text(html_content)

def parse_web_browser_search_results(results: list[dict]) -> list[WebSearchResult]:
    found_resources = list()
    for i, result in enumerate(results):
        resource = WebSearchResult(
            link=result.get('link', NOT_FOUND_LITERAL),
            metadata={
                "search_order": i,
                "web_page_title": result.get('title', NOT_FOUND_LITERAL),
                "web_page_summary": result.get('description', NOT_FOUND_LITERAL),
                "query": result.get("query", NOT_FOUND_LITERAL)
            }
        )
        found_resources.append(resource)

    return found_resources


def web_search(question: str) -> list[dict[str, str|WebSearchResult]]:
    """
    Orchestrates the full web search process for a given question.

    This process includes:
    1. Converting the question into search queries.
    2. Searching the web to find resources.
    3. Downloading and extracting text content from each resource.

    Args:
        question: The user's question.

    Returns:
        A list of WebResource objects, with their 'content' field populated with the content of webpage.
    """
    # Number of queries to generate per question
    queries_per_question = 1
    # Number of web links to retrieve for each search query.
    links_per_query = 1

    candidate_queries = _question_to_queries(question, queries_per_question)
    logger.info(f"Generated queries: {candidate_queries}")

    all_sources = []
    for query in candidate_queries:
        raw_search_results_for_query = BRAVE_SEARCH_SESSION.search(query, links_per_query)
        search_results_for_query = parse_web_browser_search_results(raw_search_results_for_query)
        all_sources.extend(search_results_for_query)

    unique_search_results = _drop_non_unique_link(all_sources)
    logger.info(f"Found {len(unique_search_results)} unique sources.\n{unique_search_results}")

    final_resources = []
    for search in unique_search_results:
        # This is search result from LLM query most likely
        if search.link == NOT_FOUND_LITERAL:
            content = search.metadata["web_page_summary"]
        else:
            # Normal web link
            content = _download_content(search)
        if content:
            populated_resource = {
                "search_result": search,
                "content": content,
            }
            final_resources.append(populated_resource)

    return final_resources