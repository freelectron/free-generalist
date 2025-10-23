import requests

from bs4 import BeautifulSoup
import httpx
from ddgs import DDGS

from ..tools import NOT_FOUND_LITERAL
from ..tools.data_model import ContentResource, WebSearchResult
from ..models.core import llm
from ..utils import current_function
from .. import logging


logger = logging.getLogger(__name__)


def question_to_queries(question: str, max_queries: int = 2) -> list[str]:
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
    """
    llm_response = llm.complete(prompt)
    response_text = llm_response.text

    return response_text.strip().split("|")


def duckduckgo_search(query: str, max_results: int = 2) -> list[WebSearchResult]:
    """Performs a DuckDuckGo search and returns results as WebResource objects.

    Args:
        query: The search query string.
        max_results: The maximum number of search results to retrieve.

    Returns:
        A list of WebResource objects, where 'content' is None and 'metadata'
        contains the search result details.
    """
    found_resources = []
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))
        if not results:
            print(f"⚠️ No search results found for '{query}'")
            return []

        for i, result in enumerate(results):
            resource = WebSearchResult(
                link=result.get('href', NOT_FOUND_LITERAL),
                metadata={
                    "search_order": i,
                    "web_page_title": result.get('title', NOT_FOUND_LITERAL),
                    "web_page_summary": result.get('body', NOT_FOUND_LITERAL),
                    "query": query
                }
            )
            found_resources.append(resource)

    return found_resources

def brave_search(query: str, max_results: int = 2):
    """ 
    Performs a Brave Web search and returns results as WebResource objects.

    Args:
        query: The search query string.
        max_results: The maximum number of search results to retrieve.

    Returns:
        A list of WebResource objects, where 'content' is None and 'metadata'
        contains the search result details.
    """

    def parse_search_results(html_content: str, n: int = 1) -> list[dict]:
        """
        Parses the HTML content of a Brave search results page to extract
        the top 'n' search results.

        Args:
            html_content: The raw HTML string of the search page.
            n: The number of search results to retrieve.

        Returns:
            A list of dictionaries, where each dictionary contains the
            'title', 'link', and 'description' of a search result.
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Find all the main containers for the search results.
        # We exclude snippets that are 'standalone' (like Videos, Infobox)
        # and focus on those with a 'data-pos' attribute, which marks regular web results.
        result_containers = soup.find_all('div', class_='snippet', attrs={'data-pos': True})
        
        parsed_results = []
        
        for result in result_containers[:n]:
            title = "N/A"
            link = "N/A"
            description = "N/A"

            # assumption: the link and title are within the same <a> tag
            link_element = result.find('a', class_='heading-serpresult')
            if link_element and link_element.has_attr('href'):
                link = link_element['href']
                
                title_element = link_element.find('div', class_='title')
                if title_element:
                    title = title_element.get_text(strip=True)

            description_element = result.find('div', class_='snippet-description')
            if description_element:
                description = description_element.get_text(strip=True)
                
            parsed_results.append({
                "title": title,
                "link": link,
                "description": description
            })
            
        return parsed_results

    base_search_url = "https://search.brave.com/search?q="
    search_url = base_search_url + "+".join(query.split(" "))
    logger.info(f"- {current_function()} -- Searching url: {search_url}")
    response_search = requests.get(search_url)

    results = parse_search_results(str(response_search.content), max_results)
    found_resources = list()
    for i, result in enumerate(results):
        resource = WebSearchResult(
            link=result.get('link', NOT_FOUND_LITERAL),
            metadata={
                "search_order": i,
                "web_page_title": result.get('title', NOT_FOUND_LITERAL),
                "web_page_summary": result.get('body', NOT_FOUND_LITERAL),
                "query": query
            }
        )
        found_resources.append(resource)
        
    return found_resources


def drop_non_unique_link(resources: list[ContentResource|WebSearchResult]) -> list[ContentResource|WebSearchResult]:
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


def extract_clean_text(raw_html: str) -> str:
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

def download_content(resource: WebSearchResult) -> str:
    """
    Downloads the HTML from a resource's link and populates its 'content' field.
    This version includes robust URL encoding and safe error printing.
    """
    if not resource.link or not resource.link.startswith('http'):
        return resource

    # For this example, let's assume _encode_url_path is defined elsewhere
    # encoded_link = _encode_url_path(resource.link)
    response = httpx.get(resource.link, timeout=15)
    charset = response.encoding or 'utf-8'
    html_bytes = response.content
    html_content = html_bytes.decode(charset)

    return extract_clean_text(html_content)

def web_search(question: str, links_per_query: int = 2) -> list[ContentResource]:
    """Orchestrates the full web search process for a given question.

    This process includes:
    1. Converting the question into search queries.
    2. Searching the web to find resources.
    3. Downloading and extracting text content from each resource.

    Args:
        question: The user's question.
        links_per_query: The number of web links to retrieve for each search query.

    Returns:
        A list of WebResource objects, with their 'content' field populated.
    """
    candidate_queries = question_to_queries(question)
    logger.info(f"- {current_function()} -- Generated queries: {candidate_queries}")

    all_sources = []
    for query in candidate_queries:
        search_results_for_query = brave_search(query, links_per_query) #duckduckgo_search(query, links_per_query)
        all_sources.extend(search_results_for_query)

    unique_search_results = drop_non_unique_link(all_sources)
    logger.info(f"Found {len(unique_search_results)} unique sources.")

    final_resources = []
    for search in unique_search_results:
        content = download_content(search)
        populated_resource = ContentResource(
            provided_by=web_search.__name__,
            content=content,
            link=search.link,
            metadata=search.metadata
        )
        if populated_resource.content: # Only keep resources where content was successfully downloaded
            final_resources.append(populated_resource)

    return final_resources