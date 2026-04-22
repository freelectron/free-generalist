from typing import Optional, List, Dict, Any
from bs4 import BeautifulSoup
import httpx

from browser.search.web import BraveBrowser
from ..tools.data_model import WebSearchResult
from ..models.core import MLFlowLLMWrapper
from ..tools.base import BaseTool
from clog import get_logger


NOT_FOUND_LITERAL = "N/A"
DEFAULT_TIMEOUT = 15.0
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'

logger = get_logger(__name__)


def generate_search_queries(question: str, max_queries: int, llm: MLFlowLLMWrapper) -> list[str]:
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

    try:
        response = llm.complete(prompt)
        response_text = response.text if hasattr(response, 'text') else str(response)
        queries = [q.strip() for q in response_text.strip().split("|") if q.strip()]
        return queries[:max_queries] if queries else [question]
    except Exception as e:
        logger.error(f"Query generation failed, falling back to original question: {e}")

        return [question]


class WebSearchTool(BaseTool):
    name = "web_search"
    description = "Searches the web and downloads page content for a given question."

    def __init__(self, search_session: BraveBrowser, llm: MLFlowLLMWrapper):
        self.search_session = search_session
        self.llm = llm

    def run(self, question: str, n_queries: int = 1, links_per_query: int = 1) -> List[Dict[str, Any]]:
        """
        Searches the web and returns downloaded page contents for the given question.

        Args:
            question: The user's query or question.
            n_queries: Number of search queries to generate from the question.
            links_per_query: Number of links to fetch per search query.

        Returns:
            A list of dicts with search result metadata and cleaned page content.
        """
        candidate_queries = generate_search_queries(question, n_queries, self.llm)
        logger.info(f"Generated search queries: {candidate_queries}")

        all_sources = []
        for query in candidate_queries:
            try:
                raw_results = self.search_session.search(query, links_per_query)
                if raw_results:
                    all_sources.extend(self._parse_results(raw_results, query))
            except Exception as e:
                logger.error(f"Search session failed for query '{query}': {e}")

        unique_results = self._drop_non_unique_links(all_sources)
        logger.info(f"Retrieved {len(unique_results)} unique links.")

        final_resources = []
        for search in unique_results:
            content = None
            if search.link != NOT_FOUND_LITERAL:
                content = self._download_content(search)

            if not content:
                content = search.metadata.get("web_page_summary")

            if content and content != NOT_FOUND_LITERAL:
                final_resources.append({
                    "search_result": search,
                    "content": content,
                })

        return final_resources

    def _extract_clean_text(self, raw_html: str) -> str:
        try:
            soup = BeautifulSoup(raw_html, 'html.parser')
            for element in soup(["script", "style", "nav", "footer", "header", "aside", "form", "svg"]):
                element.decompose()
            text = soup.get_text(separator=" ")
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            return ' '.join(chunk for chunk in chunks if chunk)
        except Exception as e:
            logger.warning(f"HTML text extraction failed: {e}")
            return ""

    def _download_content(self, resource: WebSearchResult) -> Optional[str]:
        if not resource.link or resource.link == NOT_FOUND_LITERAL or not resource.link.startswith('http'):
            return None
        headers = {
            'User-Agent': USER_AGENT,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Encoding': 'gzip, deflate',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': resource.link,
        }
        try:
            with httpx.Client(follow_redirects=True, timeout=DEFAULT_TIMEOUT) as client:
                response = client.get(resource.link, headers=headers)
                response.raise_for_status()
                return self._extract_clean_text(response.text)
        except httpx.HTTPError as e:
            logger.warning(f"HTTP error downloading {resource.link}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error downloading {resource.link}: {e}")
        return None

    def _drop_non_unique_links(self, resources: List[WebSearchResult]) -> List[WebSearchResult]:
        seen_links = set()
        unique_resources = []
        for resource in resources:
            if resource.link and resource.link not in seen_links:
                unique_resources.append(resource)
                seen_links.add(resource.link)
        return unique_resources

    def _parse_results(self, results: List[Dict[str, Any]], query: str) -> List[WebSearchResult]:
        found_resources = []
        for i, result in enumerate(results):
            resource = WebSearchResult(
                link=result.get('link', NOT_FOUND_LITERAL),
                metadata={
                    "search_order": i,
                    "web_page_title": result.get('title', NOT_FOUND_LITERAL),
                    "web_page_summary": result.get('description', NOT_FOUND_LITERAL),
                    "query": query,
                }
            )
            found_resources.append(resource)
        return found_resources
