from bs4 import BeautifulSoup, Tag

from browser import ChromeBrowser
from clog import get_logger


logger = get_logger(__name__)


class BrowserWebSearch:
    base_url: str

    def __init__(self, browser: ChromeBrowser, session_id: str):
        self.logger = get_logger(name=self.__class__.__name__)
        self.browser = browser
        self.session_id = session_id

    def search(self, query: str, max_results: int):
        raise NotImplementedError()


class BraveBrowser(BrowserWebSearch):
    base_url: str = "https://search.brave.com/search?q="

    def __init__(self, browser: ChromeBrowser, session_id: str):
        super().__init__(browser=browser, session_id=session_id)

    @staticmethod
    def parse_out_search_result(result: Tag):
        title = "N/A"
        link = "N/A"
        description = "N/A"

        # assumption: the link and title are within the same <a> tag
        link_element = result.find('a', href=True)
        if link_element and link_element.has_attr('href'):
            link = link_element['href']

            title_element = link_element.find('div', class_='title')
            if title_element:
                title = title_element.get_text(strip=True)

        description_element = result.find('div', class_='content')
        if description_element:
            description = description_element.get_text(strip=True)

        return {
            "title": title,
            "link": link,
            "description": description,
        }

    @staticmethod
    def parse_out_llm_result(result: Tag):
        for button in result.find_all('button', class_='inline-refs'):
            button.decompose()

        final_text = result.get_text(separator=' ', strip=True)

        return {
            "title": "Brave LLM",
            "description": final_text,
        }


    @staticmethod
    def  parse_search_results(html_content: str, n: int, query: str | None = None) -> list[dict]:
        """
        Parses the HTML content of a Brave search results page to extract
        the top 'n' search results.

        Args:
            html_content: The raw HTML string of the search page.
            n: The number of search results to retrieve.
            query: log the query that was used in the metadata

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
            if result:
                parsed_results.append(BraveBrowser.parse_out_search_result(result))
            else:
                logger.info(f"No results")

        # Add LLM summary
        result_container_llm_summary = soup.find_all('div', class_='chatllm-content')
        parsed_results.append(BraveBrowser.parse_out_llm_result(result_container_llm_summary[0]))

        return parsed_results

    def _raw_search(self, query: str) -> str:
        """
        Return raw html of for parsing from the search.
        """
        tab_id = self.session_id
        search_url = self.base_url + "+".join(query.split(" "))
        if tab_id not in self.browser.opened_tabs.keys():
            self.browser.driver.switch_to.new_window('tab')
            self.browser.wait(0.5)
            self.browser.driver.get(search_url)
            windows = self.browser.driver.window_handles
            self.browser.opened_tabs[tab_id] = windows[-1]
            self.browser.driver.switch_to.window(self.browser.opened_tabs[tab_id])
        else:
            self.browser.driver.switch_to.window(self.browser.opened_tabs[tab_id])
            self.browser.wait(0.5)
            self.browser.driver.get(search_url)

        self.browser.wait(10)

        return self.browser.driver.page_source

    def search(self, query: str, max_results: int) -> list[dict]:
        """
        Performs a Brave Web search and returns results as WebResource objects.

        Args:
            query: The search query string.
            max_results: The maximum number of search results to retrieve.

        Returns:
            A list of WebResource objects, where 'content' is None and 'metadata'
            contains the search result details.
        """

        search_browser_page = self._raw_search(query=query)

        results = self.parse_search_results(search_browser_page, max_results, query=query)

        return results