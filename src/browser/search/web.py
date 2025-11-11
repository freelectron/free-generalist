import os
from time import sleep, time

from requests import session

from browser import ChromeBrowser
from clog import get_logger


class BrowserWebSearch:
    base_url: str

    def __init__(self, browser: ChromeBrowser, session_id: str):
        self.logger = get_logger(name=self.__class__.__name__)
        self.browser = browser
        self.session_id = session_id

    def search(self, query: str):
        raise NotImplementedError()


class BraveBrowser(BrowserWebSearch):
    base_url: str = "https://search.brave.com/search?q="

    def __init__(self, browser: ChromeBrowser, session_id: str):
        super().__init__(browser=browser, session_id=session_id)

    def search(self, query: str) -> str:
        """
        Return raw html of for parsing from the search.
        """
        tab_id = self.session_id
        search_url = self.base_url + "+".join(query.split(" "))
        if tab_id not in self.browser.opened_tabs.keys():
            self.browser.driver.execute_script(f"window.open('{search_url}');")
            windows = self.browser.driver.window_handles
            self.browser.opened_tabs[tab_id] = windows[-1]
            self.browser.driver.switch_to.window(self.browser.opened_tabs[tab_id])
        else:
            self.browser.driver.switch_to.window(self.browser.opened_tabs[tab_id])
            self.browser.driver.get(search_url)

        self.browser.wait(2)

        return self.browser.driver.page_source