from .browser import ChromeBrowser
from .search.web import BraveBrowser

# TODO: Find a better way to interact with the browser and pass it to the tool
chrome_browser = ChromeBrowser()
# session_id should be the same name as the agent that is calling it?!
BRAVE_SEARCH_SESSION  = BraveBrowser(browser=chrome_browser, session_id="deep_web_search")