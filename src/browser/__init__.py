from .browser import ChromeBrowser
from .llms.session import ChatGPT, DeepSeek
from .search.web import BraveBrowser

# TODO: Find a better way to interact with the browser and pass it to the tool
chrome_browser = ChromeBrowser()

# TODO: should session_id be the same name as the agent that is calling it?
BRAVE_SEARCH_SESSION = BraveBrowser(browser=chrome_browser, session_id="deep_web_search")

CHATGPT_SESSION = ChatGPT(chrome_browser, session_id="closed_ai")

DEEPSEEK_SESSION = DeepSeek(chrome_browser, session_id="deep_seek")
