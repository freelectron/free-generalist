from datetime import datetime
from collections import deque
from time import time

import tiktoken

from browser.llms.session import LLMSession
from browser.llms.session import ChatGPT, DeepSeek, Gemini, Qwen, Claude, Mistral
from browser.browser import ChromeBrowser
from clog import get_logger


SMALL_CONTEXT_THRESHOLD = 32000
INITIAL_TOKENS = 3
PENALTY_TOKENS = 1
TIMEOUT_HOURS = 4

logger = get_logger(__name__)


class LLMBrowser:
    """
    Manages a pool of LLM sessions. Each session has a token credit.
    On failure the credit is reduced by PENALTY_TOKENS. When credit drops
    below zero the session is moved to a FIFO timeout queue for TIMEOUT_HOURS,
    then returned to the active pool.
    """

    def create_long_message_sessions(self) -> list[LLMSession]:
        """Sessions that allow a high number of tokens in a single paste."""
        return [
            self.CLAUDE_SESSION,
            self.DEEPSEEK_SESSION,
            self.MISTRAL_SESSION,
            self.QWEN_SESSION,
        ]

    def create_small_message_sessions(self) -> list[LLMSession]:
        """Sessions that can only (easily) handle a low number of tokens."""
        return [self.GEMINI_SESSION, self.CHATGPT_SESSION]

    def __init__(self, chrome_browser: ChromeBrowser):
        self.estimattion_n_token_encoding = tiktoken.encoding_for_model("gpt-4o")
        self.n_call = 0

        self.CHATGPT_SESSION = ChatGPT(chrome_browser, session_id="closed_ai")
        self.DEEPSEEK_SESSION = DeepSeek(chrome_browser, session_id="deep_seek")
        self.GEMINI_SESSION = Gemini(chrome_browser, session_id="gemini")
        self.QWEN_SESSION = Qwen(chrome_browser, session_id="qwen")
        self.CLAUDE_SESSION = Claude(chrome_browser, session_id="claude")
        self.MISTRAL_SESSION = Mistral(chrome_browser, session_id="mistral")

        long_sessions = self.create_long_message_sessions()
        self.long_sessions: list[list] = [[s, INITIAL_TOKENS] for s in long_sessions]

        small_sessions = self.create_small_message_sessions()
        self.small_sessions: list[list] = [[s, INITIAL_TOKENS] for s in small_sessions]

        self.timeout_queue: deque[tuple[LLMSession, float, str]] = deque()

    def _release_timed_out_sessions(self):
        """Move sessions whose timeout has expired back to their original pool."""
        now = time()
        while self.timeout_queue and self.timeout_queue[0][1] <= now:
            session, _, pool_name = self.timeout_queue.popleft()
            pool = self.long_sessions if pool_name == "long" else self.small_sessions
            pool.append([session, INITIAL_TOKENS])

    def _move_to_timeout(self, entry: list, pool: list[list]):
        """Remove entry from its pool and enqueue it for TIMEOUT_HOURS."""
        pool_name = "long" if pool is self.long_sessions else "small"
        pool.remove(entry)
        self.timeout_queue.append((entry[0], time() + TIMEOUT_HOURS * 3600, pool_name))

    def _try_sessions(self, pool: list[list], message: str) -> str:
        """
        Try each session in pool in order. On success rotate it to the end.
        On failure apply penalty; if tokens < 0 move to timeout queue.
        Raises RuntimeError if all sessions fail.
        """
        for entry in list(pool):
            session, token_credit = entry
            try:
                logger.info(f"Trying {type(session).__name__} ...")
                answer = session.send_message(message)

                pool.remove(entry)
                pool.append(entry)

                self.n_call += 1
                return answer

            except Exception as e:
                logger.error(
                    f"Error occurred when running {session} with token credit {token_credit}: {e}"
                )
                entry[1] -= PENALTY_TOKENS
                if entry[1] < 0:
                    self._move_to_timeout(entry, pool)

        raise RuntimeError("All LLM sessions failed.")

    def call(self, message: str) -> str:
        self._release_timed_out_sessions()
        logger.info(
            f"Queue for calling LLM's with long message window: "
            f"{[(type(sesh).__name__, tk_credit) for sesh, tk_credit in self.long_sessions]}"
        )

        if len(message) < SMALL_CONTEXT_THRESHOLD and self.small_sessions:
            return self._try_sessions(self.small_sessions, message)

        return self._try_sessions(self.long_sessions, message)


if __name__ == "__main__":
    import dotenv

    dotenv.load_dotenv()

    # NOTE: ChromeBrowser instance is required here
    # browser = ChromeBrowser(...)
    # llm_browser = LLMBrowser(browser)
    # print(llm_browser.call("hello"))
    # print(llm_browser.call("ola"))

    print("LLMBrowser module loaded. Provide a ChromeBrowser instance to run.")