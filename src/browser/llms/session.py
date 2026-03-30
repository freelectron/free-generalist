import os
from time import sleep, time

import pyperclip
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys

from browser import ChromeBrowser
from browser.errors import (
    BrowserTimeOutError,
    BrowserStayLoggedOutFailed,
    LogInError,
    MessageNotSentError,
)
from clog import get_logger


class LLMSession:
    llm_chat_url = None
    waiter_default_timeout = 60

    def __init__(self, browser: ChromeBrowser, session_id: str):
        self.logger = get_logger(name=self.__class__.__name__)
        self.browser = browser
        self.session_id = session_id
        self.past_questions_answers = list()

    def _validate_start_page_loaded(self):
        """Check if the LLM's start page is loaded."""
        raise NotImplementedError("This method should be implemented in subclasses.")

    def _activate_chat_session(self):
        """
        Open chatGPT or switch to the tab that has it
        """
        tab_id = self.llm_chat_url + self.session_id
        if tab_id not in self.browser.opened_tabs.keys():
            self.browser.driver.switch_to.new_window('tab')
            self.browser.driver.get(self.llm_chat_url)
            windows = self.browser.driver.window_handles
            self.browser.opened_tabs[tab_id] = windows[-1]
            self.browser.driver.switch_to.window(self.browser.opened_tabs[tab_id])

        self.browser.driver.switch_to.window(self.browser.opened_tabs[tab_id])
        self.browser.wait(2)
        self._validate_start_page_loaded()

    def _retrieve_last_answer(self, time_out: int):
        raise NotImplementedError()

    def _validate_message_sent(self):
        last_answer_memory = ""
        last_answer_on_page = ""
        # ToDo: use a datastruct to access the answer attribute
        last_answer_memory = (
            self.past_questions_answers[-1]["answer"]
            if self.past_questions_answers
            else ""
        )
        last_answer_on_page = self._retrieve_last_answer(self.waiter_default_timeout)

        # FixMe: this is a bad check to actually see if the message was sent
        if last_answer_memory == last_answer_on_page or last_answer_on_page == "":
            self.logger.error(f"Last answer in memory = current answer? {last_answer_memory == last_answer_on_page}")
            self.logger.error(f"Got empty answer after validation: {last_answer_memory == last_answer_on_page}")
            raise MessageNotSentError("No new response from LLM")
        else:
            return last_answer_on_page

    def _send_message(self, message: str):
        raise NotImplementedError
    
    def send_message(self, message: str):
        self._activate_chat_session()
        # TODO: this is suggestion from Claude, as a solution to error with non-BMP characters 
        message = "".join(c for c in message if ord(c) <= 0xFFFF)
        return self._send_message(message)

    def clean_chat_history(self):
        pass 


class ChatGPT(LLMSession):
    logging_file = "llm_browser_session_openai.log"
    llm_chat_url = "https://chat.openai.com/chat"

    def __init__(self, browser: ChromeBrowser, session_id: str = None):
        super().__init__(browser, session_id)

    def _retrieve_last_answer(self, time_out: int, n_tries: int = 3):
        start_time = time()
        last_answer = ""
        for i in range(n_tries):
            # TODO: check if this will wait for till the content is fully loaded
            answer = self.browser.waiter.until(
                EC.presence_of_all_elements_located(
                    (By.CSS_SELECTOR, "div[data-message-author-role='assistant']")
                )
            )[-1]
            # TODO: how to get notified that the new message has fully arrived?
            #  Now it is done by waiting and checking against the state of it
            if len(answer.text) > len(last_answer):
                last_answer = answer.text
            else:
                if time() - start_time > time_out:
                    break
            self.browser.wait(1)

        return last_answer

    def _validate_start_page_loaded(self, n_tries: int = 2):
        for i in range(n_tries):
            html_source = self.browser.driver.page_source
            if 'content="ChatGPT"><meta' in html_source:
                return
            else:
                self.browser.wait(10)

        raise BrowserTimeOutError(
            "Failed to start chat session. Page did not load correctly."
        )

    def _send_message(self, message: str):
        editor_div = self.browser.waiter.until(
            EC.element_to_be_clickable((By.ID, "prompt-textarea"))
        )
        editor_div.click()
        self.browser.random_mouse_move(2)
        self.browser.wait(1)

        # New way: Inject text via execCommand to trigger React's synthetic event system
        self.browser.driver.execute_script("""
            arguments[0].focus();
            document.execCommand('selectAll', false, null);
            document.execCommand('insertText', false, arguments[1]);
        """, editor_div, message)

        editor_div.send_keys(Keys.ENTER)
        # Wait till the llm the first token, that when the div for the answer appears
        # TODO: see a better way to wait for an answer
        self.browser.wait(15)

        # We assume that the answer's div is already present
        answer = self._validate_message_sent()
        # ToDo: create a datastruct for this
        self.past_questions_answers.append({"message": message, "answer": answer})

        return answer


class DeepSeek(LLMSession):
    logging_file = "llm_browser_session_deepseek.log"
    llm_chat_url = "https://chat.deepseek.com/"

    def __init__(self, browser: ChromeBrowser, session_id: str = None):
        super().__init__(browser, session_id)
        self.email = (
            os.environ.get("DEEPSEEK_EMAIL")
            if os.environ.get("DEEPSEEK_EMAIL")
            else None
        )
        self.password = (
            os.environ.get("DEEPSEEK_PASSWORD")
            if os.environ.get("DEEPSEEK_PASSWORD")
            else None
        )
        if not self.email or not self.password:
            raise ValueError(
                "DeepSeek email and password must be set in environment variables."
            )

    def _validate_start_page_loaded(self):
        self.browser.wait(3)
        if "Only login via" in self.browser.driver.page_source:
            self.logger.info("Trying to log in to DeepSeek.")
            input_field_css_placeholder_email = self.browser.waiter.until(
                EC.element_to_be_clickable(
                    (
                        By.CSS_SELECTOR,
                        "input[placeholder='Phone number / email address']",
                    )
                )
            )
            self.logger.info("Filling in email field.")
            input_field_css_placeholder_email.click()
            input_field_css_placeholder_email.send_keys(self.email)

            input_field_css_placeholder_password = self.browser.waiter.until(
                EC.element_to_be_clickable(
                    (By.CSS_SELECTOR, "input[placeholder='Password']")
                )
            )
            self.logger.info("Filling in password field.")
            input_field_css_placeholder_password.click()
            input_field_css_placeholder_password.send_keys(self.password)

            login_button_xpath_text = self.browser.waiter.until(
                EC.element_to_be_clickable(
                    (By.XPATH, "//div[text()='Log in' and @role='button']")
                )
            )
            login_button_xpath_text.click()
            self.browser.wait(2)

        if (
            "How can I help you today?" in self.browser.driver.page_source
            or "Message DeepSeek" in self.browser.driver.page_source
        ):
            self.logger.info("DeepSeek chat page loaded successfully.")
            self.browser.wait(1)
            return
        else:
            raise LogInError(
                "Failed to start chat session. Page did not load correctly."
            )

    def _retrieve_last_answer(self, time_out: int):
        start_time = time()
        last_answer = ""
        while True:
            answer = self.browser.waiter.until(
                EC.presence_of_all_elements_located(
                    (
                        By.XPATH,
                        "//div[contains(@class, 'ds-markdown')]",
                    )
                )
            )[-1]
            if len(answer.text) > len(last_answer):
                last_answer = answer.text
            elif len(answer.text) == len(last_answer):
                break
            else:
                if time() - start_time > time_out:
                    break
            self.browser.wait(3)

        return last_answer

    def _send_message(self, message: str):
        xpath_locator = "//textarea[@placeholder='Message DeepSeek']"
        chat_input_textarea = self.browser.waiter.until(
            EC.element_to_be_clickable((By.XPATH, xpath_locator))
        )
        chat_input_textarea.click()

        self.browser.driver.execute_script("""
            arguments[0].focus();
            document.execCommand('selectAll', false, null);
            document.execCommand('insertText', false, arguments[1]);
        """, chat_input_textarea, message)

        chat_input_textarea.send_keys(Keys.ENTER)
        # TODO: see a better way to wait for an answer
        self.browser.wait(15)

        answer = self._validate_message_sent()
        # ToDo: create a datastruct for this
        self.past_questions_answers.append({"message": message, "answer": answer})

        return answer


class Gemini(LLMSession):
    logging_file = "llm_browser_session_gemini.log"
    llm_chat_url = "https://gemini.google.com/app"

    def __init__(self, browser: ChromeBrowser, session_id: str = None):
        super().__init__(browser, session_id)

    def _retrieve_last_answer(self, time_out: int, n_tries: int = 3):
        start_time = time()
        last_answer = ""
        for i in range(n_tries):
            # TODO: check if this will wait for till the content is fully loaded
            answer = self.browser.waiter.until(
                EC.presence_of_all_elements_located(
                    (By.CSS_SELECTOR, "model-response")
                )
            )[-1]
            # TODO: how to get notified that the new message has fully arrived?
            #  Now it is done by waiting and checking against the state of it
            if len(answer.text) > len(last_answer):
                last_answer = answer.text
            else:
                if time() - start_time > time_out:
                    break
            self.browser.wait(2)

        return last_answer

    def _validate_start_page_loaded(self, n_tries: int = 2):
        for i in range(n_tries):
            html_source = self.browser.driver.page_source
            if 'class="chat-app' in html_source:
                return
            else:
                self.browser.wait(5)

        raise BrowserTimeOutError(
            "Failed to start chat session. Page did not load correctly."
        )

    def _send_message(self, message: str):
        editor_div = self.browser.waiter.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "div.input-area"))
        )
        editor_div.click()
        self.browser.random_mouse_move(2)
        self.browser.wait(1)

        self.browser.driver.execute_script("""
            arguments[0].focus();
            document.execCommand('selectAll', false, null);
            document.execCommand('insertText', false, arguments[1]);
        """, editor_div, message)

        send_button = self.browser.waiter.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "div.input-area button.send-button"))
        )
        send_button.click()
        # Wait till the llm the first token, that when the div for the answer appears
        # TODO: see a better way to wait for an answer
        self.browser.wait(10)

        # We assume that the answer's div is already present
        answer = self._validate_message_sent()
        # ToDo: create a datastruct for this
        self.past_questions_answers.append({"message": message, "answer": answer})

        return answer

    def pass_checks(self):
        """Click 'Stay logged out' link that needs to be clicked or something of this sort"""
        try:
            stay_logged_out_link = self.browser.waiter.until(
                EC.element_to_be_clickable((By.LINK_TEXT, "Stay logged out"))
            )
            stay_logged_out_link.click()
        except Exception as e:
            raise BrowserStayLoggedOutFailed(e.__str__())


class Qwen(LLMSession):
    logging_file = "llm_browser_session_qwen.log"
    llm_chat_url = "https://chat.qwen.ai"

    def __init__(self, browser: ChromeBrowser, session_id: str = None):
        super().__init__(browser, session_id)
        self.email = (
            os.environ.get("QWEN_EMAIL")
            if os.environ.get("QWEN_EMAIL")
            else None
        )
        self.password = (
            os.environ.get("QWEN_PASSWORD")
            if os.environ.get("QWEN_PASSWORD")
            else None
        )
        if not self.email or not self.password:
            raise ValueError(
                "Qwen email and password must be set in environment variables."
            )

    def _validate_start_page_loaded(self):
        self.browser.wait(3)
        if "message-input" in self.browser.driver.page_source:
            self.logger.info("Qwen chat page loaded successfully.")
            self.browser.wait(1)
            return
        else:
            raise LogInError(
                "Failed to start chat session. Page did not load correctly."
            )

    def _retrieve_last_answer(self, time_out: int):
        start_time = time()
        last_answer = ""
        while True:
            answer = self.browser.waiter.until(
                EC.presence_of_all_elements_located(
                    (
                        By.CLASS_NAME,
                        "qwen-chat-message",
                    )
                )
            )[-1]
            if "Thinking" in answer.text[:50] and not "Thinking completed" in answer.text[:50]:  # LOL 
                continue
            else:
                if len(answer.text) > len(last_answer):
                    last_answer = answer.text
                else:
                    if time() - start_time > time_out:
                        break
            self.browser.wait(4)

        # Do one last try to retrieve the full answer
        last_answer = self.browser.waiter.until(
            EC.presence_of_all_elements_located(
                (
                    By.CLASS_NAME,
                    "qwen-chat-message",
                )
            )
        )[-1].text
        return last_answer

    def _send_message(self, message: str):
        xpath_locator = "//textarea[@placeholder='How can I help you today?']"
        chat_input_textarea = self.browser.waiter.until(
            EC.element_to_be_clickable((By.XPATH, xpath_locator))
        )
        chat_input_textarea.click()

        if len(message) < 40000:
            self.browser.driver.execute_script("""
                arguments[0].focus();
                document.execCommand('selectAll', false, null);
                document.execCommand('insertText', false, arguments[1]);
            """, chat_input_textarea, message)
        else:
            # This only works because there is interface setting "Paste Large Text as File"
            pyperclip.copy(message)
            ActionChains(self.browser.driver) \
                .key_down(Keys.COMMAND) \
                .send_keys('v') \
                .key_up(Keys.COMMAND) \
                .perform()
            self.browser.wait(10) # wait till qwen loads the text as file

        chat_input_textarea = self.browser.driver.find_element(By.XPATH, xpath_locator)
        chat_input_textarea.send_keys(Keys.ENTER)
        self.browser.wait(1)
        pyperclip.copy("")  # free clipboard
        chat_input_textarea = self.browser.driver.find_element(By.XPATH, xpath_locator)
        chat_input_textarea.send_keys(Keys.ENTER) # do again just in case?
        # TODO: see a better way to wait for an answer
        self.browser.wait(20) # qwen in thinking mode by default, thinks long time

        answer = self._validate_message_sent()
        # ToDo: create a datastruct for this
        self.past_questions_answers.append({"message": message, "answer": answer})

        return answer


class Claude(LLMSession):
    logging_file = "llm_browser_session_claude.log"
    llm_chat_url = "https://claude.ai/new"

    def __init__(self, browser: ChromeBrowser, session_id: str = None):
        super().__init__(browser, session_id)

    def _validate_start_page_loaded(self, n_tries: int = 2):
        self.browser.wait(2)
        for i in range(n_tries):
            html_source = self.browser.driver.page_source
            if 'main-content' in html_source:
                return
            else:
                self.browser.wait(2)

        raise BrowserTimeOutError(
            "Failed to start chat session. Page did not load correctly."
        )

    def _retrieve_last_answer(self, time_out: int):
        start_time = time()
        last_answer = ""
        while True:
            answer = self.browser.waiter.until(
                EC.presence_of_all_elements_located(
                    (
                        By.CSS_SELECTOR,
                        ".font-claude-response",
                    )
                )
            )[-1]
            if len(answer.text)> 0:
                if len(answer.text) > len(last_answer):
                    last_answer = answer.text
                else:
                    break
            else:
                if time() - start_time > time_out:
                    break
            self.browser.wait(2)

        return last_answer

    def _send_message(self, message: str):
        css_selector = "//*[@data-testid='chat-input']"
        chat_input_textarea = self.browser.waiter.until(
            EC.element_to_be_clickable((By.XPATH, css_selector))
        )
        chat_input_textarea.click()

        self.browser.driver.execute_script("""
            arguments[0].focus();
            document.execCommand('selectAll', false, null);
            document.execCommand('insertText', false, arguments[1]);
        """, chat_input_textarea, message)

        chat_input_textarea.send_keys(Keys.ENTER)
        # TODO: see a better way to wait for an answer
        self.browser.wait(15)

        answer = self._validate_message_sent()
        # ToDo: create a datastruct for this
        self.past_questions_answers.append({"message": message, "answer": answer})

        return answer


class Mistral(LLMSession):
    logging_file = "llm_browser_session_mistral.log"
    llm_chat_url = "https://chat.mistral.ai/chat"

    def __init__(self, browser: ChromeBrowser, session_id: str = None):
        super().__init__(browser, session_id)

    def _validate_start_page_loaded(self, n_tries: int = 2):
        self.browser.wait(2)
        for i in range(n_tries):
            html_source = self.browser.driver.page_source
            if '@container/chat-input-row' in html_source:
                return
            else:
                self.browser.wait(2)

        raise BrowserTimeOutError(
            "Failed to start chat session. Page did not load correctly."
        )

    def _retrieve_last_answer(self, time_out: int):
        start_time = time()
        last_answer = ""
        while True:
            answer = self.browser.waiter.until(
                EC.presence_of_all_elements_located(
                    (
                        By.XPATH,
                        '//div[@data-message-author-role="assistant"]',
                    )
                )
            )[-1]
            if len(answer.text) > 0 :
                if len(answer.text) > len(last_answer):
                    last_answer = answer.text
                else:
                    break
            if time() - start_time > time_out:
                break
            self.browser.wait(5)

        return last_answer

    def _send_message(self, message: str):
        xpath_selector = "//div[@contenteditable='true' and contains(@class, 'ProseMirror')]"
        chat_input_textarea = self.browser.waiter.until(
            EC.element_to_be_clickable((By.XPATH, xpath_selector))
        )
        chat_input_textarea.click()

        self.browser.driver.execute_script("""
            arguments[0].focus();
            document.execCommand('selectAll', false, null);
            document.execCommand('insertText', false, arguments[1]);
        """, chat_input_textarea, message)

        chat_input_textarea.send_keys(Keys.ENTER)
        # TODO: see a better way to wait for an answer
        self.browser.wait(15)

        answer = self._validate_message_sent()
        # ToDo: create a datastruct for this
        self.past_questions_answers.append({"message": message, "answer": answer})

        return answer


if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv()

    chrome_browser = ChromeBrowser()
    assert chrome_browser.chrome_user_data_dir.startswith("/User")

    # closed_ai = ChatGPT(chrome_browser, session_id="closed_ai")
    # try:
    #     closed_ai.send_message("Can you execute research for me?")
    #     print(closed_ai.past_questions_answers[-1])
    #     closed_ai.send_message("Can you go online for me")
    #     print(closed_ai.past_questions_answers[-1])
    # except Exception as e:
    #     print(f"An error occurred: {e}")
    #
    # deepseek = DeepSeek(chrome_browser, session_id="deep")
    # try:
    #     deepseek.send_message("What are you trained on?")
    #     print(deepseek.past_questions_answers[-1])
    #     deepseek.send_message("What is your latest knowledge cutoff?")
    #     print(deepseek.past_questions_answers[-1])
    # except Exception as e:
    #     print(f"An error occurred: {e}")

    # from browser import ChromeBrowser, chrome_browser
    # gemini = Gemini(chrome_browser, session_id="gemini")
    # try:
    #     gemini.send_message("What are you trained on?")
    #     print(gemini.past_questions_answers[-1])
    #     gemini.send_message("What is your latest knowledge cutoff?")
    #     print(gemini.past_questions_answers[-1])
    # except Exception as e:
    #     print(f"An error occurred: {e}")

    # from browser import ChromeBrowser, chrome_browser
    # gemini = Qwen(chrome_browser, session_id="qwen")
    # try:
    #     gemini.send_message("What are you trained on?")
    #     print(gemini.past_questions_answers[-1])
    #     gemini.send_message("What is your latest knowledge cutoff?")
    #     print(gemini.past_questions_answers[-1])
    # except Exception as e:
    #     print(f"An error occurred: {e}")

    from browser import ChromeBrowser

    claude = Claude(chrome_browser, session_id="qwen")
    try:
        claude.send_message("What are you trained on?")
        print(claude.past_questions_answers[-1])
        claude.send_message("What is your latest knowledge cutoff?")
        print(claude.past_questions_answers[-1])
    except Exception as e:
        print(f"An error occurred: {e}")
    #
    # from browser import ChromeBrowser, chrome_browser
    # mistral = Mistral(chrome_browser, session_id="qwen")
    # try:
    #     mistral.send_message("What are you trained on?")
    #     print(mistral.past_questions_answers[-1])
    #     mistral.send_message("What is your latest knowledge cutoff?")
    #     print(mistral.past_questions_answers[-1])
    # except Exception as e:
    #     print(f"An error occurred: {e}")


    sleep(360)
