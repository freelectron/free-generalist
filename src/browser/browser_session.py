import os
from time import sleep
from typing import Any, Dict

import undetected as uc
from selenium.webdriver.support.ui import WebDriverWait


class ChromeBrowser:
    waiter_default_timeout = 1
    logging_file = "llm_browser_session_base.log"
    past_questions_answers = list()
    chrome_user_data_dir = os.getenv(
        "CHROME_USER_DATA_DIR", "~/Library/Application Support/Google/Chrome"
    )
    default_profile_directory_name = os.getenv(
        "CHROME_PROFILE_DIRECTORY_NAME", "Profile 1"
    )
    # Mapping from tabs tittle to their window handles
    opened_tabs: Dict[str, Any] = {}

    @staticmethod
    def wait(seconds: int = 1):
        sleep(seconds)

    @staticmethod
    def get_default_options():
        """Return default Chrome options."""
        options = uc.ChromeOptions()
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-blink-features=AutomationControlled")

        return options

    def __init__(self, profile: str):
        # FIXME: do propper logging
        import  logging
        self.logger = logging.getLogger(name=self.__class__.__name__)

        self.options = self.get_default_options()
        self.profile = profile if profile else self.default_profile_directory_name
        self.driver = uc.Chrome(
            options=self.options,
            user_data_dir=os.path.join(self.chrome_user_data_dir, self.profile),
        )
        self.waiter = WebDriverWait(self.driver, self.waiter_default_timeout)
