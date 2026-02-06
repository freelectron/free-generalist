import os
from time import sleep
from typing import Any, Dict, Optional

import undetected as uc
from selenium.webdriver.support.ui import WebDriverWait

from clog import get_logger


class ChromeBrowser:
    waiter_default_timeout = 10
    # Mapping from tabs tittle to their window handles
    opened_tabs: Dict[str, Any] = {}

    @staticmethod
    def wait(seconds: float = 1):
        sleep(seconds)

    @staticmethod
    def get_default_options():
        """Return default Chrome options."""
        options = uc.ChromeOptions()
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-blink-features=AutomationControlled")

        # options.add_argument("--user-agent=Chrome/122.0.0.0")
        options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36")

        return options

    def __init__(self, profile: Optional[str] = None):
        self.logger = get_logger(name=self.__class__.__name__)

        self.chrome_user_data_dir = os.getenv("CHROME_USER_DATA_DIR", "./browser_cache")
        default_profile_directory_name = os.getenv("CHROME_PROFILE", "Default")

        self.options = self.get_default_options()
        self.profile = profile if profile else default_profile_directory_name
        self.driver = uc.Chrome(
            options=self.options,
            user_data_dir=os.path.join(self.chrome_user_data_dir, self.profile),
        )
        self.waiter = WebDriverWait(self.driver, self.waiter_default_timeout)
        self.wait(1)