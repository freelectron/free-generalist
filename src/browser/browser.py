import os
import random
from time import sleep
from typing import Any, Dict, Optional

import undetected as uc
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.actions.action_builder import ActionBuilder
from selenium.webdriver.common.actions.pointer_input import PointerInput

from clog import get_logger


class ChromeBrowser:
    waiter_default_timeout = 30
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

    def _move_mouse_absolute(self, x: int, y: int):
        mouse = PointerInput("mouse", "mouse")
        builder = ActionBuilder(self.driver, mouse=mouse)
        builder.pointer_action.move_to_location(x, y)
        builder.perform()

    def random_mouse_move(self, n_moves: int = 3):
        viewport_w = self.driver.execute_script("return window.innerWidth")
        viewport_h = self.driver.execute_script("return window.innerHeight")
        cur_x = int(random.uniform(viewport_w * 0.2, viewport_w * 0.8))
        cur_y = int(random.uniform(viewport_h * 0.2, viewport_h * 0.8))
        self._move_mouse_absolute(cur_x, cur_y)
        for _ in range(n_moves):
            cur_x = max(0, min(viewport_w - 1, cur_x + int(random.uniform(-50, 50))))
            cur_y = max(0, min(viewport_h - 1, cur_y + int(random.uniform(-20, 20))))
            self._move_mouse_absolute(cur_x, cur_y)
            self.wait(0.2)

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
        self.waiter = WebDriverWait(driver=self.driver, timeout=self.waiter_default_timeout)
        self.actions = ActionChains(self.driver)
        self.wait(1)