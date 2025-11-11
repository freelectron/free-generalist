import logging
import os
from datetime import datetime
from typing import Optional

LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


def create_file_handler(file_path: str):
    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(LOG_FORMAT)
    file_handler.setFormatter(file_formatter)

    return file_handler

def create_console_handler():
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_formatter = logging.Formatter(LOG_FORMAT)
    console_handler.setFormatter(console_formatter)

    return console_handler


class CLogger(logging.Logger):
    def switch_to_file(self):
        self.removeHandler(self.console_handler)
        self.addHandler(self.file_handler)

    def switch_to_console(self):
        self.removeHandler(self.file_handler)
        self.addHandler(self.console_handler)

    def __init__(self, name:str, level:int, file_path:str):
        super().__init__(name, level)

        if not file_path:
            debug_folder_path = os.environ.get("DEBUG_FOLDER_LOCATION", None)
            if not debug_folder_path:
                debug_folder_path = "./logs"
                if  not os.path.exists(debug_folder_path):
                    os.makedirs("logs")
            file_path =  os.path.join(debug_folder_path, f'{datetime.today().date().isoformat()}.log')

        self.console_handler = create_console_handler()
        self.file_handler = create_file_handler(file_path)

        # Default is the console handler
        self.addHandler(self.console_handler)

    def fdebug(self, msg):
        self.switch_to_file()
        self.debug(msg)
        self.switch_to_console()

    def finfo(self, msg):
        self.switch_to_file()
        self.info(msg)
        self.switch_to_console()

    def fwarn(self, msg):
        self.switch_to_file()
        self.warning(msg)
        self.switch_to_console()

    def ferror(self, msg):
        self.switch_to_file()
        self.error(msg)
        self.switch_to_console()

    def fcritical(self, msg):
        self.switch_to_file()
        self.critical(msg)
        self.switch_to_console()

def get_logger(name: str, file_path: Optional[str] = None):
    level = logging.INFO

    return CLogger(name=name, level=level, file_path=file_path)