import textwrap
import inspect


def pprint(text):
    wrapped_lines = textwrap.wrap(text, width=130)
    for line in wrapped_lines:
        print(line)


def current_function():
    return inspect.currentframe().f_back.f_code.co_name
