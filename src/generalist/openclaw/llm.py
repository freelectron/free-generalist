"""
Call to LLM from OpenClaw.
"""
################################################################################
## FIXME:Find a better way to user browser
import os
import random

from dotenv import load_dotenv

from generalist.openclaw.tool_calling import add_tool_directive, parse_out_tool_call

load_dotenv()
assert os.getenv("CHROME_USER_DATA_DIR")
from browser import CHATGPT_SESSION, GEMINI_SESSION, DEEPSEEK_SESSION, QWEN_SESSION
################################################################################

GEMINI_TINY_WINDOW = 32000
CHAT_GPT_MAX_CHARACTERS = 62000
MIX_DEEPSEEK_QWEN = 0.01


def get_llm_response(query:str):
    query_modified = add_tool_directive(query)

    if len(query) < GEMINI_TINY_WINDOW:
        answer = GEMINI_SESSION.send_message(query_modified)
    elif len(query) < CHAT_GPT_MAX_CHARACTERS:
        answer = CHATGPT_SESSION.send_message(query_modified)
    else:
        if random.random() > MIX_DEEPSEEK_QWEN:
            answer = DEEPSEEK_SESSION.send_message(query_modified)
        else:
            answer = QWEN_SESSION.send_message(query_modified)


    tool_call = parse_out_tool_call(answer)

    return answer, tool_call
