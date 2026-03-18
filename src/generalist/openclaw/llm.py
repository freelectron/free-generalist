"""
Call to LLM from OpenClaw.
"""
################################################################################
## FIXME:Find a better way to user browser
import os
import random
from dotenv import load_dotenv
load_dotenv()
assert os.getenv("CHROME_USER_DATA_DIR")
################################################################################
from generalist.openclaw.tool_calling import add_tool_directive, parse_out_tool_call
from browser import CHATGPT_SESSION, GEMINI_SESSION, DEEPSEEK_SESSION, QWEN_SESSION, CLAUDE_SESSION, MISTRAL_SESSION


GEMINI_TINY_WINDOW = 32000
CHAT_GPT_MAX_CHARACTERS = 62000
MIX_DEEPSEEK_QWEN = 0.1


def get_llm_response(query:str):
    query_modified = add_tool_directive(query)

    if len(query) < GEMINI_TINY_WINDOW:
        answer = GEMINI_SESSION.send_message(query_modified)
    elif len(query) < CHAT_GPT_MAX_CHARACTERS:
        answer = CHATGPT_SESSION.send_message(query_modified)
    else:
        if random.random() > MIX_DEEPSEEK_QWEN:

            selector_random_int = random.random()
            if selector_random_int < 0.33:
                answer = DEEPSEEK_SESSION.send_message(query_modified)
            elif selector_random_int < 0.70:
                answer = CLAUDE_SESSION.send_message(query_modified)
            else:
                answer = MISTRAL_SESSION.send_message(query_modified)
        else:
            answer = QWEN_SESSION.send_message(query_modified)

    tool_call = parse_out_tool_call(answer)

    return answer, tool_call
