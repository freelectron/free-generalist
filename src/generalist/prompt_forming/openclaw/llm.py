"""
Call to LLM from OpenClaw.
"""

################################################################################
## FIXME:Find a better way to user browser
import os
from dotenv import load_dotenv

from generalist.prompt_forming.openclaw.tool_calling import add_tool_directive, parse_out_tool_call

load_dotenv()
assert os.getenv("CHROME_USER_DATA_DIR")
from browser import CHATGPT_SESSION, DEEPSEEK_SESSION
################################################################################


def get_llm_response(query:str):
    query_modified = add_tool_directive(query)

    # answer = CHATGPT_SESSION.send_message(query)
    answer = DEEPSEEK_SESSION.send_message(query_modified)

    tool_call = parse_out_tool_call(answer)

    return answer, tool_call
