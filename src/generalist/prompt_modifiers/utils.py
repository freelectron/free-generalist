import json
import re

def parse_out_tool_call(raw_llm_answer: str) -> dict | None:
    """
    Openclaw agent is supposed to write the tool call definition json itself, you just need to parse it out.
    TODO: add checks for whether it was an actual tool call or just plain-regular-unrelated json

    Example 1:

    Here is a tool and i only need to extrac the json that start within curly brackets
    ```json
    copy
    download
    {
        "function": {
            "name": "<tool_name>",
            "arguments":
                {
                   ...
                }
    }
    ```

    Example:

    Here is a tool and i only need to extrac the json that start within curly brackets
    ```json
    {
        "function": {
            "name": "<tool_name>",
            "arguments":
                {
                   ...
                }
    }
    ```
    """
    tool_call = None

    json_match = re.search(r"json.*?(\{.*\})", raw_llm_answer, re.DOTALL | re.IGNORECASE)
    if json_match:
        tool_call = json.loads(json_match.group(1))

    return tool_call