"""
Functionality specific to openclaw agent that needs to output a tool_call in a specific format
{
  "tool_calls": [
    {
      "function": {
        "name": "cron",
        "arguments": {
          "action": "add",
          "payload": {
            "kind": "agentTurn",
            "message": "message(action=send, to='telegram', message='привет товарищ')"
          },
          "schedule": {
            "at": "2026-03-14T16:00:00Z",
            "kind": "at"
          },
          "sessionTarget": "isolated"
        }
      }
    }
  ]
},
"""
import json
import re


def add_tool_directive(original_prompt: str) -> str:
    """
    Modify prompt for llm to return an OpenClaw's tool call as a formated json.
    """
    prompt_delta = """
IMPORTANT: If the user's request requires using one of the tools available in this prompt, respond with ONLY a JSON code block in the following format:
```json
{
    "function": {
        "name": "<tool_name>",
        "arguments": 
            {
                <arguments also in json format that given in ur prompt>
            }
}
```
Example (cron tool):
```json
{
  "function": {
    "name": "cron",
    "arguments": {
      "action": "add",
      "payload": {
        "kind": "agentTurn",
        "message": "message(action=send, to='telegram', message='привет товарищ')"
      },
      "schedule": {
        "at": "2026-03-14T16:00:00Z",
        "kind": "at"
      },
      "sessionTarget": "isolated"
    }
  }
}
```
Rule: IN CASE THE USE WANTS YOU TO PERFORM ACTIONS USING TOOL, 
YOU MAY ONLY RESPONSE WITH A SINGLE (1) TOOL CALL JSON FROM PROMPT. 
BASE IT ON THE CONTEXT AND PROGRESS OF THE TASK. 
"""
    new_prompt = original_prompt + prompt_delta

    return new_prompt


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
    json_match = re.search(r"json.*?(\{.*\})", raw_llm_answer, re.DOTALL)
    tool_call = None
    if json_match:
        tool_call = json.loads(json_match.group(1))

    return tool_call