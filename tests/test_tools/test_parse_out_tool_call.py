"""
uv run pytest tests/test_tools/test_parse_out_tool_call.py
"""
import pytest
from generalist.prompt_forming.openclaw.tool_calling import parse_out_tool_call


EXPECTED = {
    "function": {
        "name": "cron",
        "arguments": {
            "action": "add",
        },
    }
}

CLEAN_BLOCK = """\
Here is the tool call:
```json
{
    "function": {
        "name": "cron",
        "arguments": {
            "action": "add"
        }
    }
}
```
"""

JUNK_PREAMBLE_BLOCK = """\
Here is the tool call:
```json
copy
download
{
    "function": {
        "name": "cron",
        "arguments": {
            "action": "add"
        }
    }
}
```
"""

NO_CODE_BLOCK = "Just a plain text answer with no json block."

EMPTY_CODE_BLOCK = "```json\n```"

NON_TOOL_JSON = """\
```json
{"key": "value"}
```
"""

SAME_LINE = """
Here is the tool call:
```json dff {
    "function": {
        "name": "cron",
        "arguments": {
            "action": "add"
        }
    }
}
```
"""

NO_BACK_TICKS = """
Here is the tool call:
json 
dff {
    "function": {
        "name": "cron",
        "arguments": {
            "action": "add"
        }
    }
}
"""

def test_clean_block():
    result = parse_out_tool_call(CLEAN_BLOCK)
    assert result == EXPECTED


def test_junk_preamble_block():
    result = parse_out_tool_call(JUNK_PREAMBLE_BLOCK)
    assert result == EXPECTED


def test_no_code_block_returns_none():
    result = parse_out_tool_call(NO_CODE_BLOCK)
    assert result is None


def test_empty_code_block_returns_none():
    result = parse_out_tool_call(EMPTY_CODE_BLOCK)
    assert result is None


def test_non_tool_json_parses():
    result = parse_out_tool_call(NON_TOOL_JSON)
    assert result == {"key": "value"}

def test_same_line():
    result = parse_out_tool_call(SAME_LINE)
    assert result == EXPECTED, (result , EXPECTED)

def test_no_back_ticks():
    result = parse_out_tool_call(NO_BACK_TICKS)
    assert result == EXPECTED, (result , EXPECTED)