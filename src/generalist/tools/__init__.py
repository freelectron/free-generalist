from enum import Enum
from typing import Dict

from llama_index.core.tools import FunctionTool

from .code import do_table_eda, write_code, execute_code
from .summarisers import task_completed


NOT_FOUND_LITERAL = "N/A"

STRING_TOOL_OUTPUT = "string"
FILE_TOOL_OUTPUT = "file"


# Coding tools
eda_table_tool = FunctionTool.from_defaults(fn=do_table_eda)
write_code_tool = FunctionTool.from_defaults(fn=write_code)
execute_code_tool = FunctionTool.from_defaults(fn=execute_code)


# summary tools
task_completed_tool = FunctionTool.from_defaults(fn=task_completed)

class ToolOutputType(str, Enum):
    STRING = STRING_TOOL_OUTPUT
    FILE = FILE_TOOL_OUTPUT

MAPPING = {
    eda_table_tool.metadata.name: ToolOutputType.STRING,
    write_code_tool.metadata.name: ToolOutputType.STRING,
    execute_code_tool.metadata.name: ToolOutputType.STRING,

    task_completed_tool.metadata.name: ToolOutputType.STRING,
}

def get_tool_type(tool_name: str)->ToolOutputType:
    return MAPPING[tool_name]
