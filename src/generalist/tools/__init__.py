from enum import Enum

from llama_index.core.tools import FunctionTool

from .code import do_table_eda, write_code, execute_code
from .web_search import web_search


STRING_TOOL_OUTPUT = "string"
FILE_TOOL_OUTPUT = "file"


# Coding tools
eda_table_tool = FunctionTool.from_defaults(fn=do_table_eda)
write_code_tool = FunctionTool.from_defaults(fn=write_code)
execute_code_tool = FunctionTool.from_defaults(fn=execute_code)
# Web search tools
web_search_tool = FunctionTool.from_defaults(fn=web_search)


class ToolOutputType(str, Enum):
    STRING = STRING_TOOL_OUTPUT
    FILE = FILE_TOOL_OUTPUT

MAPPING = {
    eda_table_tool.metadata.name: ToolOutputType.STRING,
    write_code_tool.metadata.name: ToolOutputType.FILE,
    execute_code_tool.metadata.name: ToolOutputType.STRING,
    web_search_tool.metadata.name: ToolOutputType.FILE,
}

def get_tool_type(tool_name: str)->ToolOutputType:
    return MAPPING[tool_name]
