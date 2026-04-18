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
    do_table_eda.__name__: ToolOutputType.STRING,
    write_code.__name__: ToolOutputType.FILE,
    execute_code.__name__: ToolOutputType.STRING,
    web_search.__name__: ToolOutputType.FILE,
}

def get_tool_type(tool_name: str)->ToolOutputType:
    return MAPPING[tool_name]
