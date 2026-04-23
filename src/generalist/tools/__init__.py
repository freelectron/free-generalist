from enum import Enum

from .data_model import BaseTool
from .code import TableEdaTool, WriteCodeTool, ExecuteCodeTool
from .file_handling import ReadFileTool, ListFilesTool, FindFileTool, GrepFilesTool, CreateReplaceFileContentsTool
from .web_search import WebSearchTool


STRING_TOOL_OUTPUT = "string"
FILE_TOOL_OUTPUT = "file"


class ToolOutputType(str, Enum):
    STRING = STRING_TOOL_OUTPUT
    FILE = FILE_TOOL_OUTPUT


MAPPING: dict[str, ToolOutputType] = {
    TableEdaTool.name: ToolOutputType.STRING,
    WriteCodeTool.name: ToolOutputType.FILE,
    ExecuteCodeTool.name: ToolOutputType.STRING,
    ReadFileTool.name: ToolOutputType.STRING,
    ListFilesTool.name: ToolOutputType.STRING,
    FindFileTool.name: ToolOutputType.STRING,
    GrepFilesTool.name: ToolOutputType.STRING,
    CreateReplaceFileContentsTool.name: ToolOutputType.STRING,
    WebSearchTool.name: ToolOutputType.FILE,
}


def get_tool_type(tool_name: str) -> ToolOutputType:
    return MAPPING[tool_name]
