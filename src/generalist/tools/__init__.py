from llama_index.core.tools import FunctionTool

from generalist.tools.code import do_table_eda, write_code, execute_code

NOT_FOUND_LITERAL = "N/A"

# Coding
eda_table_tool = FunctionTool.from_defaults(fn=do_table_eda)
write_code_tool = FunctionTool.from_defaults(fn=write_code)
execute_code_tool = FunctionTool.from_defaults(fn=execute_code)
