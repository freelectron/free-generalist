import tempfile

from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.tools import FunctionTool
from langgraph.graph.state import CompiledStateGraph

from generalist.agents.workflows.workflow_base import AgentWorkflow, AgentState
from generalist.tools import ToolOutputType, eda_table_tool, write_code_tool, execute_code_tool
from generalist.tools.data_model import ContentResource
from clog import get_logger


MAX_STEPS = 4
logger = get_logger(__name__)


class CodeWriterExecutorWorkflow(AgentWorkflow):
    """Workflow builder for Code Writer Executor agent.

    Creates a workflow that can write and execute Python code.
    """
    agent_prompt: str = "You are code writer and executor"
    tools: list[FunctionTool] = [eda_table_tool, write_code_tool, execute_code_tool]
    graph: CompiledStateGraph | None = None

    def __init__(
        self,
        name: str,
        llm: FunctionCallingLLM,
        context: list[ContentResource],
        task: str,
    ):
        """
        Initialize the workflow builder.

        Args:
            name (str): agent name
            llm (FunctionCallingLLM): the brain
            task (str): task that needs to be performed
            context (list[ContentResource]): summary of what has been achieved in the previous steps
        """
        super().__init__(
            name=name,
            llm=llm,
            context=context,
            task=task,
        )

    def process_tool_output(self, state: AgentState):
        """
        """
        link = ""
        content = state["last_output"].output
        if state["last_output"].type == ToolOutputType.FILE:
            # write the output to a tempfile
            fp = tempfile.NamedTemporaryFile(delete_on_close=False, mode="w", encoding="utf-8")
            fp.write(state["last_output"].output); fp.close()
            content = (f"Output of {state["last_output"].name} run is stored in {fp.name}."
                       f"EXECUTE THIS FILE NEXT TO GET THE RESULT.")
            link = fp.name

        state["context"].append(
            ContentResource(
                provided_by=state["last_output"].name,
                link=link,
                content=content,
                metadata={},
            )
        )

        return state
