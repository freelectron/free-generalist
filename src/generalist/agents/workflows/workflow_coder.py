import tempfile

from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.tools import FunctionTool
from langgraph.graph.state import CompiledStateGraph

from generalist.agents.workflows.workflow_base import AgentWorkflow, AgentState
from generalist.tools import ToolOutputType, eda_table_tool, write_code_tool, execute_code_tool
from generalist.tools.data_model import Message
from clog import get_logger


MAX_STEPS = 4
logger = get_logger(__name__)


class CodeWriterExecutorWorkflow(AgentWorkflow):
    """Workflow builder for Code Writer Executor agent.

    Creates a workflow that can write and execute Python code.
    """
    tools: list[FunctionTool] = [eda_table_tool, write_code_tool, execute_code_tool]
    graph: CompiledStateGraph | None = None

    def __init__(
        self,
        name: str,
        agent_capability: str,
        llm: FunctionCallingLLM,
        context: list[Message],
        task: str,
    ):
        """
        Initialize the workflow builder.

        Args:
            name (str): agent name
            llm (FunctionCallingLLM): the brain
            task (str): task that needs to be performed
            context (list[Message]): summary of what has been achieved in the previous steps
        """
        super().__init__(
            name=name,
            agent_capability=agent_capability,
            llm=llm,
            context=context,
            task=task,
        )

    def process_tool_output(self, state: AgentState):
        """
        """
        link = state["last_output"].output
        content = state["last_output"].output
        if state["last_output"].type == ToolOutputType.FILE:
            # write the output to a tempfile
            fp = tempfile.NamedTemporaryFile(delete=False, delete_on_close=False, mode="w", encoding="utf-8")
            fp.write(state["last_output"].output); fp.close()
            content = (f"Output of {state["last_output"].name} for an intermediary task (task: {state['task']}) is stored in {fp.name}."
                       f"EXECUTE THIS FILE IN THE NEXT STEP TO GET THE RESULT.")
            link = fp.name

        # TODO: find how to adjust the code so that I can process the output of each tool code differently
        if state["last_output"].name == execute_code_tool.metadata.name:
           content = f"Executed code for task: {state['task']}.\nOUTPUT:" + content

        state["context"].append(
            Message(
                provided_by=state["last_output"].name,
                link=link,
                content=content,
                metadata={},
            )
        )

        return state
