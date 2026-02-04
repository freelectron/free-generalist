import os.path
import tempfile

from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.tools import FunctionTool
from langgraph.graph.state import CompiledStateGraph

from generalist.agents.workflows.workflow_base import AgentState, AgentWorkflow
from generalist.tools import ToolOutputType, web_search_tool
from generalist.tools.data_model import Context
from clog import get_logger


MAX_STEPS = 1
logger = get_logger(__name__)


class DeepWebSearchWorkflow(AgentWorkflow):
    """
    Workflow builder for Deep Web Search agent.

    Creates a workflow that can perform web searches and process results.
    """
    tools: list[FunctionTool] = [web_search_tool]
    graph: CompiledStateGraph | None = None

    def __init__(
        self,
        name: str,
        agent_capability: str,
        llm: FunctionCallingLLM,
        context: list[Context],
        task: str,
    ):
        """
        Initialize the workflow builder.

        Args:
            name (str): agent name
            llm (FunctionCallingLLM): the brain
            task (str): task that needs to be performed
            context (list[Context]): summary of what has been achieved in the previous steps
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
        link = ""
        content = state["last_output"].output
        if state["last_output"].type == ToolOutputType.FILE:
            # write the output to a tempfile
            fp = tempfile.NamedTemporaryFile(delete=False, delete_on_close=False, mode="w", encoding="utf-8")
            fp.write(state["last_output"].output); fp.close()
            content = (f"Web search SUCCESSFUL for task: {state['task']}. "
                       f"The downloaded info is stored in {fp.name}. "
                       f"PROCEED TO UNSTRUCTURED TEXT PROCESSING!")
            link = fp.name

        state["context"].append(
            Context(
                provided_by=state["last_output"].name,
                link=link,
                content=content,
                metadata={},
            )
        )

    def evaluate_completion(self, state: AgentState):
        if os.path.exists(state["context"][-1].link):
            return "end"

        if state['step'] >= MAX_STEPS:
            return "end"

        return "continue"
