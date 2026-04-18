import os.path
import tempfile
from typing import Callable

from langgraph.graph.state import CompiledStateGraph

from generalist.agents.workflows.workflow_base import AgentState, AgentWorkflow
from generalist.tools import ToolOutputType, web_search
from generalist.tools.data_model import Message
from clog import get_logger


MAX_STEPS = 1
logger = get_logger(__name__)


class DeepWebSearchWorkflow(AgentWorkflow):
    """
    Workflow builder for Deep Web Search agent.

    Creates a workflow that can perform web searches and process results.
    """
    tools: list[Callable] = [web_search]
    graph: CompiledStateGraph | None = None

    def process_tool_output(self, state: AgentState):
        link = ""
        content = state["tool_call_result"].output
        if state["tool_call_result"].type == ToolOutputType.FILE:
            # Context management trick: write the output to a tempfile
            fp = tempfile.NamedTemporaryFile(delete=False, delete_on_close=False, mode="w", encoding="utf-8")
            fp.write(state["tool_call_result"].output); fp.close()
            content = (f"Web search SUCCESSFUL for task: {state['task']}."
                       f"The downloaded info is stored in {fp.name}."
                       f"Proceed to text processing!")
            link = fp.name

        state["context"].append(
            Message(
                provided_by=state["tool_call_result"].name,
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
