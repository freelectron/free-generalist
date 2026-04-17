import tempfile
from typing import Callable

from langgraph.graph.state import CompiledStateGraph

from generalist.agents.workflows.workflow_base import AgentWorkflow, AgentState
from generalist.models.core import MLFlowLLMWrapper
from generalist.tools import ToolOutputType, execute_code_tool, do_table_eda, \
    write_code, execute_code
from generalist.tools.data_model import Message
from clog import get_logger


MAX_STEPS = 4
logger = get_logger(__name__)


class CodeWriterExecutorWorkflow(AgentWorkflow):
    """Workflow builder for Code Writer Executor agent.

    Creates a workflow that can write and execute Python code.
    """
    tools: list[Callable] = [do_table_eda,
                             write_code,
                             execute_code]
    graph: CompiledStateGraph | None = None

    def __init__(
        self,
        name: str,
        agent_capability: str,
        llm: MLFlowLLMWrapper,
        context: list[Message],
        task: str,
    ):
        """
        Initialize the workflow builder.

        Args:
            name: agent name
            llm: the brain
            task: task that needs to be performed
            context: summary of what has been achieved in the previous steps
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
        link = state["tool_call_result"].output
        content = state["tool_call_result"].output
        if state["tool_call_result"].type == ToolOutputType.FILE:
            # write the output to a tempfile
            fp = tempfile.NamedTemporaryFile(delete=False, delete_on_close=False, mode="w", encoding="utf-8")
            fp.write(state["tool_call_result"].output); fp.close()
            content = (f"Output of {state["tool_call_result"].name} for an intermediary task (task: {state['task']}) is stored in {fp.name}."
                       f"EXECUTE THIS FILE IN THE NEXT STEP TO GET THE RESULT.")
            link = fp.name

        # TODO: find how to adjust the code so that I can process the output of each tool code differently
        if state["tool_call_result"].name == execute_code_tool.metadata.name:
           content = f"Executed code for task: {state['task']}.\nOUTPUT:" + content

        state["context"].append(
            Message(
                provided_by=state["tool_call_result"].name,
                link=link,
                content=content,
                metadata={},
            )
        )

        return state
