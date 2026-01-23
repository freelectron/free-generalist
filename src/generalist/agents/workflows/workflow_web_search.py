import tempfile

from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.tools import FunctionTool
from langgraph.graph.state import CompiledStateGraph

from generalist.agents.workflows.workflow_base import AgentState, AgentWorkflow
from generalist.tools import ToolOutputType, web_search_tool
from generalist.tools.data_model import ContentResource
from clog import get_logger


MAX_STEPS = 1
logger = get_logger(__name__)


class DeepWebSearchWorkflow(AgentWorkflow):
    """
    Workflow builder for Deep Web Search agent.

    Creates a workflow that can perform web searches and process results.
    """
    agent_prompt: str = "You are a web researcher"
    tools: list[FunctionTool] = [web_search_tool]
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
            content = f"Web search was successful. The results are stored in {fp.name}. PROCEED TO TEXT PROCESSING!"
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
