import tempfile

from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.tools import FunctionTool
from langgraph.graph.state import CompiledStateGraph

from generalist.agents.workflows.workflow_base import AgentWorkflow, AgentState
from generalist.tools import ToolOutputType
from generalist.tools.data_model import Message
from clog import get_logger


MAX_STEPS = 1
logger = get_logger(__name__)


class UnstructuredTextProcessorWorkflow(AgentWorkflow):
    """
    Workflow builder for Unstructured Text Processor agent.

    Creates a workflow that can process and analyze unstructured text data.
    """
    tools: list[FunctionTool] = []
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
