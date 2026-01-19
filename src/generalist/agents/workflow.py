from typing import Callable, Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition


class AgentState(TypedDict):
    """State for agent workflow execution.

    Attributes:
        task: The task to be performed
        context: Current context from previous steps
        step: Count of how many cycles (LLM + tool call) have been performed
    """
    task: str
    context: str
    step: int


class AgentWorkflow:
    """Configurable workflow builder for agents.

    Creates a LangGraph workflow that can be customized with different tools
    and decision-making logic for different agent types.
    """

    def __init__(
        self,
        tools: list[Any],
        determine_action: Callable[[AgentState], AgentState],
        context: str,
        task: str,
    ):
        """Initialize the workflow builder.

        Args:
            tools: List of tools that the agent can use
            determine_action: Function that takes AgentState and returns
                             a dict with the LLM response and any state updates
            task (str): task that needs to be performed
            context (str): summary of what has been achieved in the previous steps
        """
        self.state = AgentState(step=0, task=task, context=context)
        self.tools: list[Callable[[Any], Any]] = tools
        self.determine_action_func: Callable[[AgentState], AgentState] = determine_action
        self.graph: CompiledStateGraph | None = None

    def build_compile(self):
        """Builds and compiles the workflow graph."""
        workflow = StateGraph(state_schema=AgentState)

        workflow.add_node("determine_action", self.determine_action_func)
        workflow.add_node("tools", ToolNode(self.tools))

        workflow.add_edge(START, "determine_action")
        workflow.add_conditional_edges("determine_action", tools_condition)
        workflow.add_edge("tools", "determine_action")
        
        self.graph = workflow.compile()

    def run(self):
        """Convenience method to compile and run the workflow.

        Returns:
            Final state after workflow execution
        """
        if self.graph is None:
            self.build_compile()

        return self.graph.invoke(self.state)