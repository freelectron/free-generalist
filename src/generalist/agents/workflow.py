from dataclasses import dataclass
from typing import Callable, Any, Literal, Union

from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.tools import FunctionTool
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START
from langgraph.graph.state import CompiledStateGraph

from generalist.tools import OUTPUT_TYPE, ToolOutputType, get_tool_type
from generalist.tools.data_model import ContentResource


@dataclass
class ExecuteToolOutput:
    name: str
    type: ToolOutputType
    output: str


class AgentState(TypedDict):
    """State for agent workflow execution.

    Attributes:
        task: The task to be performed
        context: Current context from previous steps
        step: Count of how many cycles (LLM + tool call) have been performed
    """
    task: str
    context: list[ContentResource]
    step: int
    last_output: ExecuteToolOutput


class AgentWorkflow:
    """Configurable workflow builder for agents.

    Creates a LangGraph workflow that can be customized with different tools
    and decision-making logic for different agent types.
    """

    def __init__(
        self,
        system_prompt: str,
        llm: FunctionCallingLLM,
        tools: list[Any],
        context: list[ContentResource],
        task: str,
    ):
        """Initialize the workflow builder.

        Args:
            system_prompt (str): describe what this agent is to do and responsible for
            llm (FunctionCallingLLM): the brain
            tools (list): tools that the agent can use
            task (str): task that needs to be performed
            context (str): summary of what has been achieved in the previous steps
        """
        self.llm = llm
        self.agent_prompt = system_prompt
        self.state = AgentState(step=0, task=task, context=context)
        self.tools: list[FunctionTool] = tools
        self.graph: CompiledStateGraph | None = None

    def execute_tool(self, state: AgentState):
        """
        """
        prompt = f"""
        Role: {self.agent_prompt}
        
        Task: {state["task"]}
        
        Context: {state["context"]}
        """

        response = self.llm.predict_and_call(user_msg=prompt, tools=self.tools)

        # what has been called
        tool_name = response.sources[0].tool_name

        state["last_output"] = ExecuteToolOutput(name=tool_name, type=get_tool_type(tool_name), output=response.response)

    def process_tool_call_output(self, state: AgentState):
        """
        """

    def build_compile(self):
        """Builds and compiles the workflow graph."""
        workflow = StateGraph(state_schema=AgentState)

        # you are given a task, determine what tool to call and call it
        workflow.add_node("execute_tool", self.execute_tool)

        self.graph = workflow.compile()

    def run(self):
        """Convenience method to compile and run the workflow.

        Returns:
            Final state after workflow execution
        """
        if self.graph is None:
            self.build_compile()

        return self.graph.invoke(self.state)