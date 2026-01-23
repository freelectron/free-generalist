import tempfile
from dataclasses import dataclass
from typing import Any

import mlflow
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.tools import FunctionTool
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

from generalist.tools import ToolOutputType, get_tool_type, task_completed_tool
from generalist.tools.data_model import ContentResource
from clog import get_logger
from generalist.tools.summarisers import construct_task_completion


MAX_STEPS = 5
logger = get_logger(__name__)


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
    last_output: ExecuteToolOutput | None


class AgentWorkflow:
    """Configurable workflow builder for agents.

    Creates a LangGraph workflow that can be customized with different tools
    and decision-making logic for different agent types.
    """

    def __init__(
        self,
        name: str,
        system_prompt: str,
        llm: FunctionCallingLLM,
        tools: list[Any],
        context: list[ContentResource],
        task: str,
    ):
        """
        Initialize the workflow builder.

        Args:
            name (str): agent name
            system_prompt (str): describe what this agent is to do and responsible for
            llm (FunctionCallingLLM): the brain
            tools (list): tools that the agent can use
            task (str): task that needs to be performed
            context (list[ContentResource]): summary of what has been achieved in the previous steps
        """
        self.agent_name = name
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
        
        Context from the previous steps: {state["context"]}
        """

        response = self.llm.predict_and_call(user_msg=prompt, tools=self.tools)

        # tool that has just been called
        tool_name = response.sources[0].tool_name

        state["last_output"] = ExecuteToolOutput(name=tool_name, type=get_tool_type(tool_name), output=response.response)
        state["step"] += 1

        return state

    def process_tool_output(self, state: AgentState):
        """
        """
        link = ""
        content = state["last_output"].output
        # Note: this is an attempt to keep the context for an agent small
        if state["last_output"].type == ToolOutputType.FILE:
            # write the output to a tempfile
            fp = tempfile.NamedTemporaryFile(delete_on_close=False, mode="w", encoding="utf-8")
            fp.write(state["last_output"].output); fp.close()
            logger.info(f"Wrote {state["last_output"].name} to a file {fp.name}.Output:\n{state["last_output"].output}")
            content = f"Output of {state["last_output"].name} tool is stored in {fp.name}. You might want to either read it or execute (python) it."
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

    def evaluate_completion(self, state: AgentState):
        """
        """
        decision = construct_task_completion(state["task"], str(state["context"]))
        # Early stopping if answer exists
        if decision.completed:
            return "end"

        # Early stopping if maximum number of steps reached
        if state['step'] >= MAX_STEPS:
            return "end"

        return "continue"

    def build_compile(self):
        """Builds and compiles the workflow graph."""
        workflow = StateGraph(state_schema=AgentState)

        # you are given a task, determine what tool to call and call it
        workflow.add_node("execute_tool", self.execute_tool)
        workflow.add_node("process_tool_output", self.process_tool_output)

        workflow.add_edge(START, "execute_tool")
        workflow.add_edge( "execute_tool", "process_tool_output")
        workflow.add_conditional_edges(
            "process_tool_output",
            self.evaluate_completion,
            {
                "continue": "execute_tool",
                "end": END,
            }
        )

        self.graph = workflow.compile()

    def run(self) -> AgentState:
        """Convenience method to compile and run the workflow.

        Returns:
            Final state after workflow execution
        """
        if self.graph is None:
            self.build_compile()

        mlflow.models.set_model(self.graph)
        final_state = self.graph.invoke(self.state)

        return final_state