import tempfile
from dataclasses import dataclass

import mlflow
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

from generalist.models.core import MLFlowLLMWrapper, LLMResponse
from generalist.tools import ToolOutputType, get_tool_type, BaseTool
from generalist.tools.data_model import Message, ShortAnswer
from clog import get_logger
from generalist.agents.workflows.tasks.reflection_evaluation import evaluate_task_completion
from generalist.agents.workflows.tasks.plan_action import plan_next_action
from generalist.agents.workflows.tasks.execute_tool import call_tool
from generalist.agents.workflows.tasks.reflect import reflect_on_progress


MAX_STEPS = 5
logger = get_logger(__name__)


@dataclass
class ExecuteToolOutput:
    name: str
    type: ToolOutputType | None
    output: str


class AgentState(TypedDict):
    """State for agent workflow execution.

    Attributes:
        task: The task to be performed
        context: Current context from previous steps
        step: Count of how many cycles (LLM + tool call) have been performed
        plan: Current plan or reasoning about what to do next
        reflection: Reflection on the last tool output and next steps
    """
    # Description of what us asked from an agent
    task: str
    # What needs to be executed next
    plan: str | None
    # Number of iterations through the graph
    step: int
    # Output in the specific format from tool calling LLM
    tool_call_result: ExecuteToolOutput | None
    # Summary of what has been done in the current iteration
    reflection: str | None
    # All messages that were produced
    context: list[Message]
    # Summary of the progress to see if the task has been achieved
    answers: ShortAnswer | None


class AgentWorkflow:
    """
    Configurable workflow builder for agents.

    Creates a LangGraph workflow that can be customized with different tools
    and decision-making logic for different agent types.
    """
    tools: list[BaseTool] | None
    graph: CompiledStateGraph | None = None

    def __init__(
        self,
        name: str,
        agent_capability: str,
        llm: MLFlowLLMWrapper,
        context: list[Message],
        task: str,
        tools: list[BaseTool] | None = None
    ):
        """
        Initialise the workflow builder.

        Args:
            name : agent name
            agent_capability: short description of what the agent can and supposed to do.
            llm: the brain
            task: task that needs to be performed
            context: summary of what has been achieved in the previous steps
            tools: list of tools that the llm can call
        """
        self.agent_name = name
        self.agent_capability = agent_capability
        self.llm = llm
        self.state = AgentState(step=0, task=task, context=context, answers=None, plan=None, reflection=None)
        self.tools = tools if tools else self.tools

    def plan_action(self, state: AgentState):
        """Planning node: Reason about what to do next before executing tools."""
        state["plan"] = plan_next_action(
            task=state["task"],
            context=str(state["context"]),
            agent_capability=self.agent_capability,
            tools=self.tools,
            previous_reflection=state.get("reflection"),
            llm=self.llm,
        )

        logger.info(f"[{self.agent_name}] Step_{state['step']}. Plan: {state['plan']}")
        return state

    def execute_tool(self, state: AgentState):
        """Execute a tool based on the current plan."""
        response = call_tool(
            task=state["task"],
            context=str(state["context"]),
            plan=state["plan"],
            tools=self.tools,
            llm=self.llm,
        )

        if "Encountered error" in str(response):
            raise ValueError(f"Stopping early {response}")

        if response.tool_call:
            tool_name = response.tool_call.tool_name
            state["tool_call_result"] = ExecuteToolOutput(name=tool_name, type=get_tool_type(tool_name), output=str(response))
        else:
            # TODO: is there a way to handle no-tool-call better?
            logger.warning(f"No tool was called, response: {response}")
            state["tool_call_result"] = ExecuteToolOutput(name="No tool executed", type=None, output=str(response))

        state["step"] += 1

        return state

    def process_tool_output(self, state: AgentState):
        """
        Process the tool output by either:
         - writing the output of the call to a file
         - or putting the output directly into the context
        """
        link = ""
        content = state["tool_call_result"].output
        # Note: this is an attempt to keep the context for an agent small
        if state["tool_call_result"].type == ToolOutputType.FILE:
            fp = tempfile.NamedTemporaryFile(delete=False, delete_on_close=False, mode="w", encoding="utf-8")
            fp.write(state["tool_call_result"].output)
            link = fp.name
            fp.close()
            logger.info(f"Wrote {state["tool_call_result"].name} to a file {link}.Output:\n{state["tool_call_result"].output}")
            content = (f"Tool '{state["tool_call_result"].name}' was executed for task '{state["plan"]}'. "
                       f"The full output was too large for context and has been written to file: {link}. "
                       f"You MAY use this path to read the output in the next step.")

        state["context"].append(
            Message(
                provided_by=state["tool_call_result"].name,
                link=link,
                content=content,
                metadata={},
            )
        )

        return state

    def reflect(self, state: AgentState):
        """Reflection node: Analyze the tool output and determine next steps."""
        state["reflection"] = reflect_on_progress(
            task=state["task"],
            context=str(state["context"]),
            agent_capability=self.agent_capability,
            llm=self.llm,
        )

        logger.info(f"[{self.agent_name}] Step_{state['step']}. Reflection: {state['reflection']}")
        return state

    def evaluate_completion(self, state: AgentState):
        decision = evaluate_task_completion(state["task"], str(state["context"]), self.agent_capability, llm=self.llm)
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

        workflow.add_node("plan_action", self.plan_action)
        workflow.add_node("execute_tool", self.execute_tool)
        workflow.add_node("process_tool_output", self.process_tool_output)
        workflow.add_node("reflect", self.reflect)

        # Define the flow: plan → execute → process → reflect → evaluate
        workflow.add_edge(START, "plan_action")
        workflow.add_edge("plan_action", "execute_tool")
        workflow.add_edge("execute_tool", "process_tool_output")
        workflow.add_edge("process_tool_output", "reflect")
        workflow.add_conditional_edges(
            "reflect",
            self.evaluate_completion,
            {
                "continue": "plan_action",
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