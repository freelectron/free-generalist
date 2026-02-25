import tempfile
from dataclasses import dataclass

import mlflow
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.tools import FunctionTool
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

from generalist.tools import ToolOutputType, get_tool_type
from generalist.tools.data_model import Message, ShortAnswer
from clog import get_logger
from generalist.tools.summarisers import construct_task_completion, summarise_findings

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
        plan: Current plan or reasoning about what to do next
        reflection: Reflection on the last tool output and next steps
    """
    task: str
    context: list[Message]
    answers: ShortAnswer | None
    tool_call_result: ExecuteToolOutput | None
    step: int
    plan: str | None
    reflection: str | None


class AgentWorkflow:
    """
    Configurable workflow builder for agents.

    Creates a LangGraph workflow that can be customized with different tools
    and decision-making logic for different agent types.
    """
    tools: list[FunctionTool]
    graph: CompiledStateGraph

    def __init__(
        self,
        name: str,
        agent_capability: str,
        llm: FunctionCallingLLM,
        context: list[Message],
        task: str,
        tools: list[FunctionTool] | None = None
    ):
        """
        Initialise the workflow builder.

        Args:
            name (str): agent name
            agent_capability (str): short description of what the agent can and supposed to do.
            llm (FunctionCallingLLM): the brain
            task (str): task that needs to be performed
            context (list[Message]): summary of what has been achieved in the previous steps
            tools: list of tools that the llm can call
        """
        self.agent_name = name
        self.agent_capability = agent_capability
        self.llm = llm
        self.state = AgentState(step=0, task=task, context=context, answers=None, plan=None, reflection=None)
        self.tools = tools if tools else self.tools

    def plan_action(self, state: AgentState):
        """Planning node: Reason about what to do next before executing tools."""

        # Format context
        context_str = state["context"]

        # Format available tools
        tools_str = "\n".join([f"- {tool.metadata.name}: {tool.metadata.description}" for tool in self.tools])

        prompt = f"""
        Role: {self.agent_capability}

        Task: {state["task"]}

        Context from previous steps:
        {context_str}

        {f"Previous reflection: {state['reflection']}" if state.get("reflection") else ""}

        Available tools:
        {tools_str}

        Based on the task and context, reason about:
        1. What information do you have so far?
        2. What is still missing to complete the task?
        3. Which tool should you use next and why?

        Provide a brief plan for the next action (2-3 sentences).
        """

        response = self.llm.complete(prompt)
        state["plan"] = response.text.strip()
        logger.info(f"Plan: {state['plan']}")

        return state

    def execute_tool(self, state: AgentState):
        """Execute a tool based on the current plan."""

        # Format context
        context_str = state["context"]

        prompt = f"""
        Role: {self.agent_capability}

        Task: {state["task"]}

        Context from previous steps:
        {context_str}

        Plan: {state["plan"]}

        Based on the plan above, you MUST call exactly ONE of the available tools now.
        Choose the most appropriate tool to make progress on the task.
        """

        # Sometimes llm does not call any tools so we need to retry
        try:
            response = self.llm.predict_and_call(user_msg=prompt, tools=self.tools)
        except ValueError as e:
            if "Expected at least one tool call" in str(e):
                prompt = prompt + "\n\nIMPORTANT: You MUST call one of the available tools. Review the tools and select the most appropriate one."
                response = self.llm.predict_and_call(user_msg=prompt, tools=self.tools)
            else:
                raise  # re-raise if it's a different ValueError

        # tool that has just been called
        tool_name = response.sources[0].tool_name

        state["tool_call_result"] = ExecuteToolOutput(name=tool_name, type=get_tool_type(tool_name), output=response.response)
        state["step"] += 1

        return state

    def process_tool_output(self, state: AgentState):
        """Process the tool output and add it to context."""
        link = ""
        content = state["tool_call_result"].output
        # Note: this is an attempt to keep the context for an agent small
        if state["tool_call_result"].type == ToolOutputType.FILE:
            # write the output to a tempfile
            fp = tempfile.NamedTemporaryFile(delete=False, delete_on_close=False, mode="w", encoding="utf-8")
            fp.write(state["tool_call_result"].output); fp.close()
            logger.info(f"Wrote {state["tool_call_result"].name} to a file {fp.name}.Output:\n{state["tool_call_result"].output}")
            content = f"Output of {state["tool_call_result"].name} tool is stored in {fp.name}."
            link = fp.name

        # Add to context
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

        context_str = state["context"]

        prompt = f"""
        Role: {self.agent_capability}

        Task: {state["task"]}

        Context so far:
        {context_str}

        Reflect on the progress:
        1. What did you just learn from the latest tool output?
        2. How does this help with the task?
        3. Is the task complete, or what should you do next?

        Provide a brief reflection (2-3 sentences).
        """

        response = self.llm.complete(prompt)
        state["reflection"] = response.text.strip()
        logger.info(f"Reflection: {state['reflection']}")

        # Update answers summary based on reflection
        state["answers"] = summarise_findings(task=state["task"], context=context_str)

        return state

    def evaluate_completion(self, state: AgentState):
        decision = construct_task_completion(state["task"], str(state["context"]), self.agent_capability)
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

        # Add nodes for the workflow
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