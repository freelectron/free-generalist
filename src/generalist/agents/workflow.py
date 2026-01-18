# For each agent we define a single workflow with the following structure

#########
# State
#########

# class AgentState(TypedDict):
#     task:  Task
#     context: str



###########################
# Context-Aware Graph
###########################

from langgraph.prebuilt import ToolNode, tools_condition
# llm from .models module

from ..models.core import llm

llm.predict_and_call()

# def determi(state: AgentState):
#     prompt = f"""
#     Given the {state["task"]}, and context {state["context"]}, determine what tool to call next.
#     """
#
#     # Get the LLM response
#     response = llm.complete(prompt)
#     return response.text

# # Create the builder
# workflow = StateGraph(AgentState)
#
# # Add nodes
# workflow.add_node("agent", call_model)
# workflow.add_node("tools", ToolNode(tools))
#
# # Define edges (the ReAct loop)
# workflow.add_edge(START, "agent")
# workflow.add_conditional_edges("agent", tools_condition)
# workflow.add_edge("tools", "agent")
#
# # Compile
# app = workflow.compile()