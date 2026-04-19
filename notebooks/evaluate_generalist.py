#!/usr/bin/env python3
"""
Evaluation script for the Generalist agent on GAIA benchmark tasks.
Extracted and refactored from evaluate_generalist_v2.ipynb
"""

import os
import logging
from typing import TypedDict, List, Optional, Any
from dataclasses import dataclass

from dotenv import load_dotenv
import mlflow
from huggingface_hub import snapshot_download
from datasets import load_dataset
from langgraph.graph import StateGraph, START, END

# Import your project modules - adjust these imports based on your actual package structure
try:
    from generalist.agents.core import AgentPlan, AgentDeepWebSearch, AgentUnstructuredDataProcessor, AgentCodeWriterExecutor
    from generalist.tools.data_model import ShortAnswer, Message
    from generalist.tools.planning import determine_next_step, parse_out_resource_link
    from generalist.agents.workflows.tasks.reflection_evaluation import construct_short_answer, summarise_findings
    from browser import BRAVE_SEARCH_SESSION
except ImportError as e:
    logging.warning(f"Import error: {e}. Make sure your package is installed correctly.")


# ============================================================================
# Configuration
# ============================================================================

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
MAX_STEPS = 6

# GAIA dataset configuration
GAIA_REPO_ID = "gaia-benchmark/GAIA"
GAIA_SPLIT = "validation"
GAIA_LEVEL = "2023_level1"
GAIA_PATH_ENV_VAR = "HUGGING_FACE_GAIA_FOLDER_PATH"


# ============================================================================
# Data Models
# ============================================================================

class ExecutionState(TypedDict):
    """State for the LangGraph execution workflow."""
    ask: str  # Original user question/task
    task: Optional[Message]  # Task with metadata
    step: Optional[int]  # Current step index in the plan
    context: List[Message]  # Previous messages and tool calls
    capability_plan: Optional[AgentPlan]  # Current capability plan
    capability_plan_step: Optional[int]  # Current step in capability plan
    answers: List[ShortAnswer]  # Accumulated answers
    resource: Optional[Message]  # Current focus resource
    tools_called: Optional[str]  # Tools that have been called


# ============================================================================
# State Management Functions
# ============================================================================

def init_state(ask: str, resources: Optional[List[Message]] = None) -> ExecutionState:
    """Initialize a new execution state."""
    return ExecutionState(
        ask=ask,
        task=None,
        step=None,
        context=[],
        capability_plan=None,
        capability_plan_step=None,
        answers=[],
        resource=resources[0] if resources else None,
        tools_called=None,
    )


def set_task(state: ExecutionState) -> ExecutionState:
    """Parse the user's ask and create a task message."""
    # Parse out any links from the user's message
    parsed_out_resource = parse_out_resource_link(state["ask"])

    # Create the new message
    task_msg = Message(
        provided_by="user",
        content=state["ask"],
        metadata={},
    )
    if parsed_out_resource:
        task_msg.link = parsed_out_resource["link"]

    state["task"] = task_msg
    state["step"] = 0
    return state


def plan_next_step(state: ExecutionState) -> ExecutionState:
    """Determine the next capability to execute based on current context."""
    capability_plan_json = determine_next_step(
        task=str(state["ask"]),
        resource=str(state.get("resource", "")),
        context=str(state.get("answers", [])),
    )

    state["capability_plan"] = AgentPlan.from_json(capability_plan_json)
    return state


def execute_capability(state: ExecutionState) -> ExecutionState:
    """Execute the current capability in the plan."""
    capability_plan = state["capability_plan"]
    if not capability_plan:
        logger.error("No capability plan found in state")
        return state

    capability_class = capability_plan.agent
    activity = capability_plan.activity

    # Map capability classes to their run methods
    capability_agent = capability_class(activity)
    output_resource = None

    if capability_class is AgentDeepWebSearch:
        logger.info(f"Executing web search for: {activity}")
        output_resource = capability_agent.run()
    elif capability_class is AgentUnstructuredDataProcessor:
        logger.info(f"Processing unstructured data for: {activity}")
        resources = [state.get("resource")] if state.get("resource") else []
        output_resource = capability_agent.run(resources)
    elif capability_class is AgentCodeWriterExecutor:
        logger.info(f"Executing code writing for: {activity}")
        resources = [state.get("resource")] if state.get("resource") else []
        output_resource = capability_agent.run(resources)
    else:
        logger.warning(f"Unidentified agent capability: {capability_class}")

    # Update state with the result
    if output_resource:
        state["resource"] = output_resource
        state["context"].append(output_resource)
    else:
        logger.error("Capability execution produced no output resource")
        raise ValueError("Capability execution should always produce a resource")

    state["step"] = (state["step"] or 0) + 1
    return state


def manage_context(state: ExecutionState) -> ExecutionState:
    """Summarise findings and update context."""
    capability_plan = state["capability_plan"]
    if not capability_plan:
        return state

    short_answer = summarise_findings(
        capability_plan.activity,
        str(state["context"]),
    )

    state["answers"] = [short_answer]
    return state


def evaluate_task_completion(state: ExecutionState) -> str:
    """Check if the task is complete or if we should continue."""
    capability_plan = state["capability_plan"]
    if not capability_plan:
        return "end"

    # Special case for web search - always continue after search
    if capability_plan.agent is AgentDeepWebSearch:
        short_answer = ShortAnswer(answered=False, answer="", clarification="it was search")
    else:
        short_answer = construct_short_answer(
            str(state["ask"]),
            str(state.get("answers", []))
        )

    # Early stopping conditions
    if short_answer.answered:
        logger.info("Task completed - answer found")
        return "end"

    if (state.get("step") or 0) >= MAX_STEPS:
        logger.info(f"Maximum steps ({MAX_STEPS}) reached, stopping")
        return "end"

    return "continue"


def evaluate_task_completion_quick(state: ExecutionState) -> str:
    """Quick check if there's a valid capability to execute."""
    capability_plan = state.get("capability_plan")
    if not capability_plan:
        return "end"

    # Check if the agent is a valid capability
    valid_capabilities = list(AgentPlan.capability_map.values()) if hasattr(AgentPlan, 'capability_map') else []
    if capability_plan.agent not in valid_capabilities:
        return "end"
    return "continue"


# ============================================================================
# Workflow Construction
# ============================================================================

def build_workflow() -> StateGraph:
    """Build and compile the execution workflow."""
    workflow = StateGraph(state_schema=ExecutionState)

    # Add nodes
    workflow.add_node("set_task", set_task)
    workflow.add_node("plan_next_step", plan_next_step)
    workflow.add_node("execute", execute_capability)
    workflow.add_node("manage_context", manage_context)

    # Add edges
    workflow.add_edge(START, "set_task")
    workflow.add_edge("set_task", "plan_next_step")

    # Conditional edges from planning
    workflow.add_conditional_edges(
        "plan_next_step",
        evaluate_task_completion_quick,
        {
            "continue": "execute",
            "end": END,
        }
    )

    workflow.add_edge("execute", "manage_context")

    # Conditional edges from context management
    workflow.add_conditional_edges(
        "manage_context",
        evaluate_task_completion,
        {
            "continue": "plan_next_step",
            "end": END,
        }
    )

    return workflow.compile()


# ============================================================================
# GAIA Dataset Management
# ============================================================================

def setup_gaia_dataset() -> Any:
    """
    Download and load the GAIA dataset.
    
    Returns:
        The loaded dataset.
    """
    gaia_path = os.environ.get(GAIA_PATH_ENV_VAR)
    if not gaia_path:
        raise ValueError(f"Environment variable {GAIA_PATH_ENV_VAR} not set")

    logger.info(f"Downloading GAIA dataset to {gaia_path}")
    data_dir = snapshot_download(
        local_dir=gaia_path,
        local_files_only=False,
        repo_id=GAIA_REPO_ID,
        repo_type="dataset"
    )

    dataset = load_dataset(data_dir, GAIA_LEVEL, split=GAIA_SPLIT)
    logger.info(f"Loaded GAIA dataset with {len(dataset)} samples")
    return dataset


# ============================================================================
# GAIA Task IDs (for reference)
# ============================================================================

GAIA_TASK_IDS = {
    "sosa_many_studio_albums": "8e867cd7-cff9-4e6c-867a-ff5ddc2550be",
    "running_to_the_moon": "e1fc63a2-da7a-432f-be78-7c4a95598703",
    "dr_who_season_9_eps_11_location": "4b6bb5f7-f634-410e-815d-e673ab7f8632",
    "calc_sales_xlsx": "7bd855d8-463d-4ed5-93ca-5fe35145f733",
    "just_running_python": "f918266a-b3e0-4914-865d-4faa564f1aef",
    "looking_up_paper_authors": "46719c30-f4c3-4cad-be07-d5cb21eee6bb",
    "dinosaur_article": "4fc2f1ae-8625-45b5-ab34-ad4433bc21f8",
    "merriam_webster_word": "5188369a-3bbe-43d8-8b94-11558f909a08",
    "reading_txt_and_solving_puzzle": "389793a7-ca17-4e82-81cb-2b3a2391b4b9",
    "teal_video": "9d191bce-651d-4746-be2d-7ef8ecadb9c2",
    "very_specific_web_search_download": "cabe07ed-9eca-40ea-8ead-410ef5e83f91",
    "listening_to_recipe": "99c9cc74-fdc8-46c6-8f8d-3ce2d3bfeea3",
    "scikit_learn_change_log": "d0633230-7067-47a9-9dbf-ee11e0a2cdd6",
    "polish_raymond": "305ac316-eef6-4446-960a-92d80d542f82",
    "baseball_table_processing": "3f57289b-8c60-48be-bd80-01f8099ca449",
    "audio_processing_homework": "1f975693-876d-457b-a649-393859e79bf3",
    "multistep_browser_search_on_a_website": "a0068077-79f4-461a-adfe-75c1a4148545",
    "kuznetsov_paper_st_petersburg": "bda648d7-d618-4883-88f4-3466eabd860e",
    "country_with_list_medals": "cf106601-ab4f-4af9-b045-5295fe67b37d",
    "country_no_longer_exists_web_search": "5a0c1adf-205e-4841-a666-7c3ef95def9d",
}


# ============================================================================
# Evaluation Runner
# ============================================================================

@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    task_id: str
    question: str
    expected_answer: str
    answers: List[ShortAnswer]
    experiment_url: Optional[str] = None


def run_evaluation(task_ids: List[str], dataset: Any) -> List[EvaluationResult]:
    """
    Run evaluation on specified GAIA tasks.
    
    Args:
        task_ids: List of task IDs to evaluate
        dataset: GAIA dataset
        
    Returns:
        List of evaluation results
    """
    results = []
    dataset_questions = {sample["task_id"]: sample for sample in dataset}

    for task_id in task_ids:
        sample = dataset_questions.get(task_id)
        if not sample:
            logger.warning(f"Task ID {task_id} not found in dataset")
            continue

        # Log sample information
        gaia_keys = ['task_id', 'Question', 'Level', 'Final answer', 'file_name', 'file_path', 'Annotator Metadata']
        for key in gaia_keys:
            if key in sample:
                logger.info(f"{key}: {sample[key]}")

        # Set up MLflow experiment
        mlflow.langchain.autolog()
        experiment_name = f"gaia_{task_id.replace('-', '_')}"
        mlflow.set_experiment(experiment_name)
        
        # Build and set the model
        graph = build_workflow()
        mlflow.models.set_model(graph)

        experiment_url = mlflow.get_experiment_by_name(experiment_name)

        # Prepare the question and resources
        question = sample["Question"]
        resources = []
        
        if sample.get("file_path"):
            link = os.path.join(os.environ.get(GAIA_PATH_ENV_VAR, ""), sample["file_path"])
            resource = Message(
                provided_by="user",
                content=question,
                link=link,
                metadata={}
            )
            resources.append(resource)

        # Run the workflow
        initial_state = init_state(ask=question, resources=resources)
        final_state = graph.invoke(initial_state)
        answers = final_state.get("answers", [])

        # Store results
        result = EvaluationResult(
            task_id=task_id,
            question=question,
            expected_answer=sample["Final answer"],
            answers=answers,
            experiment_url=str(experiment_url) if experiment_url else None
        )
        results.append(result)

    return results


def print_results(results: List[EvaluationResult]) -> None:
    """Pretty print evaluation results."""
    for result in results:
        print("\n" + "="*80)
        print(f"Question: {result.question}")
        print(f"Expected Answer: {result.expected_answer}")
        print("-"*40)
        for i, answer in enumerate(result.answers):
            print(f"Answer {i+1}:")
            print(f"  Was answered: {answer.answered}")
            print(f"  Answer: {answer.answer}")
            print(f"  Clarification: {answer.clarification}")
        print("="*80)


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main entry point for evaluation."""
    # Verify Chrome user data directory is set
    chrome_user_data_dir = os.getenv("CHROME_USER_DATA_DIR")
    if not chrome_user_data_dir:
        logger.warning("CHROME_USER_DATA_DIR environment variable not set")
    else:
        logger.info(f"Chrome user data dir: {chrome_user_data_dir}")
        # Optionally verify BRAVE_SEARCH_SESSION is configured
        if 'BRAVE_SEARCH_SESSION' in globals():
            logger.info(f"Brave search session browser dir: {BRAVE_SEARCH_SESSION.browser.chrome_user_data_dir}")

    # Define tasks to evaluate (uncomment as needed)
    evaluation_tasks = [
        # "f918266a-b3e0-4914-865d-4faa564f1aef",  # just_running_python
        # "8e867cd7-cff9-4e6c-867a-ff5ddc2550be",  # sosa_many_studio_albums
        # "e1fc63a2-da7a-432f-be78-7c4a95598703",  # running_to_the_moon
        # "7bd855d8-463d-4ed5-93ca-5fe35145f733",  # calc_sales_xlsx
        # "305ac316-eef6-4446-960a-92d80d542f82",  # polish_raymond
        # "4fc2f1ae-8625-45b5-ab34-ad4433bc21f8",  # dinosaur_article
        # "5188369a-3bbe-43d8-8b94-11558f909a08",  # merriam_webster_word
        # "46719c30-f4c3-4cad-be07-d5cb21eee6bb",  # looking_up_paper_authors
        # "3f57289b-8c60-48be-bd80-01f8099ca449",  # baseball_table_processing
        # "cf106601-ab4f-4af9-b045-5295fe67b37d",  # country_with_list_medals
        "a0068077-79f4-461a-adfe-75c1a4148545",  # multistep_browser_search_on_a_website
    ]

    try:
        # Load dataset
        dataset = setup_gaia_dataset()
        
        # Run evaluation
        results = run_evaluation(evaluation_tasks, dataset)
        
        # Print results
        print_results(results)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        raise

############################
# DELETED FUNCTIONS:
############################
# def construct_short_answer(task: str, context: str) -> ShortAnswer:
#     """Summarises a context and provides a structured short answer to the task.
#
#     This function sends the task and context to an LLM, which synthesizes
#     the information and return a shorten result.
#
#     Note:
#         This function requires a configured LLM client, represented here as `llm`.
#
#     Args:
#         task: The specific question or instruction to be performed.
#         context: A string containing the text/information to be analyzed.
#
#     Returns:
#         A ShortAnswer dataclass instance containing the answer and clarification.
#     """
#     prompt = f"""
# You are tasked with determining whether the following TASK can be answered using the provided information.
#
# TASK: {task}
#
# INFORMATION PROVIDED:
# {context}
#
# INSTRUCTIONS:
# 1. Review the information provided above carefully
# 2. Determine if the TASK can be fully answered using ONLY the information given
# 3. Do NOT use external knowledge or make assumptions beyond what is explicitly stated
#
# OUTPUT FORMAT (valid JSON with formatting):
# ```json
# {{
#     "answer": "<provide the direct answer as a concise word, number, or short phrase, IF the TASK is completely answered in the information, otherwise use leave blank>",
#     "clarification": "<briefly explain what the information contains and how it relates to the task>"
# }}
# ```
#
# IMPORTANT RULES:
# - Provide "answer" ONLY when the task has a complete answer in the provided information
# - If the answer is partial or incomplete, set "answered" to "false"
# - ALWAYS fill the "clarification" field with the main findings from the information, regardless of whether the task was answered
# - Base your response strictly on the provided information without adding external knowledge
# """
#
#     llm_response = llm.complete(prompt)
#     response_text = llm_response.text
#     response_text = response_text.strip()
#
#     json_match = re.search(r"json.*?(\{.*\})", response_text, re.DOTALL | re.IGNORECASE)
#     code_string = json_match.group(1) if json_match else ""
#     if len(code_string) > 1:
#         response_text = code_string
#
#     logger.info(f"Short answer:\n{response_text}.")
#     data = json.loads(response_text)
#     # FIXME: either make parsing more robust or do manually
#     if isinstance(data["answered"], bool):
#         data["answered"] = str(data["answered"])
#
#     answer = data.get("answer", None)
#     if not answer:
#         answered = False
#     elif answer in ["", "None", "blank"]:
#         answered = False
#     else:
#         answered = True
#
#     return ShortAnswer(
#         answered=answered,
#         answer=answer,
#         clarification=data.get("clarification", None)
#     )
#
# def summarise_findings(task: str, context: str) -> ShortAnswer:
#     """
#     Evaluates whether the task has been accomplished based on provided context.
#
#     The task does not require a final answer. It is considered completed
#     if the main steps or intent appear to be fulfilled based solely on
#     the given resources.
#     """
#
#     prompt = f"""
#     You are presented with a list of information describing work, actions, or outcomes of previous steps:
#     {context}
#
#     Based **ONLY** on the resources above and without any additional assumptions, provide a summary of information.
#
#     Extract (copy-paste) the information that might be needed for the task: {task}.
#     """
#
#     llm_response = llm.complete(prompt)
#     response_text = llm_response.text.strip()
#
#     logger.info(f"From `summarise_findings`:\n{response_text}.")
#
#     return ShortAnswer(
#         answered=False,
#         answer="this is just a summary of previous steps.",
#         clarification=response_text,
#     )


if __name__ == "__main__":
    main()
