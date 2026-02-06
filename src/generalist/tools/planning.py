import json

from ..agents.core import AgentDeepWebSearch, AgentUnstructuredDataProcessor, AgentPlan, AgentCodeWriterExecutor
from .data_model import Task
from ..models.core import llm
from clog import get_logger


logger = get_logger(__name__)


def create_plan(task: str) -> str:
    """
    Create a concise, JSON-formatted action plan for the given task using the LLM.

    This function constructs a detailed prompt describing the user's goal and the available
    resources, instructing the language model to return a plain JSON string with the
    following structure:
      - "objective": A single-sentence, clear summary of the end goal.
      - Optional "resource": A dictionary with "content" (short description or file contents)
        and "link" (exact URL or file/URI) only when a URL/URI is present in the task but
        not already listed in `resources`.
      - "plan": A list of the minimum number (1--3) of high-level steps required to achieve
        the objective. Each list item must be separated by the Python newline character in
        the model output.

    Important behavior:
      - The function sends the constructed prompt to `llm.complete` and returns the raw
        `text` attribute of the LLM response. It does not parse or validate the JSON.
      - The caller is responsible for parsing, validating, and handling any deviations or
        errors in the model output.

    Args:
        task (str): Human-readable description of the user's goal or request.

    Returns:
        str: The raw text produced by the LLM (expected to be a plain JSON string). This
        return value may include model formatting errors; callers must handle parsing
        and validation.

    Notes:
        - The prompt contains explicit, strict instructions to the model to output plain
          JSON only, but real model outputs can still be non-conforming. Validate before use.
        - Avoid passing sensitive or very large data via `task` or `resources` to prevent
          leakage or prompt truncation.
    """

    prompt = f"""
You are an expert project planner. Your task is to create a concise, step-by-step action plan to accomplish the given user's goal and available resources.

USER'S GOAL:
{task}

Instructions:
1. Clarify the Core Objective: Start by rephrasing the user's goal as a single, clear, and specific objective.
2. **OPTIONAL** Extract Resources: Identify any URLs or URIs mentioned in USER'S GOAL AND NOT ALREADY MENTIONED 
    in AVAILABLE RESOURCES. If available resources DO NOT already include links or files mentioned in the user's task, 
    provide a short description (content) and the link for the resource.
3. Identify what information and steps are still missing to achieve the goal, i.e., answer the question.  
4. Develop a Chronological Action Plan: Break down the objective into a logical sequence of high-level steps.

Example (ALWAYS **JSON**):
{{
  "objective": "Produce a plot with the sales data of the provided csv (/home/user_name/datat/company_balance_sheet.csv)",
  "resource":
    {{
      "content": "CSV file containing company balance sheet data",
      "link": "/home/user_name/datat/company_balance_sheet.csv"
    }},
  "plan": [
    "Analyse the information on what columns are present in the csv and which ones represent sales",
    "Produce a piece of code that would only plot sales data"
  ]
}}

Another Example:
{{
  "objective": "Examine the video and extract the dialogue by Teal'c in response to the question 'Isn't that hot?'",
  "plan": [
    "Download the video from the provided URL",
    "Extract the audio and transcribe it to text",
    "Locate the part where the question is asked and Teal'c responds"
  ]
}}
Note, that there is no "resource" field generated, because the video URL/URI was not given.

where
  "objective" 's value in the json is a clear, one-sentence summary of the end goal,
  "resource" 's value in the json is a dictionary, containing:
    - "content": a brief description of what the resource is OR the contents of the file 
    - "link": the exact URL or URI mentioned in the task (AND NOT MENTIONED IN THE 'Available resources' SECTION ALREADY)
  "plan" 's value in the json is a list **ALWAYS SEPARATED BY PYTHON NEWLINE CHARACTER** like 
  [
    "A short explanation of the first logical step", 
    "The concluding step",
  ]
  **IMPORTANT**: you should only include the minimum number of steps to accomplish the task, do not include verification steps.
  **IMPORTANT**: do not include any json formatting directives, output plain json string.
  **IMPORTANT**: only include the resource that are URLs or URIs. 
"""
    task_response = llm.complete(prompt)

    return task_response.text


def determine_capabilities(task: str, resource: str = "", context: str = "") -> dict:
    """
    Analyzes a task and automatically determines which step from the plan should be executed next
    based on the context, then selects the single most appropriate capability for that step.

    Returns:
        A dataclass containing a single sub-task with the chosen capability.
    """
    planning_prompt = f"""
Your task: Based on what's already been done (Previous Steps and Context), determine the next logical step from the Plan, then select the appropriate agent to execute it.

Plan: {task}
Previous output: {resource}
Context: {context}

Available agents:
- `{AgentDeepWebSearch.name}`: {AgentDeepWebSearch.capability}
- `{AgentUnstructuredDataProcessor.name}`: {AgentUnstructuredDataProcessor.capability}
- `{AgentCodeWriterExecutor.name}`: {AgentCodeWriterExecutor.capability}

Instructions:
1. Review the Context to understand what has already been completed
2. Identify the next step from the Plan that should be executed
3. Choose ONE agent that best matches the requirements of that step
4. Describe the activity with specific details from resources (file names, discovered information, etc.)

Example - When context has completed steps:
Plan: ["Download video from URL", "Extract audio and transcribe", "Find Teal'c's response"]
Context: "Downloaded video saved as /tmp/stargate_clip.mp4"
Output: {{"activity": "Extract audio from /tmp/stargate_clip.mp4 and transcribe it to text", "agent": "{AgentCodeWriterExecutor.name}"}}

Example - When file analysis is needed:
Plan: ["Analyze CSV columns", "Plot sales data"]
Context: "CSV file available at /home/user/sales.csv"
Output: {{"activity": "Analyze /home/user/sales.csv to identify which columns represent sales data", "agent": "{AgentUnstructuredDataProcessor.name}"}}

Output JSON format:
{{
  "activity": "Clear description of the single step to perform, including specific details from Context",
  "agent": "The agent name that should execute this activity"
}}

Rules:
- Never combine multiple plan steps into one activity
- Incorporate relevant details from Context and Previous steps (file paths, URLs, names, etc.) into the activity description
"""
    response = llm.complete(planning_prompt)
    response_text = response.text.strip()
    logger.info(f"Raw output: {response_text}")
    escaped_response_text = response_text.translate(str.maketrans({"\\":  r"\\"}))

    return json.loads(escaped_response_text)