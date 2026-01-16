import json

from ..agents.core import AgentCapabilityAudioProcessor, AgentCapabilityCodeWriterExecutor, AgentCapabilityDeepWebSearch, AgentCapabilityUnstructuredDataProcessor, CapabilityPlan
from .data_model import Task
from ..models.core import llm
from ..tools.data_model import ContentResource
from clog import get_logger
from ..utils import current_function


logger = get_logger(__name__)


def create_plan(task: str, resources: list[ContentResource]) -> str:
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
      - The `resources` argument is interpolated directly into the prompt and should be
        concise and serializable (e.g., a list of `ContentResource` objects or their
        string representations).

    Args:
        task (str): Human-readable description of the user's goal or request.
        resources (list[ContentResource]): A list of available resources (files, links,
            or in-memory contents). Elements should be representable in the prompt.

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
---
AVAILABLE RESOURCES:
{resources}
---

Instructions:
1. Clarify the Core Objective: Start by rephrasing the user's goal as a single, clear, and specific objective.
2. **OPTIONAL** Extract Resources: Identify any URLs or URIs mentioned in "User's Goal" and NOT ALREADY MENTIONED 
    in the of list of available resource. If available resources DO NOT already include links or files mentioned in the user's task, 
    provide a short description (content) and the link for the resource.
3. Identify what information and steps are still missing to achieve the goal, i.e., answer the question.  
4. Develop a Chronological Action Plan: Break down the objective into a logical sequence of high-level steps (aim for 1, 2 or 3 steps).

Example Output Format (ALWAYS **JSON**):
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

Another Example Output Format:
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
  **IMPORTANT**: Do not duplicate existing resources:
   If no resource (exact LINK or FILE/FOLDER PATH) is mentioned, do not include resource key in the answer/
"""
    task_response = llm.complete(prompt)

    return task_response.text


def determine_capabilities(task: Task, context: str = "") -> CapabilityPlan:
    """
    Analyzes a task and automatically determines which step from the plan should be executed next
    based on the context, then selects the single most appropriate capability for that step.

    Returns:
        CapabilityPlan: A dataclass containing a single sub-task with the chosen capability.
    """
    planning_prompt = f"""
You are a planning agent. Inspect the Task (including its `plan`) and the provided `context` (what was already done).
Choose the single next logical step to execute and the one best capability to perform it.

Capabilities:
- `{AgentCapabilityDeepWebSearch.name}`: search and download web resources only, not for processing the content or getting any answers on the content.
- `{AgentCapabilityUnstructuredDataProcessor.name}`: analyze or extract information from text already in memory.
- `{AgentCapabilityCodeWriterExecutor.name}`: write or run code and/or manipulate files.
- `{AgentCapabilityAudioProcessor.name}`: download/transcribe audio and store results in memory.

Rules:
1. Pick exactly one step to perform now and exactly one capability.
2. Incorporate relevant details from `context` (discovered names, etc.) into the activity description.
3. If `context` is empty, choose the first step in the plan.
4. Do not modify or autocorrect resource links (preserve exact strings).
5. Output ONLY a single JSON in this exact format (no extra text):

Output format:
- "activity": A clear and concise description of the specific action to be performed using the chosen capability, incorporating relevant details from the context
- "capability": One of the above mentioned capabilities that should be used to accomplish the activity

Example 1: Selecting next logical step based on context
Task: "Task(question='What is the age of the main actor of Inception?', objective='Identify the main actor who played in Inception and their age', plan=['Determine the main character of the movie Inception', 'Look up the age of that actor'])"
Context: "Step 0: [ShortAnswer(answered=True, answer='Leonardo DiCaprio played the main character in Inception')]"
Output:
{{
  "subplan": [
    {{
      "activity": "Search for the current age of Leonardo DiCaprio online",
      "capability": "{AgentCapabilityDeepWebSearch.name}"
    }}
  ]
}}
Reasoning: Step 0 is completed: main actor identified as Leonardo DiCaprio from previous "{AgentCapabilityDeepWebSearch.name}" -> "{AgentCapabilityUnstructuredDataProcessor.name}" steps.
    We proceed to step 1 and incorporate the discovered name into the another search activity
     which will be followed by "{AgentCapabilityUnstructuredDataProcessor.name}" to actually retrieve the age from the downloaded sources.  

Example 2: Using context to refine the next action
Task: "Task(objective='Summarize the article about climate change', plan=['Search for and download the article', 'Extract key information from the article'])"
Context: "Step 0: [ShortAnswer(answered=True, answer='Downloaded article from Nature.com about Arctic climate change, stored contents in the resources)]"
Output:
{{
  "subplan": [
    {{
        "activity": "Analyze and extract key information about Arctic climate change from the article (content is provided in resources)",
        "capability": "{AgentCapabilityUnstructuredDataProcessor.name}"
    }}
  ]
}}
Reasoning: Step 0 is completed, so we move to step 1. The activity incorporates specific details from context (Arctic focus, PDF location).

---

**IMPORTANT**: do not change or autocorrect the resource links, e.g., <>/freelectron/<> should stay as 'freelectron'     

Task: "{task}"
Context: {context}

ONLY RESPOND WITH A SINGLE JSON, in this exact JSON format:
{{
  "subplan": [
    {{
      "activity": "...",
      "capability": "..."
    }}
  ]
}}
"""
    response = llm.complete(planning_prompt)
    response_text = response.text.strip()
    logger.info(f"- {current_function()} -- Raw output: {response_text}")
    escaped_response_text = response_text.translate(str.maketrans({"\\":  r"\\"}))

    return json.loads(escaped_response_text)