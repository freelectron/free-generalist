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
    Given a task, determine the next step that needs to be done to accomplish the task.
    The most important actions that are taken:
     1. Define the goal: what result is asked to be produced.
     2. List the steps: provide a short explanation for each action that needs to be taken.
    """

    prompt = f"""
You are an expert project planner. Your task is to create a concise, step-by-step action plan to accomplish the given user's goal and available resources.

User's Goal:
{task}
---
Available resources:
{resources}
---

Instructions:
1. Clarify the Core Objective: Start by rephrasing the user's goal as a single, clear, and specific objective.
2. Extract Resources: Identify any URLs or URIs mentioned in the user's goal and list them as the resource. Provide a short description (content) and the link for the resource.
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
  "resource": 
    {{
      "content": "a youtube video that needs to be downloaded",
      "link": "https://www.youtube.com/watch?v=1htKBjuUWec"
    }},
  "plan": [
    "Download the video from the provided URL",
    "Extract the audio and transcribe it to text",
    "Locate the part where the question is asked and Teal'c responds"
  ]
}}

where
  "objective" 's value in the json is a clear, one-sentence summary of the end goal,
  "resource" 's value in the json is a dictionary, containing:
    - "content": a brief description of what the resource is OR the contents of the file 
    - "link": the URL or URI mentioned in the task (AND NOT MENTIONED Available resources section already)
  "plan" 's value in the json is a list **ALWAYS SEPARATED BY PYTHON NEWLINE CHARACTER** like 
  [
    "A short explanation of the first logical step", 
    "The concluding step",
  ]
  **IMPORTANT**: you should only include the minimum number of steps to accomplish the task, do not include verification steps.
  **IMPORTANT**: do not include any json formatting directives, output plain json string.
  **IMPORTANT**: only include the resource that is explicitly mentioned in the user's goal/ask as URLs or URIs. 
  **IMPORTANT**: DO NOT DUPLICATE EXISTING RESOURCES !!!
   If no resource (link or path) is mentioned, do not include resource key in the answer 
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
You are a highly intelligent planning agent. Your primary function is to analyze a task's overall plan and the context of what has already been accomplished, then intelligently determine which step should be executed next and which capability to use.

Capabilities:
- `{AgentCapabilityDeepWebSearch.name}`: Find, evaluate, and download web content (e.g., articles, documents). This capability is for search and downloading web resources only, not for processing the content or getting any answers on the content.
- `{AgentCapabilityUnstructuredDataProcessor.name}`: Analyze, summarize, extract information from, or answer questions about, raw text already stored in memory.
- `{AgentCapabilityCodeWriterExecutor.name}`: Generate or execute code, solve mathematical problems, or perform complex logical operations and computations on files.
- `{AgentCapabilityAudioProcessor.name}`: Download audio file, transcribes speech.

Your task:
1. Review the full task plan and the context of what has already been accomplished
2. Intelligently determine which step from the plan should be executed next (it might not be the first unexecuted step - choose based on what makes sense given the context)
3. Select the SINGLE most appropriate capability for that step
4. Define a clear activity description that incorporates information from the context

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
Reasoning: Step 0 is completed (main actor identified as Leonardo DiCaprio), so we proceed to step 1 and incorporate the discovered name into the activity.

Example 2: Starting from the beginning when context is empty
Task: "Task(question='What are the main topics discussed in the uploaded podcast episode?', objective='Summarize the key themes of the podcast', plan=['Download and transcribe the audio file', 'Analyze the transcription and extract main topics'])"
Context: ""
Output:
{{
  "subplan": [
    {{
      "activity": "Write code to download and transcribe the speech from the podcast audio file",
      "capability": "{AgentCapabilityAudioProcessor.name}"
    }}
  ]
}}
Reasoning: No context provided, so we start with the first step in the plan.

Example 3: Skipping ahead when intermediate steps are already satisfied
Task: "Task(objective='Plot financial stocks prices', plan=['Download the CSV file from the provided URL', 'Load and analyze the file structure', 'Plot the stock prices'])"
Context: "Step 0: [ShortAnswer(answered=True, answer='Downloaded file to /tmp/stocks.csv')]\nStep 1: [ShortAnswer(answered=True, answer='CSV contains columns: Date, Symbol, Price, Volume')]"
Output:
{{
  "subplan": [
    {{
      "activity": "Write code to load /tmp/stocks.csv and create a plot of stock prices over time",
      "capability": "{AgentCapabilityCodeWriterExecutor.name}"
    }}
  ]
}}
Reasoning: Steps 0 and 1 are already completed, so we proceed to step 2 (plotting) and incorporate the discovered file path and column information.

Example 4: Using context to refine the next action
Task: "Task(objective='Summarize the article about climate change', plan=['Search for and download the article', 'Extract key information from the article'])"
Context: "Step 0: [ShortAnswer(answered=True, answer='Downloaded article from Nature.com about Arctic climate change, stored in /tmp/arctic_climate.pdf')]"
Output:
{{
  "subplan": [
    {{
      "activity": "Analyze and extract key information about Arctic climate change from the downloaded PDF at /tmp/arctic_climate.pdf",
      "capability": "{AgentCapabilityUnstructuredDataProcessor.name}"
    }}
  ]
}}
Reasoning: Step 0 is completed, so we move to step 1. The activity incorporates specific details from context (Arctic focus, PDF location).

---
Task: "{task}"
Full Plan: {task.plan}
Context: {context}

**CRITICAL INSTRUCTIONS**:
1. Analyze the context to understand what has already been accomplished
2. Determine which step from the plan should be executed next based on logical progression
3. Incorporate specific information from the context into your activity description (e.g., file paths, names, data discovered)
4. Output EXACTLY ONE capability action in the subplan array
5. If context is empty, start with the first step in the plan
6. If some steps are completed, proceed to the next logical step

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