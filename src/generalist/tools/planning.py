import json

from ..agents.core import AgentDeepWebSearch, AgentUnstructuredDataProcessor, AgentPlan, AgentCodeWriterExecutor
from ..models.core import llm
from clog import get_logger


logger = get_logger(__name__)


def parse_out_resource_link(task: str) -> dict[str, str]:
    """
    Extract only the resource (URL/URI) mentioned in the task description.

    This function asks the LLM to return a plain JSON string that contains only a
    `resource` field when a URL or URI is present in `task`. If no URL/URI is found,
    the model should return an empty JSON object (`{}`).

    Output contract (plain JSON only; no extra commentary):
      - If a single URL/URI is present:
        {
            "link": "<exact URL or URI found in the task>"
          }
      - If no URL/URI is present: {}
    The function returns the raw text from the LLM; callers must parse/validate it.
    """

    prompt = f"""
You are an extractor whose only job is to find a single URL or URI mentioned in the USER'S GOAL
and output a plain JSON string containing only that URL/URI.

USER'S GOAL:
{task}

Instructions:
1. If the USER'S GOAL contains a URL or URI (e.g. starting with http://, https://, ftp://, file://, or an absolute filesystem path like /home/...), 
    output exactly one JSON object with a single key "link" whose value is the exact URL or URI as it appears in the task
2. If no URL or URI is present, output exactly `{{}}` (an empty JSON object).
3. Output plain JSON only. Do not add any explanation, markdown, or extra characters.
4. Do not invent links. Only return links that appear verbatim in the USER'S GOAL.

Examples:
{{"link": "file:///home/user/data/sales.csv"}}
{{}}
"""
    task_response = llm.complete(prompt)

    return json.loads(task_response.text)


def determine_next_step(task: str, resource: str = "", context: str = "") -> dict:
    """
    Analyses a task and automatically determines which step from the plan should be executed next
    based on the context, then selects the single most appropriate capability for that step.

    Returns:
        A dataclass containing a single sub-task with the chosen capability.
    """
    planning_prompt = f"""
Your task: Based on what's already been done (Previous Steps and Context), determine the next logical step from, then select the appropriate agent to execute it.

Task: {task}
Last output: {resource}
Context: {context}

Available agents:
- `{AgentDeepWebSearch.name}`: {AgentDeepWebSearch.capability}
- `{AgentUnstructuredDataProcessor.name}`: {AgentUnstructuredDataProcessor.capability}
- `{AgentCodeWriterExecutor.name}`: {AgentCodeWriterExecutor.capability}

Instructions:
1. Review the Context to understand what has already been completed
2. Identify the next step that should be executed
3. Choose ONE agent that best matches the requirements of that step
4. Describe the activity with specific details from resources (file names, discovered information, etc.)

Example - When context has completed steps:
Task: ["Find Teal'c's response form a youtude video"]
Context: "Downloaded video saved as /tmp/stargate_clip.mp4"
Output: {{"activity": "Extract audio from /tmp/stargate_clip.mp4 and transcribe it to text", "agent": "{AgentCodeWriterExecutor.name}"}}

Example - When file analysis is needed:
Plan: ["Plot the sales data provided"]
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