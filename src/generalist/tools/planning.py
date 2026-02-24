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
You are a planner. Based on what's already been done (Last output and Context), determine the NEXT SINGLE LOGICAL STEP to execute and select EXACTLY ONE tool to run it.

Task: {task}
Last output: {resource}
Context: {context}

Available tools:
- `{AgentDeepWebSearch.name}`: {AgentDeepWebSearch.capability}
- `{AgentUnstructuredDataProcessor.name}`: {AgentUnstructuredDataProcessor.capability}
- `{AgentCodeWriterExecutor.name}`: {AgentCodeWriterExecutor.capability}

CRITICAL RULES - YOU MUST FOLLOW THESE:
1. Output EXACTLY ONE activity that maps to EXACTLY ONE tool call
    **Break Down Tasks**: Always break down complex tasks into their smallest possible atomic steps.
    **Single Focus**: Ensure each step focuses on retrieving or performing exactly one piece of information or task at a time.
2. The activity description MUST contain ONLY ONE ACTION VERB (e.g., "search", "extract", "analyze", "calculate")
3. NEVER use "AND" in your activity description - this indicates multiple steps
4. NEVER use multiple verbs like "search and extract", "find and determine", "download and process"
5. If the task requires multiple steps, output ONLY THE VERY NEXT IMMEDIATE STEP needed right now
6. Each step should be atomic and indivisible - it should do exactly one thing

BAD EXAMPLES:
 - "Search for information AND extract the relevant details" - TWO steps (search, extract)
 - "Download the file and process its contents" - TWO steps (download, process)

GOOD EXAMPLES:
✓ "Search for Teal'c's YouTube video about Stargate SG-1"
✓ "Extract audio from /tmp/stargate_clip.mp4"
✓ "Read and parse the CSV file at /home/user/sales.csv"
✓ "Calculate the sum of values in the 'sales' column"

Output JSON format:
{{
  "activity": "Single atomic action with one verb, including specific details from Context",
  "tool": "The tool name that should execute this activity"
}}

REMEMBER:
- ONE activity = ONE verb = ONE tool call
- Break complex tasks into the smallest possible atomic steps
- If you're tempted to use "and", stop - you're doing multiple steps
- Output only the IMMEDIATE next step, not a sequence of steps
"""
    response = llm.complete(planning_prompt)
    response_text = response.text.strip()
    logger.info(f"Raw output: {response_text}")
    escaped_response_text = response_text.translate(str.maketrans({"\\":  r"\\"}))

    return json.loads(escaped_response_text)