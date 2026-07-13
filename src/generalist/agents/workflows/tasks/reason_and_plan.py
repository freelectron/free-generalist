import json
from dataclasses import dataclass

import regex as re

from generalist.dialer.core import MLFlowLLMWrapper
from generalist.tools import BaseTool
from clog import get_logger


logger = get_logger(__name__)


@dataclass
class ReasoningResult:
    """Structured output of the merged reflect + plan + evaluate step.

    Attributes:
        reflection: Brief analysis of the latest tool output (empty on cold start).
        plan: Self-contained description of the next action (empty if task complete).
        is_complete: Whether the task has been accomplished based solely on context.
        summary: Short phrase describing what was achieved.
    """
    reflection: str
    plan: str
    is_complete: bool
    summary: str


def _parse_reasoning_json(text: str) -> dict:
    """Extract a JSON object from an LLM response.

    Handles both clean JSON (enforced via response_format on litellm/glm) and
    JSON embedded in prose/markdown (free-text from browser backends).
    """
    response_text = text.strip()

    # Try a direct parse first (enforced-JSON backends).
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    # Fall back to extracting the JSON object. Try a markdown ```json``` fence
    # first, then any bare {...} object embedded in prose.
    json_match = re.search(r"json.*?(\{.*\})", response_text, re.DOTALL | re.IGNORECASE)
    if not json_match:
        json_match = re.search(r"(\{.*\})", response_text, re.DOTALL)

    code_string = json_match.group(1) if json_match else ""
    if len(code_string) <= 1:
        raise ValueError(f"Could not parse a JSON object from LLM response:\n{response_text}")

    return json.loads(code_string)


def _coerce_bool(value) -> bool:
    """Coerce a possibly-string boolean (e.g. "true"/"True"/True) to a real bool."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ["true", "1", "yes"]


def reason_and_plan(
    task: str,
    context: str,
    agent_capability: str,
    tools: list[BaseTool] | None,
    llm: MLFlowLLMWrapper,
    has_prior_output: bool,
) -> ReasoningResult:
    """Merged reflect + plan + evaluate step.

    Produces, in a single LLM call, a reflection on the latest tool output, a
    judgment of whether the task is complete, and a self-contained next plan.

    Args:
        has_prior_output: False on the cold-start iteration (no tool has run
            yet); the node then skips reflection and only plans.
    """
    tools_str = "\n".join([f"- {tool.name}: {tool.description}" for tool in (tools or [])])

    if has_prior_output:
        reflect_section = (
            "First, briefly reflect on the latest tool output (2-3 sentences):\n"
            "  - What did you just learn?\n"
            "  - How does it help with the task?\n"
        )
    else:
        reflect_section = (
            "No tool has been executed yet (cold start). Leave 'reflection' empty and "
            "'is_complete' as false; focus only on producing the first plan.\n"
        )

    prompt = f"""
    Role and agent capabilities: {agent_capability}.

    Task: {task}

    Context from previous steps:
    {context}

    {reflect_section}

    Available tools:
    {tools_str}

    Then, based **ONLY** on the resources above and without any additional assumptions,
    determine whether you have accomplished the task.

    If (and only if) the task is NOT yet complete, produce a plan for the next step:
    - DO NOT EXECUTE ANYTHING in this step; identify which tool to use next and why.
    - The plan MUST BE SELF-CONTAINED: include all key details (e.g. file paths, parameters,
      values from context) needed to execute the next step without referring back to prior context.
    - Be concise (2-3 sentences).

    Your response MUST be valid JSON in exactly this format:
    ```json
    {{
        "reflection": "<brief reflection, or empty on cold start>",
        "is_complete": <true or false>,
        "summary": "<a short phrase describing what was achieved>",
        "plan": "<self-contained next action, or empty string if the task is complete>"
    }}
    ```
    """

    # response_format is honoured by litellm/glm (enforced JSON) and silently
    # ignored by the browser backends, which return free-text parsed above.
    response = llm.complete(prompt, response_format={"type": "json_object"})

    data = _parse_reasoning_json(response.text or "")

    is_complete = _coerce_bool(data.get("is_complete", data.get("done", False)))

    result = ReasoningResult(
        reflection=str(data.get("reflection", "")).strip(),
        plan=str(data.get("plan", "")).strip(),
        is_complete=is_complete,
        summary=str(data.get("summary", "")).strip(),
    )

    logger.info(
        f"ReasoningResult: is_complete={result.is_complete} | "
        f"reflection={result.reflection!r} | plan={result.plan!r}"
    )

    return result
