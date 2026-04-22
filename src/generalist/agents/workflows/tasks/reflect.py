from generalist.models.core import MLFlowLLMWrapper
from clog import get_logger


logger = get_logger(__name__)


def reflect_on_progress(
    task: str,
    context: str,
    agent_capability: str,
    llm: MLFlowLLMWrapper,
) -> str:
    prompt = f"""
    Role: {agent_capability}

    Task: {task}

    Context so far:
    {context}

    Reflect on the progress:
    1. What did you just learn from the latest tool output?
    2. How does this help with the task?
    3. Is the task complete, or what should you do next?

    Provide a brief reflection (2-3 sentences).
    """

    response = llm.complete(prompt)
    reflection = response.text.strip()
    logger.info(f"Reflection: {reflection}")
    return reflection
