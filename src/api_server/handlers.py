import json
import time
from typing import Any, AsyncGenerator

from fastapi.responses import StreamingResponse

import tiktoken

from clog import get_logger
from generalist.dialer.core import LLMBrowserServer

logger = get_logger(__name__, simple=True)
MODEL_NAME_BROWSER = "fg/web"

# The browser LLM is scraped from a web UI and never reports real token usage.
# Estimate it with tiktoken so the proxy can log non-zero token metrics with the
# same schema as the GLM/ZAI dialer. cl100k_base is an approximation.
_TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    try:
        return len(_TIKTOKEN_ENC.encode(text or ""))
    except Exception:
        return 0


def _extract_prompt(messages: list[dict]) -> str:
    """Concatenate all message contents into a single prompt string."""
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            # vision / multi-part content — extract text parts only
            content = " ".join(p.get("text", "") for p in content if p.get("type") == "text")
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


def _sse_chunk(content: str, created: int) -> str:
    data = {
        "id": "chatcmpl-browser",
        "object": "chat.completion.chunk",
        "created": created,
        "model": MODEL_NAME_BROWSER,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": content},
                "finish_reason": None,
            }
        ],
    }
    return f"data: {json.dumps(data)}\n\n"


def _sse_done(created: int) -> str:
    data = {
        "id": "chatcmpl-browser",
        "object": "chat.completion.chunk",
        "created": created,
        "model": MODEL_NAME_BROWSER,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    return f"data: {json.dumps(data)}\n\ndata: [DONE]\n\n"


async def _stream_response(answer: str) -> AsyncGenerator[str, None]:
    created = int(time.time())
    chunk_size = 200
    for i in range(0, len(answer), chunk_size):
        yield _sse_chunk(answer[i : i + chunk_size], created)
    yield _sse_done(created)


async def handle_chat_completions(req: dict[str, Any], llm: LLMBrowserServer):
    messages = req["body"].get("messages", [])
    prompt = _extract_prompt(messages)
    llm_response = llm.complete(prompt)
    content = llm_response.text or ""
    logger.info(f"[LLM] Response:\n{content}")

    if req["body"].get("stream", False):
        return StreamingResponse(
            _stream_response(content),
            media_type="text/event-stream",
        )

    prompt_tokens = _count_tokens(prompt)
    completion_tokens = _count_tokens(content)
    return {
        "id": "chatcmpl-browser",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_NAME_BROWSER,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
