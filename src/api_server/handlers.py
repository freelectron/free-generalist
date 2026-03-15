import json
from typing import Any, AsyncGenerator
import time
from fastapi.responses import StreamingResponse

from clog import get_logger
from generalist.openclaw.llm import get_llm_response

logger = get_logger(__name__)

MODEL_NAME_OPENCLAW = 'web'

def _chat_completions_sse_chunk(content: str, created: int) -> str:
    data = {
        'id': 'chatcmpl-123',
        'object': 'chat.completion.chunk',
        'created': created,
        'model': MODEL_NAME_OPENCLAW,
        'choices': [
            {
                'index': 0,
                'delta': {'role': 'assistant', 'content': content},
                'finish_reason': None,
            }
        ],
    }
    return f"data: {json.dumps(data)}\n\n"

def _chat_completions_sse_done(created: int) -> str:
    data = {
        'id': 'chatcmpl-123',
        'object': 'chat.completion.chunk',
        'created': created,
        'model': MODEL_NAME_OPENCLAW,
        'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}],
    }
    return f"data: {json.dumps(data)}\n\ndata: [DONE]\n\n"

async def _chat_completions_stream_answer(answer: str) -> AsyncGenerator[str, None]:
    created = int(time.time())
    chunk_size = 200
    for i in range(0, len(answer), chunk_size):
        yield _chat_completions_sse_chunk(answer[i:i + chunk_size], created)
    yield _chat_completions_sse_done(created)

async def handle_chat_completions(req: dict[str, Any]):
    answer, tool = get_llm_response(str(req))

    if tool:
        raise NotImplementedError("Calling ClosedAI API with tools is not implemented.")

    # server side streaming
    if req["body"].get('stream', False):
        return StreamingResponse(
            _chat_completions_stream_answer(answer),
            media_type='text/event-stream',
        )

    return {
        'id': 'chatcmpl-123',
        'object': 'chat.completion',
        'created': int(time.time()),
        'model': MODEL_NAME_OPENCLAW,
        'choices': [
            {
                'index': 0,
                'message': {
                    'role': 'assistant',
                    'content': answer,
                },
                'finish_reason': 'stop',
            }
        ],
        'usage': {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0,
        },
    }


def _api_chat_sse_chunk(content: str, created: int) -> str:
    data = {
        'created_at': created,
        'model': MODEL_NAME_OPENCLAW,
        'message': {
            "role": "assistant",
            "content": content,
        },
        "done": False,
    }
    return json.dumps(data) + "\n"

def _api_chat_tool(created: int, tool: dict) -> str:
    data = {
        'created_at': created,
        'model': MODEL_NAME_OPENCLAW,
        'message': {
            "role": "assistant",
            "content": f"Trying to create a tool: {tool}",
            'tool_calls': [tool],
        },
        "done": False,
    }

    return json.dumps(data) + "\n"

def _api_chat_sse_done(created: int) -> str:
    data = {
        'created_at': created,
        'model': MODEL_NAME_OPENCLAW,
        'message': {
            "role": "assistant",
            "content": "",
        },
        "done": True,
        "done_reason": "stop",
    }
    return json.dumps(data) + "\n"

async def _api_chat_stream_answer(answer: str, tool: dict) -> AsyncGenerator[str, None]:
    created = int(time.time())
    chunk_size = 200
    for i in range(0, len(answer), chunk_size):
        yield _api_chat_sse_chunk(answer[i:i + chunk_size], created)

    if tool:
        yield _api_chat_tool(created, tool)

    yield _api_chat_sse_done(created)

async def handle_api_chat(req: dict):
    answer, tool = get_llm_response(str(req))

    # SSE = server side streaming
    if req["body"].get('stream'):
        return StreamingResponse(
            _api_chat_stream_answer(answer, tool),
            media_type='text/event-stream',
        )

    return json.dumps({
        'created_at': int(time.time()),
        'model': MODEL_NAME_OPENCLAW,
        'message': {
            "role": "assistant",
            "content": answer,
        },
        "done": True,
        "done_reason": "stop",
        }
    )

async def handle_models_list() -> dict[str, Any]:
    raise NotImplementedError("Implement handle_models_list in handlers.py")


async def handle_embeddings(body: dict[str, Any]) -> dict[str, Any]:
    raise NotImplementedError("Implement handle_embeddings in handlers.py")
