########
## FIXME:Find a better way to user browser
import os
from dotenv import load_dotenv
load_dotenv()
assert os.getenv("CHROME_USER_DATA_DIR")
from browser import CHATGPT_SESSION, DEEPSEEK_SESSION
########

import json
from typing import Any, AsyncGenerator
import time
from fastapi.responses import StreamingResponse
from tenacity import sleep_using_event

from clog import get_logger


logger = get_logger(__name__)


def _get_llm_response(query:str):
    # answer = CHATGPT_SESSION.send_message(query)
    answer = DEEPSEEK_SESSION.send_message(query)
    return answer


def _chat_completions_sse_chunk(content: str, created: int) -> str:
    data = {
        'id': 'chatcmpl-123',
        'object': 'chat.completion.chunk',
        'created': created,
        'model': 'web',
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
        'model': 'web',
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
    """
    Handle POST /v1/chat/completions

    Expected body format (OpenAI compatible):
    {
        "model": "string",
        "messages": [{"role": "user", "content": "string"}],
        "temperature": 0.7,
        "max_tokens": 2048,
        "stream": false,
        ...
    }
    """
    answer = _get_llm_response(str(req))

    # SSE = server side streaming
    if req["body"].get('stream', False):
        return StreamingResponse(
            _chat_completions_stream_answer(answer),
            media_type='text/event-stream',
        )

    return {
        'id': 'chatcmpl-123',
        'object': 'chat.completion',
        'created': int(time.time()),
        'model': 'web',
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
        'model': 'web',
        'message': {
            "role": "assistant",
            "content": content,
        },
        "done": False,
    }
    return json.dumps(data) + "\n"

def _api_chat_sse_done(created: int) -> str:
    data = {
        'created_at': created,
        'model': 'web',
        'message': {
            "role": "assistant",
            "content": "",
        },
        "done": True,
        "done_reason": "stop",
    }
    return json.dumps(data) + "\n"

async def _api_chat_stream_answer(answer: str) -> AsyncGenerator[str, None]:
    created = int(time.time())
    chunk_size = 200
    for i in range(0, len(answer), chunk_size):
        yield _api_chat_sse_chunk(answer[i:i + chunk_size], created)
    yield _api_chat_sse_done(created)

async def handle_api_chat(req: dict):
    answer = _get_llm_response(str(req))

    # SSE = server side streaming
    if req["body"].get('stream'):
        return StreamingResponse(
            _api_chat_stream_answer(answer),
            media_type='text/event-stream',
        )

    return json.dumps({
        'created_at': int(time.time()),
        'model': 'web',
        'message': {
            "role": "assistant",
            "content": answer,
        },
        "done": True,
        "done_reason": "stop",
        }
    )

async def handle_models_list() -> dict[str, Any]:
    """
    Handle GET /v1/models

    Return list of available models.

    Expected response format:
    {
        "object": "list",
        "data": [
            {
                "id": "model-id",
                "object": "model",
                "created": 1234567890,
                "owned_by": "organization"
            }
        ]
    }
    """
    raise NotImplementedError("Implement handle_models_list in handlers.py")


async def handle_embeddings(body: dict[str, Any]) -> dict[str, Any]:
    """
    Handle POST /v1/embeddings

    Implement your embeddings logic here.

    Expected body format:
    {
        "model": "string",
        "input": "string" | ["string1", "string2"],
        ...
    }
    """
    raise NotImplementedError("Implement handle_embeddings in handlers.py")
