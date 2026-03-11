import json
from typing import Any, AsyncGenerator
import os
import time
from dotenv import load_dotenv
from fastapi.responses import StreamingResponse

load_dotenv()

assert os.getenv("CHROME_USER_DATA_DIR", None)

from browser import CHATGPT_SESSION, DEEPSEEK_SESSION


def _sse_chunk(content: str, created: int) -> str:
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


def _sse_done(created: int) -> str:
    data = {
        'id': 'chatcmpl-123',
        'object': 'chat.completion.chunk',
        'created': created,
        'model': 'web',
        'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}],
    }
    return f"data: {json.dumps(data)}\n\ndata: [DONE]\n\n"


async def _stream_answer(answer: str) -> AsyncGenerator[str, None]:
    created = int(time.time())
    chunk_size = 200
    for i in range(0, len(answer), chunk_size):
        yield _sse_chunk(answer[i:i + chunk_size], created)
    yield _sse_done(created)


async def handle_chat_completions(body: dict[str, Any]):
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
    answer = CHATGPT_SESSION.send_message(str(body))
    # answer = DEEPSEEK_SESSION.send_message(str(body))

    if body.get('stream', False):
        return StreamingResponse(
            _stream_answer(answer),
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
