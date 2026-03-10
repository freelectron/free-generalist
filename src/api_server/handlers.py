from typing import Any


async def handle_chat_completions(body: dict[str, Any]) -> dict[str, Any]:
    """
    Handle POST /v1/chat/completions

    Implement your chat completions logic here.

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
    raise NotImplementedError("Implement handle_chat_completions in handlers.py")


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
