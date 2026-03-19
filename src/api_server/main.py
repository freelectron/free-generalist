import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from .handlers import (
    handle_chat_completions,
    handle_models_list,
    handle_embeddings, handle_api_chat,
)
from clog import get_logger


logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="OpenAI-Compatible API",
    version="1.0.0",
    lifespan=lifespan,
)

def _build_full_request(request: Request, body: dict) -> dict:
    full_request = {
        "method": request.method,
        "url": str(request.url),
        "headers": dict(request.headers),
        "query_params": dict(request.query_params),
        "path_params": request.path_params,
        "client": request.client if request.client else None,
        "body": body,
    }
    return full_request


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()

    full_request = _build_full_request(request, body)
    return await handle_chat_completions(full_request)


@app.post("/api/chat")
async def api_chat(request: Request):
    body = await request.json()

    full_request = _build_full_request(request, body)

    logger.info(f"INCOMING:\n{json.dumps(full_request, indent=2, ensure_ascii=False, default=str)}")
    return await handle_api_chat(full_request)

@app.get("/api/tags")
async def api_tags():
    return {
        "models": [
            {
                "name": "web",
                "model": "web",
                "modified_at": "2025-08-22T18:36:05.414739637+02:00",
                "size": 8988124069,
                "digest": "web",
                "details": {
                  "parent_model": "",
                  "format": "gguf",
                  "family": "web",
                  "families": [
                      "web"
                  ],
                  "parameter_size": "14.8B",
                  "quantization_level": "Q4_K_M"
                }
            },
        ]
    }

@app.get("/v1/models")
async def models_list():
    return await handle_models_list()


@app.post("/v1/embeddings")
async def embeddings(request: Request):
    body = await request.json()
    return await handle_embeddings(body)


@app.get("/health")
async def health():
    return {"status": "ok"}


def run_server(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn

    uvicorn.run(app, host=host, port=port)
