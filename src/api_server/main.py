from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .handlers import (
    handle_chat_completions,
    handle_models_list,
    handle_embeddings,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="OpenAI-Compatible API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    return await handle_chat_completions(body)


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
