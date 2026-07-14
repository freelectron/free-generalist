from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, Request, Depends

from generalist.dialer.core import LLMBrowserServer
from .handlers import handle_chat_completions
from clog import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    import os
    from dotenv import load_dotenv
    from browser import ChromeBrowser
    load_dotenv()
    assert os.getenv("CHROME_USER_DATA_DIR", None)
    chrome_browser = ChromeBrowser()
    llm = LLMBrowserServer(chrome_browser)
    assert llm
    app.state.llm = llm
    yield
    del llm


def get_llm(request: Request) -> LLMBrowserServer:
    return request.app.state.llm


LLMDep = Annotated[LLMBrowserServer, Depends(get_llm)]

app = FastAPI(
    title="OpenAI-Compatible Browser API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, llm: LLMDep):
    body = await request.json()
    return await handle_chat_completions({"body": body}, llm)


@app.get("/v1/models")
async def models_list():
    return {
        "object": "list",
        "data": [
            {
                "id": "web",
                "object": "model",
                "created": 0,
                "owned_by": "browser",
            }
        ],
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


def run_server(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)
