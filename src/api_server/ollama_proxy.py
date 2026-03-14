import json

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
import uvicorn

PROXY_PORT = 11435       # port this proxy listens on
OLLAMA_PORT = 11434      # standard ollama port
OLLAMA_BASE = f"http://localhost:{OLLAMA_PORT}"

app = FastAPI()
client = httpx.AsyncClient(base_url=OLLAMA_BASE, timeout=None)


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
async def proxy(path: str, request: Request):
    url = f"/{path}"
    if request.url.query:
        url = f"{url}?{request.url.query}"

    body = await request.body()

    # DEBUG
    try:
        parsed = json.loads(body)
        print("[REQUEST]\n", json.dumps(parsed, indent=2, ensure_ascii=False))
    except Exception:
        print("[REQUEST]\n", body)

    req = client.build_request(
        method=request.method,
        url=url,
        headers={k: v for k, v in request.headers.items() if k.lower() != "host"},
        content=body,
    )

    response = await client.send(req, stream=True)

    content = await response.aread()
    await response.aclose()

    # DEBUG
    try:
        parsed = json.loads(content)
        print("[RESPONSE]\n", json.loads(parsed, indent=2, ensure_ascii=False))
    except Exception:
        print("[RESPONSE]\n", str(content))

    # Detect streaming responses
    content_type = response.headers.get("content-type", "")
    is_streaming = (
        "text/event-stream" in content_type
        or response.headers.get("transfer-encoding", "").lower() == "chunked"
    )

    if is_streaming:
        async def stream_body():
            yield content

        return StreamingResponse(
            stream_body(),
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=content_type or None,
        )

    return Response(
        content=content,
        status_code=response.status_code,
        headers=dict(response.headers),
        media_type=content_type or None,
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PROXY_PORT)