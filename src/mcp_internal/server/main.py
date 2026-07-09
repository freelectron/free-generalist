import asyncio
import atexit
import os
import threading
from dataclasses import asdict, is_dataclass

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from clog import get_logger

logger = get_logger(__name__)

# WebSearchTool drives one Chrome driver/tab, and the browser-LLM
# (LLMBrowserServer) shares that same driver. Concurrent web_search calls would
# clobber the driver's active tab/state, so they must be serialized.
_chrome_lock = threading.Lock()

# IMPORTANT: in the MCP Python SDK the `lifespan` is NOT process-wide. With the
# streamable-http transport, StreamableHTTPSessionManager calls the low-level
# Server.run() once PER SESSION, and Server.run() is what enters/exits the
# lifespan (mcp/server/lowlevel/server.py: Server.run -> AsyncExitStack). So each
# new client session (each `initialize` handshake) re-enters lifespan and, on
# disconnect, tears it down. Building ChromeBrowser inside lifespan therefore
# spawns a new browser on every session (every request when the client
# reconnects) and quits it when the session ends.
#
# Fix: own the expensive browser at module scope (one per process) and have the
# per-session lifespan just hand out references. Teardown happens via atexit.
_shared_lock = threading.Lock()
_shared: dict | None = None

DEFAULT_SERVE_PORT = 7000


def _build_shared(llm_mode: str, dialer_host: str, dialer_port: int, dialer_token: str) -> dict:
    import mlflow
    from browser import ChromeBrowser
    from browser.search.web import BraveBrowser
    from generalist.dialer.core import LLMBrowserDialer, LLMBrowserServer, MLFlowLLMWrapper
    from generalist.tools import WebSearchTool

    load_dotenv()
    assert os.getenv("CHROME_USER_DATA_DIR"), "CHROME_USER_DATA_DIR env var is required"
    mlflow.set_experiment("mcp_web_search")

    chrome_browser = ChromeBrowser()

    if llm_mode == "dialer":
        llm_instance = LLMBrowserDialer(host=dialer_host, port=dialer_port, auth_token=dialer_token)
        logger.info(f"Using LLMBrowserDialer -> http://{dialer_host}:{dialer_port}")
    else:
        llm_instance = LLMBrowserServer(chrome_browser)
        logger.info("Using LLMBrowserServer (local Chrome)")

    llm = MLFlowLLMWrapper(llm_instance=llm_instance)
    search_session = BraveBrowser(browser=chrome_browser, session_id="mcp_brave")
    tool = WebSearchTool(search_session=search_session, llm=llm)

    shared = {"chrome_browser": chrome_browser, "tool": tool, "lock": _chrome_lock}
    atexit.register(_teardown_shared, shared)
    logger.info("Built shared Chrome browser + WebSearchTool (once per process)")
    return shared


def _teardown_shared(shared: dict) -> None:
    try:
        shared["chrome_browser"].driver.quit()
    except Exception as e:
        logger.error(f"Failed to quit Chrome driver on shutdown: {e}")


_shared_build_kwargs: dict = {}


def _get_shared() -> dict:
    global _shared
    with _shared_lock:
        if _shared is None:
            _shared = _build_shared(**_shared_build_kwargs)
        return _shared


def _jsonable(results: list[dict]) -> list[dict]:
    """Make WebSearchTool.run()'s raw output JSON-serializable.

    run() returns items shaped as {"search_result": WebSearchResult, "content": str}.
    WebSearchResult is a dataclass, which the MCP result serializer cannot handle
    directly. We convert it to a plain dict, preserving the raw structure.
    """
    out = []
    for item in results:
        new_item = dict(item)
        search_result = new_item.get("search_result")
        if is_dataclass(search_result) and not isinstance(search_result, type):
            new_item["search_result"] = asdict(search_result)
        out.append(new_item)
    return out


mcp = FastMCP(
    name="free-generalist-mcp",
    instructions="Local MCP server exposing simple utility tools.",
    host="127.0.0.1",
    port=9000,
)


@mcp.tool()
def sum_two_numbers(a: int, b: int) -> int:
    """Add two integers and return the sum."""
    logger.info(f"sum_two_numbers called with a={a}, b={b}")
    return a + b


@mcp.tool()
async def web_search(question: str) -> list[dict]:
    """Searches the web and downloads page content for a given question.

    Args:
        question: The user's query or question.
    """
    shared = _get_shared()
    tool = shared["tool"]
    lock = shared["lock"]
    logger.info(f"web_search called with question={question!r}")

    def _run_blocking() -> list[dict]:
        with lock:
            return _jsonable(tool.run(question=question))

    # tool.run() does blocking I/O (selenium + crawl4ai) so in order to not block FastMCP's event loop,
    # offload it to a worker thread.
    return await asyncio.to_thread(_run_blocking)


def run_server(
    host: str,
    port: int,
    llm_mode: str,
    dialer_host: str,
    dialer_port: int,
    dialer_token: str,
):
    global _shared_build_kwargs
    _shared_build_kwargs = {
        "llm_mode": llm_mode,
        "dialer_host": dialer_host,
        "dialer_port": dialer_port,
        "dialer_token": dialer_token,
    }
    _get_shared()
    # None disables FastMCP's Host-header validation, allowing connections from
    # any IP. Uvicorn still binds to `host`!
    mcp.settings.host = None
    mcp.settings.port = port
    logger.info(f"Starting MCP server on http://{host}:{port}{mcp.settings.streamable_http_path}")
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Free-generalist MCP server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind (use 0.0.0.0 to accept remote connections)")
    parser.add_argument("--port", type=int, default=DEFAULT_SERVE_PORT, help="Port to serve on")
    parser.add_argument(
        "--llm-mode",
        choices=["server", "dialer"],
        default="server",
        help="'server' = LLMBrowserServer (local Chrome); 'dialer' = LLMBrowserDialer (remote HTTP)",
    )
    parser.add_argument("--dialer-host", default="localhost", help="LLMBrowserDialer target host")
    parser.add_argument("--dialer-port", type=int, default=8000, help="LLMBrowserDialer target port")
    parser.add_argument("--dialer-token", default="", help="LLMBrowserDialer auth token")
    args = parser.parse_args()

    run_server(
        host=args.host,
        port=args.port,
        llm_mode=args.llm_mode,
        dialer_host=args.dialer_host,
        dialer_port=args.dialer_port,
        dialer_token=args.dialer_token,
    )
