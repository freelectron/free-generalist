import threading
import asyncio

from litellm.experimental_mcp_client import call_openai_tool, load_mcp_tools
from litellm.experimental_mcp_client.client import streamable_http_client
from mcp import ClientSession

DEFAULT_CONNECT_TIMEOUT = 5.0


class MCPConnection:
    def __init__(self, sse_url, connect_timeout: float = DEFAULT_CONNECT_TIMEOUT):
        self._loop = asyncio.new_event_loop()
        self._session = None
        # Created before the daemon thread starts so _block_until_ready can wait
        # on the same loop the runner drives.
        self._ready = asyncio.Event()
        self._connect_error: Exception | None = None
        self._connect_timeout = connect_timeout

        # Daemon thread keeps the connection alive on its own event loop.
        daemon_t = threading.Thread(target=self._runner, args=(sse_url,), daemon=True)
        daemon_t.start()

        # Block until connected, or raise (fail-safe) on timeout/error.
        self._block_until_ready()

    def _block_until_ready(self):
        fut = asyncio.run_coroutine_threadsafe(self._ready.wait(), self._loop)
        try:
            fut.result(timeout=self._connect_timeout)
        except TimeoutError as e:
            raise ConnectionError(
                f"MCP server did not connect within {self._connect_timeout}s"
            ) from e
        if self._connect_error is not None:
            raise ConnectionError(f"MCP connect failed: {self._connect_error}") from self._connect_error

    def _runner(self, url):
        asyncio.set_event_loop(self._loop)

        async def hold():
            try:
                async with streamable_http_client(url) as (r, w, _):
                    async with ClientSession(r, w) as session:
                        await session.initialize()
                        self._session = session
                        # Indicator that the session has started and client is inited
                        self._ready.set()
                        # Hold the session open for the lifetime of the process.
                        await asyncio.Event().wait()
            except Exception as e:
                # Signal failure so _block_until_ready unblocks and re-raises
                # instead of hanging forever on an unreachable server.
                self._connect_error = e
                self._ready.set()

        self._loop.run_until_complete(hold())

    def list_tools(self) -> list:
        coro = load_mcp_tools(session=self._session, format="openai")
        return asyncio.run_coroutine_threadsafe(coro=coro, loop=self._loop).result()

    def call_tool(self, tool_call: dict):
        coro = call_openai_tool(session=self._session, openai_tool=tool_call)
        return asyncio.run_coroutine_threadsafe(coro=coro, loop=self._loop).result()
