import threading
import asyncio

from litellm.experimental_mcp_client import call_openai_tool, load_mcp_tools
from litellm.experimental_mcp_client.client import streamable_http_client
from mcp import ClientSession


class MCPConnection:
    def __init__(self, sse_url):
        self._loop = asyncio.new_event_loop()
        self._session = None

        # Daemon thread to keep the connection alive?
        # it will keep the self._loop running and set it as the main execution loop
        daemon_t = threading.Thread(target=self._runner, args=(sse_url,), daemon=True)
        daemon_t.start()

        # Block the event loop until the connection and client are established
        self._ready = asyncio.Event()
        self._block_until_ready()

    def _block_until_ready(self):
        asyncio.run_coroutine_threadsafe(self._ready.wait(), self._loop).result()

    def _runner(self, url):
        asyncio.set_event_loop(self._loop)
        async def hold():
            async with streamable_http_client(url) as (r, w, _):
                async with ClientSession(r,w) as session:
                    await  session.initialize()
                    self._session = session
                    # Indicator that the session has started and client is inited
                    self._ready.set()
                    # TODO: Does it wait until
                    await  asyncio.Event().wait()
        # ToDo: is it running forever because hold never returns?
        self._loop.run_until_complete(hold())

    def list_tools(self) -> list:
        coro = load_mcp_tools(session=self._session, format="openai")
        return asyncio.run_coroutine_threadsafe(coro=coro, loop=self._loop).result()

    def call_tool(self, tool_call: dict):
        coro = call_openai_tool(session=self._session, openai_tool=tool_call)
        return asyncio.run_coroutine_threadsafe(coro=coro, loop=self._loop).result()
