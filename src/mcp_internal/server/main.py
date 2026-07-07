from mcp.server.fastmcp import FastMCP

from clog import get_logger

logger = get_logger(__name__)

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


def run_server(host: str = "127.0.0.1", port: int = 9000):
    mcp.settings.host = host
    mcp.settings.port = port
    logger.info(f"Starting MCP server on http://{host}:{port}{mcp.settings.streamable_http_path}")
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    run_server()
