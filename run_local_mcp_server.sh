#!/opt/homebrew/bin/bash
set -a
source .env
source infra/services.env
set +a
# The MCP server reaches the API server over localhost because the two services
# always co-locate (both local, or both on the same remote host).
uv run python -m mcp_internal.server.main \
    --host 0.0.0.0 \
    --port "${MCP_PORT:-7000}" \
    --dialer-host localhost \
    --dialer-port "${API_PORT:-8000}"
