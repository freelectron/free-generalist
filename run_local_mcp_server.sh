#!/opt/homebrew/bin/bash
set -a
source .env
set +a
uv run python -c "from mcp_internal.server.main import run_server; run_server()"
