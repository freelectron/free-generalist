#!/opt/homebrew/bin/bash
set -a
source .env
set +a
uv run python -m mcp_internal.server.main --host 0.0.0.0 --port 7000
