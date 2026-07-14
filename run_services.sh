#!/opt/homebrew/bin/bash
# =============================================================================
# Combined launcher: start the API server, the LiteLLM proxy and the MCP server
# together. Use this on whichever laptop hosts the services (local or remote).
#
# Topology + ports come from infra/services.env; secrets come from .env. To host
# the services on a remote laptop, edit infra/services.env there (REMOTE block)
# and run this script on that host — the services bind 0.0.0.0, so they are
# reachable from the agent laptop over the LAN/Tailscale IP.
#
# The three services always co-locate, so service-to-service links use localhost
# (MCP -> API on localhost:${API_PORT}; proxy -> API on localhost:${API_PORT}).
# Only the agent/dialer side reads the *_ENDPOINT values from services.env.
# =============================================================================
set -uo pipefail
set -a
source .env
source infra/services.env
set +a

MCP_PORT="${MCP_PORT:-7000}"
API_PORT="${API_PORT:-8000}"
PROXY_PORT="${PROXY_PORT:-4000}"

PIDS=()

cleanup() {
    echo
    echo "Stopping services ..."
    for pid in "${PIDS[@]:-}"; do
        [ -n "$pid" ] && kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 1' INT TERM

echo "Starting API server on 0.0.0.0:${API_PORT} ..."
uv run uvicorn src.api_server.main:app --host 0.0.0.0 --port "${API_PORT}" &
PIDS+=("$!")

echo "Starting LiteLLM proxy on 0.0.0.0:${PROXY_PORT} ..."
uv run --project infra/proxy litellm \
    --config infra/proxy/config.yaml \
    --host 0.0.0.0 \
    --port "${PROXY_PORT}" &
PIDS+=("$!")

echo "Starting MCP server on 0.0.0.0:${MCP_PORT} (dialer -> localhost:${API_PORT}) ..."
uv run python -m mcp_internal.server.main \
    --host 0.0.0.0 \
    --port "${MCP_PORT}" \
    --dialer-host localhost \
    --dialer-port "${API_PORT}" &
PIDS+=("$!")

echo
echo "All services up. Press Ctrl-C to stop."
echo "  API    : http://localhost:${API_PORT}/health"
echo "  Proxy  : http://localhost:${PROXY_PORT}/health/liveness"
echo "  MCP    : http://localhost:${MCP_PORT}/mcp"
wait
