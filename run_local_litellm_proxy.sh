#!/usr/bin/bash
# Launch the standalone LiteLLM proxy (forward-everything + Prometheus metrics).
#
# The proxy runs on the host (binds 0.0.0.0 so other machines on the LAN can
# reach it) using a dedicated environment under infra/proxy/. Each caller is
# responsible for sending its own upstream credential — the proxy does not
# manage provider keys. See infra/proxy/README.md for how to point clients at it.
set -a
source .env
source infra/services.env
set +a
exec uv run --project infra/proxy litellm \
  --config infra/proxy/config.yaml \
  --host 0.0.0.0 \
  --port "${PROXY_PORT:-4000}"
