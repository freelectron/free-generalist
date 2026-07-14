#!/opt/homebrew/bin/bash
set -a
source .env
source infra/services.env
set +a
uv run uvicorn src.api_server.main:app --host 0.0.0.0 --port "${API_PORT:-8000}"
