#!/opt/homebrew/bin/bash
set -a
source .env 
set +a 
uv run uvicorn src.api_server.main:app --host 0.0.0.0 --port 8000 