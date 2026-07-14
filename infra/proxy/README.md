# LiteLLM Proxy (forward-everything + Prometheus metrics)

A standalone [LiteLLM](https://docs.litellm.ai/docs/proxy/quick_start) proxy
that sits in front of every LLM the project uses, **forwards each request to
its real upstream**, and records per-request metrics (tokens, latency, cost,
cache hits) labelled by **model** and **origin** into Prometheus.

This folder is self-contained: everything you need to run or tweak the proxy
lives here. The main app does not depend on it.

```
Claude Code ─┐
opencode   ──┼─▶  LiteLLM Proxy (:4000)  ──▶  upstreams (ZAI/GLM, ollama, …)
dialer     ──┤         │  forward + emit metrics on /metrics/
             │         ▼
             │     Prometheus (:9090) ── Grafana (:3000)   [docker compose]
```

## Files

| file | purpose |
| --- | --- |
| `config.yaml` | litellm proxy config: provider wildcard routes, callbacks, bring-your-own-key |
| `metrics_callback.py` | custom callback → Prometheus instruments labelled `{model, origin}` + the credential-forwarding hook |
| `prometheus.yml` | scrape config (targets the host proxy `/metrics/`) |
| `grafana/` | auto-provisioned Prometheus datasource + starter dashboard |
| `pyproject.toml` | dedicated dependencies for this service (`litellm[proxy]`, `prometheus-client`) |

## Run it

From the **repo root**:

```bash
./run_local_litellm_proxy.sh                 # proxy on 0.0.0.0:${PROXY_PORT:-4000}
docker compose -f infra/docker-compose.yaml up -d prometheus grafana
```

- Prometheus UI → http://localhost:9090
- Grafana → http://localhost:3000 (admin / `${GRAFANA_ADMIN_PASSWORD:-admin}`), the
  *LiteLLM Proxy — Overview* dashboard is provisioned automatically.

> The proxy binds `0.0.0.0`, so tools on other machines point at this host's
> LAN IP. If the proxy runs on a different machine, update the `targets:` line
> in `prometheus.yml` accordingly.

## Environment variables (repo-root `.env`)

| var | required | default | notes |
| --- | --- | --- | --- |
| `LITELLM_MASTER_KEY` | yes | — | used only for litellm admin endpoints; **not** enforced on `/v1/chat/completions` or `/v1/messages` |
| `PROXY_PORT` | no | `4000` | port the proxy listens on |
| `PROMETHEUS_PORT` | no | `9090` | host port for Prometheus |
| `GRAFANA_PORT` | no | `3000` | host port for Grafana |
| `GRAFANA_ADMIN_USER` | no | `admin` | |
| `GRAFANA_ADMIN_PASSWORD` | no | `admin` | |
| provider keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, …) | only if you want the *proxy* to supply them | — | see Auth below |

## Auth model — bring your own key

The proxy **does not store or manage provider keys.** Each caller must send a
valid upstream credential and the proxy forwards it. This is implemented by a
`async_pre_call_hook` in `metrics_callback.py` that copies the caller's
`Authorization: Bearer …` **or** `x-api-key: …` onto the request before it is
sent upstream (litellm strips these by default; the hook restores them). The
wildcard routes in `config.yaml` deliberately define no `api_key`.

## Pointing clients at the proxy

Send models in litellm `provider/model` form so the wildcard routes match,
e.g. `openai/glm-5.2`, `anthropic/claude-sonnet-4-5-20250929`,
`ollama/qwen2.5:14b`.

- **Claude Code** → `ANTHROPIC_BASE_URL=http://<host>:4000` (uses `/v1/messages`).
- **opencode** → configure the provider `base_url: http://<host>:4000/v1`.
- **The repo's dialers** (`src/generalist/dialer/core.py`) — one line:
  ```python
  LLMZaiDialer(api_base="http://localhost:4000/v1", api_key=os.environ["ZAI_API_KEY"])
  ```
  (and set `extra_headers={"X-Client-Origin": "free-generalist"}` to tag it).

### Tagging the origin

Origin is resolved per request, first match wins:

1. `X-Client-Origin: <name>` request header (most reliable — set it explicitly).
2. `User-Agent` sniff (`claude`→`claude-code`, `opencode`→`opencode`, `cursor`,
   `continue`, `aider`, `zed`, …).
3. `user_api_key_team_alias` (if litellm virtual keys are ever enabled).
4. `unknown`.

## Metrics

All on the proxy's `/metrics/` endpoint (note the trailing slash — `/metrics`
307-redirects to it; Prometheus follows redirects). Custom family is prefixed
`proxy_llm_*`, labelled `{model, origin}`:

| metric | type | what |
| --- | --- | --- |
| `proxy_llm_requests_total` | counter | every forwarded request |
| `proxy_llm_errors_total` | counter | requests that ended in an error |
| `proxy_llm_input_tokens_total` | counter | prompt/input tokens |
| `proxy_llm_output_tokens_total` | counter | completion/output tokens |
| `proxy_llm_cache_read_tokens_total` | counter | provider prompt-cache reads |
| `proxy_llm_cache_write_tokens_total` | counter | provider prompt-cache writes |
| `proxy_llm_cost_usd_total` | counter | LiteLLM-estimated USD cost |
| `proxy_llm_request_latency_seconds` | histogram | end-to-end latency |

LiteLLM's own `litellm_*` family is also exported (the built-in `prometheus`
callback). Cost is `0` for models LiteLLM has no price table for — add cost
info when you wire concrete routes.

## Adding a concrete route (when you want an alias / custom base)

Edit `config.yaml` `model_list:` — there is a ready-to-use commented example for
the ZAI/GLM dialer. No metrics code changes are needed: every route is captured
automatically.

## Adjusting the proxy

Everything is in this folder: routes and callbacks in `config.yaml`, the
metric instruments and origin logic in `metrics_callback.py`, the scrape target
in `prometheus.yml`, dashboards under `grafana/`. Restart `run_proxy.sh` (or
`docker compose … restart prometheus grafana`) to pick up changes.
