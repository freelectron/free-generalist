"""
LiteLLM proxy callback that records per-request Prometheus metrics labelled
by ``model`` and ``origin`` for *every* request the proxy forwards, regardless
of provider or client.

Origin resolution (first match wins):
  1. ``X-Client-Origin`` request header (tools/dialers set this explicitly).
  2. Sniffed from the ``User-Agent`` header (claude-code, opencode, ...).
  3. ``user_api_key_team_alias`` from litellm virtual-key metadata (if any).
  4. ``unknown``.

The callback is registered from ``config.yaml`` via the dotted module path
``metrics_callback.origin_metrics_callback`` (litellm resolves it relative to
the config file's directory). A module-level instance is exported so litellm
can attach it directly.

All instruments live on ``prometheus_client``'s default registry, which the
built-in LiteLLM ``prometheus`` callback serves on the proxy's ``/metrics``
endpoint — so these ``proxy_llm_*`` metrics appear next to litellm's own.
"""
from __future__ import annotations

import re
from typing import Any, Mapping, Optional

from litellm.integrations.custom_logger import CustomLogger
from prometheus_client import Counter, Histogram, REGISTRY

_LABELS = ["model", "origin"]


def _metric(cls, name: str, documentation: str, labelnames: list[str], **extra):
    """Create a prometheus instrument, returning an existing one if this module
    is ever executed twice (litellm imports custom callbacks via importlib, so a
    callback referenced from more than one config list would otherwise re-run
    this module body and hit a DuplicatedMetricError)."""
    try:
        return cls(name, documentation, labelnames, **extra)
    except ValueError:
        existing = getattr(REGISTRY, "_names_to_collectors", {}).get(name)
        if existing is not None:
            return existing
        raise


requests_total = _metric(
    Counter,
    "proxy_llm_requests_total",
    "Total LLM requests forwarded by the proxy.",
    _LABELS,
)
errors_total = _metric(
    Counter,
    "proxy_llm_errors_total",
    "Total LLM requests that ended in an error.",
    _LABELS,
)
input_tokens_total = _metric(
    Counter,
    "proxy_llm_input_tokens_total",
    "Prompt/input tokens reported by the upstream.",
    _LABELS,
)
output_tokens_total = _metric(
    Counter,
    "proxy_llm_output_tokens_total",
    "Completion/output tokens reported by the upstream.",
    _LABELS,
)
cache_read_tokens_total = _metric(
    Counter,
    "proxy_llm_cache_read_tokens_total",
    "Input tokens served from the provider prompt cache (cache read).",
    _LABELS,
)
cache_write_tokens_total = _metric(
    Counter,
    "proxy_llm_cache_write_tokens_total",
    "Input tokens written to the provider prompt cache (cache creation).",
    _LABELS,
)
cost_usd_total = _metric(
    Counter,
    "proxy_llm_cost_usd_total",
    "Estimated USD cost of the request, computed by LiteLLM.",
    _LABELS,
)
latency_seconds = _metric(
    Histogram,
    "proxy_llm_request_latency_seconds",
    "End-to-end latency of a forwarded LLM request (seconds).",
    _LABELS,
    buckets=(0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 240, 480),
)

_UA_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"claude", re.IGNORECASE), "claude-code"),
    (re.compile(r"opencode", re.IGNORECASE), "opencode"),
    (re.compile(r"cursor", re.IGNORECASE), "cursor"),
    (re.compile(r"continuedev|continue", re.IGNORECASE), "continue"),
    (re.compile(r"aider", re.IGNORECASE), "aider"),
    (re.compile(r"zed", re.IGNORECASE), "zed"),
    (re.compile(r"python-httpx|httpx|openai-python", re.IGNORECASE), "http-client"),
    (re.compile(r"curl", re.IGNORECASE), "curl"),
]


def _slug(value: str) -> str:
    """Normalise a free-form label value into a safe Prometheus label."""
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip()).strip("-").lower() or "unknown"


def _deep_get(data: Any, *path: str, default: Any = None) -> Any:
    cur: Any = data
    for key in path:
        if not isinstance(cur, Mapping):
            return default
        cur = cur.get(key)
        if cur is None:
            return default
    return cur


def _header_get(headers: Mapping[str, Any] | None, name: str) -> Optional[str]:
    if not headers:
        return None
    target = name.lower()
    for key, value in headers.items():
        if isinstance(key, str) and key.lower() == target and isinstance(value, str):
            return value
    return None


def _resolve_origin(slo: Mapping[str, Any] | None, metadata: Mapping[str, Any] | None) -> str:
    headers = _deep_get(slo, "metadata", "headers") or _deep_get(metadata, "headers")
    explicit = _header_get(headers if isinstance(headers, Mapping) else None, "x-client-origin")
    if explicit:
        return _slug(explicit)

    user_agent = _deep_get(slo, "metadata", "user_agent") or _deep_get(metadata, "user_agent")
    if isinstance(user_agent, str):
        for pattern, name in _UA_PATTERNS:
            if pattern.search(user_agent):
                return name

    team_alias = _deep_get(metadata, "user_api_key_team_alias")
    if isinstance(team_alias, str) and team_alias:
        return _slug(team_alias)

    return "unknown"


def _as_float(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _extract_credential(headers: Mapping[str, Any]) -> Optional[str]:
    """Pull the caller's upstream credential from the raw request headers.

    Accepts either OpenAI-style ``Authorization: Bearer <key>`` or Anthropic-style
    ``x-api-key: <key>`` (case-insensitive). Returns the bare key, or None.
    """
    if not isinstance(headers, Mapping):
        return None
    auth = _header_get(headers, "authorization")
    if isinstance(auth, str) and auth.lower().startswith("bearer "):
        key = auth[7:].strip()
        if key:
            return key
    xapikey = _header_get(headers, "x-api-key")
    if isinstance(xapikey, str) and xapikey.strip():
        return xapikey.strip()
    return None


class OriginMetricsCallback(CustomLogger):
    """Records model/origin-labelled metrics for every proxied LLM call."""

    async def async_pre_call_hook(
        self,
        user_api_key_dict,
        cache,
        data: dict,
        call_type,
    ):
        """Bring-your-own-key: copy the caller's own upstream credential onto the
        request so the proxy can forward it without ever storing provider keys.

        LiteLLM strips ``Authorization``/``x-api-key`` before the upstream call
        (they are treated as proxy-auth headers), so without this the wildcard
        routes — which deliberately define no ``api_key`` — would reach the
        upstream with no credentials. We read the original header from
        ``data["secret_fields"]["raw_headers"]`` and set ``data["api_key"]``,
        which litellm turns into the right provider credential (Bearer for
        openai/ollama/..., x-api-key for anthropic).
        """
        try:
            if data.get("api_key"):
                return None
            raw_headers = (data.get("secret_fields") or {}).get("raw_headers") or {}
            credential = _extract_credential(raw_headers)
            if credential:
                data["api_key"] = credential
        except Exception:
            pass
        return None

    def _record_success(self, kwargs: Mapping[str, Any], start_time: Any, end_time: Any) -> None:
        slo: Mapping[str, Any] | None = kwargs.get("standard_logging_object")
        if not isinstance(slo, Mapping):
            slo = None

        metadata = kwargs.get("litellm_params", {}).get("metadata") if isinstance(kwargs.get("litellm_params"), Mapping) else None
        if not isinstance(metadata, Mapping):
            metadata = _deep_get(slo, "metadata")

        model = kwargs.get("model") or _deep_get(slo, "model_group") or "unknown"
        origin = _resolve_origin(slo, metadata if isinstance(metadata, Mapping) else None)
        labels = {"model": str(model), "origin": str(origin)}

        usage_object = _deep_get(slo, "metadata", "usage_object") or {}
        prompt_details = usage_object.get("prompt_tokens_details") if isinstance(usage_object, Mapping) else {}

        cache_read = _as_float(usage_object.get("cache_read_input_tokens"))
        if cache_read == 0.0 and isinstance(prompt_details, Mapping):
            cache_read = _as_float(prompt_details.get("cached_tokens"))
        cache_write = _as_float(usage_object.get("cache_creation_input_tokens"))
        if cache_write == 0.0 and isinstance(prompt_details, Mapping):
            cache_write = _as_float(prompt_details.get("cache_creation_tokens"))

        requests_total.labels(**labels).inc()
        input_tokens_total.labels(**labels).inc(_as_float(_deep_get(slo, "prompt_tokens")))
        output_tokens_total.labels(**labels).inc(_as_float(_deep_get(slo, "completion_tokens")))
        cost_usd_total.labels(**labels).inc(_as_float(_deep_get(slo, "response_cost")))
        if cache_read > 0:
            cache_read_tokens_total.labels(**labels).inc(cache_read)
        if cache_write > 0:
            cache_write_tokens_total.labels(**labels).inc(cache_write)

        if start_time is not None and end_time is not None:
            latency_seconds.labels(**labels).observe((end_time - start_time).total_seconds())

    def _record_failure(self, kwargs: Mapping[str, Any]) -> None:
        slo: Mapping[str, Any] | None = kwargs.get("standard_logging_object")
        if not isinstance(slo, Mapping):
            slo = None
        metadata = kwargs.get("litellm_params", {}).get("metadata") if isinstance(kwargs.get("litellm_params"), Mapping) else None
        model = kwargs.get("model") or _deep_get(slo, "model_group") or "unknown"
        origin = _resolve_origin(slo, metadata if isinstance(metadata, Mapping) else None)
        errors_total.labels(model=str(model), origin=str(origin)).inc()
        requests_total.labels(model=str(model), origin=str(origin)).inc()

    # Sync hooks (direct litellm.completion path, e.g. the repo's dialers).
    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            self._record_success(kwargs, start_time, end_time)
        except Exception:
            pass

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        try:
            self._record_failure(kwargs)
        except Exception:
            pass

    # Async hooks (proxy server path — Claude Code, opencode, ...).
    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            self._record_success(kwargs, start_time, end_time)
        except Exception:
            pass

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        try:
            self._record_failure(kwargs)
        except Exception:
            pass


# Module-level instance — litellm loads this via `metrics_callback.origin_metrics_callback`.
origin_metrics_callback = OriginMetricsCallback()
