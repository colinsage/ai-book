# vLLM — Cross-Cutting Concerns

## 8.1 Authentication & Authorization

### Auth Scheme: API Key (Bearer Token)

**Implementation:** `AuthenticationMiddleware` in `vllm/entrypoints/openai/server_utils.py`

**How it works:**
1. API keys are configured via `--api-key` CLI flag or `VLLM_API_KEY` environment variable
2. Multiple keys are supported
3. Keys are SHA-256 hashed at startup and stored as digests
4. Each request's `Authorization: Bearer <token>` header is verified using constant-time comparison (`secrets.compare_digest`)
5. Authentication is **skipped** for: OPTIONS requests, paths not starting with `/v1` (e.g., `/health`, `/metrics`)

**No permission model:** There is no role-based access control (RBAC) or per-endpoint authorization. Any valid key grants full access.

**Security notes:**
- Tokens are hashed before storage to prevent accidental exposure in logs
- Constant-time comparison prevents timing attacks
- No rate limiting on authentication failures

---

## 8.2 Observability

### Logging

**Library:** Python `logging` module with custom configuration

**Configuration:**
- `VLLM_LOGGING_LEVEL` — Default level (INFO)
- `VLLM_LOGGING_CONFIG_PATH` — Custom logging config file
- `VLLM_LOGGING_PREFIX` — Prefix for all log messages
- `VLLM_LOGGING_COLOR` — Color output (auto/on/off)
- `VLLM_CONFIGURE_LOGGING` — Enable/disable logging setup

**Logger initialization:** `vllm/logger.py:init_logger()` — creates per-module loggers

**Structured fields:** Engine core logs iteration details including:
- Number of running/waiting requests
- KV cache utilization percentage
- Tokens scheduled per step
- Prefill/decode breakdown

### Metrics

**Library:** `prometheus_client` with multiprocess support

**Setup:** `vllm/v1/metrics/prometheus.py`
- Creates `PROMETHEUS_MULTIPROC_DIR` temp directory for multiprocess metrics
- Custom registry via `get_prometheus_registry()`
- Cleanup via `shutdown_prometheus()`

**Key Metrics:**

| Metric | Type | Purpose |
|--------|------|---------|
| `vllm:num_requests_running` | Gauge | Active running requests |
| `vllm:num_requests_waiting` | Gauge | Requests in waiting queue |
| `vllm:gpu_cache_usage_perc` | Gauge | KV cache GPU utilization |
| `vllm:cpu_cache_usage_perc` | Gauge | KV cache CPU utilization |
| `vllm:num_total_tokens` | Counter | Total tokens processed |
| `vllm:iteration_tokens_total` | Counter | Tokens per iteration |
| `vllm:e2e_request_latency_seconds` | Histogram | End-to-end request latency |
| `vllm:num_preemptions` | Counter | Request preemption count |
| `vllm:cache_eviction` | Counter | KV cache block evictions |
| `vllm:spec_decode_num_drafts` | Counter | Speculative decode drafts |
| `vllm:spec_decode_num_accepts` | Counter | Speculative decode accepts |

**Metrics endpoint:** `GET /metrics` (standard Prometheus exposition format)

**Stats logger:** `vllm/v1/metrics/loggers.py` — periodic stats logging at `VLLM_LOG_STATS_INTERVAL`

**Performance stats:** `vllm/v1/metrics/perf.py` — `PerfStats` tracks per-iteration timing

### Tracing

**Library:** OpenTelemetry (optional)

**Implementation:** `vllm/tracing/` — pluggable tracing backend

**Architecture:**
- `_REGISTERED_TRACING_BACKENDS` dict maps backend names to implementations
- Default backend: `"otel"` (OpenTelemetry)
- `is_otel_available()` checks for `opentelemetry` package
- `init_otel_tracer()` initializes the tracer provider
- `instrument_otel()` — decorator for automatic span creation
- `instrument_manual()` — manual span creation

**Trace propagation:**
- Trace context extracted from HTTP request headers via `extract_trace_context()`
- `SpanAttributes` defines standard attribute names (model, request_id, etc.)
- `SpanKind` defines span kinds (SERVER, INTERNAL, etc.)

**Worker tracing:** Separate `init_otel_worker_tracer()` for worker process traces

---

## 8.3 Rate Limiting & Circuit Breaking

> vLLM does **not** implement built-in rate limiting or circuit breaking. These concerns are expected to be handled by external infrastructure (e.g., API gateways, load balancers, Kubernetes).

**Load-aware call mechanism** (`vllm/entrypoints/utils.py:load_aware_call`): A decorator that can reject requests when the engine is overloaded, but this is a backpressure mechanism, not a rate limiter.

**Engine iteration timeout** (`VLLM_ENGINE_ITERATION_TIMEOUT_S=60`): If a single engine step takes longer than this, the engine is considered stuck and will be terminated — this acts as a safety mechanism rather than a rate limiter.

**Scheduler throttling:** The scheduler naturally throttles via token budget constraints:
- `max_num_scheduled_tokens` — limits tokens per step
- `max_num_running_reqs` — limits concurrent requests
- KV cache capacity — implicit backpressure when cache is full (requests are preempted or kept waiting)
