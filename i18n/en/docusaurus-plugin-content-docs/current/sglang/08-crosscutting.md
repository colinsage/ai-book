# SGLang — Cross-Cutting Concerns

## Authentication & Authorization

> SGLang does not implement authentication or authorization. There is no API key validation, JWT checking, or permission model.

This is by design: SGLang is intended to run in trusted internal environments (behind a load balancer, API gateway, or service mesh). Authentication and rate limiting are expected to be handled by an external proxy (e.g., Nginx, Kong, AWS API Gateway).

The only security-related feature is the `--api-key` server argument, which sets a simple static API key checked in the HTTP middleware layer.

---

## Observability

### Logging

**Library:** Python `logging` module, configured in `configure_logger()` (utils)

**Log Levels:**
- Configured via `--log-level` (default: "info")
- `--log-level-http` for HTTP request logging separately

**Log Format:**
```
[sglang::scheduler_TP0] 2024-01-15 10:30:00 INFO: Batch scheduled: 8 reqs, 1024 tokens
```

Process identification prefixes: `sglang::scheduler_TP0`, `sglang::detokenizer`, etc.

**Structured Fields:** SGLang does not use structured logging by default. Logs are human-readable text.

**Request Logging:** When `--log-requests` is enabled, each request's arrival, processing, and completion are logged with timing information.

### Metrics

**Library:** `prometheus_client` (metrics_collector.py:180)

**Metrics Endpoint:** `GET /metrics` (if `--enable-metrics` is set)

**Key Metrics:**

| Metric | Type | Purpose |
|--------|------|---------|
| `num_running_reqs` | Gauge | Currently running requests |
| `num_used_tokens` | Gauge | Tokens in KV cache |
| `token_usage` | Gauge | KV cache utilization percentage |
| `full_token_usage` | Gauge | Full token usage including overhead |
| `pending_prealloc_token_usage` | Gauge | Pre-allocated but unused tokens |
| `swa_token_usage` | Gauge | Sliding window attention token usage |
| `mamba_usage` | Gauge | Mamba state cache usage |
| `gen_throughput` | Histogram | Tokens/second generation throughput |
| `prefill_time` | Histogram | Prefill latency |
| `decode_time` | Histogram | Decode latency |
| `e2e_request_latency` | Histogram | End-to-end request latency |
| `queue_time` | Histogram | Time requests spend in queue |
| `num_queue_reqs` | Gauge | Requests waiting in queue |

**Custom Metrics Collection:**
- `SchedulerMetricsCollector` (metrics_collector.py) — Per-iteration metrics from the scheduler
- `TokenizerManagerMetricsCollector` — Tokenizer-side metrics
- Priority-aware metrics track per-priority-level statistics

### Tracing

**Library:** OpenTelemetry (optional, via `--enable-trace`)

**Configuration:**
- `--otlp-traces-endpoint` — OTLP exporter endpoint
- Process and thread labels: `sglang::Scheduler`, `sglang::Prefill Scheduler`, etc.
- `trace_set_thread_info()` — Sets thread labels for trace correlation (scheduler.py:3596)

**Trace Propagation:** Traces are not propagated across process boundaries. Each process (scheduler, detokenizer, tokenizer manager) creates its own trace spans.

### Crash Dumping

When `--crash-dump-folder` is set:
- Unhandled exceptions in child processes trigger diagnostic dumps
- Dumps include scheduler state, memory pool status, and pending request info
- Files written to `crash_dump_folder/scheduler_dump_{timestamp}.json`

---

## Rate Limiting & Circuit Breaking

> SGLang does not implement rate limiting or circuit breaking at the application level.

Instead, SGLang uses **implicit backpressure** through its scheduling system:

### Implicit Rate Limiting Mechanisms

1. **Max Running Requests** — The scheduler has a hard limit on concurrent requests (`max_running_requests`). Once reached, new requests queue in the waiting list.

2. **KV Cache Capacity** — If the KV cache is full, new prefill requests are blocked until existing requests complete and free their KV cache slots. This is the primary backpressure mechanism.

3. **Max Queue Size** — `--max-queued-requests` limits how many requests can wait. Requests beyond this limit are rejected with HTTP 503.

4. **Chunked Prefill** — Long prefill requests are broken into chunks, preventing a single long request from monopolizing GPU time.

5. **Preemption** — When memory is tight, the scheduler can preempt lower-priority requests to make room for higher-priority ones. Preempted requests' KV cache is either:
   - **Swapped out** to CPU memory (if `--enable-chunked-prefill-size` is set with swap)
   - **Recomputed** from scratch when the request is rescheduled

### Priority Scheduling

When `--enable-priority-scheduling` is enabled:
- Requests have integer priority values
- Lower priority requests can be preempted to make room for higher priority ones
- `--priority-scheduling-preemption-threshold` controls when preemption kicks in
- `--schedule-low-priority-values-first` inverts the priority ordering

### Multi-Tenant Isolation

SGLang does not provide tenant-level isolation. All requests share the same KV cache pool and GPU resources.

---

## Error Handling

### Child Process Failures

- **SubprocessWatchdog** (engine.py:743) — Monitors all child processes in a background thread. If any child process dies unexpectedly, the watchdog logs the failure and can trigger server shutdown.

- **SIGQUIT propagation** — When a child process (scheduler or detokenizer) hits an unhandled exception, it sends SIGQUIT to the parent before dying, ensuring the entire server shuts down rather than continuing in a degraded state.

### Request-Level Error Handling

- Invalid requests return HTTP 400 with error details
- Timeouts (via `--request-timeout`) abort requests that exceed the time limit
- OOM (out of memory) conditions trigger preemption of running requests

### CUDA Error Handling

- `faulthandler.enable()` in the scheduler process captures native crash traces
- CUDA errors typically cause the scheduler process to crash, which is caught by the watchdog
