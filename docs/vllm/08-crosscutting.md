# vLLM — 横切关注点

## 8.1 认证与授权

### 认证方案：API 密钥（Bearer Token）

**实现：** `vllm/entrypoints/openai/server_utils.py` 中的 `AuthenticationMiddleware`

**工作原理：**
1. API 密钥通过 `--api-key` CLI 标志或 `VLLM_API_KEY` 环境变量进行配置
2. 支持多个密钥
3. 密钥在启动时进行 SHA-256 哈希处理，并以摘要形式存储
4. 每个请求的 `Authorization: Bearer ***` 头部使用常量时间比较（`secrets.compare_digest`）进行验证
5. 以下情况**跳过**认证：OPTIONS 请求、不以 `/v1` 开头的路径（例如 `/health`、`/metrics`）

**无权限模型：** 没有基于角色的访问控制（RBAC）或按端点的授权。任何有效密钥都授予完全访问权限。

**安全注意事项：**
- 令牌在存储前进行哈希处理，以防止在日志中意外泄露
- 常量时间比较防止时序攻击
- 认证失败没有速率限制

---

## 8.2 可观测性

### 日志

**库：** Python `logging` 模块，带有自定义配置

**配置：**
- `VLLM_LOGGING_LEVEL` — 默认日志级别（INFO）
- `VLLM_LOGGING_CONFIG_PATH` — 自定义日志配置文件
- `VLLM_LOGGING_PREFIX` — 所有日志消息的前缀
- `VLLM_LOGGING_COLOR` — 彩色输出（auto/on/off）
- `VLLM_CONFIGURE_LOGGING` — 启用/禁用日志设置

**日志初始化器：** `vllm/logger.py:init_logger()` — 创建按模块划分的日志器

**结构化字段：** 引擎核心记录迭代详情，包括：
- 正在运行/等待的请求数量
- KV cache 利用率百分比
- 每步调度的 token 数
- Prefill/Decode 分解

### 指标

**库：** `prometheus_client`，支持多进程

**设置：** `vllm/v1/metrics/prometheus.py`
- 为多进程指标创建 `PROMETHEUS_MULTIPROC_DIR` 临时目录
- 通过 `get_prometheus_registry()` 创建自定义注册表
- 通过 `shutdown_prometheus()` 进行清理

**关键指标：**

| 指标 | 类型 | 用途 |
|--------|------|---------|
| `vllm:num_requests_running` | Gauge | 活跃运行中的请求 |
| `vllm:num_requests_waiting` | Gauge | 等待队列中的请求 |
| `vllm:gpu_cache_usage_perc` | Gauge | KV cache GPU 利用率 |
| `vllm:cpu_cache_usage_perc` | Gauge | KV cache CPU 利用率 |
| `vllm:num_total_tokens` | Counter | 处理的总 token 数 |
| `vllm:iteration_tokens_total` | Counter | 每次迭代的 token 数 |
| `vllm:e2e_request_latency_seconds` | Histogram | 端到端请求延迟 |
| `vllm:num_preemptions` | Counter | 请求抢占计数 |
| `vllm:cache_eviction` | Counter | KV cache 块驱逐数 |
| `vllm:spec_decode_num_drafts` | Counter | 投机解码草稿数 |
| `vllm:spec_decode_num_accepts` | Counter | 投机解码接受数 |

**指标端点：** `GET /metrics`（标准 Prometheus 展示格式）

**统计日志器：** `vllm/v1/metrics/loggers.py` — 按 `VLLM_LOG_STATS_INTERVAL` 周期性记录统计信息

**性能统计：** `vllm/v1/metrics/perf.py` — `PerfStats` 跟踪每次迭代的时间

### 链路追踪

**库：** OpenTelemetry（可选）

**实现：** `vllm/tracing/` — 可插拔的追踪后端

**架构：**
- `_REGISTERED_TRACING_BACKENDS` 字典将后端名称映射到实现
- 默认后端：`"otel"`（OpenTelemetry）
- `is_otel_available()` 检查 `opentelemetry` 包是否可用
- `init_otel_tracer()` 初始化追踪器提供者
- `instrument_otel()` — 用于自动创建 span 的装饰器
- `instrument_manual()` — 手动创建 span

**追踪上下文传播：**
- 通过 `extract_trace_context()` 从 HTTP 请求头部提取追踪上下文
- `SpanAttributes` 定义标准属性名称（model、request_id 等）
- `SpanKind` 定义 span 种类（SERVER、INTERNAL 等）

**Worker 追踪：** 为 Worker 进程追踪提供单独的 `init_otel_worker_tracer()`

---

## 8.3 速率限制与熔断

> vLLM **不**实现内置的速率限制或熔断。这些关注点预期由外部基础设施处理（例如 API 网关、负载均衡器、Kubernetes）。

**负载感知调用机制**（`vllm/entrypoints/utils.py:load_aware_call`）：一个装饰器，当引擎过载时可以拒绝请求，但这是一种反压机制，而非速率限制器。

**引擎迭代超时**（`VLLM_ENGINE_ITERATION_TIMEOUT_S=60`）：如果单个引擎步骤耗时超过此值，引擎将被视为卡住并被终止——这是一种安全机制，而非速率限制器。

**调度器节流：** 调度器通过 token 预算约束自然地进行节流：
- `max_num_scheduled_tokens` — 限制每步的 token 数
- `max_num_running_reqs` — 限制并发请求数
- KV cache 容量 — 当缓存已满时的隐式反压（请求被抢占或保持等待）
