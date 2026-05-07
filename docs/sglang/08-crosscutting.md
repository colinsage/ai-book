# 横切关注点

## 8.1 认证与授权

> SGLang 不实现认证或授权。没有 API 密钥验证、JWT 检查或权限模型。

这是设计如此：SGLang 旨在受信任的内部环境中运行（位于负载均衡器、API 网关或服务网格之后）。认证和限流预期由外部代理（例如 Nginx、Kong、AWS API Gateway）处理。

唯一与安全相关的功能是 `--api-key` 服务器参数，它设置一个在 HTTP 中间件层中检查的简单静态 API 密钥。

---

## 8.2 可观测性

### 日志

**库：** Python `logging` 模块，在 `configure_logger()`（utils）中配置

**日志级别：**
- 通过 `--log-level` 配置（默认值："info"）
- `--log-level-http` 用于单独的 HTTP 请求日志

**日志格式：**
```
[sglang::scheduler_TP0] 2024-01-15 10:30:00 INFO: Batch scheduled: 8 reqs, 1024 tokens
```

进程标识前缀：`sglang::scheduler_TP0`、`sglang::detokenizer` 等。

**结构化字段：** SGLang 默认不使用结构化日志。日志为人类可读的文本。

**请求日志：** 当启用 `--log-requests` 时，每个请求的到达、处理和完成都会记录时间信息。

### 指标

**库：** `prometheus_client`（metrics_collector.py:180）

**指标端点：** `GET /metrics`（如果设置了 `--enable-metrics`）

**关键指标：**

| 指标 | 类型 | 用途 |
|--------|------|---------|
| `num_running_reqs` | Gauge | 当前运行的请求 |
| `num_used_tokens` | Gauge | KV 缓存中的 Token |
| `token_usage` | Gauge | KV 缓存利用率百分比 |
| `full_token_usage` | Gauge | 包含开销的完整 Token 使用量 |
| `pending_prealloc_token_usage` | Gauge | 已预分配但未使用的 Token |
| `swa_token_usage` | Gauge | 滑动窗口注意力 Token 使用量 |
| `mamba_usage` | Gauge | Mamba 状态缓存使用量 |
| `gen_throughput` | Histogram | Token/秒生成吞吐量 |
| `prefill_time` | Histogram | 预填充延迟 |
| `decode_time` | Histogram | 解码延迟 |
| `e2e_request_latency` | Histogram | 端到端请求延迟 |
| `queue_time` | Histogram | 请求在队列中等待的时间 |
| `num_queue_reqs` | Gauge | 在队列中等待的请求 |

**自定义指标收集：**
- `SchedulerMetricsCollector`（metrics_collector.py）— 调度器的每次迭代指标
- `TokenizerManagerMetricsCollector` — 分词器端指标
- 优先级感知指标跟踪每个优先级级别的统计信息

### 链路追踪

**库：** OpenTelemetry（可选，通过 `--enable-trace`）

**配置：**
- `--otlp-traces-endpoint` — OTLP 导出器端点
- 进程和线程标签：`sglang::Scheduler`、`sglang::Prefill Scheduler` 等
- `trace_set_thread_info()` — 设置线程标签以进行追踪关联（scheduler.py:3596）

**追踪传播：** 追踪不会跨进程边界传播。每个进程（调度器、解分词器、分词器管理器）创建自己的追踪 Span。

### 崩溃转储

当设置了 `--crash-dump-folder` 时：
- 子进程中未处理的异常会触发诊断转储
- 转储包括调度器状态、内存池状态和待处理请求信息
- 文件写入到 `crash_dump_folder/scheduler_dump_{'{timestamp}'}.json`

---

## 8.3 限流与熔断

> SGLang 不在应用层实现限流或熔断。

相反，SGLang 通过其调度系统使用**隐式反压**：

### 隐式限流机制

1. **最大运行请求数** — 调度器对并发请求有硬性限制（`max_running_requests`）。一旦达到，新请求将在等待列表中排队。

2. **KV 缓存容量** — 如果 KV 缓存已满，新的预填充请求将被阻塞，直到现有请求完成并释放其 KV 缓存槽位。这是主要的反压机制。

3. **最大队列大小** — `--max-queued-requests` 限制可以等待的请求数量。超过此限制的请求将被拒绝并返回 HTTP 503。

4. **分块预填充** — 长的预填充请求被分解为多个块，防止单个长请求独占 GPU 时间。

5. **抢占** — 当内存紧张时，调度器可以抢占低优先级请求，为高优先级请求腾出空间。被抢占请求的 KV 缓存可以：
 - **换出到** CPU 内存（如果设置了带交换的 `--enable-chunked-prefill-size`）
 - **在请求重新调度时从头重新计算**

### 优先级调度

当启用 `--enable-priority-scheduling` 时：
- 请求具有整数优先级值
- 低优先级请求可以被抢占，为高优先级请求腾出空间
- `--priority-scheduling-preemption-threshold` 控制何时触发抢占
- `--schedule-low-priority-values-first` 反转优先级排序

### 多租户隔离

SGLang 不提供租户级别的隔离。所有请求共享相同的 KV 缓存池和 GPU 资源。

---

## 8.4 错误处理

### 子进程故障

- **SubprocessWatchdog**（engine.py:743）— 在后台线程中监控所有子进程。如果任何子进程意外死亡，看门狗会记录故障并可以触发服务器关闭。

- **SIGQUIT 传播** — 当子进程（调度器或解分词器）遇到未处理的异常时，它会在死亡前向父进程发送 SIGQUIT，确保整个服务器关闭而不是在降级状态下继续运行。

### 请求级错误处理

- 无效请求返回 HTTP 400 及错误详情
- 超时（通过 `--request-timeout`）中止超过时间限制的请求
- OOM（内存不足）条件触发运行中请求的抢占

### CUDA 错误处理

- 调度器进程中的 `faulthandler.enable()` 捕获原生崩溃追踪
- CUDA 错误通常会导致调度器进程崩溃，由看门狗捕获
