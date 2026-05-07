# SGLang — 源码分析

## 概要

SGLang 是一个面向大语言模型（LLM）的快速服务框架，主要使用 Python 实现，并在性能关键路径上使用 C++/CUDA 扩展。其关键设计选择包括：（1）多进程架构，包含独立的 HTTP 服务器、GPU 调度器和解码器进程，通过 ZMQ 通信；（2）基于基数树的 KV 缓存，支持跨请求的自动前缀共享；（3）通过 NCCL 支持张量并行、流水线并行和数据并行；（4）兼容 OpenAI 的 HTTP API，并提供额外的端点用于权重管理、LoRA 适配器和受限生成。

## 文档目录

| 文件 | 内容 |
|------|------|
| [01-overview.md](01-overview.md) | 项目分类、技术栈、模块图 |
| [02-startup.md](02-startup.md) | 入口点、初始化序列、线程/进程模型、内存布局 |
| [03-api.md](03-api.md) | 所有 API 端点及序列图 |
| [04-shutdown.md](04-shutdown.md) | 信号处理、优雅关机、资源清理 |
| [05-data-structures.md](05-data-structures.md) | 核心结构体/类，逐字段分析 |
| [06-storage.md](06-storage.md) | 持久化文件格式及设计理由 |
| [07-protocol.md](07-protocol.md) | ZMQ IPC 协议、HTTP API 协议、NCCL 通信 |
| [08-crosscutting.md](08-crosscutting.md) | 认证、日志、指标、追踪、限流 |
| [09-extensions.md](09-extensions.md) | 插件系统与扩展点 |

## 建议优先阅读的关键文件

| 文件 | 原因 |
|------|------|
| `python/sglang/srt/entrypoints/engine.py` | 主引擎类 — 所有初始化和关机的入口点 |
| `python/sglang/srt/managers/scheduler.py` | 调度器 — 批处理调度、KV 缓存管理和 GPU 编排的核心 |
| `python/sglang/srt/entrypoints/http_server.py` | HTTP API 层 — 所有 REST 端点和请求路由 |
| `python/sglang/srt/mem_cache/memory_pool.py` | 内存池结构 — ReqToTokenPool、TokenToKVPool、KV 缓存实现 |
| `python/sglang/srt/mem_cache/radix_cache.py` | 基数缓存 — 实现 KV 缓存自动共享的关键创新 |
| `python/sglang/srt/managers/schedule_batch.py` | 批处理和请求数据结构 — Req、ScheduleBatch、ModelWorkerBatch |
| `python/sglang/srt/server_args.py` | ServerArgs 和 PortArgs — 所有配置及 IPC 通道定义 |
| `python/sglang/srt/managers/tokenizer_manager.py` | TokenizerManager — 分词管线和请求生命周期管理 |
