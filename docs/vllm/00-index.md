# vLLM — 源码分析索引

## 概述

vLLM 是一个高吞吐量、内存高效的大语言模型推理与服务引擎。主要使用 Python 编写，辅以 CUDA/Triton 扩展，它实现了 PagedAttention 用于 KV 缓存管理，支持 100+ 种模型架构，并暴露与 OpenAI/Anthropic 兼容的 API。关键设计选择包括：连续批处理（continuous batching）与迭代级调度（iteration-level scheduling）、带前缀缓存（prefix caching）的块级 KV 缓存（block-level KV cache）、可插拔的注意力后端（pluggable attention backends）以及推测解码（speculative decoding）。

## 文档列表

| 文件 | 内容 |
|------|------|
| [01-overview.md](01-overview.md) | 项目分类、技术栈、目录映射、模块图 |
| [02-startup.md](02-startup.md) | 入口点、初始化序列、线程/进程模型、内存布局 |
| [03-api.md](03-api.md) | 所有 API 端点及序列图（OpenAI、Anthropic、管理接口） |
| [04-shutdown.md](04-shutdown.md) | 信号处理、优雅关闭序列、资源清理清单 |
| [05-data-structures.md](05-data-structures.md) | 核心数据结构：Request、SamplingParams、KVCacheBlock、SchedulerOutput 等 |
| [06-storage.md](06-storage.md) | 模型权重格式（SafeTensors、GGUF、BnB）、KV 缓存持久化、配置 |
| [07-protocol.md](07-protocol.md) | ZMQ IPC 协议、HTTP/SSE、NCCL、KV Connector、MCP 工具协议 |
| [08-crosscutting.md](08-crosscutting.md) | 认证、日志、Prometheus 指标、OpenTelemetry 链路追踪、限流 |
| [09-extensions.md](09-extensions.md) | 插件系统、注意力后端、结构化输出、量化（quantization）、推测解码（speculative decoding） |

## 优先阅读的关键文件

| 文件 | 原因 |
|------|------|
| `vllm/v1/engine/core.py` | 引擎核心：中央调度 + 执行循环 |
| `vllm/v1/engine/async_llm.py` | AsyncLLM：API 服务器与引擎之间的主接口 |
| `vllm/v1/core/sched/scheduler.py` | 调度器（Scheduler）：决定每一步哪些请求获得 token |
| `vllm/v1/worker/gpu_model_runner.py` | GPU 模型运行器：准备输入、执行前向传播（forward pass） |
| `vllm/entrypoints/openai/api_server.py` | API 服务器：FastAPI 应用设置、路由注册 |
| `vllm/v1/request.py` | Request：流经整个管道的核心数据结构 |
| `vllm/v1/core/kv_cache_utils.py` | KV 缓存块管理：前缀缓存（prefix caching）、块哈希（block hashing） |
| `vllm/config/model.py` | ModelConfig：模型加载的中央配置 |
