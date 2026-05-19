# vLLM — Source Code Analysis Index

## Quick Summary

vLLM is a high-throughput, memory-efficient inference and serving engine for large language models. Written primarily in Python with CUDA/Triton extensions, it implements PagedAttention for KV cache management, supports 100+ model architectures, and exposes OpenAI/Anthropic-compatible APIs. Key design choices include: continuous batching with iteration-level scheduling, block-level KV cache with prefix caching, pluggable attention backends, and speculative decoding.

## Documents

| File | Contents |
|------|----------|
| [01-overview.md](01-overview.md) | Project classification, tech stack, directory map, module diagram |
| [02-startup.md](02-startup.md) | Entry point, initialization sequence, thread/process model, memory layout |
| [03-api.md](03-api.md) | All API endpoints with sequence diagrams (OpenAI, Anthropic, management) |
| [04-shutdown.md](04-shutdown.md) | Signal handling, graceful shutdown sequence, resource cleanup inventory |
| [05-data-structures.md](05-data-structures.md) | Core data structures: Request, SamplingParams, KVCacheBlock, SchedulerOutput, etc. |
| [06-storage.md](06-storage.md) | Model weight formats (SafeTensors, GGUF, BnB), KV cache persistence, configuration |
| [07-protocol.md](07-protocol.md) | ZMQ IPC protocol, HTTP/SSE, NCCL, KV Connector, MCP tool protocol |
| [08-crosscutting.md](08-crosscutting.md) | Authentication, logging, Prometheus metrics, OpenTelemetry tracing, rate limiting |
| [09-extensions.md](09-extensions.md) | Plugin system, attention backends, structured output, quantization, speculative decoding |

## Key Files to Read First

| File | Reason |
|------|--------|
| `vllm/v1/engine/core.py` | Engine core: the central scheduling + execution loop |
| `vllm/v1/engine/async_llm.py` | AsyncLLM: main interface between API server and engine |
| `vllm/v1/core/sched/scheduler.py` | Scheduler: determines which requests get tokens each step |
| `vllm/v1/worker/gpu_model_runner.py` | GPU model runner: prepares inputs, executes forward pass |
| `vllm/entrypoints/openai/api_server.py` | API server: FastAPI app setup, router registration |
| `vllm/v1/request.py` | Request: core data structure flowing through the pipeline |
| `vllm/v1/core/kv_cache_utils.py` | KV cache block management: prefix caching, block hashing |
| `vllm/config/model.py` | ModelConfig: central configuration for model loading |
