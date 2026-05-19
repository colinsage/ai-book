# SGLang — Source Code Analysis

## Quick Summary

SGLang is a fast serving framework for large language models (LLMs), implemented primarily in Python with C++/CUDA extensions for performance-critical paths. Its key design choices include: (1) a multi-process architecture with separate HTTP server, GPU scheduler, and detokenizer processes communicating via ZMQ; (2) a radix-tree-based KV cache that enables automatic prefix sharing across requests; (3) support for tensor parallelism, pipeline parallelism, and data parallelism via NCCL; and (4) OpenAI-compatible HTTP API with additional endpoints for weight management, LoRA adapters, and constrained generation.

## Documents

| File | Contents |
|------|---------|
| [01-overview.md](01-overview.md) | Project classification, tech stack, module diagram |
| [02-startup.md](02-startup.md) | Entry point, init sequence, thread/process model, memory layout |
| [03-api.md](03-api.md) | All API endpoints with sequence diagrams |
| [04-shutdown.md](04-shutdown.md) | Signal handling, graceful shutdown, resource cleanup |
| [05-data-structures.md](05-data-structures.md) | Core structs/classes, field-by-field analysis |
| [06-storage.md](06-storage.md) | Persistent file formats and design rationale |
| [07-protocol.md](07-protocol.md) | ZMQ IPC protocol, HTTP API protocol, NCCL communication |
| [08-crosscutting.md](08-crosscutting.md) | Auth, logging, metrics, tracing, rate limiting |
| [09-extensions.md](09-extensions.md) | Plugin system and extension points |

## Key Files to Read First

| File | Reason |
|------|--------|
| `python/sglang/srt/entrypoints/engine.py` | Main Engine class — entry point for all initialization and shutdown |
| `python/sglang/srt/managers/scheduler.py` | Scheduler — the heart of batch scheduling, KV cache management, and GPU orchestration |
| `python/sglang/srt/entrypoints/http_server.py` | HTTP API layer — all REST endpoints and request routing |
| `python/sglang/srt/mem_cache/memory_pool.py` | Memory pool structures — ReqToTokenPool, TokenToKVPool, KV cache implementations |
| `python/sglang/srt/mem_cache/radix_cache.py` | Radix cache — the key innovation for automatic KV cache sharing |
| `python/sglang/srt/managers/schedule_batch.py` | Batch and request data structures — Req, ScheduleBatch, ModelWorkerBatch |
| `python/sglang/srt/server_args.py` | ServerArgs and PortArgs — all configuration and IPC channel definitions |
| `python/sglang/srt/managers/tokenizer_manager.py` | TokenizerManager — tokenization pipeline and request lifecycle management |
