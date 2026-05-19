# llama.cpp — Source Code Analysis

## Quick Summary

llama.cpp is a C/C++ inference engine for large language models, implementing 100+ model architectures with a custom tensor compute library (GGML) and 15+ hardware backend accelerators. It uses a lazy compute graph with vtable-dispatched backends, a composable sampler chain for token selection, and the GGUF self-describing binary format for memory-mappable model weights. The project includes an OpenAI-compatible HTTP server with multi-slot concurrent inference.

## Documents

| File | Contents |
|------|---------|
| [01-overview.md](01-overview.md) | Project classification, tech stack, directory map, module diagram |
| [02-startup.md](02-startup.md) | Entry points, initialization sequence, thread/process model, memory layout |
| [03-api.md](03-api.md) | REST API endpoints (OpenAI/Anthropic compatible), C library API, sequence diagrams |
| [04-shutdown.md](04-shutdown.md) | Signal handling, graceful shutdown sequence, resource cleanup inventory |
| [05-data-structures.md](05-data-structures.md) | Core structs — ggml_tensor, ggml_cgraph, llama_model, llama_hparams, llama_batch, llama_layer, ggml_backend_i, llama_sampler_i |
| [06-storage.md](06-storage.md) | GGUF binary format specification, metadata keys, loading process, conversion scripts |
| [07-protocol.md](07-protocol.md) | HTTP/SSE transport patterns, streaming response format |
| [08-crosscutting.md](08-crosscutting.md) | API key authentication, Prometheus metrics, slot-based concurrency control |
| [09-extensions.md](09-extensions.md) | GGML backend vtable system, sampler chain, model architecture extension |

## Key Files to Read First

| File | Reason |
|------|--------|
| `include/llama.h` | Public API — all functions callers use (1565 lines) |
| `ggml/include/ggml.h` | GGML tensor types, ops, and core abstractions |
| `src/llama-model.cpp` | Model loading and forward pass implementation for all architectures |
| `tools/server/server.cpp` | Server entry point, route registration, startup/shutdown |
| `ggml/src/ggml-backend-impl.h` | Backend vtable definitions — the hardware extension point |
| `src/llama-sampler.cpp` | All sampling algorithm implementations |
| `ggml/include/gguf.h` | GGUF format specification and API |
| `src/llama-kv-cells.h` | KV cache cell tracking — core to understanding context management |
