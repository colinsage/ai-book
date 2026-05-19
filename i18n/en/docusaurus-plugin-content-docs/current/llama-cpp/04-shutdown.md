# llama.cpp — Shutdown & Cleanup

## 4.1 Signal Handling

The server catches the following signals:

| Signal | Platform | Handler Location | Behavior |
|--------|----------|-----------------|----------|
| SIGINT | Unix/macOS | server.cpp:302-306 `sigaction()` | Graceful shutdown |
| SIGTERM | Unix/macOS | server.cpp:307 `sigaction()` | Graceful shutdown |
| CTRL_C_EVENT | Windows | server.cpp:309-312 `SetConsoleCtrlHandler()` | Graceful shutdown |

### Signal Handler Implementation (server.cpp:27-36)

```c
static void signal_handler(int signal) {
    if (is_terminating.test_and_set()) {
        // Second Ctrl+C: force exit
        fprintf(stderr, "Received second interrupt, terminating immediately.\n");
        exit(1);
    }
    shutdown_handler(signal);
}
```

Key design: an `atomic_flag` (`is_terminating`) ensures that:
- **First SIGINT/SIGTERM**: Triggers graceful shutdown via `shutdown_handler`
- **Second SIGINT/SIGTERM**: Calls `exit(1)` immediately — escape hatch if graceful shutdown hangs

## 4.2 Shutdown Sequence

### Single-Model Server (server.cpp:294-344)

1. **Signal received** → `signal_handler()` called
2. **Set terminating flag** — `is_terminating.test_and_set()` prevents double-handling
3. **Invoke shutdown_handler** — calls `ctx_server.terminate()` (server.cpp:296)
4. **`ctx_server.terminate()`** — calls `queue_tasks.terminate()` which unblocks `start_loop()`
5. **Main loop exits** — `start_loop()` returns (server.cpp:336)
6. **Execute cleanup** (server.cpp:257-262):
   - `ctx_http.stop()` — stops HTTP server, closes listening socket
   - `ctx_server.terminate()` — waits for in-flight inference tasks
   - `llama_backend_free()` — releases GGML backend resources
7. **Join HTTP thread** (server.cpp:339-340) — `ctx_http.thread.join()`
8. **Join monitor thread** (server.cpp:342-343) — if child server mode
9. **Print timing stats** (server.cpp:346-353)
10. **Exit** with code 0

### Router Server (server.cpp:233-248)

1. **Signal received** → `signal_handler()` → `ctx_http.stop()`
2. **HTTP server stops** — main thread unblocks from `thread.join()` (server.cpp:319-321)
3. **Cleanup** (server.cpp:236-242):
   - `models_routes->models.unload_all()` — terminates all child server processes
   - `llama_backend_free()`
4. **Exit**

## 4.3 Resource Cleanup Inventory

| Resource | Cleanup Method | Location |
|----------|---------------|---------|
| HTTP server | `ctx_http.stop()` | server-http.cpp |
| Inference queue | `queue_tasks.terminate()` | server-queue.cpp |
| llama_context | `llama_free()` | llama.cpp (via server_context destructor) |
| llama_model | `llama_model_free()` | llama.cpp (via server_context destructor) |
| KV cache | Implicit in context free | llama-context.cpp:~llama_context |
| GGML backends | `llama_backend_free()` | ggml-backend.cpp |
| GPU VRAM | `ggml_backend_buffer_free()` | per-backend (cuda, metal, etc.) |
| Memory-mapped model | `munmap()` / `UnmapViewOfFile()` | llama-mmap.cpp:~llama_mmap |
| Child processes | `models.unload_all()` | server-models.cpp |
| Thread pool | `ggml_threadpool_free()` | ggml-threading.cpp |
| Sampler chain | `llama_sampler_free()` | llama-sampler.cpp |
| Grammar | `llama_grammar_free()` | llama-grammar.cpp |
| LoRA adapters | `llama_adapter_lora_free()` | llama-adapter.cpp |
