# llama.cpp — Startup Flow

## 2.1 Entry Points

llama.cpp has two main entry points:

### Server Entry Point

**File:** `tools/server/server.cpp:74` — `int main(int argc, char ** argv)`

### CLI Entry Point

**File:** `tools/cli/cli.cpp` — `int main(int argc, char ** argv)`

Both entry points follow a similar initialization pattern using shared `common_params` for argument parsing.

## 2.2 Server Initialization Sequence

The server startup flow in `tools/server/server.cpp`:

1. **Parse CLI flags** (server.cpp:82) — `common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SERVER)` parses all command-line arguments into `common_params`
2. **Validate batch parameters** (server.cpp:89) — Ensures `n_batch <= n_ubatch` for embeddings mode
3. **Auto-configure parallelism** (server.cpp:95-100) — If `n_parallel < 0`, sets `n_parallel = 4` and `kv_unified = true`
4. **Initialize backend** (server.cpp:110) — `llama_backend_init()` initializes GGML backend registry and discovers available hardware backends
5. **Initialize NUMA** (server.cpp:111) — `llama_numa_init(params.numa)` configures NUMA memory policy
6. **Create server context** (server.cpp:108) — `server_context ctx_server` — the primary inference orchestrator
7. **Initialize HTTP context** (server.cpp:116-120) — `ctx_http.init(params)` sets up the HTTP server (httplib-based), loads API keys, registers middleware
8. **Register API routes** (server.cpp:127-225) — All REST endpoints are registered via `ctx_http.get()` / `ctx_http.post()`
9. **Start HTTP server** (server.cpp:265) — `ctx_http.start()` begins listening on configured port (before model load, so `/health` is available)
10. **Load model** (server.cpp:280) — `ctx_server.load_model(params)` loads GGUF weights, creates `llama_model` and `llama_context`
11. **Register signal handlers** (server.cpp:302-313) — `sigaction(SIGINT/SIGTERM)` registers `signal_handler` for graceful shutdown
12. **Enter main loop** (server.cpp:336) — `ctx_server.start_loop()` blocks the main thread, processing inference tasks from the queue

### Router Mode

If `params.model.path` is empty, the server starts in **router mode** (server.cpp:130-170). In router mode:
- No model is loaded in the main process
- A `server_models_routes` object manages child server processes
- All inference endpoints are proxied to child servers
- Additional routes `/models/load` and `/models/unload` are registered for dynamic model management

## 2.3 CLI Initialization Sequence

1. **Parse CLI flags** — `common_params_parse()` into `common_params`
2. **Initialize backend** — `llama_backend_init()`
3. **Load model** — via `server_context::load_model()`
4. **Register signal handler** — SIGINT sets `g_is_interrupted` flag for mid-generation cancellation
5. **Enter interactive loop** — reads user input, formats chat messages, calls `generate_completion()`

## 2.4 Thread Model (Server)

| Thread | Created At | Role |
|--------|-----------|------|
| main | OS | CLI parsing, init, then blocks on `start_loop()` |
| http-thread | `ctx_http.start()` | HTTP request listener (httplib server) |
| io-worker-N | httplib internal | Process HTTP requests, dispatch to route handlers |
| compute | ggml threadpool | CPU inference workers (configurable via `n_threads`) |
| monitor-thread | `server_models::setup_child_server()` (conditional) | Heartbeat to router server when running as child |

### GPU Thread Usage

When a GPU backend is active (CUDA, Metal, Vulkan), compute graph execution is dispatched to the GPU via the backend's `graph_compute` function pointer. The CPU thread that submits the graph may block on `synchronize()` until GPU work completes. Multiple slots can be batched together for efficient GPU utilization.

## 2.5 Process Model (Router Mode)

In router mode, the server spawns **child server processes** via `server_models`:

- **Spawn**: Child processes are started with `fork/exec` or equivalent, each loading a specific model
- **IPC**: Child servers communicate with the router via HTTP (same machine, different ports)
- **Lifecycle**: The router tracks child server health via a monitor thread heartbeat mechanism
- **Scaling**: Multiple child servers can run concurrently, each on a different port

## 2.6 Memory Layout at Startup

### Model Weights (largest allocation)

- **Method**: Memory-mapped file (`mmap`) via `llama_mmap` (src/llama-mmap.cpp)
- The GGUF file is mapped into virtual address space; tensors are accessed directly from the mapping
- On GPU backends, tensors are copied to VRAM via `ggml_backend_buffer` allocation

### KV Cache

- **Allocation**: `ggml_backend_alloc_ctx_tensors()` reserves GPU or CPU memory for KV cache tensors
- Size depends on `n_ctx` (context window), `n_layer`, `n_embd_head_k`, `n_embd_head_v`, and `n_seq_max`
- Typically the second-largest memory consumer after model weights

### Compute Buffers

- **Purpose**: Intermediate tensors (activations) during graph execution
- **Allocation**: `ggml_backend_alloc_ctx_tensors()` during `sched_reserve()`
- Size depends on batch size and model architecture

### CPU Arena

- GGML uses a custom allocator (`ggml_alloc`) for tensor metadata (not data)
- `ggml_context` objects allocate tensor structs from pools, avoiding per-tensor `malloc`

### GPU VRAM Layout

On CUDA/Metal backends:
1. Model weights → VRAM (via `ggml_backend_buffer`)
2. KV cache → VRAM (via `ggml_backend_buffer`)
3. Compute scratch → VRAM (temporary, reused per graph execution)
