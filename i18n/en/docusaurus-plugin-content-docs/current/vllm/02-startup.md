# vLLM — Startup Flow

## 2.1 Entry Point

The primary entry point for server mode is:

```
vllm/entrypoints/cli/serve.py → main() via vllm CLI
```

The CLI dispatches to `vllm serve <model>` which invokes the OpenAI API server.

**Argument / Flag Parsing:**
- `vllm/entrypoints/openai/cli_args.py` defines `EngineCLIConfig` with 100+ CLI options
- Key flags: `--model`, `--tensor-parallel-size`, `--max-model-len`, `--gpu-memory-utilization`, `--api-key`, `--host`, `--port`

**Config Loading:**
- CLI flags are merged with environment variables (`VLLM_*` in `vllm/envs.py`)
- Config is assembled into a `VllmConfig` object that is passed to all subsystems

## 2.2 Core Initialization Sequence

### Server Mode (vllm serve)

1. **Parse CLI flags** (`cli_args.py:make_cli_args()` — builds `EngineCLIConfig`)
2. **Create socket** (`api_server.py:setup_server()` — bind port before engine init to avoid race conditions with Ray)
3. **Register SIGTERM handler** (`api_server.py:setup_server()` — raises `KeyboardInterrupt` on SIGTERM during init)
4. **Build AsyncLLM** (`async_llm.py:AsyncLLM.from_engine_args()`)
   - a. Resolve `VllmConfig` from engine args
   - b. Select executor class (`Executor.get_class()`): Ray, Multiproc, or Uniproc based on `distributed_executor_backend`
   - c. Initialize `InputProcessor`, `OutputProcessor`, `Detokenizer`
   - d. Launch engine core process(es) via `CoreEngineProcManager`
5. **Initialize EngineCore** (`core.py:EngineCore.__init__()`)
   - a. Create `Scheduler` with KV cache config
   - b. Initialize `StructuredOutputManager`
   - c. Call `_initialize_kv_caches()` — determines available GPU memory, computes block count, creates KV cache config
   - d. Initialize model executor (loads model weights onto GPU)
6. **Build FastAPI app** (`api_server.py:build_and_serve()`)
   - a. Register API routers (models, completions, chat completions, responses, embeddings, Anthropic, pooling, etc.)
   - b. Add `AuthenticationMiddleware` if `--api-key` or `VLLM_API_KEY` is set
   - c. Add CORS middleware
   - d. Initialize serving state objects (`OpenAIServingChat`, `OpenAIServingCompletion`, etc.)
7. **Start Uvicorn** — serves the FastAPI app on the bound socket
8. **Freeze GC heap** (`server_utils.py:lifespan()` — marks startup heap as static to reduce GC pauses)

### Engine Core Process (Multiproc Mode)

When using multiprocessing (default for multi-GPU), the engine core runs in a child process:

1. **Fork/spawn worker processes** — one per GPU via `OMPProcessManager`
2. **Initialize distributed backend** — NCCL process groups for TP/PP/DP
3. **Load model weights** — each worker loads its shard via `DefaultModelLoader`
4. **Warm up / compile** — CUDA graph capture, torch.compile tracing
5. **Enter step loop** — `EngineCore.step_with_batch_queue()` runs continuously

## 2.3 Thread Model

| Thread Name | Created At | Role |
|-------------|-----------|------|
| Main (uvicorn) | OS | ASGI server, request handling |
| Input handler | `AsyncLLM._run_output_handler()` | Reads engine outputs from ZMQ, dispatches to output processor |
| Output handler (asyncio) | `AsyncLLM.output_handler()` | Processes model outputs, drives detokenization |
| Engine core | `CoreEngineProcManager` | Scheduling + model execution loop (separate process in multiproc mode) |
| Model worker-N | `MultiprocExecutor` | GPU model execution per TP rank (separate process) |
| Signal callback | `CoreEngineProcManager` | Dedicated thread for safe signal handling |
| Stats logger | `lifespan()` | Periodic stats logging (VLLM_LOG_STATS_INTERVAL) |

## 2.4 Process Model

In **multiproc mode** (default for `tensor-parallel-size > 1` or explicit selection):

| Process | Spawned By | IPC Mechanism |
|---------|-----------|---------------|
| Engine Core (rank 0) | `CoreEngineProcManager` | ZMQ input queue + ZMQ output socket |
| Worker rank N | `MultiprocExecutor` | NCCL for tensor communication; multiprocessing Queue for control messages |
| API server | Main process | In-process (via AsyncLLM) |

In **Ray mode**, workers are Ray actors communicating via Ray's distributed runtime.

In **uniproc mode**, everything runs in a single process with no IPC.

## 2.5 Memory Layout at Startup

### GPU VRAM Allocations

| Region | Purpose | Size Determination |
|--------|---------|-------------------|
| Model weights | Transformer parameters (sharded across TP) | Model file size / TP size |
| KV cache blocks | Key-value cache for all batched requests | `(total_gpu_mem - model_mem) * gpu_memory_utilization` / `block_size` |
| Activation buffers | Intermediate tensors during forward pass | Computed from `max_num_batched_tokens` and hidden dimension |
| CUDA graph buffers | Captured graph memory for common batch sizes | Pre-allocated for specific batch size buckets |

### CPU Memory

| Region | Purpose |
|--------|---------|
| Tokenizer | HuggingFace fast tokenizer model |
| Request state | Pending/running request metadata |
| KV cache block tables | Block ID mappings (pinned memory for GPU transfer) |
| Prompt token IDs | Tokenized prompt buffers |

### Key memory calculation (`_initialize_kv_caches` in `core.py:232`):

1. Worker reports available memory via `determine_available_memory()`
2. Total KV cache memory = `available_memory * gpu_memory_utilization`
3. Block count = `total_kv_memory / bytes_per_block`
4. Each block stores `block_size` tokens × `2 (K+V)` × `num_layers` × `head_dim` × `num_kv_heads / tp_size` × `dtype_size`
