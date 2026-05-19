# vLLM — Shutdown & Cleanup

## 4.1 Signal Handling

| Signal | Handler | Location | Behavior |
|--------|---------|----------|----------|
| SIGTERM | `signal_handler()` | `api_server.py:setup_server()` (line ~567) | Raises `KeyboardInterrupt` to interrupt uvicorn during initialization |
| SIGTERM | Internal shutdown | `EngineCoreProc._perform_handshakes()` | Triggers graceful shutdown of engine core process |
| KeyboardInterrupt | Uvicorn default | — | Stops the ASGI server, triggers FastAPI lifespan shutdown |

**Note:** There is no explicit SIGINT handler — the default Python behavior (KeyboardInterrupt) is relied upon. SIGTERM handling is primarily for clean termination during the initialization phase.

## 4.2 Shutdown Sequence

### API Server Shutdown (FastAPI lifespan)

1. Cancel stats logging task (`server_utils.py:lifespan()` finally block)
2. Delete `app.state` to release engine references — triggers Python GC and `AsyncLLM.__del__()`

### AsyncLLM Shutdown (`async_llm.py:shutdown()`, line 259)

1. Call `shutdown_prometheus()` — cleanup Prometheus multiprocess directory
2. Call `renderer.shutdown()` — stop any active renderers
3. Call `engine_core.shutdown(timeout=timeout)` — graceful engine core termination

### Engine Core Shutdown (`core.py:shutdown()`, line 571)

1. Set `shutdown_state = EngineShutdownState.REQUESTED`
2. Call `model_executor.shutdown()` — stop all worker processes
3. Call `scheduler.shutdown()` — cleanup scheduler state

### Engine Core Step-Level Shutdown (`core.py:_handle_shutdown()`, line 1230)

When shutdown is requested during the step loop:

1. Check `shutdown_state` — if `RUNNING`, continue
2. If `REQUESTED`:
   - Read `shutdown_timeout` from `vllm_config`
   - If timeout == 0: immediate termination
   - Otherwise: reject new requests (`_reject_add_in_shutdown()`)
   - Set state to `SHUTTING_DOWN`
3. Wait for in-flight requests to complete (up to timeout)
4. Proceed with full shutdown

### CoreEngineProcManager Shutdown (`utils.py:shutdown()`, line 193)

1. Send shutdown signal to engine core process
2. Join process with timeout
3. Force-kill if process doesn't exit within timeout

### MultiprocExecutor Worker Shutdown

1. Send shutdown message via control queue
2. Each worker:
   - Destroys NCCL process groups (`destroy_distributed_environment()`)
   - Destroys model parallel state (`destroy_model_parallel()`)
   - Releases GPU memory (PyTorch CUDA cache clear)
3. Join worker processes with timeout

## 4.3 Resource Cleanup Inventory

| Resource | Cleanup Method | Location |
|----------|---------------|----------|
| GPU KV cache | Freed with model executor shutdown | `core.py:574` → `model_executor.shutdown()` |
| NCCL process groups | `destroy_distributed_environment()` | `multiproc_executor.py` |
| Model parallel groups | `destroy_model_parallel()` | `multiproc_executor.py` |
| ZMQ sockets | Closed on process exit | `core_client.py` — `close_sockets()` |
| Multiprocessing queues | `q.close()` | `async_llm.py:630, 850` |
| Prometheus temp dir | Auto-cleaned by `TemporaryDirectory` | `prometheus.py:setup_multiprocess_prometheus()` |
| CUDA graphs | Released with model executor | `gpu_model_runner.py` |
| Server socket | Closed by uvicorn on exit | `api_server.py` |
| AsyncIO tasks | Cancelled in lifespan finally block | `server_utils.py:lifespan()` |
| App state (engine refs) | `del app.state` in lifespan finally | `server_utils.py:lifespan()` |
| CUDA memory | PyTorch CUDA cache clearing on process exit | Worker process cleanup |
