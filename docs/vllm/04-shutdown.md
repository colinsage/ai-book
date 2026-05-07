# vLLM — 关闭与清理

## 4.1 信号处理

| 信号 | 处理器 | 位置 | 行为 |
|------|--------|------|------|
| SIGTERM | `signal_handler()` | `api_server.py:setup_server()` (第 ~567 行) | 抛出 `KeyboardInterrupt` 以在初始化期间中断 uvicorn |
| SIGTERM | 内部关闭 | `EngineCoreProc._perform_handshakes()` | 触发引擎核心进程的优雅关闭 (graceful shutdown) |
| KeyboardInterrupt | Uvicorn 默认 | — | 停止 ASGI 服务器，触发 FastAPI lifespan 关闭 |

**注意：** 没有显式的 SIGINT 处理器 — 依赖 Python 的默认行为 (KeyboardInterrupt)。SIGTERM 处理主要用于在初始化阶段实现干净终止 (clean termination)。

## 4.2 关闭序列

### API 服务器关闭 (FastAPI lifespan)

1. 取消统计日志任务 (`server_utils.py:lifespan()` finally 块)
2. 删除 `app.state` 以释放引擎引用 — 触发 Python GC (垃圾回收) 和 `AsyncLLM.__del__()`

### AsyncLLM 关闭 (`async_llm.py:shutdown()`, 第 259 行)

1. 调用 `shutdown_prometheus()` — 清理 Prometheus 多进程目录
2. 调用 `renderer.shutdown()` — 停止所有活跃的渲染器 (renderer)
3. 调用 `engine_core.shutdown(timeout=timeout)` — 优雅的引擎核心终止 (graceful termination)

### 引擎核心关闭 (`core.py:shutdown()`, 第 571 行)

1. 设置 `shutdown_state = EngineShutdownState.REQUESTED`
2. 调用 `model_executor.shutdown()` — 停止所有工作进程 (worker process)
3. 调用 `scheduler.shutdown()` — 清理调度器 (scheduler) 状态

### 引擎核心步骤级关闭 (`core.py:_handle_shutdown()`, 第 1230 行)

当在步骤循环 (step loop) 中请求关闭时：

1. 检查 `shutdown_state` — 如果为 `RUNNING`，继续执行
2. 如果为 `REQUESTED`：
   - 从 `vllm_config` 读取 `shutdown_timeout`
   - 如果 timeout (超时) == 0：立即终止 (immediate termination)
   - 否则：拒绝新请求 (`_reject_add_in_shutdown()`)
   - 将状态设置为 `SHUTTING_DOWN`
3. 等待进行中的请求 (in-flight requests) 完成（最长等待至超时）
4. 继续执行完整关闭

### CoreEngineProcManager 关闭 (`utils.py:shutdown()`, 第 193 行)

1. 向引擎核心进程发送关闭信号
2. 带超时地等待进程结束 (join with timeout)
3. 如果进程在超时内未退出则强制终止 (force-kill)

### MultiprocExecutor 工作进程关闭

1. 通过控制队列 (control queue) 发送关闭消息
2. 每个工作进程 (worker)：
   - 销毁 NCCL 进程组 (`destroy_distributed_environment()`)
   - 销毁模型并行状态 (`destroy_model_parallel()`)
   - 释放 GPU 内存 (PyTorch CUDA 缓存清理)
3. 带超时地等待工作进程结束 (join with timeout)

## 4.3 资源清理清单

| 资源 | 清理方法 | 位置 |
|------|----------|------|
| GPU KV 缓存 (KV cache) | 随模型执行器关闭而释放 | `core.py:574` → `model_executor.shutdown()` |
| NCCL 进程组 (process groups) | `destroy_distributed_environment()` | `multiproc_executor.py` |
| 模型并行组 (model parallel groups) | `destroy_model_parallel()` | `multiproc_executor.py` |
| ZMQ 套接字 (sockets) | 进程退出时关闭 | `core_client.py` — `close_sockets()` |
| 多进程队列 (multiprocessing queues) | `q.close()` | `async_llm.py:630, 850` |
| Prometheus 临时目录 | 由 `TemporaryDirectory` 自动清理 | `prometheus.py:setup_multiprocess_prometheus()` |
| CUDA 图 (CUDA graphs) | 随模型执行器释放 | `gpu_model_runner.py` |
| 服务器套接字 (server socket) | uvicorn 退出时关闭 | `api_server.py` |
| AsyncIO 任务 (tasks) | 在 lifespan finally 块中取消 | `server_utils.py:lifespan()` |
| 应用状态 (引擎引用) | 在 lifespan finally 中 `del app.state` | `server_utils.py:lifespan()` |
| CUDA 内存 | 进程退出时 PyTorch CUDA 缓存清理 | 工作进程清理 (Worker process cleanup) |
