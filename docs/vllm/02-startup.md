# vLLM — 启动流程

## 2.1 入口点

服务器模式的主入口点为：

```
vllm/entrypoints/cli/serve.py → main() via vllm CLI
```

CLI 分发到 `vllm serve <model>`，该命令调用 OpenAI API 服务器。

**参数 / 标志解析：**
- `vllm/entrypoints/openai/cli_args.py` 定义了包含 100+ CLI 选项的 `EngineCLIConfig`
- 关键标志：`--model`、`--tensor-parallel-size`、`--max-model-len`、`--gpu-memory-utilization`、`--api-key`、`--host`、`--port`

**配置加载：**
- CLI 标志与环境变量合并（`VLLM_*` 定义在 `vllm/envs.py` 中）
- 配置组装为 `VllmConfig` 对象，传递给所有子系统

## 2.2 核心初始化序列

### 服务器模式 (vllm serve)

1. **解析 CLI 标志**（`cli_args.py:make_cli_args()` — 构建 `EngineCLIConfig`）
2. **创建套接字**（`api_server.py:setup_server()` — 在引擎初始化之前绑定端口，以避免与 Ray 的竞争条件）
3. **注册 SIGTERM 处理器**（`api_server.py:setup_server()` — 在初始化期间收到 SIGTERM 时抛出 `KeyboardInterrupt`）
4. **构建 AsyncLLM**（`async_llm.py:AsyncLLM.from_engine_args()`）
   - a. 从引擎参数解析 `VllmConfig`
   - b. 选择执行器类（`Executor.get_class()`）：基于 `distributed_executor_backend` 选择 Ray、Multiproc 或 Uniproc
   - c. 初始化 `InputProcessor`、`OutputProcessor`、`Detokenizer`
   - d. 通过 `CoreEngineProcManager` 启动引擎核心进程
5. **初始化 EngineCore**（`core.py:EngineCore.__init__()`）
   - a. 创建带有 KV 缓存配置的 `Scheduler`
   - b. 初始化 `StructuredOutputManager`
   - c. 调用 `_initialize_kv_caches()` — 确定可用 GPU 显存，计算块数量，创建 KV 缓存配置
   - d. 初始化模型执行器（将模型权重加载到 GPU）
6. **构建 FastAPI 应用**（`api_server.py:build_and_serve()`）
   - a. 注册 API 路由（models、completions、chat completions、responses、embeddings、Anthropic、pooling 等）
   - b. 如果设置了 `--api-key` 或 `VLLM_API_KEY`，则添加 `AuthenticationMiddleware`
   - c. 添加 CORS 中间件
   - d. 初始化服务状态对象（`OpenAIServingChat`、`OpenAIServingCompletion` 等）
7. **启动 Uvicorn** — 在绑定的套接字上服务 FastAPI 应用
8. **冻结 GC 堆**（`server_utils.py:lifespan()` — 将启动堆标记为静态，以减少 GC 暂停）

### 引擎核心进程 (Multiproc 模式)

当使用多进程模式（多 GPU 的默认模式）时，引擎核心在子进程中运行：

1. **Fork/spawn 工作进程** — 通过 `OMPProcessManager` 为每个 GPU 创建一个
2. **初始化分布式后端** — 为 TP/PP/DP 建立 NCCL 进程组
3. **加载模型权重** — 每个工作进程通过 `DefaultModelLoader` 加载其分片
4. **预热 / 编译** — CUDA graph 捕获、torch.compile 追踪
5. **进入步进循环** — `EngineCore.step_with_batch_queue()` 持续运行

## 2.3 线程模型

| 线程名称 | 创建位置 | 角色 |
|----------|---------|------|
| Main (uvicorn) | OS | ASGI 服务器，请求处理 |
| Input handler | `AsyncLLM._run_output_handler()` | 从 ZMQ 读取引擎输出，分发到输出处理器 |
| Output handler (asyncio) | `AsyncLLM.output_handler()` | 处理模型输出，驱动反分词化 (detokenization) |
| Engine core | `CoreEngineProcManager` | 调度 + 模型执行循环（multiproc 模式下为独立进程） |
| Model worker-N | `MultiprocExecutor` | 每个 TP rank 的 GPU 模型执行（独立进程） |
| Signal callback | `CoreEngineProcManager` | 专用于安全信号处理的线程 |
| Stats logger | `lifespan()` | 周期性统计日志记录 (VLLM_LOG_STATS_INTERVAL) |

## 2.4 进程模型

在 **multiproc 模式**（`tensor-parallel-size > 1` 或显式选择时的默认模式）下：

| 进程 | 由谁创建 | IPC 机制 |
|------|---------|---------|
| Engine Core (rank 0) | `CoreEngineProcManager` | ZMQ 输入队列 + ZMQ 输出套接字 |
| Worker rank N | `MultiprocExecutor` | NCCL 用于张量通信；multiprocessing Queue 用于控制消息 |
| API server | 主进程 | 进程内（通过 AsyncLLM） |

在 **Ray 模式**下，工作进程是 Ray actor，通过 Ray 的分布式运行时进行通信。

在 **uniproc 模式**下，所有内容在单个进程中运行，无 IPC。

## 2.5 启动时的内存布局

### GPU 显存分配

| 区域 | 用途 | 大小确定方式 |
|------|------|-------------|
| 模型权重 (Model weights) | Transformer 参数（跨 TP 分片） | 模型文件大小 / TP 大小 |
| KV 缓存块 (KV cache blocks) | 所有批处理请求的键值缓存 | `(total_gpu_mem - model_mem) * gpu_memory_utilization` / `block_size` |
| 激活缓冲区 (Activation buffers) | 前向传播期间的中间张量 | 根据 `max_num_batched_tokens` 和隐藏维度计算 |
| CUDA graph 缓冲区 | 常见批大小的捕获图内存 | 为特定批大小桶预分配 |

### CPU 内存

| 区域 | 用途 |
|------|------|
| 分词器 (Tokenizer) | HuggingFace fast tokenizer 模型 |
| 请求状态 (Request state) | 待处理/运行中请求的元数据 |
| KV 缓存块表 (KV cache block tables) | 块 ID 映射（用于 GPU 传输的固定内存） |
| 提示词 token ID (Prompt token IDs) | 分词后的提示词缓冲区 |

### 关键内存计算（`_initialize_kv_caches`，位于 `core.py:232`）：

1. 工作进程通过 `determine_available_memory()` 报告可用显存
2. 总 KV 缓存显存 = `available_memory * gpu_memory_utilization`
3. 块数量 = `total_kv_memory / bytes_per_block`
4. 每个块存储 `block_size` 个 token × `2 (K+V)` × `num_layers` × `head_dim` × `num_kv_heads / tp_size` × `dtype_size`
