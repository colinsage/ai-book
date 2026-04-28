# llama.cpp — 启动流程

## 2.1 入口点

llama.cpp 有两个主要入口点：

### 服务器入口点

**文件：** `tools/server/server.cpp:74` — `int main(int argc, char ** argv)`

### 命令行入口点

**文件：** `tools/cli/cli.cpp` — `int main(int argc, char ** argv)`

两个入口点遵循相似的初始化模式，均使用共享的 `common_params` 进行参数解析。

## 2.2 服务器初始化序列

`tools/server/server.cpp` 中的服务器启动流程：

1. **解析命令行标志** (server.cpp:82) — `common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SERVER)` 将所有命令行参数解析到 `common_params` 中
2. **验证批次参数** (server.cpp:89) — 确保在嵌入模式下 `n_batch <= n_ubatch`
3. **自动配置并行度** (server.cpp:95-100) — 若 `n_parallel < 0`，则设置 `n_parallel = 4` 且 `kv_unified = true`
4. **初始化后端** (server.cpp:110) — `llama_backend_init()` 初始化 GGML 后端注册表并发现可用的硬件后端
5. **初始化 NUMA** (server.cpp:111) — `llama_numa_init(params.numa)` 配置 NUMA 内存策略
6. **创建服务器上下文** (server.cpp:108) — `server_context ctx_server` — 主要的推理调度器
7. **初始化 HTTP 上下文** (server.cpp:116-120) — `ctx_http.init(params)` 设置 HTTP 服务器（基于 httplib），加载 API 密钥，注册中间件
8. **注册 API 路由** (server.cpp:127-225) — 所有 REST 端点通过 `ctx_http.get()` / `ctx_http.post()` 注册
9. **启动 HTTP 服务器** (server.cpp:265) — `ctx_http.start()` 开始在配置的端口上监听（在模型加载之前，以便 `/health` 可用）
10. **加载模型** (server.cpp:280) — `ctx_server.load_model(params)` 加载 GGUF 权重，创建 `llama_model` 和 `llama_context`
11. **注册信号处理器** (server.cpp:302-313) — `sigaction(SIGINT/SIGTERM)` 注册 `signal_handler` 用于优雅关闭
12. **进入主循环** (server.cpp:336) — `ctx_server.start_loop()` 阻塞主线程，处理队列中的推理任务

### 路由模式

若 `params.model.path` 为空，服务器以**路由模式**启动 (server.cpp:130-170)。在路由模式下：
- 主进程中不加载模型
- 一个 `server_models_routes` 对象管理子服务器进程
- 所有推理端点被代理到子服务器
- 额外注册 `/models/load` 和 `/models/unload` 路由用于动态模型管理

## 2.3 命令行初始化序列

1. **解析命令行标志** — `common_params_parse()` 解析到 `common_params`
2. **初始化后端** — `llama_backend_init()`
3. **加载模型** — 通过 `server_context::load_model()`
4. **注册信号处理器** — SIGINT 设置 `g_is_interrupted` 标志用于生成中途取消
5. **进入交互循环** — 读取用户输入，格式化聊天消息，调用 `generate_completion()`

## 2.4 线程模型（服务器）

| 线程 | 创建位置 | 角色 |
|--------|-----------|------|
| main | OS | CLI 解析、初始化，然后阻塞在 `start_loop()` |
| http-thread | `ctx_http.start()` | HTTP 请求监听器（httplib 服务器） |
| io-worker-N | httplib 内部 | 处理 HTTP 请求，分发到路由处理器 |
| compute | ggml 线程池 | CPU 推理工作线程（通过 `n_threads` 配置） |
| monitor-thread | `server_models::setup_child_server()`（条件性） | 作为子服务器运行时向路由服务器发送心跳 |

### GPU 线程使用

当 GPU 后端（CUDA、Metal、Vulkan）处于活动状态时，计算图的执行通过后端的 `graph_compute` 函数指针分发到 GPU。提交计算图的 CPU 线程可能会在 `synchronize()` 上阻塞，直到 GPU 工作完成。多个槽位可以批量一起处理以提高 GPU 利用率。

## 2.5 进程模型（路由模式）

在路由模式下，服务器通过 `server_models` 生成**子服务器进程**：

- **生成**：子进程通过 `fork/exec` 或等效方式启动，每个加载特定模型
- **IPC**：子服务器通过 HTTP 与路由通信（同一机器，不同端口）
- **生命周期**：路由通过监控线程心跳机制跟踪子服务器健康状态
- **扩展**：多个子服务器可以并发运行，每个在不同端口上

## 2.6 启动时的内存布局

### 模型权重（最大分配）

- **方式**：通过 `llama_mmap` (src/llama-mmap.cpp) 进行内存映射文件（`mmap`）
- GGUF 文件被映射到虚拟地址空间；张量直接从映射中访问
- 在 GPU 后端上，张量通过 `ggml_backend_buffer` 分配复制到 VRAM

### KV 缓存

- **分配**：`ggml_backend_alloc_ctx_tensors()` 为 KV 缓存张量预留 GPU 或 CPU 内存
- 大小取决于 `n_ctx`（上下文窗口）、`n_layer`、`n_embd_head_k`、`n_embd_head_v` 和 `n_seq_max`
- 通常是模型权重之后第二大内存消耗

### 计算缓冲区

- **用途**：计算图执行期间的中间张量（激活值）
- **分配**：在 `sched_reserve()` 期间通过 `ggml_backend_alloc_ctx_tensors()` 分配
- 大小取决于批次大小和模型架构

### CPU 竞技场

- GGML 使用自定义分配器（`ggml_alloc`）来管理张量元数据（而非数据）
- `ggml_context` 对象从池中分配张量结构体，避免每个张量调用 `malloc`

### GPU VRAM 布局

在 CUDA/Metal 后端上：
1. 模型权重 → VRAM（通过 `ggml_backend_buffer`）
2. KV 缓存 → VRAM（通过 `ggml_backend_buffer`）
3. 计算临时区 → VRAM（临时，每次计算图执行时重用）
