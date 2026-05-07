# vLLM — 网络协议分析

## ZMQ IPC 协议 (Engine Core ↔ API Server)

在多进程模式下，API 服务器与引擎核心通过 ZMQ 套接字使用自定义二进制协议进行通信。

### 消息格式

消息使用 **msgpack**（通过 `msgspec`）进行序列化以获得高性能：

```
EngineCoreRequest → msgpack bytes → ZMQ SEND
ZMQ RECV → msgpack bytes → EngineCoreOutputs
```

### 输入路径 (API Server → Engine Core)

**套接字类型：** ZMQ PUSH（异步）

**消息类型**（`EngineCoreRequestType`）：
- `ADD_REQUEST` — 新推理请求
- `ABORT_REQUEST` — 取消请求
- `UTILITY` — 管理操作（profile、sleep、wake、LoRA 等）

**流程：**
1. `AsyncLLM.add_request()` 通过 `MsgpackEncoder` 序列化 `EngineCoreRequest`
2. 通过 ZMQ PUSH 套接字发送到引擎核心的输入队列
3. 引擎核心在步骤循环中从 PULL 套接字读取

### 输出路径 (Engine Core → API Server)

**套接字类型：** ZMQ PUB/SUB（异步）— 每个引擎核心客户端一个输出套接字

**消息格式：**
```
[msgpack bytes] EngineCoreOutputs
 ├── dict[int, list[EngineCoreOutput]] # 以 client_index 为键
 └── 包含：采样令牌、结束原因、对数概率
```

**流程：**
1. 引擎核心在每次 `step()` 后发布输出
2. `AsyncLLM.output_handler()` 从 ZMQ SUB 套接字读取
3. 分发到 `OutputProcessor` 进行反标记化和组装

### 张量 IPC (Tensor IPC)

对于大型张量传输（模型权重、KV 缓存），vLLM 使用共享内存：

**处理器：** `vllm/v1/engine/tensor_ipc.py`（`TensorIpcSender`）

**机制：** 通过 `multiprocessing.reductions` 实现 PyTorch 张量 IPC — 跨进程共享张量存储而无需拷贝。

---

## HTTP/ASGI 协议 (Client ↔ API Server)

标准 HTTP 配合 JSON 请求/响应体。流式传输使用 **Server-Sent Events (SSE)**：

```
data: {"id":"...","object":"chat.completion.chunk","choices":[...]}\n\n
```

### 认证中间件 (Authentication Middleware)

**实现：** `vllm/entrypoints/openai/server_utils.py:AuthenticationMiddleware`

纯 ASGI 中间件，检查 `Authorization: Bearer ***` 请求头：

1. 从 `Authorization` 请求头中提取 bearer token
2. 对 token 进行 SHA-256 哈希
3. 使用 `secrets.compare_digest()`（常量时间比较）与预哈希的 API 密钥进行比对
4. 跳过认证的情况：OPTIONS 请求、不以 `/v1` 开头的路径（如 `/health`）

**配置：**
- `--api-key` CLI 标志或 `VLLM_API_KEY` 环境变量
- 支持多个密钥：`--api-key key1 --api-key key2`

---

## NCCL 协议 (GPU 间通信)

用于张量并行和流水线并行，vLLM 使用 NVIDIA NCCL：

- **All-reduce** 用于张量并行线性层
- **All-gather** 用于专家并行
- **Send/recv** 用于流水线并行
- **Broadcast** 用于分离式服务中的 KV 缓存协调

---

## KV Connector 协议 (Disaggregated Serving)

**处理器：** `vllm/distributed/kv_transfer/`

实现预填充实例与解码实例之间的 KV 缓存共享：

1. 预填充实例计算 KV 缓存
2. KV Connector 序列化并将 KV 块传输到解码实例
3. 解码实例反序列化并加载到本地 KV 缓存
4. 支持 RDMA、TCP 和共享内存传输方式

---

## MCP 工具协议 (MCP Tool Protocol)

**处理器：** `vllm/entrypoints/mcp/tool_server.py`

vLLM 可以作为 MCP (Model Context Protocol) 客户端：

1. 通过 SSE 传输方式连接到外部 MCP 工具服务器
2. 通过 `session.list_tools()` 发现可用工具
3. 将 MCP 工具模式转换为 Harmony 格式，用于 responses API
4. 执行工具调用并将结果反馈到生成过程

这使得 `/v1/responses` 端点能够在生成过程中使用外部工具。
