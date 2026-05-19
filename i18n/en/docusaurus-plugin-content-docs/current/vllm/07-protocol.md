# vLLM — Network Protocol Analysis

## ZMQ IPC Protocol (Engine Core ↔ API Server)

In multiprocessing mode, the API server and engine core communicate via ZMQ sockets using a custom binary protocol.

### Message Format

Messages are serialized using **msgpack** (via `msgspec`) for high performance:

```
EngineCoreRequest → msgpack bytes → ZMQ SEND
ZMQ RECV → msgpack bytes → EngineCoreOutputs
```

### Input Path (API Server → Engine Core)

**Socket type:** ZMQ PUSH (async)

**Message types** (`EngineCoreRequestType`):
- `ADD_REQUEST` — New inference request
- `ABORT_REQUEST` — Cancel request(s)
- `UTILITY` — Administrative operations (profile, sleep, wake, LoRA, etc.)

**Flow:**
1. `AsyncLLM.add_request()` serializes `EngineCoreRequest` via `MsgpackEncoder`
2. Sends via ZMQ PUSH socket to engine core's input queue
3. Engine core reads from PULL socket in step loop

### Output Path (Engine Core → API Server)

**Socket type:** ZMQ PUB/SUB (async) — one output socket per engine core client

**Message format:**
```
[msgpack bytes] EngineCoreOutputs
  ├── dict[int, list[EngineCoreOutput]]  # keyed by client_index
  └── contains: sampled tokens, finish reasons, logprobs
```

**Flow:**
1. Engine core publishes outputs after each `step()`
2. `AsyncLLM.output_handler()` reads from ZMQ SUB socket
3. Dispatches to `OutputProcessor` for detokenization and assembly

### Tensor IPC

For large tensor transfers (model weights, KV cache), vLLM uses shared memory:

**Handler:** `vllm/v1/engine/tensor_ipc.py` (`TensorIpcSender`)

**Mechanism:** PyTorch tensor IPC via `multiprocessing.reductions` — shares tensor storage across processes without copying.

---

## HTTP/ASGI Protocol (Client ↔ API Server)

Standard HTTP with JSON request/response bodies. Streaming uses **Server-Sent Events (SSE)**:

```
data: {"id":"...","object":"chat.completion.chunk","choices":[...]}\n\n
```

### Authentication Middleware

**Implementation:** `vllm/entrypoints/openai/server_utils.py:AuthenticationMiddleware`

Pure ASGI middleware that checks `Authorization: Bearer <token>` headers:

1. Extract bearer token from `Authorization` header
2. SHA-256 hash the token
3. Compare against pre-hashed API key(s) using `secrets.compare_digest()` (constant-time comparison)
4. Skip authentication for: OPTIONS requests, paths not starting with `/v1` (e.g., `/health`)

**Configuration:**
- `--api-key` CLI flag or `VLLM_API_KEY` environment variable
- Multiple keys supported: `--api-key key1 --api-key key2`

---

## NCCL Protocol (Inter-GPU Communication)

For tensor parallelism and pipeline parallelism, vLLM uses NVIDIA NCCL:

- **All-reduce** for tensor parallel linear layers
- **All-gather** for expert parallelism
- **Send/recv** for pipeline parallelism
- **Broadcast** for KV cache coordination in disaggregated serving

---

## KV Connector Protocol (Disaggregated Serving)

**Handler:** `vllm/distributed/kv_transfer/`

Enables KV cache sharing between prefill and decode instances:

1. Prefill instance computes KV cache
2. KV Connector serializes and transfers KV blocks to decode instance
3. Decode instance deserializes and loads into local KV cache
4. Supports RDMA, TCP, and shared memory transports

---

## MCP Tool Protocol

**Handler:** `vllm/entrypoints/mcp/tool_server.py`

vLLM can act as an MCP (Model Context Protocol) client:

1. Connects to external MCP tool server via SSE transport
2. Discovers available tools via `session.list_tools()`
3. Converts MCP tool schemas to Harmony format for the responses API
4. Executes tool calls and feeds results back into generation

This enables the `/v1/responses` endpoint to use external tools during generation.
