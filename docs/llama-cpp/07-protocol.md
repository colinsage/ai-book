# llama.cpp — 网络协议分析

> llama.cpp 的服务器使用标准 HTTP/1.1，请求/响应体为 JSON，流式传输使用 Server-Sent Events (SSE)。没有自定义的二进制应用层协议。本节记录 HTTP 层的模式。

## HTTP 传输

**库：** cpp-httplib（仅头文件，`tools/server/httplib.h`）
**默认端口：** 8080（可通过 `--port` 配置）
**主机：** 127.0.0.1（可通过 `--host` 配置）

## 请求格式

所有推理端点接受 JSON 请求体：

```
POST /v1/chat/completions HTTP/1.1
Content-Type: application/json
Authorization: Bearer *** (if api_keys configured)

{
 "model": "model-name",
 "messages": [{"role": "user", "content": "Hello"}],
 "temperature": 0.8,
 "max_tokens": 256,
 "stream": true
}
```

## 流式响应 — Server-Sent Events (SSE)

当 `stream: true` 时，服务器以 SSE 响应：

```
HTTP/1.1 200 OK
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","choices":[{"delta":{"content":"Hello"}}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","choices":[{"delta":{"content":" world"}}]}

data: [DONE]
```

**关键设计选择：**
- SSE 格式完全遵循 OpenAI API 规范
- 每条 `data:` 行包含一个完整的 JSON 对象
- 流以 `data: [DONE]` 终止
- httplib 使用分块传输编码进行流式传输

## 非流式响应

当 `stream: false` 时，服务器返回单个 JSON 响应：

```
HTTP/1.1 200 OK
Content-Type: application/json

{
 "id": "chatcmpl-123",
 "object": "chat.completion",
 "choices": [{"message": {"role": "assistant", "content": "Hello world"}, "finish_reason": "stop"}],
 "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
}
```

## Router 模式 IPC

在 router 模式下，主服务器通过 localhost 上的 HTTP 将请求代理到子服务器进程。这是标准的 HTTP 代理——没有自定义协议。

**子服务器通信：**
- 每个子服务器运行在独立的端口上（由 router 分配）
- Router 通过 HTTP 将传入的 API 请求转发到相应的子服务器
- 子服务器健康状态通过心跳机制（HTTP 健康检查）监控
