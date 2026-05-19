# llama.cpp — Network Protocol Analysis

> llama.cpp's server uses standard HTTP/1.1 with JSON request/response bodies and Server-Sent Events (SSE) for streaming. There is no custom binary application-layer protocol. This section documents the HTTP-level patterns.

## HTTP Transport

**Library:** cpp-httplib (header-only, `tools/server/httplib.h`)
**Default Port:** 8080 (configurable via `--port`)
**Host:** 127.0.0.1 (configurable via `--host`)

## Request Format

All inference endpoints accept JSON request bodies:

```
POST /v1/chat/completions HTTP/1.1
Content-Type: application/json
Authorization: Bearer <api_key>   (if api_keys configured)

{
  "model": "model-name",
  "messages": [{"role": "user", "content": "Hello"}],
  "temperature": 0.8,
  "max_tokens": 256,
  "stream": true
}
```

## Streaming Response — Server-Sent Events (SSE)

When `stream: true`, the server responds with SSE:

```
HTTP/1.1 200 OK
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","choices":[{"delta":{"content":"Hello"}}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","choices":[{"delta":{"content":" world"}}]}

data: [DONE]
```

**Key Design Choices:**
- SSE format follows the OpenAI API specification exactly
- Each `data:` line contains a complete JSON object
- Stream terminates with `data: [DONE]`
- Chunked transfer encoding is used by httplib for streaming

## Non-Streaming Response

When `stream: false`, the server returns a single JSON response:

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

## Router Mode IPC

In router mode, the main server proxies requests to child server processes via HTTP on localhost. This is standard HTTP proxying — no custom protocol.

**Child Server Communication:**
- Each child server runs on a distinct port (assigned by the router)
- The router forwards incoming API requests to the appropriate child via HTTP
- Child server health is monitored via a heartbeat mechanism (HTTP health check)
