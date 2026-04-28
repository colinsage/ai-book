# llama.cpp — 关闭与清理

## 4.1 信号处理

服务器捕获以下信号：

| 信号 | 平台 | 处理器位置 | 行为 |
|--------|----------|-----------------|----------|
| SIGINT | Unix/macOS | server.cpp:302-306 `sigaction()` | 优雅关闭 |
| SIGTERM | Unix/macOS | server.cpp:307 `sigaction()` | 优雅关闭 |
| CTRL_C_EVENT | Windows | server.cpp:309-312 `SetConsoleCtrlHandler()` | 优雅关闭 |

### 信号处理器实现 (server.cpp:27-36)

```c
static void signal_handler(int signal) {
 if (is_terminating.test_and_set()) {
 // Second Ctrl+C: force exit
 fprintf(stderr, "Received second interrupt, terminating immediately.\n");
 exit(1);
 }
 shutdown_handler(signal);
}
```

关键设计：一个 `atomic_flag`（`is_terminating`）确保：
- **第一次 SIGINT/SIGTERM**：通过 `shutdown_handler` 触发优雅关闭
- **第二次 SIGINT/SIGTERM**：立即调用 `exit(1)` — 在优雅关闭挂起时的逃生出口

## 4.2 关闭序列

### 单模型服务器 (server.cpp:294-344)

1. **接收到信号** → 调用 `signal_handler()`
2. **设置终止标志** — `is_terminating.test_and_set()` 防止重复处理
3. **调用 shutdown_handler** — 调用 `ctx_server.terminate()` (server.cpp:296)
4. **`ctx_server.terminate()`** — 调用 `queue_tasks.terminate()` 解除 `start_loop()` 的阻塞
5. **主循环退出** — `start_loop()` 返回 (server.cpp:336)
6. **执行清理** (server.cpp:257-262)：
 - `ctx_http.stop()` — 停止 HTTP 服务器，关闭监听套接字
 - `ctx_server.terminate()` — 等待正在进行的推理任务
 - `llama_backend_free()` — 释放 GGML 后端资源
7. **加入 HTTP 线程** (server.cpp:339-340) — `ctx_http.thread.join()`
8. **加入监控线程** (server.cpp:342-343) — 若为子服务器模式
9. **打印计时统计** (server.cpp:346-353)
10. **退出**，返回码 0

### 路由服务器 (server.cpp:233-248)

1. **接收到信号** → `signal_handler()` → `ctx_http.stop()`
2. **HTTP 服务器停止** — 主线程从 `thread.join()` 解除阻塞 (server.cpp:319-321)
3. **清理** (server.cpp:236-242)：
 - `models_routes->models.unload_all()` — 终止所有子服务器进程
 - `llama_backend_free()`
4. **退出**

## 4.3 资源清理清单

| 资源 | 清理方法 | 位置 |
|----------|---------------|---------|
| HTTP 服务器 | `ctx_http.stop()` | server-http.cpp |
| 推理队列 | `queue_tasks.terminate()` | server-queue.cpp |
| llama_context | `llama_free()` | llama.cpp（通过 server_context 析构函数） |
| llama_model | `llama_model_free()` | llama.cpp（通过 server_context 析构函数） |
| KV 缓存 | 在上下文释放时隐式清理 | llama-context.cpp:~llama_context |
| GGML 后端 | `llama_backend_free()` | ggml-backend.cpp |
| GPU VRAM | `ggml_backend_buffer_free()` | 各后端（cuda、metal 等） |
| 内存映射模型 | `munmap()` / `UnmapViewOfFile()` | llama-mmap.cpp:~llama_mmap |
| 子进程 | `models.unload_all()` | server-models.cpp |
| 线程池 | `ggml_threadpool_free()` | ggml-threading.cpp |
| 采样器链 | `llama_sampler_free()` | llama-sampler.cpp |
| 语法 | `llama_grammar_free()` | llama-grammar.cpp |
| LoRA 适配器 | `llama_adapter_lora_free()` | llama-adapter.cpp |
