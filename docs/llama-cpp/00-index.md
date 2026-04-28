# llama.cpp — 源码分析

**生成日期：** 2026-04-28
**仓库：** /home/colin/amuse/cpp/llama.cpp
**提交：** bd28a2e7200e7e8a77bd23a82dd7afb2ccca37af

## 概要

llama.cpp 是一个用于大语言模型的 C/C++ 推理引擎，实现了 100+ 种模型架构，配备了自定义张量计算库（GGML）和 15+ 种硬件后端加速器。它采用延迟计算图与虚函数表分派后端、可组合的采样器链用于令牌选择，以及 GGUF 自描述二进制格式用于内存映射的模型权重。该项目还包含一个与 OpenAI 兼容的 HTTP 服务器，支持多槽位并发推理。

## 文档目录

| 文件 | 内容 |
|------|---------|
| [01-overview.md](01-overview.md) | 项目分类、技术栈、目录映射、模块图 |
| [02-startup.md](02-startup.md) | 入口点、初始化序列、线程/进程模型、内存布局 |
| [03-api.md](03-api.md) | REST API 端点（兼容 OpenAI/Anthropic）、C 库 API、序列图 |
| [04-shutdown.md](04-shutdown.md) | 信号处理、优雅关机序列、资源清理清单 |
| [05-data-structures.md](05-data-structures.md) | 核心结构体 — ggml_tensor、ggml_cgraph、llama_model、llama_hparams、llama_batch、llama_layer、ggml_backend_i、llama_sampler_i |
| [06-storage.md](06-storage.md) | GGUF 二进制格式规范、元数据键、加载过程、转换脚本 |
| [07-protocol.md](07-protocol.md) | HTTP/SSE 传输模式、流式响应格式 |
| [08-crosscutting.md](08-crosscutting.md) | API 密钥认证、Prometheus 指标、基于槽位的并发控制 |
| [09-extensions.md](09-extensions.md) | GGML 后端虚函数表系统、采样器链、模型架构扩展 |

## 建议优先阅读的关键文件

| 文件 | 原因 |
|------|--------|
| `include/llama.h` | 公共 API — 调用方使用的所有函数（1565 行） |
| `ggml/include/ggml.h` | GGML 张量类型、操作和核心抽象 |
| `src/llama-model.cpp` | 所有架构的模型加载与前向传播实现 |
| `tools/server/server.cpp` | 服务器入口点、路由注册、启动/关机 |
| `ggml/src/ggml-backend-impl.h` | 后端虚函数表定义 — 硬件扩展点 |
| `src/llama-sampler.cpp` | 所有采样算法的实现 |
| `ggml/include/gguf.h` | GGUF 格式规范与 API |
| `src/llama-kv-cells.h` | KV 缓存单元跟踪 — 理解上下文管理的核心 |
