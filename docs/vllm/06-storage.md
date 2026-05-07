# vLLM — 存储文件分析

## 模型权重文件

### SafeTensors（主要格式）

**位置：** 从 HuggingFace Hub 或由 `--model` 指定的本地路径加载

**处理器：** `vllm/model_executor/model_loader/default_loader.py`

**格式：** HuggingFace SafeTensors — 支持内存映射（mmap）、零拷贝（zero-copy）的格式，带有 JSON 元数据头。

**加载过程：**
1. 下载/定位模型文件（分片 SafeTensors）
2. 使用 `mmap` 对每个分片进行内存映射
3. 对于张量并行（Tensor Parallelism）：仅加载每个 TP rank 对应的分片
4. 对于量化模型：应用量化感知加载（AWQ、GPTQ、FP8 等）

### GGUF 格式

**处理器：** `vllm/model_executor/model_loader/gguf_loader.py`

**格式：** GGUF 二进制格式，魔数为 `0x47475546`：
```
[4 bytes] Magic: "GGUF"
[4 bytes] Version: uint32
[8 bytes] tensor_count: uint64
[8 bytes] kv_count: uint64
[N bytes] Key-value metadata pairs
[N bytes] Tensor descriptors
[aligned] Raw tensor data (32-byte aligned)
```

**设计意图：** 自描述格式，将权重与超参数打包在一起，无需单独的配置文件即可实现内存映射访问。

### BitsAndBytes（NF4）格式

**处理器：** `vllm/model_executor/model_loader/bitsandbytes_loader.py`

支持带有嵌套量化元数据的 4-bit 量化模型。

### Tensorizer 格式

**处理器：** `vllm/model_executor/model_loader/tensorizer_loader.py`

高性能序列化张量格式，用于从 S3 或本地存储快速加载。

### RunAI Streamer

**处理器：** `vllm/model_executor/model_loader/runai_streamer_loader.py`

流式模型加载器，通过渐进式权重加载实现更快的启动速度。

### 分片状态（Sharded State）

**处理器：** `vllm/model_executor/model_loader/sharded_state_loader.py`

支持加载/保存分布式模型状态，用于检查点（checkpointing）和恢复运行（resumption）。

---

## KV 缓存持久化（KV Cache Persistence）

vLLM 默认**不**将 KV 缓存持久化到磁盘。KV 缓存完全驻留在 GPU VRAM 中（或 CPU 后端中的 CPU RAM），并在关闭时释放。

**KV 缓存事件**（`vllm/distributed/kv_events/`）：基于事件的系统，发出分配/释放事件。外部系统可以利用此机制通过 **KV Connector** 框架（`vllm/distributed/kv_transfer/`）实现 KV 缓存持久化或跨实例共享。

---

## 配置文件

### VllmConfig（`vllm/config/`）

由 CLI 参数、环境变量和模型元数据组装而成的综合配置对象：

| 配置子类 | 文件 | 用途 |
|----------------|------|---------|
| `ModelConfig` | `config/model.py` | 模型路径、最大长度、数据类型（dtype）、分词器（tokenizer） |
| `CacheConfig` | `config/cache.py` | 块大小（block size）、交换空间（swap space）、KV 缓存数据类型 |
| `SchedulerConfig` | `config/scheduler.py` | 最大序列数、最大 token 数、调度策略 |
| `ParallelConfig` | `config/parallel.py` | TP/PP/DP 大小、分布式后端 |
| `SpeculativeConfig` | `config/speculative.py` | 推测解码（Speculative Decoding）设置 |
| `LoRAConfig` | `config/lora.py` | LoRA 适配器设置 |
| `ObservabilityConfig` | `config/observability.py` | 追踪（tracing）、指标（metrics）、KV 事件 |
| `StructuredOutputsConfig` | `config/structured_outputs.py` | 语法后端选择（grammar backend selection） |

### 环境变量（`vllm/envs.py`）

200+ 个环境变量控制 vLLM 的各个方面。关键示例：

| 变量 | 默认值 | 用途 |
|----------|---------|---------|
| `VLLM_API_KEY` | None | API 认证令牌 |
| `VLLM_CACHE_ROOT` | `~/.cache/vllm` | 下载模型的缓存目录 |
| `VLLM_LOG_STATS_INTERVAL` | 10.0 | 统计日志间隔秒数 |
| `VLLM_ENGINE_ITERATION_TIMEOUT_S` | 60 | 每个引擎步骤的最大秒数 |
| `VLLM_ENGINE_READY_TIMEOUT_S` | 600 | 等待引擎启动的最大秒数 |
| `VLLM_PLUGINS` | None | 允许的插件逗号分隔列表 |
| `VLLM_USE_PRECOMPILED` | False | 使用预编译的 C++ 扩展 |

---

## LoRA 适配器存储（LoRA Adapter Storage）

**处理器：** `vllm/lora/model_manager.py`、`vllm/lora/worker_manager.py`

LoRA 适配器按需从磁盘加载：
- 以 SafeTensors 文件存储，遵循特定的键名命名约定
- 通过 `PEFT` 辅助工具（`vllm/lora/peft_helper.py`）加载
- 加载后权重固定（pinned）在 GPU 内存中
- 支持通过 API 端点动态加载/卸载
