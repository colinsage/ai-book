# vLLM — 核心数据结构

## Request (`vllm/v1/request.py`)

**用途：** 表示引擎中单个推理请求在其整个生命周期中的抽象。

**关键字段：**

| 字段 | 类型 | 用途 |
|-------|------|---------|
| `request_id` | `str` | 请求的唯一标识符 (Unique identifier) |
| `prompt_token_ids` | `list[int]` | 分词后的提示 (Tokenized prompt) |
| `sampling_params` | `SamplingParams` | 生成参数（温度、top_p、max_tokens 等） |
| `pooling_params` | `PoolingParams` | 嵌入/池化任务的参数 |
| `arrival_time` | `float` | 请求到达的时间戳 |
| `lora_request` | `LoRARequest` | 要应用的 LoRA 适配器（如果有） |
| `mm_features` | `list[MultiModalFeatureSpec]` | 多模态特征（图像、音频、视频） |
| `priority` | `int` | 调度优先级（0 = 默认） |
| `structured_output_request` | `StructuredOutputRequest` | 结构化生成的语法约束 |
| `num_computed_tokens` | `int` | 已计算 KV cache 的 token 数量 |
| `num_output_placeholders` | `int` | 延迟输出的占位符计数 |
| `resumable` | `bool` | 该请求是否支持流式续传 |

**关键方法：**

| 方法 | 复杂度 | 说明 |
|--------|-----------|-------|
| `num_tokens_with_spec` | O(1) | 返回包含推测 token (speculative tokens) 在内的总 token 数 |
| `get_prefix_cache_block_hash()` | O(blocks) | 计算用于前缀缓存 (prefix caching) 的块哈希 |
| `is_prefilling()` | O(1) | 如果仍处于预填充阶段 (prefill phase) 则返回 True |

---

## SamplingParams (`vllm/sampling_params.py`)

**用途：** 控制文本生成行为——温度、top-p、top-k、频率/重复惩罚、结构化输出约束等。

**关键字段：**

| 字段 | 类型 | 用途 |
|-------|------|---------|
| `n` | `int` | 每个提示的补全数量 (Number of completions per prompt) |
| `temperature` | `float` | 采样温度（0 = 贪心解码） |
| `top_p` | `float` | 核采样阈值 (Nucleus sampling threshold) |
| `top_k` | `int` | Top-K 采样 |
| `max_tokens` | `int` | 最大输出 token 数 |
| `stop` | `list[str]` | 停止序列 (Stop sequences) |
| `stop_token_ids` | `list[int]` | 停止 token ID |
| `frequency_penalty` | `float` | 频率惩罚 (Frequency penalty) |
| `presence_penalty` | `float` | 存在惩罚 (Presence penalty) |
| `repetition_penalty` | `float` | 重复惩罚 (Repetition penalty) |
| `structured_outputs` | `StructuredOutputsParams` | JSON schema、regex、grammar 或 choice 约束 |
| `logprobs` | `int` | 返回的 logprobs 数量 |
| `seed` | `int` | 用于可复现性的随机种子 |

**复杂逻辑：**
- `StructuredOutputsParams` 验证互斥性——json/regex/choice/grammar/json_object/structural_tag 中只能设置一个
- `_backend` 由 `Processor._validate_structured_output` 在请求处理期间自动设置
- `SamplingType` 枚举（GREEDY、RANDOM、RANDOM_SEED）决定采样内核 (sampling kernel) 中的调度路径

---

## EngineCoreRequest (`vllm/v1/engine/__init__.py`)

**用途：** 从 API 服务器通过 ZMQ/msgpack 发送到引擎核心进程的序列化请求格式。

轻量级数据类 (dataclass)，仅包含引擎核心所需的字段——没有 Python 对象引用，从而支持跨进程序列化 (cross-process serialization)。

---

## KVCacheBlock (`vllm/v1/core/kv_cache_utils.py`)

**用途：** 表示 KV cache 内存中的单个块。由 KV cache 管理器用于跟踪分配和引用计数。

**关键字段：**

| 字段 | 类型 | 用途 |
|-------|------|---------|
| `block_id` | `int` | KV cache 张量中的物理块索引 |
| `ref_count` | `int` | 共享此块的请求数量（用于前缀缓存） |
| `block_hash` | `BlockHash` | 此块中 token 内容的哈希（用于前缀缓存查找） |

**关键概念：**
- **块大小 (Block size)**：每个块的 token 数量（可配置，通常为 16）
- **前缀缓存 (Prefix caching)**：具有相同哈希的块可以在请求之间共享
- **引用计数 (Reference counting)**：共享块仅在 ref_count 降至 0 时才被释放
- **BlockHashWithGroupId**：将块哈希 + KV cache 组 ID 打包为字节，用于高效查找

---

## KVCacheBlocks (`vllm/v1/core/kv_cache_manager.py`)

**用途：** KV cache 管理器的分配结果——调度器 (Scheduler) 与 KV cache 内部之间的接口。

**结构：**
- `blocks: tuple[Sequence[KVCacheBlock], ...]` — 通过 `[kv_cache_group][block_index]` 索引
- 支持不同的 KV cache 组（例如，全注意力 vs. Mamba 状态）
- `get_block_ids()` 转换为整数块 ID 元组，用于 GPU 传输

---

## BlockTable (`vllm/v1/worker/block_table.py`)

**用途：** GPU 端从请求槽位到 KV cache 块 ID 的映射。由注意力内核 (attention kernels) 用于定位 KV cache 数据。

**关键字段：**

| 字段 | 类型 | 用途 |
|-------|------|---------|
| `block_size` | `int` | 每块的 token 数 |
| `max_num_reqs` | `int` | 最大并发请求数 |
| `max_num_blocks_per_req` | `int` | 每个请求的最大块数 |
| `device` | `torch.device` | 表所在的 GPU 设备 |

**复杂逻辑：**
- 支持**混合块 (hybrid blocks)**，其中分配块大小与内核块大小不同（拆分/合并）
- 维护 CPU（固定内存）和 GPU 副本，用于高效的批量更新
- `CpuGpuBuffer` 模式：在 CPU 上更新，异步复制到 GPU

---

## SchedulerOutput (`vllm/v1/core/sched/output.py`)

**用途：** 调度器的输出，包含模型运行器 (model runner) 执行一步所需的所有信息。

**关键字段：**

| 字段 | 类型 | 用途 |
|-------|------|---------|
| `scheduled_new_reqs` | `list[Request]` | 新调度的请求 |
| `scheduled_resumed_reqs` | `list[Request]` | 恢复的（之前被抢占的）请求 |
| `scheduled_running_reqs` | `list[Request]` | 有新 token 的运行中请求 |
| `num_scheduled_tokens` | `dict[str, int]` | 每个请求调度的 token 数 |
| `total_num_scheduled_tokens` | `int` | 此批次的总 token 数 |
| `preempted_reqs` | `list[Request]` | 被抢占的请求 |
| `scheduled_spec_decode_tokens` | `dict[str, list[int]]` | 每个请求的推测解码 token |

---

## ModelRunnerOutput (`vllm/v1/worker/gpu_model_runner.py`)

**用途：** GPU 模型运行器的输出，包含采样的 token 和元数据。

包含每个请求的：采样的 token ID、logprobs，以及请求是否已完成。

---

## IntermediateTensors (`vllm/sequence.py`)

**用途：** 用于流水线并行 (pipeline parallelism)——在流水线阶段之间传递隐藏状态 (hidden states) 和残差 (residuals)。

**关键字段：**

| 字段 | 类型 | 用途 |
|-------|------|---------|
| `tensors` | `dict[str, torch.Tensor]` | 命名张量（隐藏状态、残差） |
| `kv_connector_output` | `KVConnectorOutput` | 用于分离式服务 (disaggregated serving) 的 KV 传输输出 |

**设计说明：** 手动定义 `__init__`，以便 `torch.compile` (Dynamo) 能够追踪构造函数的来源——dataclass 自动生成的 `__init__` 会丢失此信息。

---

## EngineCoreEvent / EngineCoreEventType (`vllm/v1/engine/__init__.py`)

**用途：** 用于 KV cache 可观测性 (observability) 的事件系统——在块被分配、释放或驱逐时发出。

使外部系统能够跟踪 KV cache 生命周期，用于分离式服务协调 (disaggregated serving coordination)。

---

## KVCacheConfig / KVCacheSpec (`vllm/v1/kv_cache_interface.py`)

**用途：** 描述每种模型架构的 KV cache 结构和需求。

**关键类型：**

| 类型 | 用途 |
|------|---------|
| `FullAttentionSpec` | 标准全注意力 KV cache |
| `MLAAttentionSpec` | 多头潜在注意力 (Multi-head Latent Attention, DeepSeek) |
| `MambaSpec` | Mamba SSM 状态缓存 |
| `ChunkedLocalAttentionSpec` | 滑动窗口 / 分块注意力 |
| `SlidingWindowSpec` | 滑动窗口注意力 (Sliding window attention) |
| `KVCacheGroupSpec` | 共享同一池的 KV cache 规格组 |
| `KVCacheTensor` | 物理张量分配描述符 |
| `KVQuantMode` | 量化模式（NONE、FP8、INT8、NVFP4） |

**设计意图：** 模型可以具有异构的 KV cache 需求（例如，Llama 只有全注意力；Jamba 同时有注意力 + Mamba）。配置系统允许具有不同规格的不同缓存池共存。
