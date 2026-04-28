# llama.cpp — 核心数据结构

## `ggml_tensor` (ggml.h:660)

**用途：** 基础的 n 维张量类型。每个模型权重、激活值和中间计算结果都表示为一个 `ggml_tensor`。张量构成惰性计算 DAG — 构建操作（例如 `ggml_mul_mat`）会创建带有 `op` 字段的张量节点；实际计算在 `ggml_graph_compute()` 执行计算图时发生。

**字段：**

| Field | Type | Purpose |
|-------|------|---------|
| `type` | enum ggml_type | 数据类型（F32, F16, BF16, Q4_0, Q4_K, Q8_0, IQ 系列等 — 42 种类型） |
| `buffer` | ggml_backend_buffer* | 拥有该张量内存的后端缓冲区（CPU, CUDA, Metal 等） |
| `ne[4]` | int64_t | 形状：每个维度的元素数量（最多 4 维） |
| `nb[4]` | size_t | 每个维度的字节步幅。nb[0] = type_size，nb[i] = nb[i-1]*ne[i-1] |
| `op` | enum ggml_op | 计算操作（叶子张量为 NONE，MUL_MAT, ADD, RMS_NORM 等） |
| `op_params[16]` | int32_t | 操作特定参数（例如 SOFT_MAX 的轴，MUL_MAT 参数） |
| `flags` | int32_t | 张量标志（例如 DONT_MMAP, NOT_PERSISTENT） |
| `src[GGML_MAX_SRC]` | ggml_tensor** | 计算图节点的输入张量（最多 10 个源） |
| `view_src` | ggml_tensor* | 视图的源张量（共享底层数据） |
| `view_offs` | size_t | 到 view_src 数据的字节偏移量 |
| `data` | void* | 原始数据指针（CPU 或 GPU 地址） |
| `name` | char[64] | 人类可读的张量名称（用于调试） |
| `extra` | void* | 后端特定的额外数据（例如 CUDA 张量附加信息） |

**关键函数：**

| Function | Complexity | Notes |
|----------|-----------|-------|
| `ggml_new_tensor()` | O(1) | 从 ggml_context 竞技场分配张量元数据 |
| `ggml_mul_mat()` | O(1) 构建 / O(mnk) 计算 | 创建惰性矩阵乘法节点 |
| `ggml_graph_compute()` | O(nodes) | 对 DAG 进行拓扑排序，分派到后端 |
| `ggml_view_tensor()` | O(1) | 创建共享相同数据缓冲区的视图 |

**复杂逻辑：** `ggml_graph_compute()` 对 DAG 进行拓扑排序，然后通过 `ggml_backend_sched` 将每个节点分派到适当的后端（CPU, CUDA, Metal）。调度器根据输入张量所在位置以及哪些后端支持该操作来决定每个操作应由哪个后端处理。跨后端拷贝会自动插入。

---

## `ggml_cgraph` (ggml-impl.h:329)

**用途：** 计算图 — 张量操作的有向无环图。在前向传播构建期间创建，由后端调度器执行。

**字段：**

| Field | Type | Purpose |
|-------|------|---------|
| `size` | int | 最大节点/叶子数 |
| `n_nodes` | int | 活跃计算节点数 |
| `n_leafs` | int | 叶子（常量）张量数 |
| `nodes` | ggml_tensor** | 计算节点数组（产生输出的操作） |
| `grads` | ggml_tensor** | 梯度张量（仅训练时使用） |
| `grad_accs` | ggml_tensor** | 梯度累加器（仅训练时使用） |
| `leafs` | ggml_tensor** | 叶子张量数组（模型权重、常量） |
| `use_counts` | int32_t* | 每个张量的引用计数（用于内存复用） |
| `visited_hash_set` | ggml_hash_set | 图构建期间用于环检测的哈希集合 |
| `order` | enum ggml_cgraph_eval_order | 求值顺序（从左到右或拓扑顺序） |
| `uid` | uint64_t | 用于图匹配的可选标识符 |

**设计：** 计算图是惰性构建的 — 每个 ggml 操作（例如 `ggml_mul_mat(ctx, a, b)`）会添加一个新的张量节点并注册其源节点。在 `ggml_graph_compute()` 期间，节点按拓扑顺序求值。`visited_hash_set` 防止重复张量插入。

---

## `ggml_backend_i` (ggml-backend-impl.h:105)

**用途：** 硬件后端实现的虚表（vtable）。这是向 GGML 添加新计算硬件的核心扩展点。每个后端（CPU, CUDA, Metal, Vulkan 等）都实现此接口。

**函数指针：**

| Function | Required | Purpose |
|----------|----------|---------|
| `get_name` | Yes | 返回后端名称字符串 |
| `free` | Yes | 释放后端资源 |
| `set_tensor_async` | No | 异步写入张量数据 |
| `get_tensor_async` | No | 异步读取张量数据 |
| `cpy_tensor_async` | No | 异步跨后端张量拷贝 |
| `synchronize` | No | 等待所有挂起的异步操作完成 |
| `graph_plan_create` | No | 创建可复用的图执行计划 |
| `graph_plan_compute` | No | 使用预构建计划执行图 |
| `graph_compute` | Yes | 执行计算图（如支持则为异步） |
| `event_record` / `event_wait` | No | 后端间同步原语 |
| `graph_optimize` | No | 后端特定的图优化 |

**相关接口：**

- `ggml_backend_buffer_type_i` (ggml-backend-impl.h:17) — 缓冲区类型虚表（分配、对齐、is_host）
- `ggml_backend_buffer_i` (ggml-backend-impl.h:41) — 缓冲区虚表（free, get_base, set_tensor, get_tensor）

---

## `llama_model` (llama-model.h:512)

**用途：** 已加载的模型状态 — 保存从 GGUF 文件加载的所有模型权重、超参数、词表和元数据。

**关键字段：**

| Field | Type | Purpose |
|-------|------|---------|
| `type` | llm_type | 模型类型（例如 LLM_TYPE_8B, LLM_TYPE_70B） |
| `arch` | llm_arch | 架构枚举（LLAMA, GPT2, FALCON, MISTRAL, PHI, GEMMA 等 — 100+ 种类型） |
| `hparams` | llama_hparams | 模型超参数（维度、层数、头数等） |
| `vocab` | llama_vocab | 分词器词表 |
| `tok_embd` | ggml_tensor* | 词元嵌入权重矩阵 |
| `output_norm` | ggml_tensor* | 最终层归一化权重 |
| `output` | ggml_tensor* | 输出投影（lm_head）权重 |
| `layers` | vector\<llama_layer\> | 每层权重（注意力 + FFN） |
| `devices` | vector\<llama_device\> | 此模型使用的设备列表 |
| `gguf_kv` | unordered_map | 原始 GGUF 元数据键值对 |
| `loras` | unordered_set | 活跃的 LoRA 适配器 |

**关键方法：**

| Method | Purpose |
|--------|---------|
| `load_arch()` | 从 GGUF 元数据确定模型架构 |
| `load_hparams()` | 从 GGUF 解析超参数 |
| `load_vocab()` | 加载分词器词表 |
| `load_tensors()` | 将所有张量权重加载到后端缓冲区 |

---

## `llama_hparams` (llama-hparams.h:36)

**用途：** 从 GGUF 元数据提取的模型超参数。这些定义了模型的架构，加载后为只读。

**关键字段：**

| Field | Type | Purpose |
|-------|------|---------|
| `n_embd` | uint32_t | 嵌入维度 |
| `n_layer` | uint32_t | Transformer 层数 |
| `n_expert` | uint32_t | MoE 专家数（0 = 稠密模型） |
| `n_expert_used` | uint32_t | 每个 token 激活的专家数（MoE） |
| `n_embd_head_k_full` | uint32_t | Key 头维度（全注意力） |
| `n_embd_head_v_full` | uint32_t | Value 头维度 |
| `n_rot_full` | uint32_t | RoPE 维度（全注意力） |
| `n_head_arr[]` | uint32_t[] | 每层查询头数（最多 512 层） |
| `n_head_kv_arr[]` | uint32_t[] | 每层 KV 头数（支持 GQA/MQA） |
| `n_ff_arr[]` | uint32_t[] | 每层 FFN 隐藏维度 |
| `f_norm_rms_eps` | float | RMS 归一化 epsilon |
| `rope_freq_base_train` | float | RoPE 基础频率 |

**设计：** 逐层数组（`n_head_arr`, `n_head_kv_arr`, `n_ff_arr`）支持具有非均匀层配置的架构（例如 DeepSeek、Command-A 具有不同的头数）。

---

## `llama_vocab` (llama-vocab.h:67)

**用途：** 分词器词表 — 在文本和 token ID 之间进行映射。支持 BPE、SPM 和 WPM 分词器类型。

**关键字段：**

| Field | Type | Purpose |
|-------|------|---------|
| `token_data` | struct | 每个 token 的数据：文本、分数、属性 |
| `token_bos/eos/eot/unk/pad/nl` | llama_token | 特殊 token ID |
| `token_fim_pre/suf/mid/pad` | llama_token | Fill-in-the-Middle token |

**关键方法：**

| Method | Purpose |
|--------|---------|
| `is_eog(id)` | 检查 token 是否为生成结束符 |
| `text_to_token(text)` | 通过文本查找单个 token |
| `token_to_byte(id)` | 将字节级 token 映射到字节值 |

---

## `llama_batch` (llama.h:235)

**用途：** 在单次 `llama_decode()` 调用中处理的 token 批次。支持多序列和特定位置的处理。

**字段：**

| Field | Type | Purpose |
|-------|------|---------|
| `n_tokens` | int32_t | 此批次中的 token 数量 |
| `token` | llama_token* | Token ID 数组 |
| `embd` | float* | 嵌入输入（token ID 的替代方式） |
| `pos` | llama_pos* | 每个 token 的位置数组 |
| `n_seq_id` | int32_t* | 每个 token 的序列 ID 数量 |
| `seq_id` | llama_seq_id** | 每个 token 的序列 ID 分配 |
| `logits` | int8_t* | 是否为每个 token 计算 logits |

**设计：** `seq_id` 字段支持多请求批处理 — 来自不同并发请求的 token 共享同一个批次，但被分配不同的序列 ID。KV 缓存使用序列 ID 来隔离上下文。`logits` 控制哪些 token 产生输出（在自回归生成中只有最后一个 token 需要 logits）。

---

## `llama_layer` (llama-model.h:213)

**用途：** 每层 Transformer 权重。模型中的每一层对应一个 `llama_layer` 实例。

**关键字段（子集 — 所有架构的每层张量超过 50 个）：**

| Field | Type | Purpose |
|-------|------|---------|
| `attn_norm` | ggml_tensor* | 注意力前层归一化权重 |
| `wq/wk/wv/wo` | ggml_tensor* | 注意力 Q/K/V/O 投影权重 |
| `wqkv` | ggml_tensor* | 融合 QKV 投影（部分架构） |
| `ffn_norm` | ggml_tensor* | FFN 前层归一化权重 |
| `ffn_up/ffn_gate/ffn_down` | ggml_tensor* | FFN up/gate/down 投影权重 |
| `ffn_gate_exp/ffn_down_exp` | ggml_tensor* | MoE 专家权重 |
| `ffn_gate_shexp` | ggml_tensor* | 共享专家权重 |
| `attn_q_a_norm/attn_kv_a_norm` | ggml_tensor* | MLA（多头潜在注意力）归一化 |
| `wq_a/wq_b` | ggml_tensor* | DeepSeek 风格低秩 Q 投影 |
| `wkv_a_mqa/wkv_b` | ggml_tensor* | DeepSeek MLA 压缩 KV |

**设计：** 该结构体是 100+ 种架构的所有可能逐层张量的联合。大多数架构仅使用其中的一个子集。未使用的张量指针为 `nullptr`。这避免了虚派发，同时支持多种不同架构。

---

## `llama_cparams` (llama-cparams.h:9)

**用途：** 上下文参数 — 推理上下文的运行时配置（区别于模型超参数）。

**关键字段：**

| Field | Type | Purpose |
|-------|------|---------|
| `n_ctx` | uint32_t | 总上下文窗口大小 |
| `n_ctx_seq` | uint32_t | 每序列上下文限制 |
| `n_batch` | uint32_t | 解码的逻辑批次大小 |
| `n_ubatch` | uint32_t | 物理（微）批次大小 |
| `n_seq_max` | uint32_t | 最大并发序列数 |
| `n_threads` | int32_t | 生成的线程数 |
| `rope_freq_base/scale` | float | RoPE 配置 |
| `embeddings` | bool | 启用嵌入模式 |
| `flash_attn` | bool | 使用 Flash Attention 内核 |
| `offload_kqv` | bool | 将 KQV 操作卸载到 GPU |
| `kv_unified` | bool | 统一 KV 缓存（跨序列共享） |

---

## `llama_kv_cells` (llama-kv-cells.h:32)

**用途：** KV 缓存单元的元数据 — 跟踪哪些位置被哪些序列占用。这是"软"KV 缓存状态（位置、序列分配、移位跟踪），与实际的 key/value 张量数据分离。

**关键字段：**

| Field | Type | Purpose |
|-------|------|---------|
| `pos` | vector\<llama_pos\> | 每个单元的位置（-1 = 空） |
| `ext` | vector\<llama_kv_cell_ext\> | 二维位置数据（用于 M-RoPE / 视觉模型） |
| `shift` | vector\<llama_pos\> | 位置移位值（用于上下文扩展） |
| `seq` | vector\<bitset\> | 每个单元的序列成员关系位集 |
| `seq_pos` | map\<llama_seq_id, set\> | 每个序列的位置集合 |
| `has_shift` | bool | 是否有单元存在待处理的位置移位 |
| `used` | bitset | 当前正在使用的单元 |

**复杂逻辑：** KV 缓存使用基于单元的跟踪系统。索引 `i` 处的每个单元存储其位置、它所属的序列（通过位集表示）以及是否有待处理的位置移位（用于 RoPE 缩放 / 上下文移位）。`seq_pos` 映射提供 O(log n) 复杂度的查找，以获取属于给定序列的所有位置。当缓存已满时，使用基于位置比较的 LRU 策略驱逐单元。

---

## `llama_sampler` / `llama_sampler_i` (include/llama.h)

**用途：** 采样接口 — 由可组合的采样操作链构成（温度、top-k、top-p、重复惩罚等）。使用类似 `ggml_backend_i` 的虚表模式。

**关键虚表函数：**

| Function | Purpose |
|----------|---------|
| `name` | 返回采样器名称 |
| `accept` | 当 token 被接受时调用（用于状态更新） |
| `apply` | 对 token 候选数组应用采样 |
| `reset` | 重置内部状态 |
| `free` | 释放资源 |

**内置采样器：** temp, top_k, top_p, min_p, typical_p, penalty (重复), mirostat, dri, grammar, dist (随机选择), infill (token 偏置), branchefield, xtcd, dry, tail_free, eta_cutoff, epsilon_cutoff

**使用模式：**
```c
llama_sampler * chain = llama_sampler_chain_init(params);
llama_sampler_chain_add(chain, llama_sampler_init_top_k(40));
llama_sampler_chain_add(chain, llama_sampler_init_top_p(0.95, 1));
llama_sampler_chain_add(chain, llama_sampler_init_temp(0.8));
llama_sampler_chain_add(chain, llama_sampler_init_dist(seed));

llama_token id = llama_sampler_sample(chain, ctx, -1);
```
