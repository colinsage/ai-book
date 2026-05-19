# vLLM — Core Data Structures

## Request (`vllm/v1/request.py`)

**Purpose:** Represents a single inference request throughout its lifecycle in the engine.

**Key Fields:**

| Field | Type | Purpose |
|-------|------|---------|
| `request_id` | `str` | Unique identifier for the request |
| `prompt_token_ids` | `list[int]` | Tokenized prompt |
| `sampling_params` | `SamplingParams` | Generation parameters (temperature, top_p, max_tokens, etc.) |
| `pooling_params` | `PoolingParams` | Parameters for embedding/pooling tasks |
| `arrival_time` | `float` | Timestamp when request arrived |
| `lora_request` | `LoRARequest` | LoRA adapter to apply (if any) |
| `mm_features` | `list[MultiModalFeatureSpec]` | Multi-modal features (images, audio, video) |
| `priority` | `int` | Scheduling priority (0 = default) |
| `structured_output_request` | `StructuredOutputRequest` | Grammar constraint for structured generation |
| `num_computed_tokens` | `int` | Number of tokens whose KV cache is already computed |
| `num_output_placeholders` | `int` | Output placeholder count for deferred output |
| `resumable` | `bool` | Whether this request supports streaming continuation |

**Key Methods:**

| Method | Complexity | Notes |
|--------|-----------|-------|
| `num_tokens_with_spec` | O(1) | Returns total token count including speculative tokens |
| `get_prefix_cache_block_hash()` | O(blocks) | Computes block hash for prefix caching |
| `is_prefilling()` | O(1) | True if still in prefill phase |

---

## SamplingParams (`vllm/sampling_params.py`)

**Purpose:** Controls text generation behavior — temperature, top-p, top-k, frequency/repetition penalties, structured output constraints, and more.

**Key Fields:**

| Field | Type | Purpose |
|-------|------|---------|
| `n` | `int` | Number of completions per prompt |
| `temperature` | `float` | Sampling temperature (0 = greedy) |
| `top_p` | `float` | Nucleus sampling threshold |
| `top_k` | `int` | Top-K sampling |
| `max_tokens` | `int` | Maximum output tokens |
| `stop` | `list[str]` | Stop sequences |
| `stop_token_ids` | `list[int]` | Stop token IDs |
| `frequency_penalty` | `float` | Frequency penalty |
| `presence_penalty` | `float` | Presence penalty |
| `repetition_penalty` | `float` | Repetition penalty |
| `structured_outputs` | `StructuredOutputsParams` | JSON schema, regex, grammar, or choice constraints |
| `logprobs` | `int` | Number of logprobs to return |
| `seed` | `int` | Random seed for reproducibility |

**Complex Logic:**
- `StructuredOutputsParams` validates mutual exclusivity — only one of json/regex/choice/grammar/json_object/structural_tag can be set
- `_backend` is auto-set by `Processor._validate_structured_output` during request processing
- `SamplingType` enum (GREEDY, RANDOM, RANDOM_SEED) determines dispatch in the sampling kernel

---

## EngineCoreRequest (`vllm/v1/engine/__init__.py`)

**Purpose:** Serialized request format sent from API server to engine core process via ZMQ/msgpack.

Lightweight dataclass containing only the fields needed for the engine core — no Python object references, enabling cross-process serialization.

---

## KVCacheBlock (`vllm/v1/core/kv_cache_utils.py`)

**Purpose:** Represents a single block of KV cache memory. Used by the KV cache manager to track allocation and reference counting.

**Key Fields:**

| Field | Type | Purpose |
|-------|------|---------|
| `block_id` | `int` | Physical block index in the KV cache tensor |
| `ref_count` | `int` | Number of requests sharing this block (for prefix caching) |
| `block_hash` | `BlockHash` | Hash of the token content in this block (for prefix cache lookup) |

**Key Concepts:**
- **Block size**: Number of tokens per block (configurable, typically 16)
- **Prefix caching**: Blocks with the same hash can be shared across requests
- **Reference counting**: Shared blocks are freed only when ref_count drops to 0
- **BlockHashWithGroupId**: Packs block hash + KV cache group ID into bytes for efficient lookup

---

## KVCacheBlocks (`vllm/v1/core/kv_cache_manager.py`)

**Purpose:** Allocation result from KV cache manager — the interface between Scheduler and KV cache internals.

**Structure:**
- `blocks: tuple[Sequence[KVCacheBlock], ...]` — indexed by `[kv_cache_group][block_index]`
- Supports different KV cache groups (e.g., full attention vs. Mamba state)
- `get_block_ids()` converts to integer block ID tuples for GPU transfer

---

## BlockTable (`vllm/v1/worker/block_table.py`)

**Purpose:** GPU-side mapping from request slots to KV cache block IDs. Used by attention kernels to locate KV cache data.

**Key Fields:**

| Field | Type | Purpose |
|-------|------|---------|
| `block_size` | `int` | Tokens per block |
| `max_num_reqs` | `int` | Maximum concurrent requests |
| `max_num_blocks_per_req` | `int` | Maximum blocks per request |
| `device` | `torch.device` | GPU device for the table |

**Complex Logic:**
- Supports **hybrid blocks** where allocation block size differs from kernel block size (splitting/merging)
- Maintains CPU (pinned memory) and GPU copies for efficient batch updates
- `CpuGpuBuffer` pattern: update on CPU, async copy to GPU

---

## SchedulerOutput (`vllm/v1/core/sched/output.py`)

**Purpose:** Output from the scheduler containing all information needed by the model runner for one step.

**Key Fields:**

| Field | Type | Purpose |
|-------|------|---------|
| `scheduled_new_reqs` | `list[Request]` | Newly scheduled requests |
| `scheduled_resumed_reqs` | `list[Request]` | Resumed (previously preempted) requests |
| `scheduled_running_reqs` | `list[Request]` | Running requests with new tokens |
| `num_scheduled_tokens` | `dict[str, int]` | Tokens scheduled per request |
| `total_num_scheduled_tokens` | `int` | Total tokens in this batch |
| `preempted_reqs` | `list[Request]` | Preempted requests |
| `scheduled_spec_decode_tokens` | `dict[str, list[int]]` | Speculative decode tokens per request |

---

## ModelRunnerOutput (`vllm/v1/worker/gpu_model_runner.py`)

**Purpose:** Output from the GPU model runner containing sampled tokens and metadata.

Contains per-request: sampled token IDs, logprobs, and whether the request finished.

---

## IntermediateTensors (`vllm/sequence.py`)

**Purpose:** For pipeline parallelism — carries hidden states and residuals between pipeline stages.

**Key Fields:**

| Field | Type | Purpose |
|-------|------|---------|
| `tensors` | `dict[str, torch.Tensor]` | Named tensors (hidden states, residuals) |
| `kv_connector_output` | `KVConnectorOutput` | KV transfer output for disaggregated serving |

**Design Note:** Manually defines `__init__` so that `torch.compile` (Dynamo) can trace the constructor's origin — dataclass-generated `__init__` would lose this information.

---

## EngineCoreEvent / EngineCoreEventType (`vllm/v1/engine/__init__.py`)

**Purpose:** Event system for KV cache observability — emitted when blocks are allocated, freed, or evicted.

Enables external systems to track KV cache lifecycle for disaggregated serving coordination.

---

## KVCacheConfig / KVCacheSpec (`vllm/v1/kv_cache_interface.py`)

**Purpose:** Describes the structure and requirements of KV caches per model architecture.

**Key Types:**

| Type | Purpose |
|------|---------|
| `FullAttentionSpec` | Standard full-attention KV cache |
| `MLAAttentionSpec` | Multi-head Latent Attention (DeepSeek) |
| `MambaSpec` | Mamba SSM state cache |
| `ChunkedLocalAttentionSpec` | Sliding window / chunked attention |
| `SlidingWindowSpec` | Sliding window attention |
| `KVCacheGroupSpec` | Group of KV cache specs sharing the same pool |
| `KVCacheTensor` | Physical tensor allocation descriptor |
| `KVQuantMode` | Quantization mode (NONE, FP8, INT8, NVFP4) |

**Design Intent:** Models can have heterogeneous KV cache requirements (e.g., Llama has full attention; Jamba has both attention + Mamba). The config system allows different cache pools with different specs to coexist.
