# llama.cpp — Core Data Structures

## `ggml_tensor` (ggml.h:660)

**Purpose:** Fundamental n-dimensional tensor type. Every model weight, activation, and intermediate computation result is represented as a `ggml_tensor`. Tensors form a lazy compute DAG — building operations (e.g., `ggml_mul_mat`) creates tensor nodes with `op` fields; actual computation happens when `ggml_graph_compute()` executes the graph.

**Fields:**

| Field | Type | Purpose |
|-------|------|---------|
| `type` | enum ggml_type | Data type (F32, F16, BF16, Q4_0, Q4_K, Q8_0, IQ series, etc. — 42 types) |
| `buffer` | ggml_backend_buffer* | Backend buffer owning this tensor's memory (CPU, CUDA, Metal, etc.) |
| `ne[4]` | int64_t | Shape: number of elements per dimension (max 4D) |
| `nb[4]` | size_t | Strides in bytes per dimension. nb[0] = type_size, nb[i] = nb[i-1]*ne[i-1] |
| `op` | enum ggml_op | Compute operation (NONE for leaf tensors, MUL_MAT, ADD, RMS_NORM, etc.) |
| `op_params[16]` | int32_t | Operation-specific parameters (e.g., axis for SOFT_MAX, MUL_MAT parameters) |
| `flags` | int32_t | Tensor flags (e.g., DONT_MMAP, NOT_PERSISTENT) |
| `src[GGML_MAX_SRC]` | ggml_tensor** | Input tensors for compute graph node (up to 10 sources) |
| `view_src` | ggml_tensor* | Source tensor for views (shares underlying data) |
| `view_offs` | size_t | Byte offset into view_src's data |
| `data` | void* | Raw data pointer (CPU or GPU address) |
| `name` | char[64] | Human-readable tensor name (for debugging) |
| `extra` | void* | Backend-specific extra data (e.g., CUDA tensor extras) |

**Key Functions:**

| Function | Complexity | Notes |
|----------|-----------|-------|
| `ggml_new_tensor()` | O(1) | Allocates tensor metadata from ggml_context arena |
| `ggml_mul_mat()` | O(1) build / O(mnk) compute | Creates lazy matrix multiply node |
| `ggml_graph_compute()` | O(nodes) | Topologically sorts DAG, dispatches to backends |
| `ggml_view_tensor()` | O(1) | Creates a view sharing the same data buffer |

**Complex Logic:** `ggml_graph_compute()` topologically sorts the DAG, then dispatches each node to the appropriate backend (CPU, CUDA, Metal) via `ggml_backend_sched`. The scheduler determines which backend should handle each operation based on where the input tensors reside and which backends support the operation. Cross-backend copies are inserted automatically.

---

## `ggml_cgraph` (ggml-impl.h:329)

**Purpose:** Compute graph — a directed acyclic graph of tensor operations. Built during forward pass construction, executed by the backend scheduler.

**Fields:**

| Field | Type | Purpose |
|-------|------|---------|
| `size` | int | Maximum number of nodes/leafs |
| `n_nodes` | int | Number of active compute nodes |
| `n_leafs` | int | Number of leaf (constant) tensors |
| `nodes` | ggml_tensor** | Array of compute nodes (ops that produce output) |
| `grads` | ggml_tensor** | Gradient tensors (training only) |
| `grad_accs` | ggml_tensor** | Gradient accumulators (training only) |
| `leafs` | ggml_tensor** | Array of leaf tensors (model weights, constants) |
| `use_counts` | int32_t* | Reference counts per tensor (for memory reuse) |
| `visited_hash_set` | ggml_hash_set | Hash set for cycle detection during graph build |
| `order` | enum ggml_cgraph_eval_order | Evaluation order (left-to-right or topological) |
| `uid` | uint64_t | Optional identifier for graph matching |

**Design:** The graph is built lazily — each ggml operation (e.g., `ggml_mul_mat(ctx, a, b)`) adds a new tensor node and registers its sources. During `ggml_graph_compute()`, nodes are evaluated in topological order. The `visited_hash_set` prevents duplicate tensor insertion.

---

## `ggml_backend_i` (ggml-backend-impl.h:105)

**Purpose:** Virtual table (vtable) for hardware backend implementations. This is the core extension point for adding new compute hardware to GGML. Each backend (CPU, CUDA, Metal, Vulkan, etc.) implements this interface.

**Function Pointers:**

| Function | Required | Purpose |
|----------|----------|---------|
| `get_name` | Yes | Return backend name string |
| `free` | Yes | Release backend resources |
| `set_tensor_async` | No | Asynchronously write tensor data |
| `get_tensor_async` | No | Asynchronously read tensor data |
| `cpy_tensor_async` | No | Async cross-backend tensor copy |
| `synchronize` | No | Wait for all pending async operations |
| `graph_plan_create` | No | Create reusable graph execution plan |
| `graph_plan_compute` | No | Execute graph with pre-built plan |
| `graph_compute` | Yes | Execute compute graph (async if supported) |
| `event_record` / `event_wait` | No | Inter-backend synchronization primitives |
| `graph_optimize` | No | Backend-specific graph optimization |

**Related Interfaces:**

- `ggml_backend_buffer_type_i` (ggml-backend-impl.h:17) — Buffer type vtable (allocation, alignment, is_host)
- `ggml_backend_buffer_i` (ggml-backend-impl.h:41) — Buffer vtable (free, get_base, set_tensor, get_tensor)

---

## `llama_model` (llama-model.h:512)

**Purpose:** Loaded model state — holds all model weights, hyperparameters, vocabulary, and metadata loaded from a GGUF file.

**Key Fields:**

| Field | Type | Purpose |
|-------|------|---------|
| `type` | llm_type | Model type (e.g., LLM_TYPE_8B, LLM_TYPE_70B) |
| `arch` | llm_arch | Architecture enum (LLAMA, GPT2, FALCON, MISTRAL, PHI, GEMMA, etc. — 100+ types) |
| `hparams` | llama_hparams | Model hyperparameters (dimensions, layers, heads, etc.) |
| `vocab` | llama_vocab | Tokenizer vocabulary |
| `tok_embd` | ggml_tensor* | Token embedding weight matrix |
| `output_norm` | ggml_tensor* | Final layer norm weight |
| `output` | ggml_tensor* | Output projection (lm_head) weight |
| `layers` | vector\<llama_layer\> | Per-layer weights (attention + FFN) |
| `devices` | vector\<llama_device\> | List of devices used for this model |
| `gguf_kv` | unordered_map | Raw GGUF metadata key-value pairs |
| `loras` | unordered_set | Active LoRA adapters |

**Key Methods:**

| Method | Purpose |
|--------|---------|
| `load_arch()` | Determine model architecture from GGUF metadata |
| `load_hparams()` | Parse hyperparameters from GGUF |
| `load_vocab()` | Load tokenizer vocabulary |
| `load_tensors()` | Load all tensor weights into backend buffers |

---

## `llama_hparams` (llama-hparams.h:36)

**Purpose:** Model hyperparameters extracted from GGUF metadata. These define the model's architecture and are read-only after loading.

**Key Fields:**

| Field | Type | Purpose |
|-------|------|---------|
| `n_embd` | uint32_t | Embedding dimension |
| `n_layer` | uint32_t | Number of transformer layers |
| `n_expert` | uint32_t | Number of MoE experts (0 = dense) |
| `n_expert_used` | uint32_t | Active experts per token (MoE) |
| `n_embd_head_k_full` | uint32_t | Key head dimension (full attention) |
| `n_embd_head_v_full` | uint32_t | Value head dimension |
| `n_rot_full` | uint32_t | RoPE dimension (full attention) |
| `n_head_arr[]` | uint32_t[] | Per-layer query head count (up to 512 layers) |
| `n_head_kv_arr[]` | uint32_t[] | Per-layer KV head count (GQA/MQA support) |
| `n_ff_arr[]` | uint32_t[] | Per-layer FFN hidden dimension |
| `f_norm_rms_eps` | float | RMS norm epsilon |
| `rope_freq_base_train` | float | RoPE base frequency |

**Design:** Per-layer arrays (`n_head_arr`, `n_head_kv_arr`, `n_ff_arr`) support architectures with non-uniform layer configurations (e.g., DeepSeek, Command-A with varying head counts).

---

## `llama_vocab` (llama-vocab.h:67)

**Purpose:** Tokenizer vocabulary — maps between text and token IDs. Supports BPE, SPM, and WPM tokenizer types.

**Key Fields:**

| Field | Type | Purpose |
|-------|------|---------|
| `token_data` | struct | Per-token data: text, score, attributes |
| `token_bos/eos/eot/unk/pad/nl` | llama_token | Special token IDs |
| `token_fim_pre/suf/mid/pad` | llama_token | Fill-in-the-Middle tokens |

**Key Methods:**

| Method | Purpose |
|--------|---------|
| `is_eog(id)` | Check if token is end-of-generation |
| `text_to_token(text)` | Look up single token by text |
| `token_to_byte(id)` | Map byte-level token to byte value |

---

## `llama_batch` (llama.h:235)

**Purpose:** Batch of tokens to process in a single `llama_decode()` call. Supports multi-sequence and position-specific processing.

**Fields:**

| Field | Type | Purpose |
|-------|------|---------|
| `n_tokens` | int32_t | Number of tokens in this batch |
| `token` | llama_token* | Token IDs array |
| `embd` | float* | Embedding input (alternative to token IDs) |
| `pos` | llama_pos* | Position array for each token |
| `n_seq_id` | int32_t* | Number of sequence IDs per token |
| `seq_id` | llama_seq_id** | Sequence ID assignments per token |
| `logits` | int8_t* | Whether to compute logits for each token |

**Design:** The `seq_id` field enables multi-request batching — tokens from different concurrent requests share a single batch but are assigned different sequence IDs. The KV cache uses sequence IDs to separate contexts. `logits` controls which tokens produce output (only the last token needs logits in autoregressive generation).

---

## `llama_layer` (llama-model.h:213)

**Purpose:** Per-layer transformer weights. One `llama_layer` instance per layer in the model.

**Key Fields (subset — there are 50+ tensors per layer for all architectures):**

| Field | Type | Purpose |
|-------|------|---------|
| `attn_norm` | ggml_tensor* | Pre-attention layer norm weight |
| `wq/wk/wv/wo` | ggml_tensor* | Attention Q/K/V/O projection weights |
| `wqkv` | ggml_tensor* | Fused QKV projection (some architectures) |
| `ffn_norm` | ggml_tensor* | Pre-FFN layer norm weight |
| `ffn_up/ffn_gate/ffn_down` | ggml_tensor* | FFN up/gate/down projection weights |
| `ffn_gate_exp/ffn_down_exp` | ggml_tensor* | MoE expert weights |
| `ffn_gate_shexp` | ggml_tensor* | Shared expert weight |
| `attn_q_a_norm/attn_kv_a_norm` | ggml_tensor* | MLA (Multi-head Latent Attention) norms |
| `wq_a/wq_b` | ggml_tensor* | DeepSeek-style low-rank Q projection |
| `wkv_a_mqa/wkv_b` | ggml_tensor* | DeepSeek MLA compressed KV |

**Design:** The struct is a union of all possible per-layer tensors across 100+ architectures. Most architectures use only a subset. Pointers are `nullptr` for unused tensors. This avoids virtual dispatch while supporting diverse architectures.

---

## `llama_cparams` (llama-cparams.h:9)

**Purpose:** Context parameters — runtime configuration for an inference context (distinct from model hyperparameters).

**Key Fields:**

| Field | Type | Purpose |
|-------|------|---------|
| `n_ctx` | uint32_t | Total context window size |
| `n_ctx_seq` | uint32_t | Per-sequence context limit |
| `n_batch` | uint32_t | Logical batch size for decode |
| `n_ubatch` | uint32_t | Physical (micro) batch size |
| `n_seq_max` | uint32_t | Maximum concurrent sequences |
| `n_threads` | int32_t | Thread count for generation |
| `rope_freq_base/scale` | float | RoPE configuration |
| `embeddings` | bool | Enable embedding mode |
| `flash_attn` | bool | Use Flash Attention kernel |
| `offload_kqv` | bool | Offload KQV operations to GPU |
| `kv_unified` | bool | Unified KV cache (shared across sequences) |

---

## `llama_kv_cells` (llama-kv-cells.h:32)

**Purpose:** Metadata for KV cache cells — tracks which positions are occupied by which sequences. This is the "soft" KV cache state (positions, sequence assignments, shift tracking), separate from the actual key/value tensor data.

**Key Fields:**

| Field | Type | Purpose |
|-------|------|---------|
| `pos` | vector\<llama_pos\> | Position of each cell (-1 = empty) |
| `ext` | vector\<llama_kv_cell_ext\> | 2D position data (for M-RoPE / vision models) |
| `shift` | vector\<llama_pos\> | Position shift values (for context extension) |
| `seq` | vector\<bitset\> | Sequence membership bitset per cell |
| `seq_pos` | map\<llama_seq_id, set\> | Position sets per sequence |
| `has_shift` | bool | Whether any cells have pending position shifts |
| `used` | bitset | Which cells are currently in use |

**Complex Logic:** The KV cache uses a cell-based tracking system. Each cell at index `i` stores its position, which sequences it belongs to (via bitset), and whether it has a pending position shift (for RoPE scaling / context shifting). The `seq_pos` map provides O(log n) lookup of all positions belonging to a given sequence. When the cache is full, cells are evicted using an LRU policy based on position comparisons.

---

## `llama_sampler` / `llama_sampler_i` (include/llama.h)

**Purpose:** Sampling interface — a composable chain of sampling operations (temperature, top-k, top-p, repetition penalty, etc.). Uses a vtable pattern similar to `ggml_backend_i`.

**Key vtable functions:**

| Function | Purpose |
|----------|---------|
| `name` | Return sampler name |
| `accept` | Called when a token is accepted (for state updates) |
| `apply` | Apply sampling to a token candidate array |
| `reset` | Reset internal state |
| `free` | Release resources |

**Built-in Samplers:** temp, top_k, top_p, min_p, typical_p, penalty (repetition), mirostat, dri, grammar, dist (random selection), infill (token bias), branchefield, xtcd, dry, tail_free, eta_cutoff, epsilon_cutoff

**Usage Pattern:**
```c
llama_sampler * chain = llama_sampler_chain_init(params);
llama_sampler_chain_add(chain, llama_sampler_init_top_k(40));
llama_sampler_chain_add(chain, llama_sampler_init_top_p(0.95, 1));
llama_sampler_chain_add(chain, llama_sampler_init_temp(0.8));
llama_sampler_chain_add(chain, llama_sampler_init_dist(seed));

llama_token id = llama_sampler_sample(chain, ctx, -1);
```
