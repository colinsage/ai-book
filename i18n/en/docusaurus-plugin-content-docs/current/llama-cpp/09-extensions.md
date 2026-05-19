# llama.cpp — Extension & Plugin System

llama.cpp has two primary extension mechanisms: the **GGML Backend** system for hardware acceleration, and the **Sampler Chain** for customizing token selection.

## 9.1 GGML Backend System

**Location:** `ggml/src/ggml-backend-impl.h` — `ggml_backend_i` vtable struct (line 105)
**Mechanism:** Function pointer vtable (C-style interface polymorphism). Each hardware backend implements this interface.

### Core Interfaces

#### `ggml_backend_buffer_type_i` (line 17)

Defines how to allocate tensors on a specific device:

```c
struct ggml_backend_buffer_type_i {
    const char * (*get_name)       (ggml_backend_buffer_type_t buft);
    ggml_backend_buffer_t (*alloc_buffer) (ggml_backend_buffer_type_t buft, size_t size);
    size_t       (*get_alignment)  (ggml_backend_buffer_type_t buft);
    size_t       (*get_max_size)   (ggml_backend_buffer_type_t buft);    // optional
    size_t       (*get_alloc_size) (ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor);  // optional
    bool         (*is_host)        (ggml_backend_buffer_type_t buft);    // optional
};
```

#### `ggml_backend_buffer_i` (line 41)

Defines how to access tensor data in a buffer:

```c
struct ggml_backend_buffer_i {
    void         (*free_buffer)    (ggml_backend_buffer_t buffer);      // optional
    void *       (*get_base)       (ggml_backend_buffer_t buffer);
    enum ggml_status (*init_tensor)(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor);  // optional
    void         (*memset_tensor)  (ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size);
    void         (*set_tensor)     (ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
    void         (*get_tensor)     (ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size);
    bool         (*cpy_tensor)     (ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst);  // optional
    void         (*clear)          (ggml_backend_buffer_t buffer, uint8_t value);
    void         (*reset)          (ggml_backend_buffer_t buffer);      // optional
};
```

#### `ggml_backend_i` (line 105)

The main backend interface for compute:

```c
struct ggml_backend_i {
    const char * (*get_name)           (ggml_backend_t backend);
    void         (*free)               (ggml_backend_t backend);
    void         (*set_tensor_async)   (ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
    void         (*get_tensor_async)   (ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size);
    bool         (*cpy_tensor_async)   (ggml_backend_t backend_src, ggml_backend_t backend_dst, const struct ggml_tensor * src, struct ggml_tensor * dst);
    void         (*synchronize)        (ggml_backend_t backend);
    ggml_backend_graph_plan_t (*graph_plan_create) (ggml_backend_t backend, const struct ggml_cgraph * cgraph);
    void         (*graph_plan_free)    (ggml_backend_t backend, ggml_backend_graph_plan_t plan);
    enum ggml_status (*graph_compute)  (ggml_backend_t backend, struct ggml_cgraph * cgraph);
    void         (*event_record)       (ggml_backend_t backend, ggml_backend_event_t event);
    void         (*event_wait)         (ggml_backend_t backend, ggml_backend_event_t event);
    void         (*graph_optimize)     (ggml_backend_t backend, struct ggml_cgraph * cgraph);
};
```

### Existing Backend Implementations

| Backend | Directory | Accelerator | Key Op Implementations |
|---------|-----------|-------------|----------------------|
| CPU | `ggml/src/ggml-cpu/` | Default | All ops (reference implementation) |
| CUDA | `ggml/src/ggml-cuda/` | NVIDIA GPU | MUL_MAT, MUL_MAT_ID, FLASH_ATTN, RMS_NORM, ROPE, CONV_* |
| Metal | `ggml/src/ggml-metal/` | Apple GPU | MUL_MAT, FLASH_ATTN, RMS_NORM, ROPE, MUL_MAT_ID |
| Vulkan | `ggml/src/ggml-vulkan/` | Cross-platform GPU | MUL_MAT, MUL_MAT_ID, RMS_NORM, ROPE |
| HIP/ROCm | `ggml/src/ggml-hip/` | AMD GPU | Wraps CUDA implementation |
| SYCL | `ggml/src/ggml-sycl/` | Intel GPU | Fork of CUDA backend |
| CANN | `ggml/src/ggml-cann/` | Ascend NPU | MUL_MAT, RMS_NORM, ROPE |
| RPC | `ggml/src/ggml-rpc/` | Remote | Proxies ops over network |

### How to Add a New Backend

1. **Create directory:** `ggml/src/ggml-<name>/`
2. **Implement `ggml_backend_i`:** Define all required function pointers
3. **Implement `ggml_backend_buffer_type_i` and `ggml_backend_buffer_i`:** For memory management
4. **Register with the scheduler:** The `ggml_backend_sched` automatically routes ops to the appropriate backend
5. **Add CMake build option:** Add `GGML_<NAME>` option in root `CMakeLists.txt`
6. **Implement critical ops:** At minimum: MUL_MAT, RMS_NORM, ROPE, SILU, ADD, MUL, RESHAPE, VIEW, CPY

**Backend Registration Flow:**
```c
// In ggml-backend.cpp:
ggml_backend_t ggml_backend_cpu_init(void) {
    // 1. Allocate backend struct
    // 2. Set vtable to cpu_backend_i
    // 3. Set device and buffer type
    return backend;
}
```

The `ggml_backend_sched` determines which backend handles each graph node by:
1. Checking where input tensors reside (CPU buffer vs GPU buffer)
2. Checking which backends support the operation
3. Inserting `CPY` nodes for cross-backend data movement

---

## 9.2 Sampler Chain System

**Location:** `include/llama.h` — `llama_sampler_i` vtable struct
**Mechanism:** Chain of responsibility pattern — each sampler in the chain modifies the candidate token distribution before passing to the next.

### Interface

```c
struct llama_sampler_i {
    const char * (*name)   (const struct llama_sampler * smpl);
    void         (*accept) (struct llama_sampler * smpl, llama_token token);
    void         (*apply)  (struct llama_sampler * smpl, llama_token_data_array * cur_p);
    void         (*reset)  (struct llama_sampler * smpl);
    void         (*free)   (struct llama_sampler * smpl);
};
```

### Built-in Sampler Implementations

| Sampler | Location | Purpose |
|---------|----------|---------|
| `llama_sampler_init_top_k` | llama-sampler.cpp | Keep only top-K candidates |
| `llama_sampler_init_top_p` | llama-sampler.cpp | Nucleus sampling (top-p) |
| `llama_sampler_init_min_p` | llama-sampler.cpp | Minimum probability threshold |
| `llama_sampler_init_typical` | llama-sampler.cpp | Typical sampling |
| `llama_sampler_init_temp` | llama-sampler.cpp | Temperature scaling |
| `llama_sampler_init_dist` | llama-sampler.cpp | Random selection from distribution |
| `llama_sampler_init_greedy` | llama-sampler.cpp | Always pick highest probability |
| `llama_sampler_init_mirostat` | llama-sampler.cpp | Mirostat v1 adaptive entropy |
| `llama_sampler_init_mirostat_v2` | llama-sampler.cpp | Mirostat v2 |
| `llama_sampler_init_penalties` | llama-sampler.cpp | Repetition/frequency/presence penalties |
| `llama_sampler_init_grammar` | llama-sampler.cpp | GBNF grammar-constrained generation |
| `llama_sampler_init_dry` | llama-sampler.cpp | DRY (Don't Repeat Yourself) penalty |
| `llama_sampler_init_xtc` | llama-sampler.cpp | XTC sampler |
| `llama_sampler_init_infill` | llama-sampler.cpp | Token biasing for code infill |
| `llama_sampler_init_dri` | llama-sampler.cpp | DRI sampler |
| `llama_sampler_init_tail_free` | llama-sampler.cpp | Tail-free sampling |
| `llama_sampler_init_eta_cutoff` | llama-sampler.cpp | Eta cutoff |
| `llama_sampler_init_epsilon_cutoff` | llama-sampler.cpp | Epsilon cutoff |
| `llama_sampler_init_branchefield` | llama-sampler.cpp | Branchefield sampler |
| `llama_sampler_init_logit_bias` | llama-sampler.cpp | Direct logit bias per token |

### How to Add a Custom Sampler

```c
// 1. Define your state
typedef struct {
    float my_param;
} my_sampler_ctx;

// 2. Implement the vtable
static const char * my_sampler_name(const struct llama_sampler * smpl) {
    return "my_sampler";
}

static void my_sampler_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p) {
    my_sampler_ctx * ctx = (my_sampler_ctx *)smpl->ctx;
    // Modify cur_p->data[].logit values here
}

static void my_sampler_free(struct llama_sampler * smpl) {
    free(smpl->ctx);
}

static const struct llama_sampler_i my_sampler_i = {
    .name   = my_sampler_name,
    .accept = NULL,  // optional
    .apply  = my_sampler_apply,
    .reset  = NULL,  // optional
    .free   = my_sampler_free,
};

// 3. Create and add to chain
struct llama_sampler * my_sampler = llama_sampler_init(&my_sampler_i, &(my_sampler_ctx){.my_param = 0.5f});
llama_sampler_chain_add(chain, my_sampler);
```

---

## 9.3 Model Architecture Extension

**Location:** `src/llama-arch.h` — `llm_arch` enum + `llm_tensor_fetch` registry

Adding a new model architecture requires:

1. Add a new `LLM_ARCH_*` enum value to `llm_arch` in `src/llama-arch.h`
2. Register the architecture's tensor name mapping in the `llm_tensor_fetch` table
3. Implement the forward pass in `src/llama-model.cpp` by adding a case to the `llm_build_*` switch
4. Add architecture detection in `load_arch()` based on GGUF `general.architecture` metadata
5. Update `models/convert_hf_to_gguf.py` to support the new architecture's weight naming convention
