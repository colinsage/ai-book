# llama.cpp — 扩展与插件系统

llama.cpp 有两个主要的扩展机制：用于硬件加速的 **GGML Backend** 系统，以及用于自定义 token 选择的 **Sampler Chain**（采样器链）。

## 9.1 GGML Backend 系统

**位置：** `ggml/src/ggml-backend-impl.h` — `ggml_backend_i` vtable 结构体（第 105 行）
**机制：** 函数指针虚表（C 风格接口多态）。每个硬件后端实现此接口。

### 核心接口

#### `ggml_backend_buffer_type_i` (第 17 行)

定义如何在特定设备上分配张量：

```c
struct ggml_backend_buffer_type_i {
 const char * (*get_name) (ggml_backend_buffer_type_t buft);
 ggml_backend_buffer_t (*alloc_buffer) (ggml_backend_buffer_type_t buft, size_t size);
 size_t (*get_alignment) (ggml_backend_buffer_type_t buft);
 size_t (*get_max_size) (ggml_backend_buffer_type_t buft); // optional
 size_t (*get_alloc_size) (ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor); // optional
 bool (*is_host) (ggml_backend_buffer_type_t buft); // optional
};
```

#### `ggml_backend_buffer_i` (第 41 行)

定义如何访问缓冲区中的张量数据：

```c
struct ggml_backend_buffer_i {
 void (*free_buffer) (ggml_backend_buffer_t buffer); // optional
 void * (*get_base) (ggml_backend_buffer_t buffer);
 enum ggml_status (*init_tensor)(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor); // optional
 void (*memset_tensor) (ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size);
 void (*set_tensor) (ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
 void (*get_tensor) (ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size);
 bool (*cpy_tensor) (ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst); // optional
 void (*clear) (ggml_backend_buffer_t buffer, uint8_t value);
 void (*reset) (ggml_backend_buffer_t buffer); // optional
};
```

#### `ggml_backend_i` (第 105 行)

计算的主后端接口：

```c
struct ggml_backend_i {
 const char * (*get_name) (ggml_backend_t backend);
 void (*free) (ggml_backend_t backend);
 void (*set_tensor_async) (ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
 void (*get_tensor_async) (ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size);
 bool (*cpy_tensor_async) (ggml_backend_t backend_src, ggml_backend_t backend_dst, const struct ggml_tensor * src, struct ggml_tensor * dst);
 void (*synchronize) (ggml_backend_t backend);
 ggml_backend_graph_plan_t (*graph_plan_create) (ggml_backend_t backend, const struct ggml_cgraph * cgraph);
 void (*graph_plan_free) (ggml_backend_t backend, ggml_backend_graph_plan_t plan);
 enum ggml_status (*graph_compute) (ggml_backend_t backend, struct ggml_cgraph * cgraph);
 void (*event_record) (ggml_backend_t backend, ggml_backend_event_t event);
 void (*event_wait) (ggml_backend_t backend, ggml_backend_event_t event);
 void (*graph_optimize) (ggml_backend_t backend, struct ggml_cgraph * cgraph);
};
```

### 现有后端实现

| 后端 | 目录 | 加速器 | 关键算子实现 |
|---------|-----------|-------------|----------------------|
| CPU | `ggml/src/ggml-cpu/` | 默认 | 所有算子（参考实现） |
| CUDA | `ggml/src/ggml-cuda/` | NVIDIA GPU | MUL_MAT, MUL_MAT_ID, FLASH_ATTN, RMS_NORM, ROPE, CONV_* |
| Metal | `ggml/src/ggml-metal/` | Apple GPU | MUL_MAT, FLASH_ATTN, RMS_NORM, ROPE, MUL_MAT_ID |
| Vulkan | `ggml/src/ggml-vulkan/` | 跨平台 GPU | MUL_MAT, MUL_MAT_ID, RMS_NORM, ROPE |
| HIP/ROCm | `ggml/src/ggml-hip/` | AMD GPU | 封装 CUDA 实现 |
| SYCL | `ggml/src/ggml-sycl/` | Intel GPU | CUDA 后端的分支 |
| CANN | `ggml/src/ggml-cann/` | Ascend NPU | MUL_MAT, RMS_NORM, ROPE |
| RPC | `ggml/src/ggml-rpc/` | 远程 | 通过网络代理算子 |

### 如何添加新后端

1. **创建目录：** `ggml/src/ggml-<name>/`
2. **实现 `ggml_backend_i`：** 定义所有必需的函数指针
3. **实现 `ggml_backend_buffer_type_i` 和 `ggml_backend_buffer_i`：** 用于内存管理
4. **注册到调度器：** `ggml_backend_sched` 自动将算子路由到相应的后端
5. **添加 CMake 构建选项：** 在根 `CMakeLists.txt` 中添加 `GGML_<NAME>` 选项
6. **实现关键算子：** 至少包括：MUL_MAT, RMS_NORM, ROPE, SILU, ADD, MUL, RESHAPE, VIEW, CPY

**后端注册流程：**
```c
// In ggml-backend.cpp:
ggml_backend_t ggml_backend_cpu_init(void) {
 // 1. Allocate backend struct
 // 2. Set vtable to cpu_backend_i
 // 3. Set device and buffer type
 return backend;
}
```

`ggml_backend_sched` 通过以下方式决定哪个后端处理每个图节点：
1. 检查输入张量驻留在何处（CPU 缓冲区 vs GPU 缓冲区）
2. 检查哪些后端支持该操作
3. 插入 `CPY` 节点用于跨后端数据移动

---

## 9.2 Sampler Chain 系统

**位置：** `include/llama.h` — `llama_sampler_i` vtable 结构体
**机制：** 责任链模式 — 链中的每个采样器在传递给下一个之前修改候选 token 分布。

### 接口

```c
struct llama_sampler_i {
 const char * (*name) (const struct llama_sampler * smpl);
 void (*accept) (struct llama_sampler * smpl, llama_token token);
 void (*apply) (struct llama_sampler * smpl, llama_token_data_array * cur_p);
 void (*reset) (struct llama_sampler * smpl);
 void (*free) (struct llama_sampler * smpl);
};
```

### 内置采样器实现

| 采样器 | 位置 | 用途 |
|---------|----------|---------|
| `llama_sampler_init_top_k` | llama-sampler.cpp | 仅保留 top-K 候选 |
| `llama_sampler_init_top_p` | llama-sampler.cpp | 核采样 (top-p) |
| `llama_sampler_init_min_p` | llama-sampler.cpp | 最低概率阈值 |
| `llama_sampler_init_typical` | llama-sampler.cpp | 典型采样 |
| `llama_sampler_init_temp` | llama-sampler.cpp | 温度缩放 |
| `llama_sampler_init_dist` | llama-sampler.cpp | 从分布中随机选择 |
| `llama_sampler_init_greedy` | llama-sampler.cpp | 始终选择最高概率 |
| `llama_sampler_init_mirostat` | llama-sampler.cpp | Mirostat v1 自适应熵 |
| `llama_sampler_init_mirostat_v2` | llama-sampler.cpp | Mirostat v2 |
| `llama_sampler_init_penalties` | llama-sampler.cpp | 重复/频率/存在惩罚 |
| `llama_sampler_init_grammar` | llama-sampler.cpp | GBNF 语法约束生成 |
| `llama_sampler_init_dry` | llama-sampler.cpp | DRY (Don't Repeat Yourself) 惩罚 |
| `llama_sampler_init_xtc` | llama-sampler.cpp | XTC 采样器 |
| `llama_sampler_init_infill` | llama-sampler.cpp | 用于代码填充的 token 偏置 |
| `llama_sampler_init_dri` | llama-sampler.cpp | DRI 采样器 |
| `llama_sampler_init_tail_free` | llama-sampler.cpp | 尾自由采样 |
| `llama_sampler_init_eta_cutoff` | llama-sampler.cpp | Eta 截断 |
| `llama_sampler_init_epsilon_cutoff` | llama-sampler.cpp | Epsilon 截断 |
| `llama_sampler_init_branchefield` | llama-sampler.cpp | Branchefield 采样器 |
| `llama_sampler_init_logit_bias` | llama-sampler.cpp | 按 token 的直接 logit 偏置 |

### 如何添加自定义采样器

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
 .name = my_sampler_name,
 .accept = NULL, // optional
 .apply = my_sampler_apply,
 .reset = NULL, // optional
 .free = my_sampler_free,
};

// 3. Create and add to chain
struct llama_sampler * my_sampler = llama_sampler_init(&my_sampler_i, &(my_sampler_ctx){.my_param = 0.5f});
llama_sampler_chain_add(chain, my_sampler);
```

---

## 9.3 模型架构扩展

**位置：** `src/llama-arch.h` — `llm_arch` 枚举 + `llm_tensor_fetch` 注册表

添加新的模型架构需要：

1. 在 `src/llama-arch.h` 中的 `llm_arch` 枚举中添加新的 `LLM_ARCH_*` 枚举值
2. 在 `llm_tensor_fetch` 表中注册该架构的张量名称映射
3. 在 `src/llama-model.cpp` 中通过向 `llm_build_*` switch 添加 case 来实现前向传播
4. 在 `load_arch()` 中基于 GGUF `general.architecture` 元数据添加架构检测
5. 更新 `models/convert_hf_to_gguf.py` 以支持新架构的权重命名约定
