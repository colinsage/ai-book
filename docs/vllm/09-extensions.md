# vLLM — 扩展与插件系统

## 9.1 Python Entry Point 插件系统

**位置：** `vllm/plugins/__init__.py`

### 插件组 (Plugin Groups)

vLLM 使用 Python 的 `importlib.metadata.entry_points()` 进行插件发现 (plugin discovery)。定义了四个插件组：

| 组名 | 加载位置 | 用途 |
|-----------|-----------|---------|
| `vllm.general_plugins` | 所有进程 | 通用插件（在 API server、engine core 和 workers 中加载） |
| `vllm.io_processor_plugins` | 仅进程 0 | I/O 处理器扩展 |
| `vllm.platform_plugins` | 所有进程 | 平台特定插件（首次访问 `current_platform` 时加载） |
| `vllm.stat_logger_plugins` | 仅进程 0 | 自定义统计日志记录器实现 |

### 插件发现 (Plugin Discovery)

```python
def load_plugins_by_group(group: str) -> dict[str, Callable[[], Any]]:
 from importlib.metadata import entry_points
 discovered_plugins = entry_points(group=group)
 # Filter by VLLM_PLUGINS env var if set
 # Load matching plugins and return dict[name, callable]
```

**访问控制：**
- `VLLM_PLUGINS` 环境变量限制加载哪些插件
- 如果未设置 `VLLM_PLUGINS`，则加载所有已发现的插件
- 如果已设置，则仅加载列出的插件名称

### 创建插件 (Creating a Plugin)

要创建 vLLM 插件：

1. 在 `pyproject.toml` 中创建带有 entry point 的 Python 包：
```toml
[project.entry-points."vllm.general_plugins"]
my_plugin = "my_package.vllm_plugin:register"
```

2. 实现注册函数：
```python
def register():
 # Register custom components, hooks, etc.
 pass
```

3. 安装该包：`pip install my_plugin_package`

---

## 9.2 Attention Backend 注册表

**位置：** `vllm/v1/attention/backends/registry.py`

### 机制 (Mechanism)

Attention backend 通过名称注册，并在启动时根据硬件 (hardware)、模型架构 (model architecture) 和用户配置 (user configuration) 进行选择。

**可用 backend：**

| Backend | 文件 | 硬件 | 说明 |
|---------|------|----------|-------|
| `flash_attn` | `flash_attn.py` | NVIDIA GPU | FlashAttention-2，CUDA 的默认选择 |
| `flashinfer` | `flashinfer.py` | NVIDIA GPU | FlashInfer 库 |
| `rocm_attn` | `rocm_attn.py` | AMD GPU | ROCm attention |
| `rocm_aiter_fa` | `rocm_aiter_fa.py` | AMD GPU | AITER FlashAttention for ROCm |
| `cpu_attn` | `cpu_attn.py` | CPU | x86 CPU attention |
| `triton_attn` | `triton_attn.py` | NVIDIA GPU | 基于 Triton 的 attention |
| `tree_attn` | `tree_attn.py` | NVIDIA GPU | 用于投机解码 (speculative decoding) 的树形 attention |
| `flex_attention` | `flex_attention.py` | NVIDIA GPU | PyTorch flex attention (SDPA) |
| `linear_attn` | `linear_attn.py` | 任意 | 线性 attention 模型 |
| `mamba1_attn` | `mamba1_attn.py` | 任意 | Mamba 1 SSM |
| `mamba2_attn` | `mamba2_attn.py` | 任意 | Mamba 2 SSM |
| `mamba_attn` | `mamba_attn.py` | 任意 | Mamba（通用） |
| `mla/*` | `mla/` | NVIDIA GPU | Multi-head Latent Attention (DeepSeek) |
| `short_conv_attn` | `short_conv_attn.py` | 任意 | 短卷积 attention (short convolution attention) |

### 选择逻辑 (Selection Logic)

平台特定的注册表根据以下条件选择最佳可用 backend：
1. 用户通过 `VLLM_ATTENTION_BACKEND` 环境变量覆盖
2. 硬件平台（CUDA → flash_attn，ROCm → rocm_attn，CPU → cpu_attn）
3. 模型架构需求（Mamba → mamba_attn，MLA → mla）

---

## 9.3 结构化输出 Backend (Structured Output Backends)

**位置：** `vllm/v1/structured_output/`

### 机制 (Mechanism)

`StructuredOutputManager` 在初始化时根据 `StructuredOutputsConfig` 选择 backend：

| Backend | 文件 | 描述 |
|---------|------|-------------|
| `xgrammar` | `backend_xgrammar.py` | XGrammar 引擎（基于 C++，速度快） |
| `outlines` | `backend_outlines.py` | Outlines 库 |
| `guidance` | `backend_guidance.py` | Guidance 库 |
| `lm-format-enforcer` | `backend_lm_format_enforcer.py` | LM Format Enforcer |

**接口：** `StructuredOutputBackend` — 抽象类，包含：
- `compile_grammar()` — 将结构化输出规范 (structured output spec) 编译为 grammar 对象
- `allocate_token bitmask()` — 获取下一个 token 的位掩码 (bitmask) 以强制执行 grammar 约束

**异步编译 (Async compilation)：** grammar 编译可以在线程池中异步进行（由于跨 TP rank 的确定性要求，在 `external_launcher` 模式下禁用）。

---

## 9.4 模型架构注册表 (Model Architecture Registry)

**位置：** `vllm/model_executor/models/`（100+ 个模型实现）

### 机制 (Mechanism)

每个模型文件通过 `_MODELS` 字典或自动发现 (auto-discovery) 注册自身。添加新模型的方式：

1. 在 `vllm/model_executor/models/` 中创建新文件
2. 按照 vLLM 规范实现模型类：
   - 继承 `nn.Module`
   - 使用 vLLM 的自定义线性层（`ColumnParallelLinear`、`RowParallelLinear` 等）
   - 实现带有 `attn_metadata` 参数的 `forward()` 方法
3. 在 `__init__.py` 中注册

---

## 9.5 工具解析器扩展 (Tool Parser Extension)

**位置：** `vllm/entrypoints/tool_parsers/`

### 机制 (Mechanism)

工具解析器 (tool parser) 将模型输出转换为结构化的工具调用 (tool call)，用于 chat completions 和 responses API。

**注册：** `ToolParserManager.import_tool_parser()` — 动态导入自定义工具解析器。

**内置解析器：** 各种模型特定的解析器（例如用于函数调用格式）。

**自定义解析器：** 可通过 `--tool-parser-plugin` CLI 标志加载。

---

## 9.6 推理解析器扩展 (Reasoning Parser Extension)

**位置：** `vllm/reasoning/`

### 机制 (Mechanism)

推理解析器 (reasoning parser) 处理推理模型（如 DeepSeek-R1）的"思考" (thinking) 输出。

**注册：** `ReasoningParserManager.import_reasoning_parser()` — 动态导入自定义解析器。

**自定义解析器：** 可通过 `--reasoning-parser-plugin` CLI 标志加载。

---

## 9.7 投机解码扩展 (Speculative Decoding Extensions)

**位置：** `vllm/v1/spec_decode/`

多种投机解码 (speculative decoding) 策略以可插拔组件 (pluggable component) 的形式实现：

| 策略 | 文件 | 描述 |
|----------|------|-------------|
| n-gram | `ngram_proposer.py`, `ngram_proposer_gpu.py` | n-gram 查找生成 draft token |
| Eagle | `eagle.py` | Eagle 投机解码头 |
| Medusa | `medusa.py` | Medusa 多头投机解码 |
| Draft model | `draft_model.py` | 用于投机解码的小型 draft 模型 |
| Suffix decoding | `suffix_decoding.py` | 基于后缀的 draft token 生成 |
| dFlash | `dflash.py` | 基于 Flash 的投机解码 |

---

## 9.8 量化方法注册表 (Quantization Method Registry)

**位置：** `vllm/model_executor/layers/quantization/`

支持 20+ 种量化方法作为可插拔 backend：

| 方法 | 文件 | 描述 |
|--------|------|-------------|
| FP8 | `fp8.py` | FP8 权重/KV 缓存量化 |
| AWQ | `awq.py`, `awq_marlin.py` | 感知激活的权重量化 (Activation-aware weight quantization) |
| GPTQ | `gptq.py`, `gptq_marlin.py` | GPTQ 权重量化 |
| BitsAndBytes | `bitsandbytes.py` | NF4/int8 BnB 量化 |
| GGUF | `gguf.py` | GGUF 量化格式 |
| FBGEMM FP8 | `fbgemm_fp8.py` | FBGEMM FP8 内核 |
| ModelOpt | `modelopt.py` | NVIDIA ModelOpt 量化 |
| Compressed Tensors | `compressed_tensors/` | Neural Magic 压缩格式 |
| INT8 Experts | `experts_int8.py` | INT8 MoE 专家量化 |
| KV Cache Quant | `kv_cache.py` | KV 缓存量化（FP8、INT8、NVFP4） |

每种量化方法实现一个 `QuantizationConfig` 子类，定义权重如何加载以及使用哪些自定义内核 (custom kernel)。
