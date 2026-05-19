# vLLM — Extension & Plugin System

## 9.1 Python Entry Point Plugin System

**Location:** `vllm/plugins/__init__.py`

### Plugin Groups

vLLM uses Python `importlib.metadata.entry_points()` for plugin discovery. Four plugin groups are defined:

| Group Name | Loaded In | Purpose |
|-----------|-----------|---------|
| `vllm.general_plugins` | All processes | General-purpose plugins (loaded in API server, engine core, and workers) |
| `vllm.io_processor_plugins` | Process 0 only | I/O processor extensions |
| `vllm.platform_plugins` | All processes | Platform-specific plugins (loaded when `current_platform` is first accessed) |
| `vllm.stat_logger_plugins` | Process 0 only | Custom stats logger implementations |

### Plugin Discovery

```python
def load_plugins_by_group(group: str) -> dict[str, Callable[[], Any]]:
    from importlib.metadata import entry_points
    discovered_plugins = entry_points(group=group)
    # Filter by VLLM_PLUGINS env var if set
    # Load matching plugins and return dict[name, callable]
```

**Access control:**
- `VLLM_PLUGINS` environment variable restricts which plugins are loaded
- If `VLLM_PLUGINS` is not set, all discovered plugins are loaded
- If set, only listed plugin names are loaded

### Creating a Plugin

To create a vLLM plugin:

1. Create a Python package with an entry point in `pyproject.toml`:
```toml
[project.entry-points."vllm.general_plugins"]
my_plugin = "my_package.vllm_plugin:register"
```

2. Implement the registration function:
```python
def register():
    # Register custom components, hooks, etc.
    pass
```

3. Install the package: `pip install my_plugin_package`

---

## 9.2 Attention Backend Registry

**Location:** `vllm/v1/attention/backends/registry.py`

### Mechanism

Attention backends are registered by name and selected at startup based on hardware, model architecture, and user configuration.

**Available backends:**

| Backend | File | Hardware | Notes |
|---------|------|----------|-------|
| `flash_attn` | `flash_attn.py` | NVIDIA GPU | FlashAttention-2, default for CUDA |
| `flashinfer` | `flashinfer.py` | NVIDIA GPU | FlashInfer library |
| `rocm_attn` | `rocm_attn.py` | AMD GPU | ROCm attention |
| `rocm_aiter_fa` | `rocm_aiter_fa.py` | AMD GPU | AITER FlashAttention for ROCm |
| `cpu_attn` | `cpu_attn.py` | CPU | x86 CPU attention |
| `triton_attn` | `triton_attn.py` | NVIDIA GPU | Triton-based attention |
| `tree_attn` | `tree_attn.py` | NVIDIA GPU | Tree attention for speculative decoding |
| `flex_attention` | `flex_attention.py` | NVIDIA GPU | PyTorch flex attention (SDPA) |
| `linear_attn` | `linear_attn.py` | Any | Linear attention models |
| `mamba1_attn` | `mamba1_attn.py` | Any | Mamba 1 SSM |
| `mamba2_attn` | `mamba2_attn.py` | Any | Mamba 2 SSM |
| `mamba_attn` | `mamba_attn.py` | Any | Mamba (generic) |
| `mla/*` | `mla/` | NVIDIA GPU | Multi-head Latent Attention (DeepSeek) |
| `short_conv_attn` | `short_conv_attn.py` | Any | Short convolution attention |

### Selection Logic

The platform-specific registry selects the best available backend based on:
1. User override via `VLLM_ATTENTION_BACKEND` env var
2. Hardware platform (CUDA → flash_attn, ROCm → rocm_attn, CPU → cpu_attn)
3. Model architecture requirements (Mamba → mamba_attn, MLA → mla)

---

## 9.3 Structured Output Backends

**Location:** `vllm/v1/structured_output/`

### Mechanism

`StructuredOutputManager` selects a backend at initialization based on `StructuredOutputsConfig`:

| Backend | File | Description |
|---------|------|-------------|
| `xgrammar` | `backend_xgrammar.py` | XGrammar engine (C++ based, fast) |
| `outlines` | `backend_outlines.py` | Outlines library |
| `guidance` | `backend_guidance.py` | Guidance library |
| `lm-format-enforcer` | `backend_lm_format_enforcer.py` | LM Format Enforcer |

**Interface:** `StructuredOutputBackend` — abstract class with:
- `compile_grammar()` — compile a structured output spec into a grammar object
- `allocate_token bitmask()` — get a bitmask for the next token to enforce grammar constraints

**Async compilation:** Grammar compilation can happen asynchronously in a thread pool (disabled for `external_launcher` mode due to determinism requirements across TP ranks).

---

## 9.4 Model Architecture Registry

**Location:** `vllm/model_executor/models/` (100+ model implementations)

### Mechanism

Each model file registers itself via `_MODELS` dict or auto-discovery. New models are added by:

1. Creating a new file in `vllm/model_executor/models/`
2. Implementing the model class following the vLLM conventions:
   - Inherit from `nn.Module`
   - Use vLLM's custom linear layers (`ColumnParallelLinear`, `RowParallelLinear`, etc.)
   - Implement `forward()` with `attn_metadata` parameter
3. Register in `__init__.py`

---

## 9.5 Tool Parser Extension

**Location:** `vllm/entrypoints/tool_parsers/`

### Mechanism

Tool parsers convert model output into structured tool calls for the chat completions and responses APIs.

**Registration:** `ToolParserManager.import_tool_parser()` — dynamically imports a custom tool parser.

**Built-in parsers:** Various model-specific parsers (e.g., for function calling formats).

**Custom parser:** Can be loaded via `--tool-parser-plugin` CLI flag.

---

## 9.6 Reasoning Parser Extension

**Location:** `vllm/reasoning/`

### Mechanism

Reasoning parsers handle "thinking" output from reasoning models (e.g., DeepSeek-R1).

**Registration:** `ReasoningParserManager.import_reasoning_parser()` — dynamically imports a custom parser.

**Custom parser:** Can be loaded via `--reasoning-parser-plugin` CLI flag.

---

## 9.7 Speculative Decoding Extensions

**Location:** `vllm/v1/spec_decode/`

Multiple speculative decoding strategies are implemented as pluggable components:

| Strategy | File | Description |
|----------|------|-------------|
| n-gram | `ngram_proposer.py`, `ngram_proposer_gpu.py` | n-gram lookup for draft tokens |
| Eagle | `eagle.py` | Eagle speculative decoding head |
| Medusa | `medusa.py` | Medusa multi-head speculative decoding |
| Draft model | `draft_model.py` | Small draft model for speculative decoding |
| Suffix decoding | `suffix_decoding.py` | Suffix-based draft token generation |
| dFlash | `dflash.py` | Flash-based speculative decoding |

---

## 9.8 Quantization Method Registry

**Location:** `vllm/model_executor/layers/quantization/`

20+ quantization methods are supported as pluggable backends:

| Method | File | Description |
|--------|------|-------------|
| FP8 | `fp8.py` | FP8 weight/KV cache quantization |
| AWQ | `awq.py`, `awq_marlin.py` | Activation-aware weight quantization |
| GPTQ | `gptq.py`, `gptq_marlin.py` | GPTQ weight quantization |
| BitsAndBytes | `bitsandbytes.py` | NF4/int8 BnB quantization |
| GGUF | `gguf.py` | GGUF quantized formats |
| FBGEMM FP8 | `fbgemm_fp8.py` | FBGEMM FP8 kernels |
| ModelOpt | `modelopt.py` | NVIDIA ModelOpt quantization |
| Compressed Tensors | `compressed_tensors/` | Neural Magic compressed format |
| INT8 Experts | `experts_int8.py` | INT8 MoE expert quantization |
| KV Cache Quant | `kv_cache.py` | KV cache quantization (FP8, INT8, NVFP4) |

Each quantization method implements a `QuantizationConfig` subclass that defines how weights are loaded and which custom kernels are used.
