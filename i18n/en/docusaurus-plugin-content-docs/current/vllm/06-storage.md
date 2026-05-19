# vLLM — Storage File Analysis

## Model Weight Files

### SafeTensors (Primary Format)

**Location:** Loaded from HuggingFace Hub or local path specified by `--model`

**Handler:** `vllm/model_executor/model_loader/default_loader.py`

**Format:** HuggingFace SafeTensors — memory-mappable, zero-copy format with JSON metadata header.

**Loading Process:**
1. Download/locate model files (sharded SafeTensors)
2. Memory-map each shard with `mmap`
3. For tensor parallelism: load only the relevant shard for each TP rank
4. For quantized models: apply quantization-aware loading (AWQ, GPTQ, FP8, etc.)

### GGUF Format

**Handler:** `vllm/model_executor/model_loader/gguf_loader.py`

**Format:** GGUF binary format with magic `0x47475546`:
```
[4 bytes] Magic: "GGUF"
[4 bytes] Version: uint32
[8 bytes] tensor_count: uint64
[8 bytes] kv_count: uint64
[N bytes] Key-value metadata pairs
[N bytes] Tensor descriptors
[aligned] Raw tensor data (32-byte aligned)
```

**Design Intent:** Self-describing format bundling weights with hyperparameters, enabling memory-mapped access without separate config files.

### BitsAndBytes (NF4) Format

**Handler:** `vllm/model_executor/model_loader/bitsandbytes_loader.py`

Supports 4-bit quantized models with nested quantization metadata.

### Tensorizer Format

**Handler:** `vllm/model_executor/model_loader/tensorizer_loader.py`

High-performance serialized tensor format for fast loading from S3 or local storage.

### RunAI Streamer

**Handler:** `vllm/model_executor/model_loader/runai_streamer_loader.py`

Streaming model loader for faster startup with progressive weight loading.

### Sharded State

**Handler:** `vllm/model_executor/model_loader/sharded_state_loader.py`

Supports loading/saving distributed model state for checkpointing and resumption.

---

## KV Cache Persistence

vLLM does **not** persist KV cache to disk by default. KV cache lives entirely in GPU VRAM (or CPU RAM for CPU backend) and is freed on shutdown.

**KV Cache Events** (`vllm/distributed/kv_events/`): An event-based system that emits allocation/free events. External systems can use this to implement KV cache persistence or cross-instance sharing via the **KV Connector** framework (`vllm/distributed/kv_transfer/`).

---

## Configuration Files

### VllmConfig (`vllm/config/`)

Comprehensive configuration object assembled from CLI args, environment variables, and model metadata:

| Config Subclass | File | Purpose |
|----------------|------|---------|
| `ModelConfig` | `config/model.py` | Model path, max length, dtype, tokenizer |
| `CacheConfig` | `config/cache.py` | Block size, swap space, KV cache dtype |
| `SchedulerConfig` | `config/scheduler.py` | Max sequences, max tokens, scheduling policy |
| `ParallelConfig` | `config/parallel.py` | TP/PP/DP size, distributed backend |
| `SpeculativeConfig` | `config/speculative.py` | Speculative decoding settings |
| `LoRAConfig` | `config/lora.py` | LoRA adapter settings |
| `ObservabilityConfig` | `config/observability.py` | Tracing, metrics, KV events |
| `StructuredOutputsConfig` | `config/structured_outputs.py` | Grammar backend selection |

### Environment Variables (`vllm/envs.py`)

200+ environment variables control every aspect of vLLM behavior. Key examples:

| Variable | Default | Purpose |
|----------|---------|---------|
| `VLLM_API_KEY` | None | API authentication token |
| `VLLM_CACHE_ROOT` | `~/.cache/vllm` | Cache directory for downloaded models |
| `VLLM_LOG_STATS_INTERVAL` | 10.0 | Seconds between stats logs |
| `VLLM_ENGINE_ITERATION_TIMEOUT_S` | 60 | Max seconds per engine step |
| `VLLM_ENGINE_READY_TIMEOUT_S` | 600 | Max seconds to wait for engine startup |
| `VLLM_PLUGINS` | None | Comma-separated list of allowed plugins |
| `VLLM_USE_PRECOMPILED` | False | Use precompiled C++ extensions |

---

## LoRA Adapter Storage

**Handler:** `vllm/lora/model_manager.py`, `vllm/lora/worker_manager.py`

LoRA adapters are loaded from disk on demand:
- Stored as SafeTensors files with specific key naming conventions
- Loaded via `PEFT` helper (`vllm/lora/peft_helper.py`)
- Weights are pinned in GPU memory when loaded
- Supports dynamic loading/unloading via API endpoints
