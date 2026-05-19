# SGLang — Extension & Plugin System

## Overview

SGLang provides several extension mechanisms, though it does not have a formal plugin system with a unified registry. Instead, extensions are built into specific subsystems through inheritance, configuration, and registration patterns.

---

## Model Backend Extension

### Model Implementations

**Location:** `python/sglang/srt/models/`

**Mechanism:** Each model architecture is implemented as a Python class inheriting from a common interface. The model class is selected by matching the HuggingFace `architectures` field in `config.json`.

**Adding a New Model:**

1. Create a new file in `python/sglang/srt/models/` (e.g., `my_model.py`)
2. Implement the model class following the pattern of existing models (e.g., `llama.py`)
3. Register it in `python/sglang/srt/models/registry.py` by adding the architecture name

**Example — LlamaModel:**
```python
# models/llama.py
class LlamaModel(nn.Module):
    def __init__(self, config, ...):
        # Build transformer layers
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config, ...) for _ in range(config.num_hidden_layers)
        ])
    
    def forward(self, input_ids, positions, ...):
        # Custom forward pass with SGLang optimizations
        ...
```

The model registry automatically maps `config.architectures` to the correct implementation class.

---

## Attention Backend Extension

**Location:** `python/sglang/srt/layers/attention/`

**Available Backends:**

| Backend | Class | Purpose |
|---------|-------|---------|
| FlashInfer | `FlashInferAttnBackend` | Default, optimized for A100/H100 |
| Triton | `TritonAttnBackend` | Fallback, works on any GPU |
| FlashAttention | `FlashAttentionBackend` | For older GPUs |
| TritonWithFLA | `TritonWithFLABackend` | FlashLinearAttention for long context |
| MLA | `MLABackend` | Multi-head Latent Attention (DeepSeek) |

**Selection:** Controlled by `--attention-backend` or auto-detected based on GPU architecture.

**Adding a New Backend:**

1. Create a new file in `python/sglang/srt/layers/attention/`
2. Implement the backend class with the required interface methods:
   - `init_forward_metadata()` — Prepare batch metadata
   - `forward()` — Execute attention computation
3. Register in the attention backend factory

---

## LoRA Adapter Extension

**Location:** `python/sglang/srt/lora/`

**Mechanism:** Dynamic LoRA adapter loading and serving. Multiple LoRA adapters can be served concurrently.

**Key Classes:**

| Class | Location | Purpose |
|-------|----------|---------|
| `LoRARef` | lora_registry.py:27 | Reference to a LoRA adapter (path, ID) |
| `LoRARegistry` | lora_registry.py:54 | Registry of loaded adapters |
| `LoRALayer` | lora.py:38 | Base LoRA layer with A/B matrices |
| `LoRAAdapter` | lora.py:48 | Full adapter module wrapping base model |
| `LoRAConfig` | lora_config.py:25 | Adapter configuration |
| `LoRAOverlapLoader` | lora_overlap_loader.py:21 | Async adapter loading with overlap |
| `EvictionPolicy` | eviction_policy.py:28 | Abstract base for adapter eviction |

**Using LoRA:**

```python
# Load adapter at startup
--lora-paths my_adapter=/path/to/adapter

# Or dynamically via API
engine.update_weights_from_disk("/path/to/new_adapter")
```

**Eviction Policies:**
- `LRUStrategy` — Evict least recently used adapter
- `LFUStrategy` — Evict least frequently used
- Custom policies can be added by subclassing `EvictionPolicy`

**Design Intent:** LoRA support is a first-class extension point. Adapters are loaded into GPU memory on demand and can be evicted when memory is needed. The `LoRAOverlapLoader` overlaps adapter loading with inference to minimize latency spikes.

---

## Grammar/Constrained Generation Extension

**Location:** `python/sglang/srt/constrained/`

**Mechanism:** Pluggable grammar backends for structured generation (JSON, regex, context-free grammar).

**Available Backends:**

| Backend | Class | Library | Purpose |
|---------|-------|---------|---------|
| XGrammar | `XGrammarGrammarBackend` | xgrammar | Fast CFG-based grammar |
| Guidance | `GuidanceBackend` | llguidance | LLGuidance-based grammar |
| Reasoner | `ReasonerGrammarBackend` | custom | Reasoning-aware generation |

**Adding a New Backend:**

1. Subclass `BaseGrammarBackend` (base_grammar_backend.py:130)
2. Implement `compile_grammar()` and `allocate_token_mask()` methods
3. Register via `--grammar-backend` flag

---

## KV Cache Eviction Policy Extension

**Location:** `python/sglang/srt/mem_cache/evict_policy.py`

**Mechanism:** Strategy pattern for KV cache page eviction in the radix cache.

**Available Policies:**

| Policy | Class | Behavior |
|--------|-------|----------|
| LRU | `LRUStrategy` | Evict least recently accessed |
| LFU | `LFUStrategy` | Evict least frequently used |
| FIFO | `FIFOStrategy` | Evict oldest |
| MRU | `MRUStrategy` | Evict most recently used |
| FILO | `FILOStrategy` | Evict newest |
| Priority | `PriorityStrategy` | Evict by request priority |
| SLRU | `SLRUStrategy` | Segmented LRU with protected/probationary segments |

**Adding a New Policy:**

1. Subclass `EvictionStrategy` (evict_policy.py:10)
2. Implement `select_victim()` method
3. Register the policy name in `RadixCache.__init__()`

---

## HiCache Storage Extension

**Location:** `python/sglang/srt/mem_cache/hicache_storage.py`

**Mechanism:** Abstract base class for hierarchical cache storage backends.

**Available Backends:**

| Backend | Class | Purpose |
|---------|-------|---------|
| HiCacheFile | `HiCacheFile` | File-based storage for KV cache offloading |

**Adding a New Storage Backend:**

1. Subclass `HiCacheStorage` (hicache_storage.py:124)
2. Implement `attach()`, `detach()`, `read()`, `write()` methods
3. Register via `--hicache-storage-backend` configuration

---

## Weight Loader Extension

**Location:** `python/sglang/srt/layers/loader.py`

**Mechanism:** Pluggable model weight loaders via `BaseModelLoader` abstract class.

**Available Loaders:**
- `DefaultModelLoader` — Standard safetensors
- `LayeredModelLoader` — Memory-efficient layer-by-layer
- `GGUFModelLoader` — GGUF format
- `BitsAndBytesModelLoader` — 4-bit/8-bit quantization
- `RemoteModelLoader` — Remote weight loading
- `ModelOptModelLoader` — NVIDIA ModelOpt

**Adding a New Loader:**

1. Subclass `BaseModelLoader` (loader.py:280)
2. Implement `load_model()` method
3. Register in the loader selection logic

---

## SGLang Native Sampling Extension

**Location:** `python/sglang/srt/sampling/`

SGLang provides custom sampling implementations optimized for batched GPU execution:
- Top-p (nucleus) sampling
- Top-k sampling
- Min-p sampling
- Frequency/presence penalty
- Custom logit processors via `--custom-logit-processor`

---

## Summary of Extension Points

| Extension Point | Mechanism | Register Via |
|----------------|-----------|-------------|
| Model architecture | Python class + registry | `models/registry.py` |
| Attention backend | Python class | `--attention-backend` flag |
| LoRA adapters | Dynamic loading + registry | `--lora-paths` or API |
| Grammar backend | Python class | `--grammar-backend` flag |
| Cache eviction policy | Strategy pattern | `--eviction-policy` flag |
| HiCache storage | Abstract base class | `--hicache-storage-backend` |
| Weight loader | Abstract base class | Auto-detection by model format |
| Custom logit processor | Python class | `--custom-logit-processor` |
