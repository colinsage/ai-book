# llama.cpp — Storage File Analysis

## GGUF Model File (`.gguf`)

**Location:** User-specified path, e.g. `./models/llama-3.gguf`
**Format Definition:** `ggml/include/gguf.h` (lines 1-31)
**Implementation:** `ggml/src/gguf.c`

### File Structure

```
[4 bytes]   Magic: 0x47475546 ("GGUF")
[4 bytes]   Version: uint32 (currently 3)
[8 bytes]   tensor_count: int64 — number of tensors in file
[8 bytes]   kv_count: int64 — number of key-value metadata pairs

--- Key-Value Metadata Section ---
For each KV pair:
  [8 bytes]   key_length: uint64
  [N bytes]   key: UTF-8 string (no null terminator)
  [4 bytes]   value_type: int32 (gguf_type enum)
  If value_type == GGUF_TYPE_ARRAY:
    [4 bytes]   array_elem_type: int32 (gguf_type)
    [8 bytes]   array_count: uint64
    [N bytes]   array elements (concatenated binary representation)
  Else:
    [N bytes]   value (binary representation per type)

--- Tensor Descriptor Section ---
For each tensor:
  [8 bytes]   name_length: uint64
  [N bytes]   name: UTF-8 string (no null terminator)
  [4 bytes]   n_dimensions: uint32
  [8*n_dim]   dimensions: int64 array (shape per dimension)
  [4 bytes]   data_type: int32 (ggml_type enum — F32, F16, Q4_0, etc.)
  [8 bytes]   data_offset: uint64 (offset into tensor data blob)

--- Tensor Data Blob (aligned) ---
Padding to alignment boundary (default 32 bytes, configurable via
"general.alignment" KV key)
[N bytes]   Raw tensor data (each tensor starts at its aligned offset)
```

### GGUF Value Types (gguf_type)

| Type ID | Name | Size |
|---------|------|------|
| 0 | UINT8 | 1 byte |
| 1 | INT8 | 1 byte |
| 2 | UINT16 | 2 bytes |
| 3 | INT16 | 2 bytes |
| 4 | UINT32 | 4 bytes |
| 5 | INT32 | 4 bytes |
| 6 | FLOAT32 | 4 bytes |
| 7 | BOOL | 1 byte (stored as int8) |
| 8 | STRING | 8-byte length + chars |
| 9 | ARRAY | type + count + elements |
| 10 | UINT64 | 8 bytes |
| 11 | INT64 | 8 bytes |
| 12 | FLOAT64 | 8 bytes |

### Standard Metadata Keys

| Key | Type | Purpose |
|-----|------|---------|
| `general.architecture` | STRING | Model architecture identifier (e.g., "llama", "mistral", "gemma") |
| `general.name` | STRING | Human-readable model name |
| `general.alignment` | UINT32 | Custom alignment for tensor data (default: 32) |
| `general.file_type` | UINT32 | Quantization type identifier |
| `llama.context_length` | UINT32 | Training context window size |
| `llama.embedding_length` | UINT32 | Embedding dimension |
| `llama.block_count` | UINT32 | Number of transformer layers |
| `llama.attention.head_count` | UINT32 | Number of attention heads |
| `llama.attention.head_count_kv` | UINT32 | Number of KV heads |
| `llama.attention.layer_norm_rms_epsilon` | FLOAT32 | RMS norm epsilon |
| `llama.rope.freq_base` | FLOAT32 | RoPE base frequency |
| `tokenizer.ggml.model` | STRING | Tokenizer type ("llama", "gpt2", "t5", etc.) |
| `tokenizer.ggml.tokens` | ARRAY(STRING) | Vocabulary tokens |
| `tokenizer.ggml.scores` | ARRAY(FLOAT32) | Token scores |
| `tokenizer.ggml.token_type` | ARRAY(INT32) | Token type attributes |

### Design Intent

GGUF is a **self-describing binary format** that bundles model weights with all hyperparameters and tokenizer data needed for inference. Key design choices:

1. **No separate config files** — All metadata (architecture, dimensions, tokenizer) is embedded in the same file as weights
2. **Memory-mappable** — Tensor data is placed at aligned offsets so the entire file can be `mmap()`'d and tensors accessed directly without copying. This enables:
   - Lazy loading (only pages touched by inference are read)
   - Multi-process sharing (OS shares physical pages between processes using the same model file)
   - Zero-copy weight access on CPU
3. **Extensible metadata** — KV pairs use string keys with typed values, allowing new metadata without format breaking changes
4. **Array support** — Tokenizer vocabularies are stored as typed arrays for efficiency
5. **Alignment configurable** — Default 32-byte alignment works for most SIMD; `general.alignment` KV allows customization

### Loading Process (src/llama-model.cpp)

1. `gguf_init_from_file()` parses header, KV pairs, and tensor descriptors
2. `llama_model::load_arch()` reads `general.architecture` to determine model type
3. `llama_model::load_hparams()` reads architecture-specific hyperparameters from KV pairs
4. `llama_model::load_vocab()` reads tokenizer data from KV pairs
5. `llama_model::load_tensors()` maps tensor data into backend buffers:
   - CPU: Uses `llama_mmap` for memory-mapped access
   - GPU: Copies tensors to VRAM via `ggml_backend_buffer`

## Model Conversion Scripts

**Location:** `models/` directory (Python scripts)

| Script | Purpose |
|--------|---------|
| `convert_hf_to_gguf.py` | Convert HuggingFace models to GGUF |
| `convert_hf_to_gguf.py --quantize` | Convert + quantize in one step |
| `convert_llama_ggml_to_gguf.py` | Legacy GGML → GGUF conversion |
| `convert_ggml_to_gguf.py` | Old format migration |

These scripts read PyTorch safetensors/ckpt files, extract hyperparameters, and write the GGUF binary format using the `gguf` Python package.

## Session State Files

The server can save/load KV cache state via the `/slots` endpoint:

**Format:** GGUF-compatible (uses gguf KV pairs to store session metadata + raw KV cache data)
**Purpose:** Allows pausing and resuming long conversations without re-processing the prompt
