# llama.cpp — 存储文件分析

## GGUF 模型文件（`.gguf`）

**位置：** 用户指定路径，例如 `./models/llama-3.gguf`
**格式定义：** `ggml/include/gguf.h`（第 1-31 行）
**实现：** `ggml/src/gguf.c`

### 文件结构

```
[4 bytes] Magic: 0x47475546 ("GGUF")
[4 bytes] Version: uint32 (currently 3)
[8 bytes] tensor_count: int64 — number of tensors in file
[8 bytes] kv_count: int64 — number of key-value metadata pairs

--- Key-Value Metadata Section ---
For each KV pair:
 [8 bytes] key_length: uint64
 [N bytes] key: UTF-8 string (no null terminator)
 [4 bytes] value_type: int32 (gguf_type enum)
 If value_type == GGUF_TYPE_ARRAY:
 [4 bytes] array_elem_type: int32 (gguf_type)
 [8 bytes] array_count: uint64
 [N bytes] array elements (concatenated binary representation)
 Else:
 [N bytes] value (binary representation per type)

--- Tensor Descriptor Section ---
For each tensor:
 [8 bytes] name_length: uint64
 [N bytes] name: UTF-8 string (no null terminator)
 [4 bytes] n_dimensions: uint32
 [8*n_dim] dimensions: int64 array (shape per dimension)
 [4 bytes] data_type: int32 (ggml_type enum — F32, F16, Q4_0, etc.)
 [8 bytes] data_offset: uint64 (offset into tensor data blob)

--- Tensor Data Blob (aligned) ---
Padding to alignment boundary (default 32 bytes, configurable via
"general.alignment" KV key)
[N bytes] Raw tensor data (each tensor starts at its aligned offset)
```

### GGUF 值类型（gguf_type）

| Type ID | Name | 大小 |
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

### 标准元数据键

| 键 | 类型 | 用途 |
|-----|------|------|
| `general.architecture` | STRING | 模型架构标识符（例如 "llama"、"mistral"、"gemma"） |
| `general.name` | STRING | 人类可读的模型名称 |
| `general.alignment` | UINT32 | 张量数据的自定义对齐方式（默认：32） |
| `general.file_type` | UINT32 | 量化类型标识符 |
| `llama.context_length` | UINT32 | 训练上下文窗口大小 |
| `llama.embedding_length` | UINT32 | 嵌入维度 |
| `llama.block_count` | UINT32 | Transformer 层数 |
| `llama.attention.head_count` | UINT32 | 注意力头数量 |
| `llama.attention.head_count_kv` | UINT32 | KV 头数量 |
| `llama.attention.layer_norm_rms_epsilon` | FLOAT32 | RMS 归一化 epsilon |
| `llama.rope.freq_base` | FLOAT32 | RoPE 基础频率 |
| `tokenizer.ggml.model` | STRING | 分词器类型（"llama"、"gpt2"、"t5" 等） |
| `tokenizer.ggml.tokens` | ARRAY(STRING) | 词汇表 token |
| `tokenizer.ggml.scores` | ARRAY(FLOAT32) | Token 分数 |
| `tokenizer.ggml.token_type` | ARRAY(INT32) | Token 类型属性 |

### 设计意图

GGUF 是一种**自描述二进制格式**，将模型权重与推理所需的所有超参数和分词器数据打包在一起。关键设计选择：

1. **无需单独的配置文件** — 所有元数据（架构、维度、分词器）都嵌入在权重所在的同一文件中
2. **支持内存映射** — 张量数据放置在对齐的偏移位置，因此整个文件可以通过 `mmap()` 映射，无需复制即可直接访问张量。这实现了：
   - 惰性加载（仅读取推理访问的页面）
   - 多进程共享（操作系统在使用同一模型文件的进程之间共享物理页面）
   - CPU 上的零拷贝权重访问
3. **可扩展的元数据** — KV 对使用字符串键和类型化值，允许添加新元数据而不破坏格式兼容性
4. **数组支持** — 分词器词汇表以类型化数组存储以提高效率
5. **对齐可配置** — 默认的 32 字节对齐适用于大多数 SIMD；`general.alignment` KV 键允许自定义

### 加载过程（src/llama-model.cpp）

1. `gguf_init_from_file()` 解析头部、KV 对和张量描述符
2. `llama_model::load_arch()` 读取 `general.architecture` 以确定模型类型
3. `llama_model::load_hparams()` 从 KV 对中读取架构特定的超参数
4. `llama_model::load_vocab()` 从 KV 对中读取分词器数据
5. `llama_model::load_tensors()` 将张量数据映射到后端缓冲区：
   - CPU：使用 `llama_mmap` 进行内存映射访问
   - GPU：通过 `ggml_backend_buffer` 将张量复制到 VRAM

## 模型转换脚本

**位置：** `models/` 目录（Python 脚本）

| 脚本 | 用途 |
|------|------|
| `convert_hf_to_gguf.py` | 将 HuggingFace 模型转换为 GGUF |
| `convert_hf_to_gguf.py --quantize` | 一步完成转换 + 量化 |
| `convert_llama_ggml_to_gguf.py` | 旧版 GGML → GGUF 转换 |
| `convert_ggml_to_gguf.py` | 旧格式迁移 |

这些脚本读取 PyTorch safetensors/ckpt 文件，提取超参数，并使用 `gguf` Python 包写入 GGUF 二进制格式。

## 会话状态文件

服务器可以通过 `/slots` 端点保存/加载 KV 缓存状态：

**格式：** GGUF 兼容（使用 gguf KV 对存储会话元数据 + 原始 KV 缓存数据）
**用途：** 允许暂停和恢复长对话，无需重新处理提示
