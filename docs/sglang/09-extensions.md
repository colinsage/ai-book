# 扩展与插件系统

## 9.1 概述

SGLang 提供了多种扩展机制，不过它并没有一个带有统一注册表的正式插件系统。相反，扩展是通过继承、配置和注册模式内置到特定子系统中的。

---

## 9.2 模型后端扩展

### 模型实现

**位置：** `python/sglang/srt/models/`

**机制：** 每个模型架构实现为一个继承自公共接口的 Python 类。模型类通过匹配 `config.json` 中的 HuggingFace `architectures` 字段来选择。

**添加新模型：**

1. 在 `python/sglang/srt/models/` 中创建新文件（例如 `my_model.py`）
2. 参照现有模型（例如 `llama.py`）的模式实现模型类
3. 在 `python/sglang/srt/models/registry.py` 中注册，添加架构名称

**示例 — LlamaModel：**
```python
# models/llama.py
class LlamaModel(nn.Module):
  def __init__(self, config, ...):
    # 构建Transformer层
    self.layers = nn.ModuleList([
      LlamaDecoderLayer(config, ...) for _ in range(config.num_hidden_layers)
    ])
    
  def forward(self, input_ids, positions, ...):
    # 使用SGLang优化的自定义前向传播
    ...
```

模型注册表自动将 `config.architectures` 映射到正确的实现类。

---

## 9.3 注意力后端扩展

**位置：** `python/sglang/srt/layers/attention/`

**可用后端：**

| 后端 | 类 | 用途 |
|---------|-------|---------|
| FlashInfer | `FlashInferAttnBackend` | 默认，针对 A100/H100 优化 |
| Triton | `TritonAttnBackend` | 备选，适用于任何 GPU |
| FlashAttention | `FlashAttentionBackend` | 适用于较旧 GPU |
| TritonWithFLA | `TritonWithFLABackend` | FlashLinearAttention，用于长上下文 |
| MLA | `MLABackend` | 多头潜在注意力（DeepSeek） |

**选择方式：** 通过 `--attention-backend` 控制，或根据 GPU 架构自动检测。

**添加新后端：**

1. 在 `python/sglang/srt/layers/attention/` 中创建新文件
2. 实现后端类，包含所需接口方法：
   - `init_forward_metadata()` — 准备批处理元数据
   - `forward()` — 执行注意力计算
3. 在注意力后端工厂中注册

---

## 9.4 LoRA 适配器扩展

**位置：** `python/sglang/srt/lora/`

**机制：** 动态 LoRA 适配器加载和服务。多个 LoRA 适配器可以并发提供服务。

**核心类：**

| 类 | 位置 | 用途 |
|-------|----------|---------|
| `LoRARef` | lora_registry.py:27 | LoRA 适配器的引用（路径、ID） |
| `LoRARegistry` | lora_registry.py:54 | 已加载适配器的注册表 |
| `LoRALayer` | lora.py:38 | 带有 A/B 矩阵的基础 LoRA 层 |
| `LoRAAdapter` | lora.py:48 | 包装基础模型的完整适配器模块 |
| `LoRAConfig` | lora_config.py:25 | 适配器配置 |
| `LoRAOverlapLoader` | lora_overlap_loader.py:21 | 带重叠的异步适配器加载 |
| `EvictionPolicy` | eviction_policy.py:28 | 适配器驱逐的抽象基类 |

**使用 LoRA：**

```python
# 启动时加载适配器
--lora-paths my_adapter=/path/to/adapter

# 或通过API动态加载
engine.update_weights_from_disk("/path/to/new_adapter")
```

**驱逐策略：**
- `LRUStrategy` — 驱逐最近最少使用的适配器
- `LFUStrategy` — 驱逐最不经常使用的适配器
- 自定义策略可通过继承 `EvictionPolicy` 添加

**设计意图：** LoRA 支持是一等扩展点。适配器按需加载到 GPU 内存中，在需要内存时可以被驱逐。`LoRAOverlapLoader` 将适配器加载与推理重叠执行，以最小化延迟峰值。

---

## 9.5 语法/约束生成扩展

**位置：** `python/sglang/srt/constrained/`

**机制：** 可插拔的语法后端，用于结构化生成（JSON、正则表达式、上下文无关文法）。

**可用后端：**

| 后端 | 类 | 库 | 用途 |
|---------|-------|---------|---------|
| XGrammar | `XGrammarGrammarBackend` | xgrammar | 基于快速 CFG 的语法 |
| Guidance | `GuidanceBackend` | llguidance | 基于 LLGuidance 的语法 |
| Reasoner | `ReasonerGrammarBackend` | custom | 感知推理的生成 |

**添加新后端：**

1. 继承 `BaseGrammarBackend`（base_grammar_backend.py:130）
2. 实现 `compile_grammar()` 和 `allocate_token_mask()` 方法
3. 通过 `--grammar-backend` 标志注册

---

## 9.6 KV 缓存驱逐策略扩展

**位置：** `python/sglang/srt/mem_cache/evict_policy.py`

**机制：** 基数缓存中 KV 缓存页面驱逐的策略模式。

**可用策略：**

| 策略 | 类 | 行为 |
|--------|-------|----------|
| LRU | `LRUStrategy` | 驱逐最近最少访问的 |
| LFU | `LFUStrategy` | 驱逐最不经常使用的 |
| FIFO | `FIFOStrategy` | 驱逐最旧的 |
| MRU | `MRUStrategy` | 驱逐最近访问的 |
| FILO | `FILOStrategy` | 驱逐最新的 |
| Priority | `PriorityStrategy` | 按请求优先级驱逐 |
| SLRU | `SLRUStrategy` | 分段 LRU，包含受保护/试用段 |

**添加新策略：**

1. 继承 `EvictionStrategy`（evict_policy.py:10）
2. 实现 `select_victim()` 方法
3. 在 `RadixCache.__init__()` 中注册策略名称

---

## 9.7 HiCache 存储扩展

**位置：** `python/sglang/srt/mem_cache/hicache_storage.py`

**机制：** 层次化缓存存储后端的抽象基类。

**可用后端：**

| 后端 | 类 | 用途 |
|---------|-------|---------|
| HiCacheFile | `HiCacheFile` | 基于文件的存储，用于 KV 缓存卸载 |

**添加新存储后端：**

1. 继承 `HiCacheStorage`（hicache_storage.py:124）
2. 实现 `attach()`、`detach()`、`read()`、`write()` 方法
3. 通过 `--hicache-storage-backend` 配置注册

---

## 9.8 权重加载器扩展

**位置：** `python/sglang/srt/layers/loader.py`

**机制：** 通过 `BaseModelLoader` 抽象类实现可插拔的模型权重加载器。

**可用加载器：**
- `DefaultModelLoader` — 标准 safetensors
- `LayeredModelLoader` — 内存高效的逐层加载
- `GGUFModelLoader` — GGUF 格式
- `BitsAndBytesModelLoader` — 4位/8位量化
- `RemoteModelLoader` — 远程权重加载
- `ModelOptModelLoader` — NVIDIA ModelOpt

**添加新加载器：**

1. 继承 `BaseModelLoader`（loader.py:280）
2. 实现 `load_model()` 方法
3. 在加载器选择逻辑中注册

---

## 9.9 SGLang 原生采样扩展

**位置：** `python/sglang/srt/sampling/`

SGLang 提供了针对批量 GPU 执行优化的自定义采样实现：
- Top-p（核）采样
- Top-k 采样
- Min-p 采样
- 频率/存在惩罚
- 通过 `--custom-logit-processor` 实现自定义 logits 处理器

---

## 9.10 扩展点总结

| 扩展点 | 机制 | 注册方式 |
|----------------|-----------|-------------|
| 模型架构 | Python 类 + 注册表 | `models/registry.py` |
| 注意力后端 | Python 类 | `--attention-backend` 标志 |
| LoRA 适配器 | 动态加载 + 注册表 | `--lora-paths` 或 API |
| 语法后端 | Python 类 | `--grammar-backend` 标志 |
| 缓存驱逐策略 | 策略模式 | `--eviction-policy` 标志 |
| HiCache 存储 | 抽象基类 | `--hicache-storage-backend` |
| 权重加载器 | 抽象基类 | 按模型格式自动检测 |
| 自定义 logits 处理器 | Python 类 | `--custom-logit-processor` |
