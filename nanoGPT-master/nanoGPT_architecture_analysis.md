# nanoGPT 机器人指令项目 — 完整架构分析

> 分析日期：2026-06-03

---

## 目录

1. [项目概述](#1-项目概述)
2. [数据准备：`prepare.py`](#2-数据准备preparepy)
3. [生成的文件详解](#3-生成的文件详解)
4. [GPT 模型定义：`model.py`](#4-gpt-模型定义modelpy)
5. [训练脚本：`train.py`](#5-训练脚本trainpy)
6. [推理与评估：`sample.py`](#6-推理与评估samplepy)
7. [配置覆盖：`configurator.py`](#7-配置覆盖configuratorpy)
8. [完整数据流](#8-完整数据流)
9. [关键设计决策总结](#9-关键设计决策总结)

---

## 1. 项目概述

本项目是 nanoGPT 的一个定制化版本，用于训练一个**微型机器人指令模型**：将自然语言指令（如 "Rotate joint 2 by 30 degrees"）映射为 JSON 格式的机器人命令序列。

核心特点：
- 使用**字符级（character-level）**分词，而非 GPT-2 的 BPE 分词
- 模型规模极小（6层 / 6头 / 384维），适合调试和学习
- 数据集共 986 条 NL→JSON 指令对（887 train / 99 val，90/10 分割）
- 训练与评估流程完整，含丰富的指标和可视化

---

## 2. 数据准备：`prepare.py`

**位置**: `data/robot_instr/prepare.py`

### 2.1 数据来源

```
优先级：
  1. 本地 input.txt（如果存在） → 直接解析
  2. HuggingFace 数据集 "Studeni/robot-instructions" → 在线下载
```

### 2.2 核心处理流程

```
┌─────────────┐     parse_input_txt()    ┌──────────────────┐
│  input.txt   │ ──────────────────────→  │  example dicts   │
│ (raw corpus) │                          │  [{input,output}]│
└─────────────┘                          └───────┬──────────┘
                                                  │
                                   90/10 split (train/val)
                                                  │
                                                  ▼
                                         ┌──────────────────┐
                                         │  format_example() │
                                         │  per example      │
                                         └───────┬──────────┘
                                                  │
                                                  ▼
                              ┌──────────────────────────────────┐
                              │  "Human: <instruction>\n         │
                              │   Robot: <json>\n\n"             │
                              │   (统一文本格式)                   │
                              └──────────────┬───────────────────┘
                                             │
                              join all → train_data / val_data
                                             │
                                             ▼
                              ┌──────────────────────────────────┐
                              │  字符级编码                        │
                              │  chars = sorted(set(all_chars))  │
                              │  stoi / itos mapping             │
                              │  encode() / decode()             │
                              └──────────────┬───────────────────┘
                                             │
                              ┌──────────────┼──────────────┐
                              ▼                             ▼
                        train.bin                     val.bin
                        (uint16 numpy)               (uint16 numpy)
                              │
                              ▼
                         meta.pkl
                         {vocab_size, stoi, itos}
```

### 2.3 关键函数详解

#### `parse_input_txt(filepath)`
将 `input.txt` 反向解析为结构化数据。解析逻辑：
- 以 `\n\n`（双换行）分割为 block
- 每个 block 内第一行以 `Human: ` 开头 → instruction
- 第二行以 `Robot: ` 开头 → JSON 输出
- 返回 `[{"input": ..., "output": ...}, ...]`

#### `format_example(ex)`
将单条 example 格式化为训练文本：
```python
f"Human: {instruction}\nRobot: {json_output}\n\n"
```
在格式化之前，会对 JSON 中的所有浮点数做 `round(..., decimals=2)` 处理，保证**数值模式一致**（所有浮点数都是 2 位小数），有利于模型学习。

#### `detect_units(instruction)` 与 `reverse_conversions(parsed_commands, instruction)`
这两个函数用于处理 **HuggingFace 数据集的单位转换问题**（仅在从 HF 下载时使用）。

HuggingFace 原始数据可能将人类友好单位（米、度）自动转换为内部单位（毫米、弧度），导致 JSON 数值与指令中的数值不一致。`reverse_conversions` 负责反向转换：
- **米 → 毫米**: `move_tcp` 的 x/y/z 参数 ÷ 1000
- **弧度 → 度**: `move_joint` 的 angle 参数 × 180/π

`detect_units` 通过正则匹配检测指令中使用了哪些单位：
- `uses_meters`: 检测 "meters" / " m" 等
- `uses_degrees`: 检测 "degrees" / "deg"
- `uses_radians`: 检测 "π" / "pi"
- `joint_degree_map`: 逐关节判断是用度还是弧度

#### `round_floats(obj, decimals=2)`
递归遍历嵌套 dict/list，将所有 float 四舍五入到指定位数。保证 JSON 中所有数值的小数位数一致。

### 2.4 字符级编码

```python
chars = sorted(list(set(train_data + val_data)))  # 所有唯一字符
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}  # string → int
itos = {i: ch for i, ch in enumerate(chars)}  # int → string
```

这不同于 GPT-2 的 BPE tokenizer（词表 50257），而是使用**原始字符**作为最小单位。本项目词表大小为 **77**（仅包含数据集中实际出现的字符，如字母、数字、JSON 符号、空格换行等）。

**为什么不直接用 GPT-2 tokenizer？**
- 数据集极小且领域特殊（JSON + 英文指令 + 数字）
- 字符级编码更简单、更透明、vocab 更小
- 避免 tokenizer 对 JSON 结构的碎片化

---

## 3. 生成的文件详解

`prepare.py` 在 `data/robot_instr/` 目录下生成以下文件：

### 3.1 `input.txt` — 原始文本语料

- **格式**: 每两条记录之间用 `\n\n` 分隔
- **内容示例**:
```
Human: Rotate joint 2 by 30 degrees, joint 7 by 45 degrees, and joint 3 by π/4 radians
Robot: [{"function": "move_joint", "kwargs": {"joint": [2, 7], "angle": [0.52, 0.79]}}, {"function": "move_joint", "kwargs": {"joint": [3], "angle": [0.79]}}]

Human: Fetch robot joint details
Robot: [{"function": "get_joint_values", "kwargs": {}}]
```
- **用途**: 
  - 作为 `prepare.py` 的输入和输出（幂等：重新运行不会改变格式）
  - `sample.py` 的评估模式从此文件读取测试样例

### 3.2 `train.bin` — 训练集二进制文件

- **格式**: 原始字节流，每个 token 占 2 字节（`dtype=np.uint16`）
- **内容**: 将训练集文本按字符级编码映射为整数序列后，以 uint16 格式写入
- **规模**: 123,013 tokens（887 条指令对，90% 数据）
- **读取方式**（train.py 第 125 行）:
```python
data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
```
- **为什么用 memmap？** 对于小数据集无所谓，但 nanoGPT 原始设计支持海量数据，memmap 允许按需从磁盘读取，不占用内存。每 batch 重新创建 memmap 对象以避免 NumPy 内存泄漏。

### 3.3 `val.bin` — 验证集二进制文件

- 格式与 `train.bin` 完全相同
- **规模**: 13,413 tokens（99 条指令对，10% 数据）

### 3.4 `meta.pkl` — 元数据文件

```python
meta = {
    "vocab_size": 77,                         # 字符词表大小
    "itos": {0: '\n', 1: ' ', 2: '"', ...},   # index → character
    "stoi": {'\n': 0, ' ': 1, '"': 2, ...},   # character → index
}
```

**用途**:
- `train.py` 读取 `vocab_size` 来确定模型输出维度
- `sample.py` 读取 `stoi`/`itos` 来编解码文本
- 如果 `meta.pkl` 不存在，`sample.py` 会回退到 GPT-2 的 tiktoken 编码器

---

## 4. GPT 模型定义：`model.py`

**位置**: `model.py`

### 4.1 整体架构

```
输入 token IDs
    │
    ▼
┌─────────────────────┐
│  Token Embedding    │  wte: (vocab_size, n_embd)
│  Position Embedding │  wpe: (block_size, n_embd)
└─────────┬───────────┘
          │  + Dropout
          ▼
┌─────────────────────┐
│  Transformer Block  │  × n_layer
│  ┌─────────────────┐│
│  │ LayerNorm → CausalSelfAttention  │
│  │     + Residual  ││
│  │ LayerNorm → MLP ││
│  │     + Residual  ││
│  └─────────────────┘│
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Final LayerNorm    │  ln_f
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  LM Head (Linear)   │  lm_head: (n_embd, vocab_size)
│  = wte.weight (tied)│  ← 权重共享 (weight tying)
└─────────┬───────────┘
          │
          ▼
       logits
```

### 4.2 模块详解

#### `GPTConfig` (dataclass)
```python
@dataclass
class GPTConfig:
    block_size: int = 1024    # 最大上下文长度
    vocab_size: int = 50304   # 词表大小（GPT-2: 50257 → 向上取整到64的倍数）
    n_layer: int = 12         # Transformer 层数
    n_head: int = 12          # 注意力头数
    n_embd: int = 768         # 嵌入维度
    dropout: float = 0.0      # Dropout 比例
    bias: bool = True         # 是否在 Linear/LayerNorm 中使用 bias
```

#### `LayerNorm`
- 支持 `bias=False` 的 LayerNorm（PyTorch 原生不支持）
- 手动实现 `F.layer_norm`

#### `CausalSelfAttention`
- **Flash Attention**: PyTorch ≥ 2.0 时自动使用 `scaled_dot_product_attention`（更快、更省显存）
- **手动实现**（降级方案）: 标准的 causal mask + softmax 注意力
- Q/K/V 通过单个 Linear 层投影 (`c_attn`: n_embd → 3×n_embd)
- 输出投影: `c_proj`: n_embd → n_embd

#### `MLP`
- 标准的两层 FFN：`Linear → GELU → Linear → Dropout`
- 中间维度 = 4 × n_embd（标准 GPT-2 设计）

#### `Block`
```
x = x + Attention(LayerNorm(x))   # Pre-LayerNorm + 残差
x = x + MLP(LayerNorm(x))         # Pre-LayerNorm + 残差
```

#### `GPT` (主模型)

| 方法 | 功能 |
|------|------|
| `forward(idx, targets)` | 训练：返回 logits + cross-entropy loss；推理：仅返回最后位置的 logits |
| `generate(idx, max_new_tokens, temperature, top_k)` | 自回归生成，支持 temperature 采样和 top-k 过滤 |
| `from_pretrained(model_type)` | 从 HuggingFace 加载 GPT-2 预训练权重，需处理 Conv1D→Linear 的转置 |
| `configure_optimizers(...)` | 构建 AdamW 优化器，2D 参数（权重）使用 weight decay，<2D 参数（bias/LayerNorm）不用 |
| `estimate_mfu(fwdbwd_per_iter, dt)` | 估算 MFU（Model FLOPs Utilization），以 A100 bfloat16 峰值 312 TFLOPS 为基准 |
| `crop_block_size(block_size)` | 模型手术：裁剪位置嵌入和 attention bias 到更小的 block_size |
| `_init_weights(module)` | 权重初始化：Linear/Embedding 用 N(0, 0.02)，bias 用零；残差投影用缩放初始化 |

**关键设计细节**:
- **Weight Tying**: `lm_head.weight` 与 `wte.weight` 共享（第 138 行）
- **残差投影缩放初始化**: `c_proj.weight` 使用 `std=0.02/√(2*n_layer)`，保证残差路径在初始化时接近恒等映射
- **推理优化**: 推理时 `lm_head` 只对最后一个位置计算，节省计算量

### 4.3 参数量估算

对于机器人指令的小模型配置：
```
n_layer=6, n_head=6, n_embd=384, vocab_size=77
参数量 ≈ 6 × (4×384² + 2×384²) + 384×77 + 384×77（weight tying 下 wte/lm_head 共享）
       ≈ 6 × 884,736 + 29,568 + 29,568
       ≈ 5.4M 参数
```

---

## 5. 训练脚本：`train.py`

**位置**: `train.py`

### 5.1 训练配置

#### 超参数（针对机器人指令任务）

| 参数 | 值 | 说明 |
|------|-----|------|
| `dataset` | `robot_instr` | 数据集名称（对应 data/ 下的目录）|
| `batch_size` | 32 | 每步样本数 |
| `block_size` | 384 | 上下文窗口 |
| `n_layer` | 6 | Transformer 层数 |
| `n_head` | 6 | 注意力头数 |
| `n_embd` | 384 | 嵌入维度 |
| `dropout` | 0.2 | Dropout（小数据集需要正则化）|
| `bias` | False | 不使用 bias（更快更好）|
| `learning_rate` | 1e-3 | 初始学习率 |
| `max_iters` | 1600 | 总训练步数 |
| `weight_decay` | 0.1 | AdamW 权重衰减 |
| `grad_clip` | 1.0 | 梯度裁剪阈值 |
| `lr_decay` | cosine | 余弦退火（warmup 50 步 → min_lr=1e-4）|

### 5.2 训练流程

```
初始化
  ├── DDP 检测与初始化（可选多 GPU）
  ├── 随机种子设置
  ├── TF32 加速启用
  ├── AMP 混合精度上下文 (bfloat16/float16)
  │
  ▼
数据加载器 get_batch(split)
  ├── np.memmap 读取 .bin 文件
  ├── 随机采样 (batch_size, block_size) 的 token 块
  │   x = tokens[i : i+block_size]
  │   y = tokens[i+1 : i+1+block_size]  ← 错位一个位置
  └── pin_memory + async transfer to GPU
  │
  ▼
模型初始化
  ├── scratch: 从头训练
  ├── resume: 从 ckpt.pt 恢复
  └── gpt2*: 从 GPT-2 预训练权重初始化（仅限大模型）
  │
  ▼
训练循环 (iter_num: 0 → max_iters)
  ├── 每步:
  │   ├── 设置学习率 (cosine schedule)
  │   ├── forward + backward (支持梯度累积)
  │   ├── 梯度裁剪 (grad_clip=1.0)
  │   ├── optimizer.step() + scaler.update()
  │   └── prefetch 下一 batch（GPU 计算与数据加载并行）
  │
  ├── 每 eval_interval=100 步:
  │   ├── estimate_loss() — 在 train/val 上各评估 eval_iters=50 次取平均
  │   ├── 记录指标: loss, perplexity, LR, MFU
  │   ├── 保存最佳 checkpoint (ckpt.pt)
  │   └── wandb 日志（可选）
  │
  └── 每 log_interval=5 步:
      └── 打印 loss, 耗时, MFU
  │
  ▼
训练结束
  ├── 生成 6 合 1 训练指标图表 (training_metrics.png)
  │   ├── Loss 曲线（eval模式下的 train/val loss）
  │   ├── 原始 Training Loss（per log_interval）
  │   ├── Perplexity 曲线
  │   ├── 学习率调度曲线
  │   ├── MFU 曲线
  │   └── 文本摘要（最佳loss、最终指标、配置）
  └── 保存指标数据 (training_metrics.npz)
```

### 5.3 关键训练技术

#### 混合精度训练 (AMP)
```python
ptdtype = torch.bfloat16  # 优先 bfloat16，回退 float16
ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype)
scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))
```
- bfloat16: 不需要 GradScaler（动态范围与 float32 相同）
- float16: 需要 GradScaler 防止梯度下溢

#### 梯度累积
```python
for micro_step in range(gradient_accumulation_steps):
    loss = loss / gradient_accumulation_steps  # 缩放
    scaler.scale(loss).backward()
```
模拟更大的 batch size（`tokens_per_iter = gradient_accumulation_steps × ddp_world_size × batch_size × block_size`）

#### DDP (Distributed Data Parallel)
```python
if ddp:
    init_process_group(backend='nccl')
    model = DDP(model, device_ids=[ddp_local_rank])
    model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
```
- 仅在最后一个 micro_step 同步梯度（优化通信效率）
- 自动将 `gradient_accumulation_steps` 按 world_size 缩放

#### 学习率调度
```
warmup (linear): iter 0 → warmup_iters
cosine decay:    warmup_iters → lr_decay_iters
constant:        lr_decay_iters → max_iters (min_lr)
```

#### 数据预取 (Prefetch)
```python
# 在 GPU 做 forward pass 的同时，异步加载下一 batch
X, Y = get_batch('train')  # 立即启动下一批数据加载
scaler.scale(loss).backward()  # GPU 继续计算当前 batch
```

### 5.4 训练指标图表

训练结束后自动生成 `training_metrics.png`（2×3 子图矩阵，含中文字体支持）：

| 位置 | 图表 | 说明 |
|------|------|------|
| (0,0) | Train & Val Loss | eval 模式下的交叉熵损失 |
| (0,1) | Training Loss (raw) | 每 log_interval 记录的训练损失 |
| (0,2) | Perplexity | PPL = exp(loss) |
| (1,0) | Learning Rate Schedule | 展示 warmup + cosine decay |
| (1,1) | MFU | Model FLOPs Utilization (%) |
| (1,2) | Summary Text | 最佳 val loss、最终指标、超参数 |

---

## 6. 推理与评估：`sample.py`

**位置**: `sample.py`

### 6.1 两种模式

```
sample.py
├── eval_mode=True  → 评估模式（默认）
│   ├── 从 input.txt 加载测试样例
│   ├── 逐个生成并计算指标
│   └── 输出图表和 JSON 结果
│
└── eval_mode=False → 交互采样模式
    ├── 给定一个起始 prompt
    ├── 生成 num_samples 个续写
    └── 打印生成的文本
```

### 6.2 评估模式详解

#### 测试数据构建
从 `input.txt` 解析所有 `Human: ... \n Robot: ...` 对，使用前 `eval_num_samples` 条（默认 50）。

#### 生成流程
```python
prompt_text = f"Human: {instruction}\nRobot:"
prompt_ids = encode(prompt_text)
x = torch.tensor(prompt_ids)[None, ...]    # (1, T)

y = model.generate(x, max_new_tokens, temperature, top_k)

generated = decode(y) - prompt_text        # 只取生成部分
generated = trim_at_double_newline(generated)  # 截断到下一个样本边界
```

#### 评估指标

| 指标 | 计算方式 | 含义 |
|------|---------|------|
| **JSON Valid Rate** | `is_valid_json(generated)` | 生成的内容是否是合法 JSON |
| **Exact Match Rate** | `normalized(generated) == normalized(expected)` | 标准化后完全匹配（先 parse 再 dumps） |
| **Function Match Rate** | `funcs(generated) == funcs(expected)` | 函数名集合完全一致 |
| **Function Overlap (Jaccard)** | `|A ∩ B| / |A ∪ B|` | 函数名集合的 Jaccard 相似度 |
| **Character Error Rate (CER)** | `(char_errors + len_diff) / max_len` | 字符级编辑错误率 |

#### 评估输出

1. **实时控制台输出**：每个样本生成后立即打印结果
2. **`eval_results.json`**：详细的逐样本结果
3. **`eval_results.png`**：可视化图表
   - 左图：聚合指标柱状图（JSON Valid、Exact Match、Function Match、Char Accuracy）
   - 右图：逐样本 CER 柱状图（颜色区分高于/低于平均值）

### 6.3 文本生成参数

```python
temperature = 0.8   # < 1.0 = 更确定, > 1.0 = 更随机
top_k = 200         # 只从 top-200 中采样
```

评估模式下使用 `temperature=0.0`（贪婪解码，完全确定性）。

### 6.4 编码器选择逻辑

```python
if meta.pkl 存在:
    使用 meta['stoi']/meta['itos']  # 字符级编码
else:
    使用 tiktoken.get_encoding("gpt2")  # GPT-2 BPE 编码
```

这保证了评估时使用的编码与训练时完全一致。

---

## 7. 配置覆盖：`configurator.py`

**位置**: `configurator.py`

一个极简的配置系统，通过命令行参数覆盖脚本中的全局变量：

```bash
# 从配置文件覆盖
python train.py config/my_config.py

# 从命令行参数覆盖
python train.py --batch_size=64 --learning_rate=0.001

# 混合使用
python train.py config/base.py --max_iters=5000
```

**实现原理**:
1. 解析 `sys.argv`，以 `=` 区分配置文件路径和键值对
2. 配置文件: `exec(open(config_file).read())` 直接在调用脚本的全局命名空间中执行
3. 键值对: `globals()[key] = literal_eval(val)`，自动匹配类型

---

## 8. 完整数据流

```
┌────────────────────────────────────────────────────────────────────┐
│                        prepare.py                                  │
│                                                                    │
│  input.txt ──→ 解析 ──→ 格式化 ──→ 字符编码 ──→ train.bin         │
│  (或HF下载)      │         │          │            val.bin         │
│                  │         │          │            meta.pkl        │
│                  │         │          │                            │
│                  ▼         ▼          ▼                            │
│           [{input,   "Human: X\n   chars → ids                     │
│             output}]  Robot: Y\n\n"  stoi/itos                     │
└────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────────────────┐
│                         train.py                                   │
│                                                                    │
│  train.bin ──→ np.memmap ──→ random chunks ──→ (X, Y) batches     │
│  meta.pkl  ──→ vocab_size ──→ GPTConfig ──→ GPT model             │
│                                                                    │
│  Training Loop:                                                    │
│    forward → loss → backward → clip_grad → optimizer.step()        │
│                                                                    │
│  Output:                                                           │
│    out-robot-instr/ckpt.pt          (best checkpoint)              │
│    out-robot-instr/training_metrics.png                            │
│    out-robot-instr/training_metrics.npz                            │
└────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────────────────┐
│                        sample.py                                   │
│                                                                    │
│  ckpt.pt ──→ GPT model ──→ model.generate()                       │
│  meta.pkl ──→ encode/decode                                       │
│  input.txt ──→ test examples                                       │
│                                                                    │
│  For each test example:                                            │
│    "Human: {instr}\nRobot:" ──→ model ──→ generated JSON          │
│                                                                    │
│  Metrics: JSON valid, Exact Match, Function Match, CER             │
│  Output:                                                           │
│    out-robot-instr/eval_results.json                               │
│    out-robot-instr/eval_results.png                                │
└────────────────────────────────────────────────────────────────────┘
```

---

## 9. 关键设计决策总结

### 9.1 为什么用字符级编码？

| 方面 | 字符级编码 | GPT-2 BPE Tokenizer |
|------|-----------|---------------------|
| 词表大小 | 77 | 50257 |
| JSON 表示 | 逐字符，完全可逆 | 可能把 `{"` 编码为单个 token |
| 数字表示 | 每个数字字符独立 | 可能有专门的数字 token |
| 适合小数据 | ✅ 无需大量文本学习 token 边界 | ❌ 很多 token 可能从未出现 |
| 嵌入层参数 | 极小 (77×384) | 较大 (50257×384) |

对于仅 986 条样本的机器人指令数据集，字符级编码是最合理的选择。

### 9.2 为什么用 memmap 而非直接加载？

原始 nanoGPT 设计用于海量数据集（OpenWebText 等），memmap 允许零拷贝按需读取。对于这个小数据集其实可以直接加载到内存，但保持了与原始设计的兼容性。

### 9.3 Weight Tying 的作用

`lm_head.weight = wte.weight`（输入嵌入 = 输出投影权重）：
- 减少参数量（共享 vocab_size × n_embd 的参数）
- 为输入和输出空间提供一致的语义表示
- 对字符级小词表，节省效果不明显，但保持了标准 GPT 设计

### 9.4 为什么 `prepare.py` 放在数据目录下而非项目根目录？

这是一种数据集自包含的设计模式：每个数据集有自己独立的预处理脚本。将来添加新数据集时，只需在 `data/<new_dataset>/` 下创建对应的 `prepare.py`，互不干扰。

### 9.5 浮点数精度统一 (`round_floats`)

确保训练数据中所有浮点数都是 2 位小数，例如：
- `0.52` 而非 `0.5235987755982988`
- `50.0` 而非 `50.0000000001`

这减少了数字的 token 长度变化，使模型更容易学习数值模式。

---

## 10. 检查点文件详解：`ckpt.pt`

**位置**: `out-robot-instr/ckpt.pt`

检查点是连接训练与推理的核心纽带——`train.py` 负责写入，`sample.py` 负责读取，二者通过 `ckpt.pt` 实现模型的无缝交接。

### 10.1 内部结构

`ckpt.pt` 是一个通过 `torch.save()` 序列化的 Python dict，包含 6 个字段：

```python
checkpoint = {
    'model':        raw_model.state_dict(),   # 模型权重
    'optimizer':    optimizer.state_dict(),    # 优化器状态
    'model_args':   model_args,               # 模型架构参数
    'iter_num':     iter_num,                 # 当前训练步数
    'best_val_loss': best_val_loss,           # 最佳验证损失
    'config':       config,                   # 完整训练配置
}
```

#### (1) `model` — 模型权重（`OrderedDict`）

存储模型所有可学习参数，键名对应 `model.py` 中的模块层次结构：

```
_model 层数结构（n_layer=6 时共 121 个 tensor）_

transformer.wte.weight          (vocab_size, n_embd)     Token 嵌入矩阵
transformer.wpe.weight          (block_size, n_embd)     位置嵌入矩阵
transformer.h.0.ln_1.weight     (n_embd,)                Block 0 的第一个 LayerNorm
transformer.h.0.ln_1.bias       (n_embd,) 或 None        (bias=False 时无此项)
transformer.h.0.attn.c_attn.weight  (3×n_embd, n_embd)   Q/K/V 投影（合并）
transformer.h.0.attn.c_attn.bias    (3×n_embd,) 或 None
transformer.h.0.attn.c_proj.weight  (n_embd, n_embd)     注意力输出投影
transformer.h.0.attn.c_proj.bias    (n_embd,) 或 None
transformer.h.0.ln_2.weight     (n_embd,)                Block 0 的第二个 LayerNorm
transformer.h.0.ln_2.bias       (n_embd,) 或 None
transformer.h.0.mlp.c_fc.weight (4×n_embd, n_embd)      FFN 第一层
transformer.h.0.mlp.c_fc.bias   (4×n_embd,) 或 None
transformer.h.0.mlp.c_proj.weight   (n_embd, 4×n_embd)  FFN 第二层
transformer.h.0.mlp.c_proj.bias     (n_embd,) 或 None
transformer.h.1.* ...                                      Block 1（同上）
...（共 n_layer 个 Block）
transformer.ln_f.weight         (n_embd,)                最终 LayerNorm
transformer.ln_f.bias           (n_embd,) 或 None
lm_head.weight                  (vocab_size, n_embd)     输出投影（= wte.weight，共享存储）
```

> **注意**：`lm_head.weight` 与 `transformer.wte.weight` 是同一块内存（weight tying），`state_dict` 中二者指向同一个 tensor，只存一份。

**为什么在 checkpoint 中保存 state_dict 而非整个 model 对象？**
- `torch.save(model, ...)` 会绑定类定义，如果 `model.py` 发生修改，旧的 checkpoint 可能无法加载
- 先保存 `state_dict`，加载时重新构造 `GPT(config)` 再 `load_state_dict()`，只要架构参数（n_layer、n_embd 等）一致即可恢复

#### (2) `optimizer` — 优化器状态（`dict`）

```python
{
    'state': {
        0: {  # 第一个参数组
            'step': tensor(...),         # 当前 step 数
            'exp_avg': tensor(...),       # Adam 一阶动量 (m)
            'exp_avg_sq': tensor(...),    # Adam 二阶动量 (v)
        },
        1: { ... },  # 第二个参数组
        ...
    },
    'param_groups': [
        {'lr': ..., 'betas': ..., 'weight_decay': ..., 'params': [0,1,2,...]},
        {'lr': ..., 'betas': ..., 'weight_decay': 0.0, 'params': [...]},
    ]
}
```

- `state` 为每个参数保存 Adam 的动量缓冲区（`exp_avg` 和 `exp_avg_sq`），shape 与对应参数相同
- `param_groups` 保存优化器超参数，其中第一组（2D 参数）有 weight_decay，第二组（<2D 参数）无 weight_decay
- 体积约为模型权重的 **2 倍**（因为每个参数额外存储一阶和二阶动量）

**作用**：恢复训练时不仅恢复模型权重，还恢复优化器的动量状态，确保学习率调度和优化轨迹无缝接续。

#### (3) `model_args` — 模型架构参数（`dict`）

```python
{
    'n_layer': 6,
    'n_head': 6,
    'n_embd': 384,
    'block_size': 384,
    'bias': False,
    'vocab_size': 77,
    'dropout': 0.2,
}
```

这 7 个参数完整描述了模型架构，`sample.py` 通过 `GPTConfig(**checkpoint['model_args'])` 重新构造一个与训练时完全一致的模型结构，然后加载权重。**任何一项不匹配都会导致 `load_state_dict` 失败**。

`train.py` 在 `init_from='resume'` 时会强制从 checkpoint 覆盖以下 6 个关键字段：
```python
for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
    model_args[k] = checkpoint_model_args[k]
```
`dropout` 则可以使用命令行传入的新值（因为 dropout 只影响训练，不影响模型结构）。

#### (4) `iter_num` — 训练步数（`int`）

当前训练已完成的迭代次数。恢复训练时，循环从 `iter_num + 1` 继续（而不是从 0 开始），学习率调度器也根据此值计算当前 LR。

#### (5) `best_val_loss` — 最佳验证损失（`float`）

训练过程中遇到的最低验证损失值。恢复训练后，只有当前 val_loss **低于**此值才会覆盖 checkpoint（除非 `always_save_checkpoint=True`）。

#### (6) `config` — 完整训练配置（`dict`）

```python
{
    'out_dir': 'out-robot-instr',
    'batch_size': 32,
    'block_size': 384,
    'learning_rate': 0.001,
    'max_iters': 1600,
    'gradient_accumulation_steps': 1,
    'device': 'cuda',
    'dtype': 'bfloat16',
    'compile': True,
    # ... 所有 train.py 的全局配置变量
}
```

记录了生成此 checkpoint 时的**所有训练超参数**。`sample.py` 利用其中的 `dataset` 字段定位 `meta.pkl` 路径：
```python
if 'config' in checkpoint and 'dataset' in checkpoint['config']:
    meta_path = os.path.join('data', 'robot_instr', 'meta.pkl')
```

### 10.2 保存时机与策略

`train.py` 中的保存逻辑（第 303-315 行）：

```python
if losses['val'] < best_val_loss or always_save_checkpoint:
    best_val_loss = losses['val']
    if iter_num > 0:
        checkpoint = { ... }
        torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
```

| 条件 | 行为 |
|------|------|
| `val_loss` 创新低 | **覆盖保存** — 始终保留最优模型 |
| `always_save_checkpoint=True` | 每次 eval 都保存（用于小数据集快速过拟合场景）|
| `iter_num == 0` | 不保存（初始随机权重没有保存价值）|

**设计意图**：对于只有 986 条样本的小数据集，模型会快速过拟合。仅保存最佳 val_loss 的 checkpoint 等价于 **early stopping**——最终得到的总是泛化能力最好的权重，而非训练到最后可能已过拟合的权重。

### 10.3 加载与恢复流程

#### 训练恢复（`train.py`, `init_from='resume'`）

```
1. torch.load(ckpt.pt)
2. 从 checkpoint['model_args'] 提取架构参数，覆盖命令行配置
3. GPTConfig(**model_args) → GPT(gptconf)  // 重建模型结构
4. model.load_state_dict(checkpoint['model'])  // 恢复权重
5. optimizer.load_state_dict(checkpoint['optimizer'])  // 恢复优化器状态
6. iter_num = checkpoint['iter_num']  // 从断点继续计数
7. best_val_loss = checkpoint['best_val_loss']  // 保持 best 记录
```

**特殊处理**：`_orig_mod.` 前缀剥离
```python
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
```
当模型经过 `torch.compile()` 后保存的 checkpoint，state_dict 的键名会被加上 `_orig_mod.` 前缀。加载时必须剥离此前缀，否则 `load_state_dict` 会因键名不匹配而失败。

#### 推理加载（`sample.py`, `init_from='resume'`）

```
1. torch.load(ckpt.pt)
2. GPTConfig(**checkpoint['model_args']) → GPT(gptconf)
3. 剥离 _orig_mod. 前缀（如有）
4. model.load_state_dict(checkpoint['model'])
5. model.eval()  // 切换到评估模式（禁用 dropout）
6. model.to(device)
```

推理时**不需要**恢复 optimizer（不训练），也**不需要** `iter_num`/`best_val_loss`（不记录指标）。

### 10.4 ckpt.pt 在全流程中的角色

```
┌──────────┐         ┌──────────┐         ┌──────────┐
│ prepare  │         │  train   │         │  sample  │
│   .py    │         │   .py    │         │   .py    │
└────┬─────┘         └────┬─────┘         └────┬─────┘
     │                    │                    │
     ▼                    │                    │
  train.bin ──────────────►                    │
  val.bin  ──────────────►│                    │
  meta.pkl ──────────────►│                    │
                           │                    │
                      【训练循环】               │
                           │                    │
                           ▼                    │
                        ckpt.pt ───────────────►│
                           │                    │
                           │              【模型重建 + 评估】
                           │                    │
                           ▼                    ▼
                    training_metrics      eval_results.json
                    .png / .npz           eval_results.png
```

**核心设计原则——模型结构与权重分离**：
- `model_args`（结构）和 `model` state_dict（权重）分开存储
- 加载方先根据 `model_args` 构造空模型，再灌入权重
- 这保证了即使 `model.py` 中的辅助方法（如 `configure_optimizers`、`estimate_mfu`）发生改变，只要核心架构不变，checkpoint 仍然可以加载
- 也是 PyTorch 社区的最佳实践（对比直接 `torch.save(model)`）

### 10.5 文件体积估算

以本项目机器人指令模型（5.4M 参数，`bias=False`）为例：

| 组件 | 参数/缓冲区数量 | 估算大小 |
|------|----------------|---------|
| model state_dict | ~5.4M 参数 × 2 bytes (bfloat16) | ~10.8 MB |
| optimizer state | ~5.4M × 2（动量 m+v）× 4 bytes (float32) | ~43.2 MB |
| model_args + config | 少量标量 | < 1 KB |
| **合计** | | **~54 MB** |

> 实际大小取决于训练时的 `dtype`。bfloat16 权重存为 2 字节/参数，但 optimizer 的动量始终以 float32 存储，因此 optimizer 状态占据 checkpoint 体积的 ~80%。

---

## 附录：文件清单

| 文件 | 位置 | 功能 |
|------|------|------|
| `model.py` | 根目录 | GPT 模型定义（架构、前向、生成、优化器）|
| `train.py` | 根目录 | 训练脚本（数据加载、训练循环、指标记录）|
| `sample.py` | 根目录 | 推理脚本（评估模式 / 交互生成）|
| `configurator.py` | 根目录 | 命令行配置覆盖工具 |
| `prepare.py` | `data/robot_instr/` | 数据预处理（文本格式化、字符编码）|
| `input.txt` | `data/robot_instr/` | 原始文本语料（986 条指令对，136,426 字符）|
| `train.bin` | `data/robot_instr/` | 训练集二进制文件（uint16 编码）|
| `val.bin` | `data/robot_instr/` | 验证集二进制文件（uint16 编码）|
| `meta.pkl` | `data/robot_instr/` | 元数据（vocab_size, stoi, itos）|
| `ckpt.pt` | `out-robot-instr/` | 训练检查点（模型权重 + 优化器状态 + 架构/配置元信息）|