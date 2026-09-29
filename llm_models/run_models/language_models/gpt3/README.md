# GPT-3 — decoder-only Transformer

GPT-3 的自回归路径：

```mermaid
flowchart LR
  A[Token + 位置嵌入] --> B[因果 MHA]
  B --> C[GELU FFN]
  C --> D[堆叠 Decoder Block]
  D --> E[词表 logits]
  E --> F[采样下一个 token]
```


## 直觉

把一切 NLP 任务都写成 "续写": 模型只做一件事 —— 看着前面的 token 猜下一个。结构是 N 个相同的积木
`x = x + Attn(LN(x)); x = x + FFN(LN(x))` (Pre-LN), 因果 mask 保证位置 t 看不到 t 之后。

## 核心原理

### 核心公式

- 目标: `L = -Σ_t log P(x_{t+1} | x_{≤t})`; 标签 = 输入右移一位。
- 初始 loss 应 ≈ `ln V` (均匀瞎猜)。V=1000 → 6.908。
- KV cache: 过去 token 的 K/V 不再变化, 缓存后每步只算 1 个新 token, 生成 C 个 token 从 O(C·T²) 降到 O(C·T)。

## 运行

```bash
python -m llm_models.run_models.language_models.gpt3.train_gpt3
python -m llm_models.run_models.language_models.gpt3.infer_gpt3
```

## 运行后应该看到什么

### (实测, CPU)

- train: 参数量 1,836,032; `初始 loss 6.947 vs ln V = 6.908 | 最终 loss 0.071` (60 步)。
  两个 assert: `|初始 − ln V| < 0.5`, `最终 < 0.5 × 初始`。
- infer: Sin-PE 与 RoPE 各跑一遍: 改动后半段 token, 前半段 logits 逐位不变 (因果性);
  生成 200 token 有/无 cache 输出完全一致, 加速约 3.4x / 3.9x (数值随机器波动)。

## 与真实系统的差距

- **规模**: train 是 2 层、d_model=256、4 头, 共 1,836,032 个参数。GPT-3 是 175B。
- **位置编码**: 本库用 Sinusoidal (零参数, 可切到 RoPE 做对照)。GPT-2/3 原版是可学习的绝对位置 embedding。
- **embedding 乘 √D 是本库约定**: GPT-2/3 不乘。本库的 LM 统一这样写, 配合 N(0, 0.02²) 初始化。
- **数据是合成的**: 固定一个随机 batch (2 条 × 32 token) 反复训 60 步, 没有 tokenizer 和语料。固定 batch 是本库约定。
- **Sin-PE 下 `attention_mask` 只收全 1**: Sin-PE 是绝对位置, 左 padding 会把真实 token 的位置整体推后, 输出就变了。所以 mask 里有 0 就抛 `NotImplementedError`, 批量生成要用等长 prompt。`use_rope=True` 时照常屏蔽 pad, 左 pad 不改真实位置的输出。
- **上下文写满之后**: 序列到 `max_len` 后窗口每步左移, cache 每步作废, `generate()` 退回每步整段重算。
- **采样**: 只有 temperature 和 top-k, 没有 top-p 和重复惩罚。

## 常见误区

- "loss 从 6.9 降到 0.07 = 学会了语言": 不是。数据是**固定的一个随机 batch**, 下降只说明模型背下了它,
  验证的是 forward/backward/优化器通路。每步换新随机 batch 时 loss 会停在 ln V。
- "初始 loss 多大无所谓": N(0,1) embedding + weight tying 会让初始 logits 标准差 ≈ sqrt(D), 首步 loss 冲到 ~255。
  本库的模型统一调 `init_weights`, 用 N(0, 0.02²)。
- "KV cache 是近似": 不是, 是精确等价; 本脚本用 `torch.equal` 断言。Sin-PE 下要记得给位置编码加 offset。

## 自测题

1. 为什么 weight tying 会放大坏初始化的后果? —— logits = h·Eᵀ, E 既是输入又是输出; E~N(0,1) 时 h 的范数 ≈ sqrt(D), logits 标准差 ≈ sqrt(D), softmax 极尖。
2. 带 cache 解码第 t 步时, 因果 mask 取哪一块? —— 行 `[past:past+1]`、列 `[:past+1]`, 即 "新 query 看全部历史"。
3. Pre-LN 末尾为什么必须有 `ln_f`? —— 残差主路从不过 norm, 范数随层数累积, 不归一化直接进 lm_head 会让 logits 尺度失控。
