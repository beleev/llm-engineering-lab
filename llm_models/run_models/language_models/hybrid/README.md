# 混合架构家族 — KDA、门控注意力、NoPE 与 iRoPE

[llm_models 目录](../../../README.md) · [交互实验](https://beleev.github.io/llm-engineering-lab/#/models/hybrid-family)

## 直觉

- 2025-2026 的主流长上下文架构都是 "大部分层便宜、少数层精确" 的混合。便宜的那一半用定长状态 (线性注意力或 Mamba), 精确的那一半保留全注意力。各家的差别只在三个选择上: 便宜层用什么、精确层用什么、位置信息放在哪。
- **Kimi Linear / Kimi K3**: 便宜层是 KDA (Gated DeltaNet 的遗忘门从每头一个标量细化到每个 key 通道一个), 精确层是不加位置编码的 MLA, 比例 3:1。顺序信息交给 KDA 的衰减门隐式携带, 所以扩上下文时不用调 RoPE 的频率基数, 也不用上 YaRN。
- **Qwen3-Next / Qwen3.5**: 便宜层是 Gated DeltaNet, 精确层是带 SDPA 输出门的全注意力, 同样 3:1。输出门让注意力可以 "读了也不用", 论文报告它把落在第 0 个 token 上的注意力从 46.7% 压到 4.8%。
- **Llama 4 iRoPE**: 不分便宜和精确, 而是让不加位置编码的层与带 RoPE 的层交错, RoPE 层用分块局部注意力, 推理时再对注意力做温度缩放。Meta 说 i 指 interleaved, 同时指向支持无限长上下文这个目标。
- **Nemotron 3.5 / Jamba**: 把便宜那一半换成 Mamba。本库的 Mamba 在 `mamba.py` 里单列一章, 这里不收。

## 核心原理

### KDA: 遗忘门的粒度

```
GDN: S_t = α_t (I − β_t k_t k_t^T) S_{t-1} + β_t k_t v_t^T        α_t 每头一个标量
KDA: S_t = (I − β_t k_t k_t^T) Diag(α_t) S_{t-1} + β_t k_t v_t^T   α_t ∈ (0, 1)^{d_k}
```

`α_t` 的各通道取同一个值时两式相等 (标量与矩阵可交换), 所以 KDA 是 GDN 的严格推广; infer 脚本断言了这一点 (最大差 3.0e-08)。α 是向量时 "先衰减再写入" 这个顺序才有意义: `Diag(α)` 乘在状态的 key 维上, 与 `(I − β k k^T)` 不交换。论文把 GDN 的门称作 coarse head-wise, KDA 让每个特征维度各有自己的遗忘速率。

### 门控注意力: 在输出侧解决 attention sink

```
y = W_o [ σ(W_g x) ⊙ SDPA(q, k, v) ]
```

softmax 的行和恒为 1, 每个 query 必须把这 1 份概率质量分完。没有合适的 key 时模型只能倒给一个无害的位置, 最常见的就是第 0 个 token。输出门让注意力读了也可以不用: 质量照样分完, 乘上一个接近 0 的门之后不进残差。

本库的 `GatedAttention` 继承 `GroupedQueryAttention`, `use_gate=False` 时与它逐位相同 (infer 脚本实测差 0.0), 所以 "有门 / 无门" 的对照是同一条代码路径, 只差那一次乘法。

### 三类掩码与位置编码

`HybridLM` 的每层按混合器类型拿到不同的掩码和位置编码:

| 混合器 | 位置编码 | 掩码 | 每 token 的缓存 |
| --- | --- | --- | --- |
| `gdn` / `kda` | 无 (衰减门隐式携带) | `[B, T]` 哪些是真 token | 0, 只有一个定长状态 |
| `gqa` / `gated_gqa` | RoPE | 全长因果 | `2·Hkv·Dh` |
| `mla_nope` | 无 | 全长因果 | `r + rope` |
| `gqa_nope` | 无 | 全长因果 | `2·Hkv·Dh` |
| `gqa_chunk` | RoPE | 只在自己那一块之内 | `2·Hkv·Dh` |

缓存元素数的算式是 `Σ_{全注意力层} T · 每 token 的 K/V + Σ_{线性层} H·Dh²`, 后一项与 T 无关。

## 运行

论文: Kimi Team, "Kimi Linear: An Expressive, Efficient Attention Architecture" (arXiv 2510.26692, 2025-10); Kimi Team, "Kimi K3: Open Frontier Intelligence" (arXiv 2607.24653, 2026-07); Qiu et al., "Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free" (Qwen, arXiv 2505.06708, NeurIPS 2025); Kazemnejad et al., "The Impact of Positional Encoding on Length Generalization in Transformers" (arXiv 2305.19466, NeurIPS 2023); NVIDIA, "Nemotron 3.5 Super and Lightning" (arXiv 2610.38095, 2026-10); Llama 4 的 iRoPE 见 Meta 的官方博客。层代码: `llm_models/layers/sparse/linear_attention.py`、`llm_models/layers/core/gated_attention.py`, 模型代码: `llm_models/models/language_models/hybrid.py`。

```bash
python -m llm_models.run_models.language_models.hybrid.infer_hybrid   # 约 3 s, 不训练
python -m llm_models.run_models.language_models.hybrid.train_hybrid   # 约 112 s, 5 个模型各 1200 步
```

## 运行后应该看到什么

`infer_hybrid` (单线程 CPU 实测, 4 层、d_model = 64、4 个头、头维 16、MLA latent 32 + rope 16、分块 8):

```
KDA 通道门全相同时 vs GatedDeltaNet: 最大差 3.0e-08 | 同一头内各通道门的最大跨度 0.0e+00
默认初始化的 KDA: 同一头内各通道门的最大跨度 0.506

GatedAttention(use_gate=False) vs GroupedQueryAttention: 最大差 0.0e+00 | 第 0 列分走的注意力质量 0.0992 (初始化时接近均匀)

一条序列的缓存元素数 (实测 vs 算式):
  Kimi Linear / K3 (3:1)       T =  32 | 实测   4608 | 算式   4608 (随 T 增长 1536, 定长状态 3072)
  Kimi Linear / K3 (3:1)       T =  64 | 实测   6144 | 算式   6144 (随 T 增长 3072, 定长状态 3072)
  Qwen3-Next / Qwen3.5 (3:1)   T =  32 | 实测   7168 | 算式   7168 (随 T 增长 4096, 定长状态 3072)
  Qwen3-Next / Qwen3.5 (3:1)   T =  64 | 实测  11264 | 算式  11264 (随 T 增长 8192, 定长状态 3072)
  Llama 4 iRoPE (3:1)          T =  32 | 实测  16384 | 算式  16384 (随 T 增长 16384, 定长状态 0)
  Llama 4 iRoPE (3:1)          T =  64 | 实测  32768 | 算式  32768 (随 T 增长 32768, 定长状态 0)
  全注意力基线                      T =  32 | 实测  16384 | 算式  16384 (随 T 增长 16384, 定长状态 0)
  全注意力基线                      T =  64 | 实测  32768 | 算式  32768 (随 T 增长 32768, 定长状态 0)

iRoPE 排布 gqa_chunk gqa_chunk gqa_chunk gqa_nope: 分块大小 8
  分块 RoPE 层: 位置 31 能看到 8 个 key (块内), 全长是 32
  NoPE 全注意力层: 位置 31 能看到 32 个 key

逐 token 增量生成 vs 整段前向:
  Kimi Linear / K3 (3:1)       kda kda kda mla_nope                     最大差 6.0e-07
  Qwen3-Next / Qwen3.5 (3:1)   gdn gdn gdn gated_gqa                    最大差 3.6e-07
  Llama 4 iRoPE (3:1)          gqa_chunk gqa_chunk gqa_chunk gqa_nope   最大差 4.8e-07
  全注意力基线                      gqa gqa gqa gqa                          最大差 3.6e-07
```

- KDA 的通道门全取同一个值时与 Gated DeltaNet 逐位相同 (3.0e-08); 默认初始化下同一头内各通道的门跨度是 0.506, 确实在各自衰减。
- 关掉输出门的 `GatedAttention` 与 `GroupedQueryAttention` 完全相同 (差 0.0), 所以后面 "有门 / 无门" 的对照只差那一次乘法。
- 四种排布的缓存元素数与算式逐个相等。T 从 32 翻到 64 时, Kimi 式只从 4,608 长到 6,144 (定长状态 3,072 不动), 全注意力基线从 16,384 翻到 32,768。
- 分块 RoPE 层每个 query 只看块内 8 个 key; 注意本库的分块只改可见性, K/V 仍然整条缓存着, 所以 iRoPE 排布的缓存和全注意力基线一样大。真实实现会像滑动窗口那样滚动裁剪。

断言: KDA 通道门全相同时与 GDN 一致、默认初始化下各通道的门确实不同、关门时与 GQA 逐位相同、四种排布两个长度下的缓存元素数与算式相等、分块层的可见 key 数不超过块大小且跨块不可见、NoPE 全注意力层看到全部历史、四种排布的增量生成与整段前向一致。

`train_hybrid`: 植入针的检索任务, 序列前半段散落着 4 对 (KEY, 值), 末尾依次给出 4 个 KEY, 只有 4 个答案位算 loss。KEY 与值每条序列都重抽。训练长度 32, 评估长度 32 / 64 / 128; 序列越长针离问话越远。seed 0 实测:

```
植入针检索任务: 4 对 (KEY, 值) 每条序列重抽, 词表 20, 训练长度 32, 分块大小 16; 初始 loss 应 ≈ ln 20 = 2.996
  gqa (无输出门)              | 参数 214,848 | 初始 loss 3.084 → 0.000 | 准确率 T=32: 1.000 T=64: 0.105 T=128: 0.062 | 第 0 列注意力质量 0.0812 | 12s
  gated_gqa (有输出门)        | 参数 231,232 | 初始 loss 3.152 → 0.000 | 准确率 T=32: 1.000 T=64: 0.082 T=128: 0.102 | 第 0 列注意力质量 0.1103 | 13s
  全 NoPE 全注意力            | 参数 214,848 | 初始 loss 3.084 → 0.003 | 准确率 T=32: 1.000 T=64: 0.969 T=128: 0.824 | 第 0 列注意力质量 0.0644 | 11s
  iRoPE (分块 RoPE + NoPE)   | 参数 214,848 | 初始 loss 3.082 → 2.322 | 准确率 T=32: 0.086 T=64: 0.094 T=128: 0.141 | 第 0 列注意力质量 0.0752 | 12s
  Kimi 式 (KDA + NoPE MLA)   | 参数 243,468 | 初始 loss 3.150 → 0.051 | 准确率 T=32: 0.996 T=64: 0.969 T=128: 0.973 | 61s

A 输出门与 attention sink (4 层平均, 在训练长度 32 上测; 均匀分布的期望是 0.0987):
  无输出门 第 0 列分走 0.0812 | 有输出门 0.1103
  逐层 (无门): [0.0373, 0.1179, 0.096, 0.0738]
  逐层 (有门): [0.0708, 0.1737, 0.0988, 0.0979]

B 超出训练长度后的准确率 (训练长度 32):
  gqa (无输出门)              T=32: 1.000  T=64: 0.105  T=128: 0.062  | T=128 相对 T=32 0.06×
  全 NoPE 全注意力            T=32: 1.000  T=64: 0.969  T=128: 0.824  | T=128 相对 T=32 0.82×
  iRoPE (分块 RoPE + NoPE)   T=32: 0.086  T=64: 0.094  T=128: 0.141  | T=128 相对 T=32 1.64×
  Kimi 式 (KDA + NoPE MLA)   T=32: 0.996  T=64: 0.969  T=128: 0.973  | T=128 相对 T=32 0.98×
```

- **位置编码决定外推**: 全 RoPE 的全注意力在训练长度上满分, 长度一翻倍就塌到 0.105、再翻倍到 0.062。全 NoPE 同样在训练长度上满分, 到 128 还有 0.824。Kimi 式 (3 层 KDA + 1 层 NoPE 的 MLA) 保持得最好, 128 上 0.973, 几乎不掉。这与 Kazemnejad 等人的结论方向一致: RoPE 在没训练过的相对距离上不可靠, 不加位置编码反而外推得更稳。
- **iRoPE 的排布在这个玩具上学不会**: 3 层分块 RoPE 加 1 层 NoPE 全注意力, 1200 步后训练长度上只有 0.086, loss 停在 2.322。分块大小 16 时针和问话几乎总在不同块里, 只有最后那一层能跨块, 信息来不及先在块内整理好。Llama 4 Scout 用 10M 上下文的这套排布, 和这里 4 层、32 长度的玩具不是一个量级, 这个结果说明不了那套排布不行。
- **输出门没有减少 attention sink, 因为这个规模上没有 sink**: 均匀分布下第 0 列的期望是 0.0987, 实测无门 0.0812、有门 0.1103, 两者都在均匀水平附近。论文报告的 46.7% → 4.8% 是在真实规模的预训练模型上; 玩具上既没有 46.7% 那样的集中, 门也就没有可减的东西, 实测带门的那一组反而略高。
- **边界**: 单种子, 4 层、d_model = 64、合成检索任务, 训练长度 32。各家论文的收益都在数十亿到上万亿参数、数十万到百万 token 的上下文上报告, 这里只验证机制与缓存算式。

断言只覆盖稳的结论: 五种排布的初始 loss 都 ≈ ln 20、训练后 loss 都明显下降、至少一种排布在训练长度上学会任务、注意力质量落在 0 到 1 之间、全 NoPE 在训练长度外的保持率明显好于全 RoPE (0.82× 对 0.06×)。谁的绝对准确率更高不做断言。

## 与真实系统的差距

- **规模**: 4 层、d_model = 64、训练长度 32。Kimi Linear 是 48B 总参数 / 3B 激活、1M 上下文; Kimi K3 是 2.8T 总参数、93 层; Qwen3.5 是 397B / 17B、60 层、原生多模态; Llama 4 Scout 声称 10M 上下文。
- **KDA 只改了门**: 论文的 q / k / v 还各过一层 ShortConv 加 Swish, α 的 logit 由低秩投影加每头一个偏置生成, 再经一个有下界的映射压到 (0, 1)。本库沿用 `GatedDeltaNet` 的参数化, 只改门的粒度, 这样 "通道门全相同时退化为 GDN" 这条断言才是严格的。
- **没有 Attention Residuals**: Kimi K3 还把残差连接换成了层间注意力 (每层用一个可学的伪 query 在之前所有层的输出上做 softmax)。本库的 `hyper_llama.py` 实现的是同类思路的 Hyper-Connections 与 mHC, 两者不是同一个方法。
- **没有 MoE**: Qwen3.5 与 Kimi K3 都是稀疏 MoE, 本库的 MoE 在 `moe/` 下单列。
- **分块只改可见性**: `gqa_chunk` 仍然缓存整条 K/V, 所以 iRoPE 排布的缓存没有省。真实实现会像 Mistral 的滑动窗口那样滚动裁剪。
- **没有推理时的注意力温度缩放**: Meta 只说 "inference time temperature scaling of attention", 具体公式来自实现而不是官方描述, 本库不猜。
- **NoPE 的 MLA 层仍留着 rope 段**: 本库的 MLA 把 latent 和 rope 段写在一个投影里, NoPE 时那一段不旋转但照样缓存。真实实现会去掉, 每 token 再少 rope 个数。
- **没有 Mamba 混合器**: Nemotron 3.5 与 Jamba 的便宜层是 Mamba, 本库的 Mamba 在 `mamba.py` 里单列一章。
- **本库约定**: Pre-LN、weight tying、embedding 乘 √D、N(0, 0.02²) 初始化, 不代表原模型的做法。

## 常见误区

1. **"KDA 就是 Gated DeltaNet 换了个名字"**: 两者的递推式只差门的粒度和顺序, 但这个差别不是零: α 是向量时 `Diag(α)` 与 `(I − β k k^T)` 不交换, 先衰减再写入和先写入再衰减不是同一件事。通道门全相同时才完全相同 (实测 3.0e-08)。
2. **"混合架构一定省缓存"**: 省的是随长度增长的那一部分。T = 32 时 Kimi 式的 4,608 个元素里有 3,072 是线性层的定长状态, 占三分之二; 要到长上下文线性项占优, 收益才显出来。
3. **"分块注意力能省 KV cache"**: 分块只改可见性。本库的 `gqa_chunk` 和全注意力层缓存一样大; 要省就得像滑动窗口那样真的把旧 K/V 丢掉, 那是另一个机制。
4. **"NoPE 没有位置信息, 所以记不住顺序"**: 因果掩码本身携带顺序, 别的层 (KDA 的衰减门、分块 RoPE 层) 也会把顺序带进残差。实测全 NoPE 在训练长度上同样做到 1.000。
5. **"输出门是为了省算力"**: 它只多一次投影和一次乘法, 算力是增加的。它解决的是注意力被迫把概率质量分完这个结构问题, 顺带让大激活变小、BF16 训练更稳。
6. **"玩具上 iRoPE 学不会, 说明 Llama 4 的排布不行"**: 4 层、训练长度 32、分块 16 的配置下针和问话几乎总跨块, 只有最后一层能跨块取信息。真实模型有几十层、块长 8K, 这个结果外推不过去。

## 自测题

1. KDA 与 Gated DeltaNet 的递推式差在哪? 什么条件下两者完全相同?
   **答**: GDN 是 `α_t (I − β_t k k^T) S + β k v^T`, α 是每头一个标量; KDA 是 `(I − β_t k k^T) Diag(α_t) S + β k v^T`, α 是每个 key 通道一个。α 的各通道取同一个值时标量与矩阵可交换, 两式相等, 实测最大差 3.0e-08。
2. 为什么 Kimi Linear 把全注意力层做成不加位置编码的 MLA, 顺序信息从哪来?
   **答**: 顺序由中间那几层 KDA 的衰减门隐式携带 (越旧的信息衰减越多), 加上因果掩码本身的方向。好处是扩上下文时不用调 RoPE 的频率基数、也不用上 YaRN, 因为全注意力层本来就没有位置参数要改。
3. 一个 12 层模型, 3 层线性配 1 层全注意力, 头维 16、4 个头、全注意力层每 token 存 128 个数。T = 32 和 T = 1024 时缓存各是多少? 定长状态各占多少?
   **答**: 9 层线性 (每层状态 4 × 16 × 16 = 1,024) 加 3 层全注意力 (每层 128 · T)。T = 32 时是 9,216 + 12,288 = 21,504, 定长占 42.9%; T = 1024 时是 9,216 + 393,216 = 402,432, 定长占 2.3%。所以混合架构的收益是渐近的。
