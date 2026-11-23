# QAT: 带着量化误差训练, 让低位宽部署不掉点

[llm_finetune 目录](../../README.md) · [交互实验](https://beleev.github.io/llm-engineering-lab/#/finetune/qat)

## 直觉

训练后量化 (RTN、GPTQ、AWQ, 见推理阶段的 [量化章](https://beleev.github.io/llm-engineering-lab/#/infer/compute)) 是模型训完再压, 模型从来没在量化误差下被训练过, 位宽低到一定程度就掉点。QAT (quantization-aware training, 量化感知训练) 在训练时就让每次前向用 "量化再反量化" 之后的权重, loss 看到的就是部署时的模型。

问题在反向: 取整的导数几乎处处为 0, 梯度传不回去。直通估计器 STE (straight-through estimator, Bengio et al. 2013 起的名字) 在反向时把取整当成恒等函数, 梯度原样交给高精度的影子权重。

它和本阶段的 [QLoRA](../qlora/README.md) 不同: QLoRA 把冻结的基座压成 NF4, 基座不训练, 只训高精度的 LoRA 支路; QAT 训练的就是将要被量化的那份权重。

真实系统里的例子:

- LLM-QAT (arXiv 2305.17888) 用预训练模型自己生成的数据做蒸馏, 同时量化权重、激活和 KV cache; LLaMA-30B 在 4-8-4 (权重 4 位、激活 8 位、KV 4 位) 下只比全精度低 1.5 分。
- Google 的 Gemma 3 QAT 版本 (2025-04-18 官方博客) 以非量化 checkpoint 的概率为目标做了约 5,000 步 QAT, 量化到 Q4_0 时 perplexity 的下降减少 54%; 27B 的权重从 54 GB (BF16) 降到 14.1 GB。
- Moonshot 的 Kimi K2 Thinking 官方说明: 后训练阶段做 QAT, 对 MoE 部分做 INT4 weight-only 量化, 原生 INT4 推理的生成速度约快 2 倍, 所有基准分数都在 INT4 下报告。

## 核心原理

### 分组对称 INT4 与假量化

每 g 个权重一组 (沿输入维), 共用一个 scale:

```
s = max|w_group| / q_max          q_max = 2^(bits−1) − 1, INT4 为 7
q = clamp(round(w / s), −q_max, q_max)
ŵ = q · s                         假量化: 量化再反量化, 前向用 ŵ
```

每个权重的存储是 bits + 16/g 位 (每组一个 fp16 scale)。组越小, 一个离群值影响的权重越少, 但 scale 的开销越大。

### STE

```python
def fake_quant(w, bits, group_size):
    return w + (quantize_dequantize(w, bits, group_size) - w).detach()
```

前向值是 ŵ; 反向时 detach 那一项没有梯度, 只剩 w 的恒等路径, 所以 ∂L/∂w = ∂L/∂ŵ。`FakeQuantLinear` 接管原来的 nn.Linear 参数, 前向用 fake_quant 后的权重; 训完 `rtn_quantize_` 把影子权重取整成普通 nn.Linear(ŵ) 部署。训练最后一步的前向和部署后的前向是同一个 ŵ。代码在 [methods/qat.py](../../methods/qat.py)。

## 运行

```bash
python -m llm_finetune.run_finetune.qat.train_qat    # 单线程约 15 s
```

## 运行后应该看到什么

```
---- STE: 梯度怎么穿过取整 ----
fake_quant: 前向值 = ŵ (True), ∂L/∂w 形状 (4, 8), 与上游梯度最大差 0.0e+00
直接对 round 求导: ∂L/∂w 里只有 8/32 个非零   ← 每组 absmax 那一个经 scale 拿到梯度, 其余的梯度在取整处断掉

---- 训练后量化 RTN: fp32 基线 (reverse, SFT 600 步) 留出集 EM 1.000 ----
                     g=8            g=32            g=64
INT4        0.996 (6.0b)    0.996 (4.5b)    0.996 (4.2b)
INT3        0.980 (5.0b)    0.973 (3.5b)    0.945 (3.2b)
INT2        0.559 (4.0b)    0.176 (2.5b)    0.004 (2.2b)

---- QAT: 从 fp32 基线再训 100 步 (lr=0.001), 训完 RTN 部署 ----
                直接 RTN     多训再 RTN   QAT 再 RTN
INT4 g=32        0.996       0.988       0.988   (训练与部署 logits 最大差 0.0e+00)
INT2 g=8         0.559       0.484       0.977   (训练与部署 logits 最大差 0.0e+00)
INT2 g=32        0.176       0.164       0.961   (训练与部署 logits 最大差 0.0e+00)
INT2 g=64        0.004       0.000       0.953   (训练与部署 logits 最大差 0.0e+00)
```

(训练日志的空行已省略。括号里是每个权重的平均位数 bits + 16/g。)

怎么读:

- **STE 让梯度原样通过**: fake_quant 的梯度与上游梯度逐位相同。不用 STE 时, ŵ = round(w/s)·s 只通过 scale 依赖 w, 32 个权重里只有每组 absmax 那 8 个拿到梯度。
- **这个规模下 INT4 RTN 几乎不掉点**: 各种组大小都是 0.996。d_model=64 的小模型权重分布窄, 没有离群值, 15 个格点够用。真实 LLM 的 INT4 RTN 会掉点, 原因是激活和权重里的离群值, 这里没有。所以 QAT 的效果在 INT2 上看。
- **INT2 RTN 掉得很厉害, 组越小越好**: g=8 / 32 / 64 分别是 0.559 / 0.176 / 0.004。
- **QAT 100 步把 INT2 拿回来**: g=32 从 0.176 到 0.961。对照组同样多训 100 步但不加假量化, 再 RTN 只有 0.164, 说明恢复来自 "训练时看见了量化误差", 不是来自多训了几步。
- **QAT 之后组大小几乎不重要**: g=64 也有 0.953, 每个权重只要 2.2 位。
- **INT4 上 QAT 没有额外收益**: 0.988 对直接 RTN 的 0.996, 差别在评估噪声内。

断言 (每条都在脚本末尾):

1. fake_quant 的梯度形状与权重相同, 数值等于上游梯度; 不用 STE 时非零梯度的个数等于组数。
2. INT4 RTN (g=32) 的 EM 降幅 ≤ 0.02。
3. INT2 RTN (g=32) 的 EM 降幅 > 0.5。
4. INT2 下 RTN 的 EM: g=8 > g=32 > g=64。
5. QAT (INT2, g=32) 部署后的 EM 比直接 RTN 高 0.5 以上, 也比 "多训再 RTN" 的对照高 0.5 以上。
6. 每种配置下 QAT 训练形态与部署形态的 logits 最大差 < 1e-5。

## 与真实系统的差距

- **只量化权重**: LLM-QAT 还量化激活和 KV cache, 激活量化要处理离群通道, 难度高得多。
- **scale 不学习**: 每步按当前影子权重的 absmax 现算。LSQ 一类方法把 scale 也当参数训, 也有方法裁掉离群值再定 scale。
- **用任务数据做 QAT**: LLM-QAT 用模型自己生成的数据做蒸馏, Gemma 3 以非量化 checkpoint 的概率为目标。这里直接用 SFT 数据和交叉熵。
- **对称 [−7, 7]**: 真实 INT4 kernel 常用非对称或 [−8, 7], 还要考虑打包和反量化融进 GEMM。这里的 ŵ 是 fp32 张量, 没有真正的 INT4 存储和 kernel。
- **小模型对 INT4 不敏感**: 这是规模带来的边界, 不能读成 "INT4 不需要 QAT"。
- **单种子、reverse 任务、2 层模型**。

## 常见误区

- **QAT 就是训练完再量化**: 那是 RTN / GPTQ 这类训练后量化。QAT 的前向在训练中就用量化后的权重。
- **round 不可导, 所以 QAT 必须用特殊优化器**: STE 一行就够, w + (ŵ − w).detach()。
- **QAT 和 QLoRA 一样**: QLoRA 的量化基座冻结, 不进训练; QAT 训练的正是要被量化的权重。两者可以组合, 但解决的问题不同。
- **QAT 的提升来自多训了几步**: 对照组同样多训 100 步, INT2 g=32 RTN 后只有 0.164。

## 自测题

1. fake_quant 写成 w + (ŵ − w).detach(), 前向值和反向梯度分别是什么? — **答**: 前向值是 ŵ; 对 w 的梯度是 1 (恒等), ∂L/∂w = ∂L/∂ŵ。
2. 不用 STE 直接对 round(w/s)·s 求导, 为什么 32 个权重里还有 8 个拿到了梯度? — **答**: s = max|w|/q_max 依赖每组的 absmax 元素, 梯度经过 s 流到这 8 个元素; 其余元素只经过 round, 导数为 0。
3. 本例 INT4 RTN 不掉点, 为什么 Gemma 3 和 Kimi K2 Thinking 还要做 INT4 QAT? — **答**: 大模型有离群值, INT4 RTN 会掉点 (Gemma 3 报告 QAT 把 Q4_0 的 perplexity 下降减少 54%)。2 层 d_model=64 的小模型权重分布窄, 测不出这一点。
