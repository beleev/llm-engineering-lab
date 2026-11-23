# 条件深度 — MoD 与 MoR: 每个 token 只走一部分层

[llm_models 目录](../../../README.md) · [交互实验](https://beleev.github.io/llm-engineering-lab/#/models/conditional-depth)

## 直觉

- MoE 在宽度上稀疏: 每个 token 只走 E 个专家里的 k 个。条件深度在深度上稀疏: 每个 token 只走一部分层, 简单的 token 少算, 难的 token 多算。
- **Mixture-of-Depths (MoD)**: 被路由的层前面有一个路由器, 给每个 token 打一个分数; 整段序列里分数前 `k = C·T` 名的 token 过这一层 (注意力和 FFN 都只在它们之间算), 其余 token 直接走残差。`C` 叫容量。
- **Mixture-of-Recursions (MoR)**: 不堆很多不同的层, 而是把一个共享的递归块反复用 `N_r` 次; 每次递归前路由器决定哪些 token 继续, 所以每个 token 的递归深度不同, 参数量却几乎不随 `N_r` 增长。
- 两者共同的麻烦: "整段序列取前 k 名" 要看到后面的 token, 生成时后面的 token 还不存在。

## 核心原理

### MoD 的一层

```
r_i = w_r · x_i                                  # 路由分数, 每个 token 一个标量
S = 整段序列里 r 的前 k = round(C·T) 名           # 入选集合
x_i ← x_i + w(r_i) · (block(x_S)_i − x_i),  i ∈ S  # 入选 token 之间按原始位置做因果注意力
x_i ← x_i,                                  i ∉ S  # 其余 token 走残差
```

路由权重 `w(r_i)` 乘在输出上, 路由器才在梯度路径上、能被训练。论文用原始分数 `r_i`; 本库用 `w = 2·sigmoid(r)`, 分数为 0 时权重恰为 1, 所以容量 100% 时可以与普通 LLaMA 逐位对照。

计算量: 入选 token 被 gather 成 `[B, k, D]` 的张量再过层, 张量只有 k 行, 形状在训练前就知道。每 token 的稠密 FLOPs = 不路由的层 + `C` × 路由的层, 随 `C` 线性下降; 注意力部分按 `C²` 下降。论文最好的配置是隔层路由、`C = 12.5%`。

### 因果问题和预测器

位置 i 是否进前 k 名, 取决于整段序列的分数分布, 包括 i 后面的 token。训练时整段都在, 生成时没有。论文给了两种解法: 在路由器输出上加一个 BCE 辅助损失, 让分数本身的正负就能判断入选; 或者另训一个小 MLP 预测器, 输入是 stop-gradient 的激活, 只预测 "这个 token 会不会进前 k 名", 生成时用它代替 top-k。本库实现后一种, 预测器的损失只更新预测器自己。

### MoR 的递归和 KV cache

本库用论文选的 Middle-Cycle 共享方式: 首尾各一个独立层, 中间一个共享块递归 `N_r` 次。路由用 expert-choice: 第 `r` 次递归只在仍在递归的 token 里取分数前 `k_r = T·(N_r − r)/N_r` 名, 第 0 次全部 token 都过, 落选的 token 不再回来。

```
参数量:   2 个独立层 + 1 个共享块 + N_r 个路由器 (各 D 个参数)
K/V 条数: 第 r 次递归只有入选的 token 存 K/V → Σ k_r = (N_r + 1)/(2N_r) × N_r·T
```

这就是论文的 recursion-wise KV caching。论文的另一种做法 recursive KV sharing 让后面的递归都复用第一次的 K/V, K/V 只存 `1/N_r`, 但读 K/V 的 IO 不省; 本库没有实现。论文另有 token-choice 路由: 一开始就给每个 token 定好递归深度, 天然满足因果, 代价是负载不均, 本库也没有实现。

## 运行

论文: Raposo et al., "Mixture-of-Depths: Dynamically allocating compute in transformer-based language models" (Google DeepMind, arXiv 2404.02258); Bae et al., "Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation" (KAIST / Google DeepMind / Mila, arXiv 2507.10524)。模型代码: `llm_models/models/language_models/conditional_depth.py`。

```bash
python -m llm_models.run_models.language_models.conditional_depth.infer_conditional_depth   # 约 1 s, 不训练
python -m llm_models.run_models.language_models.conditional_depth.train_conditional_depth   # 约 190 s, 4 个模型各 2000 步
```

## 运行后应该看到什么

`infer_conditional_depth` (单线程 CPU 实测):

```
MoD 路由层 [1, 3], 容量 100%, 路由权重 1: 与 LLaMA 的 logits 最大差 3.6e-07
只算入选的 24 行 vs 整段算再屏蔽: 最大差 2.4e-07

每 token FLOPs (4 层, 第 1、3 层路由, T = 64; 按实际过层的 token 数计):
  容量 1.000 | 稠密  427.6K | 注意力  33.3K | 合计  460.9K | 相对 100% 1.000
  容量 0.750 | 稠密  374.4K | 注意力  26.0K | 合计  400.4K | 相对 100% 0.869
  容量 0.500 | 稠密  321.2K | 注意力  20.9K | 合计  342.0K | 相对 100% 0.742
  容量 0.250 | 稠密  267.9K | 注意力  17.7K | 合计  285.6K | 相对 100% 0.620
  容量 0.125 | 稠密  241.3K | 注意力  16.9K | 合计  258.2K | 相对 100% 0.560

只改后 16 个 token, 前 48 个 token 的路由决定变了多少 (8 条序列, 第 1、3 层合计):
  topk      | 前 48 个位置里路由翻转 22 次
  predictor | 前 48 个位置里路由翻转 0 次

MoR (首尾各 1 层独立, 中间 1 层共享递归 N_r 次) vs 同样展开深度的 LLaMA:
  N_r = 1 | 展开深度 3 | MoR 参数 160,960 | LLaMA 参数 160,896 | 共享块 K/V 条数 / 全过 1.000 (公式 1.000)
  N_r = 2 | 展开深度 4 | MoR 参数 161,024 | LLaMA 参数 214,272 | 共享块 K/V 条数 / 全过 0.750 (公式 0.750)
  N_r = 3 | 展开深度 5 | MoR 参数 161,088 | LLaMA 参数 267,648 | 共享块 K/V 条数 / 全过 0.667 (公式 0.667)
  N_r = 4 | 展开深度 6 | MoR 参数 161,152 | LLaMA 参数 321,024 | 共享块 K/V 条数 / 全过 0.625 (公式 0.625)
```

- 4 层里隔层路由、容量 12.5% 时, 每 token FLOPs 是全容量的 0.560, 与 "2 + 2 × 0.125 = 2.25 层当量" 的估算相符; 论文的说法是每次前向的 FLOPs 可以省一半以上。
- 只改后 16 个 token, top-k 路由下前 48 个位置的入选决定翻转了 22 次: 这些 token 的激活一点没变, 变的是 "前 k 名" 的门槛。换成预测器后翻转 0 次。
- MoR 每多递归一次只多 64 个参数 (一个路由器), 同样展开深度的 LLaMA 每多一层多 53,376 个。

断言: 容量 100% 时与 LLaMA 一致、只算入选 token 与整段计算再屏蔽一致、稠密 FLOPs 随容量线性下降、总 FLOPs 单调下降、top-k 路由下翻转次数 > 0、预测器路由下翻转 0 次、MoR 的 K/V 条数比例等于 `(N_r+1)/(2N_r)`、MoR 和 LLaMA 的参数增量如上。

`train_conditional_depth`: 序列由 12 组 `[a, b, =, (a+b) mod 10]` 组成, a、b 随机, 只有 "=" 后面的答案需要真的算。LLaMA-4、MoD-6 (第 1、3、5 层容量 1/3)、MoR (`N_r = 3`) 每 token 的稠密 FLOPs 都是 4 层当量 (脚本按实际过层的 token 数核对), 另跑 LLaMA-3 当参照, 它就是 MoD-6 里不路由的那 3 层。seed 0 实测:

```
理论下界 (只剩 a、b 的随机性) 1.127; 只学会 '=' 时约 1.715
  LLaMA-3 (参照)  | 参数 160,896 | 稠密 FLOPs/token 320.9K | held-out loss 1.563 | 答案准确率 0.210 | 首次 ≥ 0.9 的步数 未达到 | 32s
  LLaMA-4       | 参数 214,272 | 稠密 FLOPs/token 427.4K | held-out loss 1.136 | 答案准确率 1.000 | 首次 ≥ 0.9 的步数 2000 | 43s
  MoD-6, C=1/3  | 参数 327,555 | 稠密 FLOPs/token 430.0K | held-out loss 1.722 | 答案准确率 0.095 | 首次 ≥ 0.9 的步数 未达到 | 56s
  MoR, N_r=3    | 参数 161,088 | 稠密 FLOPs/token 427.8K | held-out loss 1.136 | 答案准确率 1.000 | 首次 ≥ 0.9 的步数 1500 | 53s

MoD 预测器与 top-k 的路由一致率 0.929 | 预测器路由时实际过层比例 0.293 (top-k 固定 0.333)
MoD 用预测器路由 (采样时可用): held-out loss 1.722 | 答案准确率 0.095
MoD top-k 选中的 token 里 '=' 位置占 0.500 (随机选是 0.255)
```

(每个模型下面还有一行每 250 步的答案准确率, 这里省略。)

- 这个任务的学习曲线有一段平台, 之后突然学会。seed 0 上 MoR 最先学会 (1500 步), LLaMA-4 在 2000 步, MoD-6 和 LLaMA-3 在 2000 步内都没学会。
- 换 seed 1 (`main(seed=1)`) 再跑一次: MoR 1250 步、LLaMA-4 1500 步、MoD-6 和 LLaMA-3 都是 2000 步 (答案准确率 0.955 和 0.961)。两个种子上排序一致: MoR 最快, MoD 不快于只有它那 3 个不路由层的 LLaMA-3。
- MoD 的路由器学到了偏向答案位置: top-k 选中的 token 里 "=" 位置占 0.500, 随机选是 0.255。但这没有让它更快学会求和。
- 预测器与 top-k 的一致率 0.929 (seed 1 是 0.902), 用预测器路由时的 loss 与 top-k 相同 (seed 1: 1.228 对 1.229); 论文报告的预测器准确率在 97% 以上。

断言只覆盖稳的结论: 各模型初始 loss ≈ ln 11、训练后 loss 下降、三个对照模型的稠密 FLOPs 相差 2% 以内、预测器一致率 > 0.5。谁先学会不做断言。

## 与真实系统的差距

- **规模和任务**: 3 到 6 层、d_model=64, 合成的求和任务, 两个种子。论文在几亿到十几亿参数、真实语料上比较等 FLOPs 的 loss; MoD 论文报告等训练 FLOPs 下能持平或略好于基线, 这里的玩具没有复现出 MoD 的收益。
- **路由权重**: 本库用 `2·sigmoid(r)`, 论文用原始分数 `r`。
- **因果解法只有一种**: 只实现了辅助预测器, 没有路由器上的 BCE 辅助损失。
- **MoR 只实现了一部分**: 递归块只有 1 层; 只有 expert-choice 路由和 recursion-wise caching 的计数, 没有 token-choice 路由和 recursive KV sharing。MoR 的 expert-choice 有与 MoD 相同的因果问题, 论文用辅助路由器或辅助损失, 这里没有处理。
- **没有生成**: 两个类都没有带 KV cache 的 `generate()`。跳过某层的 token 在那一层没有 K/V, 真实实现要按层分别管理缓存。
- **FLOPs 是估算**: 按矩阵参数和注意力对数计, 没有测墙钟时间; 训练脚本打印的秒数受机器负载影响, 只作参考。
- **本库约定**: weight tying、embedding 乘 √D、N(0, 0.02²) 初始化, 不代表原模型的做法。

## 常见误区

1. **"MoD 就是给每个 token 加一个门, 门小于阈值就跳过"**: 训练时的门槛是整段序列的第 k 名, 每段的入选个数固定为 k, 张量形状才固定; 用固定阈值会让每段入选个数不定。这也正是因果问题的来源。
2. **"top-k 只用到每个 token 自己的分数, 所以是因果的"**: 每个 token 的分数确实只依赖前文, 但 "前 k 名" 的门槛依赖整段序列。实测改后 16 个 token, 前 48 个位置的入选决定翻转 22 次。
3. **"跳过的层就是 dropout"**: 落选 token 走的是残差, 输出原样保留, 不是置零; 入选 token 的输出还乘了路由权重, 这条路径给路由器提供梯度。
4. **"MoR 参数少, 计算也就少"**: 参数共享省的是参数和显存, 每 token 的计算量由递归次数和容量日程决定。本例 MoR 参数只有 LLaMA-4 的 0.75 倍, 稠密 FLOPs 与它相同。
5. **"省 FLOPs 一定省时间"**: 路由本身要排序、gather 和 scatter, 入选个数少时矩阵也小, 硬件利用率会下降。本库只按矩阵大小估算 FLOPs, 没有测墙钟时间。

## 自测题

1. 4 层模型隔层路由、容量 12.5%, 每 token 的稠密 FLOPs 大约是全容量的多少? 为什么注意力部分降得更多?
   **答**: 2 个全量层加 2 × 0.125 个路由层, 共 2.25 层当量, 约 0.56 (实测合计 0.560)。注意力的计算量与过层 token 数的平方成正比, 容量降到 C 时注意力部分按 C² 下降。
2. MoD 的辅助预测器为什么要用 stop-gradient 的输入?
   **答**: 预测器只负责在生成时模仿 top-k 的决定, 它的损失不应反过来改主模型的表示; 输入 detach 以后, 预测器的 BCE 只更新预测器自己的参数, 主模型的训练目标不变。
3. MoR 用 expert-choice、`N_r = 3`、容量日程 1、2/3、1/3 时, 共享块里缓存的 K/V 条数是每次递归都全过的多少? 和 recursive KV sharing 比呢?
   **答**: (3 + 1)/(2 × 3) = 2/3 (实测 0.667)。recursive KV sharing 只存第一次递归的 K/V, 是 1/3, 但后面每次递归仍要读全部 K/V, 读取的 IO 不省。
