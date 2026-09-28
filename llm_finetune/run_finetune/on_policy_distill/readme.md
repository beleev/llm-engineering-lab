# On-policy 蒸馏 (reverse KL)

```bash
python -m llm_finetune.run_finetune.on_policy_distill.train_on_policy_distill   # ~19 s
```

## 直觉
off-policy 蒸馏只在 teacher / 数据集的前缀上教 student; student 推理时一旦走偏, 就进了从没被教过的状态。
on-policy: 让 student **自己采样**, teacher 在它走过的每个 token 位置上打分。像 RL 一样 on-policy, 又像蒸馏一样每个 token 都有稠密信号 (对整个词表解析求 KL, 不是 REINFORCE)。

## 核心公式
`y ~ π_s(·|x)` (no_grad)，`L = mean_t KL( π_s(·|x,y_<t) ‖ π_teacher(·|x,y_<t) )`
mean_t 是对整个 batch 里所有回复 token 一起取平均 (EOS 之后的位置不算), 不是先在每条回复内平均。
reverse KL = mode-seeking; forward KL = mode-covering。

## 运行后应该看到什么
同一个 teacher (70% 排序 / 30% 照抄)、同一个热身 200 步的 student (1 层 d=48), 再各训 300 步, 留出 prompt:

| | 样本合格率 | forward KL | reverse KL | log π(sort 答案) | log π(copy 答案) |
|---|---|---|---|---|---|
| teacher | 0.781 | 0 | 0 | −0.68 | −1.73 |
| 热身后的 student | 0.010 | 1.075 | 2.552 | −5.02 | −18.07 |
| + off-policy forward KL | 0.059 | **0.880** | 2.069 | −3.30 | **−17.26** |
| + on-policy reverse KL | **0.402** | 1.360 | **0.502** | −1.34 | −33.32 |

- 各自在自己优化的指标上赢 (forward KL: 0.88 vs 1.36; reverse KL: 0.50 vs 2.07)。
- **mode-seeking**: on-policy student 采样出来的回复 40% 合格 (off-policy 只有 6%) —— 容量不够时它选择把多数派答案 (sort) 做好。
- **代价**: teacher 30% 概率的少数派答案 (copy) 被放弃得更彻底 (log π −33 vs −17)。forward KL 两头都留着概率, 结果两头都说不利索。

## 与真实系统的差距
- **不对采样求导是一种有偏近似**: 本库把采样出的前缀当常数, 丢掉 "采样分布随参数变" 那一项梯度。这是 GKD 的做法, 方差小。MiniLLM 保留这一项, 用策略梯度估计。
- **student 的样本占 100%**: 每步的数据全部由 student 自己采。GKD 会把 student 采的数据和数据集里的数据按比例混合, 这里没有混。
- **teacher 必须在线**: 每步要在 student 刚采出的前缀上跑一次 teacher。teacher 的输出没法预先算好存盘, 成本是 "采样 + 2 次前向"。
- **调用方自己冻结 teacher**: `on_policy_distill_step` 只在 `no_grad` 下跑 teacher, 不改它的 `requires_grad` 和 train / eval 状态。脚本里 teacher 由 `train_teacher()` 置成 eval, optimizer 只拿到 student 的参数。
- **采样温度固定为 1, 每条最多采 7 个 token**: 真实回复长短不一, 长回复上 student 越走越偏, teacher 在这些前缀上的打分也越不可靠。
- **teacher 和 student 都很小**: 99,648 对 26,256 参数。"容量不够时放弃少数派答案" 的程度和这个容量差有关, 换一组大小结论的数值会变。

## 常见误区
- 对采样过程回传梯度 / 用 REINFORCE: 不需要。采样只决定 "在哪些状态上算 KL", 梯度来自解析的逐位置 KL。
- 从随机初始化的 student 开始 on-policy: 它采的全是垃圾, teacher 在这些分布外前缀上的输出也没意义。先 SFT / off-policy 热身。
- 把 reverse KL 写反: `KL(p‖q) = Σ p log(p/q)`, **谁在前, 期望就在谁的分布下取**。
- 认为 on-policy 全面更好: 它丢多样性。要覆盖 teacher 的全部模式就用 forward KL 或两者混合 (GKD)。

## 自测题
1. 为什么 reverse KL 不惩罚 "漏掉 teacher 的一个模式"? — 期望在 student 分布下取, student 不去的地方权重为 0。
2. 与 GRPO 相比, 每条样本提供多少监督信号? — GRPO: 整条回复 1 个标量; on-policy 蒸馏: 每个 token 位置一个 V 维分布。
3. 为什么不接通用 Trainer? — 与 GRPO 同理: 训练数据由当前 student 现场生成, 数据生成器得拿到模型本身。
