# 知识蒸馏 (off-policy, forward KL)

```bash
python -m llm_finetune.run_finetune.distill.train_distill   # ~26 s
```

## 直觉
硬标签只说 "答案是这个 token"; teacher 的分布还说了 "其它答案各有多合理"。当同一个输入有多个合理答案时, 一条软标签 ≈ 同一前缀上许多条采样硬标签。

## 核心公式
`L = α·CE(student, y) + (1−α)·T²·KL(p_T^teacher ‖ p_T^student)`，`p_T = softmax(z/T)`。KD 项与 CE 项用**同一个 mask** (label=−100 的位置都不算)。

## 运行后应该看到什么
teacher (2 层 d=64, 99,648 参数) 学的任务有**两种正确答案**: 70% 排序 / 30% 照抄。student 1 层 d=48 (26,256 参数), 各 300 步, 留出 prompt:
- forward KL 在 teacher 的样本上量, reverse KL 在 student 自己的样本上量。
- 两个 KL 都是回复段逐 token 的平均, 只算第一个 EOS (含) 之前的位置。

| | 样本合格率 | forward KL | reverse KL | log π(sort 答案) | log π(copy 答案) |
|---|---|---|---|---|---|
| teacher | 0.781 | 0 | 0 | −0.68 | −1.73 |
| 128 条固定数据 / 硬标签 | 0.053 | 2.293 | 2.934 | −11.06 | −32.24 |
| 128 条固定数据 / 软标签 T=2 | 0.059 | **1.860** | 2.540 | −8.15 | −29.11 |
| 无限新数据 / 硬标签 | 0.039 | 0.906 | 2.122 | −3.71 | −17.10 |
| 无限新数据 / 软标签 T=2 | 0.053 | 0.960 | 1.778 | −3.51 | −18.77 |

**怎么读这张表**:
- 软标签的优势出现在**数据有限**时: forward KL 1.86 vs 2.29, 有断言。
- 数据无限时硬标签自己就能把分布采出来, 两者都卡在 student 的容量上, 差距消失 (0.96 vs 0.91, 软标签甚至略差)。这两行照列在表里, 不设断言。
- 两种 student 的样本合格率都很低 (≈5%): 小 student 在 forward KL 下把概率摊在两种答案之间, 采样时半路串台。这正是 `on_policy_distill` 要解决的。

## 与真实系统的差距
- **teacher 本身就很小**: teacher 99,648 参数, student 26,256 参数, 只压了约 3.8×。真实蒸馏里 teacher 和 student 的差距大得多。
- **teacher 每步在线前向**: `TeacherStudent` 每个 batch 都现算一次 teacher 的 logits。teacher 很大时, 真实流程会先把它的输出离线算好存盘。
- **要拿到 teacher 的完整分布**: 软标签需要 teacher 在整个词表上的 logits。teacher 藏在 API 后面时只能拿到采样出的文本, 那就是表里 "硬标签" 的两行。
- **只蒸馏输出分布**: 没有对齐中间层的隐状态或注意力。
- **"两种正确答案" 是人为构造的**: 每条样本按 70% / 30% 取 sort / copy。真实数据的多解性没有这么干净的比例, 软标签的收益要重新量。
- **只跑了 T=2、α=0.3 一组**: 温度和硬标签权重都没有扫。

## 常见误区
- KD 项不做 mask: prompt / pad 位置也在被蒸馏, 而 CE 项没有 —— 两项优化的不是同一批位置。脚本里把 teacher 在 −100 位置的 logits 打乱, 断言 KD loss 不变。
- 忘了 ×T²: 调 T 会顺带改变 KD 与 CE 的相对权重。
- 评估 KL 时把 EOS 之后的位置也平均进去: student 提前说了 EOS, 后面采到的 token 不属于回复。脚本把这些 token 换成随机数, 断言 KL 不变。
- "蒸馏一定比硬标签好": 见上表后两行。

## 自测题
1. T→∞ 时 KD 项在学什么? — 两边都趋于均匀分布, 梯度 ∝ logits 之差: 退化成对 logits 的 MSE 匹配。
2. forward KL 为什么是 mode-covering? — 期望在 teacher 分布下取: teacher 有质量而 student 没有的地方, log(p_t/p_s) → ∞。
3. 这里的训练数据来自谁? 为什么叫 off-policy? — 来自数据集 / teacher, 不来自 student; student 从没在自己会走到的前缀上被训练。
