# Agent 训练 · 轨迹到参数更新

[![Agent 训练 · 轨迹到参数更新](../../docs/screenshots/agent-agent-training-coverage.png)](https://beleev.github.io/llm-engineering-lab/#/agent/agent-training)

[Agent 目录](../README.md) · [交互实验](https://beleev.github.io/llm-engineering-lab/#/agent/agent-training)

## 直觉

m25 把合格轨迹导出成 JSON 了, 然后呢? 模型的参数什么时候才真的变?

4 个符号动作、16 个训练实体: SFT 用 5 条示范跑 60 步学会 lookup→answer, 留出的 16 个实体全过; 在线 RL 按温度 3 每步采 8 条、跑 35 步, 参数动了, 留出仍 100%。奖励由 m25 的 execute 判过程, 猜对答案也不算。

## 核心原理

### 奖励来自同一个验收器

奖励不是看答案对不对, 是 m25 的 execute 把整条动作串跑一遍: 答案要对、要先 lookup 拿到观察、不能 steal、不超 budget=3, 全过才 eligible。

小 LLaMA 每步只能从 4 个动作里选: lookup、answer:observed、answer:10、steal。环境只在 lookup 或 steal 之后把 expected 填进观察; 没 lookup 就 answer:10, 哪怕 expected 恰好是 10, reasons 里也是 unsupported, 过程不合格。

### SFT 教协议, 在线 RL 自己采样

SFT 用 5 条示范教协议: 状态 (1, 3) 该出 lookup, 状态 (2, v) 该出 answer:observed, v 取 10..13。标签在 prompt 位置写 -100, 只对动作位算 loss; AdamW lr 0.008, 60 步。

在线策略梯度不看示范: 每步让当前策略在 8 个训练任务上各采 1 条, reward 减组内均值当 advantage, loss 是 -logp × advantage 的均值, 梯度范数裁到 1, lr 0.002, 35 步; 更新的是同一份参数。

### 采样与训练概率必须一致

探索时 logits 除以温度 3 再采样, log_prob 也从同一个温度 3 的 Categorical 里取, rollout 一个函数算两件事。

若采样用温度 3、梯度按温度 1 的分布算, 或者采样和训练用的模型、数值精度不同, 算出来的就是另一个策略的梯度, 行为分布错配。

### 多轮归因有方差

一条轨迹最多 3 步, 奖励只在末端出一个数 (eligible 减 0.05 × 步数), 乘到整条序列 log-prob 之和上; 哪一步做对了没有区分, 能更新但归因粗。

PRM、逐步奖励、critic 和优势估计可以细化信用分配; 奖励越稠密, 越要检查有没有被钻空子的机会。

### 数据与结论的边界

训练用 entity-0..15, 留出用 entity-32..47, 两组不交 (main 里有 isdisjoint 断言); 动作协议和 expected 的取值范围 10..13 一样。

main 断言 before < sft == rl == 1: SFT 后留出 100%, RL 后仍 100%。这只说明协议学会了、RL 没把它弄坏; 不代表能泛化到没见过的工具或自然语言推理, 也不声称 RL 超过 SFT。

### 后续训练方法的连接

偏好对接 DPO (m25 导出的 chosen / rejected 对就是它的输入), 过程标签接 PRM, 在线采样接 PPO / GRPO。

失败轨迹、长动作截断、工具异常和安全约束都要进训练设计; 对模型环境、蒸馏与自蒸馏还要防裁判同源偏差。

### 奖励可以携带诊断

这里的 reward 是一个标量: eligible - 0.05 × calls。execute 其实还返回 step_labels 和 reasons (unsupported / authorization / budget / wrong-result), 训练时被压成一个数。

多目标可以先保留质量、成本、安全的向量, 让生成式诊断解释首错和修复方向; 优化前必须定好组合方式, 不能让更低的成本抵消一次授权违规。RLVP 一类路径约束把过程有效性也纳入验收。

### 经典 RL 与语言动作

这里的动作空间故意退化成 4 个 token (16..19), 像经典离散控制; 真实语言 Agent 一次动作是变长 token 串加工具参数。

execute 把动作语法 (unknown-action)、执行是否成功、终止 (answer 之后的 after-final) 和截断 (budget) 分开记; 不分开, 训练会把无效协议当普通低奖励轨迹。

### 读代码的顺序

- `demo.py:rollout`
- `demo.py:sft`
- `demo.py:online_update`
- `demo.py:training_run`

## 运行

```bash
python -m llm_agent.m32_agent_training.demo
```

本章复用仓库已有的 PyTorch 小 LLaMA, 在 CPU 上实际更新参数; 不访问外部模型 API。

## 运行后应该看到什么

依次运行 `untrained, sft, rl` 三种条件, 输出结果与逐步轨迹。`main()` 对预期的成功和失败分别断言; 失败对照不是程序故障。网页可选择配置、播放或点击事件, 显示 Python 已运行记录, 不在浏览器里执行训练或工具。

## 与真实系统的差距

真的在 CPU 上更新了小 LLaMA 的权重 (d_model=32, 1 层, 词表 24, seed 7), parameter_delta 断言大于 0; 但只有 4 个符号动作、16 + 16 个实体。

不是通用自然语言 Agent 训练基准; 留出满分的 RL 不报告任何泛化收益。本章要 torch, 纯标准库环境跑不了。

## 常见误区

“有相同概念的教学实验”不等于“复现完整论文或具备通用产品能力”。对照只隔离本章说明的变量; 其他假设变化需重新测。

## 自测题

SFT 和 RL 留出得分都满分, 能说明 RL 更强吗?

<details><summary>答案</summary>

不能, 只能说明当前测试未退化。参数更新与任务收益是两件事。当前任务已饱和, 需要更有区分力且独立的测试才能比较能力。

</details>
