# 元优化 · 发现树与回放

[![元优化 · 发现树与回放](../../docs/screenshots/agent-meta-optimization-coverage.png)](https://beleev.github.io/llm-engineering-lab/#/agent/meta-optimization)

[Agent 目录](../README.md) · [交互实验](https://beleev.github.io/llm-engineering-lab/#/agent/meta-optimization)

## 直觉

回放历史选出的搜索策略分更高了, 拿到新任务上为什么反而更差?

训练树上 prior 策略 0.9 赢过 breadth 0.8; 换到留出树, prior 只拿 0.2, breadth 0.95。回放只能重组走过的分支, 选出的策略还要过一次留出树。

## 核心原理

### 优化的对象上移一层

三层改的东西不同: 知识更新改模型知道什么, 程序更新改怎么执行, 元优化改怎么存取信息或组织探索。

本章优化的是搜索策略本身: breadth 按发现顺序展开, prior 展开 prior 最高的。候选也可以是 Prompt、工作流、Harness; 机制 (选哪个策略) 和更新载体 (策略是段代码还是段 prompt) 别混在一起。

### 发现树只露前沿, expand 才揭示 score

每个 Node 存 prior、score 和 children。ReplayWorld.options() 只返回前沿节点的 prior, expand 之后才揭示 score 和后继。

策略在 budget = 3 内选 3 个节点展开: breadth 走 root → a → b, prior 走 root → a → c。想绕过去直接 expand('b') 会抛 ValueError "节点尚未被发现或已展开", 离线评估偷看不了未来。

### 回放省成本, 但造不出新分支

回放省下重复调用底层 Agent 的成本: 2 个策略 × 2 棵树 × 3 次展开, 0 次真实调用。

但它只能重组历史里走过的分支, 树里没有的节点, 策略再聪明也发现不了。在线和离线的决策接口都是 options() → expand(), 预算和顺序才可比。

### 元过拟合: 训练树 0.9, 留出树 0.2

训练树上 prior 策略 best 0.9 (root → a → c), breadth 0.8 (root → a → b), 离线选中 prior。

留出树上 a 的 prior 还是 0.9 但 score 掉到 0.1, c 掉到 0.2, b 变成 0.95: prior 策略只拿到 0.2, breadth 拿到 0.95。

历史最优只是历史内的保证。selected 模式保留两个策略、用留出树验证, 报告 cost = 3 和失败。

### 读代码的顺序

- `demo.py:ReplayWorld`
- `demo.py:search`

## 运行

```bash
python -m llm_agent.m39_meta_optimization.demo
```

默认只用标准库与本仓库模块, 不访问外网。

## 运行后应该看到什么

依次运行 `breadth, prior, selected` 三种条件, 输出结果与逐步轨迹。`main()` 对预期的成功和失败分别断言; 失败对照不是程序故障。网页可选择配置、播放或点击事件, 显示 Python 已运行记录, 不在浏览器里执行训练或工具。

## 与真实系统的差距

固定两棵 4 节点的发现树和两个程序策略, 没有在线自动写代码, 也没复现论文规模实验。

prior 是 TREES 里写死的数字, 不假装是模型置信度。留出树故意把 a 的高 prior 配上 0.1 的 score, 元过拟合是构造出来的; 真实搜索里它多严重要另测。

## 常见误区

“有相同概念的教学实验”不等于“复现完整论文或具备通用产品能力”。对照只隔离本章说明的变量; 其他假设变化需重新测。

## 自测题

策略在训练发现树上更好, 能否直接上线替换?

<details><summary>答案</summary>

不能, 还要新任务验证和回滚点。离线树覆盖有限, 策略可过拟合旧分支结构。新任务是独立检验。

</details>
