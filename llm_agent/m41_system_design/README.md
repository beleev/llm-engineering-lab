# 系统选择 · 工作流、状态与缓存

[![系统选择 · 工作流、状态与缓存](../../docs/screenshots/agent-system-design-coverage.png)](https://beleev.github.io/llm-engineering-lab/#/agent/system-design)

[Agent 目录](../README.md) · [交互实验](https://beleev.github.io/llm-engineering-lab/#/agent/system-design)

## 直觉

固定流程便宜但做不完, 放开 Agent 做得完但贵, 怎么分? 前缀改了, 后缀的 KV 还能接着用吗?

有清单走 workflow, 没清单走 Agent, 歧义和高风险交人: hybrid 4 个请求全处理完, 费用 4 单位; 全 Agent 也是 4 个, 费用 6。进度用状态栏单独记; 前缀一换, KV 别拼。

## 核心原理

### 先画观察、决策和行动边界

choose 按 3 个布尔特征给 4 个请求分路: known_steps、ambiguous、high_risk。

fixed-report 有清单走 workflow, investigate 没清单走 Agent, ambiguous-owner 和 transfer 直接交人。

三种模式跑同样 4 个请求: 全 workflow 处理 3 个、费用 2; 全 Agent 处理 4 个、费用 6; hybrid 处理 4 个、费用 4。

费用是教学模型 (human 0 / workflow 1 / agent 3)。真实选型还要看工具行为、质量、成本、延迟和可恢复性。

### 框架与 Harness 的职责

框架给编排和接口, choose 这种路由函数就是编排。Harness 管权限、上下文、预算、恢复和验收, demo 里的 status_bar 和 cache_experiment 都是 Harness 的活。

选框架看它的接口能不能撑起这五件事, 不按产品名判断谁更先进。

### 结构化状态栏

status_bar 把目标、已完成项、剩余预算、待核实问题存成 4 个独立字段: goal=完成报告, completed=[检索], remaining_calls=2, pending=[核对引用]。

字段由环境事件更新, 不把模型自述直接当事实; remaining 传负数直接抛 ValueError。

状态栏本身也占上下文, 每轮都要跟着发。

### Prompt Cache 与 KV 不等价

Prompt cache 是服务接口的前缀复用和计费语义; KV 是模型层的中间状态, 两者不是一回事。

demo 用 sha256({model, prefix, positions}) 当缓存键: 前缀从 [1, 2] 换成 [9, 2], 键就不同, reusable=false。

Chat template、工具顺序、位置、模型版本和前缀, 任一变了复用条件就不成立。

### 可编辑上下文必须核对因果依赖

cache_experiment 用两层因果混合模拟 Transformer: 每个位置的状态 = 自身值 + 到它为止的前缀均值, 所以后缀状态依赖前缀。

前缀 [1, 2] 换成 [9, 2], 后缀 [3, 4] 不变: 完整重算得 17.994066, 拼旧后缀得 17.979331, 两个数不相等。

组合 KV 要么模型结构支持, 要么重算验证, 不能只按文本块 id 拼接。

### 人工介入应有明确触发

choose 先查两个触发条件: ambiguous 就先澄清对象, high_risk 就交授权方。ambiguous-owner 和 transfer 在三种模式下都路由到 human, 模式再放开也绕不过。

护栏要覆盖输入、上下文和执行三处。Sidecar 只看结构化动作, 能减少被原文诱导, 但仍要独立的执行约束。

### 上下文学习的机制边界

上下文示例能激活关联、模式匹配和任务适应, 但不更新任何持久参数, 会话结束就没了。

书里的「检索而非推理」是相关机制实验的解释, 不是所有模型和任务的定理。

压缩后还保不保得住关系、否定和约束, 得用具体任务测, 本章 demo 没测这一项。

### 读代码的顺序

- `demo.py:choose`
- `demo.py:status_bar`
- `demo.py:cache_experiment`

## 运行

```bash
python -m llm_agent.m41_system_design.demo
```

默认只用标准库与本仓库模块, 不访问外网。

## 运行后应该看到什么

依次运行 `workflow, agent, hybrid` 三种条件, 输出结果与逐步轨迹。`main()` 对预期的成功和失败分别断言; 失败对照不是程序故障。网页可选择配置、播放或点击事件, 显示 Python 已运行记录, 不在浏览器里执行训练或工具。

## 与真实系统的差距

没模拟的: 路由规则只有 3 个布尔特征, 费用是人为定的 0 / 1 / 3, 不是实测 API 价格, 也不代表普遍的成本排序。

KV 反例是标量因果混合, 17.994066 和 17.979331 差 0.0147 只说明会错, 不能外推到真实 Transformer 的误差大小和性能。

## 常见误区

“有相同概念的教学实验”不等于“复现完整论文或具备通用产品能力”。对照只隔离本章说明的变量; 其他假设变化需重新测。

## 自测题

前缀改了, 可直接复用同文本后缀的 KV 吗?

<details><summary>答案</summary>

不能, 后缀状态可能依赖旧前缀。精确前缀复用和任意片段拼接条件不同。模型版本、位置和因果依赖都需检查。

</details>
