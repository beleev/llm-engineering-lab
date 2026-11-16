# 评估体系 · 用户环境与裁判校准

[![评估体系 · 用户环境与裁判校准](../../docs/screenshots/agent-eval-suite-coverage.png)](https://beleev.github.io/llm-engineering-lab/#/agent/eval-suite)

[Agent 目录](../README.md) · [交互实验](https://beleev.github.io/llm-engineering-lab/#/agent/eval-suite)

## 直觉

通过率从 1/3 涨到 3/3, 我怎么知道是机制变好了, 还是题变简单、裁判变松了?

同一份题集 (suite 191d7ed87046) 跑三种条件: baseline 1/3, 只开授权门 2/3, 再开澄清 3/3; 分层报才看得出 safety 是授权门救的, hard 是澄清救的。裁判也要校准: 按长度判 4 条标注只对 1 条, 按 rubric 判 4 条全对。

## 核心原理

### 一道题不只是一段提示

一道题 = 隐藏账户 + 初始账户 + 是否允许写 + 难度层, 三道题各占一层: clear (easy) 初始账户就是 A; ambiguous (hard) 初始账户为空, 真实账户 B 只在 UserSimulator 里; denied (safety) 标了 allowed=False, 写了就是授权失败。

模拟器只回答 which-account, 别的问题一律返回 unknown, 策略拿不到它没问到的状态。

终态还要过 m21 的验收器: baseline 不许修复 (max_repairs=0), 环境里缺 gamma 就判未完成; 另两种条件允许修 1 次, 通过。

### 消融只开关机制, 题集不动

三种条件跑同一份 TASKS, suite id 都是 191d7ed87046。

- baseline (不澄清、不管授权): 1/3, hard 题 wrong-object, safety 题 authorization。
- no-clarify (只开授权门): 2/3, 只剩 hard 题挂。
- full (再开澄清): 3/3, hard 题 user_turns=1。

分层才看得出 safety 是授权门救的、hard 是澄清救的。换一份题集 suite id 就变, 两次结果不能当同一次 A/B。

### 裁判自己先过一遍标注

校准用 4 条预先标注的夹具, 每条记 (supported, scoped, length, label)。按长度判 (length > 2 算好) 一致率 0.25: 1 个 tp、2 个假阳性、1 个假阴性, 长而错的两条都被判成好。

rubric 判 supported 且 scoped, 一致率 1.0, 假阳性 0。baseline 用的是长度裁判, 另两种条件用 rubric。

真实 LLM judge 还要拿人工样本校准, 并交换候选顺序再判一次。

### 每道题记调用数和失败类别

每道题记 calls 和 failure: 三种条件的调用数是 3 / 2 / 3, full 多出的 1 次是澄清; failure 分 wrong-object 和 authorization, 模型失败和基础设施失败不混在一起。

真实系统还要记延迟和实报 usage; 机制开关和目标人群开关分开, 配对重采样、样本量和效果阈值一起解释。

### 模拟器也可能教坏策略

这里的 UserSimulator 只有一条规则: 问 which-account 就给隐藏账户, 否则 unknown。策略学会的只是 "问这一句", 换种说法的用户它没见过。

真实用户语气、工具延迟和错误分布可以做域随机化, 但保真度要拿真实案例校准。隐私日志只留必要字段; 失败案例进训练前先查泄漏和授权。

### 评估集需要长期维护

样本来自真实失败、人工设计或合成任务, 各有各的偏差。每道题标上来源、难度、时间和适用人群, 按同源任务分组拆分, 留一份封闭测试集。

本章 3 道题在 full 条件下已经 3/3: 一套题长期全对, 先怀疑饱和或泄漏, 不是能力无限涨。

### 读代码的顺序

- `demo.py:UserSimulator`
- `demo.py:run_task`
- `demo.py:calibration`

## 运行

```bash
python -m llm_agent.m31_eval_suite.demo
```

默认只用标准库与本仓库模块, 不访问外网。

## 运行后应该看到什么

依次运行 `baseline, no-clarify, full` 三种条件, 输出结果与逐步轨迹。`main()` 对预期的成功和失败分别断言; 失败对照不是程序故障。网页可选择配置、播放或点击事件, 显示 Python 已运行记录, 不在浏览器里执行训练或工具。

## 与真实系统的差距

用户是 1 条规则的模拟器, 校准标签是 4 条手写夹具; 没有真实人工评审, 也没有线上 A/B。

调用数来自执行 (3 / 2 / 3), 延迟和费用没有数, 不编。只有 3 道题, 一道就是 1/3, 差异谈不上显著。

## 常见误区

“有相同概念的教学实验”不等于“复现完整论文或具备通用产品能力”。对照只隔离本章说明的变量; 其他假设变化需重新测。

## 自测题

删除安全题后总体得分上涨, 能否说明新机制更好?

<details><summary>答案</summary>

不能, 任务分布已经改变。固定测试分布和分层指标才能区分机制改善与题目变简单。

</details>

## Trace / span: 把一次任务的等待时间还原出来

`python -m llm_agent.m31_eval_suite.trace_demo` 跑的是真实 Agent → ClaudeLLM → 离线流式客户端, 工具由真实线程并行执行。根 span 对应 Agent.run, 模型请求和工具是它的子节点, API attempt 再挂在模型请求下面。导出的每个 span 带 trace_id、id、parent、单调时钟区间、状态、错误类别和实报 usage; 不记录提示、工具参数、输出或异常正文。

parallel / retry / cancel 三种情形用同一套数据结构:
- 默认: 两个逻辑模型请求、两次工具执行。
- 限流重试: 模型 attempt 变成三个, 工具仍只执行两次。
- 取消: 根和子 span 全部关闭, 不能留下 running。
线程显式传入父 span, 线程池里 ContextVar 会丢。

端到端时间看根 span。子 span 相加会把父子包含和并行重叠算两遍, 不能当用户等待时间。关键路径要按依赖和实际区间找: 本例两个工具有屏障证明它们重叠, 汇合后才调用模型。`Tracer.export()` 直接导出 JSON 树; 网页回放不固定机器时钟和随机 trace id。

费用只按成功请求的实报 usage 算。演示单价是每千 token 输入 / 输出 / 缓存读 1 / 2 / 0.25 单位, 不是供应商报价; 失败请求记账不完整, 不能说总账单已知。model 和 attempt 都附 usage 方便查看, 汇总只取一层, 不重复加。每个 Agent 实例仍是单会话状态机, 不能让多个任务并发共享。

边界: 标准库内存追踪, 没有 OTLP、采样、持久化或后端检索。正式服务还要管留存和授权。自测: 两个并行工具各花 100 ms, 用户是不是等了 200 ms? 不是, 先看它们的时间重叠和上下游的串行依赖。
