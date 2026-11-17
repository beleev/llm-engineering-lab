# Coding Agent · 修改、测试与修复

[![Coding Agent · 修改、测试与修复](../../docs/screenshots/agent-coding-agent-coverage.png)](https://beleev.github.io/llm-engineering-lab/#/agent/coding-agent)

[Agent 目录](../README.md) · [交互实验](https://beleev.github.io/llm-engineering-lab/#/agent/coding-agent)

## 直觉

代码改完了, 模型也说修好了, 为什么还不能算修好?

demo 里一个写死的修复策略 (RepairPolicy, 代替真实模型) 去修 pricing.py 里的折扣函数。

- 改一行不等于修好: 它的第一个候选 return total 只过 1/3 测试。读到失败反馈再改一次才 3/3, 一共 7 次模型调用。
- 修没修好由驱动器自己跑测试判定, 没过就把文件回滚成原样; 只给 4 次调用预算时, 预算用完也回滚。

## 核心原理

### 先建立工作区证据

修复之前先在工作区里建立证据。demo 的工作区是一个临时目录, 里面有两个文件: pricing.py (待修的折扣函数 discounted) 和 check.py (3 个测试用例)。bug 是 return total * (1 - percent): percent 传的是 10 这样的百分数, 算出来是负数。

RepairPolicy 是一个写死的修复策略, 代替真实模型。它不直接改:

1. 先 search, 定位 discounted 在 pricing.py 第 1 行。

2. 再 read, 读到原文。

3. 然后才 patch。

Workspace.patch 的规矩:

- 要替换的旧文本必须在文件里恰好出现 1 次, 否则抛 "旧文本不存在或不唯一, 拒绝模糊替换"。
- 替换后的内容先 compile 一遍确认是合法 Python, 再写盘。
- 修改前后用 difflib 生成 diff, 看得见改了什么。

测试文件也要保护: Workspace 启动时记下 check.py 的 sha256 摘要 (文件内容的指纹), 每次跑测试前先核对, 不一样就不跑。模型改不了测试来让自己 "通过"。

### 测试真的运行

test 工具不让模型自己说 "测过了", 而是真的跑测试。它用 subprocess 启动一个本地 Python 子进程跑 check.py: 命令行参数固定、不经过 shell、5 秒超时。子进程真的 import 修改后的 pricing.py, 跑 3 个用例: (100, 10) 应得 90, (19.99, 15) 应得 16.99, (80, 0) 应得 80。

第一个候选 return total 只是把原价返回: (100, 10) 得 100 而不是 90, 只有 (80, 0) 那条碰巧过, 测试 1/3。策略读到这个失败反馈, 第二次 patch 换成 round(total * (1 - percent / 100), 2): 折扣按百分数算, 并保留两位小数, 测试 3/3。

### 验收与修复预算独立

策略最后说的那句 "候选修复已提交验收" 不参与成功判定。experiment 在 agent.run 返回之后, 自己再调一次 workspace.check 跑测试, 以这次的结果为准。

三种模式:

- repair: 允许修复, 7 次模型调用后测试通过, 候选保留。
- one-shot: 不允许修复, 5 次调用, 只过 1/3。
- budget: 允许修复但 max_turns=4, 4 次调用就被停下, 还没来得及第二次 patch, 只过 1/3。

后两种验收失败, experiment 把 pricing.py 恢复成原文, 工作区里不留坏代码。

### Coding Agent 的工程接口

本例只有 search / read / patch / test 四个工具, 够演示闭环。真实的大项目还要检索符号、管理上下文 (项目太大装不进窗口)、执行 lint 和构建、隔离工作区、审查差异。

代码能表达复杂规则, 但生成的代码仍要过权限门和运行约束。四个工具里:

- patch 是 medium 风险、非只读 (它改文件)。
- search、read、test 只读、low 风险。

### 读代码的顺序

- `demo.py:Workspace`
- `demo.py:ProjectTool`
- `demo.py:RepairPolicy`

## 运行

```bash
python -m llm_agent.m28_coding_agent.demo
```

默认只用标准库与本仓库模块, 不访问外网。

## 运行后应该看到什么

依次运行 `one-shot, repair, budget` 三种条件, 输出结果与逐步轨迹。`main()` 对预期的成功和失败分别断言; 失败对照不是程序故障。网页可选择配置、播放或点击事件, 显示 Python 已运行记录, 不在浏览器里执行训练或工具。

## 与真实系统的差距

策略是固定修复器, 不是真实模型。patch 只接受 3 个预审过的算术表达式 (原文那句、return total、带 round 的那句), 别的一律拒绝。

路径围栏只允许本目录的 pricing.py: ../pricing.py、check.py、/etc/passwd 都被拒。但临时目录加围栏不是 OS 沙箱, 不支持运行任意不可信的生成代码。

## 常见误区

“有相同概念的教学实验”不等于“复现完整论文或具备通用产品能力”。对照只隔离本章说明的变量; 其他假设变化需重新测。

## 自测题

候选返回“已完成”, 测试失败且预算耗尽, 应怎样结束?

<details><summary>答案</summary>

回滚候选并报告验收未通过。完成由环境判断。预算和原始测试不能因为模型坚持完成而失效。

</details>
