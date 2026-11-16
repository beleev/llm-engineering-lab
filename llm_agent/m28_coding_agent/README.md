# Coding Agent · 修改、测试与修复

[![Coding Agent · 修改、测试与修复](../../docs/screenshots/agent-coding-agent-coverage.png)](https://beleev.github.io/llm-engineering-lab/#/agent/coding-agent)

[Agent 目录](../README.md) · [交互实验](https://beleev.github.io/llm-engineering-lab/#/agent/coding-agent)

## 直觉

代码改完了, 模型也说修好了, 为什么还不能算修好?

改一行不等于修好: 首个候选 return total 只过 1/3 测试。读反馈再修一次才 3/3 (7 次模型调用); 验收由驱动器自己跑测试, 没过就回滚, 4 次调用的预算用完也回滚。

## 核心原理

### 先建立工作区证据

RepairPolicy 先 search 定位 discounted 在 pricing.py 第 1 行, 再 read 原文, 然后才 patch。

Workspace.patch 要求旧文本在文件里恰好出现 1 次, 否则抛 "旧文本不存在或不唯一, 拒绝模糊替换"; 替换后先 compile 再写盘。

修改前后用 difflib 生成 diff; check.py 保持原 sha256 摘要, 跑测试前先核对, 模型改不了测试。

### 测试真的运行

test 工具用 subprocess 启动本地 Python 子进程 (argv 固定, 无 shell, 5 秒超时), 导入修改后的 pricing.py 跑 3 个用例。

首个候选 return total 只返回原价, (100, 10) 得 100 而不是 90, 测试 1/3; 模型读到反馈后换成 round(total * (1 - percent / 100), 2), 折扣按百分数算并保留两位小数, 测试 3/3。

### 验收与修复预算独立

模型最后一句 "候选修复已提交验收" 不参与成功判定; experiment 在 agent.run 之后自己再跑一次 check。

repair 模式 7 次模型调用后通过, 保留候选; one-shot 5 次、budget 4 次 (max_turns=4) 都只过 1/3, 验收失败就把 pricing.py 恢复成原文。

### Coding Agent 的工程接口

本例只有 search / read / patch / test 四个工具; 大项目还要检索符号、管理上下文、执行 lint 和构建、隔离工作区、审查差异。

代码能表达复杂规则, 但生成的代码仍要过权限门和运行约束: patch 是 medium 风险、非只读, 其余三个只读、low 风险。

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

策略是固定修复器, patch 只接受 3 个预审算术表达式 (原文那句、return total、带 round 的那句), 不是真实模型。

临时目录加路径围栏 (../pricing.py、check.py、/etc/passwd 都被拒) 不是 OS 沙箱, 不支持运行任意不可信生成代码。

## 常见误区

“有相同概念的教学实验”不等于“复现完整论文或具备通用产品能力”。对照只隔离本章说明的变量; 其他假设变化需重新测。

## 自测题

候选返回“已完成”, 测试失败且预算耗尽, 应怎样结束?

<details><summary>答案</summary>

回滚候选并报告验收未通过。完成由环境判断。预算和原始测试不能因为模型坚持完成而失效。

</details>
