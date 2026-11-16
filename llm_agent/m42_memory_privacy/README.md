# 个人记忆 · 同意、过期与删除

[![个人记忆 · 同意、过期与删除](../../docs/screenshots/agent-memory-privacy-coverage.png)](https://beleev.github.io/llm-engineering-lab/#/agent/memory-privacy)

[Agent 目录](../README.md) · [交互实验](https://beleev.github.io/llm-engineering-lab/#/agent/memory-privacy)

## 直觉

记忆删了, 为什么检索结果还可能找到它? 记得用户的偏好, 就能主动提醒他吗?

授权在召回之前查, 不是召回后过滤: flat 模式一次就漏 1 条别家租户的、1 条过期的。删除要同时清原文、索引和缓存, forget 之后召回为空。

## 核心原理

### 记忆层次与表示

工作记忆放当下状态, 情节记忆保留经历, 语义记忆保留事实, 程序记忆保留做法。这四层是设计上的分类, 不是对模型认知能力的证明。

表示可以是列表、文档、图或可执行规则。

m42 的 MemoryService 把一条偏好事实存成 Memory 记录 (id、tenant、user、topic、content、expires、consent、kind), 跨会话按作用域找回。

### 跨会话与主动服务分开测

记住一条事实、在新会话找回、在合适时机主动提醒, 是三种能力, demo 分三条 trace 测。

首次会话按作用域召回 ['Python']; 新会话再召回一次仍是 ['Python']; notify 在 opted_in=False 时直接返回空, 三种模式 proactive 都是 0。

记得不等于能主动用, 主动使用个人信息要单独同意。

### 层级与多模态不绕过权限

索引键是 (tenant, user, topic) 三元组, 召回只查自己的键。flat 模式绕过索引直接读 raw, 3 条全出来, 其中 Rust 属于 org-b, 越界 1 条。

Memory 记录带 kind 字段: 摘要层、完整正文、图片字幕、音频转录和原文件链接都继承来源的权限, 不能只在最后过滤原文。demo 只放了 text 一种。

### 删除与压缩的区别

过期决定能不能召回: Go 的 expires=1, now=5 时 scoped 召回不含它, flat 模式含它, 过期 1 条。

压缩只减体积; 删除要清原文和派生引用。forget("org-a", "Lin", "a") 同时从 raw、index、cache 三处删掉 id, 之后召回为 []。删别人的记忆抛 PermissionError。

审计只记 operation 和 12 位散列 (scope_hash / id_hash), 不留个人正文。

### 后台整理需要冲突策略

访问频率、时间、重要性只能辅助筛选, 常被访问的错误仍是错误。

合并前保留来源与作用域: remember 拒绝覆盖已有 id, 证据不可改写。碰到冲突重新求证; 索引坏了要能从 raw 重建。

### 读代码的顺序

- `demo.py:MemoryService`

## 运行

```bash
python -m llm_agent.m42_memory_privacy.demo
```

默认只用标准库与本仓库模块, 不访问外网。

## 运行后应该看到什么

依次运行 `flat, scoped, forget` 三种条件, 输出结果与逐步轨迹。`main()` 对预期的成功和失败分别断言; 失败对照不是程序故障。网页可选择配置、播放或点击事件, 显示 Python 已运行记录, 不在浏览器里执行训练或工具。

## 与真实系统的差距

没模拟的: 身份认证、加密、备份里的删除。个人信息是内存里的 3 条合成记录, 没有生产存储。

3 条记录、1 次越界、1 次过期, 说明的是数据生命周期的检查点在哪, 不能外推成满足某项法律或平台的合规要求。

## 常见误区

“有相同概念的教学实验”不等于“复现完整论文或具备通用产品能力”。对照只隔离本章说明的变量; 其他假设变化需重新测。

## 自测题

只删原文而保留包含正文的摘要缓存, 算完整遗忘吗?

<details><summary>答案</summary>

不算, 派生数据仍可能泄露内容。遗忘需要沿数据派生关系清理。保留审计时应避免重新保存被删除的正文。

</details>
