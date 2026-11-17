# Skill 进化 · 提案、发布与整理

[![Skill 进化 · 提案、发布与整理](../../docs/screenshots/agent-skill-evolution-coverage.png)](https://beleev.github.io/llm-engineering-lab/#/agent/skill-evolution)

[Agent 目录](../README.md) · [交互实验](https://beleev.github.io/llm-engineering-lab/#/agent/skill-evolution)

## 直觉

项目 A 的 review 说该用 Rust, 我直接把默认语言改成 Rust, 项目 B 怎么就坏了?

- demo 里的 Skill 是一条 "按项目选编程语言" 的规则: 默认 Python, 可以给单个项目设例外。项目 A 的 review 说该用 Rust。
- 全局改法 (把 default 改成 Rust) 过三道门的结果是 [True, False, False]: A 修好了, 但 B 和 untrusted 项目也被改成 Rust, 被拒, 不发布。
- 局部改法 (只写 projects[A]=Rust) 三门全过, 发布成版本 1, 此时 A 答 Rust, B 还是 Python。rollback(0) 一拨, A 又答 Python。

## 核心原理

### 更新载体不只有记忆

"学到了东西" 可以存在五种不同的地方, 存在哪里决定怎么检查它。

- 知识: 改的是事实, 比如 "项目 A 用 Rust"。
- Prompt: 改的是模型做决定的顺序和措辞。
- Skill: 存的是可以反复用的步骤和规则。
- 程序: 改的是执行机制, 也就是代码。
- 参数: 改的是模型本身的概率。

本章动的是 Skill: 一条 "按项目选语言" 的规则, 存在 rules.json 里, 内容是 {"default": "Python", "projects": {...}}。act(policy, project) 先查 projects 里有没有这个项目的例外, 没有就用 default。

每类载体各用各的检查。记住一个新事实只更新了知识这一类; 改了 rules.json, 模型参数也没有变。

### 提案来自已验证反馈

改规则的证据必须是验收过的, 改的范围不能比证据大。

证据只有一条: LEARNING 里项目 A 的 review-1 说该用 Rust, 而当前策略答 Python。propose 只接受来源以 review- 开头的证据, 别的来源直接抛 ValueError。

提案有两种写法, 由 globalize 开关决定:

- 全局候选 {default: Rust}: 所有项目的默认语言都变成 Rust。
- 局部候选 {projects: {A: Rust}}: 只给 A 加一条例外, default 不动。

发布前要过三道门 GATES, 每道门是 "某个项目应该答什么" 的一条检查:

- heldout-a: 查 A 是否答 Rust (迁移: 失败修好了没)。
- retention-b: 查 B 是否还答 Python (保持: 没坏别的)。
- safety-c: 查 untrusted 项目是否还答 Python (安全: 不信任的项目没被顺带改掉)。

两种候选的结果:

- 全局候选: [True, False, False], A 过了, B 和 untrusted 都挂, 不发布, active (当前生效的版本号) 还是 0。
- 局部候选: 三门全过, 发布成版本 1。

三道门的内容不进 propose 的输入, 提案器看不到考题。

### Skill 的三层加载

一份 Skill 在磁盘上是三个文件, 分别给三个阶段用, 由 SkillStore.write 一起写出。

- index.md: 一行 "project-language: 按项目选择语言 → SKILL.md", 供发现。模型先扫这一行, 决定要不要打开这份 Skill。
- SKILL.md: 两句话, 供决策 (先查项目例外, 否则用默认语言; 详细规则见 rules.json)。
- rules.json: 完整策略, 执行时读它。

三个文件的内容跟 active 版本一致, 每次发布或回滚都重写。

Skill 正文引用脚本或资源时, 还要查路径、来源和执行权限: 目录里的描述命中了, 不等于授权执行那个脚本。

### 睡眠学习是离线整合

攒下来的学习记录要定期整理, 整理不在线上跑, 像睡眠一样离线做。

curate 拿 3 条记录在 now=5 时整理, 每条记录有作用域 scope、值 value、来源 source 和过期时间 expires:

- a: (A, Rust), 来源 review-1, expires=10。
- b: (A, Rust), 来源 review-2, expires=10。
- c: (A, Go), 来源 old, expires=1。

c 的 expires=1 已经不晚于 now=5, 归档。a 和 b 的作用域和值都相同, 合并成 1 条, 两条来源都保留。结果: 合并后 1 条, 归档 1 条。

只合并相同作用域加相同值的记录; 像 (A, Rust) 和 (A, Go) 这种冲突, 不能按谁最近被访问得多就偷偷裁决。

整理怎么跑:

- 触发可以按时间、轨迹数量、容量或错误频率。
- 在线只收证据不改规则。
- 整理完重建索引, 回归通过再激活。

### 版本控制不能代替评估

能回滚只说明出了问题可以恢复, 不说明新版本是对的。

rollback 条件里分两步:

1. 局部候选先三门全过、发布成版本 1 (A 答 Rust)。

2. rollback(0) 把 active 拨回 0, 三个文件按版本 0 重写, A 又答 Python。

版本 1 可靠, 是三道门证明的。不同载体还各有各的检查: 程序更新要测试和差异审查, 参数更新要查保持与安全。至于比较不同学习方法本身哪个好 (元优化), 见 m39。

### 怎样写一份可用的 Skill

先写四样: 什么时候触发、服务谁、步骤是什么、输出怎么验收。写法上:

- 核心规则带正反例。
- 禁止项注明合法例外。
- 资料和脚本放在引用层, 不塞进正文。

规则从真实任务的失败和人工修改反推出来: 本章的 projects[A]=Rust 就来自一条 review。不要把一次对话全文直接当成 Skill。

### 内容加载不等于固定角色

index.md 先被发现, SKILL.md 按需读取; 这只规定了加载顺序, 没规定 Skill 正文一定以 system 身份进入模型。

对话 API 里每条消息都有角色 (system、user、developer, 或工具结果), 不同客户端可能用不同角色承载 Skill 正文。角色决定模型多信任这段话, 所以要按实际 API 和信任边界检查; 不能因为 Skill 是 "加载进来的" 就推断它自动拥有更高权限。

### 读代码的顺序

- `demo.py:propose`
- `demo.py:SkillStore`
- `demo.py:curate`

## 运行

```bash
python -m llm_agent.m33_skill_evolution.demo
```

默认只用标准库与本仓库模块, 不访问外网。

## 运行后应该看到什么

依次运行 `global, scoped, rollback` 三种条件, 输出结果与逐步轨迹。`main()` 对预期的成功和失败分别断言; 失败对照不是程序故障。网页可选择配置、播放或点击事件, 显示 Python 已运行记录, 不在浏览器里执行训练或工具。

## 与真实系统的差距

提案器是两条固定规则 (改 default, 或改 projects[项目]), 发布就是往一个本地临时目录写三个文件; 不是自主生成任意自然语言 Skill。

三个文件各自先写 .tmp 再 replace, 单个文件是原子的 (要么全是旧内容要么全是新内容, 不会写一半); 但三个文件不是一个事务, 中途崩溃可能有的换了有的没换。并发发布还要额外加版本锁。

## 常见误区

“有相同概念的教学实验”不等于“复现完整论文或具备通用产品能力”。对照只隔离本章说明的变量; 其他假设变化需重新测。

## 自测题

候选修好了 A 却破坏 B, 发布前应该怎么办?

<details><summary>答案</summary>

拒绝全局候选, 缩小规则作用域。迁移收益不能掩盖负迁移。拒绝不合格提案, 再生成带作用域的规则。

</details>
