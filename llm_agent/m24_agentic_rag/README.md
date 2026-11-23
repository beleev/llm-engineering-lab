# 检索证据 · 从相关到足够回答

[![检索证据 · 从相关到足够回答](../../docs/screenshots/agent-evidence-chain-extension-1.png)](https://beleev.github.io/llm-engineering-lab/#/agent/evidence-chain)

[项目首页](../../README.md) · [Agent 目录](../README.md) · [交互实验](https://beleev.github.io/llm-engineering-lab/#/agent/evidence-chain)

## 直觉

检索器已经召回相关文档块, 为什么系统仍然不能直接给出答案?

检索命中字面相似的文档块并不等于具备足够回答问题的依据。一个数值要成为可采信的答案, 必须具备完整可核验的证据链:
- 明确对应问题所指的目标实体 (entity)、具体版本 (version) 及数据来源。
- 若问题使用项目别名 (如 comet 是 atlas v2 的别名), 证据链还必须包含证明别名映射关系的记录。
demo 语料中的 4 个文档块均涉及 upload limit:
- 仅取相似度最高单块且不含元数据前缀的基线方法在三道题上全部无法作答。
- 将实体与版本拼入检索文本并做元数据过滤后, 直接询问 atlas v2 的问题可在一跳内答出 20 MB。
- 询问别名 comet 的问题需经两跳检索并同时引用两条来源, 方可答出 20 MB。
- 语料中无任何别名记录的 nova 则如实返回 insufficient (证据不足)。

## 核心原理

### 先给块补对象与版本

词面相似度仅负责召回候选文档, 答案是否适用取决于文档块绑定的目标实体与版本元数据。demo 语料由 4 个 Document 块组成, 每块包含 entity、version、body 字段以及可选的 alias 与 limit:
- alias-comet: 元数据为 atlas v2, 正文为「comet project maps to atlas v2」, alias=comet, 不含数值。
- atlas-v1: 元数据为 atlas v1, 正文为「upload limit is 10 MB」, limit=10。
- atlas-v2: 元数据为 atlas v2, 正文为「upload limit is 20 MB」, limit=20。
- orion-v2: 元数据为 orion v2, 正文为「upload limit is 90 MB」, limit=90。
后三块正文除数值外完全相同, 均包含 upload limit。底层检索采用 TF-IDF 词频相似度算法, 若仅索引正文则无法区分 v1 与 v2、也无法区分 atlas 与 orion:
- plain (无上下文基线): 仅对正文建索引, 只取相似度最高的 1 块 (top-1) 且不做元数据过滤。直接询问「atlas v2 upload limit」时, 因仅 alias-comet 的正文同时出现 atlas 与 v2 两个词, top-1 落在无数值的 alias-comet 上, 三道题全部返回 None。
- contextual (上下文增强): 建索引时将「atlas v2」等实体与版本前缀拼入检索文本, 召回后再按 entity 与 version 严格过滤出带有 limit 的匹配块。直接询问 atlas v2 时, 单跳检索 (查询 1 次) 即可返回 20 并引用来源 atlas-v2。

### 两跳要有两段证据

涉及别名的多跳问答必须完成两跳检索, 且最终引用列表必须同时包含别名映射记录与目标属性记录。第二道测试题询问「comet upload limit」, 其中 comet 并非直接产品名而是项目别名; 语料中唯有 alias-comet 记录了它指向 atlas v2, 且该记录本身不含上传限额数值。
- contextual 模式 (单跳检索): 首跳命中 alias-comet 后虽解析出目标为 atlas v2, 但首跳仅用于确定目标实体而不跨产品块猜测数值, 因此停止检索并返回 None。
- agentic 模式 (按证据继续检索): 第一跳解析出目标实体与版本 (entity=atlas, version=v2) 并记录来源 alias-comet; 检测到查询预算尚有剩余后, 自动构造第二跳查询「atlas v2 upload limit」, 命中 atlas-v2 并提取数值 20。最终返回答案 20, sources 为 [alias-comet, atlas-v2], 累计消耗 2 次查询。
若最终仅引用 atlas-v2 的 20 MB 而省略 alias-comet, 则缺失了「comet 为何对应 atlas v2」的桥接证据, 下游独立验证器会直接判定证据链断裂。

### 预算不足时承认缺证据

当查询预算耗尽仍未凑齐完整证据链时, 系统必须返回 None 并将 status 标记为 insufficient, 绝不用其他实体或旧版本数据拼凑答案。查询预算参数 budget 默认为 2, 限定每道题允许发起的最大检索次数:
- 直接询问 atlas v2: 单跳即可凑齐证据, 消耗 1 次预算并返回 20。
- 询问别名 comet: 完整证据链需两跳, 预算为 2 时返回 20; 若将预算收紧为 1, 首跳解析完别名后预算即告耗尽, 如实返回 None。
- 询问未知名称 nova: 语料中不存在任何指向 nova 的 alias 记录, 首跳无法确定目标实体; 即便库中 atlas 与 orion 的限额数据齐全, 系统也只能返回 insufficient。
此外, 当 budget 设为 0 时系统不发起任何检索, queries 为空; 若从语料中删去 atlas-v2 后再询问 comet, 系统虽能通过别名定位到 atlas v2, 但因缺失 v2 限额块同样返回 None, 不会退而拿 v1 的 10 MB 冒充答案。

### 指标从排名走到支持度

面向 Agent 的检索评测应从传统的候选排名指标转向证据支持度核验, 即检验所引用的来源集合能否在逻辑上闭环支撑最终答案。supported() 是独立的验证函数, 不信任回答文本中的数值声明, 仅对 sources 引用的文档块逐项核查:
- 引用合法性: 来源 id 必须存在于语料库中且不得重复引用。
- 别名桥接: 若原问题包含别名, 引用列表中必须包含一条 alias 等于该别名的映射记录, 并据此锁定目标实体与版本。
- 属性吻合: 引用列表中必须存在一块 entity 与 version 均匹配目标、且 limit 等于最终答案数值的文档。
demo 基于 comet 题目的正确输出构造了三类反例供验证器检测:
- 版本错误: 将最终答案改为 10 (对应 v1 而非 v2)。
- 证据链断裂: 将 sources 删减为仅剩 [atlas-v2], 缺失别名映射记录。
- 实体错配: 将 sources 篡改为 [alias-comet, orion-v2]。
上述三类坏样本均被判为不支持, 即便回答文本中的数值碰巧正确也无法通过验收。
实验边界: 本实验采用 TF-IDF 词面检索与确定性多跳规则, 未引入稠密神经检索或模型自主查询规划; 文档元数据由离线夹具直接给定, 真实生产系统还需额外核验文档来源可信度与生效时间窗口。

### 读代码的顺序

- `demo.py:retrieve`
- `demo.py:answer`
- `demo.py:supported`

## 运行

```bash
python -m llm_agent.m24_agentic_rag.demo
```

默认只用 Python 标准库, 不访问网络。所有断言都随 demo 运行, 也被 `pytest` 自动发现。

## 运行后应该看到什么

直接查询: 上下文版一跳得到 20 MB。Comet: 一跳证据不足, 两跳得到 20 MB 并引用 alias-comet、atlas-v2。Nova 保留未知。

这些数字来自本仓库默认配置的运行。费用、时间和工作量若在正文标成模拟, 就是模型值, 不是外部 API 实测。

## 与真实系统的差距

TF-IDF 与确定性查询规划, 不测神经检索或通用 Agent 自主规划。资料的元数据在夹具中给定; 真实系统还要验证来源与时间范围。

## 常见误区

- **Comet 问题只引用 atlas-v2 限制, 缺了什么?** 答案数字和对象关系都需要支持; 别名链不能省略。
- **什么时候两跳检索只是额外开销?** 能直接定位完整证据的简单题, 不必强制多跳。

## 自测题

1. Comet 问题只引用 atlas-v2 限制, 缺了什么?

<details><summary>答案</summary>

缺少 Comet 对应哪个版本的证据。答案数字和对象关系都需要支持; 别名链不能省略。

</details>

2. 什么时候两跳检索只是额外开销?

<details><summary>答案</summary>

问题已经明确对象和版本时。能直接定位完整证据的简单题, 不必强制多跳。

</details>

3. 预算用完还缺当前版本资料, 应怎么办?

<details><summary>答案</summary>

返回证据不足并保留缺失来源。预算与证据约束同时成立; 不能拿其他对象或旧版本凑答案。

</details>
