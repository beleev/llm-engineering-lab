# 教程重叠、章节合并与知识缺口复核

审查日期：2026-11-08。基线：`fe70ee9`。本文件保留审查时的结论与改版建议。后续修复已实施，状态、实验结果与验证见 [归并验收记录](tutorial-consolidation-completion.md)。

结论：主干知识已比较齐全。当前更值得做的是收紧章节职责，而不是继续增加一串新章节。建议直接合并 4 对章节；另有 3 处适合调整内容归属。新确认的补充重点是多租户服务公平性、持久任务的版本兼容，以及缓存隔离的可验证实验。

## 范围与判定方法

- 当前目录：6 个阶段、26 个二级分组、107 个三级章节，加 6 个阶段总览，共 113 个学习入口。
- 扫描 102 个数据驱动页面的摘要、问题、知识点正文，以及全部自测题和三个 completion 数据集的挂载关系；补查 Attention、Position、Blocks、MoE、Diffusion 五个手写页的主题与组件归属。
- 使用中文三元组相似度筛选候选，再人工比较正文、实验目的和代码。相似度只用于排序，不代表“重复率”。同一概念的原理、训练和部署分别讲解，不自动判为重复。
- 长文本完全重复的口径：去空白后超过 45 个字符的整个字段。该扫描不覆盖任意子句、所有 Vue 模板和所有模块 README，不能据此声称全仓库没有重复。
- 检索 Stanford CS336、vLLM、LangGraph、Anthropic 等一手资料，并回查本地实现。区分“没有形成教学单元”“已经提及但没有实验”“有意限制的教学边界”。

## 仍需处理的重叠与表述

### R1：两处页内摘要与正文完全相同，优先级低

1. `finetune-knowledge-update` 的 `tldr` 与首个知识点“编辑和 SFT 的目标不同”。见 [finetune.js](../web/src/data/topics/finetune.js)。
2. `agent-verifiable-reasoning` 的 `tldr` 与首个知识点“形式证明有明确边界”。见 [agent.js](../web/src/data/topics/agent.js)。

这两处不是两章重复，而是同页读两遍同一句。摘要保留一句结论，首个知识点改为问题定义、适用条件和反例即可。本次自动扫描未检出跨页完全相同的长字段；自测题干未检出完全相同项；三个 completion 数据集均没有同一实验条目挂到多个路由的情况。

### R2：五组概念说明可以集中主讲

| 当前交叉 | 具体重复 | 建议主讲与引用方式 |
| --- | --- | --- |
| `train-precision-stability` / `train-low-precision` | FP8 block scale、outlier 实验的 0.126 / 0.026 数值重复出现 | 合成精度章节，先讲范围与累积，再讲缩放粒度和 FP4 |
| `train-model-parallel` / `train-pipeline-schedules` | GPipe / 1F1B 气泡公式和在途激活差别重复解释 | 前章只说明 PP 解决什么，公式、时间线及假设集中到调度章 |
| `agent-prompt-caching` / `agent-context-cost` | 动态前缀破坏命中、缓存计价函数复用 | 合成缓存与成本章节，增加摘要开销和质量验收，而非重讲缓存 |
| `agent-state-subagents` / `agent-orchestrator` | 子级独立上下文、只回摘要、权限不继承 | 子级机制放协作主章；会话章节保留落盘、恢复、压缩边界 |
| `agent-runtime-reliability` / `agent-async-runtime` | cancel-request / ack、generation、同键同参提交 | 前章讲重试与不确定结果；异步章主讲状态机和真实取消时序 |

这些属于讲解职责交叉，不意味着底层实验全部重复。尤其 m22 的失败分类与 m29 的真实 asyncio / SQLite 实验应该都保留。

### R3：实验移走后，有一处阅读指引没跟着更新

`web/src/data/topics/models.js` 中 `models-dsa.links` 仍写“models-mtp · 掩码实验台”“那里的 top-k 模式”。但 `AttnMaskLab` 目前在 `web/src/views/Attention.vue`，`models-mtp` 已不挂这个组件。建议改为 Attention 的掩码实验台，并核对实际入口。

这也说明“源码路径有效”不等于“教学指引指向的内容仍在那里”。后续合并要同时检查 `links`、`related`、演进图文字、README 和章节提示。

### R4：局部实验结果需要更明确的适用范围

- `train-precision-stability` 的“更新永远发生在 fp32 master 上”，应限定为本章实现/所演示的混合精度配方，不能写成所有训练实现的必然条件。
- `finetune-lora` 用一次 copy → sort、300 步实验反驳普遍的“LoRA 收敛更快”，结论范围过大。应写成本实验没有观察到更快收敛，不能由此推出所有任务都更慢。

建议保留实测数字，把数据集、步数、种子与评价指标紧挨结论展示。这里不是要求补新算法。

## 章节合并建议

### M1–M4：建议直接合并的 4 对

| ID | 现有章节 | 合并后的主题 | 保留的递进关系 |
| --- | --- | --- | --- |
| M1 | `train-precision-stability` + `train-low-precision` | 低精度训练：数值稳定、缩放与 FP4 | 浮点范围 → loss scaling / master → FP8 → block scale / FP4；保留两个实验台 |
| M2 | `finetune-lora` + `finetune-dora` | 参数高效微调：LoRA 与 DoRA | 低秩增量与初始化 → 幅值/方向解耦 → 训练及合并代价；保留 PEFT 对照 |
| M3 | `agent-prompt-caching` + `agent-context-cost` | 上下文缓存与成本控制 | 前缀复用 → TTL / 计价 → 摘要净成本 → 质量验收；保留两种实验 |
| M4 | `agent-orchestrator` + `agent-collaboration` | 多智能体编排与协作 | 扇出与预算 → 拓扑 → 共享状态冲突 → 证据独立性与收尾 |

这 4 对合并可把三级章节 **107 → 103**，学习入口 **113 → 109**；26 个二级分组不必改。减少的是导航入口，不是实验和知识点。合并页应按递进关系分段，不直接把两页全文拼接。特别是 LoRA 属于冲刺路线、DoRA 属于扩展内容：合并后需标出进阶阅读边界，避免把所有 DoRA 内容变成冲刺读者的必读项。

### O1–O3：可以进一步整理，需先调整归属

1. **记忆生命周期 + 记忆隐私**：可以合成“长期记忆治理”，围绕来源、范围、更新、授权、删除组织；“记忆的四种类型”移回 `agent-context-memory`。两页目前共有 10 个知识点，简单拼接会使页面更长。满足这个前提后，可再减少 1 章。
2. **QK-Norm 与 YaRN**：一个控制注意力数值，一个调整位置频率，放在同章并不自然。QK-Norm 可进入 `attention`，YaRN 进入 `position`，并保留各自实验台；这样可再减少 1 章。需要控制两个手写页的长度和锚点导航。这是重归属，不是两项技术重复。
3. **SWA · MTP · 混合线性注意力**：这一章同时回答三个问题。SWA 的掩码入口现已在 Attention；模型章应保留跨层感受野及实际架构影响。混合线性结构可以和 Mamba 作比较，但不能把 DeltaNet 当成 Mamba。MTP 保留训练目标及草稿生成主线，投机验收仍由推理章节负责。此项暂不承诺减少章节数。

M1–M4 加上前两项条件合并，三级章节最多先收敛到 **101**，包含总览的入口为 **107**。这些是方案数，不是已经完成的变更。

### 不建议硬合并

- **完成验收 / 离线评测**：前者在一次执行中决定结束或修复，后者统计跨任务可靠性。可以共用验收接口，不应混成一个指标问题。
- **DPO / SimPO-ORPO**：重复的长度偏置背景应缩成引用；DPO 页还包括 KTO 和对照实验，整体合并会变成长页。
- **QLoRA / LoRA**：前者还有量化基座、反量化及显存路径；LoRA 与 DoRA 合并后，QLoRA 仍适合独立进阶。
- **模型并行 / 流水线调度**：TP、PP、多维预算与调度时间线的学习目标不同，去重公式即可。
- **模型结构 / 推理内存**：例如 MLA、DSA、MoE 在不同阶段出现，分别回答表示、计算和服务问题，属于正常递进。
- **应用记忆删除 / 参数知识编辑与遗忘**：作用对象不同。上轮拆开的知识更新章节应保留。

## 结合搜索确认的补充重点

### K1：多租户服务的公平性与配额——尚未形成教学单元

**已有内容**：continuous batching、chunked prefill、队列容量、请求超时、取消、goodput、缓存感知路由。`llm_infer/m28_service_capacity/demo.py` 的请求没有租户与优先级字段，主要用统一队列和统一 deadline；因此不能说明某租户持续占用资源时其他租户的服务质量。

**建议补到**：`infer-scheduler`，在 `infer-multi-replica-routing` 加关联说明，不新增三级章。

**最小实验**：固定两个租户的长短请求和突发流量；比较 FIFO、优先级、带配额/老化机制的策略；按租户报告等待时间、超时率、goodput 和最长等待，加入高优先级持续到达的饥饿反例。总体吞吐变好不算通过公平性验收。

vLLM 的调度配置明确区分 FCFS 与 priority。这支持把“按什么规则分资源”独立讲清；配额和防饥饿实验是本审查的教学建议，不声称 vLLM 已提供该整套策略。[官方调度配置](https://docs.vllm.ai/en/latest/api/vllm/config/scheduler/)

### K2：持久任务跨版本恢复——恢复机制已有，兼容性教学缺失

**已有内容**：JSONL 会话恢复、悬空 tool-use 补齐、权限重新建立、SQLite 幂等账本、取消确认。`core/persistence.py` 是本库消息格式；m29 重新打开的是账本，不是完整任务执行状态。

**尚缺的问题**：任务在等待审批时升级代码，旧检查点缺字段、节点改名、工具参数改变后怎么办？“文件还能读”不代表“任务能按原业务语义继续”。版本管理与重试幂等是不同问题。

**建议补到**：`agent-state-subagents` 和 `agent-async-runtime`，前者主讲状态 schema，后者使用跨版本任务恢复实验。第一版无需实现分布式工作流框架。

**最小验收**：v1 暂停、v2 恢复；新增可选字段可以迁移；删除节点/必填字段不兼容时在副作用前明确拒绝；保存 flow version，重新校验权限；外部提交成功但本地未记结果时先查收据，不把恢复自动变成重执行。最后一项是既有幂等知识的集成验收，不算新的独立知识空白。

LangGraph 官方将旧状态与新代码的兼容、业务行为版本、节点/字段迁移分别讨论。本教程可以采用自己的小状态机，不需要增加框架使用教程。[官方兼容性文档](https://docs.langchain.com/oss/python/langgraph/backward-compatibility)

### K3：前缀缓存的身份与隔离——已提及，但缺反例和验收

**已有内容**：`llm_infer/m04_prefix_cache/README.md` 已提到 LoRA id、多模态 hash、cache salt。不能说教程完全没有多租户缓存知识。

**不足**：`prefix_cache.py:_block_hash` 只接父 hash 与 token；没有把身份、模型/adapter 版本、输入模态纳入同一可验证的缓存身份实验。现有 Agent 文档 ACL 演示也不能替代推理 KV 缓存的隔离问题。

**建议补到**：`infer-prefix-radix`，并从 Multi-LoRA 和路由章节链接到主讲页。

**最小实验**：同 token、同模型、同授权共享域允许命中；不同隔离域不共享；模型或 adapter 版本改变必须失效；客户端不能自行指定其他租户的有效隔离域；用合成时延展示“命中速度可以泄露前缀存在性”的边界。不要把教学时延当真实攻击成功率。

vLLM 官方指出 cache salt 可减少跨租户缓存侧信道风险，但同一服务进程本身不提供完整租户隔离。因此不能把“加了 salt”讲成完整安全边界；网关身份约束和必要的实例隔离也要说明。[官方安全文档](https://docs.vllm.ai/en/latest/usage/security/)

### 已知边界，不重复列成新缺口

- **OS 沙箱**：m03、m12、m50 明确说明权限门、路径围栏和 `python -I` 不是 OS 隔离，m12 已说明文件系统与网络出口限制。可以做额外的真实隔离实验，但不应再次列为“尚未讲安全”。Anthropic 的资料可用于对照文件系统、网络和子进程边界。[沙箱设计](https://www.anthropic.com/research/claude-code-sandboxing)
- **GPU kernel 实操**：已有 FlashAttention / Flash-Decoding、CUDA Graph 概念及真实 CPU profiler；没有完成 GPU 实测不能包装成完成了 GPU 性能优化课程。Triton kernel 编写属于进一步的系统专项，本轮不据此新增大章。
- **PDF/OCR 生产适配、完整 OAuth/协议一致性、真实 GPU 异步 RL、线上流量平台**：上轮报告已披露边界。对应教学补全的验收通过，不意味着这些生产功能已经实现，也不意味着本轮又发现四个新知识空白。

## 主干交叉检查结果

以 [Stanford CS336 2026 课程表](https://cs336.stanford.edu/) 的主题作参照，tokenization、架构、资源核算、并行、scaling、数据处理、评估、SFT/RLHF/RLVR 和多模态在本仓库都有入口。课程表用于核对主题，不能证明本仓库达到相同实验深度。

| 范围 | 当前入口举例 | 本轮判断 |
| --- | --- | --- |
| 基础与模型 | basic-data / forward / backward；五个手写页；models-generation | 未发现需要新增顶层模块的缺口，重点整理长序列章节归属 |
| 训练 | data-packing、scaling、parallel、precision、eval | 主干存在；精度重复和 PP 背景适合收紧 |
| 微调 | SFT、LoRA/QLoRA/DoRA、DPO/PPO/GRPO、RLVR、artifact delivery | 不重开上轮交付缺口；LoRA/DoRA 可合并，结果外推需收敛 |
| 推理 | KV、量化、解码、算子、调度、服务容量 | K1 尚缺单元；K3 缺反例及验证 |
| Agent | 工具、MCP、上下文、RAG、运行时、评测、训练和应用 | K2 尚缺迁移教学；多处内容可集中主讲 |

## 本次实际运行的检查

- `python3 scripts/check-engineering-coverage.py`：G1–G6、6 组 / 31 条件的主入口、文档、自测与关联引用通过。
- `python3 scripts/check-research-coverage.py`：45 个原图主题 / 90 条原差距有去向，25 组实验入口与文档一致。
- `python3 scripts/check-agent-book-coverage.py`：固定参考版本的 10 章、287 个标题本地入口完整。
- `npm run check:sources`：319 / 319 个源码引用有效。
- `npm run check:quiz`：393 道题通过当前检查。

这些检查验证静态映射和资料一致性，不等价于语义完整、全部实验重跑或视觉验收。本轮没有修改教程代码，也没有重做浏览器排版回归。伴随的 JSON 保存重复扫描、候选相似度与审查事项，便于后续改版对账。

## 实施时应满足的条件

先处理 R1–R4，再做 M1–M4，最后把 K1–K3 补进既有章节。O1–O3 单独确认阅读长度和归属后处理。

合并时保留旧 route 的跳转及锚点，迁移本地 visited / last / quiz 状态，并保持浏览器收藏的旧链接可达；同步目录编号、前后章、学习档位、搜索、源码映射、覆盖清单及 README。实验每项只设一个主讲入口。验收需覆盖旧链接跳转、亮暗主题、窄屏、三级标题编号、键盘导航及实验切换，不能只看构建成功。
