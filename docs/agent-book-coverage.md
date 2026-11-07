# ai-agent-book → llm-engineering-lab 知识覆盖表

> 2026-11-08 语义复查及补充：287 项标题映射只验证索引。此前确认的九项缺口已增加正文、实验/入口和失败对照，逐项证据及边界见[补齐验收记录](knowledge-completion.md)。不据此宣称全领域知识穷尽或生产能力齐备。

对照书版本：`dbc046eb896ac4e39aa19c7774c8bf49583b89a6`。本轮在此前 m20–m25 与评估扩展之上继续补齐，新增 m26–m44。

标题索引来自 ai-agent-book, 附 [上游 Apache-2.0 许可](references/ai-agent-book-LICENSE)。讲解与实验按本库方式独立编写, 没有复制上游代码实现。

## 验收口径

“补齐”指这一本书的知识范围在本库都有讲解入口、与已有模块的联系，以及适合本库定位的机制实验或明确的研究边界。不是声称实现所有书中产品、复现所有论文效果，或掌握未来所有 Agent 知识。实验采用固定输入与失败对照，核心路径不因扩展而加长。

逐项对照正文的二至四级知识标题，跳过代码围栏里的伪标题、章末复习和思考题。品牌案例保留原理映射，不能按出现一个产品名就宣布复现产品。标题覆盖只是索引，内容还要人工检查机制、失败模式与边界。

- **概念与机制对照**：讲解该知识及适用条件，对应小型实验；不会把一个小实验当完整算法。
- **机制与实际执行**：相关代码实际执行修改、测试、异步任务或模型更新；仍注明规模限制。
- **案例原理映射**：映射书中案例说明的机制，原案例数据和产品实现仍以书中来源为准。

## 本轮知识 diff

| 原缺口 | 本轮补充 | 已验证与边界 |
|---|---|---|
| Prompt / Skill 与工具发现 | m26、m33、m41 | 按需 Schema、版本核验、参数保真、分层资料加载；规则模型不能代表真实提示效果 |
| 高级知识组织 | m27、m42 | 层次摘要、关系路径、版本索引、上下文块和分组统计；说明 RAPTOR / GraphRAG 尚有哪些论文级组件 |
| Coding Agent | m28、m38 | 真正修改临时项目、执行测试、修复和回滚；仅执行预审表达式，不冒充通用沙箱 |
| 异步与长任务 | m29 | asyncio 取消确认、SQLite 幂等和账本重开、流式事件；不承诺跨服务 exactly-once |
| 多智能体 | m30、m40 | 跨文件语义约束、独立审查、相关错误、协作拓扑与激励；不是大规模真实多模型结论 |
| 评估工程 | m31 | 用户隐藏状态、分层任务、消融与裁判校准；真实线上 A/B 与人工研究未开展 |
| Agent 专项训练 | m32、m37 | 小 LLaMA 的 SFT、在线策略梯度与中期训练真实更新；留出实体仍属于相同符号协议 |
| 持续进化 | m33、m39 | 局部提案、回归、版本发布、回滚、后台整理与元搜索回放；不声称通用自主进化 |
| 语音和具身 | m34、m36 | 语音事件代际、一维控制与域变化；ASR/TTS 事件是夹具，机器人是模拟 |
| 视觉操作 | m35 | 默认像素实验加可选真实 Chromium 截图、定位、点击和核对；颜色定位不等于通用视觉模型 |
| 系统与隐私遗漏 | m41、m42 | workflow / Agent 选择、状态栏、KV 因果依赖、跨会话与主动服务、删除派生数据 |

## 复查过程

1. 首轮按十类缺口建立专题，接入原有 Agent、训练与网页接口。
2. 逐章复查后补入中期训练、隐私、系统选择、程序化行动、元优化和智能体社会，避免只补热门应用。
3. 逐节复查又补入参数逐字保真、结构化数据的分组混杂、上下文学习机制边界、同步模型接异步事件、奖励诊断、语音自然度、隐藏角色信息隔离与数据集维护。
4. 第二轮语义审查发现近似向量索引和在轨自蒸馏不能只用相邻章节替代，继续新增 m43 / m44，分别运行召回—计算量对照和特权信息隔离训练。
5. 运行失败对照、边界测试与浏览器配置核对；检查“模拟”“真实执行”“研究讨论”是否在正文和读数中一致。若书的源文件哈希变化，重新提取知识 diff，不能沿用本次闭合结论。

## 可复核命令

```bash
# 默认机制保持标准库; 完整运行加三个 PyTorch 训练扩展
python -m llm_agent.run_all
python -m llm_agent.run_all --with-training
python -m pytest tests/test_agent_coverage.py
python web/scripts/export-agent-coverage.py --check
python scripts/check-agent-book-coverage.py
# 如本机有书仓库, 检查固定版本的章节内容和标题全集
python scripts/check-agent-book-coverage.py --book /data00/ai-agent-book
# 实际 Chromium 像素操作: 可选 playwright / pillow
python -m llm_agent.m35_visual_grounding.browser
# 先在 web 启动 npm run dev
python web/scripts/check-agent-coverage-ui.py
```

网页回放来自 Python 执行。选择配置切换整条已运行记录，时间轴只浏览证据，不在浏览器训练模型。实测训练数值和模拟成本分别标注。来源摘要会检查脚本变化，但摘要匹配本身不证明实验正确，仍须重跑导出核对。

## 逐节映射

上一轮共映射 **287 个知识标题**。全部有本地文件入口；入口是否充分解释对应知识仍须语义复查。不同覆盖层次见下表，不能把这个计数当内容完整率或通用能力百分比。

### 第 1 章

| 书中知识点 | 本库讲解与代码入口 | 覆盖层次 |
|---|---|---|
| [现代 Agent = LLM + 上下文 + 工具](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter1.md#L11) | [m01_agent_loop](../llm_agent/m01_agent_loop/README.md) · [m02_tool_use](../llm_agent/m02_tool_use/README.md) · [m04_context_memory](../llm_agent/m04_context_memory/README.md) | 概念与机制对照 |
| [观察空间与动作空间：模型与世界的接口](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter1.md#L37) | [m01_agent_loop](../llm_agent/m01_agent_loop/README.md) · [m02_tool_use](../llm_agent/m02_tool_use/README.md) · [m04_context_memory](../llm_agent/m04_context_memory/README.md) | 概念与机制对照 |
| [工具：Agent 的手脚](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter1.md#L61) | [m01_agent_loop](../llm_agent/m01_agent_loop/README.md) · [m02_tool_use](../llm_agent/m02_tool_use/README.md) · [m04_context_memory](../llm_agent/m04_context_memory/README.md) | 概念与机制对照 |
| [LLM：Agent 的大脑](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter1.md#L109) | [m01_agent_loop](../llm_agent/m01_agent_loop/README.md) · [m02_tool_use](../llm_agent/m02_tool_use/README.md) · [m04_context_memory](../llm_agent/m04_context_memory/README.md) | 概念与机制对照 |
| [模型即 Agent：当模型本身成为产品](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter1.md#L115) | [m01_agent_loop](../llm_agent/m01_agent_loop/README.md) · [m02_tool_use](../llm_agent/m02_tool_use/README.md) · [m04_context_memory](../llm_agent/m04_context_memory/README.md) | 概念与机制对照 |
| [Agent 的学习机制：从上下文适应到持久更新](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter1.md#L125) | [m01_agent_loop](../llm_agent/m01_agent_loop/README.md) · [m02_tool_use](../llm_agent/m02_tool_use/README.md) · [m04_context_memory](../llm_agent/m04_context_memory/README.md) | 概念与机制对照 |
| [上下文：Agent 的眼睛](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter1.md#L137) | [m01_agent_loop](../llm_agent/m01_agent_loop/README.md) · [m02_tool_use](../llm_agent/m02_tool_use/README.md) · [m04_context_memory](../llm_agent/m04_context_memory/README.md) | 概念与机制对照 |
| [ReAct 循环](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter1.md#L161) | [m01_agent_loop](../llm_agent/m01_agent_loop/README.md) · [m02_tool_use](../llm_agent/m02_tool_use/README.md) · [m04_context_memory](../llm_agent/m04_context_memory/README.md) | 概念与机制对照 |
| [Harness 工程：模型之外的竞争力](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter1.md#L263) | [m41_system_design](../llm_agent/m41_system_design/README.md) | 概念与机制对照 |
| [从提示工程到 Loop 工程：工程范式的演进](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter1.md#L321) | [m41_system_design](../llm_agent/m41_system_design/README.md) | 概念与机制对照 |
| [构建有效 Agent 的核心原则](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter1.md#L341) | [m41_system_design](../llm_agent/m41_system_design/README.md) | 概念与机制对照 |
| [如何选择模型](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter1.md#L355) | [m41_system_design](../llm_agent/m41_system_design/README.md) | 概念与机制对照 |
| [编排模式：工作流与自主](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter1.md#L372) | [m41_system_design](../llm_agent/m41_system_design/README.md) | 概念与机制对照 |
| [工作流模式：确定性的编排](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter1.md#L384) | [m41_system_design](../llm_agent/m41_system_design/README.md) | 概念与机制对照 |
| [自主 Agent：动态自主决策](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter1.md#L418) | [m41_system_design](../llm_agent/m41_system_design/README.md) | 概念与机制对照 |
| [两种模式的选择与混合](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter1.md#L434) | [m41_system_design](../llm_agent/m41_system_design/README.md) | 概念与机制对照 |
| [主流 Agent 框架简要对比](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter1.md#L442) | [m41_system_design](../llm_agent/m41_system_design/README.md) | 概念与机制对照 |
| [护栏与安全性](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter1.md#L466) | [m12_guardrails](../llm_agent/m12_guardrails/README.md) · [m03_permissions](../llm_agent/m03_permissions/README.md) · [m41_system_design](../llm_agent/m41_system_design/README.md) | 概念与机制对照 |
| [护栏类型](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter1.md#L474) | [m12_guardrails](../llm_agent/m12_guardrails/README.md) · [m03_permissions](../llm_agent/m03_permissions/README.md) · [m41_system_design](../llm_agent/m41_system_design/README.md) | 概念与机制对照 |
| [人工干预](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter1.md#L490) | [m12_guardrails](../llm_agent/m12_guardrails/README.md) · [m03_permissions](../llm_agent/m03_permissions/README.md) · [m41_system_design](../llm_agent/m41_system_design/README.md) | 概念与机制对照 |
| [Harness 五要素与「构建」部分的对应](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter1.md#L506) | [m12_guardrails](../llm_agent/m12_guardrails/README.md) · [m03_permissions](../llm_agent/m03_permissions/README.md) · [m41_system_design](../llm_agent/m41_system_design/README.md) | 概念与机制对照 |
| [贯穿全书的设计模式](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter1.md#L525) | [m41_system_design](../llm_agent/m41_system_design/README.md) · [m01_agent_loop](../llm_agent/m01_agent_loop/README.md) | 概念与机制对照 |

### 第 2 章

| 书中知识点 | 本库讲解与代码入口 | 覆盖层次 |
|---|---|---|
| [上下文：决定 Agent 能力上限的关键](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter2.md#L7) | [m04_context_memory](../llm_agent/m04_context_memory/README.md) · [m15_claude_api](../llm_agent/m15_claude_api/README.md) | 概念与机制对照 |
| [Agent 如何调用大模型：理解 API 的上下文结构](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter2.md#L41) | [m04_context_memory](../llm_agent/m04_context_memory/README.md) · [m15_claude_api](../llm_agent/m15_claude_api/README.md) | 概念与机制对照 |
| [消息的四种角色](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter2.md#L45) | [m04_context_memory](../llm_agent/m04_context_memory/README.md) · [m15_claude_api](../llm_agent/m15_claude_api/README.md) | 概念与机制对照 |
| [单轮对话：最简单的 API 调用](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter2.md#L58) | [m04_context_memory](../llm_agent/m04_context_memory/README.md) · [m15_claude_api](../llm_agent/m15_claude_api/README.md) | 概念与机制对照 |
| [带工具调用的多轮交互：Agent 的核心循环](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter2.md#L95) | [m04_context_memory](../llm_agent/m04_context_memory/README.md) · [m15_claude_api](../llm_agent/m15_claude_api/README.md) | 概念与机制对照 |
| [用代码实现 Agent 的核心循环](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter2.md#L251) | [m04_context_memory](../llm_agent/m04_context_memory/README.md) · [m15_claude_api](../llm_agent/m15_claude_api/README.md) | 概念与机制对照 |
| [从 API 视角看上下文的构成](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter2.md#L370) | [m04_context_memory](../llm_agent/m04_context_memory/README.md) · [m15_claude_api](../llm_agent/m15_claude_api/README.md) | 概念与机制对照 |
| [KV Cache 友好的上下文设计](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter2.md#L435) | [m19_prompt_caching](../llm_agent/m19_prompt_caching/README.md) · [m23_context_cache](../llm_agent/m23_context_cache/README.md) · [m41_system_design](../llm_agent/m41_system_design/README.md) | 概念与机制对照 |
| [从 API 消息到模型 Token：Chat Template](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter2.md#L502) | [m19_prompt_caching](../llm_agent/m19_prompt_caching/README.md) · [m23_context_cache](../llm_agent/m23_context_cache/README.md) · [m41_system_design](../llm_agent/m41_system_design/README.md) | 概念与机制对照 |
| [KV Cache 的原理与约束](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter2.md#L526) | [m19_prompt_caching](../llm_agent/m19_prompt_caching/README.md) · [m23_context_cache](../llm_agent/m23_context_cache/README.md) · [m41_system_design](../llm_agent/m41_system_design/README.md) · [m01_kv_cache](../llm_infer/m01_kv_cache/README.md) | 概念与机制对照 |
| [KV Cache 与 Prompt Cache：两个层级的缓存](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter2.md#L556) | [m19_prompt_caching](../llm_agent/m19_prompt_caching/README.md) · [m23_context_cache](../llm_agent/m23_context_cache/README.md) · [m41_system_design](../llm_agent/m41_system_design/README.md) | 概念与机制对照 |
| [缓存作为架构约束](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter2.md#L560) | [m19_prompt_caching](../llm_agent/m19_prompt_caching/README.md) · [m23_context_cache](../llm_agent/m23_context_cache/README.md) · [m41_system_design](../llm_agent/m41_system_design/README.md) | 概念与机制对照 |
| [KV Cache 未必是一次性的：可编辑、可组合的“笔记”](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter2.md#L580) | [m19_prompt_caching](../llm_agent/m19_prompt_caching/README.md) · [m23_context_cache](../llm_agent/m23_context_cache/README.md) · [m41_system_design](../llm_agent/m41_system_design/README.md) | 概念与机制对照 |
| [承上启下：从缓存机制到上下文内容的设计](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter2.md#L594) | [m19_prompt_caching](../llm_agent/m19_prompt_caching/README.md) · [m23_context_cache](../llm_agent/m23_context_cache/README.md) · [m41_system_design](../llm_agent/m41_system_design/README.md) | 概念与机制对照 |
| [提示工程：优化系统提示词](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter2.md#L602) | [m26_tool_discovery](../llm_agent/m26_tool_discovery/README.md) | 概念与机制对照 |
| [语气与风格：系统提示词的“人格”](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter2.md#L610) | [m26_tool_discovery](../llm_agent/m26_tool_discovery/README.md) | 概念与机制对照 |
| [结构化提示：系统提示词的“格式”](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter2.md#L614) | [m26_tool_discovery](../llm_agent/m26_tool_discovery/README.md) | 概念与机制对照 |
| [流程驱动 vs 规则堆砌：系统提示词的“组织方式”](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter2.md#L643) | [m26_tool_discovery](../llm_agent/m26_tool_discovery/README.md) | 概念与机制对照 |
| [业务规则细化：系统提示词的“内容”](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter2.md#L672) | [m26_tool_discovery](../llm_agent/m26_tool_discovery/README.md) | 概念与机制对照 |
| [Few-shot 示例：何时给模型看例子](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter2.md#L692) | [m26_tool_discovery](../llm_agent/m26_tool_discovery/README.md) | 概念与机制对照 |
| [工具定义的设计](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter2.md#L700) | [m26_tool_discovery](../llm_agent/m26_tool_discovery/README.md) | 概念与机制对照 |
| [提示注入：上下文安全的核心威胁](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter2.md#L730) | [m12_guardrails](../llm_agent/m12_guardrails/README.md) · [m26_tool_discovery](../llm_agent/m26_tool_discovery/README.md) | 概念与机制对照 |
| [动态提示词与 Agent Skills](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter2.md#L765) | [m05_extensibility](../llm_agent/m05_extensibility/README.md) · [m33_skill_evolution](../llm_agent/m33_skill_evolution/README.md) | 概念与机制对照 |
| [Skills：领域能力的可组合单元](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter2.md#L776) | [m05_extensibility](../llm_agent/m05_extensibility/README.md) · [m33_skill_evolution](../llm_agent/m33_skill_evolution/README.md) | 概念与机制对照 |
| [如何编写一份可用的 Skill](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter2.md#L794) | [m05_extensibility](../llm_agent/m05_extensibility/README.md) · [m33_skill_evolution](../llm_agent/m33_skill_evolution/README.md) | 概念与机制对照 |
| [Skills 在上下文中的位置](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter2.md#L814) | [m05_extensibility](../llm_agent/m05_extensibility/README.md) · [m33_skill_evolution](../llm_agent/m33_skill_evolution/README.md) | 概念与机制对照 |
| [Skills 与工具的关系](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter2.md#L830) | [m05_extensibility](../llm_agent/m05_extensibility/README.md) · [m33_skill_evolution](../llm_agent/m33_skill_evolution/README.md) | 概念与机制对照 |
| [Agent 状态栏：通过元信息增强 Agent 轨迹管理](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter2.md#L857) | [m41_system_design](../llm_agent/m41_system_design/README.md) · [m23_context_cache](../llm_agent/m23_context_cache/README.md) | 概念与机制对照 |
| [Agent 状态栏的理论基础](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter2.md#L867) | [m41_system_design](../llm_agent/m41_system_design/README.md) · [m23_context_cache](../llm_agent/m23_context_cache/README.md) | 概念与机制对照 |
| [Agent 状态栏的构成](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter2.md#L908) | [m41_system_design](../llm_agent/m41_system_design/README.md) · [m23_context_cache](../llm_agent/m23_context_cache/README.md) | 概念与机制对照 |
| [Agent 状态栏在上下文中的具体位置](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter2.md#L920) | [m41_system_design](../llm_agent/m41_system_design/README.md) · [m23_context_cache](../llm_agent/m23_context_cache/README.md) | 概念与机制对照 |
| [状态更新的两种实现与缓存代价](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter2.md#L951) | [m41_system_design](../llm_agent/m41_system_design/README.md) · [m23_context_cache](../llm_agent/m23_context_cache/README.md) | 概念与机制对照 |
| [上下文压缩策略](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter2.md#L991) | [m14_context_engineering](../llm_agent/m14_context_engineering/README.md) · [m23_context_cache](../llm_agent/m23_context_cache/README.md) · [m41_system_design](../llm_agent/m41_system_design/README.md) | 概念与机制对照 |
| [为什么需要压缩：不只是长度问题](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter2.md#L995) | [m14_context_engineering](../llm_agent/m14_context_engineering/README.md) · [m23_context_cache](../llm_agent/m23_context_cache/README.md) · [m41_system_design](../llm_agent/m41_system_design/README.md) | 概念与机制对照 |
| [上下文学习的内部机制：检索而非推理](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter2.md#L1007) | [m14_context_engineering](../llm_agent/m14_context_engineering/README.md) · [m23_context_cache](../llm_agent/m23_context_cache/README.md) · [m41_system_design](../llm_agent/m41_system_design/README.md) | 概念与机制对照 |
| [压缩与 KV Cache：看似矛盾，实则互补](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter2.md#L1028) | [m14_context_engineering](../llm_agent/m14_context_engineering/README.md) · [m23_context_cache](../llm_agent/m23_context_cache/README.md) · [m41_system_design](../llm_agent/m41_system_design/README.md) | 概念与机制对照 |
| [生产级的分层压缩机制](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter2.md#L1066) | [m14_context_engineering](../llm_agent/m14_context_engineering/README.md) · [m23_context_cache](../llm_agent/m23_context_cache/README.md) · [m41_system_design](../llm_agent/m41_system_design/README.md) | 概念与机制对照 |
| [压缩策略的设计原则](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter2.md#L1076) | [m14_context_engineering](../llm_agent/m14_context_engineering/README.md) · [m23_context_cache](../llm_agent/m23_context_cache/README.md) · [m41_system_design](../llm_agent/m41_system_design/README.md) | 概念与机制对照 |
| [隔离优于压缩：子 Agent 上下文隔离](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter2.md#L1089) | [m07_subagents](../llm_agent/m07_subagents/README.md) · [m30_collaboration](../llm_agent/m30_collaboration/README.md) | 概念与机制对照 |

### 第 3 章

| 书中知识点 | 本库讲解与代码入口 | 覆盖层次 |
|---|---|---|
| [用户记忆系统](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter3.md#L15) | [m42_memory_privacy](../llm_agent/m42_memory_privacy/README.md) · [m20_memory_lifecycle](../llm_agent/m20_memory_lifecycle/README.md) | 概念与机制对照 |
| [记忆能力的评估：三层次框架](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter3.md#L43) | [m42_memory_privacy](../llm_agent/m42_memory_privacy/README.md) · [m20_memory_lifecycle](../llm_agent/m20_memory_lifecycle/README.md) | 概念与机制对照 |
| [记忆的层次结构](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter3.md#L72) | [m42_memory_privacy](../llm_agent/m42_memory_privacy/README.md) · [m20_memory_lifecycle](../llm_agent/m20_memory_lifecycle/README.md) | 概念与机制对照 |
| [用户记忆的四种存储格式](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter3.md#L88) | [m42_memory_privacy](../llm_agent/m42_memory_privacy/README.md) · [m20_memory_lifecycle](../llm_agent/m20_memory_lifecycle/README.md) | 概念与机制对照 |
| [进阶知识表示形态：可执行代码](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter3.md#L113) | [m38_programmatic_actions](../llm_agent/m38_programmatic_actions/README.md) · [m42_memory_privacy](../llm_agent/m42_memory_privacy/README.md) | 概念与机制对照 |
| [用户记忆的认知科学基础](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter3.md#L170) | [m42_memory_privacy](../llm_agent/m42_memory_privacy/README.md) · [m33_skill_evolution](../llm_agent/m33_skill_evolution/README.md) | 概念与机制对照 |
| [记忆框架案例](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter3.md#L192) | [m42_memory_privacy](../llm_agent/m42_memory_privacy/README.md) · [m33_skill_evolution](../llm_agent/m33_skill_evolution/README.md) | 案例原理映射 |
| [记忆压缩与整理机制](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter3.md#L221) | [m42_memory_privacy](../llm_agent/m42_memory_privacy/README.md) · [m33_skill_evolution](../llm_agent/m33_skill_evolution/README.md) | 概念与机制对照 |
| [隐私保护：日志脱敏](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter3.md#L231) | [m42_memory_privacy](../llm_agent/m42_memory_privacy/README.md) · [m33_skill_evolution](../llm_agent/m33_skill_evolution/README.md) | 概念与机制对照 |
| [RAG 基础：构建 Agent 的知识获取管道](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter3.md#L243) | [m16_rag](../llm_agent/m16_rag/README.md) | 概念与机制对照 |
| [文档分块（Chunking）](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter3.md#L270) | [m16_rag](../llm_agent/m16_rag/README.md) | 概念与机制对照 |
| [稠密嵌入：从词汇关联到语义理解](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter3.md#L286) | [m16_rag](../llm_agent/m16_rag/README.md) · [m43_vector_index](../llm_agent/m43_vector_index/README.md) | 概念与机制对照 |
| [从 Word2Vec 到上下文感知](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter3.md#L304) | [m16_rag](../llm_agent/m16_rag/README.md) · [m43_vector_index](../llm_agent/m43_vector_index/README.md) | 概念与机制对照 |
| [稀疏嵌入：精确匹配的关键词检索](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter3.md#L332) | [m16_rag](../llm_agent/m16_rag/README.md) | 概念与机制对照 |
| [从 TF-IDF 到 BM25](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter3.md#L336) | [m16_rag](../llm_agent/m16_rag/README.md) | 概念与机制对照 |
| [混合检索：两全其美的艺术](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter3.md#L385) | [m16_rag](../llm_agent/m16_rag/README.md) | 概念与机制对照 |
| [超越扁平文本：知识的组织与检索](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter3.md#L424) | [m27_knowledge_organization](../llm_agent/m27_knowledge_organization/README.md) | 概念与机制对照 |
| [结构化索引：从信息检索到知识建模](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter3.md#L445) | [m27_knowledge_organization](../llm_agent/m27_knowledge_organization/README.md) | 概念与机制对照 |
| [文件系统范式：用目录结构组织知识](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter3.md#L482) | [m27_knowledge_organization](../llm_agent/m27_knowledge_organization/README.md) | 概念与机制对照 |
| [知识应该如何更新](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter3.md#L505) | [m20_memory_lifecycle](../llm_agent/m20_memory_lifecycle/README.md) · [m27_knowledge_organization](../llm_agent/m27_knowledge_organization/README.md) · [m33_skill_evolution](../llm_agent/m33_skill_evolution/README.md) | 概念与机制对照 |
| [用户记忆和知识库的增量更新](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter3.md#L509) | [m20_memory_lifecycle](../llm_agent/m20_memory_lifecycle/README.md) · [m27_knowledge_organization](../llm_agent/m27_knowledge_organization/README.md) · [m33_skill_evolution](../llm_agent/m33_skill_evolution/README.md) | 概念与机制对照 |
| [用户记忆和知识库的定期整理](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter3.md#L526) | [m20_memory_lifecycle](../llm_agent/m20_memory_lifecycle/README.md) · [m27_knowledge_organization](../llm_agent/m27_knowledge_organization/README.md) · [m33_skill_evolution](../llm_agent/m33_skill_evolution/README.md) | 概念与机制对照 |
| [智能体化 RAG：将知识检索工具化的范式转变](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter3.md#L542) | [m24_agentic_rag](../llm_agent/m24_agentic_rag/README.md) | 概念与机制对照 |
| [RAG 技巧：上下文感知检索](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter3.md#L594) | [m27_knowledge_organization](../llm_agent/m27_knowledge_organization/README.md) | 概念与机制对照 |
| [从数据集中提取深度知识：从信息检索到知识发现](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter3.md#L637) | [m27_knowledge_organization](../llm_agent/m27_knowledge_organization/README.md) | 概念与机制对照 |
| [前沿探索：多模态记忆](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter3.md#L665) | [m42_memory_privacy](../llm_agent/m42_memory_privacy/README.md) · [m27_knowledge_organization](../llm_agent/m27_knowledge_organization/README.md) | 概念与机制对照 |

### 第 4 章

| 书中知识点 | 本库讲解与代码入口 | 覆盖层次 |
|---|---|---|
| [工具的分类](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter4.md#L7) | [m02_tool_use](../llm_agent/m02_tool_use/README.md) · [m38_programmatic_actions](../llm_agent/m38_programmatic_actions/README.md) | 概念与机制对照 |
| [工具设计的通用原则](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter4.md#L34) | [m02_tool_use](../llm_agent/m02_tool_use/README.md) · [m38_programmatic_actions](../llm_agent/m38_programmatic_actions/README.md) | 概念与机制对照 |
| [能力的表达形式：专用工具、通用执行器与 Skill](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter4.md#L38) | [m02_tool_use](../llm_agent/m02_tool_use/README.md) · [m38_programmatic_actions](../llm_agent/m38_programmatic_actions/README.md) | 概念与机制对照 |
| [工具描述的艺术](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter4.md#L70) | [m26_tool_discovery](../llm_agent/m26_tool_discovery/README.md) | 概念与机制对照 |
| [参数传递的保真性](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter4.md#L88) | [m26_tool_discovery](../llm_agent/m26_tool_discovery/README.md) | 概念与机制对照 |
| [工具生态：MCP 与 Skill Hub](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter4.md#L100) | [m09_mcp](../llm_agent/m09_mcp/README.md) · [m26_tool_discovery](../llm_agent/m26_tool_discovery/README.md) · [m33_skill_evolution](../llm_agent/m33_skill_evolution/README.md) | 概念与机制对照 |
| [工具太多怎么办：层次化组织与主动工具发现](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter4.md#L133) | [m26_tool_discovery](../llm_agent/m26_tool_discovery/README.md) | 概念与机制对照 |
| [层次化组织与按需加载](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter4.md#L139) | [m26_tool_discovery](../llm_agent/m26_tool_discovery/README.md) | 概念与机制对照 |
| [模型原生的主动工具发现](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter4.md#L159) | [m26_tool_discovery](../llm_agent/m26_tool_discovery/README.md) | 概念与机制对照 |
| [Skills：把工具发现变成“按需查阅”](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter4.md#L202) | [m26_tool_discovery](../llm_agent/m26_tool_discovery/README.md) | 概念与机制对照 |
| [感知工具](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter4.md#L218) | [m35_visual_grounding](../llm_agent/m35_visual_grounding/README.md) · [m34_voice_interaction](../llm_agent/m34_voice_interaction/README.md) | 概念与机制对照 |
| [多模态感知](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter4.md#L244) | [m35_visual_grounding](../llm_agent/m35_visual_grounding/README.md) · [m34_voice_interaction](../llm_agent/m34_voice_interaction/README.md) | 概念与机制对照 |
| [原生多模态处理](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter4.md#L248) | [m35_visual_grounding](../llm_agent/m35_visual_grounding/README.md) · [m34_voice_interaction](../llm_agent/m34_voice_interaction/README.md) | 概念与机制对照 |
| [提取为文本](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter4.md#L252) | [m35_visual_grounding](../llm_agent/m35_visual_grounding/README.md) · [m34_voice_interaction](../llm_agent/m34_voice_interaction/README.md) | 概念与机制对照 |
| [工具化多模态分析](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter4.md#L258) | [m35_visual_grounding](../llm_agent/m35_visual_grounding/README.md) · [m34_voice_interaction](../llm_agent/m34_voice_interaction/README.md) | 概念与机制对照 |
| [执行工具](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter4.md#L270) | [m28_coding_agent](../llm_agent/m28_coding_agent/README.md) · [m38_programmatic_actions](../llm_agent/m38_programmatic_actions/README.md) · [m12_guardrails](../llm_agent/m12_guardrails/README.md) | 概念与机制对照 |
| [协作工具](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter4.md#L382) | [m07_subagents](../llm_agent/m07_subagents/README.md) · [m11_orchestrator](../llm_agent/m11_orchestrator/README.md) · [m18_a2a](../llm_agent/m18_a2a/README.md) · [m30_collaboration](../llm_agent/m30_collaboration/README.md) | 概念与机制对照 |

### 第 5 章

| 书中知识点 | 本库讲解与代码入口 | 覆盖层次 |
|---|---|---|
| [Coding Agent](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter5.md#L13) | [m28_coding_agent](../llm_agent/m28_coding_agent/README.md) | 机制与实际执行 |
| [Coding 是 Agent 的基础能力](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter5.md#L15) | [m28_coding_agent](../llm_agent/m28_coding_agent/README.md) | 机制与实际执行 |
| [案例：从 Manus 到 OpenClaw——通用 Agent 的 Coding 内核](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter5.md#L58) | [m28_coding_agent](../llm_agent/m28_coding_agent/README.md) | 机制与实际执行 |
| [Coding Agent 的整体流程](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter5.md#L84) | [m28_coding_agent](../llm_agent/m28_coding_agent/README.md) | 机制与实际执行 |
| [Harness 工程在 Coding Agent 中的实践](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter5.md#L127) | [m28_coding_agent](../llm_agent/m28_coding_agent/README.md) | 机制与实际执行 |
| [故障与错误恢复](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter5.md#L172) | [m28_coding_agent](../llm_agent/m28_coding_agent/README.md) · [m29_async_runtime](../llm_agent/m29_async_runtime/README.md) | 机制与实际执行 |
| [Coding Agent 的实现技巧](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter5.md#L227) | [m28_coding_agent](../llm_agent/m28_coding_agent/README.md) | 机制与实际执行 |
| [Coding Agent 中的搜索工具](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter5.md#L270) | [m28_coding_agent](../llm_agent/m28_coding_agent/README.md) | 机制与实际执行 |
| [Coding Agent 中的文件编辑工具](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter5.md#L293) | [m28_coding_agent](../llm_agent/m28_coding_agent/README.md) | 机制与实际执行 |
| [Coding Agent 的安全](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter5.md#L309) | [m28_coding_agent](../llm_agent/m28_coding_agent/README.md) · [m12_guardrails](../llm_agent/m12_guardrails/README.md) | 机制与实际执行 |
| [代码：通用 Agent 的元能力](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter5.md#L351) | [m38_programmatic_actions](../llm_agent/m38_programmatic_actions/README.md) | 概念与机制对照 |
| [代码作为思考工具](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter5.md#L366) | [m38_programmatic_actions](../llm_agent/m38_programmatic_actions/README.md) | 概念与机制对照 |
| [代码作为业务规则的约束](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter5.md#L406) | [m38_programmatic_actions](../llm_agent/m38_programmatic_actions/README.md) | 概念与机制对照 |
| [代码驱动的多媒体生成](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter5.md#L502) | [m38_programmatic_actions](../llm_agent/m38_programmatic_actions/README.md) | 概念与机制对照 |
| [代码作为系统适配器](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter5.md#L589) | [m38_programmatic_actions](../llm_agent/m38_programmatic_actions/README.md) | 概念与机制对照 |
| [代码作为生成式 UI](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter5.md#L631) | [m38_programmatic_actions](../llm_agent/m38_programmatic_actions/README.md) | 概念与机制对照 |
| [代码创造代码：Agent 自举](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter5.md#L728) | [m38_programmatic_actions](../llm_agent/m38_programmatic_actions/README.md) · [m33_skill_evolution](../llm_agent/m33_skill_evolution/README.md) | 概念与机制对照 |

### 第 6 章

| 书中知识点 | 本库讲解与代码入口 | 覆盖层次 |
|---|---|---|
| [模态与触发时机的扩展](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter6.md#L5) | [m29_async_runtime](../llm_agent/m29_async_runtime/README.md) · [m34_voice_interaction](../llm_agent/m34_voice_interaction/README.md) · [m35_visual_grounding](../llm_agent/m35_visual_grounding/README.md) | 机制与实际执行 |
| [异步与事件驱动：当世界主动找上门](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter6.md#L29) | [m29_async_runtime](../llm_agent/m29_async_runtime/README.md) | 机制与实际执行 |
| [为什么需要异步](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter6.md#L35) | [m29_async_runtime](../llm_agent/m29_async_runtime/README.md) | 机制与实际执行 |
| [OpenClaw 的事件驱动机制实现](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter6.md#L49) | [m29_async_runtime](../llm_agent/m29_async_runtime/README.md) | 机制与实际执行 |
| [事件触发工具](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter6.md#L73) | [m29_async_runtime](../llm_agent/m29_async_runtime/README.md) | 机制与实际执行 |
| [用户沟通工具](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter6.md#L85) | [m29_async_runtime](../llm_agent/m29_async_runtime/README.md) | 机制与实际执行 |
| [虚拟身份与隔离执行环境](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter6.md#L101) | [m29_async_runtime](../llm_agent/m29_async_runtime/README.md) | 机制与实际执行 |
| [事件处理机制](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter6.md#L113) | [m29_async_runtime](../llm_agent/m29_async_runtime/README.md) | 机制与实际执行 |
| [不支持原生异步时的兼容方案](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter6.md#L198) | [m29_async_runtime](../llm_agent/m29_async_runtime/README.md) | 机制与实际执行 |
| [模型原生异步：GPT-6 Astra](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter6.md#L275) | [m29_async_runtime](../llm_agent/m29_async_runtime/README.md) | 机制与实际执行 |
| [从“能接收异步消息”到“可靠处理异步任务”](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter6.md#L289) | [m29_async_runtime](../llm_agent/m29_async_runtime/README.md) | 机制与实际执行 |
| [语音：最自然的人机接口](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter6.md#L307) | [m34_voice_interaction](../llm_agent/m34_voice_interaction/README.md) | 概念与机制对照 |
| [交互时序：从级联到全双工](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter6.md#L313) | [m34_voice_interaction](../llm_agent/m34_voice_interaction/README.md) | 概念与机制对照 |
| [范式一 · 级联流水线（Cascading）](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter6.md#L327) | [m34_voice_interaction](../llm_agent/m34_voice_interaction/README.md) | 概念与机制对照 |
| [从串行到流式感知](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter6.md#L358) | [m34_voice_interaction](../llm_agent/m34_voice_interaction/README.md) | 概念与机制对照 |
| [范式二 · 端到端全模态模型（Omni）](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter6.md#L389) | [m34_voice_interaction](../llm_agent/m34_voice_interaction/README.md) | 概念与机制对照 |
| [范式三 · 全双工交互模型](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter6.md#L404) | [m34_voice_interaction](../llm_agent/m34_voice_interaction/README.md) | 概念与机制对照 |
| [认知时序：实时交互与深度思考](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter6.md#L416) | [m34_voice_interaction](../llm_agent/m34_voice_interaction/README.md) | 概念与机制对照 |
| [方案一：快思考回应，慢思考回答](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter6.md#L420) | [m34_voice_interaction](../llm_agent/m34_voice_interaction/README.md) | 概念与机制对照 |
| [方案二：快思考交互，慢思考提醒](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter6.md#L428) | [m34_voice_interaction](../llm_agent/m34_voice_interaction/README.md) | 概念与机制对照 |
| [方案三：端到端思考与表达统一](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter6.md#L432) | [m34_voice_interaction](../llm_agent/m34_voice_interaction/README.md) | 概念与机制对照 |
| [快慢思考分离与端到端思考的取舍](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter6.md#L438) | [m34_voice_interaction](../llm_agent/m34_voice_interaction/README.md) | 概念与机制对照 |
| [更像人的语音合成](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter6.md#L450) | [m34_voice_interaction](../llm_agent/m34_voice_interaction/README.md) | 概念与机制对照 |
| [Computer Use：GUI 自动化 Agent](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter6.md#L460) | [m35_visual_grounding](../llm_agent/m35_visual_grounding/README.md) | 概念与机制对照 |
| [动作空间设计](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter6.md#L479) | [m35_visual_grounding](../llm_agent/m35_visual_grounding/README.md) | 概念与机制对照 |
| [视觉定位（Grounding）](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter6.md#L498) | [m35_visual_grounding](../llm_agent/m35_visual_grounding/README.md) | 概念与机制对照 |
| [能看动画、能听声音的 Computer Use Agent](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter6.md#L548) | [m35_visual_grounding](../llm_agent/m35_visual_grounding/README.md) | 概念与机制对照 |
| [Computer Use 的世界模型](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter6.md#L556) | [m35_visual_grounding](../llm_agent/m35_visual_grounding/README.md) | 概念与机制对照 |
| [移动端：生态壁垒比技术更难](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter6.md#L575) | [m35_visual_grounding](../llm_agent/m35_visual_grounding/README.md) | 概念与机制对照 |
| [机器人操作：以 XLeRobot 整理桌面为例](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter6.md#L585) | [m36_embodied_control](../llm_agent/m36_embodied_control/README.md) | 概念与机制对照 |
| [硬件与算法的分工](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter6.md#L602) | [m36_embodied_control](../llm_agent/m36_embodied_control/README.md) | 概念与机制对照 |
| [机器人控制的基本结构](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter6.md#L624) | [m36_embodied_control](../llm_agent/m36_embodied_control/README.md) | 概念与机制对照 |
| [长程规划与任务分解](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter6.md#L638) | [m36_embodied_control](../llm_agent/m36_embodied_control/README.md) | 概念与机制对照 |
| [VLA 控制](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter6.md#L664) | [m36_embodied_control](../llm_agent/m36_embodied_control/README.md) | 概念与机制对照 |
| [VLA 的局限](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter6.md#L678) | [m36_embodied_control](../llm_agent/m36_embodied_control/README.md) | 概念与机制对照 |
| [世界模型](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter6.md#L689) | [m36_embodied_control](../llm_agent/m36_embodied_control/README.md) | 概念与机制对照 |
| [从仿真环境到真实机器人](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter6.md#L726) | [m36_embodied_control](../llm_agent/m36_embodied_control/README.md) | 概念与机制对照 |

### 第 7 章

| 书中知识点 | 本库讲解与代码入口 | 覆盖层次 |
|---|---|---|
| [一条评估任务的解剖：τ²-bench 的 telecom 领域](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter7.md#L27) | [m31_eval_suite](../llm_agent/m31_eval_suite/README.md) · [m13_evals](../llm_agent/m13_evals/README.md) | 概念与机制对照 |
| [任务定义的四个组成部分](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter7.md#L31) | [m31_eval_suite](../llm_agent/m31_eval_suite/README.md) · [m13_evals](../llm_agent/m13_evals/README.md) | 概念与机制对照 |
| [一次真实运行的轨迹](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter7.md#L95) | [m31_eval_suite](../llm_agent/m31_eval_suite/README.md) · [m13_evals](../llm_agent/m13_evals/README.md) | 概念与机制对照 |
| [评估指标：成功的定义](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter7.md#L129) | [m13_evals](../llm_agent/m13_evals/README.md) | 概念与机制对照 |
| [技术奇观：用 Pass@k 看能力上限](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter7.md#L133) | [m13_evals](../llm_agent/m13_evals/README.md) | 概念与机制对照 |
| [业务可靠性：关注 Pass^k](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter7.md#L147) | [m13_evals](../llm_agent/m13_evals/README.md) | 概念与机制对照 |
| [评估环境](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter7.md#L162) | [m31_eval_suite](../llm_agent/m31_eval_suite/README.md) | 概念与机制对照 |
| [五个组成要素](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter7.md#L166) | [m31_eval_suite](../llm_agent/m31_eval_suite/README.md) | 概念与机制对照 |
| [人机交互型与工具调用型评估环境](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter7.md#L182) | [m31_eval_suite](../llm_agent/m31_eval_suite/README.md) | 概念与机制对照 |
| [评估数据集的设计](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter7.md#L203) | [m31_eval_suite](../llm_agent/m31_eval_suite/README.md) | 概念与机制对照 |
| [基准设计的横向对照](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter7.md#L207) | [m31_eval_suite](../llm_agent/m31_eval_suite/README.md) | 概念与机制对照 |
| [验证器](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter7.md#L222) | [m31_eval_suite](../llm_agent/m31_eval_suite/README.md) | 概念与机制对照 |
| [任务的难度划分](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter7.md#L232) | [m31_eval_suite](../llm_agent/m31_eval_suite/README.md) | 概念与机制对照 |
| [数据泄漏防范](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter7.md#L242) | [m31_eval_suite](../llm_agent/m31_eval_suite/README.md) | 概念与机制对照 |
| [质量控制与长期维护](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter7.md#L250) | [m31_eval_suite](../llm_agent/m31_eval_suite/README.md) | 概念与机制对照 |
| [评估集的三个来源](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter7.md#L274) | [m31_eval_suite](../llm_agent/m31_eval_suite/README.md) | 概念与机制对照 |
| [自动化评估方法](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter7.md#L288) | [m31_eval_suite](../llm_agent/m31_eval_suite/README.md) · [m13_evals](../llm_agent/m13_evals/README.md) | 概念与机制对照 |
| [LLM-as-a-Judge：自动化评估的核心](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter7.md#L304) | [m31_eval_suite](../llm_agent/m31_eval_suite/README.md) · [m13_evals](../llm_agent/m13_evals/README.md) | 概念与机制对照 |
| [失败归因：从整条轨迹定位首个错误](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter7.md#L434) | [m13_evals](../llm_agent/m13_evals/README.md) | 概念与机制对照 |
| [“做对了但说错了” 问题](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter7.md#L464) | [m13_evals](../llm_agent/m13_evals/README.md) | 概念与机制对照 |
| [作用域敏感的文档格式错误](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter7.md#L484) | [m13_evals](../llm_agent/m13_evals/README.md) | 概念与机制对照 |
| [精确复制错误：从 `old_string` mismatch 到逐层定位](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter7.md#L502) | [m13_evals](../llm_agent/m13_evals/README.md) | 概念与机制对照 |
| [端到端回归任务与轨迹前缀回归任务](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter7.md#L513) | [m13_evals](../llm_agent/m13_evals/README.md) | 概念与机制对照 |
| [配对比较与模型排名](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter7.md#L539) | [m13_evals](../llm_agent/m13_evals/README.md) | 概念与机制对照 |
| [评估驱动的模型选型](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter7.md#L558) | [m41_system_design](../llm_agent/m41_system_design/README.md) · [m31_eval_suite](../llm_agent/m31_eval_suite/README.md) · [m23_context_cache](../llm_agent/m23_context_cache/README.md) | 概念与机制对照 |
| [选型的关键维度](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter7.md#L562) | [m41_system_design](../llm_agent/m41_system_design/README.md) · [m31_eval_suite](../llm_agent/m31_eval_suite/README.md) · [m23_context_cache](../llm_agent/m23_context_cache/README.md) | 概念与机制对照 |
| [模型的行为策略](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter7.md#L583) | [m41_system_design](../llm_agent/m41_system_design/README.md) · [m31_eval_suite](../llm_agent/m31_eval_suite/README.md) · [m23_context_cache](../llm_agent/m23_context_cache/README.md) | 概念与机制对照 |
| [Agent 系统的成本分析](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter7.md#L599) | [m41_system_design](../llm_agent/m41_system_design/README.md) · [m31_eval_suite](../llm_agent/m31_eval_suite/README.md) · [m23_context_cache](../llm_agent/m23_context_cache/README.md) | 概念与机制对照 |
| [评估驱动的持续迭代](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter7.md#L646) | [m41_system_design](../llm_agent/m41_system_design/README.md) · [m31_eval_suite](../llm_agent/m31_eval_suite/README.md) · [m23_context_cache](../llm_agent/m23_context_cache/README.md) | 概念与机制对照 |
| [评估结果的统计显著性](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter7.md#L672) | [m13_evals](../llm_agent/m13_evals/README.md) | 概念与机制对照 |
| [Agent 的可观测性](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter7.md#L698) | [m31_eval_suite](../llm_agent/m31_eval_suite/README.md) | 概念与机制对照 |
| [从 Benchmark 报告到系统改进](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter7.md#L718) | [m31_eval_suite](../llm_agent/m31_eval_suite/README.md) | 概念与机制对照 |
| [读懂 Benchmark 报告：发现问题的艺术](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter7.md#L728) | [m31_eval_suite](../llm_agent/m31_eval_suite/README.md) | 概念与机制对照 |
| [从数据到假设：构建改进路线图](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter7.md#L734) | [m31_eval_suite](../llm_agent/m31_eval_suite/README.md) | 概念与机制对照 |
| [从结果到决策：数据驱动的权衡](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter7.md#L742) | [m31_eval_suite](../llm_agent/m31_eval_suite/README.md) | 概念与机制对照 |
| [持续迭代：从第一次改进到系统演化](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter7.md#L756) | [m31_eval_suite](../llm_agent/m31_eval_suite/README.md) | 概念与机制对照 |
| [从外部评估到内部评估：生产级 Agent 的评估基础设施](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter7.md#L777) | [m31_eval_suite](../llm_agent/m31_eval_suite/README.md) | 概念与机制对照 |
| [消融基础设施：理解每个特性的真实贡献](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter7.md#L781) | [m31_eval_suite](../llm_agent/m31_eval_suite/README.md) | 概念与机制对照 |
| [AB 测试方法论：区分机制与目标](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter7.md#L787) | [m31_eval_suite](../llm_agent/m31_eval_suite/README.md) | 概念与机制对照 |
| [双层特性开关系统](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter7.md#L799) | [m31_eval_suite](../llm_agent/m31_eval_suite/README.md) | 概念与机制对照 |
| [提示词敏感性评估](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter7.md#L809) | [m31_eval_suite](../llm_agent/m31_eval_suite/README.md) | 概念与机制对照 |
| [以隐私感知分析为评估基础](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter7.md#L815) | [m31_eval_suite](../llm_agent/m31_eval_suite/README.md) | 概念与机制对照 |
| [从外部到内部：评估思维的转变](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter7.md#L821) | [m31_eval_suite](../llm_agent/m31_eval_suite/README.md) | 概念与机制对照 |
| [仿真环境：从评估到后训练的桥梁](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter7.md#L825) | [m31_eval_suite](../llm_agent/m31_eval_suite/README.md) · [m36_embodied_control](../llm_agent/m36_embodied_control/README.md) · [m32_agent_training](../llm_agent/m32_agent_training/README.md) | 概念与机制对照 |
| [保真度权衡与领域随机化](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter7.md#L846) | [m31_eval_suite](../llm_agent/m31_eval_suite/README.md) · [m36_embodied_control](../llm_agent/m36_embodied_control/README.md) · [m32_agent_training](../llm_agent/m32_agent_training/README.md) | 概念与机制对照 |

### 第 8 章

| 书中知识点 | 本库讲解与代码入口 | 覆盖层次 |
|---|---|---|
| [从预训练到 RL：四阶段全景](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter8.md#L26) | [m37_midtraining](../llm_agent/m37_midtraining/README.md) · [m32_agent_training](../llm_agent/m32_agent_training/README.md) | 概念与机制对照 |
| [预训练在做什么：预测下一个词](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter8.md#L39) | [m37_midtraining](../llm_agent/m37_midtraining/README.md) · [m32_agent_training](../llm_agent/m32_agent_training/README.md) | 概念与机制对照 |
| [Mid-training 的本质：在目标分布上继续学习](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter8.md#L49) | [m37_midtraining](../llm_agent/m37_midtraining/README.md) · [m32_agent_training](../llm_agent/m32_agent_training/README.md) | 概念与机制对照 |
| [SFT 的本质：换了数据的“预测下一个词”](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter8.md#L55) | [m37_midtraining](../llm_agent/m37_midtraining/README.md) · [m32_agent_training](../llm_agent/m32_agent_training/README.md) | 概念与机制对照 |
| [什么时候需要先补底座，再做 SFT/RL](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter8.md#L68) | [m37_midtraining](../llm_agent/m37_midtraining/README.md) · [m32_agent_training](../llm_agent/m32_agent_training/README.md) | 概念与机制对照 |
| [SFT 与 RL 的本质区别](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter8.md#L86) | [m37_midtraining](../llm_agent/m37_midtraining/README.md) · [m32_agent_training](../llm_agent/m32_agent_training/README.md) | 概念与机制对照 |
| [从经典 RL Agent 到现代 Agent `[可选阅读]`](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter8.md#L122) | [m32_agent_training](../llm_agent/m32_agent_training/README.md) · [m36_embodied_control](../llm_agent/m36_embodied_control/README.md) | 概念与机制对照 |
| [Agent 与环境的交互](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter8.md#L124) | [m32_agent_training](../llm_agent/m32_agent_training/README.md) · [m36_embodied_control](../llm_agent/m36_embodied_control/README.md) | 概念与机制对照 |
| [两种动作表示：经典 RL 设置与 LLM 的变长策略](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter8.md#L158) | [m32_agent_training](../llm_agent/m32_agent_training/README.md) · [m36_embodied_control](../llm_agent/m36_embodied_control/README.md) | 概念与机制对照 |
| [模型预训练基础 `[可选阅读]`](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter8.md#L250) | [m37_midtraining](../llm_agent/m37_midtraining/README.md) · [llm_train](../llm_train/README.md) · [llm_models](../llm_models/README.md) | 概念与机制对照 |
| [Mid-training：补知识与基础能力](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter8.md#L276) | [m37_midtraining](../llm_agent/m37_midtraining/README.md) | 机制与实际执行 |
| [Mid-training 数据如何构造](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter8.md#L287) | [m37_midtraining](../llm_agent/m37_midtraining/README.md) | 机制与实际执行 |
| [SFT（监督微调）](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter8.md#L314) | [m32_agent_training](../llm_agent/m32_agent_training/README.md) · [m25_trajectory_learning](../llm_agent/m25_trajectory_learning/README.md) · [sft](../llm_finetune/run_finetune/sft/README.md) | 概念与机制对照 |
| [SFT 数据合成：从示范到可训练轨迹](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter8.md#L370) | [m32_agent_training](../llm_agent/m32_agent_training/README.md) · [m25_trajectory_learning](../llm_agent/m25_trajectory_learning/README.md) | 概念与机制对照 |
| [何时选择 Mid-training、SFT 与 RL](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter8.md#L386) | [m37_midtraining](../llm_agent/m37_midtraining/README.md) · [m32_agent_training](../llm_agent/m32_agent_training/README.md) | 概念与机制对照 |
| [单轮强化学习：记忆与泛化的对照](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter8.md#L410) | [m32_agent_training](../llm_agent/m32_agent_training/README.md) · [grpo](../llm_finetune/run_finetune/grpo/README.md) | 概念与机制对照 |
| [RL 算法：从 16 次 rollout 到一次参数更新](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter8.md#L482) | [m32_agent_training](../llm_agent/m32_agent_training/README.md) · [methods](../llm_finetune/methods/grpo.py) | 机制与实际执行 |
| [为什么 LLM RL 通常优先 On-Policy](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter8.md#L504) | [m32_agent_training](../llm_agent/m32_agent_training/README.md) · [methods](../llm_finetune/methods/grpo.py) | 概念与机制对照 |
| [看似 On-Policy，为什么仍会被数值误差拖垮](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter8.md#L517) | [m32_agent_training](../llm_agent/m32_agent_training/README.md) | 概念与机制对照 |
| [RL 环境：从评估到仿真](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter8.md#L530) | [m31_eval_suite](../llm_agent/m31_eval_suite/README.md) · [m32_agent_training](../llm_agent/m32_agent_training/README.md) | 概念与机制对照 |
| [环境：模型练习的场地](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter8.md#L534) | [m31_eval_suite](../llm_agent/m31_eval_suite/README.md) · [m32_agent_training](../llm_agent/m32_agent_training/README.md) | 概念与机制对照 |
| [造不出环境怎么办：让模型扮演环境](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter8.md#L544) | [m31_eval_suite](../llm_agent/m31_eval_suite/README.md) · [m32_agent_training](../llm_agent/m32_agent_training/README.md) | 概念与机制对照 |
| [环境、任务分布与评估隔离](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter8.md#L554) | [m31_eval_suite](../llm_agent/m31_eval_suite/README.md) · [m32_agent_training](../llm_agent/m32_agent_training/README.md) | 概念与机制对照 |
| [从单轮到多轮：任务场景与信用分配](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter8.md#L564) | [m32_agent_training](../llm_agent/m32_agent_training/README.md) | 概念与机制对照 |
| [多轮任务的核心挑战](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter8.md#L566) | [m32_agent_training](../llm_agent/m32_agent_training/README.md) | 概念与机制对照 |
| [工具调用：把环境带进 Agent](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter8.md#L584) | [m32_agent_training](../llm_agent/m32_agent_training/README.md) | 概念与机制对照 |
| [奖励设计：如何把任务目标变成学习信号](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter8.md#L610) | [m25_trajectory_learning](../llm_agent/m25_trajectory_learning/README.md) · [m32_agent_training](../llm_agent/m32_agent_training/README.md) | 概念与机制对照 |
| [奖励来自哪里：规则、人类偏好与模型评判](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter8.md#L614) | [m25_trajectory_learning](../llm_agent/m25_trajectory_learning/README.md) · [m32_agent_training](../llm_agent/m32_agent_training/README.md) | 概念与机制对照 |
| [奖励在什么时候给：结果还是过程](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter8.md#L624) | [m25_trajectory_learning](../llm_agent/m25_trajectory_learning/README.md) · [m32_agent_training](../llm_agent/m32_agent_training/README.md) · [prm](../llm_finetune/run_finetune/prm/README.md) | 概念与机制对照 |
| [奖励需要表达多少信息：标量、向量与生成式诊断](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter8.md#L634) | [m25_trajectory_learning](../llm_agent/m25_trajectory_learning/README.md) · [m32_agent_training](../llm_agent/m32_agent_training/README.md) | 概念与机制对照 |
| [结果正确还不够：路径约束与 RLVP](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter8.md#L644) | [m25_trajectory_learning](../llm_agent/m25_trajectory_learning/README.md) · [m32_agent_training](../llm_agent/m32_agent_training/README.md) | 概念与机制对照 |
| [蒸馏：提升样本效率](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter8.md#L670) | [m32_agent_training](../llm_agent/m32_agent_training/README.md) · [distill](../llm_finetune/run_finetune/distill/README.md) | 概念与机制对照 |
| [在轨蒸馏：让一次 rollout 产生密集监督](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter8.md#L678) | [m32_agent_training](../llm_agent/m32_agent_training/README.md) · [on_policy_distill](../llm_finetune/run_finetune/on_policy_distill/README.md) | 概念与机制对照 |
| [没有更强的教师怎么办：On-Policy 自蒸馏](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter8.md#L711) | [m32_agent_training](../llm_agent/m32_agent_training/README.md) · [on_policy_distill](../llm_finetune/run_finetune/on_policy_distill/README.md) · [m44_self_distillation](../llm_agent/m44_self_distillation/README.md) | 概念与机制对照 |
| [从问题案例到后训练](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter8.md#L733) | [m21_completion_verification](../llm_agent/m21_completion_verification/README.md) · [m26_tool_discovery](../llm_agent/m26_tool_discovery/README.md) · [m28_coding_agent](../llm_agent/m28_coding_agent/README.md) · [m32_agent_training](../llm_agent/m32_agent_training/README.md) | 案例原理映射 |
| [案例 1：Coding Agent 过早结束](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter8.md#L746) | [m21_completion_verification](../llm_agent/m21_completion_verification/README.md) · [m26_tool_discovery](../llm_agent/m26_tool_discovery/README.md) · [m28_coding_agent](../llm_agent/m28_coding_agent/README.md) · [m32_agent_training](../llm_agent/m32_agent_training/README.md) | 案例原理映射 |
| [案例 2：中文引号](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter8.md#L764) | [m21_completion_verification](../llm_agent/m21_completion_verification/README.md) · [m26_tool_discovery](../llm_agent/m26_tool_discovery/README.md) · [m28_coding_agent](../llm_agent/m28_coding_agent/README.md) · [m32_agent_training](../llm_agent/m32_agent_training/README.md) | 案例原理映射 |
| [案例 3：编辑文件经常失败](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter8.md#L778) | [m21_completion_verification](../llm_agent/m21_completion_verification/README.md) · [m26_tool_discovery](../llm_agent/m26_tool_discovery/README.md) · [m28_coding_agent](../llm_agent/m28_coding_agent/README.md) · [m32_agent_training](../llm_agent/m32_agent_training/README.md) | 案例原理映射 |
| [后训练实践要点](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter8.md#L796) | [m37_midtraining](../llm_agent/m37_midtraining/README.md) · [m32_agent_training](../llm_agent/m32_agent_training/README.md) | 概念与机制对照 |

### 第 9 章

| 书中知识点 | 本库讲解与代码入口 | 覆盖层次 |
|---|---|---|
| [从运行轨迹中获得学习信号](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter9.md#L19) | [m13_evals](../llm_agent/m13_evals/README.md) · [m21_completion_verification](../llm_agent/m21_completion_verification/README.md) | 概念与机制对照 |
| [Agent 持续进化的四种方法](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter9.md#L56) | [m33_skill_evolution](../llm_agent/m33_skill_evolution/README.md) | 概念与机制对照 |
| [将经验沉淀为知识](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter9.md#L75) | [m20_memory_lifecycle](../llm_agent/m20_memory_lifecycle/README.md) · [m33_skill_evolution](../llm_agent/m33_skill_evolution/README.md) | 概念与机制对照 |
| [将经验写成指令](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter9.md#L97) | [m33_skill_evolution](../llm_agent/m33_skill_evolution/README.md) | 概念与机制对照 |
| [例子一：转接边界的规则化](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter9.md#L105) | [m33_skill_evolution](../llm_agent/m33_skill_evolution/README.md) | 概念与机制对照 |
| [例子二：需求澄清 Skill——从“直接开工”到“先确认再执行”](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter9.md#L135) | [m33_skill_evolution](../llm_agent/m33_skill_evolution/README.md) | 概念与机制对照 |
| [将经验写成程序](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter9.md#L167) | [m28_coding_agent](../llm_agent/m28_coding_agent/README.md) · [m38_programmatic_actions](../llm_agent/m38_programmatic_actions/README.md) · [m33_skill_evolution](../llm_agent/m33_skill_evolution/README.md) | 概念与机制对照 |
| [案例：DeepSeek Harness 以“一切皆插件”实现自我进化](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter9.md#L228) | [m28_coding_agent](../llm_agent/m28_coding_agent/README.md) · [m38_programmatic_actions](../llm_agent/m38_programmatic_actions/README.md) · [m33_skill_evolution](../llm_agent/m33_skill_evolution/README.md) | 案例原理映射 |
| [将经验写入参数](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter9.md#L248) | [m32_agent_training](../llm_agent/m32_agent_training/README.md) | 概念与机制对照 |
| [从更新产物到更新“更新方法”](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter9.md#L256) | [m39_meta_optimization](../llm_agent/m39_meta_optimization/README.md) | 概念与机制对照 |
| [构建可长期运行的持续进化闭环](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter9.md#L286) | [full_learning_loop](../llm_agent/full_learning_loop/README.md) · [m33_skill_evolution](../llm_agent/m33_skill_evolution/README.md) | 概念与机制对照 |
| [可验证闭环的边界：当“完成”不等于“进步”](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter9.md#L329) | [full_learning_loop](../llm_agent/full_learning_loop/README.md) · [m21_completion_verification](../llm_agent/m21_completion_verification/README.md) | 概念与机制对照 |
| [持续进化的安全边界](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter9.md#L342) | [m33_skill_evolution](../llm_agent/m33_skill_evolution/README.md) · [m42_memory_privacy](../llm_agent/m42_memory_privacy/README.md) | 概念与机制对照 |
| [睡眠学习：整合、遗忘与能力保鲜](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter9.md#L352) | [m33_skill_evolution](../llm_agent/m33_skill_evolution/README.md) · [m42_memory_privacy](../llm_agent/m42_memory_privacy/README.md) | 概念与机制对照 |

### 第 10 章

| 书中知识点 | 本库讲解与代码入口 | 覆盖层次 |
|---|---|---|
| [多 Agent 协作的分类框架](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter10.md#L11) | [m30_collaboration](../llm_agent/m30_collaboration/README.md) · [m07_subagents](../llm_agent/m07_subagents/README.md) · [m11_orchestrator](../llm_agent/m11_orchestrator/README.md) | 概念与机制对照 |
| [维度一：上下文是否共享](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter10.md#L15) | [m30_collaboration](../llm_agent/m30_collaboration/README.md) · [m07_subagents](../llm_agent/m07_subagents/README.md) · [m11_orchestrator](../llm_agent/m11_orchestrator/README.md) | 概念与机制对照 |
| [维度二：协作拓扑](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter10.md#L34) | [m30_collaboration](../llm_agent/m30_collaboration/README.md) · [m07_subagents](../llm_agent/m07_subagents/README.md) · [m11_orchestrator](../llm_agent/m11_orchestrator/README.md) | 概念与机制对照 |
| [多 Agent 何时真正优于单 Agent](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter10.md#L46) | [m30_collaboration](../llm_agent/m30_collaboration/README.md) · [m07_subagents](../llm_agent/m07_subagents/README.md) · [m11_orchestrator](../llm_agent/m11_orchestrator/README.md) | 概念与机制对照 |
| [共享上下文的多 Agent 协作](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter10.md#L79) | [m30_collaboration](../llm_agent/m30_collaboration/README.md) · [m04_context_memory](../llm_agent/m04_context_memory/README.md) | 概念与机制对照 |
| [不共享上下文的多 Agent 协作](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter10.md#L102) | [m07_subagents](../llm_agent/m07_subagents/README.md) · [m30_collaboration](../llm_agent/m30_collaboration/README.md) | 概念与机制对照 |
| [Agent 眼中的文件系统](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter10.md#L133) | [m07_subagents](../llm_agent/m07_subagents/README.md) · [m30_collaboration](../llm_agent/m30_collaboration/README.md) | 概念与机制对照 |
| [Agent 间的通信与控制](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter10.md#L166) | [m07_subagents](../llm_agent/m07_subagents/README.md) · [m30_collaboration](../llm_agent/m30_collaboration/README.md) | 概念与机制对照 |
| [对等协作模式：相互制衡与迭代改进](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter10.md#L194) | [m30_collaboration](../llm_agent/m30_collaboration/README.md) | 概念与机制对照 |
| [Loop 工程](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter10.md#L200) | [m30_collaboration](../llm_agent/m30_collaboration/README.md) | 概念与机制对照 |
| [提议者-审核者范式](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter10.md#L228) | [m30_collaboration](../llm_agent/m30_collaboration/README.md) | 概念与机制对照 |
| [辩论模式](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter10.md#L268) | [m30_collaboration](../llm_agent/m30_collaboration/README.md) | 概念与机制对照 |
| [头脑风暴模式](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter10.md#L276) | [m30_collaboration](../llm_agent/m30_collaboration/README.md) | 概念与机制对照 |
| [专家小组模式](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter10.md#L280) | [m30_collaboration](../llm_agent/m30_collaboration/README.md) | 概念与机制对照 |
| [管理者模式：中心化协调](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter10.md#L284) | [m11_orchestrator](../llm_agent/m11_orchestrator/README.md) · [m30_collaboration](../llm_agent/m30_collaboration/README.md) | 概念与机制对照 |
| [去中心化模式](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter10.md#L443) | [m30_collaboration](../llm_agent/m30_collaboration/README.md) · [m29_async_runtime](../llm_agent/m29_async_runtime/README.md) | 概念与机制对照 |
| [跨组织协作：A2A 协议](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter10.md#L513) | [m18_a2a](../llm_agent/m18_a2a/README.md) | 概念与机制对照 |
| [多 Agent 协作的失败模式](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter10.md#L523) | [m30_collaboration](../llm_agent/m30_collaboration/README.md) · [m29_async_runtime](../llm_agent/m29_async_runtime/README.md) | 概念与机制对照 |
| [失败模式一：共享文件系统的并发冲突](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter10.md#L537) | [m30_collaboration](../llm_agent/m30_collaboration/README.md) · [m29_async_runtime](../llm_agent/m29_async_runtime/README.md) | 概念与机制对照 |
| [失败模式二：错误的级联放大](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter10.md#L549) | [m30_collaboration](../llm_agent/m30_collaboration/README.md) · [m29_async_runtime](../llm_agent/m29_async_runtime/README.md) | 概念与机制对照 |
| [失败模式三：同质趋同](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter10.md#L555) | [m30_collaboration](../llm_agent/m30_collaboration/README.md) · [m29_async_runtime](../llm_agent/m29_async_runtime/README.md) | 概念与机制对照 |
| [失败模式四：互相扯皮](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter10.md#L561) | [m30_collaboration](../llm_agent/m30_collaboration/README.md) · [m29_async_runtime](../llm_agent/m29_async_runtime/README.md) | 概念与机制对照 |
| [失败模式五：循环失控](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter10.md#L567) | [m30_collaboration](../llm_agent/m30_collaboration/README.md) · [m29_async_runtime](../llm_agent/m29_async_runtime/README.md) | 概念与机制对照 |
| [失败模式六：理解债与认知投降](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter10.md#L571) | [m30_collaboration](../llm_agent/m30_collaboration/README.md) · [m29_async_runtime](../llm_agent/m29_async_runtime/README.md) | 概念与机制对照 |
| [Agent 社会](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter10.md#L581) | [m40_agent_society](../llm_agent/m40_agent_society/README.md) | 概念与机制对照 |
| [斯坦福 AI 小镇：生成式 Agent 的社会模拟](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter10.md#L591) | [m40_agent_society](../llm_agent/m40_agent_society/README.md) | 案例原理映射 |
| [Agentopia：十年尺度的长期生活模拟](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter10.md#L635) | [m40_agent_society](../llm_agent/m40_agent_society/README.md) | 案例原理映射 |
| [Moltbook：当 Agent 拥有自己的社交网络](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter10.md#L650) | [m40_agent_society](../llm_agent/m40_agent_society/README.md) | 案例原理映射 |
| [从虚拟社会到经济竞争：Vending-Bench Arena](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter10.md#L656) | [m40_agent_society](../llm_agent/m40_agent_society/README.md) | 案例原理映射 |
| [Agent 经济：Pinchwork 与 RentAHuman](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter10.md#L670) | [m40_agent_society](../llm_agent/m40_agent_society/README.md) | 案例原理映射 |
| [信息不对称下的策略博弈：狼人杀](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter10.md#L678) | [m40_agent_society](../llm_agent/m40_agent_society/README.md) | 案例原理映射 |
