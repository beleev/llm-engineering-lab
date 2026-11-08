# 六阶段知识复查：仍需补齐的教学环节

复查日期：2026-11-08。Lab 基线：`30210d3`；对照书：`dbc046eb896ac4e39aa19c7774c8bf49583b89a6`。

> 后续状态：本报告保留复查时发现的问题；本轮九项补充的实现与验收请看[补齐验收记录](knowledge-completion.md)。下文“缺失”描述的是基线 30210d3。

## 结论与范围

还有遗漏。此前“书中知识已完全补齐”的判断过于宽泛：287 个二至四级标题都有索引映射，但其中有些映射只指向相邻主题，没有充分解释原知识点。Chat Template 是明确反例；可观测性的 span 树和拒绝采样微调也只覆盖了部分环节。

本次检查六阶段目录、相关正文和术语、模块 README、关键实现及书中对应段落。先用全文检索定位，再阅读实现排除误报。结论针对当前仓库的教学完整性，不是最新论文榜单，也不要求把所有 CPU 教学实验改成生产系统。以下列出 9 项确认的缺口或薄弱环节，**不是全领域的穷尽清单**。

原覆盖检查器本次输出：`287 个知识标题, 10 章, 本地入口完整; 源文与标题全集一致`。它验证标题、文件存在和源文件哈希；不会判断正文是否讲清了机制，也不逐个验收引用块中的实验及段落知识。运行成功不能替代语义审查。

## 需要优先补的三项书中知识

### 1. Chat Template 与多轮训练数据（P0，讲解和实验缺失）

- 书中入口：[第 2 章，API 消息到模型 Token](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter2.md#L502)。
- 现有覆盖：`ch2-L502` 映射到 m19/m23/m41；[m41](../llm_agent/m41_system_design/README.md) 只说明模板会影响缓存。模型 API 的消息转换已经存在，但那是 JSON 消息格式转换，不能代替消息到 token 的序列化讲解。
- 明确边界：[SFT 讲义](../llm_finetune/run_finetune/sft/README.md) 写明没有对话模板、多轮监督及变长样本实验；[SFT 实现](../llm_finetune/methods/sft.py) 也是 prompt/回复两段。
- 应补：结构化消息 → 角色/边界标记 → token → assistant 监督位置；训练和推理使用同一模板；EOS、padding 与截断如何影响多轮回复和工具调用配对。供应商具体格式应独立核实，不能把一个模型的协议推广到所有模型。
- 最小验收：同一多轮对话打印 token 与 loss mask；错误角色、重复结束标记、截断半个工具结果分别给出失败对照；保留现有左 padding 与 EOS 测试。
- 放置：`finetune-sft` 为主，`basic-data` 和 `agent-tools-permissions` 交叉引用，不新增二级主题。

### 2. 可观测性的 trace / span 树（P0，已有日志但结构不足）

- 书中入口：[第 7 章，Agent 的可观测性](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter7.md#L698)。
- 现有覆盖：`ch7-L698` 指向 [m31](../llm_agent/m31_eval_suite/README.md)，主要是“记录调用、错误、延迟、usage”的原则。[m13 diagnostics](../llm_agent/m13_evals/diagnostics.py) 已能定位首错，不能说仓库没有轨迹诊断。
- 应补：一次任务与 LLM/工具/检索子调用的父子关系；并发 span 的关键路径与总耗时；重试 attempt 与逻辑操作分开计数；真实 token usage、脱敏和错误类别如何关联到一次请求。
- 最小验收：并行工具、一次失败重试、一次取消各输出同一套可重建的 span 树；工具执行次数、模型尝试次数、费用和端到端耗时均能核对。先做标准库 JSON 导出即可，不要求部署监控平台。
- 放置：`agent-evals`、`agent-eval-suite` 和 `agent-async-runtime`。

### 3. RFT：筛选成功轨迹后再训练（P1，缺完整训练基线）

- 书中入口：[第 8 章，SFT 数据合成](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter8.md#L370)。
- 现有覆盖：[m25](../llm_agent/m25_trajectory_learning/demo.py) 验证预设候选并导出数据；[m32](../llm_agent/m32_agent_training/demo.py) 有示范 SFT 和在线策略梯度，但没有当前模型多次采样 → 筛选 → 再做 SFT 的对照。[微调首页](../llm_finetune/README.md) 明确把 RFT 列在未实现项。
- 应补：RFT 与纯示范 SFT、在线 RL 的数据来源和更新方式区别；无成功候选时不能凭空造监督；只留容易任务会产生选择偏差。
- 最小验收：在同一小模型、任务拆分和采样预算下比较示范 SFT / RFT / 在线 RL，报告候选通过率、保留数据的任务分布及留出集结果，允许 RFT 不优于基线。
- 放置：`finetune-sft`、`agent-trajectory-learning`、`agent-agent-training`。推理投机解码里的 rejection sampling 已实现，但与这里的训练数据筛选不是同一个教学问题。

## 六阶段整体还欠缺的六个环节

### 4. 真实模型接口的运行时闭环（P1，机制分散、实际适配器未串通）

[真实 API 讲义](../llm_agent/m15_claude_api/README.md) 明确列出未实现流式输出、真实 usage 记账、缓存接入和自定义限流退避；在线路径也未验收。[适配器](../llm_agent/core/claude_llm.py) 使用一次性 `messages.create`。m22 的重试与 m29 的异步事件实验有价值，但并未自动赋予这个适配器相同能力。

补充流式文本与工具参数增量、完成后才执行工具、断流与未知结果、可重试错误与不可重试错误、重试预算、计费来源。离线先用协议夹具覆盖正常/截断/限流/断连，再保留用户主动开启的真实服务验证。不能把离线通过写成线上通过。

放置：`agent-runtime-reliability`、`agent-async-runtime`、`agent-full-loop`。

### 5. 训练过的语义检索与重排（P1，概念已有，缺语义实验）

[m16](../llm_agent/m16_rag/README.md) 的 dense 是字符 n-gram 随机投影，rerank 是词面规则；讲义明确指出同义改写题无法召回。[m43](../llm_agent/m43_vector_index/README.md) 用二维固定向量验证近似索引，并已介绍上下文向量、句向量、ANNOY/HNSW 的区别。

因此不是“缺向量检索”，而是缺学习语义表示与候选联合重排这一层：池化、query/document 表示、正负样本、难负样本，以及编码器误差与索引误差怎么分开。可复用 CLIP 对比损失的教学思路，但图文配对实验不能代替文本检索验收。

最小验收：分别设置同义改写、跨语言、否定和罕见编号任务；同时报告精确向量检索、ANN 和 rerank 的误差来源。小模型实验必须明确训练集范围，不宣称通用语义能力。

放置：`agent-rag`、`agent-vector-index`，与 `agent-evidence-chain` 衔接。

### 6. 多维并行与性能预算（P1，单项丰富、组合缺失）

[训练首页边界](../llm_train/README.md) 明确写了没有多维并行组合、通信与计算重叠、真实多进程；[通信讲义](../llm_train/m09_collectives/README.md) 只计带宽项，没有延迟与拓扑。现有 DDP/TP/PP/EP/CP、ZeRO、精度和重算实验不是缺项。

需要补的是如何把这些单项组合成一个配置：rank 网格、并行维度约束、每卡状态与激活预算、通信次数、消息启动延迟、计算/访存/通信瓶颈。Roofline/MFU 不能只作为缩写出现，应说明估算口径及模拟与实测的区别。

最小验收：固定模型和全局 batch，比较两种合法 DP×TP×PP 划分，并给出一组非法配置反例；能解释显存节省为什么未必带来吞吐提高。CPU 代价模型可以完成第一版，不必以多 GPU 集群为前提。

放置：`train-model-parallel`、`train-collectives-loop`，推理算子章节复用性能预算解释。

### 7. 推理服务的容量与尾延迟（P1，指标出现过，缺整体实验）

[m06](../llm_infer/m06_chunked_prefill/README.md) 已展示 TTFT/TBT 权衡；[m15](../llm_infer/m15_pd_disaggregation/README.md) 已解释 ITL、SLO 和 goodput；[m27](../llm_infer/m27_multi_replica_routing/README.md) 已有队列与路由代价模型。因此不能把这些名词列成“完全没讲”。

未形成实验的是随到达率变化的系统行为：p95/p99、饱和与排队、准入/背压、超时/取消后的资源回收、满足时延约束的成功吞吐。当前案例主要比较固定负载下的机制，尚不足以回答“最多接多少并发”。

最小验收：固定长短请求分布，扫描到达率；同时输出吞吐、首 token/逐 token 延迟分位数、拒绝率及资源占用；比较无限排队与有界准入，保留确定性 seed，并明确毫秒数来自代价模型。

放置：`infer-scheduler`、`infer-multi-replica-routing`、`infer-engine`。

### 8. 多模态学习入口（P2，内容和代码已有，导航与教学链条薄弱）

[模型首页](../llm_models/README.md) 已有 CLIP、Whisper、Qwen2-VL、Omni；对应目录有源码、讲义和训练/推理脚本。[结构浏览器数据](../web/src/data/modelArchitectures.js) 也介绍 Qwen2-VL 等模型。不是“没有多模态”。

但[阶段 2 目录](../web/src/data/chapterGroups.js) 没有清晰的图文/语音理解阅读入口，章节级题库也未系统检验这条链。读者容易直接从 Transformer 进入生成分支，而错过 patch/mel → 编码器 → 对齐或 projector/cross-attention → 语言输出。

优先把现有讲义接入主题内的三级入口或现有页小节：解释图文对比学习与条件生成的差别、视觉 token 成本、音频前处理与模型结构的分工。沿用原代码与图，避免再次实现一套。音频预处理未实现等边界继续保留。

### 9. 基础概念入口与长上下文边界（P2，解释位置和精确口径问题）

刚问到的感受野并非完全漏掉：[滑窗实验台](../web/src/components/labs/ModelSwaLayersLab.vue) 已能展示信息跨层传播；[模型正文](../web/src/data/topics/models.js) 和 Mistral 讲义也提到了它。问题是缺一个可独立查找的基础定义，并容易把“连接得到”“可以输入多少 token”“真的能利用多少信息”混在一起。

另有口径差异：[Mistral.receptive_field](../llm_models/models/language_models/mistral.py) 返回标注为近似的 `L*W`，网页实验台使用精确的 `1+L*(W-1)`。窗口 W 包含当前位置、无 dilation/额外全局通路时，位置 t 能覆盖的输入数是 `min(t+1, 1+L*(W-1))`。例如 L=4、W=8，精确为 29，不是 32；大窗口时二者才接近。这不是新的注意力算法，而是应统一解释的教学口径。

应在术语表提供定义，在现有滑窗页说明近似条件，再把理论可达性与长上下文任务实测分开。已有压缩实验和摘要实验不能自动证明模型具备同等长度的检索/多跳能力。

放置：`basic-forward`、`models-mtp`、`models-gptoss`、`agent-context-engineering`。

## 已有内容，不重复算成缺项

- 左 padding、全遮罩 NaN、EOS 与批量结束：已有 [masks.py](../llm_models/utils/masks.py) 和 [generation.py](../llm_models/utils/generation.py)。
- 配对 bootstrap 与首错定位：已有 [m13 diagnostics](../llm_agent/m13_evals/diagnostics.py)；样本小的限制也已注明。
- ANN 机制：已有 m43；缺的是完整工程索引和真实语义表征，不是“没有近似搜索”。
- 真浏览器截图点击：已有 [m35 browser.py](../llm_agent/m35_visual_grounding/browser.py)；仍不是通用视觉模型。
- Skill 三层资料与进化：已有 [m33](../llm_agent/m33_skill_evolution/README.md)。
- SFT、在线策略更新、中期训练、自蒸馏：已有 m32/m37/m44，不能因玩具规模而否认其实际参数更新。

## 文档本身的遗漏

[Agent 首页](../llm_agent/README.md) 的旧“业界覆盖度自评”未同步后续扩展，原先写无 ANN、只有模拟浏览器、无置信区间、Skill 只有两层，分别与 m43、m35、m13 diagnostics、m33 不一致。本轮已修正这四行，并保留实际实现边界。这会同时造成两类误判：把已实现内容误报为缺项，或因另一张表写“补齐”而掩盖局部短板。后续应继续按模块及覆盖层次维护这些描述。

MCP 的 HTTP/授权、A2A 的网络传输、OS 沙箱、真实多 GPU、生产监控平台属于可选深化项。可以补概念与边界，但不应把尚未复现整个产品当成基础课程必须无限扩张的理由。

## 补齐方式与验收

先补 1–3 的原书语义缺口，再处理 4–7 的跨模块教学链条，最后整理 8–9 的入口与术语。继续保留六阶段和现有二级主题；能在现有三级页补一节的，不额外增加目录。

每项关闭需要记录：能解释的问题、正文入口、机制/失败对照、验收证据、适用边界。仅新增术语、链接或一个能通过的固定案例不能关闭缺口。本轮交付是复查记录及覆盖说明纠正，以上九项教学补充尚未实施。
