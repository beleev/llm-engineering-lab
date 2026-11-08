# 教程内容重叠与关键缺口审查

检查日期：2026-11-08。基线：`04fb92f`。

后续状态：本报告保留发现时的原始内容；本轮修订与逐项验收见 [工程补全记录](engineering-completion.md)。下文“尚缺”“建议”均描述该基线。

当前教程主题覆盖已经很广，但仍有重复入口、章节职责混杂，以及几段没有串起来的工程流程。上一轮“原图知识点已有教学覆盖”的结论，只适用于那份清单和相应实验深度，不能扩展成“整个 LLM 工程知识已经完全补齐”。

本次建议先整理已有内容，再补交付、检索生命周期、协议版本、异步训练和发布验证。六阶段、26 个二级分组可以保留，不需要为了这些缺口继续扩张二级目录。优先级是本次教学审查判断，不是对生产事故风险的评级。

## 检查范围与方法

- 全量读取 100 个数据驱动页面的导出内容，检查章节挂载、两份补全实验清单，以及 111 个题目路由下的 384 道题。概览和手写页面不纳入正文相似度分母，另检查相关组件入口。
- 对正文 subtitle/tldr/question/points[].body 中去空白后超过 45 字的块做完全相同比较；三字片段相似度只用来挑选人工复核对象，不把相似度当成“重复率”。
- 检索模块 README 和源码后，精读重叠候选、疑似缺口及相反证据。没有把 README 提到一个术语视为完整实现，也没有把“与真实系统的差距”全部算成必须补的新课。
- 搜索并阅读官方协议、框架文档和课程资料。外部资料说明工程中存在这些问题；“本仓库覆盖是否充分”是结合本地证据做出的审查判断。
- 本次是静态内容审查，未重跑全套实验、浏览器排版检查或真实硬件性能测试。之前的测试通过不等于这些教学缺口不存在。

## 1. 已确认的重复

### 六组完整回放重复挂载

同一份数据与同一个实验共挂在 15 处，相对每组保留一个完整入口，多出 9 个完整回放入口。这是阅读内容重复，代码本身是集中复用的。建议主讲页保留完整实验，其他页改成一句阶段相关结论、当前问题和跳转；综合实验可以保留集成结果。

| 实验 | 当前入口 | 建议完整实验主入口 |
| --- | --- | --- |
| RFT：当前策略采样 → 验证筛选 → SFT | `finetune-sft`、`agent-trajectory-learning`、`agent-agent-training` | `agent-trajectory-learning` |
| 流式消息：什么时候可以执行工具 | `agent-runtime-reliability`、`agent-async-runtime`、`agent-full-loop` | `agent-async-runtime` |
| 一次任务里的并行、重试与取消 | `agent-eval-suite`、`agent-evals` | `agent-eval-suite` |
| 语义表示、索引与重排分别错在哪里 | `agent-rag`、`agent-vector-index` | `agent-vector-index` |
| DP×TP×PP：显存与时间一起算 | `train-model-parallel`、`train-collectives-loop` | `train-model-parallel` |
| 到达率上升后，排队会发生什么 | `infer-scheduler`、`infer-multi-replica-routing`、`infer-engine` | `infer-multi-replica-routing` |

依据：[统一回放数据及 routes](../web/src/data/knowledge-completion.json)。这六组不包含下面的基础组件复用。

### 正文与基础组件

- 在上述扫描字段中，发现一段逐字重复的正文：“多维配置要一起算”，分别追加在 `train-model-parallel` 与 `train-collectives-loop`。见 [train.js](../web/src/data/topics/train.js) 第 930、937 行。保留在模型并行预算章，collectives 章改讲通信语义与训练循环中的位置。这个结果不代表所有 README 或手写页面只有这一处重复。
- [SoftmaxTempLab](../web/src/components/labs/SoftmaxTempLab.vue) 在基础采样、RLHF、推理解码三处使用相同内容；[AttnMaskLab](../web/src/components/labs/AttnMaskLab.vue) 在 [Attention 手写页](../web/src/views/Attention.vue)、MTP、KV memory 三处出现。两个组件都没有按章节传入教学参数，默认任务相同。可保留基础页完整版本，后续页改成应用问题或折叠复习；并非所有跨章复用都应该删除。
- 384 道题中未发现完全相同的题干；没有据此断言“考点绝不重复”。

## 2. 应保留的递进讲解

| 内容 | 分布 | 整理方式 |
| --- | --- | --- |
| KV cache | models-generation → infer-kv-memory / infer-kv-footprint → agent-prompt-caching | 模型状态、分页布局、架构容量和 API 前缀成本分别讲；统一术语和前置链接。 |
| MoE | 模型路由 → train-moe-seq → infer-moe-serving | 路由与负载均衡损失、训练通信、在线服务调度是不同目标。 |
| 低精度 | train-precision-stability / train-low-precision → finetune-qlora → 推理量化 | 区分训练数值稳定性、冻结量化基座上的梯度路径、推理存储与算子支持。 |
| 记忆 | agent-context-memory / agent-context-engineering / agent-memory-lifecycle / agent-memory-privacy | 分别回答存什么、上下文怎样压缩、版本冲突怎样处理、谁能读和怎样删除。 |
| 评测 | train-eval → agent-evals → agent-eval-suite | 基础模型指标、任务终态与轨迹、版本化套件各有职责；重复 trace 卡可合并。 |
| 流水线并行 | train-model-parallel → train-pipeline-schedules | 入口介绍 PP 与预算；后者保留 1F1B、交错与气泡推演，减少重复铺垫。 |

这些章节的目标和失败条件不同。统一概念定义、标注前置章节，比合成一篇巨长正文更合适。

## 3. 后续补充造成的章节归属问题

依据：[研究补全挂载清单](../web/src/data/research-completion.json)。下面是建议调整，尚未移动页面或改变导航。

- **knowledge_edit → finetune-sft**：模型知识编辑不同于常规 SFT；在现有微调分组内标明独立专题及前置条件，SFT 正文只留区别与入口。
- **privacy_unlearning → agent-memory-privacy**：应用记录删除与模型参数遗忘分层展示；该页以同意/过期/删除为主，参数遗忘链接到微调专题。
- **tool_data → agent-tools-permissions**：Toolformer 风格训练数据筛选的主讲解移向 agent-agent-training；权限页保留运行时校验。
- **formal_science → agent-programmatic-actions**：形式证明与科学验证不等同于行动适配器；放进同组独立三级专题或明确子节，并补前置知识链接。
- **safety_governance → agent-guardrails**：执行权限、模型有害输出、群体评测、发布治理分清各自证据，不能用一个 allowlist 实验代表四者。

## 4. 关键缺口与最小验收

### G1 · P0 · 训练产物到推理加载的完整交付约定

建议归属：`finetune-runs`、`infer-engine`。

**已有内容**：已经有训练断点恢复、LoRA 状态提取与合并前后 logits 一致性、chat template 等独立内容。

**尚缺什么**：尚未找到把权重、架构配置、tokenizer、特殊 token、聊天模板、生成参数、adapter 基座版本及量化元数据一起打包，再交给独立推理进程验证的教学链路。能继续训练与能正确交付推理是两个验收目标。

本地依据：[llm_finetune/run_finetune/lora/train_lora.py](../llm_finetune/run_finetune/lora/train_lora.py)、[llm_train/m08_checkpoint_resume/README.md](../llm_train/m08_checkpoint_resume/README.md)、[web/src/data/topics/finetune.js](../web/src/data/topics/finetune.js)。

**最小验收**：

1. 用已有小模型导出版本化 manifest，记录文件哈希、基座和模板标识。
2. 新进程加载，固定输入逐层核对 token IDs、logits 容差、greedy 输出；量化产物单独定义误差容限。
3. 故意替换 tokenizer、模板或 adapter 基座，必须明确拒绝或指出不匹配，不能静默给出不同答案。

外部核对：[Transformers Models：模型配置与保存加载](https://huggingface.co/docs/transformers/main_classes/model)；[Transformers GenerationConfig：生成行为也需要配置](https://huggingface.co/docs/transformers/main_classes/text_generation)。

### G2 · P0 · 原始文档到可更新、可授权、可追溯的检索库

建议归属：`agent-rag`、`agent-vector-index`、`agent-memory-privacy`。

**已有内容**：已有 chunk/retrieval、ANN、学习语义表示、重排；个人记忆已经在检索前按 tenant/user/topic 限定范围，并实现原文、索引、缓存删除。

**尚缺什么**：这些能力尚未串成原始 HTML/PDF/表格解析、版面/OCR 边界、来源页码或偏移、父子 chunk、embedding 版本与索引快照的完整流程。还缺权限过滤接入 ANN、文档更新或撤权后索引和答案缓存失效、引用定位回原文的联合反例。

本地依据：[llm_agent/m16_rag/README.md](../llm_agent/m16_rag/README.md)、[llm_agent/m43_vector_index/README.md](../llm_agent/m43_vector_index/README.md)、[llm_agent/m45_semantic_retrieval/demo.py](../llm_agent/m45_semantic_retrieval/demo.py)、[llm_agent/m42_memory_privacy/demo.py](../llm_agent/m42_memory_privacy/demo.py)。

**最小验收**：

1. 固定带表格的文档，保留 source_id/version/page 或 offset/parent_id；解析失败不得伪造空成功。
2. 两租户分别检索；对比 ANN 前过滤与后过滤的召回影响，任何返回和引用都不得泄漏无权文档。
3. 修改、删除或撤权后重放相同问题，旧 chunk、旧引用和缓存答案均不可继续生效；embedding 版本不兼容时显式重建或拒绝混用。

外部核对：[Unstructured：原始文档到结构化元素](https://docs.unstructured.io/open-source/core-functionality/partitioning)；[Azure AI Search：向量过滤位置会影响召回](https://learn.microsoft.com/en-us/azure/search/vector-search-filters)。

### G3 · P0 版本说明；P1 网络实验 · MCP 的协议版本边界与 HTTP 授权

建议归属：`agent-mcp`、`agent-tools-permissions`。

**已有内容**：当前客户端、服务端和网页示例固定使用 2025-06-18 的 initialize 握手；README 已说明 tools-only、stdio，以及没有 HTTP/OAuth 等边界。旧版本教学本身可以保留。

**尚缺什么**：截至本次检查，官方 2026-07-28 修订已移除 initialize/initialized 握手及 HTTP 协议会话，改用请求元数据和 server/discover。现有叙述需要显式标注历史协议配置，并给出版本对照；不能只修改日期字符串。HTTP 授权是另一个缺少实验的层次。

本地依据：[llm_agent/core/mcp.py](../llm_agent/core/mcp.py)、[llm_agent/m09_mcp/server.py](../llm_agent/m09_mcp/server.py)、[llm_agent/m09_mcp/README.md](../llm_agent/m09_mcp/README.md)、[web/src/components/labs/AgentMcpLab.vue](../web/src/components/labs/AgentMcpLab.vue)、[web/src/data/topics/agent.js](../web/src/data/topics/agent.js)。

**最小验收**：

1. 在正文、实验和 README 同步标明 2025-06-18 教学配置，说明与 2026-07-28 的差异及不支持范围。
2. 增加两个明确版本的报文约定测试，未知版本不得静默按旧报文解析。
3. 可选本地 HTTP 端点覆盖错误 audience、过期凭据、权限不足、取消与超时；明确 OAuth 授权是可选能力，不能把 HTTP 授权流程强加给 stdio。

外部核对：[MCP 2026-07-28 变更说明](https://modelcontextprotocol.io/specification/2026-07-28/changelog)；[MCP HTTP Authorization 的适用范围](https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization)。

### G4 · P1 · 异步 RL 的 rollout 与训练器之间的数据约定

建议归属：`finetune-grpo-variants`、`agent-agent-training`。

**已有内容**：GRPO 已有采样时 old_logp、温度一致性、重要性比率和裁剪，也明确说明 DAPO 的部分边界；在线偏好学习也有实验。

**尚缺什么**：尚缺 rollout 与更新并行时的 policy_version、权重同步、队列背压、样本过期、取消回收，以及生成引擎与训练引擎概率不一致的联合处理。不能把这里表述成“没有重要性采样”。

本地依据：[llm_finetune/methods/grpo.py](../llm_finetune/methods/grpo.py)、[llm_finetune/run_finetune/online_preference/demo.py](../llm_finetune/run_finetune/online_preference/demo.py)、[web/src/data/topics/finetune.js](../web/src/data/topics/finetune.js)。

**最小验收**：

1. 用两个策略版本和队列构造固定延迟，记录行为策略版本、采样概率和消费版本。
2. 比较无校正、重要性校正、过期拒收，报告偏差、有效样本量及吞吐代价；说明裁剪并不保证无偏。
3. 队列满、任务取消、权重切换均不混淆样本归属。CPU 模拟足以验证机制，真实多 GPU 接入单列选做。

外部核对：[TRL Asynchronous GRPO（官方列为 Experimental）](https://huggingface.co/docs/trl/async_grpo_trainer)。

### G5 · P1 · 评测、发布门槛、线上质量与回滚串成一次流程

建议归属：`agent-eval-suite`、`agent-system-design`。

**已有内容**：已有配对 bootstrap、trace/span、评测集、延迟和 goodput、版本管理，以及治理 README 中的小流量验收和回滚概念。

**尚缺什么**：尚缺一次完整的 model/prompt/index/tool-config 变更：离线门槛通过后进行影子或小流量验证，再由质量、安全、成本和延迟共同触发回滚，并把失败样本带回隔离的回归集。

本地依据：[llm_agent/m13_evals/diagnostics.py](../llm_agent/m13_evals/diagnostics.py)、[llm_agent/m31_eval_suite/trace_demo.py](../llm_agent/m31_eval_suite/trace_demo.py)、[llm_agent/m48_safety_governance/README.md](../llm_agent/m48_safety_governance/README.md)、[llm_infer/m28_service_capacity/demo.py](../llm_infer/m28_service_capacity/demo.py)。

**最小验收**：

1. 同一组固定请求对比旧版/候选版；让候选版平均分提升但某个重要切片下降，发布门槛应识别。
2. 模拟线上延迟或质量漂移并触发回滚，能够追溯当时的模型、提示、索引与工具版本。
3. 反馈延迟与缺失单独计数；回流样本去重并保持训练/验证/最终测试隔离。复用已有统计和 trace，避免再建一套评测基础设施。

外部核对：[MLflow：Regression Testing and CI/CD](https://mlflow.org/docs/latest/genai/eval-monitor/regression-testing/)；[MLflow：从开发评测到生产监控](https://mlflow.org/docs/latest/genai/eval-monitor/)。

### G6 · P2 可选进阶 · 从代价模型到真实 profiler 与基准测量

建议归属：`train-model-parallel`、`infer-compute`、`infer-engine`。

**已有内容**：已有时间计数、通信/显存/Roofline 模型，并清楚区分模拟读数与硬件实测。

**尚缺什么**：未找到 torch.profiler 或系统化 benchmark 教学，缺少把预测瓶颈与算子 trace、host 开销、峰值内存相互验证的入口。这是工程深度选做，不应使零 GPU 主线依赖 GPU。

本地依据：[llm_train/m22_parallel_budget/README.md](../llm_train/m22_parallel_budget/README.md)、[llm_infer/core/utils.py](../llm_infer/core/utils.py)、[llm_infer/m12_cuda_graph/graph.py](../llm_infer/m12_cuda_graph/graph.py)。

**最小验收**：

1. CPU profiler 示例记录 warmup、重复测量和环境，解释总时间与算子时间为何不同。
2. GPU 选做再讲同步、异步拷贝、峰值显存及 kernel trace，不拿 CPU 或模拟结果宣称 GPU 加速比。

外部核对：[PyTorch Profiler 官方教程](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)；[Stanford CS336 2025：Systems 作业包含 profiling 与 benchmark](https://cs336.stanford.edu/spring2025/)。

## 5. 已讲过但应标明深度的内容

- **安全与治理**：[m48](../llm_agent/m48_safety_governance/README.md) 的四个合成案例验证授权和误报；它不能验证真实模型对越狱、迎合、欺骗或群体偏差的普遍表现。数据卡字段齐全也不等于完成实际治理流程。后续若提升深度，应增加独立攻击家族留出、正常任务误拒对照和版本化回归，而不是再添加同义概念卡。
- **多模态与具身**：小字形 OCR、合成波形、简化世界模型和控制实验能说明机制；真实图片分布、流式 ASR/TTS 或真实 VLA 仍属于选做接入。原图中的主题不应因此重新记成“完全未覆盖”。
- **形式验证、编辑与遗忘**：小型证明检查、局部参数编辑和遗忘示例，分别有受限语言、编辑泛化、效用保持等边界。实验主张应限定在已验证的任务与保持集上，不能直接外推到通用证明、全局知识更新或共享模型的用户级删除保证。
- **算法边界**：[GRPO 实现](../llm_finetune/methods/grpo.py) 已写明未实现 DAPO overlong reward shaping；若继续追求论文实现深度，可补长度截断、动态筛选补样本等对照。这类属于已知局部差异，不值得新增一级或二级分类。

## 6. 复核后排除的“假缺口”

| 不应声称 | 已有证据 | 真正需要补的边界 |
| --- | --- | --- |
| 没有置信区间 | [m13 diagnostics.py](../llm_agent/m13_evals/diagnostics.py) 第 97 行 paired_interval | 已有配对 bootstrap；需要更有代表性的评测集，三条手写案例不能外推 |
| 没有污染检测 | [m21_llm_eval](../llm_train/m21_llm_eval/README.md) | 已有 n-gram 检查；不等于能识别所有语义污染 |
| GRPO 没有重要性校正 | [grpo.py](../llm_finetune/methods/grpo.py) 第 146、154 行 | 缺异步队列与策略版本约定 |
| 没有可恢复训练或 LoRA 一致性 | [m08](../llm_train/m08_checkpoint_resume/README.md)、[LoRA](../llm_finetune/run_finetune/lora/train_lora.py) | 缺完整部署产物的新进程交付验证 |
| 没有权限过滤与删除 | [m42 demo.py](../llm_agent/m42_memory_privacy/demo.py) 第 32、40 行 | 缺与 ANN、文档版本及引用的联合流程 |
| 没有观测、尾延迟或容量规划 | [trace_demo](../llm_agent/m31_eval_suite/trace_demo.py)、[m28 capacity](../llm_infer/m28_service_capacity/demo.py) | 缺评测到发布回滚的一次可执行流程 |

## 7. 建议处理顺序

1. 为六组回放各确定一个主讲章节，其他入口保留概念关联；整理两个基础组件复习入口，去掉重复并行预算段落。
2. 标明五类归属问题的层次和前置知识，保留现有六阶段和二级分组结构。
3. 优先修订 MCP 版本表述，补 G1 模型交付与 G2 文档生命周期，避免读者把相邻章节中的零件误当成已连通的流程。
4. 再补 G4 异步 RL 与 G5 发布验证；HTTP 实验和 G6 profiler 按主线/选做分开。
5. 每项完成以反例、对照、运行结果和边界说明验收；同步网页、README 和审查状态，不能仅凭新增名词关闭缺口。

本次产物为审查报告及 [结构化记录](tutorial-overlap-gap-audit.json)，其中保存基线、重复入口、优先级、验收项和证据文件 SHA-256。教程内容和页面行为尚未因本次审查而改变。
