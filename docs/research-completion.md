# 研究主题与实验指南

25 组 CPU 实验、76 个执行条件。研究主题索引按 45 个主题列出检查维度、实验入口和验证边界。

每组实验提供原理、对照条件、自测与源码。网页展示 Python 实际执行记录；切换条件不调用模型 API，也不在浏览器训练模型。

## 实验入口

| 主题 | 运行模块 | 条件数 |
| --- | --- | --- |
| [数据课程：相同样本数，学到的一样吗](../llm_train/m23_data_curriculum/README.md) | `llm_train.m23_data_curriculum.demo` | 3 |
| [合成数据：多代反馈与长尾消失](../llm_train/m24_synthetic_feedback/README.md) | `llm_train.m24_synthetic_feedback.demo` | 3 |
| [可靠性：校准、拒答与人工评价](../llm_train/m25_reliability_metrics/README.md) | `llm_train.m25_reliability_metrics.demo` | 3 |
| [多语言：采样、分词与翻译偏差](../llm_train/m26_multilingual/README.md) | `llm_train.m26_multilingual.demo` | 3 |
| [IPO：偏好间隔为何有有限目标](../llm_finetune/run_finetune/ipo/README.md) | `llm_finetune.run_finetune.ipo.demo` | 2 |
| [PEFT：Prompt、Prefix、Adapter 与 LoRA](../llm_finetune/run_finetune/peft_comparison/README.md) | `llm_finetune.run_finetune.peft_comparison.demo` | 4 |
| [在线偏好：更新后还符合谁的偏好](../llm_finetune/run_finetune/online_preference/README.md) | `llm_finetune.run_finetune.online_preference.demo` | 3 |
| [知识编辑：改对一个事实，伤到多少邻居](../llm_finetune/run_finetune/knowledge_edit/README.md) | `llm_finetune.run_finetune.knowledge_edit.demo` | 3 |
| [隐私：记忆提取、DP 与参数遗忘](../llm_finetune/run_finetune/privacy_unlearning/README.md) | `llm_finetune.run_finetune.privacy_unlearning.demo` | 3 |
| [推理图：链、树、合并与逐步核验](../llm_infer/m29_reasoning_graph/README.md) | `llm_infer.m29_reasoning_graph.demo` | 3 |
| [剪枝与融合：零值不等于少做计算](../llm_infer/m30_pruning_fusion/README.md) | `llm_infer.m30_pruning_fusion.demo` | 3 |
| [端侧预算：能放下、能跑快、能持续吗](../llm_infer/m31_edge_budget/README.md) | `llm_infer.m31_edge_budget.demo` | 3 |
| [RAG：答案误差如何反传到检索器](../llm_agent/m46_joint_rag/README.md) | `llm_agent.m46_joint_rag.demo` | 4 |
| [有效上下文：看得到，不等于用得上](../llm_agent/m47_context_utilization/README.md) | `llm_agent.m47_context_utilization.demo` | 3 |
| [安全与治理：误拒、漏拦和责任记录](../llm_agent/m48_safety_governance/README.md) | `llm_agent.m48_safety_governance.demo` | 3 |
| [学习世界模型：预测和反馈的差别](../llm_agent/m49_learned_world/README.md) | `llm_agent.m49_learned_world.demo` | 3 |
| [代码任务：跨文件修改与独立验收](../llm_agent/m50_repository_verification/README.md) | `llm_agent.m50_repository_verification.demo` | 3 |
| [形式证明与科学假设：三种正确](../llm_agent/m51_formal_science/README.md) | `llm_agent.m51_formal_science.demo` | 3 |
| [专业领域：证据时效、地域与复核](../llm_agent/m52_domain_workflows/README.md) | `llm_agent.m52_domain_workflows.demo` | 3 |
| [机制解释：可读出，不等于起作用](../llm_models/run_models/foundation/interpretability/README.md) | `llm_models.run_models.foundation.interpretability.demo` | 3 |
| [视觉任务：OCR、图表与不确定性](../llm_models/run_models/multimodal/vision_tasks/README.md) | `llm_models.run_models.multimodal.vision_tasks.demo` | 3 |
| [语音信号：波形、Mel 与 codec](../llm_models/run_models/multimodal/audio_pipeline/README.md) | `llm_models.run_models.multimodal.audio_pipeline.demo` | 3 |
| [媒体生成：一致性、水印与来源](../llm_models/run_models/generative/media_evaluation/README.md) | `llm_models.run_models.generative.media_evaluation.demo` | 3 |
| [递归记忆与分层读取：压缩丢了什么](../llm_models/run_models/foundation/hierarchical_memory/README.md) | `llm_models.run_models.foundation.hierarchical_memory.demo` | 3 |
| [工具学习：从候选调用到训练数据](../llm_agent/m53_tool_data/README.md) | `llm_agent.m53_tool_data.demo` | 3 |

## 运行与校验

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python web/scripts/export-research-completion.py --check
python scripts/check-research-coverage.py
```

实验清单见 [research-completion-manifest.json](research-completion-manifest.json)。
主题索引见 [research-completion-coverage.json](research-completion-coverage.json)，其中 `scope` 是需要检查的知识维度或验证边界，不代表所有真实系统能力都已实现。

## 实验边界

- **数据课程**：二维回归机制实验。验证分布若错误，主动选择也会偏；没有证明课程在 LLM 上普遍更好。
- **合成数据**：有限类别的多项式估计器，不是 LLM。只说明有限采样和选择偏差如何丢失长尾，不把所有合成数据都叫 model collapse。
- **可靠性**：确定性评测夹具，不是模型能力榜单。拒答提高已答准确率会减少覆盖；Kappa 受标签流行率影响。
- **多语言**：两种受控语言模板，不是低资源语言 benchmark。分词器扩词还需处理新增 embedding、旧分词兼容和训练成本。
- **IPO**：四个独立的条件 logits，验证 loss 的梯度与平衡点，不代表真实偏好泛化，也不说明 IPO 总比 DPO 好。
- **PEFT**：固定 32 条合成样本验证参数路径与梯度。没有留出泛化结论；Prompt Tuning 与提示词写作不是一回事。
- **在线偏好**：有限动作上的实际 DPO 梯度更新，评审器是公开规则，不是人工偏好或真实 LLM Judge。奖励规则与部署验收应独立审计。
- **知识编辑**：三维线性关联记忆，说明低秩编辑约束，不声称复现 ROME/MEMIT 或 LLM 的事实存储。别名向量是给定的。
- **隐私**：全部是合成记录。公开固定种子仅用于复现演示，不能用于保护真实私密数据；该精确遗忘方法仅适用于此凸闭式模型，不适用于一般 LLM。
- **推理图**：有限整数状态搜索，不是完整 ToT/GoT 论文复现。检验文本步骤只检验外部一致性，内部因果忠实性需激活干预等独立证据。
- **剪枝与融合**：numpy 的通道裁剪与理论流量。没有稀疏 GPU kernel 或融合 kernel 的实测加速；模型压缩还需看长尾、推理、安全切片。
- **端侧预算**：预算模型，不是手机或芯片跑分，不提供特定设备续航承诺。真实能耗需扣除/报告空闲基线并说明测量边界。
- **RAG**：三篇文档、一个答案的可微机制实验；不是通用 RAG 模型或端到端自然语言生成。固定 generator 的对照用来识别答案记忆捷径。
- **有效上下文**：这是带公开位置偏好的 softmax 阅读器，不是实测 LLM 的 Lost in the Middle 曲线。不能把本例阈值或窗口长度外推成任何模型的有效上下文。
- **安全与治理**：公开合成夹具与动作门，没有实际攻击或真实安全模型。字符规则不能解决通用提示注入；字段齐全也不等于获得法律授权或通过合规认证。
- **学习世界模型**：一维线性、全状态可观测、无延迟环境。学习到的是此分布下的转移近似，不能推断掌握真实物理因果或安全机器人控制。
- **代码任务**：两个模块的受控 API 迁移，候选补丁由程序给定，不是通用 Coding Agent 或安全沙箱；-I 只隔离 Python 环境，不隔离操作系统权限。
- **形式证明与科学假设**：有限命题逻辑内核与合成测量，不是通用定理证明器或科学发现系统。证明只相对于假设成立，现实假设仍需实验检验。
- **专业领域**：所有地区、日期和规则均为虚构夹具，不提供医疗、法律或投资建议。未实现真实专家系统；这里验证资料和工作流边界。
- **机制解释**：线性读出与合成激活。SAE 特征没有自动获得语义名称；注意力图、相关探针或可读 CoT 都不能单独证明模型机制。
- **视觉任务**：已知字体和合成柱图的像素算法，不是通用 VLM 或真实 OCR；距离差只是拒答规则，不是校准后的正确概率。
- **语音信号**：真实数值信号处理，但输入是合成音调，不是语音。没有声纹识别、ASR/TTS 或神经 codec 能力；音频输出必须和说话人授权一起管理。
- **媒体生成**：合成短片与公开演示密钥，不是安全水印、C2PA 实现或通用深伪检测器；签名验证的是完整性与密钥持有者，不证明内容真实。
- **递归记忆与分层读取**：可控数组读写对照，不是 Transformer-XL 或训练过的分层注意力。读取单元是本例算法账本，不是端到端 FLOPs 或 GPU 延迟。
- **工具学习**：Toolformer 数据构造机制的最小对照。工具输出按观察而非用户指令插入；真实场景还要隔离执行、去重并计入延迟与权限成本。

## 维护

修改实验源码或清单后，运行导出器更新回放，再用 `--check` 核对。源码摘要覆盖教学包；执行记录中的参数、指标与模块 README 应一致。浏览器交互与页面布局的检查命令见 [验证指南](validation.md)。
