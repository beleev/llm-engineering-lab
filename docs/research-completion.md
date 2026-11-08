# 研究方向长图：补充覆盖与验证边界

原始审计固定在提交 `23f1c93`，见 [45 项原始对照](image-knowledge-audit.md)。此文记录后续补充，不回写旧审计的源码哈希。

本轮增加 **25 组 CPU 实验、76 个实际执行条件、25 道自测**，放进既有三级章节。原图的 **11 个方向、45 个主题**均有去向。这里的“补齐”指教学知识、机制对照与评测方法；真实大模型、多机 GPU、真实机器人和专业领域上线验证仍明确列为边界。

## 一次运行

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/run-research-completion.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python web/scripts/export-research-completion.py --check
python scripts/check-research-coverage.py
```

网页回放只展示 Python 预运行证据；不是浏览器现场训练。讲解和实际运行条件由 [补充清单](research-completion-manifest.json) 对照。

## 新增实验入口

| 实验 | 运行模块 | 原图对应 |
| --- | --- | --- |
| [数据课程：相同样本数，学到的一样吗](../llm_train/m23_data_curriculum/README.md) | `llm_train.m23_data_curriculum.demo` | 1.1, 1.2, 1.4, 3.1 |
| [合成数据：多代反馈与长尾消失](../llm_train/m24_synthetic_feedback/README.md) | `llm_train.m24_synthetic_feedback.demo` | 1.3, 3.1 |
| [可靠性：校准、拒答与人工评价](../llm_train/m25_reliability_metrics/README.md) | `llm_train.m25_reliability_metrics.demo` | 3.4, 5.4, 9.1, 9.2, 10.4 |
| [多语言：采样、分词与翻译偏差](../llm_train/m26_multilingual/README.md) | `llm_train.m26_multilingual.demo` | 11.4, 10.4 |
| [IPO：偏好间隔为何有有限目标](../llm_finetune/run_finetune/ipo/README.md) | `llm_finetune.run_finetune.ipo.demo` | 3.2 |
| [PEFT：Prompt、Prefix、Adapter 与 LoRA](../llm_finetune/run_finetune/peft_comparison/README.md) | `llm_finetune.run_finetune.peft_comparison.demo` | 3.5 |
| [在线偏好：更新后还符合谁的偏好](../llm_finetune/run_finetune/online_preference/README.md) | `llm_finetune.run_finetune.online_preference.demo` | 3.2, 3.5, 10.1 |
| [知识编辑：改对一个事实，伤到多少邻居](../llm_finetune/run_finetune/knowledge_edit/README.md) | `llm_finetune.run_finetune.knowledge_edit.demo` | 5.4 |
| [隐私：记忆提取、DP 与参数遗忘](../llm_finetune/run_finetune/privacy_unlearning/README.md) | `llm_finetune.run_finetune.privacy_unlearning.demo` | 10.3 |
| [推理图：链、树、合并与逐步核验](../llm_infer/m29_reasoning_graph/README.md) | `llm_infer.m29_reasoning_graph.demo` | 4.1, 4.2, 4.3, 4.4 |
| [剪枝与融合：零值不等于少做计算](../llm_infer/m30_pruning_fusion/README.md) | `llm_infer.m30_pruning_fusion.demo` | 2.1, 8.2, 8.3 |
| [端侧预算：能放下、能跑快、能持续吗](../llm_infer/m31_edge_budget/README.md) | `llm_infer.m31_edge_budget.demo` | 8.4 |
| [RAG：答案误差如何反传到检索器](../llm_agent/m46_joint_rag/README.md) | `llm_agent.m46_joint_rag.demo` | 5.1 |
| [有效上下文：看得到，不等于用得上](../llm_agent/m47_context_utilization/README.md) | `llm_agent.m47_context_utilization.demo` | 2.3, 5.2 |
| [安全与治理：误拒、漏拦和责任记录](../llm_agent/m48_safety_governance/README.md) | `llm_agent.m48_safety_governance.demo` | 1.1, 9.1, 10.1, 10.2, 10.4 |
| [学习世界模型：预测和反馈的差别](../llm_agent/m49_learned_world/README.md) | `llm_agent.m49_learned_world.demo` | 6.3, 7.4 |
| [代码任务：跨文件修改与独立验收](../llm_agent/m50_repository_verification/README.md) | `llm_agent.m50_repository_verification.demo` | 11.1 |
| [形式证明与科学假设：三种正确](../llm_agent/m51_formal_science/README.md) | `llm_agent.m51_formal_science.demo` | 4.2, 4.4, 11.2 |
| [专业领域：证据时效、地域与复核](../llm_agent/m52_domain_workflows/README.md) | `llm_agent.m52_domain_workflows.demo` | 11.3 |
| [机制解释：可读出，不等于起作用](../llm_models/run_models/foundation/interpretability/README.md) | `llm_models.run_models.foundation.interpretability.demo` | 9.4 |
| [视觉任务：OCR、图表与不确定性](../llm_models/run_models/multimodal/vision_tasks/README.md) | `llm_models.run_models.multimodal.vision_tasks.demo` | 7.1 |
| [语音信号：波形、Mel 与 codec](../llm_models/run_models/multimodal/audio_pipeline/README.md) | `llm_models.run_models.multimodal.audio_pipeline.demo` | 7.2, 7.3 |
| [媒体生成：一致性、水印与来源](../llm_models/run_models/generative/media_evaluation/README.md) | `llm_models.run_models.generative.media_evaluation.demo` | 7.3, 10.3 |
| [递归记忆与分层读取：压缩丢了什么](../llm_models/run_models/foundation/hierarchical_memory/README.md) | `llm_models.run_models.foundation.hierarchical_memory.demo` | 2.3 |
| [工具学习：从候选调用到训练数据](../llm_agent/m53_tool_data/README.md) | `llm_agent.m53_tool_data.demo` | 6.1 |

## 逐项复查

“机制实验 + 方法讲解”不表示该主题的每个真实系统都已复现。每条旧差距保留原文，新增证据和超出本地验证的部分分别列出。

### 1.1 数据清洗与配比

原有基础：精确/近似去重、质量规则、来源温度采样；另有实际训练后拟合 scaling law。

本次处理：机制实验 + 方法讲解。

- 原审计：语言识别与预训练语料 PII 过滤
- 原审计：数据来源、授权和许可证记录
- 原审计：依据验证损失动态调整数据配比

- [llm_train/m23_data_curriculum/README.md](../llm_train/m23_data_curriculum/README.md)
- [llm_agent/m48_safety_governance/README.md](../llm_agent/m48_safety_governance/README.md)

验收方法与边界：固定模型、随机种子与训练样本预算，比较静态、课程、主动策略；再把验证分布改偏，检查主动选择是否过拟合验证域。额外候选梯度计算必须计入总成本。；四个合成案例只验证具体授权与误报逻辑。生产应把策略版本、决策日志、人工复核与撤销测试纳入验收；这不构成通用防越狱保证或法律合规判定。

新增实验的实际边界：
- 二维回归机制实验。验证分布若错误，主动选择也会偏；没有证明课程在 LLM 上普遍更好。
- 公开合成夹具与动作门，没有实际攻击或真实安全模型。字符规则不能解决通用提示注入；字段齐全也不等于获得法律授权或通过合规认证。

### 1.2 数据选择与课程学习

原有基础：m37 提到长度课程；m17 提到 DoReMi/RegMix，已有静态配比。

本次处理：机制实验 + 方法讲解。

- 原审计：按难度/长度改变训练顺序的对照实验
- 原审计：主动样本选择、数据影响与预算收益

- [llm_train/m23_data_curriculum/README.md](../llm_train/m23_data_curriculum/README.md)

验收方法与边界：固定模型、随机种子与训练样本预算，比较静态、课程、主动策略；再把验证分布改偏，检查主动选择是否过拟合验证域。额外候选梯度计算必须计入总成本。

新增实验的实际边界：
- 二维回归机制实验。验证分布若错误，主动选择也会偏；没有证明课程在 LLM 上普遍更好。

### 1.3 合成数据

原有基础：轨迹验收后生成 SFT/偏好/过程标签；RLAIF 与 RFT 展示自动反馈和成功样本筛选。

本次处理：机制实验 + 方法讲解。

- 原审计：多代自生成数据的多样性、长尾损失与 model collapse
- 原审计：真实/合成混合比例、去重和难度控制的联合实验

- [llm_train/m24_synthetic_feedback/README.md](../llm_train/m24_synthetic_feedback/README.md)

验收方法与边界：先比较三种实际采样路径，再改变 rounds 和种子复查尾部。要研究真实 LLM 的 model collapse，必须训练多代模型并固定总预算；本例只隔离有限采样和选择偏差。

新增实验的实际边界：
- 有限类别的多项式估计器，不是 LLM。只说明有限采样和选择偏差如何丢失长尾，不把所有合成数据都叫 model collapse。

### 1.4 Continual Pre-training

原有基础：m37 实际训练小 LLaMA：继续 next-token、新域适应、旧域遗忘与 replay 对照。

本次处理：机制实验 + 方法讲解。

- 原审计：真实领域语料、未见知识泛化、长序列课程仍未验证；属于深度边界，不再列成空白

- [llm_train/m23_data_curriculum/README.md](../llm_train/m23_data_curriculum/README.md)

验收方法与边界：固定模型、随机种子与训练样本预算，比较静态、课程、主动策略；再把验证分布改偏，检查主动选择是否过拟合验证域。额外候选梯度计算必须计入总成本。

新增实验的实际边界：
- 二维回归机制实验。验证分布若错误，主动选择也会偏；没有证明课程在 LLM 上普遍更好。

### 2.1 Transformer 结构优化

原有基础：MHA/MQA/GQA/MLA、FlashAttention、RoPE/YaRN、norm/激活、KV cache 均有教学实现。

本次处理：机制实验 + 方法讲解。

- 原审计：CPU 算法实验不等同 GPU kernel 性能复现

- [llm_infer/m30_pruning_fusion/README.md](../llm_infer/m30_pruning_fusion/README.md)

验收方法与边界：先核对数学输出，再计算理想访存次数。这里没有 GPU kernel 或真实加速比；实测需 warm-up、同步、固定 batch/上下文、设备功率与分位数。

新增实验的实际边界：
- numpy 的通道裁剪与理论流量。没有稀疏 GPU kernel 或融合 kernel 的实测加速；模型压缩还需看长尾、推理、安全切片。

### 2.2 Mixture of Experts

原有基础：稀疏路由、共享专家、负载均衡、专家并行与推理负载调度。

本次处理：沿用已有闭环 + 补充评测协议。

- 原审计：专家专业化的实证分析与真实跨卡测量未覆盖

- [llm_models/run_models/moe/deepseek/README.md](../llm_models/run_models/moe/deepseek/README.md)
- [llm_train/m11_expert_parallel/README.md](../llm_train/m11_expert_parallel/README.md)
- [llm_train/m11_expert_parallel/demo.py](../llm_train/m11_expert_parallel/demo.py)
- [llm_infer/m21_moe_serving/README.md](../llm_infer/m21_moe_serving/README.md)
- [llm_infer/m21_moe_serving/demo.py](../llm_infer/m21_moe_serving/demo.py)

验收方法与边界：专家专业化需统计各领域的路由分布、专家输出差异，再替换或消融专家测任务损失；路由热图本身不是专业化的因果证据。跨卡验收记录 dispatch/combine 字节、拥塞、最忙 rank 与端到端吞吐，不能从 numpy 等价性推断。

### 2.3 长序列架构

原有基础：滑窗、跨层感受野、RoPE 外推、稀疏 decode、SSM 与上下文压缩。

本次处理：机制实验 + 方法讲解。

- 原审计：递归记忆/分层注意力的专门对照
- 原审计：长输入可接受、理论可达、信息有效利用三者的端到端评测

- [llm_agent/m47_context_utilization/README.md](../llm_agent/m47_context_utilization/README.md)
- [llm_models/run_models/foundation/hierarchical_memory/README.md](../llm_models/run_models/foundation/hierarchical_memory/README.md)

验收方法与边界：18 个位置×长度×证据组合全部执行。这里是显式机制模型，不是某个 LLM 的 Lost in the Middle 实测；真实评测要控制输出预算和检索器变化。；核对 24 个查询的准确率，同时报告访问量和持久存储。块 ID 已知，因此这里不测语义检索；连接 agent-context-engineering 阅读有效上下文实验。

新增实验的实际边界：
- 这是带公开位置偏好的 softmax 阅读器，不是实测 LLM 的 Lost in the Middle 曲线。不能把本例阈值或窗口长度外推成任何模型的有效上下文。
- 可控数组读写对照，不是 Transformer-XL 或训练过的分层注意力。读取单元是本例算法账本，不是端到端 FLOPs 或 GPU 延迟。

### 2.4 SSM 与混合架构

原有基础：Mamba 选择性状态更新；Qwen3-Next 的 Gated DeltaNet/attention 混合。

本次处理：沿用已有闭环 + 补充评测协议。

- 原审计：合成训练验证结构，不是长文本 benchmark 对比

- [llm_models/run_models/language_models/mamba/README.md](../llm_models/run_models/language_models/mamba/README.md)
- [llm_models/models/language_models/mamba.py](../llm_models/models/language_models/mamba.py)
- [llm_models/run_models/language_models/qwen3_next/README.md](../llm_models/run_models/language_models/qwen3_next/README.md)

验收方法与边界：比较 SSM 与混合架构时固定训练数据和计算预算，分别测内容检索、顺序记忆和长度外推；状态固定只说明内存复杂度，不保证任意远细节可恢复。已有合成训练验证保留，真实长文本 benchmark 未执行。

### 3.1 Supervised Fine-Tuning

原有基础：prompt mask、chat template、多轮与工具格式、冷启动/轨迹 SFT、RFT 连接。

本次处理：机制实验 + 方法讲解。

- 原审计：真实 SFT 语料的质量、长短答案配比和任务分布消融仍可深化

- [llm_train/m23_data_curriculum/README.md](../llm_train/m23_data_curriculum/README.md)
- [llm_train/m24_synthetic_feedback/README.md](../llm_train/m24_synthetic_feedback/README.md)

验收方法与边界：固定模型、随机种子与训练样本预算，比较静态、课程、主动策略；再把验证分布改偏，检查主动选择是否过拟合验证域。额外候选梯度计算必须计入总成本。；先比较三种实际采样路径，再改变 rounds 和种子复查尾部。要研究真实 LLM 的 model collapse，必须训练多代模型并固定总预算；本例只隔离有限采样和选择偏差。

新增实验的实际边界：
- 二维回归机制实验。验证分布若错误，主动选择也会偏；没有证明课程在 LLM 上普遍更好。
- 有限类别的多项式估计器，不是 LLM。只说明有限采样和选择偏差如何丢失长尾，不把所有合成数据都叫 model collapse。

### 3.2 偏好对齐与 AI Feedback

原有基础：RM/PPO、DPO/KTO/SimPO/ORPO、RLAIF/Constitutional AI，含奖励捷径和裁判偏差。

本次处理：机制实验 + 方法讲解。

- 原审计：IPO 明确未实现
- 原审计：在线偏好采集→更新循环
- 原审计：迎合、长度偏好和持续/个性化对齐的专门实验

- [llm_finetune/run_finetune/ipo/README.md](../llm_finetune/run_finetune/ipo/README.md)
- [llm_finetune/run_finetune/online_preference/README.md](../llm_finetune/run_finetune/online_preference/README.md)

验收方法与边界：同一组四个单 token 偏好分别训练 DPO 与 IPO，比较 loss 和 log-ratio margin，而不是只检查训练集准确率。长文本上还应控制回答长度与参考模型概率。；固定随机种子和交互预算，比较偏好变化前后各用户指标。模型层迎合与有意欺骗不能只由一个错误答案判定，需要可核验的任务行为和多个条件。

新增实验的实际边界：
- 四个独立的条件 logits，验证 loss 的梯度与平衡点，不代表真实偏好泛化，也不说明 IPO 总比 DPO 好。
- 有限动作上的实际 DPO 梯度更新，评审器是公开规则，不是人工偏好或真实 LLM Judge。奖励规则与部署验收应独立审计。

### 3.3 面向推理的 RL

原有基础：RLVR、GRPO/DAPO/Dr.GRPO/GSPO、拒绝采样式 RFT，奖励依赖 prompt 并有常数基线。

本次处理：沿用已有闭环 + 补充评测协议。

- 原审计：短序列规则任务不代表数学长链推理或开放任务泛化

- [llm_finetune/run_finetune/grpo/README.md](../llm_finetune/run_finetune/grpo/README.md)
- [llm_finetune/methods/grpo.py](../llm_finetune/methods/grpo.py)
- [llm_finetune/run_finetune/rft/README.md](../llm_finetune/run_finetune/rft/README.md)
- [llm_finetune/data/tasks.py](../llm_finetune/data/tasks.py)

验收方法与边界：把奖励验证器与策略训练集分开；用未见模板、数值范围和推理长度测试，并统计格式错误与 reward hacking。现有短序列参数更新成立，不能推出数学长链或开放任务泛化。

### 3.4 On-policy 蒸馏与强到弱迁移

原有基础：学生轨迹上 teacher 逐 token 反馈、reverse KL；与离线 forward KL 对照；另有特权信息自蒸馏。

本次处理：机制实验 + 方法讲解。

- 原审计：已有实现偏 GKD 式停止采样梯度，不是完整 MiniLLM；校准收益未验证

- [llm_train/m25_reliability_metrics/README.md](../llm_train/m25_reliability_metrics/README.md)

验收方法与边界：实验固定同一组正确/错误标签，只改变置信度或拒答阈值，并实际拟合成对胜负得分。真实校准必须使用未参与阈值选择的测试集；不要从教学数组推断某个模型可靠。

新增实验的实际边界：
- 确定性评测夹具，不是模型能力榜单。拒答提高已答准确率会减少覆盖；Kappa 受标签流行率影响。

### 3.5 参数高效微调

原有基础：LoRA/QLoRA/DoRA、适配器保存与合并、Multi-LoRA serving。

本次处理：机制实验 + 方法讲解。

- 原审计：瓶颈 Adapter、Prompt Tuning、Prefix Tuning 仅部分提名，缺机制对照
- 原审计：适配器组合与持续个性化对齐

- [llm_finetune/run_finetune/peft_comparison/README.md](../llm_finetune/run_finetune/peft_comparison/README.md)
- [llm_finetune/run_finetune/online_preference/README.md](../llm_finetune/run_finetune/online_preference/README.md)

验收方法与边界：四种模式使用相同冻结基座和固定 32 条样本，报告可训练参数数与 MSE，断言基座不变。参数量没有配平，结果只验证可学习路径，不作为方法排名。；固定随机种子和交互预算，比较偏好变化前后各用户指标。模型层迎合与有意欺骗不能只由一个错误答案判定，需要可核验的任务行为和多个条件。

新增实验的实际边界：
- 固定 32 条合成样本验证参数路径与梯度。没有留出泛化结论；Prompt Tuning 与提示词写作不是一回事。
- 有限动作上的实际 DPO 梯度更新，评审器是公开规则，不是人工偏好或真实 LLM Judge。奖励规则与部署验收应独立审计。

### 4.1 CoT 与搜索式推理

原有基础：多步程序链、自一致性、PRM beam search；发现树展示搜索与回放。

本次处理：机制实验 + 方法讲解。

- 原审计：显式 CoT 的提示与训练区别
- 原审计：CoT 忠实性及解释不等于内部机制
- 原审计：ToT/GoT 的状态表示、分支评估与图合并对照

- [llm_infer/m29_reasoning_graph/README.md](../llm_infer/m29_reasoning_graph/README.md)

验收方法与边界：三个策略使用同一初态、目标和动作集合；统计展开数并拒绝伪造路径。迁移到 LLM 后还需审计步骤粒度、验证器相关错误与未见任务泛化。

新增实验的实际边界：
- 有限整数状态搜索，不是完整 ToT/GoT 论文复现。检验文本步骤只检验外部一致性，内部因果忠实性需激活干预等独立证据。

### 4.2 过程监督与验证器

原有基础：PRM/ORM 训练与对照、带噪验证器影响搜索、执行验收与过程标签。

本次处理：机制实验 + 方法讲解。

- 原审计：形式化证明器见 11.2；缺长推理 verifier overcredit 专门评测

- [llm_infer/m29_reasoning_graph/README.md](../llm_infer/m29_reasoning_graph/README.md)
- [llm_agent/m51_formal_science/README.md](../llm_agent/m51_formal_science/README.md)

验收方法与边界：三个策略使用同一初态、目标和动作集合；统计展开数并拒绝伪造路径。迁移到 LLM 后还需审计步骤粒度、验证器相关错误与未见任务泛化。；伪造证明必须被拒；拟合误差与留出误差分别显示。本例是命题逻辑和合成数值实验，不冒充数学竞赛解题器或真实科研发现。

新增实验的实际边界：
- 有限整数状态搜索，不是完整 ToT/GoT 论文复现。检验文本步骤只检验外部一致性，内部因果忠实性需激活干预等独立证据。
- 有限命题逻辑内核与合成测量，不是通用定理证明器或科学发现系统。证明只相对于假设成立，现实假设仍需实验检验。

### 4.3 Test-Time Scaling

原有基础：BoN、投票、PRM 搜索、budget forcing，比较预算、随机错与系统性误解。

本次处理：机制实验 + 方法讲解。

- 原审计：程序 sampler/verifier，不是实测大模型；判分器前向未计入 token 账

- [llm_infer/m29_reasoning_graph/README.md](../llm_infer/m29_reasoning_graph/README.md)

验收方法与边界：三个策略使用同一初态、目标和动作集合；统计展开数并拒绝伪造路径。迁移到 LLM 后还需审计步骤粒度、验证器相关错误与未见任务泛化。

新增实验的实际边界：
- 有限整数状态搜索，不是完整 ToT/GoT 论文复现。检验文本步骤只检验外部一致性，内部因果忠实性需激活干预等独立证据。

### 4.4 RL for Reasoning

原有基础：与 3.3 共用实际参数更新及可执行验证任务，不重复建设。

本次处理：机制实验 + 方法讲解。

- 原审计：形式证明与开放式奖励可靠性仍需深化

- [llm_infer/m29_reasoning_graph/README.md](../llm_infer/m29_reasoning_graph/README.md)
- [llm_agent/m51_formal_science/README.md](../llm_agent/m51_formal_science/README.md)

验收方法与边界：三个策略使用同一初态、目标和动作集合；统计展开数并拒绝伪造路径。迁移到 LLM 后还需审计步骤粒度、验证器相关错误与未见任务泛化。；伪造证明必须被拒；拟合误差与留出误差分别显示。本例是命题逻辑和合成数值实验，不冒充数学竞赛解题器或真实科研发现。

新增实验的实际边界：
- 有限整数状态搜索，不是完整 ToT/GoT 论文复现。检验文本步骤只检验外部一致性，内部因果忠实性需激活干预等独立证据。
- 有限命题逻辑内核与合成测量，不是通用定理证明器或科学发现系统。证明只相对于假设成立，现实假设仍需实验检验。

### 5.1 RAG

原有基础：切块、BM25/稠密/混合、重排、多跳、引用验收、图/树知识组织；m45 真训练检索与重排。

本次处理：机制实验 + 方法讲解。

- 原审计：通用查询改写与训练反馈
- 原审计：Retriever 与 Generator 联合训练；m45 联合输入重排不是检索—生成联合训练

- [llm_agent/m46_joint_rag/README.md](../llm_agent/m46_joint_rag/README.md)

验收方法与边界：先比较冻结与只更新检索，再看共同训练是否走向记忆捷径。检查不同模式实际更新的参数和 top-1，不以“joint”命名代替梯度证据。

新增实验的实际边界：
- 三篇文档、一个答案的可微机制实验；不是通用 RAG 模型或端到端自然语言生成。固定 generator 的对照用来识别答案记忆捷径。

### 5.2 Long Context

原有基础：窗口/感受野区别、YaRN、滑窗、稀疏 KV、压缩与预算。

本次处理：机制实验 + 方法讲解。

- 原审计：证据位置×长度×干扰量的有效利用评测
- 原审计：Lost in the Middle 与多证据组合；稀疏注意力的合成 needle 召回不能替代此项

- [llm_agent/m47_context_utilization/README.md](../llm_agent/m47_context_utilization/README.md)

验收方法与边界：18 个位置×长度×证据组合全部执行。这里是显式机制模型，不是某个 LLM 的 Lost in the Middle 实测；真实评测要控制输出预算和检索器变化。

新增实验的实际边界：
- 这是带公开位置偏好的 softmax 阅读器，不是实测 LLM 的 Lost in the Middle 曲线。不能把本例阈值或窗口长度外推成任何模型的有效上下文。

### 5.3 Long-term Memory

原有基础：来源、作用域、版本、冲突更新、回滚、过期、原文/索引/缓存删除。

本次处理：沿用已有闭环 + 补充评测协议。

- 原审计：以有限夹具验收，不代表通用自动记忆决策或生产存储系统

- [llm_agent/m20_memory_lifecycle/README.md](../llm_agent/m20_memory_lifecycle/README.md)
- [llm_agent/m20_memory_lifecycle/demo.py](../llm_agent/m20_memory_lifecycle/demo.py)
- [llm_agent/m42_memory_privacy/README.md](../llm_agent/m42_memory_privacy/README.md)
- [llm_agent/m42_memory_privacy/demo.py](../llm_agent/m42_memory_privacy/demo.py)

验收方法与边界：长期记忆按写入、读取、冲突更新、过期与删除逐步验收；使用带时间戳的事实变化任务核对原文、索引、缓存是否一致删除。已有夹具验证机制，生产存储还要测备份、并发、跨副本及恢复语义。

### 5.4 知识编辑与幻觉缓解

原有基础：引用支持、缺证据拒答、版本核对、独立任务验收。

本次处理：机制实验 + 方法讲解。

- 原审计：参数知识编辑及编辑成功/改写泛化/局部性/关联事实指标
- 原审计：模型回答置信校准与选择性回答

- [llm_train/m25_reliability_metrics/README.md](../llm_train/m25_reliability_metrics/README.md)
- [llm_finetune/run_finetune/knowledge_edit/README.md](../llm_finetune/run_finetune/knowledge_edit/README.md)

验收方法与边界：实验固定同一组正确/错误标签，只改变置信度或拒答阈值，并实际拟合成对胜负得分。真实校准必须使用未参与阈值选择的测试集；不要从教学数组推断某个模型可靠。；固定旧模型，对同一目标做三种更新，同时输出目标误差与保护邻域误差。真实编辑还要批量累积、撤销、关联问答和校准测试，避免只挑原问题验收。

新增实验的实际边界：
- 确定性评测夹具，不是模型能力榜单。拒答提高已答准确率会减少覆盖；Kappa 受标签流行率影响。
- 三维线性关联记忆，说明低秩编辑约束，不声称复现 ROME/MEMIT 或 LLM 的事实存储。别名向量是给定的。

### 6.1 工具学习与 Function Calling

原有基础：Schema、参数验证、权限、工具结果循环、工具发现；另有轨迹 SFT/RL。

本次处理：机制实验 + 方法讲解。

- 原审计：Toolformer 式自动插入调用数据的训练路线未专门复现

- [llm_agent/m53_tool_data/README.md](../llm_agent/m53_tool_data/README.md)

验收方法与边界：无过滤、收益过滤和高成本门槛分别验收调用数与正确数。真实 Toolformer 的候选生成、语言模型条件似然和再训练规模不在此小模型结果中。

新增实验的实际边界：
- Toolformer 数据构造机制的最小对照。工具输出按观察而非用户指令插入；真实场景还要隔离执行、去重并计入延迟与权限成本。

### 6.2 规划、反思与执行

原有基础：计划/行动循环、重试幂等、首错诊断、任务验收后继续修复。

本次处理：沿用已有闭环 + 补充评测协议。

- 原审计：脚本策略与小型任务环境，不宣称通用长期规划能力

- [llm_agent/m10_planning/README.md](../llm_agent/m10_planning/README.md)
- [llm_agent/m10_planning/demo.py](../llm_agent/m10_planning/demo.py)
- [llm_agent/m21_completion_verification/README.md](../llm_agent/m21_completion_verification/README.md)
- [llm_agent/m21_completion_verification/demo.py](../llm_agent/m21_completion_verification/demo.py)
- [llm_agent/m22_runtime_reliability/README.md](../llm_agent/m22_runtime_reliability/README.md)
- [llm_agent/m22_runtime_reliability/demo.py](../llm_agent/m22_runtime_reliability/demo.py)

验收方法与边界：长程规划应逐步记录观察、动作、预算与终态，反思建议只能在独立回归通过后进入策略。对比同预算的无反思、固定反思与按失败触发反思；本仓库既有任务不证明通用长期自主能力。

### 6.3 Computer Use 与具身 Agent

原有基础：元素 ref、失效截图、真实本地浏览器像素/点击、动作块与反馈控制。

本次处理：机制实验 + 方法讲解。

- 原审计：VLA 参数训练、视觉—动作真实数据
- 原审计：真实传感/延迟/接触等具身任务；一维模拟仅覆盖反馈机制

- [llm_agent/m49_learned_world/README.md](../llm_agent/m49_learned_world/README.md)

验收方法与边界：检查拟合参数、同状态反向动作差异和部署漂移下的预测误差。迁移机器人时需真实传感/执行延迟、部分可观测和接触任务，先离线验证再受控执行。

新增实验的实际边界：
- 一维线性、全状态可观测、无延迟环境。学习到的是此分布下的转移近似，不能推断掌握真实物理因果或安全机器人控制。

### 6.4 Multi-Agent Systems

原有基础：orchestrator/workers、独立审查、共享版本冲突、相关错误、预算和终态验收。

本次处理：沿用已有闭环 + 补充评测协议。

- 原审计：不是增加 Agent 数量就提高能力；真实模型通信收益需要另测

- [llm_agent/m11_orchestrator/README.md](../llm_agent/m11_orchestrator/README.md)
- [llm_agent/m11_orchestrator/demo.py](../llm_agent/m11_orchestrator/demo.py)
- [llm_agent/m30_collaboration/README.md](../llm_agent/m30_collaboration/README.md)
- [llm_agent/m30_collaboration/demo.py](../llm_agent/m30_collaboration/demo.py)

验收方法与边界：多 Agent 比较固定总模型调用预算，单独消融角色、通信与审查。对相同错误来源的多个 Agent 不能假设错误独立；记录协调成本、冲突和最终通过率，真实模型收益仍需另测。

### 7.1 视觉语言理解

原有基础：CLIP、视觉编码器/投影、Qwen2-VL/M-RoPE；视觉定位有像素夹具。

本次处理：机制实验 + 方法讲解。

- 原审计：VQA/OCR/文档与图表理解任务级实验
- 原审计：数量/空间关系、视觉不确定性与拒答

- [llm_models/run_models/multimodal/vision_tasks/README.md](../llm_models/run_models/multimodal/vision_tasks/README.md)

验收方法与边界：切换遮挡和歧义，检查正确读取与拒答是否合理。模板 OCR 只识别四个合成字形；真实 VLM 还需独立图像集、未见字体和文档版式。

新增实验的实际边界：
- 已知字体和合成柱图的像素算法，不是通用 VLM 或真实 OCR；距离差只是拒答规则，不是校准后的正确概率。

### 7.2 Speech 与 Omni

原有基础：Whisper、Thinker/Talker 架构，语音交互的回声、打断、快慢响应事件。

本次处理：机制实验 + 方法讲解。

- 原审计：波形前处理、codec→波形、真实流式感知生成
- 原审计：说话人识别、口音/噪声/韵律评测；m34 不是 ASR/TTS 实现

- [llm_models/run_models/multimodal/audio_pipeline/README.md](../llm_models/run_models/multimodal/audio_pipeline/README.md)

验收方法与边界：比较相同信号的噪声和量化失真。此例不训练 ASR/TTS；已有语音交互状态机用于打断与快慢响应，两者共同覆盖信号链和交互链。

新增实验的实际边界：
- 真实数值信号处理，但输入是合成音调，不是语音。没有声纹识别、ASR/TTS 或神经 codec 能力；音频输出必须和说话人授权一起管理。

### 7.3 图像、视频与音频生成

原有基础：VAE/3D VAE、DiT/MM-DiT/Video DiT/VAR；Omni 只到 codec logits。

本次处理：机制实验 + 方法讲解。

- 原审计：可控编辑、运动/身份一致性、镜头控制与音画同步
- 原审计：音频生成完整路径
- 原审计：水印、内容溯源与深伪检测

- [llm_models/run_models/multimodal/audio_pipeline/README.md](../llm_models/run_models/multimodal/audio_pipeline/README.md)
- [llm_models/run_models/generative/media_evaluation/README.md](../llm_models/run_models/generative/media_evaluation/README.md)

验收方法与边界：比较相同信号的噪声和量化失真。此例不训练 ASR/TTS；已有语音交互状态机用于打断与快慢响应，两者共同覆盖信号链和交互链。；分别修改身份属性与字节内容，检查时序指标和签名验证反应。公开教学密钥不能保护真实来源；完整内容凭证还需要身份、密钥管理和编辑链。

新增实验的实际边界：
- 真实数值信号处理，但输入是合成音调，不是语音。没有声纹识别、ASR/TTS 或神经 codec 能力；音频输出必须和说话人授权一起管理。
- 合成短片与公开演示密钥，不是安全水印、C2PA 实现或通用深伪检测器；签名验证的是完整性与密钥持有者，不证明内容真实。

### 7.4 3D 空间智能与世界模型

原有基础：m36 解释世界模型/规划、模型偏差、反馈与 sim2real。

本次处理：机制实验 + 方法讲解。

- 原审计：从观测学习动力学的实验
- 原审计：多步预测误差与动作条件因果对照
- 原审计：3D 空间表征与空间推理；视频生成不算掌握动力学

- [llm_agent/m49_learned_world/README.md](../llm_agent/m49_learned_world/README.md)

验收方法与边界：检查拟合参数、同状态反向动作差异和部署漂移下的预测误差。迁移机器人时需真实传感/执行延迟、部分可观测和接触任务，先离线验证再受控执行。

新增实验的实际边界：
- 一维线性、全状态可观测、无延迟环境。学习到的是此分布下的转移近似，不能推断掌握真实物理因果或安全机器人控制。

### 8.1 分布式训练

原有基础：DP/TP/PP/EP、ZeRO/FSDP、集合通信、重算、状态/RNG/游标恢复、多维预算。

本次处理：沿用已有闭环 + 补充评测协议。

- 原审计：numpy 仿真与成本模型，不是多机多卡故障/吞吐实测

- [llm_train/m05_zero_fsdp/README.md](../llm_train/m05_zero_fsdp/README.md)
- [llm_train/m05_zero_fsdp/demo.py](../llm_train/m05_zero_fsdp/demo.py)
- [llm_train/m08_checkpoint_resume/README.md](../llm_train/m08_checkpoint_resume/README.md)
- [llm_train/m08_checkpoint_resume/demo.py](../llm_train/m08_checkpoint_resume/demo.py)
- [llm_train/m22_parallel_budget/README.md](../llm_train/m22_parallel_budget/README.md)
- [llm_train/m22_parallel_budget/demo.py](../llm_train/m22_parallel_budget/demo.py)

验收方法与边界：多机实测先校验梯度/优化器状态等价，再测通信重叠、rank 拖尾与故障恢复；检查 checkpoint 一致点和恢复后样本重复/遗漏。仓库成本模型不能代替带拓扑、设备与故障注入的运行记录。

### 8.2 模型压缩

原有基础：权重量化、KV 量化、离线/在线蒸馏，多项误差与字节对照。

本次处理：机制实验 + 方法讲解。

- 原审计：权重/结构化剪枝、稀疏性与实际加速区别
- 原审计：压缩前后长尾能力、推理与安全退化的切片评测

- [llm_infer/m30_pruning_fusion/README.md](../llm_infer/m30_pruning_fusion/README.md)

验收方法与边界：先核对数学输出，再计算理想访存次数。这里没有 GPU kernel 或真实加速比；实测需 warm-up、同步、固定 batch/上下文、设备功率与分位数。

新增实验的实际边界：
- numpy 的通道裁剪与理论流量。没有稀疏 GPU kernel 或融合 kernel 的实测加速；模型压缩还需看长尾、推理、安全切片。

### 8.3 推理与 Serving

原有基础：分页 KV、前缀/Radix、连续批处理、chunked prefill、投机、P/D、路由与容量。

本次处理：机制实验 + 方法讲解。

- 原审计：通用算子融合/编译执行的机制实验；CUDA Graph 调度复用不能代替融合
- 原审计：真实 GPU 吞吐/延迟/质量联合测量为生产深度边界

- [llm_infer/m30_pruning_fusion/README.md](../llm_infer/m30_pruning_fusion/README.md)

验收方法与边界：先核对数学输出，再计算理想访存次数。这里没有 GPU kernel 或真实加速比；实测需 warm-up、同步、固定 batch/上下文、设备功率与分位数。

新增实验的实际边界：
- numpy 的通道裁剪与理论流量。没有稀疏 GPU kernel 或融合 kernel 的实测加速；模型压缩还需看长尾、推理、安全切片。

### 8.4 端侧 LLM 与 Green AI

原有基础：量化和内存预算可以复用，但没有端侧专题。

本次处理：机制实验 + 方法讲解。

- 原审计：设备内存/带宽/功耗/散热/电池约束
- 原审计：质量—延迟—能耗联合预算
- 原审计：模型、编译器、kernel 与设备协同设计

- [llm_infer/m31_edge_budget/README.md](../llm_infer/m31_edge_budget/README.md)

验收方法与边界：全部参数是公开假设：1B 参数、2 GiB 内存等。切换精度和热态观察可行性及成本；真实选型还需测量量化质量、温度、功率曲线和电池。

新增实验的实际边界：
- 预算模型，不是手机或芯片跑分，不提供特定设备续航承诺。真实能耗需扣除/报告空闲基线并说明测量边界。

### 9.1 能力与可靠性评测

原有基础：PPL、污染、pass@k、失败切片、配对 bootstrap、独立留出与套件版本。

本次处理：机制实验 + 方法讲解。

- 原审计：回答置信校准：可靠性图/ECE/Brier 与拒答权衡
- 原审计：群体公平性、毒性、输入扰动鲁棒性
- 原审计：HELM 式同场景多指标比较

- [llm_train/m25_reliability_metrics/README.md](../llm_train/m25_reliability_metrics/README.md)
- [llm_agent/m48_safety_governance/README.md](../llm_agent/m48_safety_governance/README.md)

验收方法与边界：实验固定同一组正确/错误标签，只改变置信度或拒答阈值，并实际拟合成对胜负得分。真实校准必须使用未参与阈值选择的测试集；不要从教学数组推断某个模型可靠。；四个合成案例只验证具体授权与误报逻辑。生产应把策略版本、决策日志、人工复核与撤销测试纳入验收；这不构成通用防越狱保证或法律合规判定。

新增实验的实际边界：
- 确定性评测夹具，不是模型能力榜单。拒答提高已答准确率会减少覆盖；Kappa 受标签流行率影响。
- 公开合成夹具与动作门，没有实际攻击或真实安全模型。字符规则不能解决通用提示注入；字段齐全也不等于获得法律授权或通过合规认证。

### 9.2 人工评测与 LLM Judge

原有基础：位置交换、长度偏好、rubric、预标注夹具一致率/假阳性/假阴性。

本次处理：机制实验 + 方法讲解。

- 原审计：人工成对评价协议、标注者一致性
- 原审计：Arena 类排名与不确定性
- 原审计：真实多裁判/自我偏好消融；规则裁判并非真实 LLM Judge

- [llm_train/m25_reliability_metrics/README.md](../llm_train/m25_reliability_metrics/README.md)

验收方法与边界：实验固定同一组正确/错误标签，只改变置信度或拒答阈值，并实际拟合成对胜负得分。真实校准必须使用未参与阈值选择的测试集；不要从教学数组推断某个模型可靠。

新增实验的实际边界：
- 确定性评测夹具，不是模型能力榜单。拒答提高已答准确率会减少覆盖；Kappa 受标签流行率影响。

### 9.3 Agent 与系统评测

原有基础：任务终态、动作授权、首错、恢复、pass^k、trace/span、预算与版本固定。

本次处理：沿用已有闭环 + 补充评测协议。

- 原审计：在线真实任务大样本评测不在教学验收范围

- [llm_agent/m13_evals/README.md](../llm_agent/m13_evals/README.md)
- [llm_agent/m13_evals/demo.py](../llm_agent/m13_evals/demo.py)
- [llm_agent/m31_eval_suite/README.md](../llm_agent/m31_eval_suite/README.md)
- [llm_agent/m31_eval_suite/demo.py](../llm_agent/m31_eval_suite/demo.py)
- [llm_agent/m31_eval_suite/trace_demo.py](../llm_agent/m31_eval_suite/trace_demo.py)

验收方法与边界：在线任务评测保留任务抽样框、失败与取消分母、用户群体和版本；按任务而非单个 span 统计成功，分别报告成本、尾延迟与副作用。真实在线大样本和用户满意度不属于本地教学实验的已验证结果。

### 9.4 机制可解释性

原有基础：形状/注意力可视化、BERT 梯度检查与检索探针均不是机制解释课程。

本次处理：机制实验 + 方法讲解。

- 原审计：表示探针与相关/因果区别
- 原审计：activation patching、causal tracing、circuit analysis
- 原审计：SAE 稀疏特征与干预验证

- [llm_models/run_models/foundation/interpretability/README.md](../llm_models/run_models/foundation/interpretability/README.md)

验收方法与边界：先比较两个同样相关的通道在干预后的差别，再看 SAE 的重建与稀疏权衡。这里使用已知小网络做因果校验，不宣称解释真实 LLM 的全部机制。

新增实验的实际边界：
- 线性读出与合成激活。SAE 特征没有自动获得语义名称；注意力图、相关探针或可读 CoT 都不能单独证明模型机制。

### 10.1 对齐安全与 Red Teaming

原有基础：RLAIF、奖励捷径、Agent safety 任务、执行权限。

本次处理：机制实验 + 方法讲解。

- 原审计：有害/正常请求的配对安全训练和评测
- 原审计：拒绝边界、过度拒答、迎合/欺骗
- 原审计：对抗改写与模型层红队流程

- [llm_finetune/run_finetune/online_preference/README.md](../llm_finetune/run_finetune/online_preference/README.md)
- [llm_agent/m48_safety_governance/README.md](../llm_agent/m48_safety_governance/README.md)

验收方法与边界：固定随机种子和交互预算，比较偏好变化前后各用户指标。模型层迎合与有意欺骗不能只由一个错误答案判定，需要可核验的任务行为和多个条件。；四个合成案例只验证具体授权与误报逻辑。生产应把策略版本、决策日志、人工复核与撤销测试纳入验收；这不构成通用防越狱保证或法律合规判定。

新增实验的实际边界：
- 有限动作上的实际 DPO 梯度更新，评审器是公开规则，不是人工偏好或真实 LLM Judge。奖励规则与部署验收应独立审计。
- 公开合成夹具与动作门，没有实际攻击或真实安全模型。字符规则不能解决通用提示注入；字段齐全也不等于获得法律授权或通过合规认证。

### 10.2 Jailbreak 与 Prompt Injection

原有基础：工具权限、路径围栏、提示注入案例、敏感信息脱敏。

本次处理：机制实验 + 方法讲解。

- 原审计：模型越狱与系统注入的独立威胁模型和测试
- 原审计：多载体间接注入、策略误报/漏报评测；关键词护栏不是通用防御

- [llm_agent/m48_safety_governance/README.md](../llm_agent/m48_safety_governance/README.md)

验收方法与边界：四个合成案例只验证具体授权与误报逻辑。生产应把策略版本、决策日志、人工复核与撤销测试纳入验收；这不构成通用防越狱保证或法律合规判定。

新增实验的实际边界：
- 公开合成夹具与动作门，没有实际攻击或真实安全模型。字符规则不能解决通用提示注入；字段齐全也不等于获得法律授权或通过合规认证。

### 10.3 隐私、记忆与版权

原有基础：个人记忆同意/作用域/删除、脱敏；通用评测里提到成员推断。

本次处理：机制实验 + 方法讲解。

- 原审计：差分隐私、训练数据提取、成员推断实验
- 原审计：参数级 machine unlearning 与效用/遗忘验证
- 原审计：数据授权、输出相似性与内容溯源

- [llm_finetune/run_finetune/privacy_unlearning/README.md](../llm_finetune/run_finetune/privacy_unlearning/README.md)
- [llm_models/run_models/generative/media_evaluation/README.md](../llm_models/run_models/generative/media_evaluation/README.md)

验收方法与边界：仅用合成记录重现三条路径。固定公开随机种子方便教学复现，却不提供真实私密数据的 DP 保护；真实发布必须使用合适随机源、审计 accountant 与全部数据访问。；分别修改身份属性与字节内容，检查时序指标和签名验证反应。公开教学密钥不能保护真实来源；完整内容凭证还需要身份、密钥管理和编辑链。

新增实验的实际边界：
- 全部是合成记录。公开固定种子仅用于复现演示，不能用于保护真实私密数据；该精确遗忘方法仅适用于此凸闭式模型，不适用于一般 LLM。
- 合成短片与公开演示密钥，不是安全水印、C2PA 实现或通用深伪检测器；签名验证的是完整性与密钥持有者，不证明内容真实。

### 10.4 公平性、社会影响与治理

原有基础：Agent 社会实验讨论收益分配，任务评测记录版本和人群，已有权限/人工介入。

本次处理：机制实验 + 方法讲解。

- 原审计：群体公平性及语言/文化差异评测
- 原审计：数据卡/模型卡、责任与部署风险记录
- 原审计：治理生命周期与持续监控的完整案例

- [llm_train/m25_reliability_metrics/README.md](../llm_train/m25_reliability_metrics/README.md)
- [llm_train/m26_multilingual/README.md](../llm_train/m26_multilingual/README.md)
- [llm_agent/m48_safety_governance/README.md](../llm_agent/m48_safety_governance/README.md)

验收方法与边界：实验固定同一组正确/错误标签，只改变置信度或拒答阈值，并实际拟合成对胜负得分。真实校准必须使用未参与阈值选择的测试集；不要从教学数组推断某个模型可靠。；先跑三个采样条件，再核对错误来自训练覆盖还是翻译标签。当前短语分类不验证通用跨语言语义迁移；真实模型需留出未见实体和原生表达。；四个合成案例只验证具体授权与误报逻辑。生产应把策略版本、决策日志、人工复核与撤销测试纳入验收；这不构成通用防越狱保证或法律合规判定。

新增实验的实际边界：
- 确定性评测夹具，不是模型能力榜单。拒答提高已答准确率会减少覆盖；Kappa 受标签流行率影响。
- 两种受控语言模板，不是低资源语言 benchmark。分词器扩词还需处理新增 embedding、旧分词兼容和训练成本。
- 公开合成夹具与动作门，没有实际攻击或真实安全模型。字符规则不能解决通用提示注入；字段齐全也不等于获得法律授权或通过合规认证。

### 11.1 Code LLMs

原有基础：真实修改本地代码、运行测试、修复循环、沙箱与验收；有程序化行动。

本次处理：机制实验 + 方法讲解。

- 原审计：代码模型训练目标/补全与测试生成
- 原审计：跨文件依赖和仓库级任务
- 原审计：漏洞/私有仓库适配评测；单函数夹具不是仓库级能力

- [llm_agent/m50_repository_verification/README.md](../llm_agent/m50_repository_verification/README.md)

验收方法与边界：对完整、局部和投机补丁分别启动独立 Python 进程验收。可见测试被替换为 assert True 也不能影响外部判分；本例只运行本仓库生成的可信小片段。

新增实验的实际边界：
- 两个模块的受控 API 迁移，候选补丁由程序给定，不是通用 Coding Agent 或安全沙箱；-I 只隔离 Python 环境，不隔离操作系统权限。

### 11.2 数学推理与 AI for Science

原有基础：算术 verifier、RLVR、受限程序计算与候选验收可复用。

本次处理：机制实验 + 方法讲解。

- 原审计：形式化证明与 theorem prover 闭环
- 原审计：科学数据/模拟器/假设生成→独立实验验证
- 原审计：数值正确、形式正确与现实假设正确的区别

- [llm_agent/m51_formal_science/README.md](../llm_agent/m51_formal_science/README.md)

验收方法与边界：伪造证明必须被拒；拟合误差与留出误差分别显示。本例是命题逻辑和合成数值实验，不冒充数学竞赛解题器或真实科研发现。

新增实验的实际边界：
- 有限命题逻辑内核与合成测量，不是通用定理证明器或科学发现系统。证明只相对于假设成立，现实假设仍需实验检验。

### 11.3 医疗、法律与金融 LLM

原有基础：通用 RAG、授权与验收机制可作基础，但未见具体领域教学案例。

本次处理：机制实验 + 方法讲解。

- 原审计：领域数据适配与独立工作流评测
- 原审计：时效/地域/证据、专家复核和责任边界
- 原审计：校准、隐私及高代价错误的案例化验收

- [llm_agent/m52_domain_workflows/README.md](../llm_agent/m52_domain_workflows/README.md)

验收方法与边界：比较不加约束、证据过滤和带外部复核的输出状态。全部术语与规则均为虚构测试夹具，不提供专业结论；真实上线需由领域专家制定标签与验收阈值。

新增实验的实际边界：
- 所有地区、日期和规则均为虚构夹具，不提供医疗、法律或投资建议。未实现真实专家系统；这里验证资料和工作流边界。

### 11.4 多语言与低资源语言

原有基础：中文检索切词、m45 中英同义检索小探针；BPE 基础。

本次处理：机制实验 + 方法讲解。

- 原审计：跨语言迁移与语言均衡训练
- 原审计：tokenizer 分词膨胀和低资源数据预算
- 原审计：文化语境、翻译偏差与本地化独立评测

- [llm_train/m26_multilingual/README.md](../llm_train/m26_multilingual/README.md)

验收方法与边界：先跑三个采样条件，再核对错误来自训练覆盖还是翻译标签。当前短语分类不验证通用跨语言语义迁移；真实模型需留出未见实体和原生表达。

新增实验的实际边界：
- 两种受控语言模板，不是低资源语言 benchmark。分词器扩词还需处理新增 embedding、旧分词兼容和训练成本。

## 仍需外部资源的验证

以下工作没有执行，也不写成“能力已经证明”：真实语料长周期预训练、真实多机 GPU 吞吐/功耗、通用 VLM/ASR/TTS 与神经 codec 质量、VLA/3D 接触控制、完整 Lean 证明搜索、真实领域专家与在线用户大样本评测。各章节已给出机制、拆分方式、指标与验收协议，后续可替换教学夹具开展这些实验。

知识覆盖的范围以这张长图和固定版本的 ai-agent-book 为依据，不把有限对照表外推为整个 LLM 研究领域永无遗漏。
