// 研究补充术语：连接既有章节。
export default [
  {
    "term": "课程学习",
    "aka": "Curriculum learning",
    "oneliner": "按训练进度分阶段编排样本难度或领域配比的训练策略; 评估收益时须在相同总计算预算下与静态均匀混合基线严格对照。",
    "route": "train-data-packing",
    "stage": "train"
  },
  {
    "term": "合成反馈退化",
    "aka": "Model collapse",
    "oneliner": "模型在自身生成数据上多代递归重训引发的分布窄化与长尾坍缩现象; 防控时需同时锚定真实数据比例并审查质量过滤器的选择偏差。",
    "route": "train-data-packing",
    "stage": "train"
  },
  {
    "term": "回答校准",
    "aka": "Calibration / ECE / Brier",
    "oneliner": "衡量模型预测置信度与真实样本正确率一致程度的属性; 引入基于置信度的拒答机制时须联合报告准确率与有效回答覆盖率。",
    "route": "train-eval",
    "stage": "train"
  },
  {
    "term": "IPO",
    "aka": "Identity Preference Optimization",
    "oneliner": "用平方损失将策略相对参考模型的偏好 log-ratio 约束至有限目标值的对齐方法, 缓解 DPO 在弱噪声偏好下的过拟合。",
    "route": "finetune-dpo",
    "stage": "finetune"
  },
  {
    "term": "Prompt Tuning",
    "aka": "Soft prompt",
    "oneliner": "冻结预训练模型全部参数, 仅在输入嵌入层前端拼接并训练一组连续软提示向量的微调方法; 区别于离散词表上的手工提示词工程。",
    "route": "finetune-lora",
    "stage": "finetune"
  },
  {
    "term": "Prefix Tuning",
    "aka": "Layer-wise KV prefix",
    "oneliner": "冻结主干参数并在每一层注意力前端直接学习一组可训练 K/V 前缀状态的参数高效微调方法。",
    "route": "finetune-lora",
    "stage": "finetune"
  },
  {
    "term": "瓶颈 Adapter",
    "aka": "Bottleneck adapter",
    "oneliner": "冻结主干权重并在子层残差通路中串联插入「线性降维、非线性激活与线性升维」小瓶颈模块的参数高效微调结构。",
    "route": "finetune-lora",
    "stage": "finetune"
  },
  {
    "term": "参数知识编辑",
    "aka": "Knowledge editing",
    "oneliner": "定向修改权重中存储的特定事实, 评测时须同步核验同义改写泛化、无关知识局部性与关联事实一致性。",
    "route": "finetune-sft",
    "stage": "finetune"
  },
  {
    "term": "参数遗忘",
    "aka": "Machine unlearning",
    "oneliner": "调整参数使模型在行为与分布上逼近从未见过目标删除数据的重训参照模型; 本质上不同于推理期的表层输出过滤屏蔽。",
    "route": "agent-memory-privacy",
    "stage": "agent"
  },
  {
    "term": "差分隐私",
    "aka": "Differential privacy",
    "oneliner": "限制任意两组仅差单条记录的相邻数据集在训练输出分布上可区分度的隐私保证框架, 须明确邻接粒度与累计组合预算。",
    "route": "agent-memory-privacy",
    "stage": "agent"
  },
  {
    "term": "成员推断",
    "aka": "Membership inference",
    "oneliner": "根据模型输出特征推断特定样本是否曾参与训练的隐私审计方法, 评测时须报告给定假阳率下的 TPR/FPR。",
    "route": "agent-memory-privacy",
    "stage": "agent"
  },
  {
    "term": "推理忠实性",
    "aka": "CoT faithfulness",
    "oneliner": "模型输出的外部推理解释是否真实反映其内部决策因果机制; 推导步骤本身可被验证并不自动代表该解释忠实。",
    "route": "infer-test-time-compute",
    "stage": "infer"
  },
  {
    "term": "结构化剪枝",
    "aka": "Structured pruning",
    "oneliner": "按通道或注意力头等完整规整粒度裁除参数以实质缩小矩阵算子维度的压缩方法; 单纯将离散权重置零并不必然带来硬件加速。",
    "route": "infer-compute",
    "stage": "infer"
  },
  {
    "term": "有效上下文",
    "aka": "Effective context",
    "oneliner": "模型在任意位置深度下均能稳定检索并融合关键证据的实际可用上下文跨度, 通常小于架构名义上允许输入的最大窗口长度。",
    "route": "agent-context-engineering",
    "stage": "agent"
  },
  {
    "term": "激活修补",
    "aka": "Activation patching",
    "oneliner": "因果机制定位方法: 用干净输入运行在特定层位的激活值替换受扰动运行的对应激活, 根据目标输出恢复程度识别关键内部组件。",
    "route": "blocks",
    "stage": "models"
  },
  {
    "term": "稀疏自编码器",
    "aka": "SAE",
    "oneliner": "用高维稀疏过完备字典分解并重建模型隐层激活的特征解耦模型; 字典特征对应的可解释语义仍须通过因果干预实验验证。",
    "route": "blocks",
    "stage": "models"
  },
  {
    "term": "说话人分段",
    "aka": "Speaker diarization",
    "oneliner": "从连续音频中切分标注「谁在何时说话」的语音任务; 评测时须与说话人身份识别和语音转写分别独立衡量。",
    "route": "agent-voice-interaction",
    "stage": "agent"
  },
  {
    "term": "形式证明内核",
    "aka": "Proof kernel",
    "oneliner": "定理证明系统中独立核验每步形式推导规则合法性的极简可信核心; 验收通过仅保证结论相对于给定公理与定义严格成立。",
    "route": "agent-programmatic-actions",
    "stage": "agent"
  },
  {
    "term": "内容来源链",
    "aka": "Content provenance",
    "oneliner": "以密码学签名记录数字内容生成源头与历次编辑流转关系的溯源机制; 签名合法仅证明链路完整而并不保证所描绘事件本身真实。",
    "route": "diffusion",
    "stage": "models"
  },
  {
    "term": "分词膨胀",
    "aka": "Token fertility",
    "oneliner": "表达同等语义内容在不同语言中消耗的 token 数量差异, 会直接造成非优势语种的计费成本上升与有效上下文窗口缩水。",
    "route": "basic-data",
    "stage": "basic"
  }
]
