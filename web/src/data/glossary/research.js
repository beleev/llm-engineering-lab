// 研究补充术语：连接既有章节。
export default [
  {
    "term": "课程学习",
    "aka": "Curriculum learning",
    "oneliner": "按阶段调整样本难度或分布；与静态混合在相同预算下对照",
    "route": "train-data-packing",
    "stage": "train"
  },
  {
    "term": "合成反馈退化",
    "aka": "Model collapse",
    "oneliner": "递归自采样可能丢失长尾；真实锚点和过滤偏差需共同检查",
    "route": "train-data-packing",
    "stage": "train"
  },
  {
    "term": "回答校准",
    "aka": "Calibration / ECE / Brier",
    "oneliner": "置信度与实际正确频率是否匹配；拒答时同时看覆盖率",
    "route": "train-eval",
    "stage": "train"
  },
  {
    "term": "IPO",
    "aka": "Identity Preference Optimization",
    "oneliner": "把相对参考模型的偏好 log-ratio 拉向有限平方目标",
    "route": "finetune-dpo",
    "stage": "finetune"
  },
  {
    "term": "Prompt Tuning",
    "aka": "Soft prompt",
    "oneliner": "冻结模型，训练输入前缀的向量；不同于手写提示词",
    "route": "finetune-lora",
    "stage": "finetune"
  },
  {
    "term": "Prefix Tuning",
    "aka": "Layer-wise KV prefix",
    "oneliner": "直接学习各层注意力的 K/V 前缀状态",
    "route": "finetune-lora",
    "stage": "finetune"
  },
  {
    "term": "瓶颈 Adapter",
    "aka": "Bottleneck adapter",
    "oneliner": "在残差中插入降维、非线性与升维的可训练分支",
    "route": "finetune-lora",
    "stage": "finetune"
  },
  {
    "term": "参数知识编辑",
    "aka": "Knowledge editing",
    "oneliner": "定向修改参数事实，同时核验改写、局部性与关联事实",
    "route": "finetune-sft",
    "stage": "finetune"
  },
  {
    "term": "参数遗忘",
    "aka": "Machine unlearning",
    "oneliner": "使删除后模型接近不含该数据的重训参照；不同于输出屏蔽",
    "route": "agent-memory-privacy",
    "stage": "agent"
  },
  {
    "term": "差分隐私",
    "aka": "Differential privacy",
    "oneliner": "限制相邻数据集的输出分布差异，需明确邻接和组合预算",
    "route": "agent-memory-privacy",
    "stage": "agent"
  },
  {
    "term": "成员推断",
    "aka": "Membership inference",
    "oneliner": "依据模型输出判断某记录是否参加训练，需报告 TPR/FPR",
    "route": "agent-memory-privacy",
    "stage": "agent"
  },
  {
    "term": "推理忠实性",
    "aka": "CoT faithfulness",
    "oneliner": "外部解释是否反映实际决策机制；过程可验证并不自动保证忠实",
    "route": "infer-test-time-compute",
    "stage": "infer"
  },
  {
    "term": "结构化剪枝",
    "aka": "Structured pruning",
    "oneliner": "删除通道等完整结构以改变算子形状；置零不必然加速",
    "route": "infer-compute",
    "stage": "infer"
  },
  {
    "term": "有效上下文",
    "aka": "Effective context",
    "oneliner": "能稳定利用证据的范围，不等于允许输入的最大窗口",
    "route": "agent-context-engineering",
    "stage": "agent"
  },
  {
    "term": "激活修补",
    "aka": "Activation patching",
    "oneliner": "用干净运行的激活替换受损运行位置，观察输出恢复",
    "route": "blocks",
    "stage": "models"
  },
  {
    "term": "稀疏自编码器",
    "aka": "SAE",
    "oneliner": "用稀疏过完备特征重建激活，特征语义仍需干预验证",
    "route": "blocks",
    "stage": "models"
  },
  {
    "term": "说话人分段",
    "aka": "Speaker diarization",
    "oneliner": "识别谁在何时说话；与说话人身份识别和转写分别评测",
    "route": "agent-voice-interaction",
    "stage": "agent"
  },
  {
    "term": "形式证明内核",
    "aka": "Proof kernel",
    "oneliner": "独立核验每一步推导是否合法，结论相对于定义和公理成立",
    "route": "agent-programmatic-actions",
    "stage": "agent"
  },
  {
    "term": "内容来源链",
    "aka": "Content provenance",
    "oneliner": "记录内容来源和编辑签署关系；有效签名不保证事件真实",
    "route": "diffusion",
    "stage": "models"
  },
  {
    "term": "分词膨胀",
    "aka": "Token fertility",
    "oneliner": "同等语义在不同语言中消耗的 token 数差异，影响成本和窗口",
    "route": "basic-data",
    "stage": "basic"
  }
]
