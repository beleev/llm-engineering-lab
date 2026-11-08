// 每个实验保留一个主讲入口；分段顺序由正文决定。
const map = {
  "finetune-runs": [
    "EngineeringCompletionLab"
  ],
  "agent-rag": [
    "EngineeringCompletionLab"
  ],
  "agent-mcp": [
    "EngineeringCompletionLab"
  ],
  "finetune-grpo-variants": [
    "EngineeringCompletionLab"
  ],
  "agent-eval-suite": [
    "EngineeringCompletionLab"
  ],
  "infer-compute": [
    "EngineeringCompletionLab"
  ]
}

map['infer-scheduler'] = ['EngineeringCompletionLab']
map['agent-state-subagents'] = ['EngineeringCompletionLab']
map['infer-prefix-radix'] = ['EngineeringCompletionLab']
export default map
