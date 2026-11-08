// 二级主题决定阅读顺序；三级章节沿用原 route，已有链接和学习进度不变。
// 新增章节必须明确归组，避免悄悄回到一长串平铺目录。
const group = (id, title, routes) => ({ id, title, routes: routes.split(' ') })
export const chapterGroups = {
  basic: [
    group('forward', '数据与前向计算', 'basic-data basic-forward'),
    group('learning', '梯度、优化与生成', 'basic-backward basic-optim-sample'),
  ],
  models: [
    group('transformer', 'Transformer 基础', 'attention position blocks models-generation'),
    group('sequence', '长序列与混合架构', 'models-qknorm-yarn models-dsa models-mamba models-mtp'),
    group('experts', '稀疏专家模型', 'moe models-moe-balance models-gptoss'),
    group('generation', '生成模型分支', 'diffusion models-llada models-var'),
  ],
  train: [
    group('data-scale', '数据与训练规模', 'train-data-packing train-scaling'),
    group('parallel', '并行与通信', 'train-batch-ddp train-model-parallel train-pipeline-schedules train-moe-seq train-ulysses'),
    group('optimization', '显存、精度与优化', 'train-memory train-precision-stability train-low-precision train-lr-schedule train-muon'),
    group('validation', '评测与完整训练', 'train-eval train-collectives-loop'),
  ],
  finetune: [
    group('adaptation', '监督微调与参数适配', 'finetune-sft finetune-lora finetune-qlora finetune-dora finetune-knowledge-update'),
    group('preference', '偏好与奖励建模', 'finetune-dpo finetune-simpo-orpo finetune-rlaif finetune-prm'),
    group('rl', '强化学习与蒸馏', 'finetune-rlhf finetune-ppo finetune-grpo-variants finetune-rlvr finetune-onpolicy-distill'),
    group('delivery', '模型合并与落盘', 'finetune-merge finetune-runs'),
  ],
  infer: [
    group('cache', 'KV 缓存与内存', 'infer-kv-memory infer-prefix-radix infer-kv-footprint infer-attention-sinks infer-kv-offload'),
    group('decoding', '解码与推理计算', 'infer-decode-control infer-structured-output infer-tree-speculation infer-test-time-compute'),
    group('compute', '量化与算子优化', 'infer-compute infer-quant-awq infer-sparse-decode'),
    group('serving', '服务调度与引擎', 'infer-scheduler infer-moe-serving infer-multi-replica-routing infer-engine'),
  ],
  agent: [
    group('foundations', '循环、规划与系统选择', 'agent-loop agent-planning agent-system-design'),
    group('tools', '工具、协议与安全', 'agent-tools-permissions agent-extensibility agent-mcp agent-tool-discovery agent-guardrails agent-programmatic-actions agent-verifiable-reasoning'),
    group('memory', '上下文与记忆管理', 'agent-context-memory agent-context-engineering agent-prompt-caching agent-context-cost agent-memory-lifecycle agent-memory-privacy'),
    group('knowledge', '检索与知识组织', 'agent-rag agent-evidence-chain agent-knowledge-organization agent-vector-index'),
    group('runtime', '运行时与多智能体协作', 'agent-state-subagents agent-runtime-reliability agent-async-runtime agent-a2a agent-orchestrator agent-collaboration agent-agent-society'),
    group('evaluation', '验收、评估与轨迹数据', 'agent-completion-verification agent-evals agent-eval-suite agent-trajectory-learning'),
    group('learning', '训练与持续改进', 'agent-agent-training agent-midtraining agent-self-distillation agent-learning-loop agent-skill-evolution agent-meta-optimization'),
    group('applications', '交互应用与完整系统', 'agent-coding-agent agent-computer-use agent-visual-grounding agent-voice-interaction agent-embodied-control agent-full-loop'),
  ],
}

export function organizeStage(stage) {
  const chapters = new Map(stage.chapters.map((c) => [c.route, c]))
  const seen = new Set()
  const groups = chapterGroups[stage.id].map((g, gi) => ({
    id: `${stage.id}-${g.id}`, title: g.title, number: `${stage.idx}.${gi + 1}`,
    chapters: g.routes.map((route, ci) => {
      if (!chapters.has(route) || seen.has(route)) throw new Error(`章节归组重复或不存在: ${route}`)
      seen.add(route)
      return { ...chapters.get(route), number: `${stage.idx}.${gi + 1}.${ci + 1}`, groupId: `${stage.id}-${g.id}`, groupTitle: g.title }
    }),
  }))
  const missing = [...chapters.keys()].filter((route) => !seen.has(route))
  if (missing.length) throw new Error(`章节尚未归组: ${missing.join(', ')}`)
  // 保留导出的 *Chapters 数组引用，让总览、翻章和侧栏共用同一个顺序。
  stage.chapters.splice(0, stage.chapters.length, ...groups.flatMap((g) => g.chapters))
  stage.groups = groups
}
