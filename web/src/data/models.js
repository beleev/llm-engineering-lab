// 全站的结构数据: 六个阶段、每阶段的章节目录、训练 / 推理 / Agent 的模块表、模型时间轴。
// 章节正文不在这里, 在 data/topics/<stage>.js。
// 时间轴的三条主线: left (语言), eye (多模态理解), right (图像视频生成)

import { extraChapters } from 'virtual:course-catalog'
import { organizeStage } from './chapterGroups.js'

// 收集原始章节；最终分组和阅读顺序由 chapterGroups.js 决定。
// topics/*.js 追加的章节默认接在本阶段末尾 (压轴章之前); 写了 after: '某章 route' 的挪到那一章后面
function place(list) {
  const out = list.filter((c) => !c.after)
  for (const c of list.filter((c) => c.after)) {
    const i = out.findIndex((x) => x.route === c.after)
    out.splice(i < 0 ? out.length - 1 : i + 1, 0, c)   // after 指向的章不存在: 插到倒数第二, 也就是压轴章之前
  }
  return out
}

export const tracks = {
  left:  { label: '语言',         color: 'var(--left)',  cls: 'left'  },
  eye:   { label: '多模态理解',   color: 'var(--eye)',   cls: 'eye'   },
  right: { label: '图像视频生成', color: 'var(--right)', cls: 'right' },
}

export const basicChapters = place([
  { route: 'basic-data', label: '数据与 tokenizer', hint: 'input.txt → train.bin / val.bin / meta.npz' },
  { route: 'basic-forward', label: 'forward 与形状流', hint: 'embedding → block → logits → loss' },
  { route: 'basic-backward', label: '手写 backward', hint: 'cache、链式法则、gradcheck' },
  { route: 'basic-optim-sample', label: 'Adam 与采样', hint: '优化器状态与自回归生成' },
  ...(extraChapters.basic || []),
])

export const modelChapters = place([
  { route: 'attention', label: '注意力演进',     hint: 'MHA → GQA → MLA → DSA' },
  { route: 'position',  label: '位置编码',       hint: 'Sin / Learnable → RoPE → M-RoPE' },
  { route: 'blocks',    label: 'Block 组装器',  hint: '4 个零件 × 几种组合 = 所有主流 LLM' },
  { route: 'moe',       label: 'MoE 路由',       hint: 'Mixtral 与 DeepSeek 的两套哲学' },
  { route: 'models-mtp', label: 'SWA · MTP · 混合线性', hint: 'Mistral 滑动窗口 + 多 token 预测 + Qwen3-Next DeltaNet' },
  { route: 'diffusion', label: '扩散生成',       hint: 'DDPM ε-pred → Flow Matching v-pred' },
  ...(extraChapters.models || []),
])

export const trainChapters = place([
  { route: 'train-batch-ddp', label: 'batch 与 DDP', hint: '梯度累积、数据并行、all-reduce' },
  { route: 'train-model-parallel', label: 'TP / PP 切模型', hint: '层内矩阵切分与层间流水线' },
  { route: 'train-memory', label: '状态与显存', hint: 'ZeRO/FSDP、activation checkpoint、resume' },
  { route: 'train-precision-stability', label: '精度与稳定性', hint: 'FP16 / BF16、loss scaling、FP8 消融、clip' },
  { route: 'train-moe-seq', label: 'EP 与序列并行', hint: 'MoE all-to-all 路由 + Ring Attention' },
  ...(extraChapters.train || []),
  { route: 'train-collectives-loop', label: '通信与 full_loop', hint: 'collectives 到完整训练主循环' },  // 压轴章: 把前面的机制组合起来, 始终排最后
])

export const finetuneChapters = place([
  { route: 'finetune-sft', label: 'SFT 数据与 loss', hint: 'prompt mask 与 response-only CE' },
  { route: 'finetune-lora', label: 'LoRA 参数高效微调', hint: '低秩补丁、注入、merge、QLoRA NF4 量化基座' },
  { route: 'finetune-dpo', label: 'DPO 偏好对齐', hint: 'chosen/rejected 与 reference policy' },
  { route: 'finetune-rlhf', label: 'RM · GRPO · 蒸馏', hint: '奖励模型、组内相对优势、软标签蒸馏' },
  ...(extraChapters.finetune || []),
  { route: 'finetune-runs', label: '选型与落盘', hint: '16 种方法的代价结构: 前向次数、显存、产物' },  // 压轴章: 把前面的机制组合起来, 始终排最后
])

export const inferChapters = place([
  { route: 'infer-kv-memory', label: 'KV 与缓存内存', hint: 'KV cache、paged attention' },
  { route: 'infer-scheduler', label: '调度与 prefill', hint: 'continuous batching、chunked prefill、P/D 分离' },
  { route: 'infer-decode-control', label: '解码加速与约束', hint: 'speculative decoding、sampling 流水线' },
  { route: 'infer-compute', label: '算子与调度开销', hint: 'FlashAttention、CUDA Graph、TP' },
  ...(extraChapters.infer || []),
  { route: 'infer-engine', label: 'mini-vLLM 引擎', hint: 'full_engine 主循环与 Multi-LoRA serving' },  // 压轴章: 把前面的机制组合起来, 始终排最后
])

export const agentChapters = place([
  { route: 'agent-loop', label: 'Agent loop', hint: 'messages → model action → tool result → final' },
  { route: 'agent-tools-permissions', label: '工具与权限', hint: 'tool schema、execute、deny-first、auto approval' },
  { route: 'agent-context-memory', label: '上下文与记忆', hint: 'file memory、retrieval、compaction' },
  { route: 'agent-extensibility', label: 'Hooks / Skills', hint: '确定性扩展点与渐进式披露' },
  { route: 'agent-state-subagents', label: '持久化与子智能体', hint: 'JSONL resume、summary-only delegation' },
  ...(extraChapters.agent || []),
  { route: 'agent-full-loop', label: 'mini Agent harness', hint: '组合成应用层最小闭环' },  // 压轴章: 把前面的机制组合起来, 始终排最后
])

// ─────────────────────────────────────────────────────────────────────────
// 全书主线 — 一个 LLM 工程师从原理到应用层的六个阶段。
// 每个阶段对应仓库根目录下的一个子项目, Web 章节直接对照原始代码。
// ─────────────────────────────────────────────────────────────────────────
export const stages = [
  {
    id: 'basic',
    idx: 1,
    code: 'llm_basic/',
    title: '最小可跑闭环',
    oneliner: 'numpy 手写 forward / backward / Adam / 采样, 看清梯度怎么流过 Transformer。',
    status: 'ready',
    route: 'basic',
    chapters: basicChapters,
    files: ['model.py', 'train.py', 'optim.py', 'sample.py', 'gradcheck.py'],
  },
  {
    id: 'models',
    idx: 2,
    code: 'llm_models/',
    title: '常见模型结构',
    oneliner: '把零件 (attn / ffn / norm / pos) 装进 Pre-LN Block, 堆出二十多种主流模型。',
    status: 'ready',
    route: 'models',
    // 这一阶段的章节见 modelChapters; 总览对照表 (Compare) 放在终章。
    chapters: modelChapters,
  },
  {
    id: 'train',
    idx: 3,
    code: 'llm_train/',
    title: '规模化训练',
    oneliner: '把单机训练循环拆成 batch / 层 / 状态 / 精度 / 容错五条工程主线。',
    status: 'ready',
    route: 'train',
    chapters: trainChapters,
    files: ['m01..m21/demo.py', 'core/collectives.py', 'full_loop/demo.py'],
  },
  {
    id: 'finetune',
    idx: 4,
    code: 'llm_finetune/',
    title: '微调与对齐',
    oneliner: '冻结大部分参数, 用极少量数据把 base model 拨到具体任务/偏好上。',
    status: 'ready',
    route: 'finetune',
    chapters: finetuneChapters,
    // 18 种方法各一个文件, 这里按用途分组列全
    files: [
      'methods/{sft,lora,qlora,dora}.py',
      'methods/{dpo,orpo,simpo,kto}.py',
      'methods/{reward_model,prm,ppo,grpo,rlaif}.py',
      'methods/{distill,on_policy_distill,merge}.py',
      'methods/{is_correction,qat}.py',
    ],
  },
  {
    id: 'infer',
    idx: 5,
    code: 'llm_infer/',
    title: '推理与部署优化',
    oneliner: '从逐 token 重算到 mini-vLLM: 缓存、分页、连续批、前缀复用与采样约束。',
    status: 'ready',
    route: 'infer',
    chapters: inferChapters,
    files: ['m01..m27/demo.py', 'full_engine/engine.py', 'core/tiny_model.py'],
  },
  {
    id: 'agent',
    idx: 6,
    code: 'llm_agent/',
    title: 'Agent 应用层',
    oneliner: '把推理服务接成能行动的系统: 工具、权限、上下文、持久化、扩展与子智能体。',
    status: 'ready',
    route: 'agent',
    chapters: agentChapters,
    files: ['core/agent.py', 'core/tools.py', 'core/permissions.py', 'full_loop/demo.py'],
  },
]

stages.forEach(organizeStage)
export const chapterByRoute = Object.fromEntries(stages.flatMap((s) => s.chapters.map((c) => [c.route, c])))

export const stageBy = Object.fromEntries(stages.map(s => [s.id, s]))

export const learningPath = [
  { route: 'home', label: '主线总览' },
  ...stages.flatMap(s => [
    ...(s.route ? [{ route: s.route, label: `阶段 ${s.idx}.0 · 阶段总览` }] : []),
    ...(s.chapters || []).map(c => ({ route: c.route, label: `${c.number} · ${c.label}`, groupId: c.groupId })),
  ]),
  { route: 'compare', label: '总览对照表' },
]

// 章节正文全部住在 data/topics/<stage>.js, 这里只汇总。
// 这个文件只留结构性数据: 阶段、章节目录、模块表、模型时间轴。
export { topicPages } from './topics/index.js'
