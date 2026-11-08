// 阅读分层：冲刺、主干、扩展与速查。数量由实际目录计算，新增章节默认进入扩展。
// 冲刺是主干的子集；主线总览是序章，速查页独立列出。
//
// 分层只是阅读建议, 不改变任何章节的内容或可达性。
// 新增章节默认归为扩展 —— 主线是要守住的, 加内容不该让它变长。

// 冲刺: 每个阶段挑出"不读这页就接不上下一阶段"的那几页
const SPRINT = [
  'basic-forward', 'basic-backward',                      // 形状怎么流、梯度怎么回来
  'attention', 'blocks', 'models-generation',             // 注意力、Block 怎么拼、KV cache
  'train-batch-ddp', 'train-memory',                      // 为什么多卡等价于大 batch、显存花在哪
  'finetune-sft', 'finetune-lora',                        // 怎么教模型做事、怎么只训 1% 参数
  'infer-kv-memory', 'infer-scheduler',                   // 推理的瓶颈、怎么把请求喂进去
  'agent-loop',                                           // 模型怎么变成会行动的系统
]

// 主干: 冲刺 + 课程主线的其余部分
const CORE = [
  ...SPRINT,
  'home',
  'basic', 'basic-data', 'basic-optim-sample',
  'models', 'position', 'moe',
  'train', 'train-model-parallel', 'train-precision-stability', 'train-collectives-loop',
  'train-scaling', 'train-eval',                          // 模型该多大、分数该怎么读
  'finetune', 'finetune-dpo', 'finetune-rlhf', 'finetune-ppo', 'finetune-runs',
  'infer', 'infer-decode-control', 'infer-test-time-compute', 'infer-compute', 'infer-engine',
  'agent', 'agent-tools-permissions', 'agent-context-memory',
  'agent-extensibility', 'agent-state-subagents', 'agent-full-loop',
  'agent-rag', 'agent-prompt-caching',
]

// 工具页: 不属于阅读主线, 需要时来查
const REF = ['compare', 'glossary', 'fast-track']

const sprintSet = new Set(SPRINT)
const coreSet = new Set(CORE)
const refSet = new Set(REF)

export const isSprint = (route) => sprintSet.has(route)
export const tierOf = (route) =>
  refSet.has(route) ? 'ref' : coreSet.has(route) ? 'core' : 'ext'

export const TIER_META = {
  core: { label: '主干', mark: '●', desc: '课程主线, 建议按顺序读完' },
  ext: { label: '扩展', mark: '○', desc: '深水区或分支, 跳过不影响主线' },
  ref: { label: '速查', mark: '◆', desc: '工具页, 需要时再来' },
}

// 阅读档位 —— 速成路线页和侧栏筛选共用
// 从窄到宽排: 冲刺 ⊂ 主干 ⊂ 全部。颜色也按这个顺序由深到浅 (main.css 的 .lv-*)
export const LEVELS = [
  { id: 'sprint', label: '冲刺', hint: '只有一天' },
  { id: 'core', label: '主干', hint: '完整主线' },
  { id: 'all', label: '全部', hint: '一个都不落' },
]

// 某个档位下该不该显示这个路由
export const inLevel = (route, level) =>
  level === 'all' ? true : level === 'core' ? tierOf(route) === 'core' : isSprint(route)


