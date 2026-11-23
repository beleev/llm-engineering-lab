<!--
  Orchestrator–workers 记账 (对应 llm_agent/core/subagents.py 的 DelegateTool + agent.py 的并行执行)。
  只讲一件事: 子智能体把"读了什么"留在自己的上下文里, lead 只收摘要 —— 换来干净的主上下文和并行, 代价要用 token 账本看。
-->
<template>
  <LabFrame
    title="Orchestrator–workers — 把调研拆给几个子 agent, 三本账怎么变: lead 上下文 / 总 token / 墙钟时间"
    sub="- 模拟什么: 针对需阅读大量长文档的调研任务, 对比两种多步执行架构的资源开销:
  - 「单 agent 全自己读」: 由单个 agent 将全部 $N \times r$ 份文档逐一读入同一个上下文窗口。
  - 「orchestrator-workers」: 由顶层 lead agent 将任务拆解为 $N$ 个独立子任务, 通过 delegate 工具分发给 $N$ 个 worker (子 agent); 每个 worker 在各自隔离的上下文中阅读文档, 完成后仅向上游返回一条精简摘要 (由 harness 硬截断至 200 字符, 约 50 token), 最后由 lead 统一汇总。worker 既可逐个串行执行, 也可通过线程池并行执行 (并发上限为 4)。
- 图上是什么: 执行时间甘特图, 横轴表示墙钟时间 (即用户实际等待秒数, 每次「模型调用 + 工具执行」按 3 秒折算):
  - orchestrator 模式: 首行为 lead 进程, 两段橙色时间条分别代表起始的「扇出」分发与末尾的「汇总」生成; 下方每行对应一个 worker, 紫色时间条表示该 worker 从接收任务到返回摘要的持续时间 (含轻微随机抖动), 条内标注其实际阅读的文档份数。
  - 单 agent 模式: 仅有一行连续长条, 从头至尾串行读完所有文档。
  - 点击任意 worker 时间条: 图表下方会展示该 worker 自身读取的文档总 token 数、累计消耗的 input token 数, 以及最终回传给 lead 的摘要 token 数。
- 可以做什么: 第一排切换三种执行模式, 点击「换一组耗时」可重新采样各 worker 的随机耗时; 下方提供四根参数滑杆:
  - 「子任务数 N」: 设定拆分的子任务及对应 worker 数量。
  - 「每个子任务读几份文档」: 设定单个 worker 需阅读的文档数 $r$, 单 agent 模式下则需顺序阅读全部 $N \times r$ 份。
  - 「每份文档」: 设定单份文档作为 tool_result 返回时的 token 长度。
  - 「worker 多探索」: 模拟真实场景下子智能体自主扩充检索范围的行为, 使每个 worker 实际阅读文档数增至 $r$ 乘以该倍率, 单 agent 基线不受此参数影响。
- 应该看到: 峰值窗口、累计成本与等待延迟三项指标呈现不同的变化规律。右侧读数含义如下:
  - 「峰值上下文」: 单次模型调用所承受的最大上下文长度。由于长文档原文被隔离在各 worker 内部、lead 仅接收短摘要, lead 的峰值上下文始终低于单 agent。
  - 「总 input tokens」: 全部模型调用的输入 token 累加和 (含 lead 与所有 worker)。当文档数量少且篇幅短时, orchestrator 总消耗反而更高 (默认配置下为 11.1K 对 9.4K), 原因是每个 worker 均需独立携带一份基础开销 (system prompt 与工具定义 800 token 加任务简报 150 token); 随着文档增多变长, 单 agent 每轮都必须重发此前读过的全部历史文档, 累计成本按轮数平方级增长, orchestrator 架构反而更节省总 token。
  - 「总 token 相对单 agent」: 当前架构总输入 token 与单 agent 基线的比值。
  - 「墙钟时间」: 默认小规模配置下两者耗时大致持平, 当子任务规模扩大后并行扇出才能显著缩短等待时间。
  - 读数下方灰字: 显示对照组基线数值, 当前值较基线降低 10% 以上时标绿, 高出 10% 以上时标红。"
    module="llm_agent/m11"
    run="python -m llm_agent.m11_orchestrator.demo"
    :challenge="{
      ask: '默认配置对应 m11 demo 的小规模场景: 3 个子任务, 每个子任务仅读取 1 份短文档。先预测: 若将「每个子任务读几份文档」与「每份文档」长度同时调大, orchestrator 的总 token 消耗会始终高于单 agent 吗? 随后再将「worker 多探索」调至 3× 观察结果变化。',
      answer: '不会始终高于单 agent, 大小关系会发生两次翻转。\n- 任务量少且文档较短时 (默认状态): orchestrator 总开销更高, 为 11.1K 对 9.4K。原因是每个 worker 均需从零构建独立上下文 (system prompt 与工具定义占 800 token, 任务简报占 150 token), 固定启动开销占据主导。m11 demo 实测数据为 770 对 413 (见右栏), 相对关系一致。\n- 文档数量与长度增加后: 单 agent 的累计成本呈平方级增长, 因为每推进一轮都必须将此前读过的全部文档重新发送一遍 (读取第 $k$ 份文档时, 前 $k-1$ 份文档仍驻留在上下文中)。而在多智能体架构下, 每个 worker 仅重发各自负责的少量文档, 相同总阅读量下累计消耗反而显著低于单 agent。\n- 将「worker 多探索」调高后: orchestrator 总开销再次反超。真实场景中各 worker 往往会自主追加检索轮次, Anthropic 曾披露生产级多智能体系统的平均 token 消耗约为普通单轮对话的 15 倍。\n全程不随参数翻转的指标只有一项: lead 的峰值上下文始终小于单 agent (默认配置下为 2.2K 对 3.1K), 且文档越长差距越显著, 因为 lead 每轮仅接收硬截断后的 50 token 摘要。\n墙钟时间则取决于并发规模: 默认状态下两种模式均经历 4 轮模型调用深度, 并行扇出与单 agent 耗时相近; 当子任务数量或单任务阅读轮数扩大后, 并行执行才能明显拉开速度差距。',
    }"
  >
    <template #controls>
      <div class="row">
        <button v-for="m in MODES" :key="m.id" type="button" :class="{ active: mode === m.id }" @click="mode = m.id">{{ m.label }}</button>
        <button type="button" @click="seed++">换一组耗时</button>
      </div>
      <LabSlider v-model="n" label="子任务数 N" :min="1" :max="8" />
      <LabSlider v-model="r" label="每个子任务读几份文档" :min="1" :max="5" />
      <LabSlider v-model="t" label="每份文档的长度 (tool_result)" :min="500" :max="8000" :step="500" unit=" tok" />
      <LabSlider v-model="explore" label="worker 多探索 (实际读几倍的文档)" :min="1" :max="3" :step="0.5" unit="×" />
    </template>

    <div class="gantt">
      <div v-for="row in plan.rows" :key="row.id" class="g-row">
        <span class="mono g-lbl">{{ row.label }}</span>
        <div class="g-track">
          <button
            v-for="(b, i) in row.bars" :key="i" type="button" class="g-bar" :class="[b.kind, { sel: sel === b.worker && b.worker != null }]"
            :style="{ left: b.start / plan.span * 100 + '%', width: b.dur / plan.span * 100 + '%' }"
            :title="`${b.text} · ${b.dur.toFixed(0)}s`" @click="sel = b.worker ?? sel"
          >{{ b.text }}</button>
        </div>
      </div>
      <p class="axis mono">0s ─── 墙钟时间 ─── {{ plan.span.toFixed(0) }}s (线程池 max_parallel = 4)</p>
    </div>
    <p v-if="mode !== 'single'" class="detail">
      <b>worker {{ selW + 1 }}</b> 在自己的上下文里读了 <b class="mono">{{ fmtNum(acct.workerRead) }}</b> tok 文档、花了 <b class="mono">{{ fmtNum(acct.workerInput) }}</b> input tok;
      lead 只收到一条 <b class="mono">{{ S }}</b> tok 的摘要 tool_result (硬上限 200 字符)。完整 transcript 落盘在 child_{{ String(selW).padStart(2, '0') }}.jsonl, 可审计, 但不回流。
    </p>
    <p v-else class="detail">单 agent: {{ n * r }} 份文档全部堆在同一个上下文里, 每多一轮都要把前面读过的再发一遍。</p>

    <template #stats>
      <div class="kv"><span>峰值上下文 (单次模型调用最大) <em>对照: {{ vs.name }} {{ fmtNum(vs.peak) }}</em></span><b :class="cmp(acct.peak, vs.peak)">{{ fmtNum(acct.peak) }}</b></div>
      <div class="kv"><span>总 input tokens (lead + 全部 worker) <em>对照: {{ vs.name }} {{ fmtNum(vs.total) }}</em></span><b :class="cmp(acct.total, vs.total)">{{ fmtNum(acct.total) }}</b></div>
      <div class="kv"><span>总 token 相对单 agent</span><b>{{ (acct.total / acct.single.total).toFixed(2) }}×</b></div>
      <div class="kv"><span>墙钟时间 (用户实际等多久) <em>对照: {{ vs.name }} {{ vs.wall.toFixed(1) }}s</em></span><b :class="cmp(plan.wall, vs.wall)">{{ plan.wall.toFixed(1) }}s</b></div>
      <p class="lab-note">指标颜色均以每项下方灰字所示的对照组为基准: 较对照组低 10% 以上标绿, 高 10% 以上标红。处于 orchestrator 模式时以「单 agent」为对照, 处于单 agent 模式时则以「并行 orchestrator」为对照。</p>
      <div class="lab-note">
        <p><b>m11 demo 实测的两本账, 不随滑杆变化:</b></p>
        <ul class="pts">
          <li>峰值上下文: lead 247, 单 agent 375。lead 小, 因为长文档只进了 worker 的上下文。</li>
          <li>总输入: 770 (lead 285 + workers 485), 单 agent 413。总账大, 因为每个 worker 都要重建自己的上下文。</li>
        </ul>
        <p>代码 demo 中的 system prompt 仅有一句话, 总消耗仅数百 token; 上方代价模型按真实工程 harness 规模设定基础开销 (system prompt 加工具定义共 1500 token), 绝对读数更大, 但成本翻转规律完全一致。</p>
      </div>
      <p class="lab-note">delegate 在实现上只是一个标准工具。当 lead 在同一轮回复中并发输出 $N$ 个 tool_use 时, 底层 agent loop 原生支持并行执行同一轮内的多个工具调用, 无需额外引入专用调度器即可实现任务扇出。</p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { fmtNum, mulberry32, range, sum } from '@/utils/labmath.js'

const MODES = [{ id: 'single', label: '单 agent 全自己读' }, { id: 'serial', label: 'orchestrator · worker 串行' }, { id: 'parallel', label: 'orchestrator · worker 并行' }]
const mode = ref('parallel'), n = ref(3), r = ref(1), t = ref(500), explore = ref(1), seed = ref(1), sel = ref(0)
const selW = computed(() => Math.min(sel.value, n.value - 1))

// 固定开销 (token): 主 system+工具定义 / 子 system+工具定义 / 用户问题 / 每次 tool_use / 任务简报 / 摘要
// S = 50: DelegateTool.max_summary_chars = 200 字符, 按 4 字符/token 粗估
const B = 1500, BW = 800, Q = 50, U = 30, BRIEF = 150, S = 50
const TURN = 3, POOL = 4 // 每轮 (模型调用+工具) 约 3s; 线程池 4
// ★ 一个 agent 连续读 k 份文档: 第 j 次调用要重发此前全部 j 份 → input 之和是 k 的平方级
const session = (base, k) => ({ total: sum(range(k + 1).map((j) => base + j * (t.value + U))), peak: base + k * (t.value + U) })

const acct = computed(() => {
  const single = session(B + Q, n.value * r.value)
  const kw = Math.round(r.value * explore.value)
  const worker = session(BW + BRIEF, kw)
  const leadPeak = B + Q + n.value * (U + BRIEF + S)
  const lead = (B + Q) + leadPeak // 两次调用: 扇出 + 汇总
  const orch = { total: lead + n.value * worker.total, peak: leadPeak }
  const cur = mode.value === 'single' ? single : orch
  return { ...cur, single, orch, workerInput: worker.total, workerRead: kw * t.value, kw }
})

const plan = computed(() => {
  const rand = mulberry32(seed.value * 97 + 3)
  const kw = acct.value.kw
  const durs = range(n.value).map(() => (kw + 1) * TURN * (0.7 + 0.6 * rand()))
  // 线程池: 谁先空闲谁接下一个 (pool.map 的效果); 串行 = 池大小 1
  const fan = (pool) => {
    const free = Array(pool).fill(TURN)
    const rows = durs.map((d, i) => {
      const k = free.indexOf(Math.min(...free)), start = free[k]
      free[k] = start + d
      return { id: i, label: `worker ${i + 1}`, bars: [{ kind: 'worker', worker: i, start, dur: d, text: `读 ${kw} 份 → 摘要` }] }
    })
    const end = Math.max(...free)
    rows.unshift({ id: 'lead', label: 'lead', bars: [{ kind: 'lead', start: 0, dur: TURN, text: '扇出' }, { kind: 'lead', start: end, dur: TURN, text: '汇总' }] })
    return { rows, wall: end + TURN }
  }
  const d = (n.value * r.value + 1) * TURN
  const solo = { rows: [{ id: 'a', label: 'agent', bars: [{ kind: 'lead', start: 0, dur: d, text: `${n.value * r.value} 份文档 + final` }] }], wall: d }
  const par = fan(POOL), ser = fan(1)
  const cur = mode.value === 'single' ? solo : mode.value === 'serial' ? ser : par
  return { ...cur, span: Math.max(d, ser.wall), solo: d, par: par.wall }
})

// 对照: orchestrator 模式和单 agent 比; 单 agent 模式和并行 orchestrator 比
const vs = computed(() => (mode.value === 'single'
  ? { name: '并行 orchestrator', peak: acct.value.orch.peak, total: acct.value.orch.total, wall: plan.value.par }
  : { name: '单 agent', peak: acct.value.single.peak, total: acct.value.single.total, wall: plan.value.solo }))
const cmp = (a, b) => (a < b * 0.9 ? 'good' : a > b * 1.1 ? 'bad' : '')
</script>

<style scoped>
.gantt { display: flex; flex-direction: column; gap: 4px; min-width: 420px; }
.g-row { display: grid; grid-template-columns: 64px 1fr; gap: 8px; align-items: center; }
.g-lbl { font-size: 11px; color: var(--text-dim); }
.g-track { position: relative; height: 24px; border-bottom: 1px dashed var(--border); }
.g-bar { position: absolute; top: 2px; height: 20px; min-height: 0; padding: 0 4px; font-size: 10px; line-height: 18px; overflow: hidden; white-space: nowrap; border-radius: 3px; border: 1px solid var(--accent); background: var(--accent-soft); color: var(--text); }
.g-bar.lead { border-color: var(--eye); background: color-mix(in srgb, var(--eye) 22%, transparent); cursor: default; }
.g-bar.sel { outline: 2px solid var(--warn); outline-offset: 1px; }
.axis { font-size: 10px; color: var(--text-dim); margin-top: 4px; }
.detail { margin-top: 12px; font-size: 13px; color: var(--text-muted); line-height: 1.7; }
.detail b { color: var(--text); font-weight: 500; }
.kv em { display: block; font-style: normal; font-size: 10.5px; color: var(--text-dim); }
</style>
