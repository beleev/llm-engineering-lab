<!--
  Orchestrator–workers 记账 (对应 llm_agent/core/subagents.py 的 DelegateTool + agent.py 的并行执行)。
  只讲一件事: 子智能体把"读了什么"留在自己的上下文里, lead 只收摘要 —— 换来干净的主上下文和并行, 代价要用 token 账本看。
-->
<template>
  <LabFrame
    title="Orchestrator–workers — 把调研拆给几个子 agent, 三本账怎么变: lead 上下文 / 总 token / 墙钟时间"
    sub="- 模拟什么: 一个调研任务要读很多长文档, 比较两种做法。「单 agent 全自己读」: 一个 agent 把全部 N×r 份文档一份一份读进自己的上下文。「orchestrator-workers」: 一个 lead agent 把任务拆成 N 个子任务, 用 delegate 工具分给 N 个 worker (子 agent), 每个 worker 在自己独立的上下文里读文档, 做完只交回一条很短的摘要 (harness 硬截断到 200 字符, 约 50 token), lead 再汇总。worker 可以一个接一个串行跑, 也可以用线程池并行跑 (最多 4 个同时)。
- 图上是什么: 一张甘特图, 横轴是墙钟时间 (用户实际要等的秒数, 每次「模型调用 + 工具执行」按 3 秒算)。orchestrator 模式下第一行是 lead, 两段橙色条分别是开头的「扇出」和结尾的「汇总」; 下面每行一个 worker, 紫色条是它从领到任务到交回摘要的这段时间, 长度带一点随机, 条上写着它读了几份。单 agent 模式只有一行, 一条长条从头读到尾。点任意一个 worker 条, 图下方会显示这个 worker 自己读了多少 token 的文档、花了多少 input token, 以及 lead 实际只收到多少。
- 可以做什么: 第一排切换三种做法, 「换一组耗时」重新随机每个 worker 的耗时。四根滑杆: 「子任务数 N」是有几个 worker; 「每个子任务读几份文档」是每个 worker 要读几份, 单 agent 则要读 N×r 份; 「每份文档」是一份文档的 tool_result 有多少 token; 「worker 多探索」模拟真实系统里 worker 常常会自己多翻几份资料, 这个倍数让每个 worker 读 r 乘倍数那么多份, 单 agent 不受它影响。
- 应该看到: 三本账方向不一样。「峰值上下文」是单次模型调用里上下文最大有多少 token, lead 的永远比单 agent 小, 因为长文档都在 worker 那边, lead 只看到摘要。「总 input tokens」是所有模型调用的输入加起来 (lead 加全部 worker): 文档少又短时 orchestrator 更贵 (默认 11.1K 对 9.4K), 因为每个 worker 都要重建一份 system prompt 加任务简报 (800 + 150 token); 文档一多, 单 agent 每一轮都要重发前面读过的全部文档, 账按平方涨, orchestrator 反而便宜。「总 token 相对单 agent」就是这个倍数。「墙钟时间」默认大致打平, 子任务多了并行才拉开差距。每一项下面的灰字是对照组的数, 比它小 10% 以上标绿, 大 10% 以上标红。"
    module="llm_agent/m11"
    run="python -m llm_agent.m11_orchestrator.demo"
    :challenge="{
      ask: '默认状态和 m11 demo 是同一种情形: 3 个子任务, 各读 1 份短文档。先猜: 把「读几份文档」和「每份文档」都拖大, orchestrator 的总 token 会一直比单 agent 多吗? 最后把「worker 多探索」拖到 3×, 再看一次。',
      answer: '不会一直多, 会翻转两次。\n- 活少、文档短 (默认状态): orchestrator 更贵, 11.1K 对 9.4K。每个 worker 都要从零建一份自己的上下文 (system prompt 加工具定义 800 token, 任务简报 150 token), 这份固定开销占了大头。m11 demo 实测是 770 对 413 (右栏), 方向一样。\n- 活一多: 单 agent 的账按平方涨, 因为它每一轮都要把此前读过的全部文档重发一遍 (读第 k 份时前 k-1 份都还在上下文里)。每个 worker 只重发自己那几份, 同样的工作量下总 token 反而更少。\n- 「worker 多探索」拖上去: 总账又反超。真实系统里 worker 会各自多翻几份资料, Anthropic 报告过多智能体系统的 token 用量约为普通聊天的 15 倍。\n不随滑杆翻转的只有一件事: lead 的峰值上下文始终比单 agent 小 (默认 2.2K 对 3.1K), 文档越长差得越多, 因为 lead 永远只收 50 token 的摘要。\n墙钟时间要看活有多少: 默认状态下两边都是 4 轮模型调用, 并行扇出和单 agent 大致打平; 子任务或文档一多, 并行才拉开差距。',
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
      <p class="lab-note">颜色是和每项下面灰字的对照组比: 比它小 10% 以上变绿, 大 10% 以上变红。orchestrator 模式和「单 agent」比, 单 agent 模式和「并行 orchestrator」比。</p>
      <div class="lab-note">
        <p><b>m11 demo 实测的两本账, 不随滑杆变化:</b></p>
        <ul class="pts">
          <li>峰值上下文: lead 247, 单 agent 375。lead 小, 因为长文档只进了 worker 的上下文。</li>
          <li>总输入: 770 (lead 285 + workers 485), 单 agent 413。总账大, 因为每个 worker 都要重建自己的上下文。</li>
        </ul>
        <p>demo 的 system prompt 只有一句话, 所以只有几百 tok。上面的玩具模型按真实 harness 取值 (system prompt + 工具定义 1500 tok), 数字更大, 但方向一样。</p>
      </div>
      <p class="lab-note">delegate 只是一个普通工具。lead 在同一轮发出 N 个 tool_use, agent loop 本来就会并行执行同一轮的多个调用, 扇出自然就有了, 不需要另一套调度器。</p>
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
