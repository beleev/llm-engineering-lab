<!-- Chunked prefill 实验台 (llm_infer/m06): token 预算 B 是 TBT 与 TTFT 之间的旋钮。代价模型与 Python CostModel 相同。 -->
<template>
  <LabFrame
    title="Chunked prefill — 长 prompt 切块混进 decode, 用 token 预算封顶卡顿"
    sub="- 模拟什么: 推理引擎在每个迭代步对一个 batch 执行一次前向计算。其中 prefill 阶段需一次性并行处理新到达 prompt 的全部 token (受限于计算吞吐, 耗时随 token 数量线性增长); decode 阶段则为每条活跃请求各生成 1 个新 token (受限于显存带宽, 单步耗时近似恒定)。场景设定为当前有 D 个在线用户正在逐 token 解码, 第 3 步时新到达一条长度为 P 个 token 的长 prompt。注: 单步耗时采用线性代价模型估算 (每步 20 ms 固定启动与访存开销加上 0.25 ms 乘以本步处理的 token 总数), 并非真实 GPU 实测。对比两种混批调度策略:
  - 「prefill 优先」: 将整段长 prompt 在单步内一次性算完, 所有活跃 decode 请求被迫暂停一步等待其完成。
  - chunked prefill (分块预填充, 即 Sarathi 方案): 设定单步处理的 token 总预算 B, 每步优先为 D 个活跃 decode 请求各分配 1 个 token 配额, 再用剩余预算切取一小块长 prompt 拼入同一 batch 混合执行, 将长 prefill 拆分为多步消化。
- 图上是什么: 视图由上方迭代步耗时柱状图与下方用户体验时间轴组成:
  - 上方柱状图: 每根柱对应一步模型前向, 柱高代表该步总耗时 (ms)。灰色底座表示每步固定的 20 ms 基础开销, 绿色段代表活跃 decode 请求消耗的配额, 黄色段代表新请求 prefill 消耗的配额; 红色水平虚线标出分块模式下的单步耗时上限 ($20 + 0.25 \times B$)。
  - 下方时间轴: 横轴为真实物理时间 (ms)。每个绿色圆点代表在线 decode 用户收到一个新生成 token 的时刻, 相邻两点的间距即为 token 间延迟 TBT (time between tokens), 红色粗线段标出全程最长的单次卡顿间隔; 黄色横线段标出新到达长请求的首 token 延迟 TTFT (time to first token)。
- 可以做什么: 按面板自上而下可进行以下操作:
  - 点击「prefill 优先 (整段)」或「分块混批 (Sarathi)」切换调度策略。
  - 拖动「token 预算 B」滑杆调整分块混批模式下单步允许处理的最大 token 总数。
  - 拖动「长 prompt 长度 P」与「正在 decode 的用户 D」滑杆改变并发负载规模。
  - 使用步进播放器逐步回放调度过程, 或点击任意一根耗时柱查看该迭代步的批次配额构成。
- 应该看到: 在整段 prefill 模式下会出现一根极高的峰值耗时柱, 对应下方时间轴上一段明显的生成停顿空白 (所有在线 decode 用户被长请求阻塞); 切换至分块混批模式后, 所有耗时柱均被压制在红线阈值以内, 生成停顿消失, 但黄色 prefill 段分散至更多迭代步, 导致新请求的 TTFT 相应拉长。右侧读数并列对比两种策略的指标, 分块模式较整段模式降低 10% 以上时标绿、升高 10% 以上时标红:
  - decode 用户的最大 TBT: 在线生成流中出现的最长单次 token 间隔。
  - 新请求的 TTFT: 长 prompt 从进入系统至产出首个 token 所经历的总等待时间。
  - prefill 切成几块: 长度为 P 的长 prompt 被拆分执行的迭代步数。
  - 第 x 步耗时: 当前选中迭代步的单步前向耗时明细。"
    module="llm_infer/m06"
    run="python -m llm_infer.m06_chunked_prefill.demo"
    :challenge="{
      ask: '将单步 token 预算 B 从 128 拖至 1024 观察读数变化: 是否存在某个预算值 B 能同时使在线用户的最大 TBT 与新请求的 TTFT 均达到最优?',
      answer: '不存在。在默认参数 ($P = 1024$、$D = 4$) 下各档位的实测读数如下:\n- 当 $B = 128$ 时: 最大 TBT 压低至 52 ms, 但新请求 TTFT 增至 445 ms。\n- 当 $B = 1024$ 时: 最大 TBT 升至 276 ms, 新请求 TTFT 缩短至 298 ms。\n- 整段 prefill 基线: 最大 TBT 为 297 ms, 新请求 TTFT 最短, 为 276 ms。\n预算 B 越小, 单步前向耗时的上界 $20 + 0.25 \\cdot B$ 越低, 在线 decode 请求遭遇的最长单步阻塞 (最大 TBT) 随之显著下降; 但这会从两方面拉长新请求的 TTFT:\n- 长 prompt 被切分为更多微块, 每多跨一个迭代步便需额外承担一次 20 ms 的固定启动与权重读取开销。\n- 每个混批迭代步还需优先扣减 D 个活跃 decode 请求占用的配额, 实际留给 prefill 推进的有效容量仅为 $B - D$ 个 token。\n因此新请求需要经历更多迭代步才能完成全部 prefill 并产出首个 token。单步预算 B 本质上是在线流式平滑度 (TBT) 与新请求首字响应速度 (TTFT) 之间的权衡旋钮, 压低其中一项必然推高另一项。在主流生产级推理引擎中, 该控制参数即对应 vLLM 的 max_num_batched_tokens 或 SGLang 的 chunked_prefill_size。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: mode === 'whole' }" @click="mode = 'whole'">prefill 优先 (整段)</button>
        <button type="button" :class="{ active: mode === 'chunk' }" @click="mode = 'chunk'">分块混批 (Sarathi)</button>
      </div>
      <LabSlider v-model="budget" label="token 预算 B" :min="32" :max="1024" :step="32" />
      <LabSlider v-model="P" label="长 prompt 长度 P" :min="256" :max="2048" :step="128" />
      <LabSlider v-model="D" label="正在 decode 的用户 D" :min="1" :max="16" />
      <StepPlayer :stepper="stepper" :label="`第 ${stepper.step.value} 步`" />
    </template>

    <svg class="tl" :viewBox="`0 0 ${W} 264`" role="group" aria-label="每步耗时与 token 到达时间轴">
      <line :x1="0" :x2="W" :y1="yOf(capMs)" :y2="yOf(capMs)" class="cap" />
      <text :x="W - 4" :y="yOf(capMs) - 4" text-anchor="end" class="lbl">{{ mode === 'chunk' ? `预算上限 20 + 0.25×${budget} = ${capMs} ms` : `无上限: 整段 prefill ${capMs} ms` }}</text>
      <!-- 整排柱子只有一个 Tab 停靠点, 左右方向键换步 -->
      <g
        class="bars" tabindex="0" role="slider" aria-label="当前步" aria-valuemin="0" :aria-valuemax="cur.steps.length - 1"
        :aria-valuenow="stepper.step.value" :aria-valuetext="`第 ${stepper.step.value} 步 ${now.ms} ms`"
        @keydown.left.prevent="stepper.prev()" @keydown.right.prevent="stepper.next()"
      >
        <g
          v-for="(s, i) in cur.steps" :key="i" class="bar"
          :class="{ dim: i > stepper.step.value, now: i === stepper.step.value }" @click="stepper.step.value = i"
        >
          <rect :x="i * bw" y="0" :width="bw" height="180" class="bg" />
          <rect :x="i * bw + gap" :y="yOf(20)" :width="bw - 2 * gap" :height="180 - yOf(20)" class="fixed" />
          <rect :x="i * bw + gap" :y="yOf(20 + 0.25 * s.dec)" :width="bw - 2 * gap" :height="yOf(20) - yOf(20 + 0.25 * s.dec)" class="dec" />
          <rect :x="i * bw + gap" :y="yOf(s.ms)" :width="bw - 2 * gap" :height="yOf(20 + 0.25 * s.dec) - yOf(s.ms)" class="pre" />
        </g>
      </g>
      <!-- 时间轴: 横坐标是真实时间 (ms) -->
      <line x1="0" :x2="W" y1="212" y2="212" class="axis" />
      <line :x1="xT(cur.maxGap.from)" :x2="xT(cur.maxGap.to)" y1="212" y2="212" class="maxgap" />
      <text :x="(xT(cur.maxGap.from) + xT(cur.maxGap.to)) / 2" y="229" text-anchor="middle" class="lbl bad">最大 TBT {{ cur.maxTbt.toFixed(0) }} ms</text>
      <circle v-for="(t, i) in cur.emits" :key="i" :cx="xT(t)" cy="212" r="3" class="tok" />
      <line :x1="xT(cur.arriveT)" :x2="xT(cur.firstT)" y1="244" y2="244" class="ttft" />
      <text :x="xT(cur.arriveT)" y="260" class="lbl">新请求 TTFT {{ cur.ttft.toFixed(0) }} ms</text>
      <text x="0" y="200" class="lbl">● decode 用户收到 token 的时刻 (ms)</text>
    </svg>
    <p class="legend">
      <span class="sw fixed" />固定开销 20 ms <span class="sw dec" />decode token <span class="sw pre" />prefill token · 点柱子看这一步
    </p>

    <template #stats>
      <div class="kv head"><span></span><span class="mono">整段 → 分块</span></div>
      <div class="kv"><span>decode 用户的最大 TBT (最长停顿)</span><b :class="cmp(sims.whole.maxTbt, sims.chunk.maxTbt)">{{ sims.whole.maxTbt.toFixed(0) }} → {{ sims.chunk.maxTbt.toFixed(0) }} ms</b></div>
      <div class="kv"><span>新请求的 TTFT (首 token 延迟)</span><b :class="cmp(sims.whole.ttft, sims.chunk.ttft)">{{ sims.whole.ttft.toFixed(0) }} → {{ sims.chunk.ttft.toFixed(0) }} ms</b></div>
      <div class="kv"><span>prefill 切成几块</span><b>1 → {{ sims.chunk.nChunks }}</b></div>
      <div class="kv"><span>第 {{ stepper.step.value }} 步耗时</span><b>{{ now.ms.toFixed(1) }} ms</b></div>
      <div class="lab-note">
        <p>这一步算了 {{ now.dec }} 个 decode token 和 {{ now.pre }} 个 prefill token, 耗时 20 + 0.25 × {{ now.dec + now.pre }} = {{ now.ms.toFixed(2) }} ms。</p>
        <p>分块预填充在数学上等价于整段计算: 在因果掩码约束下, 任意位置的 token 仅关注自身及前序上下文, 前序位置的 KV 表示完全不依赖后续未到达的 token, 因此将长序列拆分为多步递推计算与单步完整前向所得结果严格一致 (demo 脚本实测最大数值差异为 0 或 1e-6 浮点舍入量级)。</p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import StepPlayer from '@/components/lab/StepPlayer.vue'
import { useStepper } from '@/composables/useStepper.js'

const W = 720, WARM = 3, TAIL = 4
const stepMs = (n) => 20 + 0.25 * n // 与 m06 CostModel 相同: decode 访存受限 ≈ 固定开销, prefill 计算受限 ∝ token 数

const mode = ref('chunk')
const budget = ref(128)
const P = ref(1024)
const D = ref(4)

const simulate = (chunked) => {
  const steps = []
  const push = (dec, pre) => steps.push({ dec, pre, ms: stepMs(dec + pre) })
  for (let i = 0; i < WARM; i++) push(D.value, 0)
  let left = P.value
  if (!chunked) { push(0, left); left = 0 } // prefill 优先: 这一步只跑整段 prefill, 所有 decode 停一步
  // ★ 分块: 每步先给每个 decode 用户 1 个 token, 剩下的预算 B − D 才切给 prefill
  while (left > 0) { const c = Math.min(budget.value - D.value, left); push(D.value, c); left -= c }
  const lastPrefill = steps.length - 1
  for (let i = 0; i < TAIL; i++) push(D.value + 1, 0) // 新请求也进入 decode
  let t = 0
  const ends = steps.map((s) => (t += s.ms))
  const emits = ends.filter((_, i) => steps[i].dec > 0) // decode 用户在每个含 decode 的步末收到 1 个 token
  const gaps = emits.slice(1).map((e, i) => ({ from: emits[i], to: e, d: e - emits[i] }))
  const maxGap = gaps.reduce((a, b) => (b.d > a.d ? b : a))
  const arriveT = ends[WARM - 1]
  return { steps, ends, emits, maxGap, maxTbt: maxGap.d, arriveT, firstT: ends[lastPrefill], ttft: ends[lastPrefill] - arriveT,
    total: t, nChunks: lastPrefill - WARM + 1 }
}
const sims = computed(() => ({ whole: simulate(false), chunk: simulate(true) }))
// 颜色跟数值走: 分块比整段低 10% 以上才绿, 高 10% 以上才红, 差不多就不上色
const cmp = (whole, chunk) => (chunk < whole * 0.9 ? 'good' : chunk > whole * 1.1 ? 'bad' : '')
const cur = computed(() => sims.value[mode.value === 'chunk' ? 'chunk' : 'whole'])
const capMs = computed(() => (mode.value === 'chunk' ? stepMs(budget.value) : stepMs(P.value)))

const bw = computed(() => W / cur.value.steps.length)
const gap = computed(() => Math.min(2, bw.value * 0.15))
const yMax = computed(() => Math.max(capMs.value, 60) * 1.15)
const yOf = (ms) => 180 - (ms / yMax.value) * 170
const xT = (t) => (t / cur.value.total) * W

const stepper = useStepper(() => cur.value.steps.length, { interval: 250 })
const now = computed(() => cur.value.steps[stepper.step.value] || cur.value.steps[0])
watch(cur, (c) => { stepper.pause(); stepper.step.value = c.steps.length - 1 }, { immediate: true })
</script>

<style scoped>
.tl { min-width: 560px; } /* 窄屏: 图保持可读, 由 .lab-viz 横向滚动 */
.bars { outline: none; }
.bar { cursor: pointer; }
.bar.dim { opacity: 0.3; }
.bg { fill: transparent; }
.bar.now .bg { fill: var(--accent-soft); }
.bars:focus-visible .bar.now .bg { stroke: var(--accent); }
.fixed { fill: var(--border-strong); background: var(--border-strong); }
.dec { fill: var(--left); background: var(--left); }
.pre { fill: var(--warn); background: var(--warn); }
.cap { stroke: var(--danger); stroke-dasharray: 4 3; }
.axis { stroke: var(--border-strong); }
.maxgap { stroke: var(--danger); stroke-width: 4; }
.ttft { stroke: var(--warn); stroke-width: 3; }
.tok { fill: var(--left); }
.lbl { font-size: 12px; fill: var(--text-muted); }
.lbl.bad { fill: var(--danger); }
.legend { font-size: 11px; color: var(--text-dim); margin-top: 6px; }
.sw { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin: 0 4px 0 10px; }
.kv.head { border: 0; padding: 0; font-size: 11px; color: var(--text-dim); }
</style>
