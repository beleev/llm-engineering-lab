<!-- Chunked prefill 实验台 (llm_infer/m06): token 预算 B 是 TBT 与 TTFT 之间的旋钮。代价模型与 Python CostModel 相同。 -->
<template>
  <LabFrame
    title="Chunked prefill — 长 prompt 切块混进 decode, 用 token 预算封顶卡顿"
    sub="- 模拟什么: 推理引擎每一步前向处理一个 batch。prefill 是把一条新 prompt 的全部 token 一次算完 (算力瓶颈, 耗时随 token 数涨); decode 是给每条正在生成的请求算 1 个新 token (访存瓶颈, 耗时几乎固定)。D 个用户正在逐 token 生成, 第 3 步来了一条 P 个 token 的长 prompt。「prefill 优先」把它整段一步算完, 所有 decode 用户停一步等它。chunked prefill (Sarathi 的做法) 定一个每步 token 预算 B: 先给每个 decode 用户 1 个 token, 剩下的预算切一块 prefill 混进同一个 batch, 长 prompt 分几步算完。耗时用代价模型: 每步 20 ms 固定开销 + 0.25 ms × 这一步的 token 数, 不是实测。
      - 图上是什么: 上面每根柱子是一步前向, 高度是这一步的耗时 (ms): 灰色底是 20 ms 固定开销, 绿色是 decode 的 token, 黄色是 prefill 的 token; 红色虚线是每步耗时的上限 (分块时是 20 + 0.25 × B)。下面的时间轴横坐标是真实时间: 每个绿点是一个 decode 用户收到一个 token 的时刻, 相邻两点的间隔就是 TBT (time between tokens, 用户看到的两个字之间的停顿), 红色粗线标出最长的那一段; 黄色线段是新请求的 TTFT (time to first token, 从到达到吐出第一个 token 的时间)。
      - 可以做什么: 点「prefill 优先 (整段)」或「分块混批 (Sarathi)」切换调度方式; 拖「token 预算 B」改分块时每步最多算几个 token; 拖「长 prompt 长度 P」和「正在 decode 的用户 D」改负载; 用播放器逐步看, 或点一根柱子看那一步的构成。
      - 应该看到: 整段模式下有一根特别高的柱子, 时间轴上对应一个很长的空档 (decode 用户卡住了); 分块模式下柱子都被压在红线以下, 空档消失, 但黄色柱子变多, 新请求的 TTFT 变长。右侧读数把两种模式放在一起比: 「decode 用户的最大 TBT」是时间轴上最长的那个间隔; 「新请求的 TTFT」是长 prompt 等了多久才出第一个 token; 「prefill 切成几块」是 P 个 token 被分成了几步; 「第 x 步耗时」是当前选中那一步的耗时。分块比整段低 10% 以上显示绿色, 高 10% 以上红色。"
    module="llm_infer/m06"
    run="python -m llm_infer.m06_chunked_prefill.demo"
    :challenge="{
      ask: '把 token 预算 B 从 128 拖到 1024。有没有一个 B 能让最大 TBT 和 TTFT 两头都好?',
      answer: '没有。默认 P = 1024、D = 4 时:\n- B = 128: 最大 TBT 52 ms, TTFT 445 ms。\n- B = 1024: 最大 TBT 276 ms, TTFT 298 ms。\n- 整段 prefill: 最大 TBT 297 ms, TTFT 276 ms。\nB 越小, 每步耗时的上限 $20 + 0.25 \\cdot B$ 越低, decode 用户之间最长的等待 (最大 TBT) 就越小。\n代价: 长 prompt 被切成更多块, 每块都要多付一次 20 ms 固定开销, 还要和 D 个 decode 用户分享预算 (每步只有 B − D 个 token 给 prefill), 所以新请求要等更久才出第一个 token, TTFT 变长。\nB 是「decode 卡顿」和「首 token 延迟」之间的旋钮, 压低一头就抬高另一头。真实系统里这个旋钮叫 max_num_batched_tokens (vLLM) 或 chunked_prefill_size (SGLang)。',
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
        <p>分块不改变计算结果: 因果 attention 下每个 token 只看它前面的 token, 前面 token 的 KV 不依赖后面的 token, 所以分几块算和一次算完一样 (demo 实测差 0 或 1e-6 量级)。</p>
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
