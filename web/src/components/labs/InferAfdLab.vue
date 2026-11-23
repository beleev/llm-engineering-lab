<!-- AFD 实验台 (llm_infer/m40): afd.py 代价模型 (粗粒度专家模型) 的 JS 移植 + 一个按资源排队的 ping-pong 时间线。全部是估算。 -->
<template>
  <LabFrame
    title="Attention–FFN 分离 — 注意力卡和 FFN 卡各配多少, micro-batch 切几份"
    sub="- 模拟什么: MoE decode 的每一层拆到两组卡上: 注意力卡只跑 attention, 显存全给 KV; FFN 卡只放专家 (这里固定 16 张)。全局 batch 切成 m 个 micro-batch 在两边之间来回 (ping-pong), 每次过网络都是一次 all-to-all。时间来自 afd.py 的代价模型 (每卡 3.35 TB/s、1.98 PFLOPS, 模型是 48 选 3 的粗粒度专家), 全部是估算。
- 图上是什么: ① 前 3 层的时间线, 四条泳道是注意力卡、注意力→FFN 的网络、FFN 卡、FFN→注意力的网络, 同色块是同一个 micro-batch, 泳道上的空白是气泡; ② 每卡吞吐随注意力卡数的曲线, 圆点是当前配置。
- 可以做什么: 拖「注意力卡数」或点 ② 曲线上的任意位置; 拖「micro-batch 数」和「网络带宽」; 切换上下文长度; 「跳到曲线顶点」把注意力卡数设成当前条件下的最优值。
- 应该看到: 默认 184 张注意力卡、3 个 micro-batch、450 GB/s 时 t_a ≈ t_f, 时间线上两边都没有空白, 每卡吞吐在曲线顶点 (4108 token/s, 与 Python demo 一致)。micro-batch 降到 2 出现气泡; 网络降到 50 GB/s 时网络泳道排满, 曲线的顶点左移。上下文翻到 8K, 顶点右移到约 368 张。"
    module="llm_infer/m40"
    run="python -m llm_infer.m40_af_disaggregation.demo"
    :challenge="{
      ask: '保持 184 张注意力卡和 450 GB/s, 把 micro-batch 从 3 拖到 6。通信藏得更好了, 每卡吞吐会更高吗?',
      answer: '不会, 从 4108 降到 3039 token/s, 低了 26%。\n3 个 micro-batch 时 t_a ≈ t_f ≈ 160 µs, t_c = 46 µs, 满足 m ≥ 2·(1 + t_c/t_f) ≈ 2.6, 通信已经完全藏住。再往下切, 每一份的 token 变少, 但每一份都要把专家权重和注意力权重从显存读一遍: FFN 卡的 t_f 掉到「只读权重」的 99 µs 就不再降了, 一层的周期变成 6 × 109 µs。切得越碎, 读权重的固定开销被重复得越多。\n这就是 MegaScale-Infer 把 micro-batch 数限制在 4 以内的原因: 够藏通信就停, 每一份还要大到能让专家的矩阵乘跑满。',
    }"
  >
    <template #controls>
      <LabSlider v-model="nA" label="注意力卡数 (FFN 卡固定 16)" :min="8" :max="512" :step="8" />
      <LabSlider v-model="mb" label="micro-batch 数" :min="1" :max="6" />
      <LabSlider v-model="net" label="每卡网络带宽" :min="50" :max="450" :step="50" unit=" GB/s" />
      <div class="row">
        <button v-for="c in CTXS" :key="c" type="button" :class="{ active: ctx === c }" @click="ctx = c">上下文 {{ c / 1024 }}K</button>
        <button type="button" @click="nA = best.n">跳到曲线顶点</button>
      </div>
    </template>

    <svg viewBox="0 0 560 300" role="group" aria-label="ping-pong 时间线与每卡吞吐曲线">
      <text x="4" y="12" class="cap">① 前 3 层的时间线 (共 {{ (gantt.end * 1e6).toFixed(0) }} µs; 同色 = 同一个 micro-batch)</text>
      <g v-for="(lane, li) in LANES" :key="lane">
        <text x="4" :y="34 + li * 24" class="lbl">{{ lane }}</text>
        <rect x="70" :y="22 + li * 24" width="486" height="18" class="lane" />
      </g>
      <rect v-for="(s, i) in gantt.segs" :key="`s${i}`" :x="70 + s.t0 * gs" :y="22 + s.lane * 24" :width="Math.max(1, (s.t1 - s.t0) * gs - 0.5)" height="18" :class="`mb${s.mb % 6}`" />
      <text x="4" y="138" class="cap">② 每卡吞吐 (token/s) 随注意力卡数; 点曲线换配置</text>
      <polyline :points="curve.map((p) => `${cx(p.n)},${cy(p.tgs)}`).join(' ')" class="curve" />
      <rect v-for="p in curve" :key="`h${p.n}`" :x="cx(p.n) - 4" y="146" width="8" height="140" class="hit" @click="nA = p.n" />
      <circle :cx="cx(nA)" :cy="cy(cur.tgs)" r="5" class="dot" />
      <text v-for="n in [8, 128, 256, 384, 512]" :key="`t${n}`" :x="cx(n)" y="298" class="tick">{{ n }}</text>
    </svg>

    <template #stats>
      <div class="kv"><span>t_a / t_f / t_c (每层每份, µs)</span><b>{{ us(cur.t_a) }} / {{ us(cur.t_f) }} / {{ us(cur.t_c) }}</b></div>
      <div class="kv"><span>通信藏住了吗 (需要 m ≥ {{ need.toFixed(1) }})</span><b :class="cur.hidden ? 'good' : 'bad'">{{ cur.hidden ? '藏住' : mb === 1 ? '不重叠' : '有气泡' }}</b></div>
      <div class="kv"><span>TPOT (估算)</span><b :class="cur.tpot <= 0.05 ? 'good' : 'bad'">{{ (cur.tpot * 1e3).toFixed(1) }} ms</b></div>
      <div class="kv"><span>每卡吞吐 (曲线顶点 {{ best.tgs.toFixed(0) }} @ {{ best.n }} 张)</span><b :class="cur.tgs > 0.95 * best.tgs ? 'good' : cur.tgs < 0.7 * best.tgs ? 'bad' : ''">{{ cur.tgs.toFixed(0) }} token/s</b></div>
      <div class="lab-note">
        <p>Python demo [4] 的结论 (实测, 不随左侧变化; 也是估算): 8 个场景里 AFD 比 16 卡同卡部署快 1.25~4.0 倍, 但比调好的大 EP 同卡部署只有 0.68~0.97 倍; 短上下文 + 50 GB/s 网络时最差。</p>
        <p>每张注意力卡塞满显存: 4K 上下文 125 条序列, 32K 只有 16 条。</p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'

// afd.py: HW() 与 COARSE (48 选 3, 每专家 3×7168×5120 参数), FP8 1 字节
const BW = 3.35e12, FLOPS = 1.98e15, MEM = 72e9
const L = 58, DM = 7168, E = 48, TOPK = 3, PE = 3 * 7168 * 5120, WA = 0.19e9, KVB = 2048, NF = 16
const CTXS = [4096, 8192, 32768], LANES = ['注意力卡', '网络 A→F', 'FFN 卡', '网络 F→A']

const nA = ref(184), mb = ref(3), net = ref(450), ctx = ref(4096)

function afd(n_a, m) {
  const b = (MEM - L * WA) / (ctx.value * KVB * L)                    // 注意力卡: 显存全给 KV
  const tokens = n_a * b / m                                          // 一个 micro-batch 的 token 总数
  const t_a = (WA + (b / m) * ctx.value * KVB) / BW
  const t_f = Math.max((E / NF) * PE / BW, (2 * PE * tokens * TOPK) / NF / FLOPS)   // ★ 读权重 vs 算, 取慢的
  const t_c = Math.max((b / m) * TOPK, (tokens * TOPK) / NF) * DM * 2 / (net.value * 1e9)
  const cycle = Math.max(t_a + t_f + 2 * t_c, m * t_a, m * t_f, m * t_c)
  const tpot = L * cycle
  return { t_a, t_f, t_c, tpot, tgs: n_a * b / tpot / (n_a + NF), hidden: m * Math.max(t_a, t_f) >= Math.max(t_a + t_f + 2 * t_c, m * t_c) }
}
const cur = computed(() => afd(nA.value, mb.value))
const need = computed(() => 2 * (1 + cur.value.t_c / Math.max(cur.value.t_a, cur.value.t_f)))
const curve = computed(() => Array.from({ length: 64 }, (_, i) => ({ n: 8 + i * 8, tgs: afd(8 + i * 8, mb.value).tgs })))
const best = computed(() => curve.value.reduce((a, p) => (p.tgs > a.tgs ? p : a)))

// 按资源排队的 ping-pong: 每份依次占用 注意力 → 网络 → FFN → 网络, 下一层从回来的时刻开始
const gantt = computed(() => {
  const { t_a, t_f, t_c } = cur.value, dur = [t_a, t_c, t_f, t_c]
  const free = [0, 0, 0, 0], ready = Array(mb.value).fill(0), segs = []
  for (let layer = 0; layer < 3; layer++) {
    for (let i = 0; i < mb.value; i++) {
      let t = ready[i]
      for (let lane = 0; lane < 4; lane++) {
        const t0 = Math.max(free[lane], t)
        free[lane] = t = t0 + dur[lane]
        segs.push({ lane, mb: i, t0, t1: t })
      }
      ready[i] = t
    }
  }
  return { segs, end: Math.max(...free) }
})
const gs = computed(() => 486 / gantt.value.end)
const maxTgs = computed(() => Math.max(...curve.value.map((p) => p.tgs)) * 1.1)
const cx = (n) => 70 + ((n - 8) / 504) * 480
const cy = (v) => 284 - (v / maxTgs.value) * 136
const us = (s) => (s * 1e6).toFixed(0)
</script>

<style scoped>
svg { min-width: 540px; }
.cap { font-size: 10px; fill: var(--text-muted); }
.lbl, .tick { font-size: 9px; fill: var(--text-dim); }
.tick { text-anchor: middle; }
.lane { fill: none; stroke: var(--border); }
.curve { fill: none; stroke: var(--accent); stroke-width: 2; }
.hit { fill: transparent; cursor: pointer; }
.dot { fill: var(--warn); }
.mb0 { fill: var(--accent); }
.mb1 { fill: var(--left); }
.mb2 { fill: var(--eye); }
.mb3 { fill: var(--right); }
.mb4 { fill: var(--warn); }
.mb5 { fill: var(--danger); }
</style>
