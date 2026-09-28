<!--
  多副本路由实验台 (对应 llm_infer/m27_multi_replica_routing/router.py:route / simulate)。
  一件事: 送到有前缀的副本能省 prefill, 但热点会把一个副本压垮 —— 负载阈值在两者之间取舍。
-->
<template>
  <LabFrame
    title="多副本路由 — 命中率 vs 负载均衡"
    sub="8 个副本各有一个 24k token 的 RadixCache, 互不共享 KV。1084 个请求来自 200 段多轮对话, 共用 4 个 system prompt。
      - 图 ①: demo 整段模拟的结果, 跟着策略和阈值换。
      - 图 ②: 一个请求到来的那一刻。拖积压柱顶的把手, 看 route() 按当前策略选谁。"
    module="llm_infer/m27"
    run="python -m llm_infer.m27_multi_replica_routing.demo"
    :challenge="{
      ask: '阈值拖到 ∞ (纯前缀感知), 命中率最高 (93.9%)。TTFT 会比最少负载 (0.469 s) 低吗?',
      answer: '不会, 反而是它的 200 多倍: 105.954 s。\n纯前缀感知下, 新对话只在最先缓存它 system prompt 的副本上命中 512 token, 于是永远选它。\n只有 3 个副本在干活 (52% / 32% / 16%), max/mean 4.18, 热点副本的队列越排越长。\nTTFT 105.954 s 是队列模型过载发散的结果, 只说明「会爆」。\n阈值往紧里拖, TTFT 也不会一直降:\n- 1s: 命中 83.6%, TTFT 0.314 s, 比最少负载低 33%。\n- 0.5s: TTFT 最优 0.284 s。\n- 0.05s: 命中跌到 72.9%, TTFT 回升到 0.308 s, 在往最少负载退化。',
    }"
  >
    <template #controls>
      <div class="row">
        <button v-for="(name, p) in POLICY" :key="p" type="button" :class="{ active: policy === p }" :aria-pressed="policy === p" @click="policy = p">{{ name }}</button>
      </div>
      <LabSlider v-if="policy === 'prefix'" v-model="thi" label="负载阈值" :min="0" :max="6" :format="(i) => THR_LABEL[i]" />
    </template>

    <svg ref="svg" viewBox="0 0 560 330" role="group" aria-label="各副本负载与单次路由决策">
      <text x="4" y="12" class="cap">① 实测, 不随图 ② 的拖动变化: 各副本分到的活 (占比), 虚线 = 平均 12.5%</text>
      <g v-for="(s, i) in cfg.share" :key="`s${i}`">
        <rect :x="bx(i)" :y="140 - s * 2.2" width="44" :height="s * 2.2" class="share" />
        <text :x="bx(i) + 22" :y="136 - s * 2.2" class="tick">{{ s }}%</text>
        <text :x="bx(i) + 22" y="152" class="tick">R{{ i }}</text>
      </g>
      <line x1="10" x2="550" :y1="140 - 12.5 * 2.2" :y2="140 - 12.5 * 2.2" class="mean" />

      <text x="4" y="176" class="cap">② 前端现算: 这个请求在各副本上的前缀命中 (token) 与积压 (秒, 可拖)</text>
      <g v-for="(r, i) in REPS" :key="`r${i}`">
        <rect :x="bx(i) - 3" y="182" width="50" height="136" class="pick" :class="{ on: i === choice }" />
        <rect :x="bx(i)" :y="310 - r.hit / 20" width="20" :height="r.hit / 20" class="hitb" />
        <rect :x="bx(i) + 24" :y="310 - backlog[i] * 40" width="20" :height="backlog[i] * 40" class="load" />
        <g class="draggable knob" tabindex="0" role="slider" :aria-label="`副本 R${i} 积压 (秒)`" aria-valuemin="0" aria-valuemax="3" :aria-valuenow="backlog[i]"
          @pointerdown="start($event, { svg, onMove: (e) => drag(i, e) })"
          @keydown.up.prevent="nudge(i, 0.1)" @keydown.down.prevent="nudge(i, -0.1)">
          <rect :x="bx(i) + 21" y="190" width="26" height="120" class="grab" />
          <rect :x="bx(i) + 22" :y="307 - backlog[i] * 40" width="24" height="6" rx="3" class="handle" />
        </g>
        <text :x="bx(i) + 22" y="326" class="tick">{{ r.hit }} · {{ backlog[i].toFixed(1) }}s</text>
      </g>
    </svg>
    <p class="legend"><span class="sw hitb" /> 前缀命中 <span class="sw load" /> 积压 (拖柱顶的把手) · 绿框 = 路由器选中的副本</p>
    <p class="lab-note why">{{ why }}</p>

    <template #stats>
      <div class="kv"><span>{{ cfg.name }}: 命中率</span><b :class="cfg.hit > 80 ? 'good' : ''">{{ cfg.hit }}%</b></div>
      <div class="kv"><span>负载 max / mean</span><b :class="cfg.bal > 2 ? 'bad' : cfg.bal < 1.2 ? 'good' : ''">{{ cfg.bal.toFixed(2) }}</b></div>
      <div class="kv"><span>TTFT 均值 / p90 (秒, 估算)</span><b :class="cfg.ttft > 1 ? 'bad' : cfg.ttft < 0.35 ? 'good' : ''">{{ cfg.ttft.toFixed(3) }} / {{ cfg.p90.toFixed(3) }}</b></div>
      <div class="kv"><span>② 这个请求发往</span><b>R{{ choice }} (命中 {{ REPS[choice].hit }} token)</b></div>
      <div class="lab-note">
        <p>① 的命中率 / max/mean / TTFT 来自 demo 的 [1] [3] 段。各副本占比是用同一个 simulate() 对每种配置跑出的 share。demo 只打印了纯前缀那一行: 52 / 32 / 0 / 0 / 0 / 16 / 0 / 0。</p>
        <p>TTFT 来自 FIFO 代价模型 (prefill 1000 tok/s, decode 100 tok/s), 命中率是真跑 RadixCache 得到的。</p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { useDrag } from '@/composables/useDrag.js'
import { argmax, clamp, range } from '@/utils/labmath.js'

// 数据: python -m llm_infer.m27_multi_replica_routing.demo 的 [1] [3] 段;
// share = router.simulate(chat_stream(), policy, threshold_s=thr)['share'] (四舍五入到整数 %)
const POLICY = { round_robin: '轮询', least_load: '最少负载', prefix: '前缀感知' }
const RR = { name: '轮询', hit: 59.3, bal: 1.03, ttft: 0.507, p90: 1.09, share: [13, 12, 13, 13, 12, 12, 12, 12] }
const LL = { name: '最少负载', hit: 59.3, bal: 1.02, ttft: 0.469, p90: 0.999, share: [13, 13, 12, 13, 12, 12, 12, 13] }
const THR = [Infinity, 4, 2, 1, 0.5, 0.2, 0.05]
const THR_LABEL = ['∞ (纯前缀)', '4 s', '2 s', '1 s', '0.5 s', '0.2 s', '0.05 s']
const PX = [
  { hit: 93.9, bal: 4.18, ttft: 105.954, p90: 300.605, share: [52, 32, 0, 0, 0, 16, 0, 0] },
  { hit: 93.4, bal: 1.45, ttft: 0.689, p90: 2.102, share: [15, 15, 11, 8, 9, 18, 13, 11] },
  { hit: 90.5, bal: 1.15, ttft: 0.461, p90: 1.486, share: [14, 14, 11, 12, 12, 14, 12, 11] },
  { hit: 83.6, bal: 1.19, ttft: 0.314, p90: 0.912, share: [14, 15, 10, 11, 12, 13, 12, 13] },
  { hit: 77.9, bal: 1.12, ttft: 0.284, p90: 0.823, share: [13, 14, 12, 12, 11, 13, 12, 13] },
  { hit: 75.7, bal: 1.07, ttft: 0.285, p90: 0.83, share: [12, 13, 12, 12, 13, 12, 12, 13] },
  { hit: 72.9, bal: 1.12, ttft: 0.308, p90: 0.908, share: [12, 14, 12, 13, 12, 13, 11, 13] },
]
// ② 示意: 一段对话的第 3 轮, R1 缓存了它的整段历史, R0/R5 只缓存了同一个 system prompt (512)
const REPS = [512, 1900, 0, 0, 0, 512, 0, 0].map((hit) => ({ hit }))

const policy = ref('prefix'), thi = ref(3)
const backlog = ref([1.2, 2.4, 0.6, 0.3, 0.8, 0.9, 0.4, 0.5])
const svg = ref(null)
const { start } = useDrag()
const thr = computed(() => (policy.value === 'prefix' ? THR[thi.value] : Infinity))

const cfg = computed(() => {
  if (policy.value === 'round_robin') return RR
  if (policy.value === 'least_load') return LL
  return { name: `前缀感知, 阈值 ${THR_LABEL[thi.value]}`, ...PX[thi.value] }
})

// ★ 与 route() 相同: 命中最长者优先 (平手选积压低的); 它比最闲副本多积压超过阈值, 就改走最闲的
const least = computed(() => argmax(backlog.value.map((b) => -b)))
const best = computed(() => argmax(range(8).map((i) => REPS[i].hit * 1e3 - backlog.value[i])))
const gap = computed(() => backlog.value[best.value] - backlog.value[least.value])
const rr = 5   // round_robin 的请求序号 k: 这个请求是第 k 个, 发往 k mod 8
const choice = computed(() => {
  if (policy.value === 'round_robin') return rr % 8
  if (policy.value === 'least_load') return least.value
  return gap.value > thr.value ? least.value : best.value
})

// 这一次为什么选它: 三种策略各说各的理由
const why = computed(() => {
  if (policy.value === 'round_robin') return `轮询: 这是第 ${rr} 个请求, 发往 ${rr} mod 8 = R${choice.value}。不看命中, 也不看积压。`
  if (policy.value === 'least_load') return `最少负载: 只看积压, 选最闲的 R${least.value} (${backlog.value[least.value].toFixed(1)} s)。命中多少不管。`
  const g = `前缀最长的 R${best.value} 比最闲的 R${least.value} 多积压 ${gap.value.toFixed(1)} s`
  return gap.value > thr.value ? `${g}, 超过阈值 ${THR_LABEL[thi.value]} → 让位给 R${least.value}。` : `${g}, 没超过阈值 ${THR_LABEL[thi.value]} → 仍发往 R${best.value}。`
})

const bx = (i) => 14 + i * 67
const drag = (i, { y }) => { backlog.value[i] = Math.round(clamp((310 - y) / 40, 0, 3) * 10) / 10 }
const nudge = (i, d) => { backlog.value[i] = Math.round(clamp(backlog.value[i] + d, 0, 3) * 10) / 10 }
</script>

<style scoped>
svg { min-width: 540px; touch-action: pan-x pan-y; }
.cap { font-size: 10px; fill: var(--text-muted); }
.tick { font-size: 9px; fill: var(--text-dim); text-anchor: middle; }
.share { fill: var(--accent); opacity: 0.8; }
.mean { stroke: var(--text-muted); stroke-dasharray: 5 4; }
.pick { fill: transparent; stroke: var(--border); rx: 4; }
.pick.on { fill: var(--accent-soft); stroke: var(--left); stroke-width: 2; }
.hitb { fill: var(--accent); }
.load { fill: var(--warn); }
.grab { fill: transparent; }
.knob { cursor: ns-resize; outline: none; }
.handle { fill: var(--text); stroke: var(--bg-card); }
.knob:focus-visible .handle { stroke: var(--accent); stroke-width: 2; }
.legend { font-size: 11px; color: var(--text-dim); margin-top: 6px; }
.sw { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin: 0 4px 0 10px; }
.sw.hitb { background: var(--accent); }
.sw.load { background: var(--warn); }
.why { margin-top: 8px; min-height: 1.7em; }
</style>
