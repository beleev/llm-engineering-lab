<!--
  MoE 路由实验台 (对应 llm_train/m11_expert_parallel/demo.py)。
  只讲一件事: router 选谁决定了每个专家收多少 token; 超过 capacity 的被丢弃, 最热的卡决定 step 时间。
  均衡用 demo 里 balance_bias 的办法: 选专家时加一个 bias, 过载减、欠载加。
  与 demo 的差别: 步长逐步衰减并在达到均衡时停下。固定步长在 64 个 token 上会绕着均衡点来回跳。
-->
<template>
  <LabFrame
    title="MoE 路由实验台 — 倾斜、溢出与 all-to-all"
    sub="64 个 token 经 top-1 路由发往 E 个专家 (均分在 D 张卡上)。路由越倾斜,
      热点卡越忙、容量溢出丢的 token 越多。红色的一段是被丢弃的 token。点一根柱子看这个专家的明细。"
    module="llm_train/m11"
    run="python -m llm_train.m11_expert_parallel.demo"
    :challenge="{
      ask: '默认状态丢了 4 个 token。三件事各猜一次, 每次只动一个控件: skew 拖到 0 会不会归零? cf 拖到 2 呢? 打开「bias 均衡」呢?',
      answer: '- skew = 0: 仍丢 2 个。router 不偏心时是纯随机路由, 总有专家碰巧收到 12 个, 超过 capacity 10。\n- cf = 2: 丢弃归零。代价是显存和计算按最坏情况 padding, 开销是两倍。\n- bias 均衡: 任何 skew 下每个专家都正好 8 个, 丢弃 0, 最热的卡收 16 个。它只改「选谁」, gate 权重不变。\ndemo 实测: 倾斜 router 是 1.62x、丢 5 个; bias 均衡后 1.00x、丢 0 个; aux loss 训练 60 步后 1.12x、丢 0 个。',
    }"
  >
    <template #controls>
      <div class="row">
        <button v-for="d in [2, 4]" :key="d" type="button" :class="{ active: D === d }" @click="D = d">D = {{ d }} 卡</button>
        <button type="button" :class="{ active: balanced }" :aria-pressed="balanced" @click="balanced = !balanced">
          bias 均衡 (aux-loss-free): {{ balanced ? '开' : '关' }}
        </button>
        <button type="button" @click="seed++">换一组 token</button>
      </div>
      <LabSlider v-model="E" label="专家数 E" :min="4" :max="16" :step="4" />
      <LabSlider v-model="cf" label="容量因子 cf" :min="1" :max="2" :step="0.25" :format="(v) => v.toFixed(2)" />
      <LabSlider v-model="skew" label="路由倾斜度 skew" :min="0" :max="3" :step="0.1" :format="(v) => v.toFixed(1)" />
    </template>

    <div class="chart-wrap">
      <div class="chart">
        <div class="cap-line" :style="{ bottom: hPct(capacity) }">
          <span class="mono">capacity = {{ capacity }}</span>
        </div>
        <button
          v-for="(c, e) in route.counts" :key="e" type="button" class="col" :class="{ sel: e === pick }"
          :aria-label="`专家 ${e}: 收到 ${c} 个 token`" @click="pick = e"
        >
          <span class="seg over" :style="{ height: hPct(Math.max(0, c - capacity)) }" />
          <span class="seg" :style="{ height: hPct(Math.min(c, capacity)), background: devColors[devOf(e)] }" />
        </button>
      </div>
      <div class="xlabs">
        <span v-for="e in E" :key="e" class="mono">e{{ e - 1 }}</span>
      </div>
    </div>
    <p class="lab-note" style="margin-top: 10px;">
      专家 e{{ sel.e }} 在卡 {{ sel.dev }}: 收到 {{ sel.c }} 个 token, 处理 {{ sel.c - sel.over }} 个, 丢弃 {{ sel.over }} 个 (只走残差)。
    </p>

    <template #stats>
      <div class="kv"><span>不均衡度 max/mean</span><b :class="st.imb < 1.2 ? 'good' : 'bad'">{{ st.imb.toFixed(2) }}x</b></div>
      <div class="kv"><span>容量溢出丢弃的 token</span><b :class="st.dropped ? 'bad' : 'good'">{{ st.dropped }} / {{ N }}</b></div>
      <div class="kv"><span>最热的卡收到 (均匀应为 {{ N / D }})</span><b :class="st.hot > st.hotOk ? 'bad' : 'good'">{{ st.hot }}</b></div>
      <div class="a2a">
        <span>all-to-all 发送矩阵 (行 = 源卡, 列 = 目标卡)</span>
        <div class="a2a-grid" :style="{ gridTemplateColumns: `repeat(${D}, 36px)` }">
          <template v-for="(row, s) in dispatch" :key="s">
            <span v-for="(v, t) in row" :key="t" class="a2a-cell mono" :style="{ background: heat((v / st.cellMax) * 0.8) }">{{ v }}</span>
          </template>
        </div>
      </div>
      <div class="legend">
        <span v-for="d in D" :key="d" class="leg mono"><i :style="{ background: devColors[d - 1] }" />卡{{ d - 1 }}</span>
      </div>
      <p class="lab-note">
        bias 均衡是 DeepSeek-V3 的做法: 选专家用 s_e + b_e, 过载的专家 b_e 减一点, 欠载的加一点, 重复几百步。
        另一条路是 Switch aux loss: 加一项 loss 去训练 router。
      </p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { heat, mulberry32, range, sum } from '@/utils/labmath.js'

const N = 64
const D = ref(4)
const E = ref(8)
const cf = ref(1.25)
const skew = ref(0.2)
const balanced = ref(false)
const seed = ref(1)
const pick = ref(0)
const devColors = ['var(--accent)', 'var(--left)', 'var(--eye)', 'var(--right)']
watch(E, (e) => { if (pick.value >= e) pick.value = 0 })

const devOf = (e) => Math.floor(e / (E.value / D.value))

// 路由打分: score_e = skew·lin_e + noise, lin 从 +1 线性递减到 -1 (专家 0 天然最热)
const scores = computed(() => range(N).map((n) => {
  const rand = mulberry32(seed.value * 7919 + n * 1000 + 7)
  return range(E.value).map((e) => skew.value * (1 - (2 * e) / (E.value - 1)) + (rand() - 0.5) * 2.5)
}))
const choose = (bias) => scores.value.map((row) => row.reduce((b, s, e) => (s + bias[e] > row[b] + bias[b] ? e : b), 0))
const load = (expertOf) => { const c = Array(E.value).fill(0); expertOf.forEach((e) => c[e]++); return c }

// ★ 同 demo 的 balance_bias: b_e += γ·sign(mean − load_e)。bias 只影响「选谁」
const bias = computed(() => {
  const b = Array(E.value).fill(0), mean = N / E.value
  if (!balanced.value) return b
  let best = b.slice(), bestMax = Infinity
  for (let i = 0, gamma = 0.05; i < 1000; i++, gamma *= 0.99) {
    const c = load(choose(b)), mx = Math.max(...c)
    if (mx < bestMax) { bestMax = mx; best = b.slice() }
    if (mx <= Math.ceil(mean)) break
    c.forEach((v, e) => { b[e] += gamma * Math.sign(mean - v) })
  }
  return best
})
const route = computed(() => { const expertOf = choose(bias.value); return { expertOf, counts: load(expertOf) } })

const capacity = computed(() => Math.ceil((N / E.value) * cf.value))
// 发送矩阵: token n 的源卡 = floor(n / (N/D)), 目标卡 = 其专家所在卡
const dispatch = computed(() => {
  const M = range(D.value).map(() => Array(D.value).fill(0))
  route.value.expertOf.forEach((e, n) => { M[Math.floor(n / (N / D.value))][devOf(e)] += 1 })
  return M
})
const st = computed(() => ({
  imb: Math.max(...route.value.counts) / (N / E.value),
  dropped: sum(route.value.counts.map((c) => Math.max(0, c - capacity.value))),
  hot: Math.max(...range(D.value).map((d) => sum(dispatch.value.map((r) => r[d])))),
  hotOk: Math.ceil(N / E.value) * (E.value / D.value),   // 每个专家都不超过 ⌈mean⌉ 时, 一张卡最多收这么多
  cellMax: Math.max(1, ...dispatch.value.flat()),
}))
const sel = computed(() => {
  const c = route.value.counts[pick.value]
  return { e: pick.value, dev: devOf(pick.value), c, over: Math.max(0, c - capacity.value) }
})

const scaleMax = computed(() => Math.max(capacity.value, ...route.value.counts) * 1.15)
const hPct = (v) => `${((v / scaleMax.value) * 100).toFixed(1)}%`
</script>

<style scoped>
.chart-wrap { background: var(--code-bg); border-radius: var(--radius-sm); padding: 10px; }
.chart { position: relative; height: 180px; display: flex; align-items: flex-end; gap: 4px; }
.col { flex: 1; display: flex; flex-direction: column; justify-content: flex-end; height: 100%; min-width: 0; min-height: 0; padding: 0; border: 0; border-radius: 0; background: none; cursor: pointer; }
.col.sel { box-shadow: inset 0 -2px 0 var(--text); }
.seg { display: block; border-radius: 2px 2px 0 0; }
.seg.over { background: var(--danger); }
.cap-line { position: absolute; left: 0; right: 0; border-top: 2px dashed var(--warn); z-index: 1; pointer-events: none; }
.cap-line span { position: absolute; right: 0; top: -16px; font-size: 10px; color: var(--warn); }
.xlabs { display: flex; gap: 4px; margin-top: 4px; }
.xlabs span { flex: 1; text-align: center; font-size: 10px; color: var(--text-dim); }
.a2a { display: flex; flex-direction: column; gap: 6px; }
.a2a-grid { display: grid; gap: 3px; }
.a2a-cell { height: 28px; display: flex; align-items: center; justify-content: center; font-size: 11px; color: var(--text); border: 1px solid var(--border); border-radius: 3px; }
.legend { display: flex; gap: 12px; flex-wrap: wrap; }
.leg { display: inline-flex; align-items: center; gap: 5px; font-size: 11px; color: var(--text-muted); }
.leg i { width: 10px; height: 10px; border-radius: 2px; }
</style>
