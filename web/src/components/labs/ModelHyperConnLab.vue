<!--
  Hyper-Connections 实验台 (对应 llm_models/layers/core/hyper_connections.py: sinkhorn_knopp / composite_gain)。
  只讲一件事: 流间混合矩阵 H_res 连乘 L 层之后还是不是恒等映射。不加约束时行和会漂、连乘后指数放大;
  Sinkhorn-Knopp 投到双随机之后, 行和恒为 1, 连乘多少层都还是 1。
  玩具简化: 所有层共用同一个 H_res, 所以复合映射就是 H_res 的 L 次幂; Python 里每层各有一个。
-->
<template>
  <LabFrame
    title="加宽残差流 — H_res 连乘 L 层之后还剩多少恒等"
    sub="- 模拟什么: Hyper-Connections 把 1 条残差流加宽成 n 条, 每层按 x ← H_res · x + H_post 乘层输出更新。H_res 是 n×n 的流间混合矩阵, 第 j 层到第 k 层的有效连接是中间各层 H_res 的连乘。这里 n = 4, 所有层共用同一个 H_res, 所以连乘就是它的 L 次幂。
- 图上是什么: 左边 4×4 的格子是 H_res, 颜色深浅是数值大小, 格子里写着数值; 右侧一列是每行的和, 下面一排是每列的和, 偏离 1 的标红。右边是连乘 L 层之后的复合映射, 同样画成 4×4 的格子, 下面一条横条按最大绝对行和的大小伸缩。
- 可以做什么: 两个按钮在 '不约束 (HC)' 和 '双随机 (mHC)' 之间切换。层数滑杆调 L, 偏离强度滑杆调 H_res 离单位矩阵多远。点左边任意一个格子 (或选中后按回车), 把那一格的原始打分加 1, 点满一圈回到 0。'换一组' 重抽噪声。
- 应该看到: 不约束模式下行和本来就不是 1, L 越大复合映射的最大绝对行和越大, 偏离强度调高时增长更快; 切到双随机模式, 行和立刻变成 1, 而且不管 L 调到多少、哪一格点多高, 复合增益都还是 1.000000。右侧三个读数里, 行和偏差和复合增益在双随机模式下绿, 不约束模式下随 L 变红。"
    module="llm_models/layers/core/hyper_connections.py"
    run="python -m llm_models.run_models.language_models.hyper_connections.infer_hyper_connections"
    :challenge="{
      ask: '把偏离强度拉到最大, 任选一格点到最高, 层数设成 24。不约束模式下复合增益大概是几? 切到双随机模式呢? 为什么点哪一格、点多高都不影响后者?',
      answer: '答案: 不约束模式下复合增益会涨到几十甚至更高, 并且随层数继续涨; 双随机模式下始终是 1.000000。\n原因:\n- 复合映射是 H_res 的 L 次幂, 它的增益大致按最大特征值的 L 次方走。只要 H_res 的谱半径超过 1, 深度一加就是指数放大; 小于 1 就指数衰减, 底层的信号传不到顶层。\n- 双随机矩阵 (元素非负、每行和每列的和都是 1) 的集合对矩阵乘法封闭, 谱范数不超过 1。连乘之后仍然是双随机, 所以最大绝对行和恒为 1, 恒等映射的性质不随深度丢掉。\n- Sinkhorn-Knopp 先把打分取 exp 变成正矩阵, 再交替按列、按行归一化。点某一格只改变那一格在自己行列里的占比, 归一化会把增加的部分从同行同列其他格子里扣掉, 行和列和不变。\n- Python 的 infer 脚本用 24 个各不相同的随机 H_res: 不约束时最大绝对行和从 L = 1 的 1.782 涨到 L = 24 的 10.795, 双随机下 5 个深度全是 1.000000。mHC 论文在 27B 上实测 HC 的增益峰值约 3000, mHC 约 1.6。'
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: mode === 'hc' }" @click="mode = 'hc'">不约束 (HC)</button>
        <button type="button" :class="{ active: mode === 'mhc' }" @click="mode = 'mhc'">双随机 (mHC)</button>
        <button type="button" @click="seed++; bumps = {}">换一组</button>
      </div>
      <LabSlider v-model="depth" label="连乘层数 L" :min="1" :max="24" />
      <LabSlider v-model="sigma" label="H_res 偏离单位矩阵的强度" :min="0" :max="60" :format="(v) => (v / 100).toFixed(2)" />
    </template>

    <svg :viewBox="`0 0 400 210`" class="mats" role="group" aria-label="流间混合矩阵与它的 L 次幂, 格子可点击">
      <text x="0" y="12" class="lbl">单层 H_res (点格子改原始打分)</text>
      <g v-for="(v, c) in cells" :key="'a' + c" :transform="`translate(${(c % N) * 34}, 20)`">
        <rect :y="Math.floor(c / N) * 34" width="32" height="32" class="cellbox" :style="{ fill: shade(v) }"
              tabindex="0" role="button" :aria-label="`第 ${Math.floor(c / N) + 1} 行第 ${(c % N) + 1} 列, 当前 ${v.toFixed(2)}`"
              @click="bump(c)" @keydown.enter="bump(c)" />
        <text x="16" :y="Math.floor(c / N) * 34 + 20" text-anchor="middle" class="num mono">{{ v.toFixed(2) }}</text>
      </g>
      <text v-for="(s, r) in rowSums" :key="'r' + r" x="142" :y="r * 34 + 40" class="sum mono" :class="{ off: Math.abs(s - 1) > 1e-6 }">{{ s.toFixed(3) }}</text>
      <text v-for="(s, c) in colSums" :key="'c' + c" :x="c * 34 + 16" y="172" text-anchor="middle" class="sum mono" :class="{ off: Math.abs(s - 1) > 1e-6 }">{{ s.toFixed(2) }}</text>
      <text x="142" y="186" class="lbl">↑ 列和 &nbsp; → 行和</text>

      <text x="210" y="12" class="lbl">连乘 {{ depth }} 层后的复合映射</text>
      <g v-for="(v, c) in powCells" :key="'b' + c" :transform="`translate(${210 + (c % N) * 34}, 20)`">
        <rect :y="Math.floor(c / N) * 34" width="32" height="32" class="cellbox" :style="{ fill: shade(v / Math.max(1, gain)) }" />
        <text x="16" :y="Math.floor(c / N) * 34 + 20" text-anchor="middle" class="num mono">{{ fmt(v) }}</text>
      </g>
      <rect x="210" y="162" :width="Math.min(186, 186 * gain / 12)" height="10" class="gainbar" :class="{ hot: gain > 1.05 }" />
      <text x="210" y="186" class="lbl">最大绝对行和 {{ gain.toFixed(6) }} (满条 = 12)</text>
    </svg>

    <template #stats>
      <div class="kv"><span>单层行和最大偏差</span><b :class="rowErr < 1e-6 ? 'good' : 'bad'">{{ rowErr.toExponential(1) }}</b></div>
      <div class="kv"><span>连乘 {{ depth }} 层的最大绝对行和</span><b :class="Math.abs(gain - 1) < 1e-5 ? 'good' : 'bad'">{{ gain.toFixed(6) }}</b></div>
      <div class="kv"><span>单层最小元素 (非负才不会相消)</span><b :class="minEl >= 0 ? 'good' : 'bad'">{{ minEl.toFixed(3) }}</b></div>
      <div class="lab-note">
        <p>复合映射的最大绝对行和就是 mHC 论文的 Amax Gain Magnitude (前向口径), 反向看的是最大绝对列和。偏离 1 表示整条残差流的信号被放大或衰减。</p>
        <p>Python 实测, 不随左侧变化: 24 个各不相同的 H_res, 不约束时最大绝对行和 L = 1 为 1.782、L = 24 为 10.795; 双随机下 1、4、8、16、24 层全是 1.000000。Sinkhorn 固定迭代 20 次, 行和偏差 1.2e-07, 列和偏差 5.6e-05。</p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { mulberry32 } from '@/utils/labmath.js'

const N = 4
const mode = ref('mhc'), depth = ref(8), sigma = ref(30), seed = ref(1)
const bumps = ref({})

// 原始打分 H̃_res: 单位矩阵加噪声, 再加上读者点出来的增量
const raw = computed(() => {
  const r = mulberry32(seed.value * 7919), s = sigma.value / 100
  return Array.from({ length: N * N }, (_, c) => {
    const base = (Math.floor(c / N) === c % N ? 1 : 0) + s * (r() * 2 - 1)
    return base + (bumps.value[c] || 0)
  })
})

// Sinkhorn-Knopp: exp 后交替按列、按行归一化, 与 Python 的 sinkhorn_knopp 同一件事
function sinkhorn(v) {
  const mx = Math.max(...v)
  let m = v.map((x) => Math.exp(x - mx))
  for (let it = 0; it < 20; it++) {
    for (let c = 0; c < N; c++) {                                   // T_c: 每列和为 1
      let s = 0
      for (let r = 0; r < N; r++) s += m[r * N + c]
      for (let r = 0; r < N; r++) m[r * N + c] /= s
    }
    for (let r = 0; r < N; r++) {                                   // ★ T_r: 每行和为 1, 最后一步, 所以行和恰为 1
      let s = 0
      for (let c = 0; c < N; c++) s += m[r * N + c]
      for (let c = 0; c < N; c++) m[r * N + c] /= s
    }
  }
  return m
}

const cells = computed(() => (mode.value === 'mhc' ? sinkhorn(raw.value) : raw.value))
const rowSums = computed(() => Array.from({ length: N }, (_, r) => cells.value.slice(r * N, r * N + N).reduce((a, b) => a + b, 0)))
const colSums = computed(() => Array.from({ length: N }, (_, c) => cells.value.filter((_, i) => i % N === c).reduce((a, b) => a + b, 0)))
const rowErr = computed(() => Math.max(...rowSums.value.map((s) => Math.abs(s - 1))))
const minEl = computed(() => Math.min(...cells.value))

// H_res 的 L 次幂: 复合映射。所有层共用一个矩阵时, 连乘就是矩阵幂
const powCells = computed(() => {
  let p = Array.from({ length: N * N }, (_, c) => (Math.floor(c / N) === c % N ? 1 : 0))
  for (let l = 0; l < depth.value; l++) {
    const q = new Array(N * N).fill(0)
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) for (let k = 0; k < N; k++) q[i * N + j] += cells.value[i * N + k] * p[k * N + j]
    p = q
  }
  return p
})
const gain = computed(() => Math.max(...Array.from({ length: N }, (_, r) =>
  powCells.value.slice(r * N, r * N + N).reduce((a, b) => a + Math.abs(b), 0))))

const shade = (v) => `color-mix(in srgb, var(--accent) ${Math.round(Math.min(1, Math.abs(v)) * 80)}%, transparent)`
const fmt = (v) => (Math.abs(v) >= 100 ? v.toExponential(0) : v.toFixed(2))
function bump(c) { bumps.value = { ...bumps.value, [c]: ((bumps.value[c] || 0) + 1) % 4 } }
</script>

<style scoped>
.mats { width: 100%; min-width: 420px; display: block; }
.cellbox { stroke: var(--border); stroke-width: 1; cursor: pointer; }
.cellbox:focus { outline: none; stroke: var(--text); stroke-width: 2; }
.num { font-size: 9px; fill: var(--text); }
.sum { font-size: 10px; fill: var(--text-muted); }
.sum.off { fill: var(--danger); }
.lbl { font-size: 9px; fill: var(--text-dim); }
.gainbar { fill: var(--eye); }
.gainbar.hot { fill: var(--danger); }
</style>
