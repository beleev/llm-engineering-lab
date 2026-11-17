<!--
  KV cache 量化分组实验台 (对应 llm_infer/m08_quantization/kv_quant.py:quantize_kv)。
  一件事: 同一个量化器, 只是"谁和谁共用一组 scale"不同 —— K 有固定离群通道所以按通道分组, V 没有所以按 token 分组 (KIVI)。
-->
<template>
  <LabFrame
    title="KV cache 量化 — 谁和谁共用一个 scale, 决定离群值害到谁"
    sub="- 模拟什么: KV cache 占大量显存, 一种省法是把里面的浮点数压成 2~8 bit 的整数, 这叫量化。做法是: 一组数共用一个范围 [min, max], 把范围等分成 $2^{\text{bits}}$ 个格点, 每个数吸附到最近的格点, 格子间距叫 scale。组里混进一个特别大的数 (离群值), 范围被撑大, scale 变粗, 同组其它数的误差一起变大。所以「谁和谁分到一组」决定了误差有多大。这个台子用一个 16 个 token × 12 个通道的 K 矩阵对比三种分组。
      - 图上是什么: 左边一张原矩阵, 右边三张误差图, 对应三种分组:
        - 最左一张: K 矩阵本身, 一行是一个 token, 一列是一个通道 (hidden 维度); 两种颜色分别代表正负, 越深绝对值越大。
        - 顶上一排小方块: 列头, 亮起的列是「固定离群通道」, 即每个 token 在这一维都是大值。
        - 右边三张: 三种分组量化后的误差 $|K - \hat{K}|$ 热力图, 越红误差越大, 三张共用一个色标, 可以直接比。每张图标题后面的数字是这张图的整体相对误差。
        - per-tensor: 整张矩阵共用一个 scale。
        - per-token: 每一行 (每个 token) 一个 scale。
        - per-channel: 每一列 (每个通道) 一个 scale。
      - 可以做什么: 按界面从上到下:
        - 「K 型」把矩阵设成有两个固定离群通道 (第 3 列和第 9 列), 这是真实 K 的样子。
        - 「V 型」去掉固定离群列, 改成每个 token 整行幅度不同, 模拟 V 的样子。
        - 「换一组」换随机数。
        - 拖「比特数」改量化用几比特。
        - 拖「离群通道幅度」改离群列有多大 (单位是正常值标准差 σ 的倍数)。
        - 点最左那张的某个列头, 把这一列变成或取消离群通道。
        - 悬停或点任何格子, 图下面一行会显示它的原值、三种分组下量化再还原的值和各自的 scale。
      - 应该看到: K 型下 per-channel 误差最小: 离群列只把自己那一列的 scale 撑大, 其它列不受影响; per-token 每一行都含离群值, 每一行的 scale 都被撑大, 整张图都偏红。切到 V 型, 排名反过来, per-token 最好。右侧读数:
        - 三个「相对误差」是 $\|K-\hat{K}\|_F / \|K\|_F$, 越小越好, 最小的标绿、最大的标红。
        - 「per-token 误差是 per-channel 的几倍」是两者的比值, 比特数越低这个倍数越大。"
    module="llm_infer/m08"
    run="python -m llm_infer.m08_quantization.demo"
    :challenge="{
      ask: 'K 型 + INT4: 三种分组谁误差最小? 先猜再看。',
      answer: 'K 型下 per-channel 最好。\n原因是离群通道「在所有 token 上都大」:\n- per-token 分组: 每一行 (每个 token) 单独算 min/max。每一行都含这两个离群值, 所以每一行的范围都被撑到离群值那么大, scale 变粗, 其余 10 个正常通道只能落在一两个格点上, 误差很大。\n- per-channel 分组: 每一列单独算 min/max。离群值被关在自己那一列里, 那一列的 scale 大, 但它本来数值就大, 相对误差不大; 其它列各用自己的小 scale, 误差很小。\n- per-tensor: 整张矩阵一个 scale, 被离群值撑大, 所有格子都粗, 最差。\n切到 V 型, 排名反过来: V 没有固定离群列, 但不同 token 的整行幅度差别大 (有的 token 整行都大, 有的整行都小), 按行分组正好把大 token 和小 token 隔开, 所以 per-token 最好。按 token 分组还有一个工程好处: 新 token 写入 cache 时可以立刻单独量化, 不用等别的 token。\nKIVI 这篇论文的做法: K 按通道分组 (per-channel), V 按 token 分组 (per-token), 两边各用自己最合适的分组。\n在 K 型下把「比特数」从 4 拖到 2, 差距更大: 比特越低格点越少, scale 被撑大的代价越重。Python demo 里 K 的相对误差:\n- INT4: per-channel 0.026 vs per-token 0.18。\n- INT2: per-channel 0.134 vs per-token 1.02, per-token 的误差已经超过信号本身, 基本不可用。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: kind === 'K' }" @click="setKind('K')">K 型 (固定离群通道)</button>
        <button type="button" :class="{ active: kind === 'V' }" @click="setKind('V')">V 型 (无固定离群通道)</button>
        <button type="button" @click="seed++">换一组</button>
      </div>
      <LabSlider v-model="bits" label="比特数" :min="2" :max="8" unit=" bit" />
      <LabSlider v-model="mag" label="离群通道幅度" :min="0" :max="20" :step="0.5" unit="×σ" />
    </template>

    <svg viewBox="0 0 572 205" role="group" aria-label="K 矩阵, 以及 per-tensor / per-token / per-channel 三种分组的量化误差热力图" @mouseleave="hover = null">
      <g v-for="(p, k) in PANELS" :key="p.id" :transform="`translate(${8 + k * 142}, 0)`">
        <text x="0" y="10" class="cap" :class="{ best: k > 0 && k - 1 === bestI, worst: k > 0 && k - 1 === worstI }">{{ p.label }}{{ k > 0 ? ' ' + q.errs[k - 1].toFixed(3) : '' }}</text>
        <template v-if="k === 0">
          <rect v-for="c in D" :key="`h${c}`" :x="(c - 1) * CS" y="16" :width="CS - 1" height="9" tabindex="0" role="button"
            :aria-label="`通道 ${c - 1}: 设为或取消离群通道`" :aria-pressed="outs.has(c - 1)" :class="['colhead', { on: outs.has(c - 1) }]"
            @click="toggle(c - 1)" @keydown.enter="toggle(c - 1)" />
        </template>
        <g v-for="t in T" :key="t">
          <rect v-for="c in D" :key="c" :x="(c - 1) * CS" :y="28 + (t - 1) * CS" :width="CS - 1" :height="CS - 1"
            :style="{ fill: k === 0 ? valColor(q.K[t - 1][c - 1]) : heat(Math.abs(q.K[t - 1][c - 1] - q.hats[k - 1][t - 1][c - 1]) / q.eMax, 'var(--danger)') }"
            :class="['c', { hov: hover && hover.t === t - 1 && hover.c === c - 1, clk: k === 0 }]"
            @mouseenter="hover = { t: t - 1, c: c - 1 }" @click="tap(k, t - 1, c - 1)" />
        </g>
      </g>
    </svg>
    <p class="read mono">{{ readout }}</p>

    <template #stats>
      <div v-for="(n, i) in NAMES" :key="n" class="kv">
        <span>{{ n }} 分组的相对误差</span><b :class="i === bestI ? 'good' : i === worstI ? 'bad' : ''">{{ q.errs[i].toFixed(4) }}</b>
      </div>
      <div class="kv"><span>per-token 误差是 per-channel 的几倍</span><b>{{ (q.errs[1] / q.errs[2]).toFixed(1) }}×</b></div>
      <div class="lab-note">
        <p>★ 三种分组只差一件事: 在哪个范围内统计 min/max。per-tensor 看整张矩阵, per-token 看一行, per-channel 看一列。于是 scale 的组数分别是 1 / {{ T }} / {{ D }}: 组越多, 每组的范围越贴合自己的数, 但要多存几个 scale。</p>
        <p><Tex text="三种分组用的是同一个量化器 (非对称 RTN, 即直接舍入到最近格点): $q = \mathrm{round}((x - \text{lo})/\text{scale})$, 还原时 $\hat{x} = q \cdot \text{scale} + \text{lo}$。相对误差 $= \|K - \hat{K}\|_F / \|K\|_F$, 即误差矩阵的长度除以原矩阵的长度, 0.1 意味着误差约占信号的 10%。" /></p>
        <p>V 型是玩具假设: 没有固定离群列, 各 token 整行的幅度服从对数正态分布 (有的行大有的行小)。真实 V 的分布更复杂, 但「没有固定离群列」这一点成立。</p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import Tex from '@/components/Tex.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { argmax, clamp, heat, mulberry32, randn, range } from '@/utils/labmath.js'

const T = 16, D = 12, CS = 11
const NAMES = ['per-tensor', 'per-token', 'per-channel']
const PANELS = [{ id: 'k', label: 'K 矩阵 (点列头切换离群通道)' }, ...NAMES.map((n) => ({ id: n, label: `${n} 误差` }))]
const kind = ref('K'), bits = ref(4), mag = ref(8), seed = ref(1), hover = ref(null)
const outs = ref(new Set([3, 9]))
const toggle = (c) => { const s = new Set(outs.value); s.has(c) ? s.delete(c) : s.add(c); outs.value = s }
const tap = (k, t, c) => { hover.value = { t, c }; if (k === 0) toggle(c) }   // 点一下也能看 scale, 触屏没有悬停
const setKind = (k) => { kind.value = k; outs.value = new Set(k === 'K' ? [3, 9] : []) }

// 同一个非对称 min/max 量化器, 输入是"共用一组 scale 的那些数"
const quant = (xs, b) => {
  const lo = Math.min(...xs), L = 2 ** b - 1, scale = Math.max((Math.max(...xs) - lo) / L, 1e-4)
  return { scale, hat: xs.map((v) => clamp(Math.round((v - lo) / scale), 0, L) * scale + lo) }
}

const q = computed(() => {
  const r = mulberry32(seed.value * 53 + 11)
  const base = range(T).map(() => range(D).map(() => randn(r)))
  const rowS = range(T).map(() => Math.exp(randn(r) * 0.8)), sign = range(D).map(() => (r() < 0.5 ? -1 : 1))
  // K 型: 离群通道 = 每个 token 在这一维都是 ±mag 附近的大值; V 型: 每个 token 整行乘一个幅度
  const K = base.map((row, t) => row.map((v, c) => (kind.value === 'V' ? v * rowS[t] : v)
    + (outs.value.has(c) ? mag.value * sign[c] * (1 + 0.15 * base[t][(c + 1) % D]) : 0)))
  const all = quant(K.flat(), bits.value)
  const rows = K.map((row) => quant(row, bits.value))
  const cols = range(D).map((c) => quant(K.map((row) => row[c]), bits.value))
  const hats = [
    range(T).map((t) => all.hat.slice(t * D, (t + 1) * D)),   // axis=None
    rows.map((x) => x.hat),                                    // ★ axis=1: 每个 token 一组
    range(T).map((t) => cols.map((x) => x.hat[t])),            // ★ axis=0: 每个通道一组 (KIVI 对 K 的选择)
  ]
  const kf = K.flat(), hf = hats.map((H) => H.flat()), nk = Math.hypot(...kf)
  const errs = hf.map((h) => Math.hypot(...kf.map((v, i) => v - h[i])) / nk)
  const eMax = Math.max(1e-9, ...hf.flatMap((h) => h.map((v, i) => Math.abs(v - kf[i]))))
  const scales = (t, c) => [all.scale, rows[t].scale, cols[c].scale]
  return { K, hats, errs, eMax, scales, vMax: Math.max(...kf.map(Math.abs)) }
})
const bestI = computed(() => argmax(q.value.errs.map((e) => -e))), worstI = computed(() => argmax(q.value.errs))
const valColor = (v) => heat(Math.abs(v) / q.value.vMax, v >= 0 ? 'var(--accent)' : 'var(--right)')
const readout = computed(() => {
  if (!hover.value) return '悬停或点一个格子: 显示它的原值, 以及三种分组下量化再还原的值和各自所在组的 scale (格点间距)。scale 越大, 格点越粗, 还原值离原值越远。'
  const { t, c } = hover.value, s = q.value.scales(t, c)
  return `token ${t}, 通道 ${c}: 原值 ${q.value.K[t][c].toFixed(2)} → ` + NAMES.map((n, i) => `${n} ${q.value.hats[i][t][c].toFixed(2)} (scale ${s[i].toFixed(3)})`).join(' · ')
})
</script>

<style scoped>
/* 窄屏: 保住可读的最小宽度, 由 .lab-viz 横向滚动; 只有 .draggable 把手拦截触摸 */
svg { min-width: 540px; touch-action: pan-x pan-y; }
.cap { font-size: 9.5px; fill: var(--text-muted); }
.cap.best { fill: var(--left); font-weight: 600; }
.cap.worst { fill: var(--danger); }
.c { stroke: var(--border); stroke-width: 0.5; }
.c.clk { cursor: pointer; }
.c.hov { stroke: var(--text); stroke-width: 1.5; }
.colhead { fill: var(--bg-elev); stroke: var(--border-strong); stroke-width: 0.5; cursor: pointer; outline: none; }
.colhead.on { fill: var(--warn); }
.colhead:hover, .colhead:focus-visible { stroke: var(--accent); stroke-width: 1.5; }
.read { font-size: 11px; color: var(--text-muted); margin-top: 8px; line-height: 1.6; min-height: 3.2em; }
</style>
