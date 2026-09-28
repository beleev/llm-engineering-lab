<!--
  Flash-Decoding 实验台 (对应 llm_infer/m26_flash_decoding/flash_decoding.py:decode_latency_us)。
  一件事: decode 只有 1 个 query, 并行单位只剩 B×H; 沿 KV 长度切 S 段, 把闲着的 SM 用上。
  延迟用与 Python 完全相同的代价模型在前端现算 (A100 参数), 不是实测。
-->
<template>
  <LabFrame
    title="Flash-Decoding — Q 切不动, 就切 KV"
    sub="108 个 SM, 每个 thread block 负责 (序列, head, 一段 KV)。$H = 32$。
      - 上图: 108 个格子是 SM, 颜色越深表示这个 SM 要跑的波次越多。
      - 下图: 延迟随切分段数 $S$ 的变化, 点圆点选 $S$。
      按钮切 batch 大小 $B$, 滑杆改上下文长度 $T$。"
    module="llm_infer/m26"
    run="python -m llm_infer.m26_flash_decoding.demo"
    :challenge="{
      ask: 'B=1、T=32k 时, S 从 2 加到 4 能快多少?',
      answer: '一点也不快: S=2 和 S=4 都是 459.0 μs。\n- S=2: 64 个 block, 1 波就跑完。\n- S=4: 128 个 block, 108 个 SM 要跑 2 波, 每波读的 KV 减半, 总时间不变。\n要让 $B \\cdot H \\cdot S$ 接近 108 的整数倍。S=64 时 3.30×, 上限是 $108 / (B \\cdot H) = 3.38$×。\n切到 B=64, $B \\cdot H = 2048$, SM 早就满了, 最优的 S 也只有 1.00×, 多一个 reduce kernel 只会亏。\n这些都来自代价模型: 真卡上单个 SM 能拿到的带宽比 1/108 高, 真实收益更小。',
    }"
  >
    <template #controls>
      <div class="row">
        <button v-for="b in BS" :key="b" type="button" :class="{ active: B === b }" :aria-pressed="B === b" @click="B = b">B = {{ b }}</button>
      </div>
      <LabSlider v-model="si" label="切分段数 S" :min="0" :max="7" :format="(i) => 2 ** i" />
      <LabSlider v-model="ti" label="上下文长度 T" :min="0" :max="3" :format="(i) => TS[i].toLocaleString()" />
    </template>

    <div class="sms" role="img" :aria-label="`108 个 SM, ${cur.waves} 波`">
      <span v-for="i in N_SM" :key="i" class="sm" :style="{ background: smFill(i - 1) }" :title="`SM ${i - 1}: ${smWaves(i - 1)} 波`" />
    </div>

    <svg viewBox="0 0 560 190" role="group" :aria-label="`延迟随 S 的曲线, 当前 S=${cur.S}: ${cur.t.toFixed(1)} μs`">
      <line x1="40" x2="550" y1="160" y2="160" class="axis" />
      <polyline :points="rows.map((r, i) => `${sx(i)},${ly(r.t)}`).join(' ')" class="line" />
      <g v-for="(r, i) in rows" :key="i" class="pt" tabindex="0" role="button" :aria-label="`S=${r.S}: ${r.t.toFixed(1)} μs`" :aria-pressed="i === si" @click="si = i" @keydown.enter="si = i">
        <circle :cx="sx(i)" :cy="ly(r.t)" r="10" class="hit" />
        <circle :cx="sx(i)" :cy="ly(r.t)" :r="i === si ? 5 : 3.5" :class="{ cur: i === si, best: i === bestI }" />
        <text :x="sx(i)" :y="ly(r.t) - 9" class="tick">{{ r.t.toFixed(0) }}</text>
        <text :x="sx(i)" y="174" class="tick">S={{ r.S }}</text>
      </g>
      <text x="4" y="12" class="cap">一层 decode attention 的估算延迟 (μs), 绿点 = 最优 S</text>
    </svg>

    <template #stats>
      <div class="kv"><span>并行单位 B·H·S / 波次</span><b>{{ cur.units }} / {{ cur.waves }}</b></div>
      <div class="kv"><span>SM 利用率</span><b :class="cur.util < 0.5 ? 'bad' : cur.util > 0.9 ? 'good' : ''">{{ (cur.util * 100).toFixed(0) }}%</b></div>
      <div class="kv"><span>估算延迟</span><b>{{ cur.t.toFixed(1) }} μs</b></div>
      <div class="kv"><span>相对 S=1 加速 (最优 S={{ rows[bestI].S }})</span><b :class="speed > 1.5 ? 'good' : ''">{{ speed.toFixed(2) }}× ({{ (rows[0].t / rows[bestI].t).toFixed(2) }}×)</b></div>
      <div class="lab-note">
        <p>加速上限 <Tex text="$N_{\text{SM}} / (B \cdot H)$" /> = {{ B * H < N_SM ? `${(N_SM / (B * H)).toFixed(2)}×` : '— (SM 已满)' }}。</p>
        <p><Tex text="★ $t = 3 + \text{waves} \cdot (T/S) \cdot 512\,\text{B} / (2000\,\text{GB/s} / 108)$; $S \gt 1$ 时再加一个 reduce kernel: $3 + \text{units} \cdot 129 \cdot 4\,\text{B} / 2000\,\text{GB/s}$。" /></p>
        <p>公式与 decode_latency_us 相同, 可对 demo 的 [2] [3] 段: B=1、T=32k 时 S=1 909.0 μs, S=64 275.5 μs (3.30×)。</p>
        <p>分段结果按 <Tex text="$\exp(\mathrm{lse}_s - \mathrm{lse})$" /> 加权合并, 与普通 attention 数学相等 (13 组配置最坏误差 2.06e-07)。</p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import Tex from '@/components/Tex.vue'
import { argmax, range } from '@/utils/labmath.js'

// 与 flash_decoding.py 的常量一致: A100, 108 SM, 2000 GB/s, 每 token 每 head 512 B, 每个 kernel 3 μs
const N_SM = 108, HBM = 2000, KV_B = 512, LAUNCH = 3, H = 32
const BS = [1, 4, 16, 64], TS = [512, 4096, 32768, 131072]

const B = ref(1), si = ref(0), ti = ref(2)

// ★ decode_latency_us 的逐行翻译
const latency = (b, T, S) => {
  const units = b * H * S, waves = Math.ceil(units / N_SM)
  let t = LAUNCH + waves * (T / S) * KV_B / (HBM * 1e3 / N_SM)
  if (S > 1) t += LAUNCH + units * (128 + 1) * 4 / (HBM * 1e3)
  return { S, units, waves, t, util: units / (waves * N_SM) }
}
const rows = computed(() => range(8).map((i) => latency(B.value, TS[ti.value], 2 ** i)))
const cur = computed(() => rows.value[si.value])
const bestI = computed(() => argmax(rows.value.map((r) => -r.t)))
const speed = computed(() => rows.value[0].t / cur.value.t)

// 每个 SM 分到几波: units 按轮转分给 108 个 SM
const smWaves = (i) => Math.floor(cur.value.units / N_SM) + (i < cur.value.units % N_SM ? 1 : 0)
const smFill = (i) => {
  const w = smWaves(i)
  return w === 0 ? 'transparent' : `color-mix(in srgb, var(--accent) ${Math.round(30 + 70 * w / cur.value.waves)}%, transparent)`
}

const sx = (i) => 60 + i * 68
const maxT = computed(() => Math.max(...rows.value.map((r) => r.t)) * 1.1)   // B 大时 S 越大越慢, 最高点不在 S=1
const ly = (t) => 160 - (t / maxT.value) * 130
</script>

<style scoped>
svg { min-width: 540px; }
.sms { display: grid; grid-template-columns: repeat(27, 1fr); gap: 2px; max-width: 540px; margin-bottom: 10px; }
.sm { aspect-ratio: 1; border: 1px solid var(--border); border-radius: 2px; }
.cap { font-size: 10px; fill: var(--text-muted); }
.axis { stroke: var(--border-strong); }
.tick { font-size: 9px; fill: var(--text-dim); text-anchor: middle; }
.line { fill: none; stroke: var(--accent); stroke-width: 2; }
.pt { cursor: pointer; }
.pt circle:not(.hit) { fill: var(--bg-elev); stroke: var(--accent); stroke-width: 1.5; }
.pt circle.best { fill: var(--left); stroke: var(--left); }
.pt circle.cur { stroke: var(--text); stroke-width: 2.5; }
.hit { fill: transparent; stroke: none; }
</style>
