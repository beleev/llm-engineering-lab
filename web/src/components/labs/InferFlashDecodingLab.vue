<!--
  Flash-Decoding 实验台 (对应 llm_infer/m26_flash_decoding/flash_decoding.py:decode_latency_us)。
  一件事: decode 只有 1 个 query, 并行单位只剩 B×H; 沿 KV 长度切 S 段, 把闲着的 SM 用上。
  延迟用与 Python 完全相同的代价模型在前端现算 (A100 参数), 不是实测。
-->
<template>
  <LabFrame
    title="Flash-Decoding — decode 的 Q 只有 1 行切不动, 就把 KV 切成段并行"
    sub="- 模拟什么: decode 阶段每条序列只有 1 个 query 要算注意力, 但要把它和整段上下文的 KV (T 个 token) 做点积。GPU 上并行的单位是 thread block, 一个 block 对应 (一条序列, 一个注意力头): FlashAttention 本来还按 Q 分块, 可 decode 的 Q 只有 1 行, 切不动, 并行单位只剩 $B \times H$ 个 (batch 大小乘头数)。A100 有 108 个 SM (流式多处理器, 真正干活的单元), $B = 1$、$H = 32$ 时只有 32 个 block, 70% 的 SM 闲着。Flash-Decoding 的做法 (split-K): 把 KV 沿长度切成 S 段, 每段一个 block 各算局部结果, 并行单位变成 $B \times H \times S$; 最后再跑一个小的 reduce kernel 按 lse (每段 softmax 分母的对数) 把 S 段合并, 结果和普通注意力完全相等。延迟用和 Python 相同的代价模型现算 (A100 参数), 不是实测。
      - 图上是什么: 上面 108 个小格是 108 个 SM, 颜色越深表示这个 SM 要跑的波次越多 (block 数超过 108 时, 多出来的要排到下一波), 没颜色的 SM 在闲着。下面的折线是一层 decode attention 的估算延迟 (μs) 随切分段数 S 的变化, 每个点标着延迟, 绿点是延迟最低的 S, 描边加粗的是当前选中的 S。
      - 可以做什么: 点 B = 1 / 4 / 16 / 64 改 batch 大小 (同时在 decode 的序列数); 拖「切分段数 S」或点折线上的圆点改 S; 拖「上下文长度 T」改 KV 有多长。
      - 应该看到: B = 1 时 S 从 1 加到 64, 延迟从 909 μs 降到 275.5 μs, 上面的 SM 格子从大半空白变成全部填满; B = 64 时 SM 本来就满了, S 越大反而越慢 (多了 reduce 的开销)。右侧读数: 「并行单位 / 波次」是 $B \cdot H \cdot S$ 个 block 要分几波跑完 (每波最多 108 个); 「SM 利用率」是这些 block 平均把 108 个 SM 用满了几成; 「估算延迟」是当前 S 的延迟; 「相对 S=1 加速」是当前 S 比不切快多少倍, 括号里是最优 S 能达到的倍数。"
    module="llm_infer/m26"
    run="python -m llm_infer.m26_flash_decoding.demo"
    :challenge="{
      ask: 'B=1、T=32k 时, S 从 2 加到 4 能快多少?',
      answer: '一点也不快: S = 2 和 S = 4 都是 459.0 μs。\n- S = 2: $1 \\times 32 \\times 2 = 64$ 个 block, 108 个 SM 一波就跑完, 每个 block 读 T/2 的 KV。\n- S = 4: 128 个 block, 超过 108, 要跑 2 波; 每波每个 block 只读 T/4 的 KV, 两波加起来和 S = 2 一波读的一样多, 总时间不变。\n要让 $B \\cdot H \\cdot S$ 接近 108 的整数倍才划算。S = 64 时 2048 个 block 跑 19 波, SM 利用率 100%, 加速 3.30×; 上限是 $108 / (B \\cdot H) = 3.38$× (把闲着的 SM 全用上)。\n切到 B = 64: $B \\cdot H = 2048$, 不切 SM 就已经满了, 最优的 S 也只有 1.00×, 多一个 reduce kernel 只会亏。\n这些都来自代价模型 (假设每个 SM 最多拿到 1/108 的显存带宽): 真卡上单个 SM 能拿到的带宽比这高, 真实收益更小。',
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
      <div class="kv"><span>并行单位 B·H·S (block 数) / 要跑几波</span><b>{{ cur.units }} / {{ cur.waves }}</b></div>
      <div class="kv"><span>SM 利用率 (108 个 SM 平均用满几成)</span><b :class="cur.util < 0.5 ? 'bad' : cur.util > 0.9 ? 'good' : ''">{{ (cur.util * 100).toFixed(0) }}%</b></div>
      <div class="kv"><span>当前 S 的估算延迟</span><b>{{ cur.t.toFixed(1) }} μs</b></div>
      <div class="kv"><span>相对 S=1 的加速 (括号里是最优 S={{ rows[bestI].S }} 的)</span><b :class="speed > 1.5 ? 'good' : ''">{{ speed.toFixed(2) }}× ({{ (rows[0].t / rows[bestI].t).toFixed(2) }}×)</b></div>
      <div class="lab-note">
        <p>加速上限 <Tex text="$N_{\text{SM}} / (B \cdot H)$" /> = {{ B * H < N_SM ? `${(N_SM / (B * H)).toFixed(2)}×` : '— (不切 SM 就已经满了)' }}: 最多把闲着的 SM 全用上。</p>
        <p><Tex text="★ 代价模型: $t = 3 + \text{waves} \cdot (T/S) \cdot 512\,\text{B} / (2000\,\text{GB/s} / 108)$; $S \gt 1$ 时再加一个 reduce kernel: $3 + \text{units} \cdot 129 \cdot 4\,\text{B} / 2000\,\text{GB/s}$。" /></p>
        <p>其中每个 kernel 3 μs 固定开销, 每 token 每 head 的 KV 是 512 B (fp16, head_dim 128), 显存带宽 2000 GB/s 平分给 108 个 SM, 每波每个 block 要读 T/S 个 token 的 KV。</p>
        <p>公式与 decode_latency_us 相同, 可对 demo 的 [2] [3] 段: B=1、T=32k 时 S=1 909.0 μs, S=64 275.5 μs (3.30×)。</p>
        <p>分段结果按 <Tex text="$\exp(\mathrm{lse}_s - \mathrm{lse})$" /> 加权合并 (每段按它占的 softmax 质量加权), 与普通 attention 数学相等 (13 组配置最坏误差 2.06e-07)。</p>
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
