<!-- KV cache 实验台 (对应 llm_infer/m01_kv_cache): 每步到底前向了几个 token。 -->
<template>
  <LabFrame
    title="KV cache — 每一步到底重算了多少 token"
    sub="柱子 = 这一步送进模型前向的 token 数。
      - 没有 cache: 第 $t$ 步要把 prompt + 已生成的 $t$ 个 token 全部重跑。
      - 有 cache: prefill 只跑一次, 之后每步只前向 1 个新 token。
      点柱子 (或聚焦图后按左右方向键) 跳到那一步, 看下方 token 条里谁被重算、谁直接从 cache 读。"
    module="llm_infer/m01"
    run="python -m llm_infer.m01_kv_cache.demo"
    :challenge="{
      ask: '把生成长度 N 从 32 拖到 8, 长度变成 1/4。无 cache 的总前向 token 数也变成 1/4 吗?',
      answer: '降得更多: 688 降到 76, 约 1/9。\n- 无 cache: 总量 $= N \\cdot P + N(N-1)/2$, 含 $N^2$ 项。\n- 有 cache: 总量 $= P + N - 1$, 只从 37 降到 13。\ncache 用 $O(T)$ 的显存换掉了 $O(T^2)$ 的重复计算。所以后面所有章节都在管这块显存。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: !cache }" @click="cache = false">无 cache (每步重跑整段)</button>
        <button type="button" :class="{ active: cache }" @click="cache = true">有 KV cache</button>
      </div>
      <LabSlider v-model="P" label="prompt 长度 P" :min="4" :max="32" />
      <LabSlider v-model="N" label="生成 token 数 N" :min="2" :max="32" />
      <StepPlayer :stepper="stepper" :label="`第 ${t} 步`" />
    </template>

    <svg :viewBox="`0 0 ${W} ${H + 18}`" role="group" aria-label="每步前向 token 数">
      <!-- 整张图只有一个 Tab 停靠点, 左右方向键换步 -->
      <g
        class="bars" tabindex="0" role="slider" aria-label="当前步" aria-valuemin="0" :aria-valuemax="N - 1" :aria-valuenow="t"
        :aria-valuetext="`第 ${t} 步, 无 cache 前向 ${steps[t].no} 个, 有 cache 前向 ${steps[t].yes} 个`"
        @keydown.left.prevent="stepper.prev()" @keydown.right.prevent="stepper.next()"
      >
        <g v-for="s in steps" :key="s.t" :opacity="s.t <= t ? 1 : 0.3" @click="stepper.step.value = s.t">
          <rect :x="x(s.t)" :y="H - s.no * k" :width="bw" :height="s.no * k" class="bar no" :class="{ sel: !cache }" />
          <rect :x="x(s.t) + bw" :y="H - s.yes * k" :width="bw" :height="Math.max(s.yes * k, 1.5)" class="bar yes" :class="{ sel: cache }" />
          <rect v-if="s.t === t" :x="x(s.t) - 1" y="0" :width="bw * 2 + 2" :height="H" class="now" />
        </g>
      </g>
      <text x="0" :y="H + 14" class="ax">step 0 (含 prefill)</text>
      <text :x="W" :y="H + 14" class="ax" text-anchor="end">step {{ N - 1 }}</text>
    </svg>

    <p class="strip-title">第 {{ t }} 步送进注意力的 {{ P + t }} 个位置:</p>
    <div class="strip">
      <span
        v-for="i in P + t" :key="i" class="cell"
        :class="cellClass(i - 1)" :title="cellTitle(i - 1)"
      >{{ i - 1 < P ? 'p' : 'g' }}</span>
    </div>
    <p class="legend">
      <span class="sw no" /> 无 cache 的柱子
      <span class="sw yes" /> 有 cache 的柱子
      <span class="cell hot">p</span> 重新前向 (重算 K/V + MLP)
      <span class="cell ok">p</span> 直接读 cache
      <span class="cell on">g</span> 本步新算的 token
    </p>

    <template #stats>
      <div class="kv"><span>本步前向 token 数</span><b :class="nowCls">{{ cur.now }}</b></div>
      <div class="kv"><span>累计前向 (到第 {{ t }} 步)</span><b>{{ cur.cum }}</b></div>
      <div class="kv"><span>全程总量: 无 / 有 cache</span><b>{{ total.no }} / {{ total.yes }} ({{ (total.no / total.yes).toFixed(1) }}×)</b></div>
      <div class="kv"><span>显存里攥着的 KV 条目</span><b>{{ cache ? P + t : 0 }} × 层数</b></div>
      <div class="lab-note">
        <p>
          闭式: 无 cache <Tex text="$= N \cdot P + N(N-1)/2 =$" /> {{ N * P + N * (N - 1) / 2 }}; 有 cache <Tex text="$= P + N - 1 =$" /> {{ P + N - 1 }}。
          与上面逐步累加的数相同。
        </p>
        <p>cache 省掉的是旧 token 的 K/V 投影和 MLP 重算。</p>
        <p>新 token 的 query 仍要和全部 {{ P + t }} 个 key 做点积, 每步访存依旧是 <Tex text="$O(t)$" />。所以长上下文 decode 是带宽瓶颈。</p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import Tex from '@/components/Tex.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import StepPlayer from '@/components/lab/StepPlayer.vue'
import { useStepper } from '@/composables/useStepper.js'
import { range, sum } from '@/utils/labmath.js'

const cache = ref(true)
// 默认值与 m01 demo 相同: prompt 6 + 生成 32, 累计前向 688 → 37 (18.6×)
const P = ref(6)
const N = ref(32)
const W = 640, H = 150

// ★ 全部的差别: 无 cache 第 t 步前向 P+t 个 token; 有 cache 只有第 0 步前向 P 个, 之后每步 1 个
const steps = computed(() => range(N.value).map((t) => ({ t, no: P.value + t, yes: t === 0 ? P.value : 1 })))
const total = computed(() => ({ no: sum(steps.value.map((s) => s.no)), yes: sum(steps.value.map((s) => s.yes)) }))

const stepper = useStepper(() => N.value, { interval: 450 })
const t = computed(() => stepper.step.value)
const key = computed(() => (cache.value ? 'yes' : 'no'))
const cur = computed(() => ({
  now: steps.value[t.value][key.value],
  cum: sum(steps.value.slice(0, t.value + 1).map((s) => s[key.value])),
}))
// 和这一步的两种做法比: 比重跑整段少就绿, 比只算新 token 多就红; 第 0 步两者相同, 不上色
const nowCls = computed(() => {
  const s = steps.value[t.value]
  return cur.value.now < s.no ? 'good' : cur.value.now > s.yes ? 'bad' : ''
})
// 换参数后直接跳到最后一步, 先看到全部柱子, 再按播放重看过程
watch([P, N], () => { stepper.pause(); stepper.step.value = N.value - 1 }, { immediate: true })

const bw = computed(() => W / N.value / 2.4)
const x = (i) => (i + 0.1) * (W / N.value)
const k = computed(() => H / (P.value + N.value - 1)) // 最高的柱子 = 无 cache 的最后一步

const isNew = (i) => i === P.value + t.value - 1 || t.value === 0
const cellClass = (i) => (!cache.value ? 'hot' : isNew(i) ? 'on' : 'ok')
const cellTitle = (i) => `位置 ${i}: ${!cache.value ? '重新前向' : isNew(i) ? '本步计算并写入 cache' : '从 cache 读 K/V'}`
</script>

<style scoped>
svg { min-width: 520px; } /* 窄屏: 图保持可读, 由 .lab-viz 横向滚动 */
.bars { outline: none; }
.bars:focus-visible .now { stroke-width: 3; }
.bar { cursor: pointer; opacity: 0.45; }
.bar.sel { opacity: 1; }
.bar.no { fill: var(--warn); }
.bar.yes { fill: var(--left); }
.now { fill: none; stroke: var(--accent); stroke-width: 1.5; pointer-events: none; }
.ax { font-size: 11px; fill: var(--text-dim); }
.strip-title { font-size: 12px; color: var(--text-muted); margin: 12px 0 6px; }
.strip { display: flex; flex-wrap: wrap; gap: 3px; }
.sw { display: inline-block; width: 12px; height: 12px; border-radius: 2px; }
.sw.no { background: var(--warn); }
.sw.yes { background: var(--left); }
.legend { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; font-size: 11px; color: var(--text-dim); margin-top: 8px; }
</style>
