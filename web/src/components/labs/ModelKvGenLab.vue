<!--
  KV cache 的边界 (对应 llm_models/utils/generation.py:GenerationMixin.generate)。
  只讲一件事: 序列长度超过 max_len 后窗口左移, 绝对位置全变, cache 作废, 之后每一步都整段重算。
  cache 本身怎么省计算, 由阶段 5 的 InferKvCacheLab 讲。
-->
<template>
  <LabFrame
    title="窗口满了 — KV cache 作废, 之后每一步都整段重算"
    sub="默认 max_len = 12, 而 prompt 加生成一共 22 个 token。前 7 步 cache 正常工作, 第 8 步起每步都要重算 12 个。
      - 上排: 当前这一步喂进模型的 token, 颜色含义看图例。
      - 下排: 每一步喂进模型的 token 数, 点柱子跳到那一步。"
    module="llm_models/utils/generation.py"
    run="python -m llm_models.run_models.language_models.llama.infer_llama"
    :challenge="{
      ask: 'P=6, 生成 16 个 token, max_len=12 时累计 120 次 token 前向。先猜: max_len 至少拖到多少, 柱子才全部落回 1? 那时累计是多少?',
      answer: '- 答案: max_len ≥ 21。最后一步开始时序列长 21, cache 里已有 20 条, 再喂 1 个新 token 正好占满 21 个位置。\n- 窗口够长, 有 cache: 只在第一步 prefill 6 个, 之后每步 1 个, $6+15 = 21$。\n- 窗口够长, 无 cache: 每步重算整个前缀, $6+7+\\dots+21 = 216$ 次 token 前向。约 10 倍, 且差距随长度平方增长。\n- max_len=12, 有 cache: 前 7 步 $6+6 = 12$, 后 9 步每步重算 12 个, 合计 $12 + 108 = 120$。\nmax_len=12 时, 长度一超过 12, 窗口就要左移。每个 token 的绝对位置都变了 (RoPE 角度/位置 embedding 全错位), 旧 K/V 不能复用。generate() 只能丢掉 cache 重新 prefill, 退化回无 cache。\n所以生产系统要么不让超长, 要么用 SWA 滚动缓存。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: useCache }" @click="useCache = true">use_cache = True</button>
        <button type="button" :class="{ active: !useCache }" @click="useCache = false">use_cache = False (每步重算前缀)</button>
      </div>
      <LabSlider v-model="P" label="prompt 长度 P" :min="2" :max="10" />
      <LabSlider v-model="N" label="生成 token 数" :min="4" :max="20" />
      <LabSlider v-model="maxLen" label="上下文上限 max_len" :min="8" :max="32" />
      <StepPlayer :stepper="stepper" :label="`第 ${stepper.step.value + 1} 步`" />
    </template>

    <div class="cells" :style="{ gridTemplateColumns: `repeat(${P + N}, minmax(18px, 26px))` }">
      <span
        v-for="(c, i) in cellsNow" :key="i" class="cell" :class="c.cls"
        :title="c.tip"
      >{{ i < P ? 'p' + i : 'g' + (i - P) }}</span>
    </div>
    <p class="legend">
      <span class="cell hot">·</span> 本步重算 K/V <span class="cell ok">·</span> 读缓存 <span class="cell on">·</span> 本步新采样出的 token
      <span class="cell dim">·</span> 尚未生成 / 已滑出窗口
    </p>

    <!-- 柱子只给鼠标和触屏点; 键盘用上面 StepPlayer 的时间轴, 免得多出 N 个 Tab 停靠点 -->
    <div class="bars" role="group" aria-label="每步计算量">
      <button
        v-for="(f, s) in sim.frames" :key="s" type="button" tabindex="-1" class="bar" :class="{ now: s === stepper.step.value, prefill: f.computed > 1 }"
        :style="{ height: 8 + (f.computed / sim.maxC) * 92 + 'px' }" :aria-label="`第 ${s + 1} 步, 重算 ${f.computed} 个 token`"
        @click="stepper.step.value = s"
      ><i>{{ f.computed }}</i></button>
    </div>
    <p class="axis-cap">每步喂进模型的 token 数 (= 每层 K/V 投影次数) · 点柱子跳转</p>

    <template #stats>
      <div class="kv"><span>本步重算 token 数</span><b :class="now.computed === 1 ? 'good' : stepper.step.value > 0 ? 'bad' : ''">{{ now.computed }}</b></div>
      <div class="kv"><span>cache 从第几步起作废</span><b :class="breakCls">{{ breakText }}</b></div>
      <div class="kv"><span>累计 token 前向 (全程)</span><b :class="sim.total <= P + N ? 'good' : 'bad'">{{ sim.total }}</b></div>
      <div class="kv"><span>是「cache 全程有效」的几倍</span><b :class="ratio < 1.5 ? 'good' : 'bad'">{{ ratio.toFixed(1) }}×</b></div>
      <p class="lab-note">
        {{ now.why }}
      </p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import StepPlayer from '@/components/lab/StepPlayer.vue'
import { useStepper } from '@/composables/useStepper.js'
import { range, sum } from '@/utils/labmath.js'

// 默认 max_len=12 < P+N-1=21: 打开就能看到窗口满了之后的样子
const useCache = ref(true), P = ref(6), N = ref(16), maxLen = ref(12)

// 逐步复刻 GenerationMixin.generate 的三个分支
const sim = computed(() => {
  const frames = []
  let pos = 0 // cache.pos: 缓存里已有多少条
  for (let s = 0; s < N.value; s++) {
    const len = P.value + s                      // 本步开始时序列总长
    const win = Math.min(len, maxLen.value)      // 模型实际能看到的后缀
    let computed_, why
    if (!useCache.value) {
      computed_ = win; pos = 0
      why = `无 cache: 把最近 ${win} 个 token 全部重新前向, 只为了拿最后一个位置的 logits。`
    } else if (pos === 0 || pos + 1 > maxLen.value) {
      // ★ prefill, 或窗口已满: 左移后绝对位置全变, 旧 cache 作废
      computed_ = win; pos = win
      why = s === 0 ? `prefill: 一次性把 ${win} 个 prompt token 的 K/V 算好写入 cache。`
        : `上下文已满 (max_len=${maxLen.value}): 窗口左移使所有绝对位置改变, 旧 cache 作废, 重新 prefill ${win} 个。`
    } else {
      computed_ = 1; pos += 1
      why = `decode: 只喂 1 个新 token, 它的 Q 去查缓存里的 ${pos} 条 K/V (含它自己)。`
    }
    frames.push({ len, win, computed: computed_, why })
  }
  const total = sum(frames.map((f) => f.computed))
  // 第一个「有 cache 却整段重算」的步: 窗口在这一步满了
  const breakAt = useCache.value ? frames.findIndex((f, s) => s > 0 && f.computed > 1) : -1
  return { frames, total, breakAt, maxC: Math.max(...frames.map((f) => f.computed)) }
})
const breakText = computed(() =>
  !useCache.value ? '没开 cache' : sim.value.breakAt < 0 ? '没有作废' : `第 ${sim.value.breakAt + 1} 步`)
const breakCls = computed(() => (!useCache.value ? '' : sim.value.breakAt < 0 ? 'good' : 'bad'))
const ratio = computed(() => sim.value.total / (P.value + N.value - 1))

const stepper = useStepper(() => sim.value.frames.length, { interval: 450 })
// 参数一变就停到 cache 作废的那一步; 没有作废时只把当前帧夹回合法范围
watch(sim, (v) => {
  stepper.pause()
  stepper.step.value = v.breakAt >= 0 ? v.breakAt : Math.min(stepper.step.value, N.value - 1)
}, { immediate: true })
const now = computed(() => sim.value.frames[stepper.step.value] || sim.value.frames[0])

const cellsNow = computed(() => range(P.value + N.value).map((i) => {
  const f = now.value, lo = f.len - f.win
  if (i === f.len) return { cls: 'on', tip: '本步采样出的新 token' }
  if (i > f.len || i < lo) return { cls: 'dim', tip: i > f.len ? '尚未生成' : '已滑出窗口' }
  const fresh = i >= f.len - f.computed
  return { cls: fresh ? 'hot' : 'ok', tip: fresh ? '本步重算 K/V' : '读缓存' }
}))
</script>

<style scoped>
.legend { display: flex; flex-wrap: wrap; align-items: center; gap: 6px 10px; margin: 10px 0 16px; font-size: 12px; color: var(--text-muted); }
.legend .cell { min-width: 16px; width: 16px; height: 16px; }
.bars { display: flex; align-items: flex-end; gap: 3px; height: 120px; padding-top: 18px; border-bottom: 1px solid var(--border-strong); }
.bar { flex: 1; min-width: 12px; max-width: 30px; padding: 0; min-height: 0; border-radius: 3px 3px 0 0; border: 1px solid var(--left); background: color-mix(in srgb, var(--left) 30%, transparent); position: relative; cursor: pointer; }
.bar.prefill { border-color: var(--warn); background: color-mix(in srgb, var(--warn) 30%, transparent); }
.bar.now { outline: 2px solid var(--accent); outline-offset: 1px; }
.bar i { position: absolute; top: -15px; left: 0; right: 0; text-align: center; font-style: normal; font-size: 9px; color: var(--text-dim); font-family: "SF Mono", Menlo, monospace; }
.axis-cap { font-size: 11px; color: var(--text-dim); margin-top: 6px; }
</style>
