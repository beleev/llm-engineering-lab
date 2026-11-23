<!--
  KV cache 的边界 (对应 llm_models/utils/generation.py:GenerationMixin.generate)。
  只讲一件事: 序列长度超过 max_len 后窗口左移, 绝对位置全变, cache 作废, 之后每一步都整段重算。
  cache 本身怎么省计算, 由阶段 5 的 InferKvCacheLab 讲。
-->
<template>
  <LabFrame
    title="KV cache 与上下文上限 — 窗口一满, cache 作废, 之后每一步都整段重算"
    sub="- 模拟什么: 语言模型自回归逐 token 生成过程分为两阶段: 首先一次性处理全部 P 个 prompt token (prefill 阶段), 随后每步生成 1 个新 token (decode 阶段), 共执行 N 步。KV cache 将历史 token 在各层计算出的 K 与 V 缓存起来, decode 阶段新 token 只需计算自身的 Q、K、V 并读取历史缓存, 无需重复计算前缀。但模型受上下文窗口上限 max_len 限制: 一旦序列总长度超过 max_len, 模型只能截取最近 max_len 个 token, 窗口整体左移导致各 token 的绝对位置发生变化, 原有 K/V 缓存因位置错位而失效, 只能丢弃缓存并重新执行 prefill。本实验逐步复现 generate() 中的三条执行路径: 禁用 cache、prefill (含重跑 prefill) 与常规 decode。
- 图上是什么: 上排展示序列 token 状态格, 下排展示各生成步的计算量柱状图:
  - 上排格子: 每格代表一个 token, p0 至 p5 为 prompt token, g0 起为新生成的 token; 鼠标悬停可查看详细状态。
  - 格子颜色 (对应下方图例): 黄色表示本步需重新计算 K/V 的 token, 绿色表示直接从 KV cache 读取的 token, 紫色高亮表示本步刚采样出的新 token, 灰色表示尚未生成或已滑出上下文窗口的 token。
  - 下排柱子: 每根柱子对应一个生成步, 柱高代表该步送入模型前向计算的 token 数量 (即每层执行的 K/V 投影次数); 绿柱表示仅计算 1 个新 token, 黄柱表示整段窗口重算。
- 可以做什么: 按界面控件顺序操作:
  - 「use_cache = True / False」: 切换是否启用 KV cache。
  - 三根滑杆: 分别调节「prompt 长度 P」「生成 token 数 N」「上下文上限 max_len」; 修改参数后视图会自动定位至缓存首次失效的步数。
  - 播放控件: 单步回放生成过程, 或直接点击某根柱子跳转至对应步。
- 应该看到: 默认配置下 max_len 为 12, 而 prompt 加生成总长度为 22 个 token。结合柱状图与右侧读数可观察到:
  - 第 1 步: 执行 prefill, 一次计算 6 个 token。
  - 第 2 至第 7 步: 缓存命中, 每步仅计算 1 个新 token。
  - 第 8 步起: 序列长度超过 12, 窗口每步左移并重新 prefill 全部 12 个 token, 柱高跃升至 12 并一直维持在该水平。
  - 「本步重算 token 数」: 当前步柱子的高度。
  - 「cache 从第几步起作废」: 除第 1 步初始 prefill 外, 第一根黄柱出现的步数。
  - 「累计 token 前向」: 所有生成步柱高之和。
  - 「是『cache 全程有效』的几倍」: 累计前向 token 数与理想下界 $P + N - 1$ 的比值 (后者对应 prefill 计算 P 个、后续每步仅算 1 个的全程命中情形), 1.0× 表示无额外重算开销。"
    module="llm_models/utils/generation.py"
    run="python -m llm_models.run_models.language_models.llama.infer_llama"
    :challenge="{
      ask: '当 $P = 6$、生成 token 数为 16、$\\text{max\\_len} = 12$ 时, 累计需要 120 次 token 前向计算。请先思考: max_len 至少需要调至多少, 后续 decode 步的柱高才能全部保持为 1? 此时的累计 token 前向次数是多少?',
      answer: '答案: max_len 至少需要设为 21, 此时累计 token 前向次数为 21。\n- 为什么临界值是 21: 到最后一步 (第 16 步) 开始时, 序列中已有 6 个 prompt token 与 15 个已生成 token, 合计 21 个 token; 此时缓存中存有前 20 个位置的 K/V, 再送入第 21 个位置的 token 恰好占满 21 个位置。只要 max_len 不小于 21, 上下文窗口就无需左移裁剪。\n- 窗口充足且启用 cache: 仅在第 1 步 prefill 6 个 token, 后续 15 步每步计算 1 个 token, 累计 $6 + 15 = 21$ 次。\n- 窗口充足但禁用 cache (切换「use_cache = False」观察): 每步均需重算完整前缀, 从第 1 步的 6 个递增至第 16 步的 21 个, 累计 $6 + 7 + \\dots + 21 = 216$ 次, 约为启用 cache 的 10 倍, 且差距随生成长度呈平方级扩大。\n- $\\text{max\\_len} = 12$ 且启用 cache (默认配置): 累计 120 次。\n  - 前 7 步: 第 1 步 prefill 6 个加 6 次 decode 各 1 个, 小计 12 次。\n  - 第 8 步起: 序列长度超过 12, 后续 9 步每步均需对最近 12 个 token 重新执行 prefill, 小计 108 次。\n窗口满载后 cache 失效的原因如下:\n1. 序列长度一旦超过 12, 模型仅保留最近 12 个 token, 窗口整体左移一格。\n2. 窗口内各 token 的绝对位置索引随之改变, RoPE 旋转角度与位置强绑定而全部错位, 原有 K/V 缓存无法直接复用。\n3. generate() 只能丢弃旧缓存并重新执行 prefill, 计算复杂度退化回无缓存状态。\n因此生产推理系统要么严格限制序列不超出上下文窗口, 要么采用滑动窗口注意力 (SWA) 配合滚动缓存机制。',
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
    <p class="axis-cap">每根柱子 = 一步; 高度 = 这一步喂进模型的 token 数 (= 每层 K/V 投影次数) · 绿柱只喂 1 个, 黄柱整段重算 · 点柱子跳转</p>

    <template #stats>
      <div class="kv"><span>本步重算 token 数 (有 cache 时理想是 1)</span><b :class="now.computed === 1 ? 'good' : stepper.step.value > 0 ? 'bad' : ''">{{ now.computed }}</b></div>
      <div class="kv"><span>cache 从第几步起作废 (窗口在那一步满了)</span><b :class="breakCls">{{ breakText }}</b></div>
      <div class="kv"><span>累计 token 前向 (全程所有柱子之和)</span><b :class="sim.total <= P + N ? 'good' : 'bad'">{{ sim.total }}</b></div>
      <div class="kv"><span>是「cache 全程有效」(最少 {{ P + N - 1 }} 次) 的几倍</span><b :class="ratio < 1.5 ? 'good' : 'bad'">{{ ratio.toFixed(1) }}×</b></div>
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
        : `上下文已满 (max_len=${maxLen.value}): 窗口左移, 所有绝对位置都变了, 旧 cache 作废, 重新 prefill ${win} 个。`
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
