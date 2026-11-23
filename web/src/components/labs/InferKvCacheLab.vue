<!-- KV cache 实验台 (对应 llm_infer/m01_kv_cache): 每步到底前向了几个 token。 -->
<template>
  <LabFrame
    title="KV cache — 没有它, 每生成一个 token 都要把前面的全部重算一遍"
    sub="- 模拟什么: 自回归解码每步仅新增 1 个输出 token, 但自注意力层需要将该新 token 的 query 与前文所有历史位置的 K、V 向量做点积加权。在因果掩码约束下, 历史位置的隐藏状态与 K/V 投影结果不会因后续新 token 的追加而发生任何改变; 将每层已算出的历史 K/V 张量驻留显存直接复用而非每步重算, 即构成 KV cache。实验对比有无缓存的两种前向执行方式:
  - 无 cache 模式: 在第 $t$ 个生成步, 必须将初始 prompt 连同已生成的 $t$ 个历史 token 整段拼接后重新送入模型执行完整前向。
  - 有 KV cache 模式: 仅在第 0 步 (prefill 阶段) 并行计算整段 prompt 并将各层 K/V 写入缓存; 此后每个 decode 步仅向模型送入刚采样出的 1 个新 token, 算得的新 K/V 直接追加至缓存尾部。
- 图上是什么: 视图由上方逐步前向工作量柱状图与下方序列位置状态条组成:
  - 上方对比柱状图: 每个生成步并列绘制两根柱条, 柱高代表该步实际送入模型前向计算的 token 数量。黄色柱对应无 cache 模式, 绿色柱对应有 KV cache 模式 (当前激活选中的模式以实色高亮, 另一模式半透明显示); 横轴为生成步序号 (第 0 步包含初始 prefill)。
  - 下方序列位置格: 展开第 $t$ 步参与注意力交互的全部 $P + t$ 个上下文位置, 标有 p 的方格代表初始 prompt, 标有 g 的方格代表已生成的输出 token。黄色底纹表示该位置在本步被迫重新执行全层前向 (重算 K/V 投影与 MLP), 绿色底纹表示直接从 KV cache 读取历史键值免于重算, 紫色底纹标识本步唯一新执行前向并追加写入缓存的最新 token。
- 可以做什么: 提供三组交互控件:
  - 点击「无 cache」或「有 KV cache」切换前向执行机制。
  - 拖动「prompt 长度 P」与「生成 token 数 N」滑杆改变输入与生成序列长度。
  - 使用步进播放器逐步回放生成过程, 或点击柱条 (亦可聚焦图表后按键盘左右方向键) 跳转至指定生成步观察序列状态。
- 应该看到: 无 cache 模式下的黄色柱随步数推进呈线性阶梯递增 (每步均需重跑不断增长的完整前缀), 而有 KV cache 模式下的绿色柱仅在第 0 步高度为 P, 后续所有 decode 步恒定保持为 1。右侧四项读数含义如下:
  - 「本步前向 token 数」: 当前选中步实际送入模型前向计算的 token 数 (低于整段重算规模时标绿)。
  - 「累计前向」: 从第 0 步起截至当前步模型累计前向处理的 token 总量。
  - 「全程总量」: 完成全部 N 个 token 生成后无 cache 与有 cache 两种模式的累计前向总量对比及节省倍数 (在默认参数 $P = 6$、$N = 32$ 下分别为 688 与 37, 相差 18.6×, 与本节 demo 输出一致)。
  - 「cache 里存着的 token 数」: 开启 KV cache 时当前步显存缓存中驻留的序列位置总数 (模型每一层均各自保存一份)。"
    module="llm_infer/m01"
    run="python -m llm_infer.m01_kv_cache.demo"
    :challenge="{
      ask: '将生成长度 N 从 32 拖至 8 (生成长度缩减为原来的 1/4), 无 cache 模式下的全程总前向 token 数也会等比例缩减为 1/4 吗?',
      answer: '缩减幅度远超线性比例: 无 cache 模式的全程前向总量从 688 骤降至 76, 约为原来的 1/9 而非 1/4。\n两种机制的计算复杂度对比如下:\n- 无 cache 模式的平方增长: 第 $t$ 步需重复前向处理 $P + t$ 个 token, 全程 $N$ 步累加总量为 $N \\cdot P + N(N-1)/2$, 包含关于生成长度的二次项 $N^2$; 因此当 $N$ 缩短为 1/4 时, 二次主导项以平方比例迅速收缩。\n- 有 KV cache 模式的线性增长: 全程累计前向总量为 $P + N - 1$ (仅包含第 0 步 prefill 的 $P$ 个位置以及后续 $N-1$ 个单 token 解码步), 仅从 37 线性降至 13。\nKV cache 的本质是以 $O(T)$ 的线性显存占用 (为每个上下文位置在每一层持久化存储一份 K/V 向量) 换取消除 $O(T^2)$ 的冗余历史重算。如何压缩这块显存占用、消除其碎片化分配并在多请求间高效共享复用, 构成了后续推理系统优化章节的核心主线。',
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
      <div class="kv"><span>累计前向 (到第 {{ t }} 步一共过了多少 token)</span><b>{{ cur.cum }}</b></div>
      <div class="kv"><span>全程总量: 无 / 有 cache</span><b>{{ total.no }} / {{ total.yes }} ({{ (total.no / total.yes).toFixed(1) }}×)</b></div>
      <div class="kv"><span>cache 里存着的 token 数 (每层各一份 K/V)</span><b>{{ cache ? P + t : 0 }} × 层数</b></div>
      <div class="lab-note">
        <p>
          闭式: 无 cache <Tex text="$= N \cdot P + N(N-1)/2 =$" /> {{ N * P + N * (N - 1) / 2 }}; 有 cache <Tex text="$= P + N - 1 =$" /> {{ P + N - 1 }}。
          与上面逐步累加的数相同。
        </p>
        <p>KV cache 省去的是历史 token 在所有 Transformer 层上的线性 K/V 投影、前馈网络 MLP 及层归一化重算开销, 避免在每个解码步将全部历史上下文重新前向传播一遍。</p>
        <p>但新 token 的 query 仍要和全部 {{ P + t }} 个 key 做点积, 每步的访存量依旧是 <Tex text="$O(t)$" />: 要把整个 cache 读一遍只为算 1 个 token。所以长上下文 decode 是带宽瓶颈, 不是算力瓶颈。</p>
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
