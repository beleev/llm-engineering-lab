<!--
  采样流水线实验台 (对应 llm_infer/m10_sampling/samplers.py:sample)。
  一件事: 采样参数不是一堆独立旋钮, 而是一条有先后顺序的流水线 —— 每一级看到的都是上一级处理过的分布。
-->
<template>
  <LabFrame
    title="采样流水线 — 重复惩罚、温度、top-k、top-p、min-p 按顺序各砍一刀"
    sub="- 模拟什么: 自回归生成每一步中, 模型对词表内所有候选 token 输出原始未归一化得分 (logits), 经 softmax 转化为概率分布后再按概率随机抽取下一个输出 token, 这一过程称为解码采样。各项采样超参数并非彼此独立的调节旋钮, 而是串联成一条具有严格先后顺序的处理流水线, 每一级算子接收并修改上一级输出的分布。本实验以「今天天气真」之后的 16 个候选 token 为例完整演示五级采样流水线:
  1. 重复惩罚: 压低上文历史中已出现过的 token 的 logit。
  2. 温度缩放: 将所有 logit 除以温度 $T$ 后执行 softmax; $T$ 增大使分布趋于平坦 (生成更具随机性), $T$ 减小使分布趋于尖锐 (生成更趋确定)。
  3. top-k 过滤: 仅保留当前概率排名最高的 $k$ 个候选。
  4. top-p (核采样) 过滤: 按概率从高到低累加, 首次达到或超过阈值 $p$ 时截断并剔除后续尾部候选。
  5. min-p 过滤: 剔除概率低于「当前最高单 token 概率 × min_p」的候选。
- 图上是什么: 每根柱子对应一个候选 token, 按概率由高到低排列, 柱顶数字为对应概率值:
  - 实线空心外框: 截断过滤前 (仅历经重复惩罚与温度缩放) 的原始概率。
  - 紫色实心柱: 历经全部过滤后在保留候选集上重新归一化的最终采样概率。
  - 仅剩虚线空心框的列: 已被截断剔除的候选 token, 柱底标注具体将其剔除的过滤阶段 (top-k / top-p / min-p)。
  - 标有「已出现」的候选: 已加入上下文历史、当前正承受重复惩罚的 token。
  - 红色竖直虚线: top-k 的截断边界, 位于该线右侧的候选全部被 top-k 剔除。
- 可以做什么: 调节滑杆、顺序开关并直接点击候选柱交互:
  - 五根滑杆分别对应流水线的五道工序, 滑杆前的圆圈序号标示默认执行顺序。
  - 将「top-k」拖至 0 表示关闭该级过滤, 也可直接左右拖动图中的红色竖直虚线。
  - 切换「温度放在过滤之后」执行流水线顺序对照实验: 将温度缩放移至最后一级后, 三级过滤器看到的是未缩放 ($T = 1$) 的基准概率分布。
  - 点击任意候选柱将其加入或移出「已出现过」的上文历史集合。
  - 点击「清空历史」移除全部历史标记, 点击「换一组 logits」重新生成候选得分。
- 应该看到: 右侧展示四项核心统计读数, 拖动温度滑杆时可观察执行顺序带来的差异:
  - 「存活」对比过滤后最终保留的候选 token 数, 并列给出温度在前 (默认流水线) 与温度「在后」两种顺序下的存活数量。
  - 「保留的原始概率质量」统计所有存活候选在过滤前的原始概率之和, 低于一半时标红提示截断过重。
  - 「熵」度量最终采样分布的离散程度 (单位为 nats), 截断过滤并重归一化后熵必然降低。
  - 「各级砍掉」分别统计三级过滤器各自剔除的候选个数。
  - 在默认 top-k 设为 8、top-p 设为 0.9 时共存活 5 个候选; 调高温度使分布变平后, top-p 为凑足 0.9 的累计概率质量需纳入更多候选, 默认顺序下的存活数随之增加, 而温度「在后」的对照存活数保持不变。"
    module="llm_infer/m10"
    run="python -m llm_infer.m10_sampling.demo"
    :challenge="{
      ask: '在默认 top-k 为 8、top-p 为 0.9 的配置下共有 5 个候选存活。若将温度从 1.0 拖至 2.0, 存活候选数会增加还是减少?',
      answer: '存活候选数会增加, 从 5 个变为 7 个。\n由于默认流水线中温度缩放位于截断过滤之前, 调节温度会直接改变后续各级过滤器所接收的输入概率分布。当温度 $T$ 从 1.0 调高至 2.0 时, 所有候选的 logit 均除以 2, 原本高分与低分之间的绝对差距缩小, 经 softmax 后概率分布趋于平坦: 头部最高概率下降, 尾部候选概率上升。三级过滤器对此表现出不同的响应特性:\n- top-p: 需按概率从高到低累加至达到 0.9 才停止。分布变平后头部单个候选贡献的概率减小, 凑足 0.9 累计质量需要纳入更多候选, 因此保留数量增加。\n- min-p: 截断阈值定义为 $\\text{min\\_p} \\times p_{\\max}$。分布变平使最大概率 $p_{\\max}$ 下降, 截断门槛随之等比降低, 保留的候选同样增多。\n- top-k: 仅依据概率名次固定保留前 $k$ 个候选, 由于温度缩放是严格保序变换, 其截断集合与温度无关。\ntop-p 与 min-p 的共同优势在于能够根据分布熵值自适应伸缩候选池 (模型分布越平坦不确定时保留越多候选, 分布越尖锐确定时截断越严格), 这是固定名次的 top-k 无法实现的。\n若开启「温度放在过滤之后」再拖动温度滑杆, 三级过滤器接收的始终是 $T = 1$ 的原始概率分布, 存活候选数将固定停留在 5 个; 此时温度仅改变已确定存活集合内部的最终重归一化抽样概率, 不再影响候选集边界。\n工程实践中同一组采样超参数在不同推理框架 (如 vLLM、Hugging Face Transformers、llama.cpp) 下生成行为存在差异, 往往正是源于各框架内部采样算子 (例如 min-p 与 top-p 的先后次序, 或温度与过滤的相对位置) 的流水线顺序不完全统一。',
    }"
  >
    <template #controls>
      <LabSlider v-model="pen" label="① 重复惩罚" :min="1" :max="2" :step="0.05" :format="(t) => t.toFixed(2)" />
      <LabSlider v-model="T" :label="`${NO.T} 温度 T`" :min="0.2" :max="3" :step="0.1" :format="(t) => t.toFixed(1)" />
      <LabSlider v-model="topK" :label="`${NO.k} top-k (0 = 关)`" :min="0" :max="16" />
      <LabSlider v-model="topP" :label="`${NO.p} top-p`" :min="0.05" :max="1" :step="0.05" :format="(t) => t.toFixed(2)" />
      <LabSlider v-model="minP" :label="`${NO.m} min-p`" :min="0" :max="0.5" :step="0.01" :format="(t) => t.toFixed(2)" />
      <div class="row">
        <button type="button" :class="{ active: tempLast }" :aria-pressed="tempLast" @click="tempLast = !tempLast">温度放在过滤之后</button>
        <button type="button" @click="seed++">换一组 logits</button>
        <button type="button" @click="hist = new Set()">清空历史</button>
      </div>
    </template>

    <svg ref="svg" :viewBox="`0 0 ${X0 + V * BW + 10} 250`" role="group" aria-label="16 个候选 token 按概率排序的柱状图: 空心是过滤前, 实心是过滤后">
      <line :x1="X0" :x2="X0 + V * BW" y1="190" y2="190" class="axis" />
      <g v-for="(b, r) in bars" :key="b.i" tabindex="0" role="button" class="bar"
        :aria-label="`${b.tok}: ${hist.has(b.i) ? '移出' : '放进'}历史`" :aria-pressed="hist.has(b.i)"
        @click="toggle(b.i)" @keydown.enter="toggle(b.i)">
        <rect :x="X0 + r * BW" y="10" :width="BW" height="232" class="hit" />
        <rect :x="X0 + r * BW + 4" :y="190 - b.pre * SCALE" :width="BW - 8" :height="b.pre * SCALE" class="pre" :class="{ cut: b.by }" />
        <rect v-if="!b.by" :x="X0 + r * BW + 4" :y="190 - b.post * SCALE" :width="BW - 8" :height="b.post * SCALE" class="post" />
        <text :x="X0 + (r + 0.5) * BW" :y="184 - Math.max(b.pre, b.post) * SCALE" class="num">{{ (b.by ? b.pre : b.post).toFixed(2) }}</text>
        <text :x="X0 + (r + 0.5) * BW" y="206" class="tok" :class="{ h: hist.has(b.i) }">{{ b.tok }}</text>
        <text v-if="hist.has(b.i)" :x="X0 + (r + 0.5) * BW" y="220" class="tag h">已出现</text>
        <text v-if="b.by" :x="X0 + (r + 0.5) * BW" y="234" class="tag" :class="b.by">{{ NAMES[b.by] }}</text>
      </g>
      <g class="draggable focus-inset" tabindex="0" role="slider" aria-label="top-k 的切线: 左右拖动改变 top-k, 线右边的 token 被砍掉" :aria-valuenow="topK" aria-valuemin="0" aria-valuemax="16"
        @pointerdown="start($event, { svg, onMove })"
        @keydown.left.prevent="topK = clamp((topK || V) - 1, 1, V)" @keydown.right.prevent="topK = topK && topK < V - 1 ? topK + 1 : 0">
        <rect :x="cutX - 8" y="4" width="16" height="190" class="grab" />
        <line :x1="cutX" :x2="cutX" y1="14" y2="192" class="cutline" />
        <text :x="Math.min(cutX, X0 + V * BW - 22)" y="10" class="cutlbl">top-k = {{ topK || '关' }}</text>
      </g>
    </svg>

    <template #stats>
      <div class="kv"><span>存活的候选数: 温度在过滤前 / 温度在过滤后</span><b>{{ aliveBoth[0] }} / {{ aliveBoth[1] }}</b></div>
      <div class="kv"><span>幸存者在过滤前的概率加起来 (保留的原始概率质量)</span><b :class="mass < 0.5 ? 'bad' : 'good'">{{ (mass * 100).toFixed(1) }}%</b></div>
      <div class="kv"><span>熵 (分布有多分散, nats): 过滤前 → 过滤后</span><b>{{ hPre.toFixed(2) }} → {{ hPost.toFixed(2) }}</b></div>
      <div class="kv"><span>各级砍掉的 token 数</span><b class="sm">top-k {{ cutBy.k }} · top-p {{ cutBy.p }} · min-p {{ cutBy.m }}</b></div>
      <div class="lab-note">
        <ul class="pts">
          <li><b>重复惩罚:</b> 出现过的 token, 正 logit 除以惩罚系数、负 logit 乘以惩罚系数, 两种情况目标都是「让它变小」(负数乘以大于 1 的数更负)。它在温度之前, 所以就算温度为 0 的 greedy 也受影响。</li>
          <li><b>温度:</b> logit 除以 <Tex text="$T$" /> 再 softmax。<Tex text="$T$" /> 大于 1 时分布变平 (更随机), 小于 1 时变尖 (更保守), 它不砍任何 token。</li>
          <li><b>top-p:</b> 按概率从高到低累加, 留下「累计质量首次 <Tex text="$\ge p$" /> 的最小集合」, 所以模型越犹豫留得越多。</li>
          <li><b>min-p:</b> 砍掉 <Tex text="$p \lt  \text{min\_p} \cdot p_{\max}$" /> 的, 阈值跟着最高概率走, 模型越确定砍得越狠。</li>
        </ul>
        <p>后面的级只看到前面留下的 token, 每砍一刀都在幸存者上重新归一化。温度放到过滤之后, 三级过滤看到的就是 <Tex text="$T = 1$" /> 的分布, 存活数不再跟着温度变。</p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import Tex from '@/components/Tex.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { useDrag } from '@/composables/useDrag.js'
import { clamp, entropy, mulberry32, randn, softmax, sum } from '@/utils/labmath.js'

const TOKS = ['好', '不错', '热', '冷', '棒', '糟', '的', '舒服', '闷', '差', '怪', '是', '吗', '了', '猫', '量子']
const BASE = [4.2, 3.6, 3.1, 2.8, 2.4, 1.9, 1.5, 1.2, 0.8, 0.4, 0, -0.5, -1, -1.5, -2.5, -3.5]
const NAMES = { k: 'top-k', p: 'top-p', m: 'min-p' }
const V = TOKS.length, X0 = 10, BW = 34
// 默认开着 top-k 和 top-p: 一打开就能看到两级过滤各砍了谁
const pen = ref(1), T = ref(1), topK = ref(8), topP = ref(0.9), minP = ref(0), seed = ref(1)
const tempLast = ref(false)
const NO = computed(() => (tempLast.value ? { T: '⑤', k: '②', p: '③', m: '④' } : { T: '②', k: '③', p: '④', m: '⑤' }))
const hist = ref(new Set())
const svg = ref(null)
const { start } = useDrag()
const toggle = (i) => { const s = new Set(hist.value); s.has(i) ? s.delete(i) : s.add(i); hist.value = s }

const logits = computed(() => { const rand = mulberry32(seed.value * 31 + 5); return BASE.map((z) => z + (seed.value > 1 ? 0.8 * randn(rand) : 0)) })

// ★ 与 samplers.sample 同序: rep penalty → /T → top-k → top-p → min-p; 每一级只在上一级的幸存者上重新 softmax
// late = true 是对照组: 三级过滤先在 T = 1 的分布上做完, 最后才除温度
const run = (late) => {
  let z = logits.value.map((v, i) => (hist.value.has(i) && pen.value !== 1 ? (v > 0 ? v / pen.value : v * pen.value) : v))
  if (!late) z = z.map((v) => v / T.value)
  const pre = softmax(z), by = Array(V).fill('')
  const order = () => z.map((v, i) => i).filter((i) => z[i] > -Infinity).sort((a, b) => z[b] - z[a])
  const kill = (ids, tag) => ids.forEach((i) => { z[i] = -Infinity; by[i] = tag })
  if (topK.value > 0 && topK.value < V) kill(order().slice(topK.value), 'k')
  if (topP.value < 1) {
    const p = softmax(z), o = order()
    let cum = 0, keep = 0
    while (keep < o.length) { cum += p[o[keep++]]; if (cum >= topP.value - 1e-12) break }   // 首次 ≥ p 的那个也留下
    kill(o.slice(keep), 'p')
  }
  if (minP.value > 0) { const p = softmax(z), pm = Math.max(...p); kill(order().filter((i) => p[i] < minP.value * pm), 'm') }
  return { pre, post: softmax(z, late ? T.value : 1), by, alive: by.filter((b) => !b).length }
}
const pipe = computed(() => run(tempLast.value))
const aliveBoth = computed(() => [run(false).alive, run(true).alive])

const bars = computed(() => TOKS.map((tok, i) => ({ i, tok, pre: pipe.value.pre[i], post: pipe.value.post[i], by: pipe.value.by[i] }))
  .sort((a, b) => b.pre - a.pre))
const alive = computed(() => bars.value.filter((b) => !b.by))
const mass = computed(() => sum(alive.value.map((b) => b.pre)))
const hPre = computed(() => entropy(pipe.value.pre)), hPost = computed(() => entropy(pipe.value.post))
const cutBy = computed(() => Object.fromEntries(['k', 'p', 'm'].map((t) => [t, bars.value.filter((b) => b.by === t).length])))

// 纵轴按最大概率向上取整到 0.25 的倍数: 柱子够高, 又不会每动一下就缩放
const SCALE = computed(() => 165 / (Math.ceil(Math.max(...pipe.value.pre, ...pipe.value.post) * 4) / 4))
const cutX = computed(() => X0 + (topK.value || V) * BW)
const onMove = ({ x }) => { const k = clamp(Math.round((x - X0) / BW), 1, V); topK.value = k >= V ? 0 : k }
</script>

<style scoped>
/* 窄屏: 保住可读的最小宽度, 由 .lab-viz 横向滚动; 只有 .draggable 把手拦截触摸 */
svg { min-width: 540px; touch-action: pan-x pan-y; }
.axis { stroke: var(--border-strong); }
.bar { cursor: pointer; outline: none; }
.hit { fill: transparent; }
.bar:hover .hit, .bar:focus-visible .hit { fill: var(--accent-soft); }
.pre { fill: none; stroke: var(--text-dim); stroke-width: 1; }
.pre.cut { stroke-dasharray: 3 2; }
.post { fill: var(--accent); opacity: 0.85; }
.num { font-size: 8px; fill: var(--text-muted); text-anchor: middle; font-family: "SF Mono", Menlo, monospace; }
.tok { font-size: 11px; fill: var(--text); text-anchor: middle; }
.tok.h { fill: var(--warn); font-weight: 600; }
.tag { font-size: 8px; text-anchor: middle; fill: var(--text-dim); }
.tag.h { fill: var(--warn); }
.tag.k { fill: var(--danger); } .tag.p { fill: var(--eye); } .tag.m { fill: var(--right); }
.grab { fill: transparent; }
.cutline { stroke: var(--danger); stroke-width: 2; stroke-dasharray: 5 3; }
.cutlbl { font-size: 9px; fill: var(--danger); text-anchor: middle; }
.sm { font-size: 12px !important; }
</style>
