<!--
  采样流水线实验台 (对应 llm_infer/m10_sampling/samplers.py:sample)。
  一件事: 采样参数不是一堆独立旋钮, 而是一条有先后顺序的流水线 —— 每一级看到的都是上一级处理过的分布。
-->
<template>
  <LabFrame
    title="采样流水线 — 重复惩罚、温度、top-k、top-p、min-p 按顺序各砍一刀"
    sub="- 模拟什么: 模型每一步给词表里每个 token 打一个分 (logit), 过 softmax 变成概率, 再按概率抽一个作为输出, 这一步叫采样。采样参数不是互相独立的旋钮, 而是一条有先后顺序的流水线, 每一级只能看到上一级处理过的分布。这个台子用「今天天气真」后面的 16 个候选 token 走一遍这条流水线, 五级依次是:
        1. 重复惩罚把已经出现过的 token 的 logit 压低。
        2. 温度 $T$ 把所有 logit 除以 $T$ 再 softmax, $T$ 大分布变平 (更随机), $T$ 小分布变尖 (更保守)。
        3. top-k 只留概率最高的 $k$ 个。
        4. top-p 按概率从高到低累加, 加到 $p$ 为止, 后面的全砍。
        5. min-p 砍掉概率不到「最高概率 × min_p」的。
      - 图上是什么: 每根柱子是一个候选 token, 按概率从高到低排, 柱顶的数字是对应的值:
        - 空心框: 过滤前 (只做了重复惩罚和温度) 的概率。
        - 紫色实心柱: 过滤后、在幸存者上重新归一化的概率。
        - 只剩虚线空心框的: 被砍掉的 token, 下面标着是哪一级砍的 (top-k / top-p / min-p)。
        - 标「已出现」的 token: 被放进历史、正在吃重复惩罚的。
        - 红色竖虚线: top-k 的切线, 线右边的全被 top-k 砍掉。
      - 可以做什么: 滑杆、开关和柱子:
        - 五根滑杆对应流水线的五级, 前面的圈号就是执行顺序。
        - 「top-k」拖到 0 表示关掉, 也可以直接左右拖图里的红色竖线。
        - 「温度放在过滤之后」是一个对照实验, 把温度挪到最后一级, 三级过滤看到的就是 $T = 1$ 的分布。
        - 点任意柱子把这个 token 放进或移出「已出现过」的历史。
        - 「清空历史」全部移出; 「换一组 logits」换一组随机分数。
      - 应该看到: 右侧四个读数, 最后一条是拖温度时的变化:
        - 「存活」是过滤后还剩几个候选, 两个数分别是温度在前 (默认顺序) 和温度在后两种顺序下的结果。
        - 「保留的原始概率质量」是幸存者在过滤前的概率加起来占多少, 低于一半标红, 说明砍得太狠。
        - 「熵」衡量分布有多分散 (单位 nats), 过滤后一定变小。
        - 「各级砍掉」是三级各砍了几个。
        - 默认 top-k=8、top-p=0.9 时存活 5 个; 把温度拖大, 分布变平, top-p 要凑够 0.9 需要更多 token, 存活数增加, 而「在后」那个数纹丝不动。"
    module="llm_infer/m10"
    run="python -m llm_infer.m10_sampling.demo"
    :challenge="{
      ask: '默认 top-k = 8、top-p = 0.9, 存活 5 个。把温度从 1.0 拖到 2.0, 存活数变多还是变少?',
      answer: '变多: 5 个变成 7 个。\n温度在过滤之前, 它改的是过滤器看到的分布。$T$ 从 1.0 拖到 2.0, 所有 logit 除以 2, 高分降得多、低分降得少, softmax 之后分布变平: 最高概率下降, 尾部概率上升。三级过滤对这件事的反应不一样:\n- top-p: 要从高到低累加到 0.9 才停。分布平了, 每个 token 分到的概率小了, 凑够 0.9 需要更多 token, 留下的变多。\n- min-p: 阈值是 $\\text{min\\_p} \\times p_{\\max}$。分布平了 $p_{\\max}$ 下降, 阈值跟着下降, 留下的也变多。\n- top-k: 不看概率, 永远留前 $k$ 个, 与温度无关。\n前两者的共同点是「模型越犹豫留得越多, 越确定砍得越狠」, 固定的 top-k 做不到这一点。\n打开「温度放在过滤之后」再拖温度: 三级过滤看到的都是 $T = 1$ 的分布, 存活数停在 5; 温度只改最后抽签时各候选的概率, 不再改候选集。\n同一组参数在不同推理框架里结果不同, 多半是顺序不同: vLLM、HF、llama.cpp 对 min-p 放在 top-p 之前还是之后处理得不完全一致。',
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
          <li><b>min-p:</b> 砍掉 <Tex text="$p < \text{min\_p} \cdot p_{\max}$" /> 的, 阈值跟着最高概率走, 模型越确定砍得越狠。</li>
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
