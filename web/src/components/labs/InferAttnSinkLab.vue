<!-- Attention sink 实验台 (对应 llm_infer/m16_attention_sinks): 合成注意力 + 人为植入的 sink, 不是真模型的测量。 -->
<template>
  <LabFrame
    title="Attention sinks — 为什么滑动窗口不能扔掉开头几个 token"
    sub="- 模拟什么: 流式场景 (长对话、实时字幕) 里 token 无限增长, KV cache (模型存下来的每个旧 token 的 key 和 value, 新 token 靠它们算注意力) 装不下, 只能逐出一部分。最直接的做法是只留最近 window 个 token (滑动窗口)。但训练过的 LLM 有个习惯: 当前 token 没什么可看时, softmax 会把多余的注意力倒在开头几个 token 上, 这几个 token 叫 attention sink (注意力的「垃圾桶」)。把它们逐出, softmax 的分母突然少一大块, 输出就畸变了。StreamingLLM 的做法是开头几个 sink 永远不逐出。这里的注意力是合成的: 人为在开头 4 个 token 上植入了 sink, 因为随机权重的 TinyLM 没训练过, 长不出这个现象。
      - 图上是什么: 两行柱状图, 横轴都是 token 的位置 (最左是位置 0, 最老的 token; 最右是当前正在生成的 token), 柱高是「当前 token 看这个位置」的注意力权重。第一行是完整上下文下的注意力: 黄色柱是开头 4 个 sink, 实心柱是还留在 cache 里的 token, 半透明柱是被逐出的。第二行是逐出之后的注意力: 被逐出的 key 不存在了, softmax 只在幸存的 token 里重新归一化 (所有权重重新凑成 1), 柱子会整体变高。下面一排小格是 cache 的槽位, 格里的数字是住在这个槽的 token 的绝对位置, 黄色的是 sink; 点一格, 下方会显示它喂给 RoPE (按位置给 key 做旋转编码, 模型靠它知道远近) 的位置是多少。
      - 可以做什么: 拖「流长度 T」改一共流过了多少 token; 拖「最近窗口 window」改保留最近几个; 拖「保留的 sink 数」在 0 (纯滑动窗口) 到 4 之间改留几个开头 token; 拖「sink 强度」改植入的 sink 有多强, 0 就是没有 sink 现象 (模拟随机权重模型)。「位置按槽位重编号」和「沿用绝对位置」切换 RoPE 用哪种位置: 前者按 cache 槽位 0, 1, 2 … 重新编号, 后者用 token 原来的绝对位置。「换一组」换一组随机噪声。
      - 应该看到: sink 数拖到 0 时, 第一行黄色的大柱子变成半透明 (被逐出), 第二行幸存的柱子被整体放大好几倍。右侧「逐出丢掉的注意力」是被逐出的 token 原本分到的权重之和, 越大越糟; 「幸存权重被放大」等于 1 除以 (1 − 丢掉的部分), 是幸存的每个权重被乘上的倍数, 离 1 越远, 注意力输出离训练时见过的分布越远。「cache 里的 token / 流过的 token」是 cache 实际存了几个, 对比流过了多少。「喂给 RoPE 的最大位置」在「沿用绝对位置」时等于 T − 1, 超过训练长度 64 就标红: 模型没见过这么大的位置, 这就是为什么要重编号。"
    module="llm_infer/m16"
    run="python -m llm_infer.m16_attention_sinks.demo"
    :challenge="{
      ask: '把「保留的 sink 数」从 4 拖到 0 (纯滑动窗口), 幸存权重会被放大多少倍?',
      answer: '默认参数下 (T = 96, window = 16, sink 强度 4), 「幸存权重被放大」从 1.24× 涨到 4.70×, 「逐出丢掉的注意力」从 19.2% 涨到 78.7%。\n为什么差这么多: softmax 的权重之和恒为 1, 植入的 sink 让开头 4 个 token 吃掉了大部分权重 (第一行最高的那几根黄色柱子)。\n- 纯滑动窗口 (sink 数 = 0): 这 4 个 token 被逐出, 分母一下少了近 80%, 幸存的每个权重都要乘 $1/(1-0.787) \\approx 4.7$ 才能重新凑成 1。注意力输出离训练时见过的分布很远, 真实模型上表现为困惑度爆炸。\n- 留住 4 个 sink: 分母基本没变, 丢掉的只是中间那些本来就没分到多少注意力的 token, 放大只有 1.24×。\n再把「sink 强度」拖到 0 (没有 sink 现象): 留 4 个是 1.86×, 不留是 1.95×, 留不留开头几乎没差别。这就是随机权重的 TinyLM 上看不到效果的原因: 它没训练过, 开头 token 没有被养成垃圾桶。',
    }"
  >
    <template #controls>
      <LabSlider v-model="T" label="流长度 T" :min="16" :max="128" unit=" tok" />
      <LabSlider v-model="win" label="最近窗口 window" :min="4" :max="64" unit=" tok" />
      <LabSlider v-model="nSink" label="保留的 sink 数" :min="0" :max="N_SINK" />
      <LabSlider v-model="strength" label="sink 强度 (植入的 logit)" :min="0" :max="6" :step="0.5" />
      <div class="row">
        <button type="button" :class="{ active: reindex }" @click="reindex = true">位置按槽位重编号</button>
        <button type="button" :class="{ active: !reindex }" @click="reindex = false">沿用绝对位置</button>
        <button type="button" @click="seed++">换一组</button>
      </div>
    </template>

    <svg :viewBox="`0 0 ${W} ${2 * RH + 56}`" role="img" aria-label="逐出前后的注意力分布">
      <g v-for="(row, ri) in [m.full, m.renorm]" :key="ri" :transform="`translate(0, ${ri * (RH + 22) + 12})`">
        <text x="0" y="-2" class="ax">{{ ri ? '② 逐出后: 只在 cache 里幸存的 token 之间重新归一化' : '① 完整上下文: 当前 token 对每个位置的注意力 (半透明的将被逐出)' }}</text>
        <rect
          v-for="(p, j) in row" :key="j" :x="j * bw" :y="RH - (p / m.peak) * (RH - 10)" :width="Math.max(1, bw - 0.6)"
          :height="(p / m.peak) * (RH - 10)" :class="['bar', j < N_SINK ? 'sink' : '', { hl: j === hl }]" :opacity="m.kept[j] ? 1 : ri ? 0 : 0.25"
        />
        <line x1="0" :x2="W" :y1="RH" :y2="RH" class="base" />
      </g>
      <text x="0" :y="2 * RH + 50" class="ax">位置 0 (最老)</text>
      <text :x="W" :y="2 * RH + 50" class="ax" text-anchor="end">位置 {{ T - 1 }} (当前 query)</text>
    </svg>

    <p class="cap">cache 槽位 (共 {{ m.slots.length }} 个): 格里的数字是住在这个槽的 token 的绝对位置, 黄色是 sink。点一格看它喂给 RoPE 的位置:</p>
    <div class="slots">
      <button
        v-for="(abs, s) in m.slots" :key="s" type="button" class="cell slot"
        :class="[abs < N_SINK ? 'hot' : 'on', { hl: abs === hl }]" :title="`槽位 ${s}: 住着绝对位置 ${abs} 的 token`"
        @click="hl = abs" @mouseenter="hl = abs"
      >{{ abs }}</button>
    </div>
    <p v-if="hlSlot >= 0" class="lab-note detail">
      绝对位置 <b>{{ hl }}</b> 的 token 现在住在槽位 <b>{{ hlSlot }}</b>; 喂给 RoPE 的位置是 <b>{{ reindex ? hlSlot : hl }}</b>,
      与当前 query 的距离是 {{ reindex ? m.slots.length - 1 - hlSlot : T - 1 - hl }}。
    </p>

    <template #stats>
      <div class="kv"><span>逐出丢掉的注意力 (被逐出 token 的权重之和)</span><b :class="m.lost > 0.4 ? 'bad' : m.lost < 0.25 ? 'good' : ''">{{ (m.lost * 100).toFixed(1) }}%</b></div>
      <div class="kv"><span>幸存权重被放大 (重新归一化后)</span><b :class="m.lost > 0.4 ? 'bad' : m.lost < 0.25 ? 'good' : ''">{{ (1 / (1 - m.lost)).toFixed(2) }}×</b></div>
      <div class="kv"><span>cache 里的 token / 流过的 token</span><b>{{ m.slots.length }} / {{ T }}</b></div>
      <div class="kv">
        <span>喂给 RoPE 的最大位置 (训练长度 {{ TRAIN }})</span>
        <b :class="maxPos >= TRAIN ? 'bad' : 'good'">{{ maxPos }}</b>
      </div>
      <div class="lab-note">
        <p><Tex text="放大倍数 $= 1 / (1 - \text{丢掉的注意力})$: softmax 的分母少了多少, 幸存的每个权重就等比例涨多少。" /></p>
        <p><Tex text="★ 重编号: 位置 = cache 槽位 $0, \dots, L-1$ ($L$ 是 cache 里的 token 数), 永远不超过训练长度, 流再长也不会越界。" /></p>
        <p>
          为了每步都能按新槽位重转, <code class="inline">SinkCache</code> 里存的是<b>未旋转 (pre-RoPE)</b> 的 K。
          普通 KV cache 存的是已经按原位置旋转过的 K, 位置烙死在里面, 没法改号。代价是每步要把 cache 里全部 K 重新旋转一遍, <Tex text="$O(L \cdot D)$" />。
        </p>
        <p>被逐出的中间内容是真的忘了: sink 只保证流不崩, 不保证记得很久以前的内容。</p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import Tex from '@/components/Tex.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { mulberry32, randn, softmax, range, sum } from '@/utils/labmath.js'

const W = 640, RH = 90, TRAIN = 64
const N_SINK = 4   // sink 只植入在开头 4 个 token 上, 所以最多也只值得留 4 个
const T = ref(96), win = ref(16), nSink = ref(4), strength = ref(4), seed = ref(1)
const reindex = ref(true), hl = ref(0)

const m = computed(() => {
  const rand = mulberry32(seed.value), n = T.value
  const noise = range(128).map(() => 0.4 * randn(rand))                    // 先抽满 128 个, 拖 T 时旧 token 的噪声不变
  // 合成 logit: 开头 4 个 token 的 sink 加成 + 越近越高的 recency + 噪声
  const logits = range(n).map((j) => (j < N_SINK ? strength.value : 0) + 3 * Math.exp(-(n - 1 - j) / 6) + noise[j])
  const full = softmax(logits)
  const kept = range(n).map((j) => j < nSink.value || j >= n - win.value) // ★ SinkCache 留下的: 开头 n_sink 个 + 最近 window 个
  const lost = sum(full.filter((_, j) => !kept[j]))
  const renorm = full.map((p, j) => (kept[j] ? p / (1 - lost) : 0))       // 被逐出的 key 不存在了 → softmax 只在幸存者里归一
  return { full, kept, lost, renorm, peak: Math.max(...renorm), slots: range(n).filter((j) => kept[j]) }
})
const bw = computed(() => W / T.value)
const hlSlot = computed(() => m.value.slots.indexOf(hl.value))
const maxPos = computed(() => (reindex.value ? m.value.slots.length - 1 : T.value - 1))
</script>

<style scoped>
svg { min-width: 520px; } /* 窄屏: 图保持可读, 由 .lab-viz 横向滚动 */
.bar { fill: var(--accent); }
.bar.sink { fill: var(--warn); }
.bar.hl { fill: var(--text); }
.base { stroke: var(--border-strong); }
.ax { font-size: 11px; fill: var(--text-dim); }
.cap { font-size: 12px; color: var(--text-muted); margin: 12px 0 6px; }
.slots { display: flex; flex-wrap: wrap; gap: 3px; }
.slot { min-width: 26px; padding: 0 2px; cursor: pointer; font-size: 10px; min-height: 0; }
.slot.hl { outline: 2px solid var(--text); outline-offset: 1px; }
.detail { margin-top: 10px; }
.detail b { color: var(--text); }
</style>
