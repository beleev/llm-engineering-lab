<!-- Attention sink 实验台 (对应 llm_infer/m16_attention_sinks): 合成注意力 + 人为植入的 sink, 不是真模型的测量。 -->
<template>
  <LabFrame
    title="Attention sinks — 为什么滑动窗口不能扔掉开头几个 token"
    sub="- 模拟什么: 在长对话或实时流式生成场景中, 序列 token 数持续增长, 受显存容量限制必须逐出部分旧的 KV cache。直观做法是仅保留最近 window 个 token (滑动窗口), 但经过预训练的 LLM 普遍存在一个分布特性: 当新生成的 token 缺乏明确的上下文关注对象时, softmax 会将富余的注意力权重集中分配给序列起始处的若干个 token, 即 attention sink (注意力汇聚点)。若滑动窗口直接逐出这些起始 token, softmax 分母会骤然缺失主要项, 导致剩余位置的注意力分布严重失真。StreamingLLM 的解法是在滑动窗口之外永久保留序列开头的若干个 sink token。注: 由于未训练的随机权重 TinyLM 不会自发涌现该现象, 本实验使用合成注意力分布, 在序列前 4 个 token 上显式注入了 sink 偏置。
- 图上是什么: 主体为上下两行注意力权重柱状图, 横轴均为序列绝对位置 (最左端位置 0 为最早的 token, 最右端为当前生成步), 柱高代表当前步对该位置的注意力权重:
  - 第一行: 完整上下文下的原始注意力分布。黄色柱代表起始处的 4 个 sink token, 实心柱代表仍保留在 cache 中的 token, 半透明柱代表已被逐出的历史 token。
  - 第二行: 逐出旧键值后的实际注意力分布。被逐出位置的 key 已从缓存移除, softmax 仅在留存 token 集合上重新归一化 (权重总和仍为 1), 因此留存柱高度会被整体等比例放大。
  - 下方槽位格: 展示物理 KV cache 的各个槽位, 格内数字为当前驻留 token 的原始绝对位置, 黄色格对应 sink 槽位; 点击任意一格可在下方查看该槽位实际传入 RoPE 的位置索引。
- 可以做什么: 提供四个滑杆与三个切换按钮:
  - 拖动「流长度 T」: 调整已流过的历史 token 总数。
  - 拖动「最近窗口 window」: 调整滑动窗口保留的近期 token 数量。
  - 拖动「保留的 sink 数」: 在 0 (纯滑动窗口) 至 4 之间调整永久保留的起始 token 数量。
  - 拖动「sink 强度」: 调节起始位置的汇聚偏置强度, 设为 0 即退化为无 sink 现象的均匀噪声分布 (对应随机权重模型)。
  - 点击「位置按槽位重编号」与「沿用绝对位置」: 切换 RoPE 编码使用的位置索引来源, 前者按当前缓存槽位 0, 1, 2 … 连续重编号, 后者沿用 token 在原始流中的绝对位置。
  - 点击「换一组」: 重新采样一组随机注意力分数。
- 应该看到: 将保留的 sink 数拖至 0 时, 第一行高耸的黄色柱变为半透明 (被逐出), 第二行留存 token 的权重柱被整体放大数倍。右侧读数含义如下:
  - 「逐出丢掉的注意力」: 被逐出 token 在完整上下文下原本占据的注意力权重之和, 数值越大代表分布截断损失越严重。
  - 「幸存权重被放大」: 等于 1 除以 (1 − 逐出权重之和), 表示每个留存位置的注意力权重被统一乘上的重归一化倍数; 偏离 1 越远, 注意力输出越偏离训练期分布。
  - 「cache 里的 token / 流过的 token」: 对比当前缓存实际占用的槽位数与累计流过的序列总长。
  - 「喂给 RoPE 的最大位置」: 在「沿用绝对位置」模式下等于 T − 1, 一旦超出模型预训练上下文长度 64 即标红报警, 说明流式推理必须配合缓存内位置重编号。"
    module="llm_infer/m16"
    run="python -m llm_infer.m16_attention_sinks.demo"
    :challenge="{
      ask: '在默认参数下将「保留的 sink 数」从 4 拖至 0 (退化为纯滑动窗口), 留存 token 的注意力权重会被放大至多少倍?',
      answer: '在默认参数下 (流长度 T 为 96, 窗口 window 为 16, sink 强度为 4), 「幸存权重被放大」读数从 1.24× 升至 4.70×, 「逐出丢掉的注意力」从 19.2% 升至 78.7%。\n由于 softmax 输出的权重总和恒为 1, 注入的 sink 偏置使起始 4 个 token 占据了绝大部分注意力权重 (对应第一行最高的四根黄色柱):\n- 纯滑动窗口 (保留 sink 数为 0): 起始 4 个 token 被移出缓存, softmax 分母骤减近 80%, 每个留存位置的权重均需乘以 $1/(1-0.787) \\approx 4.7$ 才能重新归并为 1。重归一化后的注意力输出严重偏离训练分布, 在真实预训练模型上会导致生成困惑度急剧恶化。\n- 保留 4 个 sink token: softmax 分母的主体部分得以保留, 被逐出的仅是中间区段原本权重极低的普通 token, 留存权重放大倍数仅为 1.24×。\n若进一步将「sink 强度」拖至 0 (消除 sink 现象): 保留 4 个起始 token 时的放大倍数为 1.86×, 不保留时为 1.95×, 是否保留序列开头几乎没有区别。未经过训练的随机权重 TinyLM 不会在起始位置形成注意力汇聚点, 因此在其上直接测试无法观测到该现象。',
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
        <p><Tex text="重归一化放大倍数满足 $\text{放大倍数} = 1 / (1 - \text{丢掉的注意力})$: softmax 分母因逐出而缩减的比例越大, 每个留存 token 的注意力权重就被等比例放大越多。" /></p>
        <p><Tex text="★ 槽位重编号: RoPE 位置索引直接取当前 cache 槽位编号 $0, \dots, L-1$ ($L$ 为缓存内驻留的 token 总数), 始终限制在预训练上下文窗口之内, 无论输入流持续多长均不会发生位置越界。" /></p>
        <p>
          为了每步都能按新槽位重转, <code class="inline">SinkCache</code> 里存的是<b>未旋转 (pre-RoPE)</b> 的 K。
          普通 KV cache 存的是已经按原位置旋转过的 K, 位置烙死在里面, 没法改号。代价是每步要把 cache 里全部 K 重新旋转一遍, <Tex text="$O(L \cdot D)$" />。
        </p>
        <p>被移出滑动窗口的中间历史信息已从缓存中永久丢弃: 保留 attention sink 仅能维持流式生成的注意力分布稳定与输出流畅, 并不能赋予模型检索长程早期记忆的能力。</p>
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
