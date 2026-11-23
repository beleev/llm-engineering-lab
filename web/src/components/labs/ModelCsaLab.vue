<!--
  压缩注意力实验台 (对应 llm_models/layers/core/compressed_attention.py: compress_entries / topk_mask)。
  只讲一件事: 先把序列按块压短, 再在压缩块上选 top-k。块要凑满才可见, 所以压缩比越大, 近处越要靠滑窗补。
  玩具简化: indexer 的打分用随机数, 其中 "针" 所在的块分数偏高 (Python 里是训练出来的)。
-->
<template>
  <LabFrame
    title="压缩稀疏注意力 — 先把序列压短, 再在块上选 top-k"
    sub="- 模拟什么: DeepSeek-V4 的注意力。每 m 个 token 压成 1 个压缩条目 (学出来的加权和, 不是池化), 一个条目同时当 key 和 value; 再由一个便宜的 indexer 给这些条目打分, 每个 query 只读分数最高的 k 个。另外每个 query 都额外读最近几个未压缩的 token (滑动窗口)。CSA 用 m = 4 加 top-k, HCA 用 m = 128 且不做选择。
- 图上是什么: 上面一排 32 个小格是 token, 带框的那个是当前 query, 浅色的是它的滑动窗口。下面一排长方块是压缩条目, 每块横跨它覆盖的 m 个 token: 深灰是还没凑满、query 看不到的, 浅色是可见但没被选中的, 高亮色是被 indexer 选中的, 带虚线边框的是 '针' 所在的块 (答案藏在那里)。竖线标出 query 的位置。
- 可以做什么: 两个按钮在 '稀疏选 top-k (CSA)' 和 '稠密读全部 (HCA)' 之间切换。两个滑杆分别调压缩块大小 m 和保留的条目数 k。拖动上面一排里带框的格子 (或选中后按左右方向键) 换 query 的位置。'换一组' 重抽 indexer 的打分。
- 应该看到: m 越大, 块越少、KV 字节越省, 但最后那一块要等更久才凑满, query 附近只能靠滑窗读到原始 token。k 调小时读到的条目数跟着降, 命中针所在块的概率也降; 切到稠密模式读全部可见块, 一定命中, 代价是读的条目数等于可见块数。把 query 拖到序列开头, 可见块数会掉到 0, 这时注意力全靠滑窗。"
    module="llm_models/layers/core/compressed_attention.py"
    run="python -m llm_models.run_models.moe.deepseek_v4.infer_deepseek_v4"
    :challenge="{
      ask: '把 m 调到 8、k 调到 2, query 放在最后一个位置。读到的条目数是多少? 再切到稠密模式看一次。为什么压缩比越大, 滑动窗口就越重要?',
      answer: '答案: m = 8 时 32 个 token 压成 4 块, query 在末尾时 4 块都凑满了; k = 2 只读 2 块加滑窗, 稠密模式读 4 块加滑窗。\\n原因:\\n- 压缩条目要等它覆盖的 m 个 token 全部到齐才可见, 到齐位置是 (i+1)m − 1。m 越大, 最近那一块悬空的时间越长: 最坏情况下 query 前面 m − 1 个 token 还没进任何可见条目。\\n- 这段空缺正好由滑动窗口补上: 它读的是未压缩的 token, 一个也不丢。所以 HCA 的压缩比敢开到 128, 靠的是同一条滑窗分支。\\n- KV 字节按每 m 个 token 一条摊: m = 4 时每 token 每层 144 字节, m = 128 时 4.65 字节 (原文配置 c = 512、RoPE 段 64 用 BF16、其余用 FP8、滑窗 128, 1M 上下文)。省下来的代价是块内的细节被加权和抹平了, 所以 V4 把两种层交错着用。\\n- 选得准不准是另一件事: top-k 不可导, indexer 拿不到 LM loss 的梯度, 要靠拟合主注意力分布的对齐 loss 单独训。'
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: sparse }" @click="sparse = true">稀疏选 top-k (CSA)</button>
        <button type="button" :class="{ active: !sparse }" @click="sparse = false">稠密读全部 (HCA)</button>
        <button type="button" @click="seed++">换一组</button>
      </div>
      <LabSlider v-model="mExp" label="压缩块大小 m" :min="0" :max="4" :format="() => String(m)" />
      <LabSlider v-model="topk" label="保留的条目数 k" :min="1" :max="8" />
    </template>

    <svg :viewBox="`0 0 ${W} 150`" class="seq" role="group" aria-label="token 序列与压缩条目, query 位置可拖动">
      <text x="0" y="10" class="lbl">token (带框的是 query, 浅色是滑动窗口)</text>
      <g v-for="t in T" :key="'t' + t">
        <rect :x="(t - 1) * CW" y="16" :width="CW - 1" height="14"
              :class="['tok', { win: inWindow(t - 1), q: t - 1 === qpos }]" />
      </g>
      <rect :x="qpos * CW - 1" y="14" :width="CW + 1" height="18" class="grip draggable"
            tabindex="0" role="slider" :aria-valuenow="qpos" :aria-valuemin="0" :aria-valuemax="T - 1"
            aria-label="query 的位置" @keydown.left="qpos = Math.max(0, qpos - 1)" @keydown.right="qpos = Math.min(T - 1, qpos + 1)" />
      <line :x1="qpos * CW + CW / 2" :x2="qpos * CW + CW / 2" y1="16" y2="110" class="qline" />

      <text x="0" y="52" class="lbl">压缩条目 (深灰 = 还没凑满, 虚线框 = 针所在的块)</text>
      <g v-for="(b, i) in blocks" :key="'b' + i">
        <rect :x="i * m * CW" y="58" :width="m * CW - 2" height="22"
              :class="['blk', b.state, { needle: i === needle }]" />
        <text :x="i * m * CW + (m * CW - 2) / 2" y="73" text-anchor="middle" class="num mono">{{ b.score.toFixed(1) }}</text>
      </g>
      <text x="0" y="100" class="lbl">
        可见 {{ visible.length }} 块, 读到 {{ chosen.length }} 块 + {{ winCount }} 个未压缩 token
      </text>
    </svg>

    <template #stats>
      <div class="kv"><span>读到的 KV 条目数 (稠密因果是 {{ qpos + 1 }})</span><b :class="chosen.length + winCount < qpos + 1 ? 'good' : 'bad'">{{ chosen.length + winCount }}</b></div>
      <div class="kv"><span>命中针所在的块</span><b :class="hit ? 'good' : 'bad'">{{ hit ? '是' : '否' }}</b></div>
      <div class="kv"><span>每 token 每层 KV 字节 (原文配置, 1M 上下文)</span><b :class="bytes < 1152 ? 'good' : 'bad'">{{ bytes.toFixed(2) }}</b></div>
      <div class="lab-note">
        <p>字节按 c = 512、RoPE 段 64 用 BF16、其余 448 维用 FP8、滑窗 128 条算: 一个条目 576 字节, 摊到每 m 个 token, 再加上不随长度增长的滑窗。MLA 的对照是每 token 每层 1152 字节。</p>
        <p>Python 实测, 不随左侧变化: 压缩比 1、不重叠、top-k 取全部、去掉滑窗时, CSA 与稠密因果注意力的最大差 2.4e-07; 逐 token 增量生成与整段前向最大差 4.8e-07。按原文配置的代价模型, 61 层 1:1 交错时每 token 4,469 字节, 是 MLA 的 6.4%。</p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { useDrag } from '@/composables/useDrag.js'
import { mulberry32 } from '@/utils/labmath.js'

const T = 32, CW = 12, W = T * CW, WINDOW = 4
const sparse = ref(true), mExp = ref(2), topk = ref(3), qpos = ref(27), seed = ref(1)
const m = computed(() => 1 << mExp.value)                      // 1 / 2 / 4 / 8 / 16

// 针藏在前半段的某个 token 里, 它所在的块就是 indexer 该选的那一块
const needleTok = computed(() => 5 + (mulberry32(seed.value * 31)() * 10) | 0)
const needle = computed(() => Math.floor(needleTok.value / m.value))

// indexer 打分: 随机, 针所在的块偏高 (Python 里这是训练出来的)
const scores = computed(() => {
  const r = mulberry32(seed.value * 7717)
  return Array.from({ length: Math.ceil(T / m.value) }, (_, i) => r() * 2 + (i === needle.value ? 2.2 : 0))
})
// 条目 i 的到齐位置 (i+1)m − 1: 覆盖的 token 全到齐才可见  ★
const ready = (i) => (i + 1) * m.value - 1
const visible = computed(() => scores.value.map((_, i) => i).filter((i) => ready(i) <= qpos.value))
const chosen = computed(() => {
  if (!sparse.value) return visible.value
  return [...visible.value].sort((a, b) => scores.value[b] - scores.value[a]).slice(0, topk.value)
})
const blocks = computed(() => scores.value.map((score, i) => ({
  score,
  state: !visible.value.includes(i) ? 'pending' : chosen.value.includes(i) ? 'picked' : 'seen',
})))
const inWindow = (t) => t <= qpos.value && qpos.value - t < WINDOW
const winCount = computed(() => Math.min(WINDOW, qpos.value + 1))
const hit = computed(() => chosen.value.includes(needle.value))
// 代价模型: 一个条目 576 字节, 每 m 个 token 一条; 滑窗 128 条, 1M 上下文下摊薄到可忽略
const bytes = computed(() => 576 / m.value + (128 * 2 * 576) / 1e6)

const { start } = useDrag((p) => { qpos.value = Math.max(0, Math.min(T - 1, Math.round(p.x / CW - 0.5))) })
</script>

<style scoped>
.seq { width: 100%; min-width: 400px; display: block; }
.tok { fill: var(--border-strong); }
.tok.win { fill: var(--eye); }
.tok.q { fill: var(--accent); }
.grip { fill: transparent; stroke: var(--text); stroke-width: 1.5; cursor: ew-resize; }
.grip:focus { outline: none; stroke: var(--accent); stroke-width: 2.5; }
.qline { stroke: var(--text-dim); stroke-dasharray: 3 3; }
.blk { stroke: var(--border); }
.blk.pending { fill: var(--border-strong); }
.blk.seen { fill: var(--left); opacity: 0.45; }
.blk.picked { fill: var(--accent); }
.blk.needle { stroke: var(--warn); stroke-width: 2; stroke-dasharray: 4 2; }
.num { font-size: 9px; fill: var(--text); }
.lbl { font-size: 9px; fill: var(--text-dim); }
</style>
