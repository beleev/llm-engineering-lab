<!--
  Ring Attention 步进器 (对应 llm_train/m12_sequence_parallel/demo.py 的连续切分, T = 32)。
  只讲一件事: KV 块沿环传 D−1 次, 每张卡都见过完整序列, 但任何时刻只持有 1/D 的 KV。
  因果 mask 下未来块直接跳过, 可是每一轮要等最慢的那张卡。
-->
<template>
  <LabFrame
    title="Ring Attention 步进器 — KV 块沿环传递"
    sub="序列切成 $D$ 段常驻 $D$ 张卡; 每一步 KV 块向右邻居传一格, $D-1$ 步后每张卡都见过完整序列, 但任何时刻只持有 $1/D$ 的 KV。
      每张卡下面的小方块是它对各个 KV 块的进度。点一张卡看它的明细。"
    module="llm_train/m12"
    run="python -m llm_train.m12_sequence_parallel.demo"
    :challenge="{
      ask: 'D = 4, 一步一步走到第 3 步。卡 0 和卡 3 各算了几个 KV 块? 因果 mask 跳过了 6 个块对, 墙钟省了多少?',
      answer: '卡 0 只算 1 块, 卡 3 算满 4 块。每卡总工作量是 [36, 100, 164, 228] 个 q·k 对。\n墙钟按每一步最慢的卡算: 36 + 64 + 64 + 64 = 228。不利用因果性是 256, 只省了约 10%。\n跳过的计算都落在编号小的卡上, 它们算完只能干等。zigzag 切分让每张卡拿一早一晚两块, 每轮工作量相同, 墙钟降到 132 (1.73x)。',
    }"
  >
    <template #controls>
      <div class="row">
        <button v-for="d in [2, 4, 8]" :key="d" type="button" :class="{ active: D === d }" @click="D = d">D = {{ d }} 卡</button>
        <button type="button" :disabled="step === 0" @click="step = Math.max(0, step - 1)">‹ 上一步</button>
        <button type="button" :disabled="step === D - 1" @click="step = Math.min(D - 1, step + 1)">下一步 ›</button>
      </div>
      <LabSlider v-model="step" label="步骤 step" :min="0" :max="D - 1" />
    </template>

    <div class="ring">
      <template v-for="(c, r) in cards" :key="r">
        <button type="button" class="dev" :class="{ sel: r === pick }" :aria-pressed="r === pick" @click="pick = r">
          <span class="dev-head mono">卡 {{ r }}</span>
          <span class="chips">
            <span class="chip q mono">Q_{{ r }}</span>
            <span class="chip kv mono">KV_{{ c.src }}</span>
          </span>
          <span class="strip">
            <i v-for="(s, b) in c.blocks" :key="b" class="sq" :class="s" />
          </span>
          <span class="dev-head mono">本步 {{ c.work }}</span>
        </button>
        <span v-if="r < D - 1" class="arrow">→</span>
        <span v-else class="ring-back mono">⟲ 环回</span>
      </template>
    </div>
    <div class="legend mono">
      <span><i class="sq done" />已计算</span>
      <span><i class="sq skip" />因果跳过 (未来块)</span>
      <span><i class="sq todo" />还没传到</span>
    </div>
    <div class="lab-note" style="margin-top: 10px;">
      <p>{{ stepText }}</p>
      <p>{{ pickText }}</p>
    </div>

    <template #stats>
      <div class="kv"><span>通信轮数</span><b>{{ step }} / {{ D - 1 }}</b></div>
      <div class="kv"><span>块对: 已算 · 跳过 / 全部</span><b>{{ st.done }} · {{ st.skipped }} / {{ D * D }}</b></div>
      <div class="kv"><span>本步各卡 q·k 对数</span><b class="small">[{{ cards.map((c) => c.work).join(', ') }}]</b></div>
      <div class="kv"><span>累计墙钟 (每步取最慢的卡)</span><b>{{ st.wall }}</b></div>
      <div class="cover">
        <span>覆盖矩阵 (行 = Q 块/卡, 列 = KV 块)</span>
        <div class="cover-grid" :style="{ gridTemplateColumns: `repeat(${D}, 16px)` }">
          <template v-for="(c, r) in cards" :key="r">
            <i v-for="(s, b) in c.blocks" :key="b" class="cv" :class="s" />
          </template>
        </div>
      </div>
      <p class="lab-note">
        <Tex :text="`每卡常驻 KV: $1/${D}$ 段 $= T/${D} \\cdot d \\cdot 2$ floats。走完全程不利用因果性要 ${(T * T) / D} 个 q·k 对的墙钟。`" />
      </p>
      <p class="lab-note">
        增量合并用的 online softmax 与 FlashAttention 完全同一个技巧:
        单卡分块是 Flash, 跨卡传块就是 Ring。通信只发生在相邻卡之间, 可与计算重叠。
      </p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import Tex from '@/components/Tex.vue'
import { range, sum } from '@/utils/labmath.js'

const T = 32               // 与 demo 相同的序列长度
const D = ref(4)
const step = ref(0)
const pick = ref(3)

// 切换卡数时 step 和选中的卡不能越界
watch(D, (d) => { if (step.value > d - 1) step.value = d - 1; if (pick.value > d - 1) pick.value = d - 1 })

// 卡 r 在第几步遇到 KV 块 b (块沿环向右传 = 编号向后回退)
const seenAt = (r, b) => (r - b + D.value) % D.value
// ★ 卡 r 在第 s 步的 q·k 对数: 自己那块只算下三角, 过去的块算满, 未来的块跳过
const work = (r, s) => {
  const n = T / D.value, b = (r - s + D.value) % D.value
  return s === 0 ? (n * (n + 1)) / 2 : b < r ? n * n : 0
}

const cards = computed(() => range(D.value).map((r) => ({
  src: (r - step.value + D.value) % D.value,           // 这一步手里的 KV 块编号
  work: work(r, step.value),
  total: sum(range(step.value + 1).map((s) => work(r, s))),
  blocks: range(D.value).map((b) => (seenAt(r, b) > step.value ? 'todo' : b <= r ? 'done' : 'skip')),
})))

const st = computed(() => {
  const all = cards.value.flatMap((c) => c.blocks)
  return {
    done: all.filter((s) => s === 'done').length,
    skipped: all.filter((s) => s === 'skip').length,
    wall: sum(range(step.value + 1).map((s) => Math.max(...range(D.value).map((r) => work(r, s))))),
  }
})

const stepText = computed(() => {
  if (step.value === 0) return '第 0 步: 还没有通信。每张卡用自己那一段的 Q 和 KV 算块内注意力, 只算下三角。'
  const idle = cards.value.filter((c) => c.work === 0).length
  return `第 ${step.value} 步: 每张卡把手里的 KV 块传给右邻居, 最后一张卡传回卡 0。`
    + `${D.value - idle} 张卡拿到的是过去的块, 用 online softmax 并进结果; ${idle} 张卡拿到的是未来块, 跳过不算。`
})
const pickText = computed(() => {
  const c = cards.value[pick.value]
  const ids = (k) => range(D.value).filter((b) => c.blocks[b] === k).join(', ') || '无'
  return `卡 ${pick.value}: 手里是 KV_${c.src}。已计算的块 ${ids('done')}; 跳过的块 ${ids('skip')}; 还没传到的块 ${ids('todo')}。累计 ${c.total} 个 q·k 对。`
})
</script>

<style scoped>
.ring { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; background: var(--code-bg); border-radius: var(--radius-sm); padding: 12px; }
.dev { border: 1px solid var(--border); border-radius: var(--radius-sm); background: var(--bg-elev); padding: 8px; display: flex; flex-direction: column; align-items: flex-start; gap: 6px; min-width: 86px; min-height: 0; cursor: pointer; }
.dev.sel { border-color: var(--accent); }
.dev-head { font-size: 11px; color: var(--text-muted); }
.chips { display: flex; gap: 4px; flex-wrap: wrap; }
.chip { font-size: 10px; padding: 2px 6px; border-radius: 999px; border: 1px solid; }
.chip.q { color: var(--accent); border-color: var(--accent); background: var(--accent-soft); }
.chip.kv { color: var(--eye); border-color: var(--eye); background: color-mix(in srgb, var(--eye) 14%, transparent); }
.strip { display: flex; gap: 3px; }
.sq { display: inline-block; width: 12px; height: 12px; border-radius: 3px; }
.sq.done { background: var(--left); }
.sq.skip { background: var(--border-strong); }
.sq.todo { background: var(--code-bg); border: 1px solid var(--border); }
.arrow { color: var(--text-dim); font-size: 16px; }
.ring-back { font-size: 11px; color: var(--text-dim); }
.legend { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 8px; font-size: 11px; color: var(--text-muted); }
.legend span { display: inline-flex; align-items: center; gap: 5px; }
.kv b.small { font-size: 12px; }
.cover { display: flex; flex-direction: column; gap: 6px; }
.cover-grid { display: grid; gap: 3px; }
.cover-grid i { display: block; width: 16px; height: 16px; border-radius: 3px; }
.cv.done { background: var(--accent); }
.cv.skip { background: var(--border-strong); }
.cv.todo { background: var(--code-bg); border: 1px solid var(--border); }
</style>
