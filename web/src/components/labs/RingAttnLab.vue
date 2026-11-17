<!--
  Ring Attention 步进器 (对应 llm_train/m12_sequence_parallel/demo.py 的连续切分, T = 32)。
  只讲一件事: KV 块沿环传 D−1 次, 每张卡都见过完整序列, 但任何时刻只持有 1/D 的 KV。
  因果 mask 下未来块直接跳过, 可是每一轮要等最慢的那张卡。
-->
<template>
  <LabFrame
    title="Ring Attention — 序列切给 D 张卡, KV 块沿环传一圈, 每张卡都算出精确的注意力"
    sub="- 模拟什么: 一条 32 个 token 的序列太长, 一张卡放不下, 于是切成 $D$ 段, 每张卡常驻自己那一段的 Q 和 KV。注意力要让每个 Q 看到全部 KV, 所以 KV 块像击鼓传花一样沿环向右邻居传, 传 $D-1$ 次后每张卡都见过完整序列, 但任何时刻只持有 $1/D$ 的 KV; 每收到一块就用 online softmax 并进结果 (和 FlashAttention 的分块技巧一样, 结果精确等于完整注意力)。因果 mask 下, 一张卡拿到「未来」的 KV 块可以整块跳过不算, 但每一轮大家都要等最慢的那张卡。
      - 图上是什么: 每个方框是一张卡。
        - 方框里 Q_r 是它常驻的 query 块, KV_x 是它这一步手里的 KV 块编号。
        - 方框下面一排小方块是它对 $D$ 个 KV 块的进度: 绿色已经算过, 灰色是未来块、跳过, 空的还没传到。
        - 「本步」是这张卡这一步要算的 q·k 对数。
        - 方框之间的箭头是传递方向, 最后一张卡环回卡 0。
        - 右侧的覆盖矩阵: 同一信息的总表, 行是 Q 块 (也就是卡), 列是 KV 块。
      - 可以做什么: 控件从上到下:
        - 「D = 2 / 4 / 8 卡」改环上有几张卡。
        - 「上一步 / 下一步」或拖「步骤 step」走通信轮次, 第 0 步还没通信, 每张卡只算自己块内的注意力。
        - 点任意一张卡, 图下面一行字给它的明细。
      - 应该看到: 每走一步, 每张卡手里的 KV 编号减 1 (块向右传, 卡收到的是左邻居的块)。编号小的卡很快就只剩未来块, 「本步」变成 0 干等; 编号最大的卡每一步都满载。右侧读数:
        - 「通信轮数」: 走到第几轮, 共 $D-1$ 轮。
        - 「块对」(已算 · 跳过 / 全部): 覆盖矩阵里紫色、灰色格子的数量。
        - 「本步各卡 q·k 对数」: 每张卡这一步的工作量。自己的块只算下三角 n(n+1)/2 (n = 32/D), 过去的块算满 n², 未来块为 0。
        - 「累计墙钟」: 每一步取最慢的卡, 再把各步加起来, 这才是真正的耗时。"
    module="llm_train/m12"
    run="python -m llm_train.m12_sequence_parallel.demo"
    :challenge="{
      ask: 'D = 4, 一步一步走到第 3 步。先猜: 卡 0 和卡 3 各算了几个 KV 块? 因果 mask 一共跳过了 6 个块对, 墙钟省了多少?',
      answer: '答案:\n- 卡 0 只算 1 块, 卡 3 算满 4 块。\n- 墙钟只从 256 降到 228, 省约 10%。\n为什么:\n- 卡 r 常驻第 r 段 Q, 因果 mask 下它只需要看编号不大于 r 的 KV 块: 卡 0 只看 KV_0, 卡 3 要看 KV_0 到 KV_3。所以卡 0 算 1 块, 卡 3 算 4 块。\n- 每块 8 个 token。自己那块只算下三角 8 × 9 / 2 = 36 个 q·k 对, 过去的块算满 8 × 8 = 64 个。每卡总工作量是 [36, 100, 164, 228]。\n- 墙钟按每一步最慢的卡算:\n  - 第 0 步大家都是 36。\n  - 第 1 到 3 步总有卡在算满块, 每步 64。\n  - 合计 36 + 64 + 64 + 64 = 228。不利用因果性每步都是 64, 合计 256。\n- 跳过的 6 个块对 (计算量省了近一半) 都落在编号小的卡上, 它们算完只能干等编号大的卡。省下的计算没有变成省下的时间。\nzigzag 切分解决这个问题: 序列切成 2D = 8 段, 每张卡拿一早一晚两段 (卡 0 拿第 0 和第 7 段), 每轮工作量相同, 墙钟降到 132, 比连续切分快 1.73×。',
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
      <span><i class="sq done" />已计算 (过去的块或自己的块)</span>
      <span><i class="sq skip" />因果跳过 (未来块, 不用算)</span>
      <span><i class="sq todo" />还没传到这张卡</span>
    </div>
    <div class="lab-note" style="margin-top: 10px;">
      <p>{{ stepText }}</p>
      <p>{{ pickText }}</p>
    </div>

    <template #stats>
      <div class="kv"><span>通信轮数 (共 {{ D - 1 }} 轮传完)</span><b>{{ step }} / {{ D - 1 }}</b></div>
      <div class="kv"><span>块对: 已算 · 跳过 / 全部 ({{ D }}×{{ D }})</span><b>{{ st.done }} · {{ st.skipped }} / {{ D * D }}</b></div>
      <div class="kv"><span>本步各卡的工作量 (q·k 对数, 卡 0 在前)</span><b class="small">[{{ cards.map((c) => c.work).join(', ') }}]</b></div>
      <div class="kv"><span>累计墙钟 (每步取最慢的卡, 再逐步累加)</span><b>{{ st.wall }}</b></div>
      <div class="cover">
        <span>覆盖矩阵 (行 = Q 块, 也就是卡; 列 = KV 块)</span>
        <div class="cover-grid" :style="{ gridTemplateColumns: `repeat(${D}, 16px)` }">
          <template v-for="(c, r) in cards" :key="r">
            <i v-for="(s, b) in c.blocks" :key="b" class="cv" :class="s" />
          </template>
        </div>
      </div>
      <p class="lab-note">
        <Tex :text="`每张卡任何时刻只持有 $1/${D}$ 的 KV: $T/${D} \\cdot d \\cdot 2$ 个数 (T = 32 个 token, d 是每头维度, K 和 V 各一份)。走完全程、不利用因果性时, 墙钟是 ${(T * T) / D} 个 q·k 对 (每步 (T/D)² 个, 共 D 步)。`" />
      </p>
      <p class="lab-note">
        增量合并用的 online softmax 和 FlashAttention 是同一个技巧: 每收到一块 KV, 就用新的最大值把之前的 (最大值, 分母, 加权和) 重新缩放一遍再累加, 结果和一次算完全部 KV 完全相等 (demo 实测误差 4.4e-16)。
        单卡内分块是 Flash, 跨卡传块就是 Ring。通信只发生在相邻卡之间, 可以和计算重叠。
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
