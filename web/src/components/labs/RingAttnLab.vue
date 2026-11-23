<!--
  Ring Attention 步进器 (对应 llm_train/m12_sequence_parallel/demo.py 的连续切分, T = 32)。
  只讲一件事: KV 块沿环传 D−1 次, 每张卡都见过完整序列, 但任何时刻只持有 1/D 的 KV。
  因果 mask 下未来块直接跳过, 可是每一轮要等最慢的那张卡。
-->
<template>
  <LabFrame
    title="Ring Attention — 序列切给 D 张卡, KV 块沿环传一圈, 每张卡都算出精确的注意力"
    sub="- 模拟什么: 当一条长度为 32 个 token 的长序列超出单卡显存容量时, 序列并行将其沿长度维均分为 $D$ 段, 每张卡常驻各自对应区段的 Q 块与初始 KV 块。为使每个位置的 Q 都能访问全序列的 KV, 各卡将手中的 KV 块沿环形拓扑依次向右邻卡传递, 历经 $D - 1$ 轮点对点通信后每张卡均遍历完整个序列, 而任意时刻单卡仅需驻留 $1/D$ 的 KV 显存; 每收到一块新 KV 即利用 online softmax 增量合并至局部输出 (与 FlashAttention 的分块归约数学等价, 最终结果与单卡全序列注意力精确一致)。在因果 mask 约束下, 当某张卡收到位置编号大于自身 Q 块的「未来」KV 块时可直接整块跳过计算, 但每轮环形同步仍需等待本轮负载最重的那张卡完成后才能进入下一轮。
- 图上是什么: 左侧环形拓扑中每个方框代表一张计算卡, 右侧为全局块覆盖矩阵:
  - 方框内部标注: Q_r 表示该卡常驻的第 r 段 query 块, KV_x 表示该卡在当前通信轮次所持有的 KV 块编号。
  - 方框下方进度条: 对应全部 $D$ 个 KV 块的处理状态, 绿色表示已完成计算, 灰色表示因属未来位置而整块跳过, 空白格表示尚未传递到该卡。
  - 「本步」读数: 该卡在当前轮次实际需计算的 $q \cdot k$ 点积对数量。
  - 卡间连线箭头: 指示 KV 块沿环传递的方向, 末卡环回连接至卡 0。
  - 右侧覆盖矩阵: 汇总全环进度, 行对应 Q 块 (即各张计算卡), 列对应各段 KV 块。
- 可以做什么: 自上而下操作界面控件:
  - 「D = 2 / 4 / 8 卡」: 切换环形拓扑中的设备卡数。
  - 「上一步 / 下一步」或拖动「步骤 step」滑杆: 逐步推进通信轮次; 第 0 步尚未发起跨卡通信, 各卡仅计算自身常驻块内部的局部注意力。
  - 点击任意设备方框: 图表底部文字栏展示该卡在当前轮次的计算与通信明细。
- 应该看到: 每前进一步, 各卡所持 KV 块的编号循环减 1 (KV 块向右传递, 即每张卡接收左邻卡发来的块)。编号较小的头部卡很快便只收到未来 KV 块, 「本步」计算量降为 0 并陷入同步空等; 而持有序列末尾 Q 块的最后一张卡在每一轮均满负荷计算。右侧读数含义如下:
  - 「通信轮数」: 当前所处的环形传递轮次, 全程共需执行 $D - 1$ 轮通信。
  - 「块对」(已算 · 跳过 / 全部): 覆盖矩阵中已完成计算格与因果跳过格的统计数量。
  - 「本步各卡 q·k 对数」: 当前轮次各卡的实际点积计算量; 对角线自身块仅计算下三角 $n(n + 1)/2$ 对 ($n = 32 / D$), 历史块计算完整 $n^2$ 对, 未来块计算量为 0。
  - 「累计墙钟」: 每一步取全环最慢卡的计算量作为当步耗时并逐轮累加, 反映存在木桶效应时的真实端到端耗时。"
    module="llm_train/m12"
    run="python -m llm_train.m12_sequence_parallel.demo"
    :challenge="{
      ask: '在 $D = 4$ 配置下逐步推进至第 3 步结束。请先思考: 卡 0 与卡 3 全程各自实际计算了多少个 KV 块? 因果 mask 合计跳过了 6 个未来块对, 累计墙钟耗时相应节省了多少?',
      answer: '答案:\n- 卡 0 全程仅实际计算 1 个 KV 块, 卡 3 则计算了全部 4 个 KV 块。\n- 累计墙钟耗时仅从无因果跳过的 256 降至 228, 仅节省约 10%。\n原因分析:\n- 负载分布不均: 第 r 张卡常驻第 r 段 Q, 在因果 mask 约束下只需关注编号不大于 r 的 KV 块, 因此卡 0 仅需计算自身的 KV_0 共 1 块, 而位于序列末尾的卡 3 必须遍历 KV_0 至 KV_3 全部 4 块。\n- 单卡计算量核算: 每段包含 8 个 token; 计算自身所在对角块时仅需算下三角 $8 \\times 9 / 2 = 36$ 个 $q \\cdot k$ 对, 计算更早的历史块时需算满 $8 \\times 8 = 64$ 个点积对, 由此四张卡的全程总计算量依次为 [36, 100, 164, 228]。\n- 同步墙钟耗时受限于每轮最慢卡:\n  - 第 0 步各卡均计算自身下三角块, 耗时均为 36。\n  - 第 1 至第 3 步中, 每一轮都至少有一张高编号卡在计算完整的历史块, 单步耗时恒为 64。\n  - 四步累计墙钟耗时为 $36 + 64 + 64 + 64 = 228$; 若不跳过未来块则四步均为 64, 合计 256。\n- 根源在于被因果 mask 省去的 6 个未来块对 (总算力原本减少了近一半) 全部集中在低编号卡上, 低编号卡提前算完后只能同步等待仍在满载计算的高编号卡, 省下的浮点运算未能转化为端到端时间收益。\n工程上通常采用交错均衡切分 (zigzag Ring Attention) 解决该木桶效应: 将整条序列细分为 $2D = 8$ 段, 让每张卡同时持有一头一尾两段互补位置 (例如卡 0 常驻第 0 段与第 7 段), 使各卡在每一轮通信中的有效计算量完全对齐, 累计墙钟耗时可降至 132, 相比连续切分提速 1.73×。',
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
      <p class="lab-note">跨块增量归约所用的 online softmax 与 FlashAttention 基于同一数学恒等式: 每接收一块新 KV, 便根据更新后的局部最大 logit 对已累积的统计量 (当前最大值、指数和分母、加权输出分子) 做重缩放修正后再累加, 最终输出与一次性拼接全序列 KV 计算在数学上严格等价 (示例脚本实测浮点误差仅为 4.4e-16)。在单卡 SRAM 内分块流式归约即为 FlashAttention, 将 KV 块在多卡环网间流转归约即为 Ring Attention; 由于每轮仅需与紧邻卡做点对点收发, 跨卡通信可与当步局部注意力计算异步重叠隐藏。</p>
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
