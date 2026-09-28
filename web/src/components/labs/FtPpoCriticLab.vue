<!--
  PPO vs GRPO: 同样每步 256 条回复, 预算怎么花。
  只讲一件事: GRPO 把 256 条分成 32 题 × 8 条, 组内全对/全错时 A ≡ 0, 这组白花; PPO 一题一采, 256 道题都有信号。
  格子里的 0/1 奖励是前端按随机难度抽的示意 (可以点格子改); 下方表格来自
  python -m llm_finetune.run_finetune.ppo.train_ppo 的结果表 (与 run_finetune/ppo/readme.md 一致), 不随格子变化。
-->
<template>
  <LabFrame
    title="同样 256 条回复: 组均值 vs critic"
    sub="每个格子是一条回复, 绿 = 这条答对 (R=1)。
      - GRPO: 每行是同一道题的 8 条, baseline = 行均值。
      - PPO: 256 格是 256 道不同的题, baseline = critic 的 $V(s_0)$。
      点格子翻转对错, 看哪些回复还有梯度 (淡掉的 = $A \equiv 0$)。键盘: 方向键移动, 回车翻转。"
    module="llm_finetune/methods/ppo.py"
    run="python -m llm_finetune.run_finetune.ppo.train_ppo"
    :challenge="{
      ask: '先看格子: GRPO 下有几组是 A≡0? 切到 PPO 后还剩几条没有梯度? 再看实测表: critic 冻结后 pass@1 和正常 PPO 一样 (0.389 vs 0.381), 那 PPO 比 GRPO 高的 0.09 从哪来?',
      answer: '格子: 默认这一批 GRPO 有 9 组 A≡0, 72 条回复没有梯度; 切到 PPO 后 256 条都有梯度。\n冻结的 critic 给出 $V \\approx 0$, 优势退化成「回报再做批内白化」, 相当于 REINFORCE 减批均值。它照样让 256 道题里答对的上、答错的下。\n高出的 0.09 来自采样方式:\n- GRPO: 32 道题里约 1/3 组全对或全错, $A\\equiv 0$, 白占预算。\n- PPO: 256 道题各采 1 条, 批均值当 baseline 也几乎不浪费。\ncritic 本身在 60 步里几乎没学会: $V(s_0)$ 只解释 2.5% 的回报方差, 免费的组均值解释 36.5%。在这个规模上, 它换来的只有 ×2 的要训参数和 ×2 的耗时。',
    }"
  >
    <template #controls>
      <div class="row">
        <span class="lbl">baseline 来源:</span>
        <button v-for="m in METHODS" :key="m.id" type="button" :class="{ active: mode === m.id }" @click="mode = m.id">{{ m.name }}</button>
        <button type="button" @click="reseed">换一批</button>
      </div>
    </template>

    <div class="grid" :class="{ grouped: mode === 'grpo' }" role="group" aria-label="256 条回复的对错, 方向键移动, 回车翻转">
      <!-- ★ 整张表只占一个 Tab 位: 当前行可聚焦, 方向键在行内和行间移动 -->
      <div
        v-for="(row, g) in cells" :key="g" :ref="(el) => { rowEls[g] = el }"
        class="grow" :class="{ dead: mode === 'grpo' && row.dead }"
        :tabindex="g === cur.g ? 0 : -1" role="group"
        :aria-label="`第 ${g + 1} 行: 8 条里对 ${row.n} 条; 光标在第 ${cur.i + 1} 条, ${row.r[cur.i] ? '对' : '错'}`"
        @keydown.left.prevent="move(0, -1)" @keydown.right.prevent="move(0, 1)"
        @keydown.up.prevent="move(-1, 0)" @keydown.down.prevent="move(1, 0)"
        @keydown.enter.prevent="flip(cur.g, cur.i)" @keydown.space.prevent="flip(cur.g, cur.i)"
      >
        <button
          v-for="(c, i) in row.r" :key="i" type="button" tabindex="-1" class="cell rc"
          :class="{ ok: c, cur: g === cur.g && i === cur.i }"
          :aria-label="`第 ${g * G + i + 1} 条: ${c ? '对' : '错'}, 点击翻转`" @click="tap(g, i)"
        />
        <span v-if="mode === 'grpo'" class="gl mono">{{ row.dead ? 'A≡0' : 'μ=' + row.mu.toFixed(2) }}</span>
      </div>
    </div>

    <p class="tcap">train_ppo.py 留出集 (实测, 不随上面的格子变化; SFT 起点 pass@1 0.186)</p>
    <table class="res mono">
      <thead><tr><th /><th>pass@1</th><th>训练奖励 前 10 → 后 10 步</th><th>baseline 解释的方差</th><th>常驻 / 要训参数</th><th>60 步耗时</th></tr></thead>
      <tbody>
        <tr v-for="m in METHODS" :key="m.id" :class="{ sel: mode === m.id }">
          <td>{{ m.name }}</td><td>{{ m.pass1.toFixed(3) }}</td><td>{{ m.first.toFixed(3) }} → {{ m.last.toFixed(3) }}</td>
          <td>{{ m.ev.toFixed(3) }}</td><td>{{ m.params.toLocaleString() }} / {{ m.train.toLocaleString() }}</td><td>{{ m.sec }}</td>
        </tr>
      </tbody>
    </table>

    <template #stats>
      <div class="kv"><span>有梯度的回复 (本批示意)</span><b :class="st.live < 256 * 0.8 ? 'bad' : 'good'">{{ st.live }} / 256</b></div>
      <div class="kv"><span>整组全对或全错的题</span><b>{{ st.dead }} / {{ NQ }}</b></div>
      <div class="kv"><span>本批平均奖励</span><b>{{ st.mean.toFixed(3) }}</b></div>
      <p class="lab-note">{{ cur_m.note }}</p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, nextTick, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import { clamp, mulberry32, range, sum } from '@/utils/labmath.js'

const G = 8, NQ = 32
// 来自 train_ppo.py 结果表 (每步 256 条, 60 步, 留出集); 耗时两次实测有波动, 取 readme 的数
const METHODS = [
  { id: 'grpo', name: 'GRPO (32 题 × 8)', pass1: 0.287, ev: 0.365, first: 0.205, last: 0.295, params: 99648, train: 99648, sec: '6.1 s',
    note: 'baseline = 同题 8 条的均值, 不用训。组内全对或全错时 A 全为 0, 这 8 条白占预算。' },
  { id: 'ppo', name: 'PPO (256 题 × 1)', pass1: 0.381, ev: 0.025, first: 0.2, last: 0.356, params: 299008, train: 199360, sec: '11.8 s',
    note: 'baseline = critic 的 V(s_0)。常驻 policy + ref + critic 三份, 要训两份。V(s_0) 只解释 2.5% 的回报方差, 几乎是个常数。' },
  { id: 'frozen', name: 'PPO, critic 冻结 (lr=0)', pass1: 0.389, ev: 0.004, first: 0.204, last: 0.357, params: 299008, train: 199360, sec: '12.0 s',
    note: 'critic 一步没学, pass@1 和正常 PPO 在噪声内。高出 GRPO 的那截来自一题一采, 不是 critic。(lr=0 时梯度照算, 参数照记。)' },
]
const mode = ref('grpo'), seed = ref(1)
const cur_m = computed(() => METHODS.find((x) => x.id === mode.value))

// 示意: 每道题的通过率 p 偏低且不均 (训练奖励约 0.2–0.36), 抽 0/1 奖励
const draw = (s) => {
  const rand = mulberry32(s * 4099)
  return range(NQ).map(() => { const p = 0.7 * rand() ** 2; return range(G).map(() => +(rand() < p)) })
}
const R = ref(draw(1))
const reseed = () => { seed.value++; R.value = draw(seed.value) }
const flip = (g, i) => { R.value = R.value.map((row, k) => (k === g ? row.map((c, j) => (j === i ? 1 - c : c)) : row)) }

// 键盘光标: 方向键改 (行, 列), 焦点跟着行走
const cur = ref({ g: 0, i: 0 })
const rowEls = []
const tap = (g, i) => { cur.value = { g, i }; flip(g, i) }
const move = (dg, di) => {
  cur.value = { g: clamp(cur.value.g + dg, 0, NQ - 1), i: clamp(cur.value.i + di, 0, G - 1) }
  nextTick(() => rowEls[cur.value.g]?.focus())
}

const cells = computed(() => R.value.map((r) => {
  const n = sum(r)
  return { r, n, mu: n / G, dead: r.every((c) => c === r[0]) }   // ★ 组内奖励全相同 → r − μ ≡ 0
}))
const st = computed(() => {
  const all = R.value.flat(), dead = cells.value.filter((c) => c.dead).length
  // PPO: 批内白化后, 只要整批不全相同, 每条都有非零 A
  const live = mode.value === 'grpo' ? (NQ - dead) * G : all.every((c) => c === all[0]) ? 0 : all.length
  return { live, dead, mean: sum(all) / all.length }
})
</script>

<style scoped>
.lbl { font-size: 12px; color: var(--text-muted); }
.grid { display: flex; flex-direction: column; gap: 2px; width: max-content; }
.grid.grouped { gap: 4px; }
.grow { display: flex; gap: 2px; align-items: center; }
.grow:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.grow:focus-visible .rc.cur { outline: 2px solid var(--text); outline-offset: 0; }
.grow.dead .rc { opacity: 0.3; }
.rc { min-width: 0; min-height: 0; width: 18px; height: 14px; padding: 0; cursor: pointer; }
.gl { font-size: 10px; color: var(--text-dim); margin-left: 6px; width: 44px; }
.grow.dead .gl { color: var(--danger); }
.tcap { margin: 14px 0 4px; font-size: 12px; color: var(--text); }
.res { border-collapse: collapse; font-size: 11px; min-width: 520px; }
.res th { font-size: 10.5px; color: var(--text-dim); font-weight: 400; text-align: right; padding: 3px 6px; }
.res td { padding: 3px 6px; text-align: right; color: var(--text-muted); border-top: 1px solid var(--border); }
.res td:first-child, .res th:first-child { text-align: left; }
.res tr.sel td { background: var(--bg-elev); color: var(--text); }
</style>
