<!--
  PPO vs GRPO: 同样每步 256 条回复, 预算怎么花。
  只讲一件事: GRPO 把 256 条分成 32 题 × 8 条, 组内全对/全错时 A ≡ 0, 这组白花; PPO 一题一采, 256 道题都有信号。
  格子里的 0/1 奖励是前端按随机难度抽的示意 (可以点格子改); 右侧读数全部来自
  python -m llm_finetune.run_finetune.ppo.train_ppo 的结果表 (与 run_finetune/ppo/readme.md 一致)。
-->
<template>
  <LabFrame
    title="同样 256 条回复: 组均值 vs critic"
    sub="每个格子是一条回复, 绿 = 这条答对 (R=1)。
      - GRPO: 每行是同一道题的 8 条, baseline = 行均值。
      - PPO: 256 格是 256 道不同的题, baseline = critic 的 $V(s_0)$。
      点格子翻转对错, 看哪些回复还有梯度 (淡掉的 = $A \equiv 0$)。"
    module="llm_finetune/methods/ppo.py"
    run="python -m llm_finetune.run_finetune.ppo.train_ppo"
    :challenge="{
      ask: '切到「PPO, critic 冻结」。critic 不学了, 为什么 pass@1 反而和正常 PPO 一样 (0.389 vs 0.381)? 那 PPO 比 GRPO 高的 0.09 从哪来?',
      answer: '冻结的 critic 给出 $V \\approx 0$, 优势退化成「回报再做批内白化」, 相当于 REINFORCE 减批均值。它照样让 256 道题里答对的上、答错的下。\n高出的 0.09 来自采样方式:\n- GRPO: 32 道题里约 1/3 组全对或全错, $A\\equiv 0$, 白占预算。\n- PPO: 256 道题各采 1 条, 批均值当 baseline 也几乎不浪费。\ncritic 本身在 60 步里几乎没学会: $V(s_0)$ 只解释 2.5% 的回报方差, 免费的组均值解释 36.5%。在这个规模上, 它换来的只有 ×2 的要训参数和 ×2 的耗时。',
    }"
  >
    <template #controls>
      <div class="row">
        <button v-for="m in METHODS" :key="m.id" type="button" :class="{ active: mode === m.id }" @click="mode = m.id">{{ m.name }}</button>
        <button type="button" @click="reseed">换一批</button>
      </div>
    </template>

    <div class="grid" :class="{ grouped: mode === 'grpo' }">
      <div v-for="(row, g) in cells" :key="g" class="grow" :class="{ dead: mode === 'grpo' && row.dead }">
        <button
          v-for="(c, i) in row.r" :key="i" type="button" class="cell rc" :class="c ? 'ok' : ''"
          :aria-label="`第 ${g * G + i + 1} 条: ${c ? '对' : '错'}, 点击翻转`" @click="flip(g, i)"
        />
        <span v-if="mode === 'grpo'" class="gl mono">{{ row.dead ? 'A≡0' : 'μ=' + row.mu.toFixed(2) }}</span>
      </div>
    </div>

    <template #stats>
      <div class="kv"><span>有梯度的回复 (本批示意)</span><b :class="live < 256 * 0.8 ? 'bad' : 'good'">{{ live }} / 256</b></div>
      <div class="kv"><span>留出集 pass@1 (SFT 起点 0.186)</span><b :class="m.pass1 > 0.35 ? 'good' : ''">{{ m.pass1.toFixed(3) }}</b></div>
      <div class="kv"><span>baseline 解释的方差</span><b :class="m.ev < 0.1 ? 'bad' : 'good'">{{ m.ev.toFixed(3) }}</b></div>
      <div class="kv"><span>训练奖励 前 10 步 → 后 10 步</span><b>{{ m.first.toFixed(3) }} → {{ m.last.toFixed(3) }}</b></div>
      <div class="kv"><span>常驻 / 要训的参数</span><b :class="m.train > 1e5 ? 'bad' : ''">{{ m.params.toLocaleString() }} / {{ m.train.toLocaleString() }}</b></div>
      <div class="kv"><span>60 步耗时</span><b>{{ m.sec }}</b></div>
      <p class="lab-note">{{ m.note }}</p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import { mulberry32, range, sum } from '@/utils/labmath.js'

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
const m = computed(() => METHODS.find((x) => x.id === mode.value))

// 示意: 每道题的通过率 p 偏低且不均 (训练奖励约 0.2–0.36), 抽 0/1 奖励
const draw = (s) => {
  const rand = mulberry32(s * 4099)
  return range(NQ).map(() => { const p = 0.7 * rand() ** 2; return range(G).map(() => +(rand() < p)) })
}
const R = ref(draw(1))
const reseed = () => { seed.value++; R.value = draw(seed.value) }
const flip = (g, i) => { R.value = R.value.map((row, k) => (k === g ? row.map((c, j) => (j === i ? 1 - c : c)) : row)) }

const cells = computed(() => R.value.map((r) => {
  const mu = sum(r) / G
  return { r, mu, dead: r.every((c) => c === r[0]) }   // ★ 组内奖励全相同 → r − μ ≡ 0
}))
const live = computed(() => {
  if (mode.value === 'grpo') return cells.value.filter((c) => !c.dead).length * G
  const all = R.value.flat()                          // PPO: 批内白化后, 只要整批不全相同, 每条都有非零 A
  return all.every((c) => c === all[0]) ? 0 : all.length
})
</script>

<style scoped>
.grid { display: flex; flex-direction: column; gap: 2px; width: max-content; }
.grid.grouped { gap: 4px; }
.grow { display: flex; gap: 2px; align-items: center; }
.grow.dead .rc { opacity: 0.3; }
.rc { min-width: 0; min-height: 0; width: 16px; height: 12px; padding: 0; cursor: pointer; }
.gl { font-size: 10px; color: var(--text-dim); margin-left: 6px; width: 44px; }
.grow.dead .gl { color: var(--danger); }
</style>
