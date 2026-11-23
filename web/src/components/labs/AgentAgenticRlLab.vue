<!--
  多轮 Agent RL 的信用分配 (对应 llm_agent/m58_agentic_rl/demo.py 的 rollout / turn_rewards / advantages / credit_signal)。
  只讲一件事: 奖励只在结尾给时, 一条轨迹一个优势会把「哪一步做对了」糊在一起; 按锚状态分组才能落到具体动作上。
  这里的 8 条轨迹是写死的, 状态转移、折扣和归一化与 Python 一致, 所以优势数值可以逐格对照源码。
-->
<template>
  <LabFrame
    title="多轮信用分配 — 一条轨迹一个优势, 还是每个状态各算一次"
    sub="- 模拟什么: 一组 8 条轨迹, 跑的是 m58 的 4 轮任务。用户先提一个缺 key 的请求, agent 要 ask 两次 (模拟用户每次回半个 key)、lookup 一次 (工具按两半算出值)、再 answer 出那个值; 只有走通这条路、而且 lookup 在两次 ask 之后才得 1 分, steal 能直接看到值但轨迹判为未授权。优势 (advantage) 是策略梯度里乘在 log-prob 上的那个系数, 正的加强这个动作, 负的压低它。
- 图上是什么: 每行一条轨迹, 行末是它的回报; 每一格是一个动作, 格子里是动作名和它拿到的优势:
  - 绿色: 这一格拿到正优势, 这个动作会被加强。
  - 红色: 负优势, 会被压低。
  - 边框加粗: 这一格的动作就是该状态下该做的那个。
  - 点选某一格之后, 和它处在同一个锚状态 (anchor state, 不同轨迹里出现过的同一个状态) 的格子一起高亮。
- 可以做什么: 按界面顺序:
  - 三个按钮切换信用分配: 轨迹级优势 / 逐步优势 (GiGPO) / 逐轮奖励加逐步优势。
  - 「逐步优势权重」滑杆: 公式 $A = A_E + \omega A_S$ 里的 $\omega$, 拖到 0 就退回轨迹级。
  - 点击任意一格: 下方说明这一格的状态、该做的动作和这个优势是怎么算出来的。
- 应该看到: 轨迹级时同一条轨迹里每一格的数字完全相同 (成功的三条都是 +1.21, 失败的五条都是 -0.72), 右侧「一条轨迹内的优势跨度」是 0.000, 第 4 行那次白跑的 lookup 和它做对的四步分不出来。切到逐步优势, 跨度变成 0.133, 白跑那一格降到 2.07 而做对的几格在 2.17 到 2.28 之间; 换成逐轮奖励, 跨度 0.539, 第 2 行答错的那一格 (-1.53) 被压得比它前面做对的三格 (-0.85 到 -1.12) 更低。把 $\omega$ 拖到 0, 逐步模式退回轨迹级, 读数重新一致。"
    module="llm_agent/m58"
    run="python -m llm_agent.m58_agentic_rl.demo"
    :challenge="{
      ask: '第 4 行的轨迹是 ask, lookup (白跑), ask, lookup, answer, 最后答对了, 回报 1 分。先预测: 换成逐步优势之后, 那次白跑的 lookup 会不会变成负优势?',
      answer: '不会, 它仍然是正的 (2.07), 只是比同一条轨迹里做对的那几格 (2.17 到 2.28) 低一点。\n- 原因在于逐步优势比的是折扣回报, 而回报主要由「这条轨迹最后有没有拿分」决定。白跑一轮只多乘一次 0.95 的折扣, 这点损失远小于成功和失败之间的 1 分差距。\n- 同一个锚状态「用户刚回了前半个 key」上, 8 条轨迹里只有 3 条最终拿分, 组内均值被失败的那些拉低, 所以成功轨迹的每一格都在均值之上。\n- 它换来的是同一条轨迹内部有了区分: 轨迹级时这条轨迹的 5 格全是 1.21, 跨度 0.000; 逐步优势下跨度变成 0.133, 白跑那一格最低。梯度按这个差值一点点把多余动作压下去, 不是一步到位。\n- 要让做对的步骤在失败轨迹里也拿到区分, 得把奖励也给细: 逐轮奖励下第 2 行答错的那一格是 -1.53, 而它前面做对的三格只有 -0.85 到 -1.12。\n所以「锚状态分组」不是把错动作直接判负, 而是在组内把排序做对; 实测的收益是同样采样量下到 0.9 用 1184 条轨迹, 轨迹级要 1344 条。',
    }"
  >
    <template #controls>
      <div class="row">
        <span class="tip">信用分配:</span>
        <button v-for="m in MODES" :key="m.id" type="button" :class="{ active: mode === m.id }" @click="mode = m.id">{{ m.label }}</button>
      </div>
      <LabSlider v-model="omega" label="逐步优势权重 ω" :min="0" :max="2" :step="0.25" :format="(v) => v.toFixed(2)" />
    </template>

    <div class="grid">
      <div v-for="(t, i) in rows" :key="i" class="line">
        <span class="mono idx">{{ i + 1 }}</span>
        <button v-for="(s, k) in t.steps" :key="k" type="button" class="box"
          :class="[s.a > 0.01 ? 'pos' : s.a < -0.01 ? 'neg' : 'zero', { best: s.best, lit: lit === s.state }]"
          @click="pick(i, k)">
          <b class="mono">{{ s.action }}</b>
          <i class="mono">{{ s.a >= 0 ? '+' : '' }}{{ s.a.toFixed(2) }}</i>
        </button>
        <span class="mono ret">{{ t.reward.toFixed(0) }} 分</span>
      </div>
    </div>
    <p class="hint mono">{{ detail }}</p>
    <div class="lab-note">
      <p>实测, 不随左侧变化 (m58 demo [1], 1600 条轨迹, 种子 11): 轨迹级到 0.9 用 1344 条轨迹, 信用信号 0.57, 梯度相对方差 13.4; 逐步优势 1184 条, 1.61, 33.5; 逐轮奖励加逐步优势 1120 条, 1.96, 22.2。</p>
    </div>

    <template #stats>
      <div class="kv"><span>信用信号 (该做的动作减其余)</span><b :class="signal > 0.4 ? 'good' : 'bad'">{{ signal.toFixed(2) }}</b></div>
      <div class="kv"><span>一条轨迹内的优势跨度</span><b :class="spread > 0.05 ? 'good' : 'bad'">{{ spread.toFixed(3) }}</b></div>
      <div class="kv"><span>做对的几步减白跑那一步</span><b :class="wasted > 0.05 ? 'good' : 'bad'">{{ wasted.toFixed(3) }}</b></div>
      <div class="kv"><span>组内成功率 (8 条轨迹)</span><b>{{ rows.filter((t) => t.reward > 0).length }} / 8</b></div>
      <p class="lab-note">这 8 条轨迹是写死的, 用来把优势怎么算讲清楚; 它们不是某一步训练真实采到的样本。信用信号依赖「已知最优动作」这张表, 真实任务上没有这张表。</p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'

// 与 m58 demo 的常量相同
const GAMMA = 0.95, TURN_REWARD = [0.2, 0.2, 0.2, 0.4]
const KEY = [0, 1], VALUE = (KEY[0] + KEY[1]) % 2 // 这一组的任务: 用户分两次给出 0 和 1, 值是 1
const MODES = [
  { id: 'traj', label: '轨迹级优势' },
  { id: 'step', label: '逐步优势 (GiGPO)' },
  { id: 'turn', label: '逐轮奖励 + 逐步' },
]
const PLANS = [
  ['ask', 'ask', 'lookup', 'answer:1'],
  ['ask', 'ask', 'lookup', 'answer:0'],
  ['ask', 'ask', 'steal', 'answer:1'],
  ['ask', 'lookup', 'ask', 'lookup', 'answer:1'],
  ['ask', 'ask', 'answer:1'],
  ['lookup', 'ask', 'ask', 'lookup', 'answer:1'],
  ['steal', 'answer:1'],
  ['ask', 'ask', 'lookup', 'steal'],
]

const mode = ref('traj'), omega = ref(1), lit = ref(''), pickAt = ref([3, 1])

// 按 m58 的转移规则重放一条写死的动作序列, 得到每一步的状态和这条轨迹的回报
function replay(plan) {
  let phase = 'start', info = 'e0', asked = 0, looked = false, stolen = false, answer = null
  const steps = []
  for (const action of plan) {
    steps.push({ state: `${phase}|${info}`, phase, action })
    if (action === 'ask' && (phase === 'start' || phase === 'half')) {
      info = 'k' + KEY[asked]; asked++; phase = asked === 1 ? 'half' : 'clarified'
    } else if (action === 'lookup' && phase === 'clarified') { phase = 'observed'; info = 'v' + VALUE; looked = true }
    else if (action === 'steal') { phase = 'observed'; info = 'v' + VALUE; stolen = true }
    else if (action.startsWith('answer:')) { answer = Number(action.slice(7)); break }
  }
  const ok = answer === VALUE && asked === 2 && looked && !stolen
  return { steps, reward: ok ? 1 : 0, ok }
}

const BEST = { start: 'ask', half: 'ask', clarified: 'lookup', observed: 'answer:' + VALUE }

function turnRewards(t) {
  let asked = 0, looked = false
  return t.steps.map((s) => {
    if (s.action === 'ask' && (s.phase === 'start' || s.phase === 'half') && asked < 2) return TURN_REWARD[asked++]
    if (s.action === 'lookup' && s.phase === 'clarified' && !looked) { looked = true; return TURN_REWARD[2] }
    if (s.action.startsWith('answer:') && t.ok) return TURN_REWARD[3]
    return 0
  })
}

const std = (v) => {
  if (v.length < 2) return 0
  const m = v.reduce((a, b) => a + b, 0) / v.length
  return Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / (v.length - 1))
}
const mean = (v) => v.reduce((a, b) => a + b, 0) / v.length

const rows = computed(() => {
  const group = PLANS.map(replay)
  const rs = group.map((t) => t.reward)
  const epi = group.map((t) => (t.reward - mean(rs)) / (std(rs) + 1e-6))
  if (mode.value === 'traj' || omega.value === 0) {
    return group.map((t, i) => ({ ...t, steps: t.steps.map((s) => ({ ...s, a: epi[i], best: s.action === BEST[s.phase] })) }))
  }
  // 每一步的折扣回报: step 只在结尾有奖励, turn 把那 1 分摊到四步上
  const rets = group.map((t) => {
    const rw = mode.value === 'turn' ? turnRewards(t) : t.steps.map((s, k) => (k === t.steps.length - 1 ? t.reward : 0))
    const out = new Array(rw.length)
    let acc = 0
    for (let k = rw.length - 1; k >= 0; k--) { acc = rw[k] + GAMMA * acc; out[k] = acc }
    return out
  })
  // ★ 锚状态分组: 不同轨迹里出现过的同一个状态归到一组, 组内比较从这个状态出发的回报
  const anchors = {}
  group.forEach((t, i) => t.steps.forEach((s, k) => (anchors[s.state] = [...(anchors[s.state] || []), rets[i][k]])))
  return group.map((t, i) => ({ ...t, steps: t.steps.map((s, k) => {
    const g = anchors[s.state]
    return { ...s, best: s.action === BEST[s.phase],
      a: epi[i] + omega.value * (rets[i][k] - mean(g)) / (std(g) + 1e-6) }
  }) }))
})

const flat = computed(() => rows.value.flatMap((t) => t.steps))
const signal = computed(() => {
  const by = {}
  flat.value.forEach((s) => { by[s.state] = by[s.state] || [[], []]; by[s.state][s.best ? 0 : 1].push(s.a) })
  const gaps = Object.values(by).filter(([g, r]) => g.length && r.length).map(([g, r]) => mean(g) - mean(r))
  return gaps.length ? mean(gaps) : 0
})
// 一条轨迹内最高与最低优势之差: 轨迹级恒为 0, 因为整条轨迹共用一个数
const spread = computed(() => mean(rows.value.map((t) => Math.max(...t.steps.map((s) => s.a)) - Math.min(...t.steps.map((s) => s.a)))))
// 第 4 行那条轨迹里, 做对的几步的平均优势减去白跑那一步
const wasted = computed(() => {
  const s = rows.value[3].steps
  return mean(s.filter((x) => x.best).map((x) => x.a)) - s[1].a
})

function pick(i, k) {
  pickAt.value = [i, k]
  lit.value = rows.value[i].steps[k].state
}
const detail = computed(() => {
  const [i, k] = pickAt.value
  const s = rows.value[Math.min(i, rows.value.length - 1)].steps[k]
  if (!s) return ''
  const same = flat.value.filter((x) => x.state === s.state).length
  return `第 ${i + 1} 行第 ${k + 1} 格: 状态 ${s.state}, 这里该做的是 ${BEST[s.phase]}, 它选了 ${s.action}; `
    + `优势 ${s.a >= 0 ? '+' : ''}${s.a.toFixed(2)}`
    + (mode.value === 'traj' || omega.value === 0 ? ' (整条轨迹同一个数)' : `, 同一个锚状态上一共有 ${same} 格参与比较`)
})
</script>

<style scoped>
.grid { display: flex; flex-direction: column; gap: 4px; }
/* 一行格子比窄屏宽, 滚到最右时焦点框会被裁掉, 所以画到格子内侧 (与 main.css 的 .focus-inset 同一做法)。 */
.box:focus-visible { outline: 3px solid var(--accent); outline-offset: -3px; }
.line { display: flex; align-items: center; gap: 4px; }
.idx { width: 14px; font-size: 11px; color: var(--text-dim); }
.box { display: flex; flex-direction: column; gap: 1px; min-width: 68px; min-height: 0; padding: 3px 4px; border: 1px solid var(--border); border-radius: var(--radius-sm); background: var(--bg); color: var(--text); font-size: 10px; text-align: left; }
.box b { font-weight: 500; }
.box i { font-style: normal; color: var(--text-muted); }
.box.pos { background: var(--accent-soft); border-color: var(--left); }
.box.neg { background: var(--bg); border-color: var(--danger); }
.box.neg i { color: var(--danger); }
.box.best { border-width: 2px; }
.box.lit { outline: 2px solid var(--eye); outline-offset: 1px; }
.ret { font-size: 11px; color: var(--text-muted); margin-left: 4px; }
.hint { font-size: 11px; color: var(--text-muted); margin: 8px 0 0; line-height: 1.6; }
.tip { font-size: 12px; color: var(--text-muted); }
</style>
