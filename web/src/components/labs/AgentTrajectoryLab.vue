<!-- 对照实验: m25_trajectory_learning。数值由 Python / JS 一致性测试校验。 -->
<template>
  <LabFrame
    class="agent-experiment"
    title="轨迹学习 — 答案碰巧猜对的轨迹, 能拿去当训练数据吗"
    sub="- 模拟什么: 想用 agent 自己跑出来的轨迹 (trajectory, 一次任务里的动作序列加最终回答) 去训练模型, 先得决定哪些轨迹算「好」。这里 5 条轨迹回答同一道题「某个配额是多少」, 标准答案是 10: verified 先用 lookup 查了再答; lucky-guess 没查, 直接猜了个 10; forbidden 用 steal 这种没授权的方式拿到数据再答; over-budget 查了三次才答; wrong 查了却答成 11。每一步都会被检查三件事: 来源 (答案有没有查过的依据), 授权 (动作是不是允许的), 预算 (动作数有没有超)。
- 图上是什么: 五个按钮是五条轨迹, 下面一行是当前奖励口径下它的奖励分; 绿边的是通过了全部检查、可以进训练集的轨迹 (这个边框不随奖励口径变)。点一条, 下面的面板逐步列出它的动作和每一步的检查结果: 通过, 或者一个原因 (unsupported 是没查过依据就作答, authorization 是用了未授权的动作, budget 是超出动作预算, wrong-result 是答错, after-final 是作答之后还有动作)。面板里还显示当前奖励最高的是哪条。
- 可以做什么: 「奖励口径」下拉框: 「只看结果」答案对就给 1 分, 再扣 0.05 乘动作数; 「结果与过程」要结果对而且每一步检查都通过才给 1 分, 同样扣动作成本。「动作预算」是允许的最多动作数, 拖到 4 看 over-budget 那条会不会翻身。
- 应该看到: 「只看结果」下 lucky-guess 奖励最高 (0.95), 因为它答对了又只用了 1 个动作, 没查依据反而占了便宜; forbidden 也拿 0.90, 偷来的数据照样算对。切到「结果与过程」, 只有 verified 还是正分, 其他全部变负, 最高奖励换成 verified。右侧「结果正确」是答案等于 10 的条数 (默认 3 条); 「可学习轨迹」是结果和过程都通过的条数, 默认只有 verified 1 条, 预算拖到 4 时 over-budget 也通过, 变 2 条; 「最高奖励」是当前口径下的最高分。"
    module="llm_agent/m25_trajectory_learning"
    run="python -m llm_agent.m25_trajectory_learning.demo"
    :challenge="{
      ask: '只奖励答案正确, 再扣一点调用成本, 会偏好哪条轨迹?',
      answer:
        '会偏好 lucky-guess: 没查任何依据却碰巧猜中的那条最短轨迹。\n一步步算: 奖励 = 验收分 − 0.05 × 动作数。「只看结果」时验收分只看答案是不是 10: verified 对, 2 个动作, 1 − 0.10 = 0.90; lucky-guess 对, 1 个动作, 1 − 0.05 = 0.95; forbidden 对 (数据是偷来的, 但这个口径不管), 0.90; over-budget 和 wrong 答案不对, 只剩负的成本。所以猜中的短路径奖励最高, 这样的奖励会教出「别查, 直接猜」的捷径。\n切到「结果与过程」: 验收分要求每一步都通过来源、授权、预算检查。lucky-guess 的作答一步被标成 unsupported (没有查过依据), forbidden 的 steal 被标成 authorization, 两条都拿 0 分再扣成本, 只有 verified 还是 0.90。\n通过验收的轨迹才能进 SFT 数据; 同一道题的好坏轨迹组成偏好对; 逐步标签用来做过程监督。但数据通过验收不等于训练后的模型已经改善, 那要另外评测。',
    }"
  >
    <template #controls>
      <label
        >奖励口径
        <select v-model="aware">
          <option :value="false">只看结果</option>
          <option :value="true">结果与过程</option>
        </select></label
      >
      <LabSlider v-model="budget" label="动作预算" :min="1" :max="5" />
    </template>
    <div class="trace" aria-label="候选轨迹">
      <button
        v-for="(row, i) in rows"
        :key="row.name"
        type="button"
        :class="{ active: selected === i, good: row.report.eligible }"
        @click="selected = i"
      >
        {{ row.name }}<br />奖励 {{ row.reward.toFixed(2) }}
      </button>
    </div>
    <div class="detail">
      <p v-for="(action, i) in trajectories[selected].actions" :key="i">
        {{ i + 1 }} · {{ action }} → {{ rows[selected].report.reasons[i] || '通过' }}
      </p>
      <p>当前奖励最高的轨迹: {{ winner.name }}</p>
      <p>通过验收的轨迹进 SFT 数据; 同一道题的好坏轨迹组成偏好对 (chosen / rejected); 每一步的检查结果就是过程监督用的逐步标签。</p>
    </div>
    <template #stats>
      <div class="kv">
        <span>结果正确 (最终答案等于 10)</span><b>{{ rows.filter((r) => r.report.result).length }} / 5</b>
      </div>
      <div class="kv">
        <span>可学习轨迹 (结果和过程都通过)</span><b>{{ rows.filter((r) => r.report.eligible).length }} / 5</b>
      </div>
      <div class="kv">
        <span>最高奖励 (当前口径)</span><b>{{ winner.reward.toFixed(2) }}</b>
      </div>
      <p class="lab-note">
        奖励 = 验收分 (通过 1, 不通过 0) − 0.05 × 动作数。「只看结果」的验收分只看答案, 「结果与过程」还要每一步检查都通过。这里只构造样本和标签; 模型训练复用微调阶段的方法, 没有在本实验里执行。
      </p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import '@/styles/agent-experiments.css'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { trajectories, trajectoryReport } from '@/utils/agent-reliability.js'
const aware = ref(false),
  budget = ref(3),
  selected = ref(1)
const rows = computed(() =>
  trajectories.map((t) => {
    const report = trajectoryReport(t.actions, budget.value)
    return { ...t, report, reward: Number(aware.value ? report.eligible : report.result) - 0.05 * report.calls }
  }),
)
const winner = computed(() => rows.value.reduce((best, row) => (row.reward > best.reward ? row : best)))
</script>
