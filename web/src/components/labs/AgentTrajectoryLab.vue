<!-- 对照实验: m25_trajectory_learning。数值由 Python / JS 一致性测试校验。 -->
<template>
  <LabFrame
    class="agent-experiment"
    title="猜对了，能拿去训练吗？"
    sub="五条轨迹回答同一个配额问题。切换奖励口径，点击每条轨迹检查逐步标签。"
    module="llm_agent/m25_trajectory_learning"
    run="python -m llm_agent.m25_trajectory_learning.demo"
    :challenge="{
      ask: '只奖励答案正确，再扣一点调用成本，会偏好哪条轨迹？',
      answer:
        '会偏好没有查证据却碰巧猜中的短轨迹。把来源、授权和预算也计入验收后，先查再答才是可用样本。数据通过验收不等于训练后的模型已经改善。',
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
      <p>当前最高奖励：{{ winner.name }}</p>
      <p>通过的轨迹用于 SFT；同题好坏轨迹组成偏好对；逐步标签供过程监督使用。</p>
    </div>
    <template #stats>
      <div class="kv">
        <span>结果正确</span><b>{{ rows.filter((r) => r.report.result).length }} / 5</b>
      </div>
      <div class="kv">
        <span>可学习轨迹</span><b>{{ rows.filter((r) => r.report.eligible).length }} / 5</b>
      </div>
      <div class="kv">
        <span>最高奖励</span><b>{{ winner.reward.toFixed(2) }}</b>
      </div>
      <p class="lab-note">
        奖励 = 验收通过 − 0.05 × 动作数。这里构造样本与标签；模型训练复用微调阶段，未在本实验中执行。
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
