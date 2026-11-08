<!-- 对照实验: m13_evals。数值由 Python / JS 一致性测试校验。 -->
<template>
  <LabFrame
    class="agent-experiment"
    title="哪一步先错了？"
    sub="同一句“已完成”，可能来自四种不同轨迹。切换判分口径，再点动作查看首错位置。"
    module="llm_agent/m13_evals"
    run="python -m llm_agent.m13_evals.demo"
    :challenge="{
      ask: '提前结束和越界写入都回答“已完成”。只看这句话能分出谁做对了吗？',
      answer:
        '不能。环境验收检查 alpha、beta 是否都在，轨迹验收还检查是否写过范围外的笔记。首错前的快照能定位该改哪一步。',
    }"
  >
    <template #controls>
      <label
        >轨迹
        <select v-model="chosen">
          <option v-for="(c, i) in cases" :key="c.name" :value="i">{{ labels[c.name] }}</option>
        </select></label
      >
      <label
        >判分口径
        <select v-model="grader">
          <option value="text">最终回答</option>
          <option value="state">环境终态</option>
          <option value="trace">全过程</option>
        </select></label
      >
    </template>
    <div class="trace" aria-label="动作时间轴">
      <button
        v-for="(action, i) in row.actions"
        :key="i"
        type="button"
        :class="{ active: step === i, bad: grader === 'trace' && row.failure?.step === i + 1 }"
        @click="step = i"
      >
        {{ i + 1 }} · {{ action }}
      </button>
    </div>
    <div class="detail" aria-live="polite">
      <p>最终回答：{{ row.final || '模型不可用，没有回答' }}</p>
      <p v-if="grader !== 'text'">最终笔记：{{ Object.keys(row.state).join('、') || '空' }}</p>
      <p v-if="grader === 'trace'">首错：{{ row.failure ? `${row.failure.step} · ${row.failure.category}` : '无' }}</p>
      <p v-if="grader === 'trace' && row.failure && step + 1 === row.failure.step">
        首错前环境：{{ row.before_error }}
      </p>
      <p v-else>当前选中第 {{ Math.min(step + 1, row.actions.length) }} 步。红框表示首错，不表示后面没有错误。</p>
    </div>
    <template #stats>
      <div class="kv">
        <span>当前判分</span><b :class="passed ? 'good' : 'bad'">{{ passed ? '通过' : '失败' }}</b>
      </div>
      <div class="kv">
        <span>工具调用</span><b>{{ row.tool_calls }}</b>
      </div>
      <div class="kv">
        <span>首错步骤</span><b>{{ row.failure?.step || '—' }}</b>
      </div>
      <p class="lab-note">记录由 Python Agent 实际执行后导出。题目固定，不是线上模型成功率。</p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import '@/styles/agent-experiments.css'
import LabFrame from '@/components/lab/LabFrame.vue'
import cases from '@/data/agent-diagnostics.json'
const chosen = ref(1),
  grader = ref('text'),
  step = ref(0)
const labels = {
  correct: '真正完成',
  premature: '提前结束',
  'extra-write': '多写了别的对象',
  'tool-failure': '工具执行失败',
  outage: '模型不可用',
}
const row = computed(() => cases[chosen.value])
const passed = computed(() =>
  grader.value === 'text'
    ? row.value.final === '已完成'
    : grader.value === 'state'
      ? Object.keys(row.value.state).sort().join(',') === 'alpha,beta'
      : row.value.passed,
)
</script>
