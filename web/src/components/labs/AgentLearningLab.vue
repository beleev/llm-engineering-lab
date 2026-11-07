<!-- 对照实验: full_learning_loop。数值由 Python / JS 一致性测试校验。 -->
<template>
  <LabFrame
    class="agent-experiment"
    title="改进以后，旧任务还会做吗？"
    sub="按时间走完学习、迁移、规则变化和保持测试。点击任务，查看反馈是在回答之前还是之后出现。"
    module="llm_agent/full_learning_loop"
    run="python -m llm_agent.full_learning_loop.demo"
    :challenge="{
      ask: '项目 A 从 TypeScript 改成 Rust 的第一题为什么应该失败一次？',
      answer:
        '新规则只在这一题回答后作为反馈出现。若第一题就用了新答案，说明泄漏了未来信息。下一题恢复正确，再检查项目 B 仍用 Python，才能分别说明适应变化和保留旧能力。',
    }"
  >
    <template #controls>
      <label
        >更新策略
        <select v-model="mode">
          <option value="static">静态规则</option>
          <option value="append">直接追加记忆</option>
          <option value="versioned">按范围更新版本</option>
        </select></label
      >
    </template>
    <div class="trace" aria-label="跨任务时间轴">
      <button
        v-for="(row, i) in result.rows"
        :key="row.id"
        type="button"
        :class="{ active: selected === i, good: row.passed, bad: !row.passed }"
        @click="selected = i"
      >
        {{ row.id }} · {{ names[row.phase] }}<br />{{ row.action }}
      </button>
    </div>
    <div class="detail">
      <p>{{ current.id }}：先回答 {{ current.action }}，再按 {{ current.expected }} 验收。</p>
      <p>记忆激活：{{ current.activated ? '是' : '否' }}；验收后更新：{{ current.updated ? '是' : '否' }}。</p>
      <p>本题事件：{{ result.order.filter((e) => e.startsWith(current.id + ':')).join(' → ') }}</p>
      <p>迁移与保持阶段只读。规则更新只在学习期和明确变化事件之后开放。</p>
    </div>
    <template #stats>
      <div v-for="(score, phase) in result.scores" :key="phase" class="kv">
        <span>{{ names[phase] }}通过率</span
        ><b :class="score === 1 ? 'good' : score === 0 ? 'bad' : ''">{{ (score * 100).toFixed(0) }}%</b>
      </div>
      <p class="lab-note">
        9 个固定任务的外部记忆实验，未训练权重。负迁移任务
        {{ result.negative_transfer }} 个；这个数包含新规则尚未反馈时的失败。
      </p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import '@/styles/agent-experiments.css'
import LabFrame from '@/components/lab/LabFrame.vue'
import { learning } from '@/utils/agent-reliability.js'
const mode = ref('append'),
  selected = ref(5)
const names = { learning: '学习', transfer: '迁移', change: '规则变化', retention: '旧能力保持' }
const result = computed(() => learning(mode.value))
const current = computed(() => result.value.rows[selected.value])
</script>
