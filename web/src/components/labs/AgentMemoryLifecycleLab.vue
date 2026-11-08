<!-- 对照实验: m20_memory_lifecycle。数值由 Python / JS 一致性测试校验。 -->
<template>
  <LabFrame
    class="agent-experiment"
    title="记住了，为什么还用错？"
    sub="同一批证据：Alice 通常使用 Python，只有项目 A 使用 TypeScript。点击查询对象，观察三种更新策略。"
    module="llm_agent/m20_memory_lifecycle"
    run="python -m llm_agent.m20_memory_lifecycle.demo"
    :challenge="{
      ask: '把项目 A 的新偏好覆盖到全局，项目 B 会怎样？',
      answer:
        'B 会错误地变成 TypeScript。按对象和范围保存版本后，A 使用项目例外，B 回退到全局。Bob 没有证据，必须保留未知。',
    }"
  >
    <template #controls>
      <label
        >更新策略
        <select v-model="mode">
          <option value="append-first">追加后取首条</option>
          <option value="overwrite">最新覆盖全局</option>
          <option value="versioned">按范围审核</option>
        </select></label
      >
    </template>
    <div class="trace" aria-label="选择查询对象">
      <button
        v-for="(q, i) in queries"
        :key="q"
        type="button"
        :class="{ active: selected === i, good: answers[i] === expected[i], bad: answers[i] !== expected[i] }"
        @click="selected = i"
      >
        {{ q }}<br />{{ answers[i] }}
      </button>
    </div>
    <div class="detail">
      <p>原始证据 1：Alice / 全局 / Python</p>
      <p>原始证据 2：Alice / 项目 A / TypeScript</p>
      <p>当前查询：{{ queries[selected] }}；正确结果：{{ expected[selected] }}</p>
      <p>版本化审核保留 entity、scope、source_id；撤销项目例外后可以回退到全局。</p>
    </div>
    <template #stats>
      <div class="kv">
        <span>正确查询</span><b :class="correct === 3 ? 'good' : 'bad'">{{ correct }} / 3</b>
      </div>
      <div class="kv">
        <span>跨范围错误</span
        ><b :class="answers[1] === expected[1] ? 'good' : 'bad'">{{ Number(answers[1] !== expected[1]) }}</b>
      </div>
      <div class="kv">
        <span>无证据时编造</span
        ><b :class="answers[2] === 'unknown' ? 'good' : 'bad'">{{ Number(answers[2] !== 'unknown') }}</b>
      </div>
      <p class="lab-note">这里测更新与检索规则，不测模型会不会服从记忆。完整 demo 还验证审核拒绝、回滚和索引重建。</p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import '@/styles/agent-experiments.css'
import LabFrame from '@/components/lab/LabFrame.vue'
import { memoryAnswers } from '@/utils/agent-reliability.js'
const mode = ref('overwrite'),
  selected = ref(1)
const queries = ['Alice / 项目 A', 'Alice / 项目 B', 'Bob / 项目 A'],
  expected = ['TypeScript', 'Python', 'unknown']
const answers = computed(() => memoryAnswers(mode.value))
const correct = computed(() => answers.value.filter((a, i) => a === expected[i]).length)
</script>
