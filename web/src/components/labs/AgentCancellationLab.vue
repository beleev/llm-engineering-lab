<!-- 对照实验: m22_runtime_reliability。数值由 Python / JS 一致性测试校验。 -->
<template>
  <LabFrame
    class="agent-experiment"
    title="最快返回的，可能是错的"
    sub="三个子任务的完成时刻为 1、3、6。第一个结果不合格，只有通过验收的结果才能触发取消。"
    module="llm_agent/m22_runtime_reliability"
    run="python -m llm_agent.m22_runtime_reliability.demo"
    :challenge="{
      ask: '第 1 格已经有人返回了，为什么还要继续工作？',
      answer:
        '那个结果没有通过验收。第 3 格出现合格结果后，可以请求取消慢任务。本实验让执行器当场确认；真实系统还要等待确认，已经发生的副作用不会自动回滚。',
    }"
  >
    <template #controls>
      <label
        >收尾策略
        <select v-model="mode">
          <option value="all">等待全部</option>
          <option value="first-valid">首个合格结果后取消</option>
        </select></label
      >
      <LabSlider v-model="budget" label="总工作量上限" :min="0" :max="12" />
    </template>
    <div class="trace" aria-label="逻辑时钟">
      <button
        v-for="(frame, i) in result.frames"
        :key="i"
        type="button"
        :class="{ active: selected === i }"
        @click="selected = i"
      >
        t={{ frame.tick }}<br />累计工作 {{ frame.work }}
      </button>
    </div>
    <div class="detail">
      <p v-for="(state, job) in frame.states" :key="job">{{ job }}：{{ state }}</p>
      <p>取消请求 → 执行器确认 → 不再发新动作。旧任务的迟到结果还要检查任务版本。</p>
    </div>
    <template #stats>
      <div class="kv">
        <span>合格结果</span><b :class="result.accepted.length ? 'good' : 'bad'">{{ result.accepted.length }}</b>
      </div>
      <div class="kv">
        <span>总工作量</span><b>{{ result.work }}</b>
      </div>
      <div class="kv">
        <span>最后工作时刻</span><b>{{ result.elapsed }}</b>
      </div>
      <p class="lab-note">时间和工作量是逻辑单位，没有真实线程或网络延迟。</p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import '@/styles/agent-experiments.css'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { schedule } from '@/utils/agent-reliability.js'
const mode = ref('all'),
  budget = ref(12),
  selected = ref(0)
const result = computed(() => schedule(mode.value, budget.value))
const frame = computed(() => result.value.frames[Math.min(selected.value, result.value.frames.length - 1)])
</script>
