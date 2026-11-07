<!-- 对照实验: m21_completion_verification。数值由 Python / JS 一致性测试校验。 -->
<template>
  <LabFrame
    class="agent-experiment"
    title="“完成”之后，再看一眼环境"
    sub="任务要求三条笔记，规则模型第一轮只写两条。选择验收方式，并限制修复次数。"
    module="llm_agent/m21_completion_verification"
    run="python -m llm_agent.m21_completion_verification.demo"
    :challenge="{
      ask: '允许修复一次，就一定能成功吗？',
      answer:
        '不一定。模型可以一直只说完成，或者工具一直失败。验收器只能指出缺口；修复预算耗尽后必须报告未完成，不能把验证器变成代写工具。',
    }"
  >
    <template #controls>
      <label
        >验收方式
        <select v-model="mode">
          <option value="direct">直接接受 final</option>
          <option value="text">回答包含“完成”</option>
          <option value="verified">读取环境验收</option>
        </select></label
      >
      <LabSlider v-model="repairs" label="最多修复次数" :min="0" :max="2" />
      <label><input v-model="stubborn" type="checkbox" />模型始终只说完成</label>
    </template>
    <div class="trace" aria-label="执行与验收时间轴">
      <button
        v-for="(frame, i) in result.frames"
        :key="i"
        type="button"
        :class="{ active: selected === i }"
        @click="selected = i"
      >
        {{ i + 1 }} · {{ frame.text }}
      </button>
    </div>
    <div class="detail" aria-live="polite">
      <p>{{ frame.text }}</p>
      <p>这一时刻实际写入 {{ frame.count }} / 3 条。</p>
      <div class="meter"><span :style="{ width: (frame.count / 3) * 100 + '%' }" /></div>
      <p>已有笔记 existing 始终保留。模型总调用预算为 4。</p>
    </div>
    <template #stats>
      <div class="kv">
        <span>系统宣告通过</span><b>{{ result.passed ? '是' : '否' }}</b>
      </div>
      <div class="kv">
        <span>环境实际正确</span><b :class="result.actual ? 'good' : 'bad'">{{ result.actual ? '是' : '否' }}</b>
      </div>
      <div class="kv">
        <span>模型调用</span><b>{{ result.calls }} / 4</b>
      </div>
      <p class="lab-note">时间轴按确定性策略重算。检查本身不写笔记，Stop Hook 仍只负责通知。</p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import '@/styles/agent-experiments.css'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { completion } from '@/utils/agent-reliability.js'
const mode = ref('direct'),
  repairs = ref(1),
  stubborn = ref(false),
  selected = ref(0)
const result = computed(() => completion(mode.value, repairs.value, stubborn.value))
const frame = computed(() => result.value.frames[Math.min(selected.value, result.value.frames.length - 1)])
</script>
