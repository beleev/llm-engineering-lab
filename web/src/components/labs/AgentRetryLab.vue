<!-- 对照实验: m22_runtime_reliability。数值由 Python / JS 一致性测试校验。 -->
<template>
  <LabFrame
    class="agent-experiment"
    title="响应丢了，要不要再写一次？"
    sub="服务端已经记账，但第一次响应丢失。切换重试方式，查看真实写入次数。"
    module="llm_agent/m22_runtime_reliability"
    run="python -m llm_agent.m22_runtime_reliability.demo"
    :challenge="{
      ask: '一次超时，能证明服务端没执行吗？',
      answer:
        '不能。盲目重试可能再扣一次；服务端幂等键返回原收据，或先查询状态，都可以避免这个例子中的重复写入。授权拒绝不能靠重试绕过。',
    }"
  >
    <template #controls>
      <label
        >重试方式
        <select v-model="mode">
          <option value="blind">盲目重试</option>
          <option value="idempotent">同一幂等键</option>
          <option value="query">先查询收据</option>
        </select></label
      >
      <LabSlider v-model="attempts" label="提交次数上限" :min="1" :max="3" />
      <label><input v-model="allowed" type="checkbox" />有写入权限</label>
    </template>
    <div class="trace" aria-label="提交与确认过程">
      <button
        v-for="(text, i) in result.frames"
        :key="i"
        type="button"
        :class="{ active: selected === i }"
        @click="selected = i"
      >
        {{ i + 1 }} · {{ text }}
      </button>
    </div>
    <div class="detail">
      <p>{{ result.frames[Math.min(selected, result.frames.length - 1)] }}</p>
      <p>客户端状态：{{ result.status }}</p>
      <p>unknown 表示无法确认；它既不等于未执行，也不等于已成功确认。</p>
    </div>
    <template #stats>
      <div class="kv">
        <span>服务端写入</span><b :class="result.writes > 1 ? 'bad' : 'good'">{{ result.writes }} 次</b>
      </div>
      <div class="kv">
        <span>实际金额</span><b :class="result.amount > 10 ? 'bad' : ''">{{ result.amount }} 元</b>
      </div>
      <div class="kv">
        <span>提交 / 查询</span><b>{{ result.requests }} / {{ result.queries }}</b>
      </div>
      <p class="lab-note">内存账本模拟。幂等记录与副作用同处服务端；不代表跨服务事务已经实现 exactly-once。</p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import '@/styles/agent-experiments.css'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { retryWrite } from '@/utils/agent-reliability.js'
const mode = ref('blind'),
  attempts = ref(2),
  allowed = ref(true),
  selected = ref(0)
const result = computed(() => retryWrite(mode.value, allowed.value, attempts.value))
</script>
