<!-- 对照实验: m23_context_cache。数值由 Python / JS 一致性测试校验。 -->
<template>
  <LabFrame
    class="agent-experiment"
    title="压短了，就一定省钱吗？"
    sub="一起计算摘要调用、缓存读写与动态记忆。点击某一轮，查看这次真正送入的上下文大小。"
    module="llm_agent/m23_context_cache"
    run="python -m llm_agent.m23_context_cache.demo"
    :challenge="{
      ask: '把轮数调成 2，摘要阈值调成 2。输入变短后，为什么总费用可能变高？',
      answer:
        '摘要自己也读输入、生成输出，并改写后续前缀。短会话来不及摊薄这些开销。完整实验还检查范围和否定是否保留，费用相同也可能丢掉关键约束。',
    }"
  >
    <template #controls>
      <label
        >动态状态位置
        <select v-model="placement">
          <option value="front">最前部</option>
          <option value="tail">历史之后</option>
        </select></label
      >
      <LabSlider v-model="turns" label="会话轮数" :min="2" :max="10" />
      <label
        >摘要阈值
        <select v-model="threshold">
          <option :value="0">不摘要</option>
          <option :value="2">2 个历史块</option>
          <option :value="4">4 个历史块</option>
          <option :value="6">6 个历史块</option>
        </select></label
      >
      <label><input v-model="keepScope" type="checkbox" />摘要保留范围与禁止事项</label>
    </template>
    <div class="trace" aria-label="每轮请求">
      <button
        v-for="row in result.rows"
        :key="row.turn"
        type="button"
        :class="{ active: selected === row.turn - 1 }"
        @click="selected = row.turn - 1"
      >
        第 {{ row.turn }} 轮<br />{{ row.input }} tok
      </button>
    </div>
    <div class="detail">
      <p>本轮输入 {{ frame.input }} tok；请求费用 {{ frame.cost.toFixed(1) }}。</p>
      <div class="meter"><span :style="{ width: (frame.input / result.peak) * 100 + '%' }" /></div>
      <p>同位置不摘要：{{ baseline.cost.toFixed(1) }}；当前总价：{{ result.cost.toFixed(1) }}。</p>
      <p>三条约束：项目 A、TypeScript、禁止删除。原始证据仍有 {{ result.audit_blocks }} 块。</p>
    </div>
    <template #stats>
      <div class="kv">
        <span>模拟总费用</span
        ><b :class="result.cost < baseline.cost ? 'good' : result.cost > baseline.cost ? 'bad' : ''">{{
          result.cost.toFixed(1)
        }}</b>
      </div>
      <div class="kv">
        <span>其中摘要费用</span><b>{{ result.summary_cost.toFixed(1) }}</b>
      </div>
      <div class="kv">
        <span>约束保留</span><b :class="result.retained === 3 ? 'good' : 'bad'">{{ result.retained }} / 3</b>
      </div>
      <div class="kv">
        <span>峰值上下文</span><b>{{ result.peak }} tok</b>
      </div>
      <p class="lab-note">
        固定大小文本块与 m19 的费用模型：读 ×0.1，写 ×1.25，摘要输出 ×1。不是实际 API 价格或 tokenizer 测量。
      </p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import '@/styles/agent-experiments.css'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { contextCache } from '@/utils/agent-reliability.js'
const turns = ref(6),
  placement = ref('tail'),
  threshold = ref(4),
  keepScope = ref(true),
  selected = ref(0)
const result = computed(() => contextCache(turns.value, placement.value, threshold.value, keepScope.value))
const baseline = computed(() => contextCache(turns.value, placement.value, 0, true))
const frame = computed(() => result.value.rows[Math.min(selected.value, result.value.rows.length - 1)])
</script>
