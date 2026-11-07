<!-- 对照实验: m24_agentic_rag。数值由 Python / JS 一致性测试校验。 -->
<template>
  <LabFrame
    class="agent-experiment"
    title="相关，不代表足够回答"
    sub="先确认 Comet 指的是哪个产品版本，再找对应上传限制。点击文档查看它是否在证据链中。"
    module="llm_agent/m24_agentic_rag"
    run="python -m llm_agent.m24_agentic_rag.demo"
    :challenge="{
      ask: '知道 Atlas v2 的限制是 20 MB，却没有 Comet 的别名资料，能回答 Comet 的问题吗？',
      answer: '不能。数字正确也可能用错对象。别名证据和版本限制共同支持两跳答案；如果只问 Atlas v2，一次查询就够。',
    }"
  >
    <template #controls>
      <label
        >问题
        <select v-model="question">
          <option :value="0">Atlas v2 上传限制？</option>
          <option :value="1">Comet 上传限制？</option>
          <option :value="2">Nova 上传限制？</option>
        </select></label
      >
      <label
        >检索方式
        <select v-model="mode">
          <option value="plain">无上下文的一跳</option>
          <option value="contextual">加对象/版本的一跳</option>
          <option value="agentic">按证据继续检索</option>
        </select></label
      >
      <LabSlider v-model="budget" label="查询预算" :min="0" :max="3" />
    </template>
    <div class="trace" aria-label="证据文档">
      <button
        v-for="(doc, i) in ragDocs"
        :key="doc.id"
        type="button"
        :class="{ active: selected === i, good: result.sources.includes(doc.id) }"
        @click="selected = i"
      >
        {{ doc.id }}<br />{{ result.sources.includes(doc.id) ? '已引用' : '未引用' }}
      </button>
    </div>
    <div class="detail">
      <p>{{ ragDocs[selected].body }}</p>
      <p>对象：{{ ragDocs[selected].entity }}；版本：{{ ragDocs[selected].version }}</p>
      <p v-for="(query, i) in result.queries" :key="i">查询 {{ i + 1 }}：{{ query }}</p>
    </div>
    <template #stats>
      <div class="kv">
        <span>回答</span><b>{{ result.answer === null ? '证据不足' : result.answer + ' MB' }}</b>
      </div>
      <div class="kv">
        <span>查询次数</span><b>{{ result.queries.length }} / {{ budget }}</b>
      </div>
      <div class="kv">
        <span>引用证据</span><b>{{ result.sources.length }}</b>
      </div>
      <p class="lab-note">沿用 Python 的 TF-IDF 和确定性多跳规则；没有神经 embedding，也没有自由规划模型。</p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import '@/styles/agent-experiments.css'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { evidenceChain, ragDocs } from '@/utils/agent-reliability.js'
const question = ref(1),
  mode = ref('contextual'),
  budget = ref(2),
  selected = ref(0)
const result = computed(() => evidenceChain(question.value, mode.value, budget.value))
</script>
