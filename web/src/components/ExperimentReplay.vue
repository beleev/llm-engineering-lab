<!-- 研究主题沿用现有章节与 LabFrame; 数据与讲解随实验组件按需加载。 -->
<template>
  <LabFrame v-for="entry in entries" :key="entry.id" :title="entry.title" :sub="records.kind"
            :module="entry.module" :run="entry.run" :challenge="entry.challenge"
            class="research-lab" v-bind="{ ['data-' + namespace]: entry.id }">
    <template #controls>
      <div class="row" role="group" :aria-label="entry.title + ': 选择实验条件'">
        <button v-for="condition in entry.conditions" :key="condition.id" type="button"
                :class="{ active: current(entry).id === condition.id }"
                :aria-pressed="current(entry).id === condition.id"
                @click="selected[entry.id] = condition.id">{{ condition.label }}</button>
      </div>
    </template>
    <!-- 讲解放全宽: 演进链 → 直觉 → 原理与验证方法 (默认展开); 下面左栏是这个条件下的执行记录, 右栏是指标 -->
    <template #intro>
      <EvolutionChain v-if="entry.evolution" :title="'演进逻辑链 · ' + entry.evolution.title" :subtitle="entry.evolution.subtitle" :steps="evoSteps(entry.evolution)" />
      <Prose class="intuition" :text="entry.intuition" />
      <details class="reading" open>
        <summary>原理与验证方法</summary>
        <Prose v-for="(paragraph, i) in entry.principles" :key="i" :text="paragraph" />
        <p class="protocol"><strong>怎样验证: </strong><Prose :text="entry.protocol" /></p>
        <ul v-if="entry.references.length" class="references">
          <li v-for="ref in entry.references" :key="ref.url"><a :href="ref.url" target="_blank" rel="noopener noreferrer">{{ ref.title }} ↗</a></li>
        </ul>
      </details>
    </template>
    <div class="research-content">
      <p class="exec-label">执行记录 · {{ current(entry).label }}</p>
      <div class="execution" aria-live="polite">
        <p v-for="(step, i) in current(entry).steps" :key="i" class="step">{{ step }}</p>
        <p class="lab-note">{{ current(entry).note }}</p>
      </div>
      <SourceSnippet :src="entry.module + ':experiment'" />
    </div>
    <template #stats>
      <div aria-live="polite" aria-atomic="true">
        <div v-for="(value, key) in current(entry).metrics" :key="key" class="kv">
          <span>{{ key }}</span><b>{{ format(value) }}</b>
        </div>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, reactive } from 'vue'
import { useRoute } from 'vue-router'
import EvolutionChain from '@/components/EvolutionChain.vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import Prose from '@/components/Prose.vue'
import SourceSnippet from '@/components/SourceSnippet.vue'
const props = defineProps({ records: { type: Object, required: true }, namespace: { type: String, required: true } })
const route = useRoute()
const selected = reactive({})
// 和 StageTopic.vue 的 EVO_COLORS 一致: 按步序配色
const EVO_COLORS = ['var(--text-muted)', 'var(--accent)', 'var(--left)', 'var(--right)', 'var(--eye)']
const evoSteps = e => e.steps.map((s, i) => ({ color: EVO_COLORS[i % EVO_COLORS.length], ...s }))
const entries = computed(() => props.records.entries.filter(entry => entry.routes.includes(route.name)))
const current = entry => entry.conditions.find(condition => condition.id === selected[entry.id]) || entry.conditions[0]
const format = value => value == null ? '不适用' : Array.isArray(value) ? value.map(format).join(', ') : typeof value === 'boolean' ? (value ? '是' : '否') : value
</script>

<style scoped>
.row { display: flex; flex-wrap: wrap; gap: 6px; }
.research-content { min-width: 0; }
.intuition { margin: 4px 0 12px; font-size: 14px; line-height: 1.75; max-width: 960px; }
.reading { border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 10px 14px; margin-bottom: 16px; max-width: 960px; }
.reading summary { cursor: pointer; color: var(--accent); font-size: 13px; }
.reading[open] summary { margin-bottom: 12px; }
.reading :deep(p), .protocol, .step { font-size: 13.5px; overflow-wrap: anywhere; line-height: 1.8; margin-bottom: 10px; }
.protocol { display: flex; gap: 6px; flex-wrap: wrap; }
.protocol :deep(.prose) { flex: 1; min-width: 200px; }
.exec-label { font-size: 11px; letter-spacing: 0.8px; text-transform: uppercase; color: var(--text-dim); margin-bottom: 8px; }
.references { padding-left: 20px; font-size: 12px; }
.kv { gap: 12px; align-items: baseline; }
.kv > span, .kv > b { overflow-wrap: anywhere; min-width: 0; }
.kv > b { max-width: 62%; }
</style>
