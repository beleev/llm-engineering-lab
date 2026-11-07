<!-- 研究主题沿用现有章节与 LabFrame；数据与讲解随实验组件按需加载。 -->
<template>
  <LabFrame v-for="entry in entries" :key="entry.id" :title="entry.title" :sub="records.kind"
            :module="entry.module" :run="entry.run" :challenge="entry.challenge"
            class="research-lab" :data-research="entry.id">
    <template #controls>
      <div class="row" role="group" :aria-label="entry.title + '：选择实验条件'">
        <button v-for="condition in entry.conditions" :key="condition.id" type="button"
                :class="{ active: current(entry).id === condition.id }"
                :aria-pressed="current(entry).id === condition.id"
                @click="selected[entry.id] = condition.id">{{ condition.label }}</button>
      </div>
    </template>
    <div class="research-content">
      <Prose class="intuition" :text="entry.intuition" />
      <details class="reading">
        <summary>先理解原理与验证方法</summary>
        <Prose v-for="(paragraph, i) in entry.principles" :key="i" :text="paragraph" />
        <p class="protocol"><strong>怎样验证：</strong>{{ entry.protocol }}</p>
        <ul v-if="entry.references.length" class="references">
          <li v-for="ref in entry.references" :key="ref.url"><a :href="ref.url" target="_blank" rel="noopener noreferrer">{{ ref.title }} ↗</a></li>
        </ul>
      </details>
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
import LabFrame from '@/components/lab/LabFrame.vue'
import Prose from '@/components/Prose.vue'
import SourceSnippet from '@/components/SourceSnippet.vue'
import records from '@/data/research-completion.json'
const route = useRoute()
const selected = reactive({})
const entries = computed(() => records.entries.filter(entry => entry.routes.includes(route.name)))
const current = entry => entry.conditions.find(condition => condition.id === selected[entry.id]) || entry.conditions[0]
const format = value => Array.isArray(value) ? value.map(format).join(', ') : typeof value === 'boolean' ? (value ? '是' : '否') : value
</script>

<style scoped>
.row { display: flex; flex-wrap: wrap; gap: 6px; }
.research-content { min-width: 0; }
.intuition { margin-bottom: 12px; }
.reading { border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 10px 12px; margin-bottom: 16px; }
.reading summary { cursor: pointer; color: var(--accent); font-size: 13px; }
.reading[open] summary { margin-bottom: 12px; }
.reading :deep(p), .protocol, .step { font-size: 13px; overflow-wrap: anywhere; line-height: 1.8; margin-bottom: 10px; }
.references { padding-left: 20px; font-size: 12px; }
.kv { gap: 12px; align-items: baseline; }
.kv > span, .kv > b { overflow-wrap: anywhere; min-width: 0; }
.kv > b { max-width: 62%; }
</style>
