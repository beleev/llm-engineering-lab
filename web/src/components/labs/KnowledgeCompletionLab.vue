<!-- 回放来自 Python 实际执行；沿用 LabFrame，与现有章节和目录合并。 -->
<template>
  <LabFrame v-for="entry in entries" :key="entry.id" :title="entry.title" :sub="records.kind"
            :module="entry.module" :run="entry.run" :challenge="entry.challenge" class="completion-lab" :data-experiment="entry.id">
    <template #controls>
      <div class="row" role="group" :aria-label="entry.title + '：选择实验条件'">
        <button v-for="condition in entry.conditions" :key="condition.id" type="button"
                :class="{ active: current(entry).id === condition.id }" :aria-pressed="current(entry).id === condition.id"
                @click="selected[entry.id] = condition.id">{{ condition.label }}</button>
      </div>
    </template>
    <div class="replay-details">
      <p v-for="(step, i) in current(entry).steps" :key="i" class="step">{{ step }}</p>
      <p v-if="current(entry).note" class="lab-note">{{ current(entry).note }}</p>
      <SourceSnippet :key="entry.module" :src="entry.module + ':experiment'" />
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
import SourceSnippet from '@/components/SourceSnippet.vue'
import records from '@/data/knowledge-completion.json'
const route = useRoute()
const selected = reactive({})
const entries = computed(() => records.entries.filter(entry => entry.routes.includes(route.name)))
const current = entry => entry.conditions.find(condition => condition.id === selected[entry.id]) || entry.conditions[0]
const format = value => Array.isArray(value) ? value.join(', ') : typeof value === 'boolean' ? (value ? '是' : '否') : value
</script>

<style scoped>
.row { display: flex; flex-wrap: wrap; gap: 6px; }
.replay-details { min-width: 0; }
.step { white-space: pre-wrap; overflow-wrap: anywhere; margin-bottom: 10px; font-size: 13px; }
.kv { gap: 12px; align-items: baseline; }
.kv > span, .kv > b { overflow-wrap: anywhere; min-width: 0; }
.kv > b { max-width: 65%; }
</style>
