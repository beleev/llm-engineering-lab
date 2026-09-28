<!--
  真源码片段: <SourceSnippet src="llm_infer/m07_speculative_decoding/demo.py:spec_decode_greedy" />
  内容在构建期直接读自 Python 文件, 不是手抄的, 所以永远和仓库一致。
-->
<template>
  <details class="source" :open="open" @toggle="onToggle">
    <summary>
      <span class="toggle"><span class="caret" aria-hidden="true">▸</span> {{ isOpen ? '收起' : '展开真源码' }}</span>
      <span class="mono">{{ parsed.path }}</span>
      <span v-if="parsed.symbol" class="sym mono">· {{ parsed.symbol }}</span>
      <span v-if="data?.start" class="lines mono">L{{ data.start }}–{{ data.end }}</span>
    </summary>
    <p v-if="data?.error" class="err" role="alert">{{ data.error }}</p>
    <CodeBlock v-else-if="data" :code="data.code" />
    <p v-else class="loading">加载中…</p>
    <RepoLink :path="parsed.path" label="在 GitHub 上看完整文件" tiny />
  </details>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import CodeBlock from '@/components/CodeBlock.vue'
import RepoLink from '@/components/RepoLink.vue'
import { loadSource, parseRef } from '@/utils/sources.js'

const props = defineProps({
  src: { type: String, required: true },
  open: { type: Boolean, default: false },
})
const parsed = computed(() => parseRef(props.src))
const data = ref(null)
const isOpen = ref(props.open)
const fetchIt = async () => { data.value = await loadSource(props.src) }
// 加载失败后收起再展开会重试一次
const onToggle = (e) => {
  isOpen.value = e.target.open
  if (e.target.open && (!data.value || data.value.error)) fetchIt()
}
watch(() => props.src, () => { data.value = null; if (props.open) fetchIt() }, { immediate: true })
</script>

<style scoped>
.source { border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 8px 12px; margin-top: 10px; background: var(--bg-elev); }
.source summary { cursor: pointer; font-size: 12px; color: var(--text-muted); display: flex; gap: 8px; align-items: baseline; flex-wrap: wrap; }
.source summary::-webkit-details-marker { display: none; }
.source[open] summary { margin-bottom: 8px; }
.toggle { color: var(--accent); white-space: nowrap; }
.caret { display: inline-block; transition: transform 0.15s; }
.source[open] .caret { transform: rotate(90deg); }
.sym { color: var(--accent); }
.lines { margin-left: auto; color: var(--text-dim); font-size: 11px; }
.loading { font-size: 12px; color: var(--text-dim); }
.err { font-size: 12px; color: var(--danger); margin-bottom: 6px; }
</style>
