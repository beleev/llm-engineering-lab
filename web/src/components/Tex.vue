<!--
  一段可能夹着公式的文字: $…$ 行内, $$…$$ 独立成行。语法见 utils/tex.js。
  只有公式部分走 v-html (KaTeX 的输出), 普通文字照旧转义。
-->
<template>
  <template v-for="(p, i) in parts" :key="i">
    <span v-if="p.html" :class="p.display ? 'tex-block' : 'tex'" v-html="p.html" />
    <template v-else>{{ p.text }}</template>
  </template>
</template>

<script setup>
import { computed } from 'vue'
import { renderTex, texParts } from '@/utils/tex'

const props = defineProps({ text: { type: [String, Number], default: '' } })
const parts = computed(() => texParts(String(props.text)).map((p) => (p.tex != null ? { html: renderTex(p.tex, p.display), display: p.display } : p)))
</script>
