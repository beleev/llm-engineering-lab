<template>
  <section :id="section.id" class="section topic-supplement" tabindex="-1">
    <h2>{{ section.title }}</h2>
    <Prose class="lead" :text="section.tldr" />
    <nav v-if="section.related?.length" aria-label="相关段落" class="supplement-links">
      <RouterLink v-for="link in section.related" :key="link.route" :to="{ name: link.route, hash: link.anchor ? '#' + link.anchor : '' }">{{ link.label }} →</RouterLink>
    </nav>
    <LabMount :names="section.labNames || []" />
    <EvolutionChain v-if="section.evolution" v-bind="section.evolution" />
    <div class="grid grid-3">
      <div v-for="point in section.points" :key="point.title" class="card">
        <h3><Tex :text="point.title" /></h3>
        <Prose class="desc" :text="point.body" />
      </div>
    </div>
    <details class="card supplement-source">
      <summary>对照源码与实现边界</summary>
      <Prose v-if="section.question" class="lead" :text="section.question" />
      <div v-for="row in section.sourceRows" :key="row.concept" class="source-row">
        <h3><Tex :text="row.concept" /></h3><code>{{ row.code }}</code><Prose :text="row.takeaway" />
      </div>
      <CodeBlock v-if="section.snippet" :code="section.snippet" />
      <SourceSnippet v-for="src in [].concat(section.source || [])" :key="src" :src="src" />
      <p v-if="section.run" class="hint">运行：<code>{{ section.run }}</code></p>
      <div v-for="link in section.links" :key="link.from + link.to" class="source-row">
        <strong>{{ link.from }} → {{ link.to }}</strong><Prose :text="link.body" />
      </div>
    </details>
  </section>
</template>
<script setup>
import Prose from './Prose.vue'
import Tex from './Tex.vue'
import LabMount from './LabMount.vue'
import EvolutionChain from './EvolutionChain.vue'
import CodeBlock from './CodeBlock.vue'
import SourceSnippet from './SourceSnippet.vue'
defineProps({ section: { type: Object, required: true } })
</script>
<style scoped>
.topic-supplement { scroll-margin-top: 24px; }
.topic-supplement:focus { outline: none; }
.supplement-source { margin-top: 20px; overflow-wrap: anywhere; }
summary { cursor: pointer; color: var(--accent); }
.source-row { margin: 16px 0; }
.supplement-links { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 16px; }
code { overflow-wrap: anywhere; }
</style>
