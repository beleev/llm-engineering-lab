<template>
  <section class="section stage-contents" aria-label="本阶段目录">
    <h2>按主题阅读</h2>
    <p class="lead">{{ stage.groups.length }} 个主题，{{ stage.chapters.length }} 个章节。展开主题查看具体章节；目录按当前阅读档位筛选。</p>
    <div class="topic-grid">
      <details v-for="g in groups" :key="g.id" class="topic-group">
        <summary><span class="mono">{{ g.number }}</span> {{ g.title }} <span class="count">{{ g.chapters.length }} 章</span></summary>
        <ol>
          <li v-for="c in g.chapters" :key="c.route">
            <router-link :to="{ name: c.route }"><span class="mono">{{ c.number }}</span> {{ c.label }}</router-link>
            <p>{{ c.hint }}</p>
          </li>
        </ol>
      </details>
    </div>
  </section>
</template>

<script setup>
import { computed } from 'vue'
import { stageBy } from '@/data/models.js'
import { inLevel } from '@/data/tiers.js'
import { useProgress } from '@/composables/useProgress.js'
const props = defineProps({ stageId: { type: String, required: true } })
const progress = useProgress()
const stage = computed(() => stageBy[props.stageId])
const groups = computed(() => stage.value.groups.map((g) => ({ ...g, chapters: g.chapters.filter((c) => inLevel(c.route, progress.state.level)) })).filter((g) => g.chapters.length))
</script>

<style scoped>
/* 后面可能直接接演进卡片；与普通 section 使用相同的段落间距。 */
.stage-contents { margin-bottom: 36px; }
.topic-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; align-items: start; }
.topic-group { border: 1px solid var(--border); border-radius: var(--radius-sm); background: var(--bg-card); }
summary { padding: 14px; cursor: pointer; font-size: 14px; font-weight: 600; line-height: 1.6; }
summary:hover { color: var(--accent); }
.mono { color: var(--text-dim); font-size: 11px; margin-right: 4px; }
.count { white-space: nowrap; font-size: 11px; color: var(--text-muted); font-weight: 400; }
ol { list-style: none; padding: 0 14px 10px; margin: 0; }
li { padding: 9px 0; border-top: 1px solid var(--border); font-size: 13px; }
li a { display: block; padding: 3px 0; }
li p { margin-top: 4px; color: var(--text-dim); font-size: 11px; line-height: 1.6; overflow-wrap: anywhere; }
@media (max-width: 650px) { .topic-grid { grid-template-columns: 1fr; } }
</style>
