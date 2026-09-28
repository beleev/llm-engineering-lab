<template>
  <nav class="chapter-nav">
    <router-link v-if="prev" :to="{ name: prev.name }" class="nav-side prev">
      <span class="dir">← 上一章<template v-if="levelLabel"> · {{ levelLabel }}档</template></span>
      <span class="title">{{ prev.label }}</span>
      <span v-if="prev.hint" class="hint">{{ prev.hint }}</span>
    </router-link>
    <span v-else class="nav-side disabled" />

    <router-link v-if="next" :to="{ name: next.name }" class="nav-side next">
      <span class="dir">下一章<template v-if="levelLabel"> · {{ levelLabel }}档</template> →</span>
      <span class="title">{{ next.label }}</span>
      <span v-if="next.hint" class="hint">{{ next.hint }}</span>
    </router-link>
    <span v-else class="nav-side disabled" />
  </nav>
</template>

<script setup>
// 上一章 / 下一章由 useChapterNav 从 learningPath 推, 并按当前阅读档位过滤。
//
// props 什么时候生效:
//   当前路由在 learningPath 里 (所有章节页) → 忽略 props, 用推出来的结果。
//     手写的 prev / next 在插入新章后会过期, 所以章节页上传进来的值不会被采用。
//   当前路由不在 learningPath 里 (速成路线、术语速查、兜底页) → 用 props。
import { computed } from 'vue'
import { useChapterNav } from '@/composables/useChapterNav.js'

const props = defineProps({
  prev: { type: Object, default: null },   // { name, label, hint? }, 只在路由不属于 learningPath 时使用
  next: { type: Object, default: null },
})
const nav = useChapterNav()
const levelLabel = computed(() => (nav.inPath.value ? nav.levelLabel.value : ''))
const prev = computed(() => (nav.inPath.value ? nav.prev.value : props.prev))
const next = computed(() => (nav.inPath.value ? nav.next.value : props.next))
</script>

<style scoped>
.chapter-nav {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
  margin-top: 48px;
  padding-top: 24px;
  border-top: 1px solid var(--border);
}
.nav-side {
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 14px 16px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  color: inherit;
  text-decoration: none;
  transition: background-color 150ms ease-out, border-color 150ms ease-out, transform 150ms ease-out;
}
.nav-side:hover {
  border-color: var(--accent);
  transform: translateY(-1px);
  text-decoration: none;
}
.nav-side.next { text-align: right; }
.nav-side.disabled {
  background: transparent;
  border: 1px dashed var(--border);
  pointer-events: none;
}
.dir {
  font-size: 11px;
  color: var(--text-dim);
  letter-spacing: 0.8px;
  text-transform: uppercase;
}
.title {
  font-size: 15px;
  font-weight: 600;
  color: var(--text);
}
.hint {
  font-size: 12px;
  color: var(--text-muted);
  line-height: 1.5;
}
</style>
