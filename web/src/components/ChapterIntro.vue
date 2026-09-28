<template>
  <div class="chapter-intro">
    <!-- 核心一句话 + 关键问题 -->
    <div class="intro-thesis">
      <div class="thesis-row">
        <span class="thesis-label">本章一句话</span>
        <Prose class="thesis-text" :text="tldr" />
      </div>
      <div v-if="question" class="thesis-row">
        <span class="thesis-label q">关键问题</span>
        <Prose class="thesis-text" :text="question" />
      </div>
    </div>

    <!-- 学完本章你能做什么 (可选) -->
    <div v-if="goals.length" class="intro-goals">
      <span class="goals-label">读完本章你能</span>
      <ul class="goals-list">
        <li v-for="g in goals" :key="g">
          <span class="goal-check" aria-hidden="true">✓</span>
          <Tex :text="g" />
        </li>
      </ul>
    </div>

    <!-- 元信息: 代码路径 + 上一章 + 下一章 -->
    <div class="intro-meta">
      <div v-if="codeRefs.length" class="meta-block">
        <span class="meta-label">对应代码</span>
        <span class="meta-codes">
          <RepoLink
            v-for="r in codeRefs"
            :key="r.path + (r.line || '')"
            :path="r.path"
            :line="r.line"
            :label="r.label"
            tiny
          />
        </span>
      </div>
      <div v-if="before" class="meta-block">
        <span class="meta-label">上一章</span>
        <router-link :to="{ name: before.name }" class="meta-link">
          ← {{ before.label }}
        </router-link>
      </div>
      <div v-if="after" class="meta-block">
        <span class="meta-label">下一章</span>
        <router-link :to="{ name: after.name }" class="meta-link">
          {{ after.label }} →
        </router-link>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import RepoLink from '@/components/RepoLink.vue'
import Prose from '@/components/Prose.vue'
import Tex from '@/components/Tex.vue'
import { parseRef } from '@/utils/repo.js'
import { useChapterNav } from '@/composables/useChapterNav.js'

const props = defineProps({
  tldr:     { type: String, required: true },
  question: { type: String, default: '' },
  // 字符串写法: 单条, 或用 ' · ' / ',' 分隔多条 (路径里不含逗号或中点)
  code:     { type: String, default: '' },
  // 数组写法: 每项 { path, line?, label? }。两种都传时用数组
  codes:    { type: Array, default: () => [] },
  // "读完本章你能..." 的小目标清单, 帮助新手判断是否要往下读
  goals:    { type: Array, default: () => [] },
  // 上一章 / 下一章。和 ChapterNav 同一条规则: 当前路由在 learningPath 里时忽略这两个 prop,
  // 用 learningPath 按当前档位推出来的结果; 只有路由不在 learningPath 里时才用传进来的值。
  // 这里列的是阅读顺序上的前后两章, 不表示「必须先学会」的依赖关系。
  prereq:   { type: Object, default: null },
  nextStep: { type: Object, default: null },
})

const nav = useChapterNav()
const before = computed(() => (nav.inPath.value ? nav.prev.value : props.prereq))
const after = computed(() => (nav.inPath.value ? nav.next.value : props.nextStep))

// 把字符串形态拆成可链接的 ref 列表;  显式 codes 优先。
const codeRefs = computed(() => {
  if (props.codes && props.codes.length) {
    return props.codes.map(c => ({
      path: c.path,
      line: c.line ?? null,
      label: c.label ?? c.path,
    }))
  }
  if (!props.code) return []
  // 拆分: 中点 ' · ' 或英文逗号; 含 "{a,b}" 通配的整体保留为单条 (不易拆)
  const raw = props.code.includes(' · ')
    ? props.code.split(' · ')
    : props.code.split(/,(?![^{]*\})/)  // 不切分 {a,b} 内的逗号
  return raw
    .map(s => s.trim())
    .filter(Boolean)
    .map(s => {
      const r = parseRef(s)
      return { path: r.path, line: r.line, label: r.label || s }
    })
})
</script>

<style scoped>
.chapter-intro {
  margin: 0 0 28px;
  border: 1px solid var(--border);
  border-left: 3px solid var(--accent);
  border-radius: var(--radius-sm);
  background: linear-gradient(
    180deg,
    color-mix(in srgb, var(--accent) 6%, var(--bg-card)) 0%,
    var(--bg-card) 100%
  );
  overflow: hidden;
}

.intro-thesis {
  padding: 14px 18px 12px;
  border-bottom: 1px dashed var(--border);
}
.thesis-row {
  display: flex;
  gap: 14px;
  align-items: baseline;
  font-size: 13.5px;
  line-height: 1.65;
}
.thesis-row + .thesis-row { margin-top: 6px; }
.thesis-label {
  flex: 0 0 auto;
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 1px;
  text-transform: uppercase;
  color: var(--accent);
  background: var(--accent-soft);
  padding: 3px 8px;
  border-radius: 4px;
  font-family: "SF Mono", Menlo, monospace;
}
.thesis-label.q {
  color: var(--eye);
  background: color-mix(in srgb, var(--eye) 18%, transparent);
}
.thesis-text { color: var(--text); }

.intro-goals {
  display: flex;
  align-items: flex-start;
  gap: 14px;
  padding: 10px 18px;
  border-bottom: 1px dashed var(--border);
  font-size: 12.5px;
}
.goals-label {
  flex: 0 0 auto;
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.8px;
  text-transform: uppercase;
  color: var(--left);
  background: color-mix(in srgb, var(--left) 18%, transparent);
  padding: 3px 8px;
  border-radius: 4px;
  font-family: "SF Mono", Menlo, monospace;
  margin-top: 2px;
}
.goals-list {
  list-style: none;
  margin: 0;
  padding: 0;
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 4px 18px;
  color: var(--text);
  line-height: 1.55;
}
.goals-list li {
  display: flex;
  gap: 8px;
  align-items: baseline;
}
.goal-check {
  color: var(--left);
  font-size: 11px;
  font-weight: 600;
  flex: 0 0 auto;
}

.intro-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 18px 28px;
  padding: 10px 18px;
  font-size: 12px;
}
.meta-block {
  display: flex;
  align-items: center;
  gap: 8px;
}
.meta-label {
  color: var(--text-dim);
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.8px;
}
.meta-code {
  background: var(--code-bg);
  border: 1px solid var(--border);
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 11.5px;
  color: var(--code-text);
  font-family: "SF Mono", Menlo, monospace;
}
.meta-codes {
  display: inline-flex;
  flex-wrap: wrap;
  gap: 4px 6px;
  align-items: center;
}
.meta-link {
  color: var(--text);
  font-size: 12px;
  border-bottom: 1px dashed var(--border-strong);
}
.meta-link:hover { color: var(--accent); border-bottom-color: var(--accent); text-decoration: none; }
</style>
