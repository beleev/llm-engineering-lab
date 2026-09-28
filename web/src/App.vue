<template>
  <div class="app" :class="{ 'nav-open': navOpen }" :style="{ '--side-w': `${sideW}px` }">
    <!-- 键盘用户的第一个 Tab 停在这里, 回车直接进正文, 不用先走完整个侧栏。
         用按钮不用 <a href="#main">: 站点是 hash 路由, 地址栏变成 #main 会被当成换页 -->
    <button type="button" class="skip-link" @click="skipToMain">跳到正文</button>
    <!-- 窄屏: 顶栏 + 抽屉式侧栏 -->
    <header class="topbar">
      <button ref="burger" type="button" class="hamburger" :aria-expanded="navOpen" aria-controls="sidebar"
              :aria-label="navOpen ? '关闭章节导航' : '打开章节导航'" @click="navOpen = !navOpen">{{ navOpen ? '✕' : '☰' }}</button>
      <span class="topbar-title">{{ route.meta.title || 'LLM 全栈教程' }}</span>
    </header>
    <div class="backdrop" @click="navOpen = false" />

    <!-- 窄屏抽屉关着时整个侧栏 inert: 看不见的 90 多个链接不进 Tab 顺序, 读屏也不读 -->
    <aside id="sidebar" class="sidebar" :inert="(narrow && !navOpen) || null">
      <div class="brand">
        <div class="brand-row">
          <h1>LLM 全栈教程</h1>
          <button
            class="theme-toggle"
            type="button"
            @click="toggleTheme"
            :aria-label="theme === 'dark' ? '切换到浅色主题' : '切换到深色主题'"
            :title="theme === 'dark' ? '切到浅色 ☀︎' : '切到深色 ☾'"
          >
            <span v-if="theme === 'dark'" aria-hidden="true">☀︎</span>
            <span v-else aria-hidden="true">☾</span>
          </button>
        </div>
        <p>原理、架构、训练、微调、推理、Agent 六段闭环</p>
        <div class="progress" :title="`已读 ${readCount} / ${chapters.length} 章`">
          <div class="progress-bar"><span :style="{ width: `${(readCount / chapters.length) * 100}%` }" /></div>
          <span class="mono">已读 {{ readCount }}/{{ chapters.length }}</span>
          <button type="button" class="reset-progress" aria-label="重置进度" title="清空已读记录和自测成绩" @click="resetProgress">重置</button>
        </div>
        <a class="repo-link" :href="repoUrl()" target="_blank" rel="noopener">
          <span class="gh-icon" aria-hidden="true">↗</span>
          GitHub · 仓库源码
        </a>
      </div>

      <nav>
        <!-- 阅读档位: 主干/冲刺时收起扩展章, 当前所在章始终可见 -->
        <div class="level-bar" role="group" aria-label="阅读档位">
          <button
            v-for="l in LEVELS" :key="l.id" type="button"
            :class="['level-btn', 'lv', `lv-${l.id}`, { active: level === l.id }]"
            :aria-pressed="level === l.id"
            :title="l.hint" @click="progress.setLevel(l.id)"
          >{{ l.label }} {{ levelCount(l.id) }}</button>
        </div>
        <!-- 图例常显: 触屏没有悬停, 只写在 title 里的说明永远看不到 -->
        <div class="level-legend">
          <p>{{ levelNow.label }}档: {{ levelNow.hint }}, 共 {{ levelCount(level) }} 章。侧栏和翻章都按这一档走。</p>
          <p><span class="tier-dot core">★</span>冲刺 <span class="tier-dot core">●</span>主干 <span class="tier-dot ext">○</span>扩展</p>
          <p><span class="mark done">✓</span> 自测全对 <span class="mark">已读</span> 打开过这一章</p>
        </div>
        <router-link :to="{ name: 'fast-track' }" class="nav-link fast">
          <span class="idx">→</span>
          <span>速成路线</span>
        </router-link>

        <!-- 序章 -->
        <div class="section-label">序章</div>
        <router-link :to="{ name: 'home' }" class="nav-link">
          <span class="idx">✦</span>
          <span>主线总览</span>
        </router-link>

        <!-- 六阶段 -->
        <template v-for="s in stages" :key="s.id">
          <section class="nav-stage" :class="{ open: isStageOpen(s.id) }">
            <button
              class="stage-toggle"
              type="button"
              :aria-expanded="isStageOpen(s.id)"
              :aria-controls="`stage-${s.id}-nav`"
              @click="toggleStage(s.id)"
            >
              <span class="idx">{{ s.idx }}</span>
              <span class="stage-toggle-title">阶段 {{ s.idx }} · {{ s.title }}</span>
              <span v-if="s.status === 'planned'" class="planned-tag">待补</span>
              <span class="chevron" aria-hidden="true">▾</span>
            </button>

            <div
              v-show="isStageOpen(s.id)"
              :id="`stage-${s.id}-nav`"
              class="stage-links"
            >
              <!-- ready & 阶段总览 -->
              <router-link
                v-if="s.status === 'ready' && s.route"
                :to="{ name: s.route }"
                class="nav-link sub overview"
              >
                <span class="idx">{{ s.idx }}.0</span>
                <span>阶段总览</span>
              </router-link>

              <!-- ready & 阶段内小标题 -->
              <router-link
                v-for="c in shown(s)"
                :key="c.route"
                :to="{ name: c.route }"
                class="nav-link sub"
              >
                <span class="idx">{{ s.idx }}.{{ subIdx(s, c.route) }}</span>
                <span class="chapter-label">
                  <span class="tier-dot" :class="tierOf(c.route)" :title="TIER_META[tierOf(c.route)].desc">{{ isSprint(c.route) ? '★' : TIER_META[tierOf(c.route)].mark }}</span>
                  {{ c.label }}
                </span>
                <span v-if="progress.isMastered(c.route)" class="mark done" title="自测全对">✓</span>
                <span v-else-if="progress.isVisited(c.route)" class="mark" title="打开过这一章">已读</span>
              </router-link>

              <p v-if="hiddenCount(s)" class="more-hint">
                本档位外还有 {{ hiddenCount(s) }} 章, 切到「全部」可见
              </p>

              <!-- planned: 灰显, 不可点击 -->
              <div v-if="s.status === 'planned'" class="nav-link disabled sub">
                <span class="idx">·</span>
                <span>{{ s.title }}</span>
              </div>
            </div>
          </section>
        </template>

        <!-- 终章 -->
        <div class="section-label">终章</div>
        <router-link :to="{ name: 'compare' }" class="nav-link">
          <span class="idx">∎</span>
          <span>总览对照表</span>
        </router-link>
        <router-link :to="{ name: 'glossary' }" class="nav-link">
          <span class="idx">?</span>
          <span>术语速查</span>
        </router-link>

        <div class="section-label">关于</div>
        <div class="footer">
          <p>六个目录已接入 Web 教程, 每章都对照原始代码阅读。</p>
          <p>键盘 ← / → 翻章, 按当前档位走。</p>
          <p>进度只存在本机浏览器, 进度条旁的「重置」可以清空。</p>
        </div>
      </nav>
    </aside>
    <!-- 拖动调目录宽度; 双击恢复默认, 聚焦后左右方向键也能调 -->
    <div class="side-resize" role="separator" aria-orientation="vertical" aria-label="拖动调整目录宽度"
         :aria-valuenow="sideW" :aria-valuemin="SIDE_MIN" :aria-valuemax="SIDE_MAX" tabindex="0"
         @pointerdown.prevent="startResize" @dblclick="sideW = SIDE_DEFAULT"
         @keydown.left.prevent="sideW -= 16" @keydown.right.prevent="sideW += 16" />

    <main id="main" ref="mainEl" class="main" tabindex="-1">
      <router-view v-slot="{ Component, route }">
        <div v-if="route.meta.chapter" class="breadcrumb">
          {{ route.meta.chapter }} · {{ route.meta.title }}
          <span class="crumb-tier" :class="tierOf(route.name)">
            {{ isSprint(route.name) ? '★ 冲刺' : TIER_META[tierOf(route.name)].label }}
          </span>
          <span class="crumb-desc">{{ isSprint(route.name) ? '只有一天也要读的 12 章之一' : TIER_META[tierOf(route.name)].desc }}</span>
        </div>
        <transition name="page" mode="out-in">
          <component :is="Component" :key="route.fullPath" />
        </transition>
      </router-view>
    </main>
  </div>
</template>

<script setup>
import { computed, ref, onMounted, onBeforeUnmount, watch } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { learningPath, stages } from '@/data/models.js'
import { LEVELS, TIER_META, inLevel, isSprint, tierOf } from '@/data/tiers.js'
import { repoUrl } from '@/utils/repo.js'
import { useProgress } from '@/composables/useProgress.js'
import { useChapterNav } from '@/composables/useChapterNav.js'

const router = useRouter()
const route = useRoute()

// ── 学习进度 + 窄屏抽屉 ─────────────────────────────────────────────
const progress = useProgress()
const navOpen = ref(false)
const burger = ref(null)
const mainEl = ref(null)
const skipToMain = () => mainEl.value?.focus()
// 侧栏是不是抽屉形态, 和 main.css 里的 900px 断点保持一致
const narrowQuery = typeof window !== 'undefined' && window.matchMedia ? window.matchMedia('(max-width: 900px)') : null
const narrow = ref(!!narrowQuery?.matches)
const onNarrow = (e) => { narrow.value = e.matches }
const level = computed(() => progress.state.level)
const levelNow = computed(() => LEVELS.find((l) => l.id === level.value) || LEVELS[LEVELS.length - 1])
// 章节数的口径: learningPath 去掉序章 (home), 共 86 章
const chapters = learningPath.filter((p) => p.route !== 'home')
const levelCount = (id) => chapters.filter((c) => inLevel(c.route, id)).length
// 当前所在章即使不在档位里也要显示, 否则筛选会让人丢失位置
const shown = (s) => (s.chapters || []).filter((c) => inLevel(c.route, level.value) || c.route === route.name)
const hiddenCount = (s) => (s.chapters || []).length - shown(s).length

// ── 目录宽度: 可拖, 存本机 ──────────────────────────────────────────
const SIDE_MIN = 200, SIDE_MAX = 480, SIDE_DEFAULT = 260, SIDE_KEY = 'llm-side-w'
const clampW = (w) => Math.min(SIDE_MAX, Math.max(SIDE_MIN, Math.round(w)))
const sideW = ref(clampW(Number((() => { try { return localStorage.getItem(SIDE_KEY) } catch (_) { return null } })()) || SIDE_DEFAULT))
watch(sideW, (w) => {
  if (clampW(w) !== w) { sideW.value = clampW(w); return }
  try { localStorage.setItem(SIDE_KEY, String(w)) } catch (_) { /* 隐私模式下不记 */ }
})
const startResize = () => {
  const move = (e) => { sideW.value = clampW(e.clientX) }
  const up = () => {
    removeEventListener('pointermove', move); removeEventListener('pointerup', up)
    document.body.classList.remove('side-resizing')
  }
  addEventListener('pointermove', move); addEventListener('pointerup', up)
  document.body.classList.add('side-resizing')
}

const readCount = computed(() => chapters.filter((p) => progress.isVisited(p.route)).length)
watch(() => route.name, (name) => { progress.visit(name); navOpen.value = false }, { immediate: true })
const resetProgress = () => {
  if (!window.confirm('清空已读记录和自测成绩? 阅读档位和主题不受影响。清空后找不回来。')) return
  progress.reset()
  progress.visit(route.name)      // 眼前这一章正开着, 仍算打开过
}

// ── 键盘: Esc 关抽屉, ← / → 翻章 ─────────────────────────────────────
const { prev, next } = useChapterNav()
const onKey = (e) => {
  if (e.key === 'Escape' && navOpen.value) {
    navOpen.value = false
    burger.value?.focus()         // 侧栏随即变 inert, 焦点要先挪出来
    return
  }
  if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return
  // Shift+→ 是扩选文字; 按住不放会一口气翻过一串章并全记成已读
  if (e.altKey || e.ctrlKey || e.metaKey || e.shiftKey || e.repeat || e.defaultPrevented) return
  // 只在焦点没落在任何控件上时翻章 (页面本身、正文容器、普通链接)。
  // 滑杆、按钮、<summary>、SVG 里带 tabindex 的节点都有自己的方向键用法, 不抢。
  const t = e.target
  const free = t === document.body || t === document.documentElement || t === mainEl.value || t.tagName === 'A'
  if (!free) return
  const to = e.key === 'ArrowRight' ? next.value : prev.value
  if (to) router.push({ name: to.name })
}
onMounted(() => {
  window.addEventListener('keydown', onKey)
  narrowQuery?.addEventListener?.('change', onNarrow)
})
onBeforeUnmount(() => {
  window.removeEventListener('keydown', onKey)
  narrowQuery?.removeEventListener?.('change', onNarrow)
})

const subIdx = (stage, route) =>
  (stage.chapters || []).findIndex(c => c.route === route) + 1

const openStages = ref(Object.fromEntries(stages.map(s => [s.id, true])))
const isStageOpen = (id) => openStages.value[id] !== false
const toggleStage = (id) => {
  openStages.value = { ...openStages.value, [id]: !isStageOpen(id) }
}

watch(
  () => route.meta.stage,
  (stage) => {
    if (stage) openStages.value = { ...openStages.value, [stage]: true }
  },
  { immediate: true },
)

// ── 主题切换 ──────────────────────────────────────────────────────
const THEME_KEY = 'llm-theme'
const initialTheme = (() => {
  try {
    const stored = localStorage.getItem(THEME_KEY)
    if (stored === 'light' || stored === 'dark') return stored
  } catch (_) { /* localStorage 不可用时降级 */ }
  if (typeof window !== 'undefined' && window.matchMedia) {
    return window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark'
  }
  return 'dark'
})()
const theme = ref(initialTheme)
const applyTheme = (t) => { document.documentElement.dataset.theme = t }
applyTheme(theme.value)
onMounted(() => applyTheme(theme.value))
watch(theme, (t) => {
  applyTheme(t)
  try { localStorage.setItem(THEME_KEY, t) } catch (_) { /* ignore */ }
})
const toggleTheme = () => { theme.value = theme.value === 'dark' ? 'light' : 'dark' }
</script>

<style scoped>
.brand-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}
.theme-toggle {
  width: 40px;
  height: 40px;
  padding: 0;
  display: grid;
  place-items: center;
  font-size: 14px;
  line-height: 1;
  background: var(--bg-card);
  border: 1px solid var(--border);
  color: var(--text-muted);
  border-radius: var(--radius-sm);
}
.theme-toggle:hover {
  color: var(--accent);
  border-color: var(--accent);
}
.skip-link {
  position: fixed; z-index: 100; left: 8px; top: 8px; padding: 8px 14px;
  background: var(--accent); border-color: var(--accent); color: #fff; font-size: 13px;
  transform: translateY(-200%);
}
.skip-link:focus, .skip-link:active { transform: none; }
.main:focus { outline: none; }
.level-bar { display: flex; gap: 4px; padding: 0 24px 6px; }
.level-btn { flex: 1; min-height: 28px; padding: 3px 0; font-size: 11px; border-radius: var(--radius-sm); }
.level-legend { padding: 0 24px 10px; font-size: 10.5px; line-height: 1.7; color: var(--text-muted); }
.level-legend .tier-dot { margin: 0 3px 0 6px; }
.level-legend .tier-dot:first-child { margin-left: 0; }
.level-legend .mark { margin: 0 2px 0 8px; }
.level-legend .mark:first-child { margin-left: 0; }
.reset-progress { min-height: 0; padding: 1px 6px; font-size: 10px; color: var(--text-muted); background: transparent; }
.nav-link.fast { color: var(--accent); font-size: 12.5px; }
.chapter-label { min-width: 0; }
.tier-dot { font-size: 9px; margin-right: 5px; vertical-align: 1px; }
.tier-dot.core { color: var(--accent); }
.tier-dot.ext { color: var(--text-dim); }
.more-hint { padding: 4px 24px 6px 38px; font-size: 11px; color: var(--text-dim); line-height: 1.6; }
.crumb-tier { margin-left: 8px; padding: 1px 6px; border-radius: 3px; font-size: 10px; border: 1px solid var(--border-strong); }
.crumb-tier.core { color: var(--accent); border-color: var(--accent); }
.crumb-tier.ext { color: var(--text-dim); }
.crumb-desc { margin-left: 6px; font-size: 11px; color: var(--text-dim); }
.progress { display: flex; align-items: center; gap: 8px; margin-top: 10px; font-size: 10px; color: var(--text-dim); }
.progress-bar { flex: 1; height: 4px; border-radius: 2px; background: var(--border); overflow: hidden; }
.progress-bar span { display: block; height: 100%; background: var(--left); transition: width 0.3s; }
.mark { margin-left: auto; flex: 0 0 auto; font-size: 10px; color: var(--text-dim); white-space: nowrap; }
.mark.done { font-size: 11px; color: var(--left); }
.repo-link {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  margin-top: 10px;
  font-size: 11px;
  color: var(--text-muted);
  border-bottom: 1px dashed var(--border-strong);
  padding-bottom: 1px;
  width: max-content;
}
.repo-link:hover { color: var(--accent); border-color: var(--accent); text-decoration: none; }
.gh-icon { font-size: 10px; color: var(--accent); }

.footer {
  padding: 8px 24px 24px;
  font-size: 11px;
  color: var(--text-dim);
  line-height: 1.7;
}
.footer p { margin-top: 4px; }

.planned-tag {
  margin-left: 6px;
  padding: 1px 6px;
  border-radius: 3px;
  background: var(--bg-card);
  border: 1px dashed var(--border);
  color: var(--text-dim);
  font-size: 9px;
  text-transform: uppercase;
  letter-spacing: 0.6px;
}

.nav-stage {
  margin: 2px 0 4px;
}
.stage-toggle {
  width: 100%;
  min-height: 40px;
  padding: 8px 24px;
  display: flex;
  align-items: center;
  gap: 12px;
  background: transparent;
  border: 0;
  border-left: 2px solid transparent;
  border-radius: 0;
  color: var(--text);
  text-align: left;
}
.stage-toggle:hover {
  background: var(--bg-card);
  border-left-color: var(--border-strong);
}
.stage-toggle .idx {
  flex: 0 0 auto;
  width: 24px;
  height: 24px;
  border-radius: 5px;
  background: var(--bg-card);
  color: var(--text-muted);
  display: grid;
  place-items: center;
  font-size: 11px;
  font-family: "SF Mono", Menlo, monospace;
}
.stage-toggle-title {
  min-width: 0;
  flex: 1;
  font-size: 13px;
  font-weight: 600;
  line-height: 1.35;
}
.chevron {
  color: var(--text-dim);
  font-size: 12px;
  transition: transform 0.15s;
}
.nav-stage.open .chevron { transform: rotate(180deg); }
.stage-links {
  padding: 1px 0 5px;
}
.nav-link.sub { padding-left: 38px; font-size: 12.5px; }
.nav-link.sub .idx { width: 34px; }
.nav-link.overview { color: var(--text); }
.nav-link.disabled {
  display: flex; align-items: center; gap: 12px;
  padding: 10px 24px;
  font-size: 13px;
  color: var(--text-dim);
  border-left: 2px solid transparent;
  cursor: not-allowed;
  opacity: 0.55;
}
.nav-link.disabled .idx { background: transparent; }

.page-enter-active, .page-leave-active {
  transition: opacity 0.18s, transform 0.18s;
}
.page-enter-from { opacity: 0; transform: translateY(4px); }
.page-leave-to   { opacity: 0; transform: translateY(-4px); }
</style>
