<!--
  数据驱动章节的统一版式。顺序固定成一条学习线, 页首的「本章顺序」导航与之对应:
    本章一句话 / 关键问题 → 1 它从哪来 (演进链 + 它接在哪) → 2 要点 (单栏, 逐条编号)
    → 3 动手验证 (实验台, 读完要点再操作) → 4 去代码里看 (对照表 + 骨架) → 补充段 → 5 章末自测
  没有实验台的章, 编号自动顺延。
-->
<template>
  <div v-if="page">
    <h1 class="page-title">{{ page.title }}</h1>
    <Prose class="page-subtitle" :text="page.subtitle" />

    <ChapterIntro
      :tldr="page.tldr"
      :question="page.question"
      :code="page.code"
    />

    <nav v-if="page.related?.length" class="card related-topics" aria-label="前置知识与相关实验">
      <p class="related-title">前置知识与相关实验</p>
      <ul>
        <li v-for="link in page.related" :key="link.route + link.label">
          <RouterLink :to="{ name: link.route }">{{ link.label }} →</RouterLink>
          <p>{{ link.reason }}</p>
        </li>
      </ul>
    </nav>

    <!-- 本章顺序: 让读者先知道这一页分几步、每步干什么 -->
    <nav class="page-outline" aria-label="本章顺序">
      <span class="outline-label">本章顺序</span>
      <RouterLink v-for="(s, i) in outline" :key="s.id" :to="{ hash: '#' + s.id }" class="outline-item">
        <span class="outline-num">{{ i + 1 }}</span>{{ s.label }}
      </RouterLink>
    </nav>

    <!-- 1 它从哪来: 根问题 → 每一步补上一步的窟窿 → 本章实现的那一步; 再看它替换了什么、后来被用在哪 -->
    <section v-if="evolution || page.links?.length" :id="ids.origin" class="section" tabindex="-1">
      <h2><span class="sec-num">{{ num('origin') }}</span>它从哪来</h2>
      <p class="lead">
        顺着箭头读一遍: 最早的做法卡在哪, 后来每一代补了什么, 本章实现的是其中哪一步。
      </p>
      <EvolutionChain
        v-if="evolution"
        :title="evolution.title"
        :subtitle="evolution.subtitle"
        :steps="evolution.steps"
      />
      <template v-if="page.links?.length">
        <h3 class="sub-h">它接在哪</h3>
        <p class="lead">左边是它替换掉的东西, 右边是它后来被用在哪。</p>
        <div class="card link-card">
          <div v-for="l in page.links" :key="`${l.from}-${l.to}`" class="link-row">
            <component
              :is="topicRef(l.from).linked ? RepoLink : 'span'"
              v-bind="topicRef(l.from).linked ? { path: topicRef(l.from).path, label: l.from, tiny: true } : {}"
              class="mono endpoint"
            ><Tex :text="l.from" /></component>
            <span class="arrow">→</span>
            <component
              :is="topicRef(l.to).linked ? RepoLink : 'span'"
              v-bind="topicRef(l.to).linked ? { path: topicRef(l.to).path, label: l.to, tiny: true } : {}"
              class="mono endpoint"
            ><Tex :text="l.to" /></component>
            <Prose class="body" :text="l.body" />
          </div>
        </div>
      </template>
    </section>

    <!-- 2 要点: 单栏逐条, 每条先结论再机制再数字再边界 -->
    <section :id="ids.points" class="section" tabindex="-1">
      <h2><span class="sec-num">{{ num('points') }}</span>要点</h2>
      <p class="lead">
        本章共 {{ page.points.length }} 条要点, 按顺序读。每条先说结论, 再说它是怎么发生的。标了「重点」的那条最该掌握。
      </p>
      <ol class="points-list">
        <li v-for="(p, i) in page.points" :key="p.title" class="card point-card" :class="{ key: p.key }">
          <h3><span class="point-num">{{ i + 1 }}</span><Tex :text="p.title" /> <span v-if="p.key" class="key-tag">重点</span></h3>
          <Prose class="desc" :text="p.body" lead />
        </li>
      </ol>
    </section>

    <!-- 3 动手验证: 放在要点之后, 读者已经知道名词是什么 -->
    <section v-if="hasLabs" :id="ids.labs" class="section" tabindex="-1">
      <h2><span class="sec-num">{{ num('labs') }}</span>动手验证</h2>
      <p class="lead">
        把要点里的结论亲手拖出来。每个实验台开头都写了它在模拟什么、图上是什么、可以做什么、应该看到什么; 「试一试」先猜再看答案。
      </p>
      <LabMount :names="primaryLabs" />
    </section>

    <!-- 4 去代码里看: 对照表 + 只留控制流的骨架 + 真源码 -->
    <section :id="ids.code" class="section" tabindex="-1">
      <h2><span class="sec-num">{{ num('code') }}</span>去代码里看</h2>
      <p class="lead">
        打开源码之前先看这张表: 每一行告诉你一个概念落在代码的哪一行、那一行在说什么。只盯这几行, 其余的先放过。
      </p>
      <div class="card" style="padding: 0; overflow-x: auto;">
        <table class="topic-table">
          <thead>
            <tr>
              <th>概念</th>
              <th>代码位置 / 表达式</th>
              <th>这一行在说什么</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="r in page.sourceRows" :key="r.concept">
              <td class="concept"><Tex :text="r.concept" /></td>
              <td class="mono small">
                <RepoLink
                  v-if="topicRef(r.code).linked"
                  :path="topicRef(r.code).path"
                  :label="r.code"
                  tiny
                />
                <span v-else>{{ r.code }}</span>
              </td>
              <td><Prose :text="r.takeaway" /></td>
            </tr>
          </tbody>
        </table>
      </div>

      <h3 class="sub-h">骨架长这样</h3>
      <p class="lead">
        只留控制流, 细节都砍了。想看完整实现就展开下面的真源码, 它直接从仓库文件里读出来, 不会和代码脱节。
      </p>
      <div class="card">
        <h3>{{ page.snippetTitle }} <span v-if="page.run" class="tag">可运行</span></h3>
        <CodeBlock :code="page.snippet" />
        <!-- page.source: 直接从仓库 Python 文件取的真源码, 不会和代码漂移 -->
        <SourceSnippet v-for="s in sources" :key="s" :src="s" />
        <p v-if="page.run" class="hint">
          跑一下: <code class="inline">{{ page.run }}</code>
        </p>
      </div>
    </section>

    <TopicSupplement v-for="section in page.sections || []" :key="section.id" :section="section" />

    <QuizCard :num="num('quiz')" />

    <!-- 上一章 / 下一章由组件自己从 learningPath 推, 这里不用传 -->
    <ChapterNav />
  </div>

  <!-- 路由在, 但这一章还没有正文: 给读者看的提示和出口, 与地址写错时同一页 -->
  <NotFound v-else />
</template>

<script setup>
import { computed } from 'vue'
import TopicSupplement from '@/components/TopicSupplement.vue'
import { useRoute } from 'vue-router'
import ChapterIntro from '@/components/ChapterIntro.vue'
import EvolutionChain from '@/components/EvolutionChain.vue'
import Prose from '@/components/Prose.vue'
import Tex from '@/components/Tex.vue'
import ChapterNav from '@/components/ChapterNav.vue'
import RepoLink from '@/components/RepoLink.vue'
import LabMount from '@/components/LabMount.vue'
import CodeBlock from '@/components/CodeBlock.vue'
import SourceSnippet from '@/components/SourceSnippet.vue'
import QuizCard from '@/components/QuizCard.vue'
import NotFound from '@/views/NotFound.vue'
import { labMap } from '@/data/labMap.js'
import { stageBy, topicPages } from '@/data/models.js'
import { looksLikeRepoRef, normalizeRepoRef, splitRefs } from '@/utils/repo.js'

const route = useRoute()
const page = computed(() => topicPages[route.name])
// 数据里不写颜色: 按步序取色, 和 Attention / Position 两章手写的演进链同一套顺序
const EVO_COLORS = ['var(--text-muted)', 'var(--accent)', 'var(--left)', 'var(--right)', 'var(--eye)']
const evolution = computed(() => {
  const e = page.value?.evolution
  if (!e) return null
  const steps = e.steps.map((s, i) => ({ color: EVO_COLORS[i % EVO_COLORS.length], ...s }))
  return { ...e, title: e.title ? `演进逻辑链 · ${e.title}` : '', steps }
})
const primaryLabs = computed(() => {
  const extra = new Set((page.value?.sections || []).flatMap(s => s.labNames || []))
  return [...new Set([...(page.value?.widgets || []), ...(labMap[route.name] || [])])].filter(n => !extra.has(n))
})
const hasLabs = computed(() => primaryLabs.value.length > 0)
const sources = computed(() => [].concat(page.value?.source || []))

// 段落锚点 (与 chapterAliases 里的补充段锚点不重名), 以及页首「本章顺序」
const ids = { origin: 'sec-origin', points: 'sec-points', labs: 'sec-labs', code: 'sec-code', quiz: 'sec-quiz' }
const outline = computed(() => [
  (evolution.value || page.value?.links?.length) && { id: ids.origin, key: 'origin', label: '它从哪来' },
  { id: ids.points, key: 'points', label: '要点' },
  hasLabs.value && { id: ids.labs, key: 'labs', label: '动手验证' },
  { id: ids.code, key: 'code', label: '去代码里看' },
  { id: ids.quiz, key: 'quiz', label: '章末自测' },
].filter(Boolean))
const num = (key) => outline.value.findIndex(s => s.key === key) + 1

const stageCode = computed(() => stageBy[route.meta.stage]?.code || '')
const contextRefs = computed(() =>
  splitRefs(page.value?.code || '', 'auto').map(r => normalizeRepoRef(r.label || r.path).path)
)

const topicRef = (ref) => {
  const raw = String(ref || '').trim()
  if (!looksLikeRepoRef(raw)) return { linked: false, path: raw }

  const parsed = normalizeRepoRef(raw)
  if (parsed.path.startsWith('llm_')) return { linked: true, path: raw }

  const path = parsed.path
  const match = contextRefs.value.find(p => p.endsWith(path) || p.endsWith(`/${path}`))
  if (match) return { linked: true, path: match }

  if (stageCode.value === 'llm_basic/' && /^[\w.-]+\.(py|txt|bin|npz|md)(?::|$)/.test(raw)) {
    return { linked: true, path: `${stageCode.value}${raw}` }
  }

  if (stageCode.value && path.includes('/')) {
    return { linked: true, path: `${stageCode.value}${raw}` }
  }

  return { linked: false, path: raw }
}
</script>

<style scoped>
.hint code.inline { white-space: normal; overflow-wrap: anywhere; }
.related-topics { margin: 20px 0; }
.related-title { font-weight: 600; margin-bottom: 10px; }
.related-topics ul { padding-left: 20px; }
.related-topics li { margin: 10px 0; overflow-wrap: anywhere; }
.related-topics li p { color: var(--text-muted); font-size: 13px; margin-top: 4px; }
.section:focus { outline: none; }
.section { scroll-margin-top: 24px; }

/* 一个大节里的小标题 (它接在哪 / 骨架长这样) */
.sub-h {
  font-size: 15px;
  font-weight: 600;
  margin: 28px 0 6px;
}
.sub-h + .lead { margin-bottom: 12px; }

.link-card {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.link-row {
  display: grid;
  grid-template-columns: minmax(120px, 0.9fr) 24px minmax(120px, 0.9fr) 2fr;
  gap: 10px;
  align-items: baseline;
  padding: 10px 12px;
  background: var(--bg-elev);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  font-size: 13.5px;
}
.link-row .endpoint {
  color: var(--text);
  font-size: 12px;
}
.link-row .arrow {
  color: var(--accent);
  text-align: center;
}
.link-row .body {
  color: var(--text);
  line-height: 1.65;
}

table.topic-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13.5px;
}
table.topic-table th {
  text-align: left;
  padding: 12px 14px;
  background: var(--bg-elev);
  color: var(--text-muted);
  font-size: 11px;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.7px;
  border-bottom: 1px solid var(--border-strong);
}
table.topic-table td {
  padding: 12px 14px;
  border-bottom: 1px solid var(--border);
  vertical-align: top;
  line-height: 1.65;
}
table.topic-table .concept {
  color: var(--text);
  font-weight: 600;
  white-space: nowrap;
}
table.topic-table .small {
  color: var(--text-muted);
  font-size: 12px;
}
.hint {
  margin-top: 10px;
  font-size: 12px;
  color: var(--text-dim);
  line-height: 1.6;
}

@media (max-width: 960px) {
  .link-row {
    grid-template-columns: 1fr;
    gap: 4px;
  }
  .link-row .arrow {
    text-align: left;
  }
}
</style>
