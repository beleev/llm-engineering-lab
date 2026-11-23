<!--
  速成路线: 86 章太多了, 这一页回答"我到底该读哪些、按什么顺序、要多久"。
  三个档位 (冲刺 / 主干 / 全部) 共用同一条 learningPath 顺序, 只是筛掉的多少不同。
  档位读写的是全局的 progress.state.level, 和侧栏的档位条是同一个值: 在这里选了, 侧栏和翻章跟着变。
-->
<template>
  <div>
    <h1 class="page-title">速成路线</h1>
    <div class="page-subtitle lead-group">
      <p>全书六个阶段共计 {{ all.length }} 章, 无需一次性逐章通读; 建议先根据可用时间选择阅读档位并按序推进。</p>
      <p>
        完成主干档位即可贯通「大模型从底层训练、对齐微调、高吞吐部署到智能体构建」的完整工程链路, 其余扩展章节可在后续按专题选读。
      </p>
    </div>

    <div class="btn-group levels" role="group" aria-label="阅读档位">
      <button
        v-for="l in LEVELS" :key="l.id" type="button" :aria-pressed="level === l.id"
        :class="['lv', `lv-${l.id}`, { active: level === l.id }]" @click="progress.setLevel(l.id)"
      >{{ l.label }} · {{ countOf(l.id) }} 章, {{ l.hint }}</button>
    </div>
    <p class="assume legend">此处所选档位与左侧导航栏实时同步, 页尾「下一章」跳转及键盘方向键翻页均遵循当前档位范围。</p>
    <ul class="assume legend pts">
      <li><b>章节前缀标记:</b> <span class="tier core">★</span> 冲刺必读 <span class="tier core">●</span> 主干核心 <span class="tier ext">○</span> 专题扩展。</li>
      <li><b>行尾进度状态:</b> <span class="state">✓</span> 表示章末自测全部答对, <span class="state">已读</span> 表示已访问过该章节。</li>
    </ul>

    <div class="card summary">
      <div class="sum-item"><span class="v mono">{{ list.length }}</span><span class="k">章</span></div>
      <div class="sum-item"><span class="v mono">{{ totalTime(list.map((c) => c.route)) }}</span><span class="k">预计用时 (含实验操作)</span></div>
      <div class="sum-item"><span class="v mono">{{ labCount }}</span><span class="k">个实验台</span></div>
      <div class="sum-item"><span class="v mono" :class="{ good: readCount === list.length }">{{ readCount }}</span><span class="k">已读</span></div>
      <router-link v-if="nextUp" :to="{ name: nextUp.route }" class="btn active">
        {{ readCount ? '继续阅读' : '从首章开始' }}: {{ nextUp.label }} →
      </router-link>
    </div>
    <p class="assume">阅读耗时按每分钟 300 字估算, 每个交互实验台另计 4 分钟, 仅供安排学习计划时参考。</p>

    <section v-for="g in groups" :key="g.stage" class="section">
      <h2>{{ g.title }} <span class="stage-code mono">{{ g.code }}</span></h2>
      <template v-for="topic in g.topics" :key="topic.id">
      <h3 v-if="topic.title" class="path-topic">{{ topic.title }}</h3>
      <ol class="path">
        <li v-for="c in topic.items" :key="c.route" :class="{ done: progress.isVisited(c.route) }">
          <router-link :to="{ name: c.route }" class="row">
            <span class="tier" :class="tierOf(c.route)" :title="TIER_META[tierOf(c.route)].desc">
              {{ isSprint(c.route) ? '★' : TIER_META[tierOf(c.route)].mark }}
            </span>
            <span class="label">{{ c.label }}</span>
            <span class="hint">{{ c.hint || '' }}</span>
            <span class="meta mono">
              <span v-if="labsOf(c.route)" class="lab-n" :title="`${labsOf(c.route)} 个实验台`">⚙{{ labsOf(c.route) }}</span>
              {{ minutesOf(c.route) }}′
            </span>
            <span class="state">{{ progress.isMastered(c.route) ? '✓' : progress.isVisited(c.route) ? '已读' : '' }}</span>
          </router-link>
        </li>
      </ol>
      </template>
    </section>

    <section class="section">
      <h2>三档阅读范围的划分依据</h2>
      <div class="grid grid-3">
        <div v-for="l in LEVELS" :key="l.id" :class="['card', 'lv', `lv-${l.id}`]">
          <h3>{{ l.label }}</h3>
          <p class="desc">{{ WHY[l.id] }}</p>
        </div>
      </div>
      <p class="assume">
        扩展章节聚焦架构变体与工程深水区, 涵盖 2026 年主流生产系统广泛采用的关键技术; 建议先建立主干框架, 再结合具体问题按需深入研读。
      </p>
    </section>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { learningPath, stages } from '@/data/models.js'
import { LEVELS, TIER_META, inLevel, isSprint, tierOf } from '@/data/tiers.js'
import { labsOf, minutesOf, totalTime } from '@/utils/readtime.js'
import { useProgress } from '@/composables/useProgress.js'

const WHY = {
  all: '覆盖六个阶段的全部章节。每章均与仓库中的可运行源码一一对应, 适合系统性精读与案头查阅。',
  core: '全书核心主线。从纯 NumPy 手写反向传播出发, 循序渐进贯通至具备工具调用能力的 Agent 系统, 逻辑链条完整无跳步。',
  sprint: '单日速览精选的 12 章。从每个阶段提炼出承上启下的最小核心集合, 帮助读者在最短时间内建立全栈全局视图。',
}

const progress = useProgress()
const level = computed(() => progress.state.level)

const all = computed(() => learningPath.filter((p) => p.route !== 'home'))
const list = computed(() => all.value.filter((c) => inLevel(c.route, level.value)))

const countOf = (id) => all.value.filter((c) => inLevel(c.route, id)).length
const labCount = computed(() => list.value.reduce((s, c) => s + labsOf(c.route), 0))
const readCount = computed(() => list.value.filter((c) => progress.isVisited(c.route)).length)
const nextUp = computed(() => list.value.find((c) => !progress.isVisited(c.route)) || list.value[0])

// 按阶段分组, 组内保持 learningPath 的顺序
const stageGroups = computed(() =>
  stages
    .map((s) => ({
      stage: s.id,
      title: `阶段 ${s.idx} · ${s.title}`,
      code: s.code,
      items: list.value.filter((c) => c.route === s.route || (s.chapters || []).some((ch) => ch.route === c.route))
        .map((c) => ({ ...c, hint: (s.chapters || []).find((ch) => ch.route === c.route)?.hint })),
    }))
    .filter((g) => g.items.length),
)
const groups = computed(() => stageGroups.value.map((g) => {
  const stage = stages.find((s) => s.id === g.stage)
  const overview = g.items.filter((c) => !c.groupId)
  return { ...g, topics: [
    ...(overview.length ? [{ id: 'overview', title: '', items: overview }] : []),
    ...(stage?.groups || []).map((topic) => ({
      id: topic.id, title: `${topic.number} · ${topic.title}`,
      items: g.items.filter((c) => c.groupId === topic.id),
    })).filter((topic) => topic.items.length),
  ] }
}))
</script>

<style scoped>
.path-topic { font-size: 14px; margin: 20px 0 8px; color: var(--text-muted); }
.levels { margin-bottom: 16px; }
.summary { display: flex; flex-wrap: wrap; align-items: center; gap: 24px; }
.sum-item { display: flex; flex-direction: column; }
.sum-item .v { font-size: 20px; color: var(--text); }
.sum-item .v.good { color: var(--left); }
.sum-item .k { font-size: 11px; color: var(--text-dim); }
.assume { font-size: 12px; color: var(--text-dim); margin-top: 8px; line-height: 1.7; }
.stage-code { font-size: 11px; font-weight: 400; color: var(--text-dim); }

.path { list-style: none; counter-reset: step; }
.path .row {
  display: grid;
  grid-template-columns: 18px minmax(120px, 1.1fr) 2fr 62px 28px;
  gap: 10px;
  align-items: baseline;
  padding: 7px 10px;
  border-bottom: 1px solid var(--border);
  color: var(--text-muted);
}
.path .row:hover { background: var(--bg-card); text-decoration: none; }
.path li.done .label { color: var(--text-dim); }
.tier { font-size: 10px; text-align: center; }
.tier.core { color: var(--accent); }
.tier.ext { color: var(--text-dim); }
.label { color: var(--text); font-size: 13px; }
.hint { font-size: 12px; color: var(--text-dim); }
.meta { font-size: 11px; color: var(--text-dim); text-align: right; font-variant-numeric: tabular-nums; }
.lab-n { color: var(--accent); margin-right: 4px; }
.state { font-size: 11px; color: var(--left); white-space: nowrap; }
.legend .tier, .legend .state { display: inline; margin: 0 2px 0 4px; }  /* 窄屏隐藏的是行尾那一列, 图例里的照常显示 */

@media (max-width: 900px) {
  .path .row { grid-template-columns: 18px 1fr 52px; }
  .hint, .state { display: none; }
}
</style>
