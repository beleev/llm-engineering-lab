<!-- 术语速查表: 搜索 + 按阶段过滤。每个词条一句话 + 一个关键数字 + 跳到对应章节。 -->
<template>
  <div>
    <h1 class="page-title">术语速查</h1>
    <p class="page-subtitle">读到陌生缩写先来这里。每条只给一句话和一个值得记住的数字, 想深入就点进对应章节。</p>

    <div class="bar">
      <input v-model="q" type="search" class="search" placeholder="搜索: KV cache / ZeRO / GRPO / 投机解码 …" aria-label="搜索术语" />
      <div class="btn-group">
        <button type="button" :class="{ active: !stage }" @click="stage = ''">全部</button>
        <button v-for="s in stages" :key="s.id" type="button" :class="{ active: stage === s.id }" @click="stage = s.id">
          {{ s.idx }} · {{ s.title }}
        </button>
      </div>
    </div>

    <p class="count mono" aria-live="polite">{{ shown.length }} / {{ glossary.length }} 条</p>
    <p v-if="!shown.length" class="card empty">
      没有匹配的术语。换个关键词<template v-if="stage">, 或切回「全部」</template>。
      <button v-if="q || stage" type="button" @click="q = ''; stage = ''">清空筛选</button>
    </p>
    <div class="grid grid-2">
      <article v-for="g in shown" :key="g.id" class="card term">
        <h3>{{ g.term }} <span v-if="g.aka" class="aka"><Tex :text="g.aka" /></span></h3>
        <!-- 同名词条在不同阶段各有一条 (如 KV cache), 靠这个标签分清是哪个阶段的说法 -->
        <p v-if="stageBy[g.stage]" class="stage-tag">阶段 {{ stageBy[g.stage].idx }} · {{ stageBy[g.stage].title }}</p>
        <p class="desc"><Tex :text="g.oneliner" /></p>
        <p v-if="g.number" class="num" :class="{ mono: !g.number.includes('$') }"><Tex :text="g.number" /></p>
        <router-link v-if="g.route" :to="{ name: g.route }" class="go">去这一章 →</router-link>
      </article>
    </div>
  </div>
</template>

<script setup>
import Tex from '@/components/Tex.vue'
import { texPlain } from '@/utils/tex'
import { computed, ref } from 'vue'
import { glossary } from '@/data/glossary.js'
import { stageBy, stages } from '@/data/models.js'

const q = ref('')
const stage = ref('')
// 词条名会重复, 用在原数组里的序号当 key
const entries = glossary.map((g, id) => ({ ...g, id }))
const shown = computed(() => {
  const k = q.value.trim().toLowerCase()
  return entries
    .filter((g) => !stage.value || g.stage === stage.value)
    .filter((g) => !k || texPlain(`${g.term} ${g.aka || ''} ${g.oneliner}`).toLowerCase().includes(k))
    .sort((a, b) => a.term.localeCompare(b.term))
})
</script>

<style scoped>
.bar { display: flex; flex-direction: column; gap: 10px; margin-bottom: 12px; }
.search { width: 100%; max-width: 520px; min-height: 40px; padding: 8px 12px; font-size: 14px; color: var(--text); background: var(--bg-elev); border: 1px solid var(--border); border-radius: var(--radius-sm); }
.count { font-size: 11px; color: var(--text-dim); margin-bottom: 10px; }
.term h3 { margin-bottom: 6px; }
.stage-tag { font-size: 11px; color: var(--text-dim); margin: -2px 0 6px; }
.empty { display: flex; flex-wrap: wrap; align-items: center; gap: 12px; font-size: 13px; color: var(--text-muted); }
.aka { font-size: 11px; font-weight: 400; color: var(--text-dim); }
.num { margin-top: 8px; font-size: 13px; color: var(--accent); overflow-x: auto; }
.go { display: inline-block; margin-top: 8px; font-size: 12px; }
</style>
