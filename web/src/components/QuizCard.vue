<!--
  章末自测: 每题选一次就锁定, 立刻给对错和一句解释。
  全部答完记一次成绩, 存的是最好成绩 (useProgress.setQuiz); 最好成绩全对时侧栏打 ✓。
  对错不只靠颜色: 选项前有「✓ 正确项」「✗ 你的选择」文字标记。
-->
<template>
  <section v-if="items.length" class="section quiz">
    <h2>
      章末自测
      <span class="score mono">
        <template v-if="best">最好成绩 {{ best.correct }} / {{ best.total }} · </template>这一遍 {{ correct }} / {{ items.length }}
      </span>
    </h2>
    <p class="lead">合上代码答一遍。每题只能选一次, 选完就给答案和解释。答错不要紧: 看解释, 再回实验台拖一拖验证。</p>
    <p v-if="mastered" class="mastered">✓ 已掌握: 这一章的自测全对过, 侧栏里这一章已打 ✓。重做不会把它弄丢。</p>
    <div v-for="(it, i) in items" :key="i" class="card q">
      <p class="q-text"><span class="mono q-idx">Q{{ i + 1 }}</span><Tex :text="it.q" /></p>
      <div class="opts" role="radiogroup" :aria-label="`第 ${i + 1} 题`">
        <button
          v-for="(o, k) in it.options" :key="k" type="button" role="radio"
          :aria-checked="picked[i] === k"
          :class="['opt', stateOf(i, k)]" :disabled="picked[i] != null"
          @click="pick(i, k)"
        ><b v-if="tagOf(i, k)" class="opt-tag">{{ tagOf(i, k) }}</b><Tex :text="o" /></button>
      </div>
      <!-- 容器常驻, 内容后插: 读屏器只播报已存在的 live 区域里的变化 -->
      <div aria-live="polite">
        <div v-if="picked[i] != null" class="why" :class="picked[i] === it.answer ? 'good' : 'bad'">
          <b class="verdict">{{ picked[i] === it.answer ? '✓ 正确' : '✗ 选错了, 正确项已标出' }}</b>
          <Prose :text="it.why" />
        </div>
      </div>
    </div>
    <button v-if="done" type="button" @click="retry">重做一遍</button>
  </section>
</template>

<script setup>
import Tex from '@/components/Tex.vue'
import { computed, ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import { quizBank } from '@/data/quiz/index.js'
import Prose from '@/components/Prose.vue'
import { useProgress } from '@/composables/useProgress.js'

const route = useRoute()
const progress = useProgress()
const items = computed(() => quizBank[route.name] || [])
const picked = ref([])
watch(items, () => (picked.value = []), { immediate: true })

const correct = computed(() => items.value.filter((it, i) => picked.value[i] === it.answer).length)
const done = computed(() => items.value.length > 0 && items.value.every((_, i) => picked.value[i] != null))
const pick = (i, k) => { picked.value[i] = k }
const retry = () => { picked.value = [] }
const stateOf = (i, k) => {
  if (picked.value[i] == null) return ''
  if (k === items.value[i].answer) return 'right'
  return picked.value[i] === k ? 'wrong' : ''
}
// 选项前的文字标记, 和边框颜色说的是同一件事
const tagOf = (i, k) => {
  const s = stateOf(i, k)
  if (s === 'right') return picked.value[i] === k ? '✓ 正确项 · 你的选择' : '✓ 正确项'
  return s === 'wrong' ? '✗ 你的选择' : ''
}
const best = computed(() => progress.quizOf(route.name))
const mastered = computed(() => progress.isMastered(route.name))
watch(done, (d) => { if (d) progress.setQuiz(route.name, correct.value, items.value.length) })
</script>

<style scoped>
.score { font-size: 12px; color: var(--accent); margin-left: auto; }
.q { margin-bottom: 12px; }
.q-text { font-size: 14px; margin-bottom: 10px; line-height: 1.6; }
.q-idx { color: var(--accent); margin-right: 8px; font-size: 12px; }
.opts { display: grid; gap: 6px; }
.opt { text-align: left; line-height: 1.5; padding: 8px 12px; }
.opt:disabled { opacity: 1; cursor: default; }
.opt.right { border-color: var(--left); background: color-mix(in srgb, var(--left) 14%, transparent); }
.opt.wrong { border-color: var(--danger); background: color-mix(in srgb, var(--danger) 12%, transparent); }
.opt-tag { display: block; font-size: 11px; font-weight: 600; margin-bottom: 2px; }
.opt.right .opt-tag { color: var(--left); }
.opt.wrong .opt-tag { color: var(--danger); }
.mastered { font-size: 13px; color: var(--left); margin: -8px 0 16px; }
.why { margin-top: 10px; font-size: 13px; line-height: 1.6; padding-left: 10px; border-left: 2px solid var(--border); color: var(--text-muted); }
.why.good { border-left-color: var(--left); }
.why.bad { border-left-color: var(--danger); }
.why .verdict { display: block; margin-bottom: 4px; color: var(--text); }
</style>
