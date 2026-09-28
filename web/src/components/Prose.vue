<!--
  文案渲染: 一个字符串里, 换行 = 分段, "- " 开头的连续行 = 要点列表。
  要点以 "短关键词: " 开头时, 关键词加粗, 方便扫读。
  数据里照旧写字符串, 只是多了换行; 没有换行的旧文案原样显示成一段。
  $…$ 是行内公式, 单独一行的 $$…$$ 是独立公式。一个公式不能跨 \n, 要多行就在一个 $$…$$ 里用 aligned。
-->
<template>
  <div class="prose">
    <template v-for="(b, i) in blocks" :key="i">
      <ul v-if="b.items">
        <li v-for="x in b.items" :key="x.body"><b v-if="x.label"><Tex :text="x.label" /></b><Tex :text="x.body" /></li>
      </ul>
      <div v-else-if="b.display" class="tex-row"><Tex :text="b.text" /></div>
      <p v-else><Tex :text="b.text" /></p>
    </template>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import Tex from './Tex.vue'

const props = defineProps({ text: { type: String, default: '' } })

const blocks = computed(() => {
  const out = []
  for (const line of props.text.split('\n').map((l) => l.trim()).filter(Boolean)) {
    const item = line.match(/^- (.*)/)?.[1]
    // 冒号前不超过 20 个字符才算关键词, 否则是句子里本来的冒号。关键词里可以有公式 ($β$ 大:), 公式按 3 个字符算;
    // 公式内部的冒号不算
    const m = item?.match(/^((?:\$[^$]+\$|[^:：$])+?[:：] ?)(.*)/)
    const short = m && m[1].replace(/\$[^$]+\$/g, 'xxx').length <= 21
    const [, label, body] = short ? m : [null, '', item]
    const last = out[out.length - 1]
    if (item && last?.items) last.items.push({ label, body })
    else out.push(item ? { items: [{ label, body }] } : { text: line, display: /^\$\$.*\$\$$/.test(line) })
  }
  return out
})
</script>

<style scoped>
.prose > * + * { margin-top: 0.55em; }
.prose p, .prose ul { margin: 0; }
.prose ul { padding-left: 1.2em; }
.prose li + li { margin-top: 0.3em; }
.prose .tex-row { overflow-x: auto; overflow-y: hidden; }
.prose li b { color: var(--text); font-weight: 600; }
</style>
