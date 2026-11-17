<!--
  文案渲染: 一个字符串里, 换行 = 分段, "- " 开头的连续行 = 无序列表, "1. " "2. " 开头的连续行 = 有序列表 (步骤)。
  列表项下面再缩进 2 格的 "- " / "1. " 行是子项 (只支持一层)。
  要点以 "短关键词: " 开头时, 关键词加粗, 方便扫读。
  段落以 "结论: " "边界: " 这类 6 字以内的中文短标签开头时, 标签显示成小色块。
  lead: 第一段的第一句 (要点都是结论先行) 单独加粗, 扫一遍加粗句就能拿到主干。
  文字里的括号注解、代码标识符、关键数字由 Tex 的 rich 模式分层 (utils/richText.js)。
  数据里照旧写字符串, 只是多了换行; 没有换行的旧文案原样显示成一段。
  $…$ 是行内公式, 单独一行的 $$…$$ 是独立公式。一个公式不能跨 \n, 要多行就在一个 $$…$$ 里用 aligned。
-->
<template>
  <div class="prose">
    <template v-for="(b, i) in blocks" :key="i">
      <component :is="b.ordered ? 'ol' : 'ul'" v-if="b.items" :start="b.ordered && b.start !== 1 ? b.start : undefined">
        <!-- 正文包一层 span: 实验台说明把「标签 / 正文」排成两列网格时要两个元素 -->
        <li v-for="(x, xi) in b.items" :key="xi" :class="{ labeled: x.label }"><b v-if="x.label"><Tex :text="x.label" /></b><span class="li-body"><Tex :text="x.body" :rich="rich" /><component :is="x.subOrdered ? 'ol' : 'ul'" v-if="x.sub" class="sub-list"><li v-for="(y, yi) in x.sub" :key="yi"><b v-if="y.label"><Tex :text="y.label" /></b><Tex :text="y.body" :rich="rich" /></li></component></span></li>
      </component>
      <div v-else-if="b.display" class="tex-row"><Tex :text="b.text" /></div>
      <p v-else :class="{ 'has-lead': b.lead }">
        <span v-if="b.label" class="p-label" :class="b.tone">{{ b.label }}</span>
        <strong v-if="b.lead" class="lead-sentence"><Tex :text="b.lead" :rich="rich" /></strong>
        <Tex :text="b.text" :rich="rich" />
      </p>
    </template>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import Tex from './Tex.vue'
import { splitLead } from '@/utils/richText'

const props = defineProps({
  text: { type: String, default: '' },
  lead: { type: Boolean, default: false },
  rich: { type: Boolean, default: true },
})

// 段首标签的配色: 结论类用主色, 代价 / 边界类用警示色, 其余中性
const KEY = new Set(['结论', '重点', '一句话', '核心'])
const WARN = new Set(['边界', '代价', '注意', '风险', '局限', '没模拟的', '反例', '误区'])

// "关键词: 正文" → { label, body }。冒号前不超过 20 个字符才算关键词, 否则是句子里本来的冒号。
// 关键词里可以有公式 ($β$ 大:), 公式按 3 个字符算; 公式内部的冒号不算
function splitLabel(item) {
  // 冒号后面要跟空格 (或在行尾) 才算关键词: 代码里的 answer:observed、{type:tool_use} 不算
  const m = item.match(/^((?:\$[^$]+\$|[^:：$])+?(?:: |：|:$))(.*)/)
  const short = m && m[1].replace(/\$[^$]+\$/g, 'xxx').length <= 21
  // 关键词全是英文时 (GQA/MLA:、q (query):) zhPunct 会跳过它, 末尾冒号在这里统一成中文冒号, 和同一列表里的中文关键词一致
  return short ? { label: m[1].replace(/: ?$/, '：'), body: m[2] } : { label: '', body: item }
}

// 行的缩进: 模板属性里的多行文字, 第 2 行起都带着模板缩进, 第 1 行没有。所以第 1 行不参与计算,
// 其余行减去它们共同的最小缩进 (同 Python 的 inspect.cleandoc)。去缩进后还缩进 2 格以上的列表行是子项。
function lines(text) {
  const raw = text.split('\n')
  const rest = raw.slice(1).filter((l) => l.trim())
  const base = rest.length ? Math.min(...rest.map((l) => l.length - l.trimStart().length)) : 0
  return raw.map((l, i) => ({ indent: i === 0 ? 0 : Math.max(0, l.length - l.trimStart().length - base), text: l.trim() }))
    .filter((l) => l.text)
}

const blocks = computed(() => {
  const out = []
  for (const { indent, text: line } of lines(props.text)) {
    // "- 要点" 是无序项, "1. 第一步" 是有序项
    const un = line.match(/^- (.*)/)?.[1]
    const om = un == null ? line.match(/^(\d{1,2})\. (.*)/) : null
    const item = un ?? om?.[2]
    const last = out[out.length - 1]
    if (item != null && indent >= 2 && last?.items) {
      const parent = last.items[last.items.length - 1]
      ;(parent.sub ||= []).push(splitLabel(item))
      parent.subOrdered = om != null
      continue
    }
    if (item != null) {
      const ordered = om != null
      if (last?.items && last.ordered === ordered) last.items.push(splitLabel(item))
      else out.push({ items: [splitLabel(item)], ordered, start: ordered ? Number(om[1]) : 1 })
      continue
    }
    if (/^\$\$.*\$\$$/.test(line)) { out.push({ text: line, display: true }); continue }
    const lab = line.match(/^([\u4e00-\u9fff]{1,6})[:：] ?(.+)/)
    const p = lab ? { label: lab[1], tone: KEY.has(lab[1]) ? 'key' : WARN.has(lab[1]) ? 'warn' : '', text: lab[2] } : { text: line }
    out.push(p)
  }
  // 首句加粗只给第一段, 而且第一段得是普通段落 (不是列表 / 公式)
  const first = out[0]
  if (props.lead && first && !first.items && !first.display) {
    const [lead, rest] = splitLead(first.text)
    // 第一段只有一句时整段就是结论; 但整条要点只有这一段时不加粗 (全粗等于没粗)
    if (rest.trim() || out.length > 1) Object.assign(first, { lead, text: rest })
  }
  return out
})
</script>

<style scoped>
.prose > * + * { margin-top: 0.55em; }
.prose p, .prose ul, .prose ol { margin: 0; }
.prose ul, .prose ol { padding-left: 1.4em; }
.prose ol { list-style: decimal; }
.prose .sub-list { margin-top: 0.3em; padding-left: 1.2em; }
.prose .sub-list li + li { margin-top: 0.2em; }
.prose li + li { margin-top: 0.3em; }
.prose .tex-row { overflow-x: auto; overflow-y: hidden; }
.prose li b { color: var(--text); font-weight: 600; }
</style>
