<!--
  Sequence packing 的三样补丁 (对应 llm_train/m18_sequence_packing:doc_mask / reset_positions)。
  只讲一件事: 几篇文档拼进一行后, 要文档 mask + 位置重置 + 边界 label 屏蔽才与逐篇训练等价; RoPE 下位置重置可以省。
  max|Δ| 与 CE 取自 m18 demo 第 [2] 段输出 (一行 4 篇, 长度 [42,41,41,4], 共 128 token)。
  图里的 mask / 位置 / label 是按当前开关在前端现算的, 用一行 9 个 token 的小例子。
-->
<template>
  <LabFrame
    title="Packing — 拼在一行的文档会互相看见吗?"
    sub="方阵是注意力 mask: 第 $t$ 行是 query, 第 $s$ 列是 key。绿格 = 同一篇里能看, 红格 = 看到了别的文档 (串文档)。
      下面两行是位置 id 和每个 token 要预测的 label。点某一行, 看这个 token 到底能看见谁。"
    module="llm_train/m18"
    run="python -m llm_train.m18_sequence_packing.demo"
    :challenge="{
      ask: '开着文档 mask、关掉位置重置。RoPE 和绝对位置编码, 哪个还与逐篇训练等价? 为什么?',
      answer: '- RoPE 还等价: m18 max|Δ| 1.3e-15。RoPE 的 $\\langle R(m)q, R(n)k \\rangle$ 只依赖 $n-m$。有文档 mask 时只在同一篇里算注意力, 块内相对位置没变。\n- 绝对位置编码不等价: 差 4.6。第 2 篇起每个 token 都带着错的位置向量进了 embedding, 连它看见的内容本身都变了。\n真实系统用 RoPE 时仍然重置位置 (HF 的做法)。位置很大时 bf16 下的角度有误差, 重置更稳。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: docMask }" @click="docMask = !docMask">文档 mask {{ docMask ? '开' : '关' }}</button>
        <button type="button" :class="{ active: reset }" @click="reset = !reset">位置重置 {{ reset ? '开' : '关' }}</button>
        <button type="button" :class="{ active: labelMask }" @click="labelMask = !labelMask">边界 label 屏蔽 {{ labelMask ? '开' : '关' }}</button>
      </div>
      <div class="row">
        <button type="button" :class="{ active: pe === 'rope' }" @click="pe = 'rope'">RoPE</button>
        <button type="button" :class="{ active: pe === 'abs' }" @click="pe = 'abs'">绝对位置编码</button>
        <button type="button" @click="lp = (lp + 1) % LENS.length">换一组长度 {{ LENS[lp].join(' + ') }}</button>
      </div>
    </template>

    <div class="pk">
      <div class="grid" :style="{ gridTemplateColumns: `40px repeat(${T}, 24px)` }">
        <span />
        <span v-for="s in T" :key="'h' + s" class="hd mono" :class="'d' + (doc[s - 1] % 4)">{{ doc[s - 1] + 1 }}</span>
        <template v-for="t in T" :key="'r' + t">
          <button type="button" class="hd rowh mono" :class="['d' + (doc[t - 1] % 4), { sel: q === t - 1 }]" @click="q = t - 1">q{{ t - 1 }}</button>
          <span v-for="s in T" :key="s" class="cell" :class="cellCls(t - 1, s - 1)" />
        </template>
        <span class="lbl">位置</span>
        <span v-for="t in T" :key="'p' + t" class="cell mono" :class="posBad(t - 1) ? 'bad' : ''">{{ pos[t - 1] }}</span>
        <span class="lbl">label</span>
        <span v-for="t in T" :key="'l' + t" class="cell mono" :class="labelCls(t - 1)">{{ t < T ? '→' + t : '·' }}</span>
      </div>
    </div>

    <template #stats>
      <div class="kv"><span>q{{ q }} 能看见的 token</span><b>{{ seen.all }} 个</b></div>
      <div class="kv"><span>其中来自别的文档</span><b :class="seen.other ? 'bad' : 'good'">{{ seen.other }} 个</b></div>
      <div class="kv"><span>有效预测 (本图 {{ T }} token)</span><b>{{ valid }}</b></div>
      <div class="kv"><span>m18 实测 max|Δ| ({{ pe === 'rope' ? 'RoPE' : '绝对位置' }})</span><b :class="delta === null ? '' : delta < 1e-10 ? 'good' : 'bad'">{{ delta === null ? '未测' : delta.toExponential(1) }}</b></div>
      <div class="kv"><span>与逐篇训练等价?</span><b :class="equiv ? 'good' : 'bad'">{{ equiv ? '是' : '否' }}</b></div>
      <div class="lab-note">
        <p>★ 文档 mask 只有一行: <code class="inline">(t ≥ s) & (doc[t] == doc[s])</code>。</p>
        <p>m18 的 CE 总和: 三样都做 566.353069 = 逐篇 566.353069; 不屏蔽边界 577.563075 (有效预测 124 → 127)。</p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import { range } from '@/utils/labmath.js'

// m18 demo [2] 的 max|Δ| 表: 键 = 文档mask|位置重置; 两列 RoPE / 绝对位置。demo 没测 "两样都不做"。
const DELTA = { 'true|true': [1.3e-15, 2.7e-15], 'true|false': [1.3e-15, 4.6], 'false|true': [3.0, 4.3] }
const LENS = [[3, 2, 4], [5, 4], [2, 2, 2, 3]]
const docMask = ref(false), reset = ref(true), labelMask = ref(false), pe = ref('rope'), lp = ref(0), q = ref(6)

const doc = computed(() => LENS[lp.value].flatMap((n, d) => Array(n).fill(d)))
const T = computed(() => doc.value.length)
const start = computed(() => doc.value.map((d) => doc.value.indexOf(d)))
const pos = computed(() => range(T.value).map((t) => (reset.value ? t - start.value[t] : t)))
const allowed = (t, s) => t >= s && (!docMask.value || doc.value[t] === doc.value[s]) // ★ 因果 ∧ 同一篇
const cellCls = (t, s) => {
  if (!allowed(t, s)) return q.value === t ? 'sel-row' : 'dim'
  const cross = doc.value[t] !== doc.value[s]
  return [cross ? 'bad' : 'ok', q.value === t ? 'sel-row' : '']
}
// 绝对位置编码 + 不重置: 第 2 篇起的位置都是错的
const posBad = (t) => pe.value === 'abs' && start.value[t] > 0 && !reset.value
// label: 每篇最后一个 token 去预测下一篇的开头 = 跨篇
const boundary = (t) => t < T.value - 1 && doc.value[t] !== doc.value[t + 1]
const labelCls = (t) => (t === T.value - 1 ? 'dim' : boundary(t) ? (labelMask.value ? 'dim' : 'bad') : 'ok')
const valid = computed(() => range(T.value - 1).filter((t) => !(labelMask.value && boundary(t))).length)

const seen = computed(() => {
  const vis = range(T.value).filter((s) => allowed(q.value, s))
  return { all: vis.length, other: vis.filter((s) => doc.value[s] !== doc.value[q.value]).length }
})
const delta = computed(() => {
  const row = DELTA[`${docMask.value}|${reset.value}`]
  return row ? row[pe.value === 'rope' ? 0 : 1] : null
})
const equiv = computed(() => docMask.value && (reset.value || pe.value === 'rope') && labelMask.value)
</script>

<style scoped>
.pk { overflow-x: auto; }
.grid { display: grid; gap: 3px; width: max-content; }
.grid .cell { min-width: 0; width: 24px; }
.hd { font-size: 10px; text-align: center; color: var(--text-muted); align-self: center; }
.rowh { padding: 0; min-height: 22px; font-size: 10px; }
.rowh.sel { box-shadow: 0 0 0 2px var(--accent); }
.d0 { color: var(--accent); }
.d1 { color: var(--eye); }
.d2 { color: var(--right); }
.d3 { color: var(--warn); }
.lbl { font-size: 10px; color: var(--text-dim); align-self: center; }
.cell.sel-row { outline: 1px solid var(--accent); }
</style>
