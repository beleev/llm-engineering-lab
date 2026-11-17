<!--
  Sequence packing 的三样补丁 (对应 llm_train/m18_sequence_packing:doc_mask / reset_positions)。
  只讲一件事: 几篇文档拼进一行后, 要文档 mask + 位置重置 + 边界 label 屏蔽才与逐篇训练等价; RoPE 下位置重置可以省。
  max|Δ| 与 CE 取自 m18 demo 第 [2] 段输出 (一行 4 篇, 长度 [42,41,41,4], 共 128 token)。
  图里的 mask / 位置 / label 是按当前开关在前端现算的, 用一行 9 个 token 的小例子。
-->
<template>
  <LabFrame
    title="Packing — 几篇文档拼进同一行训练, 它们会互相看见吗?"
    sub="- 模拟什么: 训练用的序列是定长的 (本例 m18 里一行 128 个 token), 但文档长短不一, 中位数只有 28 个 token。每篇单独占一行、后面补空位 (padding), 七成算力花在空位上。packing 就是把几篇文档首尾相接塞进同一行, 不留空位。代价是同一行里后一篇的 token 在注意力里能看见前一篇, 学到的是一个本不存在的上下文。这个台子用一行 9 个 token 的小例子, 把需要补的三样开关逐个打开给你看。要和逐篇单独训练完全一样, 要补这三样:
        - 文档 mask: 只许看同一篇里排在自己前面的 token。
        - 位置重置: 每篇的位置编号从 0 重新数。
        - 边界 label 屏蔽: 每篇最后一个 token 不去预测下一篇的第一个。
      - 图上是什么: 方阵是注意力 mask, 第 $t$ 行是正在提问的 token (query), 第 $s$ 列是它可能看的 token (key)。列头和行头的数字、颜色表示这个 token 属于第几篇文档。
        - 格子颜色对照下面的图例: 绿色是能看而且是同一篇的; 红色是能看但看到了别的文档 (串文档了); 灰色是被 mask 挡住看不见的。
        - 方阵下面的「位置」行: 每个 token 的位置编号 (错的标红)。
        - 方阵下面的「label」行: 每个 token 要预测的下一个 token 编号 (跨到下一篇去预测的标红, 被屏蔽的变灰)。
      - 可以做什么: 控件从上到下:
        - 第一排三个开关分别打开或关闭文档 mask、位置重置、边界 label 屏蔽, 看方阵和下面两行怎么变。
        - 第二排选位置编码方式: RoPE (旋转位置编码, 注意力只看两个 token 的位置差) 或绝对位置编码 (把位置编号本身加进输入向量)。
        - 「换一组长度」换一种文档长度组合。
        - 点方阵左边的 q0、q1 等行头, 右侧第一行读数会告诉你这个 token 能看见几个 token、其中几个来自别的文档。
      - 应该看到: 文档 mask 关着时, 第二篇起的行里有红格子, 后面的文档能看见前面的。打开文档 mask, 红格子全部变灰, 方阵只剩沿对角线的几个绿色小方块 (每篇一块)。右侧读数:
        - 「有效预测」: 这一行里真正算 loss 的 token 数, 开边界屏蔽后会减去篇数减 1。
        - 「m18 实测 max|Δ|」: m18 在真实 128 token 的例子上按当前开关组合测出来的 packing 输出与逐篇单独前向输出的最大差。1e-15 量级是浮点零头 (等价), 3 到 5 是真的不一样。
        - 「与逐篇训练等价?」: 只有三样都开 (RoPE 下位置重置可以关) 才是「是」。"
    module="llm_train/m18"
    run="python -m llm_train.m18_sequence_packing.demo"
    :challenge="{
      ask: '开着文档 mask、关掉位置重置。RoPE 和绝对位置编码, 哪个还与逐篇训练等价? 为什么?',
      answer: 'RoPE 还等价, 绝对位置编码不等价。\n- RoPE 等价: m18 实测 max|Δ| 是 1.3e-15, 浮点零头。原因是 RoPE 把位置编号变成一个旋转角, 两个 token 的注意力分数 $\\langle R(m)q, R(n)k \\rangle$ 只依赖位置差 $n-m$, 和各自的绝对编号无关。开着文档 mask 时, 注意力只在同一篇文档内部算, 同一篇里任意两个 token 的位置差, 不管编号从 0 数还是从 5 数都一样, 所以结果不变。\n- 绝对位置编码不等价: 差 4.6。它把位置编号对应的向量直接加进 token 的输入表示。第 2 篇起每个 token 拿到的是「我在整行里排第几」而不是「我在本篇里排第几」, 输入向量本身就错了, 后面怎么算都是错的, 和 mask 挡没挡住无关。\n真实系统用 RoPE 时一般仍然重置位置 (HF 的做法)。原因是 bf16 精度下位置编号很大时旋转角有误差, 重置能让每篇都从小角度开始, 更稳。',
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

    <div class="pk lab-scroll">
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
    <p class="legend">
      <span class="cell ok" /> 同一篇里能看 <span class="cell bad" /> 看到了别的文档, 或位置 / label 错了 <span class="cell dim" /> 被 mask 掉 · 列头的数字 = 第几篇文档
    </p>

    <template #stats>
      <div class="kv"><span>选中的 q{{ q }} 能看见 {{ seen.all }} 个 token, 其中来自别的文档的</span><b :class="seen.other ? 'bad' : 'good'">{{ seen.other }} 个</b></div>
      <div class="kv"><span>有效预测 (本图 {{ T }} 个 token 里真正算 loss 的个数)</span><b>{{ valid }}</b></div>
      <div class="kv"><span>m18 实测 max|Δ| ({{ pe === 'rope' ? 'RoPE' : '绝对位置' }}): packing 输出与逐篇输出的最大差</span><b :class="delta === null ? '' : delta < 1e-10 ? 'good' : 'bad'">{{ delta === null ? '未测' : delta.toExponential(1) }}</b></div>
      <div class="kv"><span>与逐篇训练等价?</span><b :class="equiv ? 'good' : 'bad'">{{ equiv ? '是' : '否' }}</b></div>
      <div class="lab-note">
        <p>★ 文档 mask 只有一行代码: <code class="inline">(t ≥ s) & (doc[t] == doc[s])</code>, 意思是「s 排在 t 前面 (因果), 而且两者属于同一篇」才许看。</p>
        <p>max|Δ| 是 m18 在一行 4 篇、共 128 个 token 的真实例子上测的, 按当前开关组合查表。1e-15 量级是浮点运算的零头, 算等价; 3 到 5 是输出真的变了。demo 没测「文档 mask 关、位置重置也关」这一格, 所以显示「未测」。</p>
        <p>边界 label 的账 (m18 实测, 不随开关变化), 一行 4 篇 128 个 token:</p>
        <ul class="pts">
          <li><b>屏蔽边界:</b> 有效预测 124 个, 交叉熵 (CE) 总和 566.353069, 与逐篇训练的 566.353069 完全相同。</li>
          <li><b>不屏蔽边界:</b> 多出 3 个跨篇预测 (有效预测 127), CE 总和变成 577.563075, 多出来的都是在学「上一篇的结尾接什么」这种没有意义的事。</li>
        </ul>
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
.legend { display: flex; flex-wrap: wrap; align-items: center; gap: 6px 8px; margin-top: 10px; font-size: 11px; color: var(--text-muted); }
.legend .cell { min-width: 14px; width: 14px; height: 14px; }
</style>
