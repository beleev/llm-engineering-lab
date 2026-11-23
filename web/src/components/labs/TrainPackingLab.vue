<!--
  Sequence packing 的三样补丁 (对应 llm_train/m18_sequence_packing:doc_mask / reset_positions)。
  只讲一件事: 几篇文档拼进一行后, 要文档 mask + 位置重置 + 边界 label 屏蔽才与逐篇训练等价; RoPE 下位置重置可以省。
  max|Δ| 与 CE 取自 m18 demo 第 [2] 段输出 (一行 4 篇, 长度 [42,41,41,4], 共 128 token)。
  图里的 mask / 位置 / label 是按当前开关在前端现算的, 用一行 9 个 token 的小例子。
-->
<template>
  <LabFrame
    title="Packing — 几篇文档拼进同一行训练, 它们会互相看见吗?"
    sub="- 模拟什么: 预训练批次张量的序列长度通常固定 (如 m18 中每行固定为 128 个 token), 但真实语料中的单篇文档长短悬殊, 中位长度仅为 28 个 token。若每篇文档独占一行并在末尾补齐空位 (padding), 约七成算力将浪费在无效占位符上。序列拼接 (packing) 将多篇短文档首尾相连拼入同一条定长序列以消除空位浪费, 但若不加隔离, 同一序列中后一篇文档的 token 在自注意力计算时会关注到前一篇文档的内容, 从而学习到虚假的跨文档上下文关联。本实验台通过一条含 9 个 token 的微型拼接序列, 逐一演示恢复严格隔离所需的三项修正机制。要使拼接训练在数学上与逐篇独立训练完全等价, 必须同时补齐以下三项设置:
  - 文档注意力 mask: 限制每个 token 仅能关注同一篇文档内部且排在自身之前的 token。
  - 位置编号重置: 每篇新文档起始处将位置索引重置为从 0 开始计数。
  - 文档边界 label 屏蔽: 屏蔽每篇文档末尾 token 对下一篇文档首 token 的跨界预测损失。
- 图上是什么: 主体方阵展示自注意力可见性 mask, 第 $t$ 行对应发起查询的 token (query), 第 $s$ 列对应被关注的上下文 token (key), 行列标题的数字与底色标明各 token 所属的文档编号。
  - 方阵格颜色与下方图例对应: 绿色格表示允许关注且同属一篇文档; 红色格表示未被屏蔽从而发生了跨文档注意力泄露; 灰色格表示已被 mask 阻断。
  - 方阵下方的「位置」行: 显示每个 token 当前馈入模型的位置索引 (未按文档重置的错误索引标红)。
  - 方阵下方的「label」行: 显示每个位置用于计算交叉熵损失的目标下一 token 编号 (跨文档边界预测的目标标红, 已屏蔽不计入损失的位置呈灰色)。
- 可以做什么: 控件从上到下依次为:
  - 第一排提供文档 mask、位置重置与边界 label 屏蔽三个独立开关, 切换时可实时观察注意力方阵与底部两行标记的变化。
  - 第二排切换位置编码机制: 可选 RoPE (旋转位置编码, 注意力内积仅依赖两位置的相对差值) 或绝对位置编码 (直接将位置索引对应的嵌入向量加至输入表征)。
  - 点击「换一组长度」切换不同的多文档长度拼接组合。
  - 点击方阵左侧的 q0、q1 等行头选中特定 query 位置, 右侧首行读数将统计该 token 当前可见的上下文总数及其中混入的跨文档 token 数量。
- 应该看到: 关闭文档 mask 时, 自第二篇文档起各行左下角出现红色泄露格, 表明后文关注到了前序无关文档; 开启文档 mask 后红格全部转为灰色阻断态, 方阵仅保留沿主对角线排列的若干独立绿色下三角块 (每篇文档对应一块)。右侧读数含义如下:
  - 「有效预测」: 当前拼接序列中实际参与 loss 计算的预测位置总数; 开启边界屏蔽后会扣除文档篇数减 1 个跨界位置。
  - 「m18 实测 max|Δ|」: m18 在长度 128 的真实拼接样本上按当前开关组合实测得到的拼接前向输出与逐篇独立前向输出之间的最大绝对偏差; 1e-15 量级代表仅含浮点舍入误差 (严格等价), 而 3 至 5 量级代表模型输出已发生实质偏离。
  - 「与逐篇训练等价?」: 仅当三项隔离机制全部满足要求 (采用 RoPE 时位置重置开关可省略) 时才判定为「是」。"
    module="llm_train/m18"
    run="python -m llm_train.m18_sequence_packing.demo"
    :challenge="{
      ask: '在开启文档注意力 mask 但关闭位置重置的配置下, RoPE 与绝对位置编码哪一种仍与逐篇独立训练保持等价? 原因是什么?',
      answer: '采用 RoPE 时仍然严格等价, 而采用绝对位置编码时不再等价。\n- RoPE 保持等价的机制: m18 实测该配置下的最大绝对偏差 max|Δ| 仅为 1.3e-15, 属于纯浮点舍入误差。这是因为 RoPE 将位置索引映射为复平面旋转角, 任意两个 token 之间的注意力打分 $\\langle R(m)q, R(n)k \\rangle$ 仅取决于两者的相对位置差 $n-m$, 与各自起始编号的绝对偏移无关。只要开启了文档注意力 mask, 自注意力计算便严格限制在单篇文档内部进行; 而同一篇文档内部任意两位置的相对距离, 无论该篇起始编号是从 0 起算还是从 5 起算均完全不变, 因此前向输出严格一致。\n- 绝对位置编码失效的原因: 实测最大输出偏差高达 4.6。绝对位置编码将位置索引对应的嵌入向量直接加到 token 的初始输入向量上; 若不重置位置编号, 自第 2 篇文档起每个 token 注入的都是其在整条拼接行中的全局绝对坐标而非篇内相对坐标, 输入表征从源头即发生改变, 即使后续注意力 mask 阻断了跨文档交互也无法消除这一偏差。\n尽管数学上等价, 生产级训练框架 (如 Hugging Face 实现) 在使用 RoPE 配合 packing 时通常仍会显式重置每篇文档的位置编号。原因在于 bf16 低精度浮点下大位置索引会导致旋转角计算累积舍入误差, 按文档重置可确保每篇文本始终从高精度的小角度区间起步。',
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
        <p>右侧 max|Δ| 读数取自 m18 在单行拼接 4 篇文档、总长 128 个 token 的样本上实测的结果表, 随当前开关组合同步查表显示。1e-15 量级仅源于浮点运算顺序差异, 代表数学等价; 3 至 5 量级则表明前向输出已发生实质改变。由于离线脚本未单独测量「文档 mask 关闭且位置重置同时关闭」这一组合, 选中该状态时面板显示为「未测」。</p>
        <p>跨文档边界 label 的损失统计对比 (m18 固定实测基准, 不随上方开关变化, 单行含 4 篇文档共 128 个 token):</p>
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
