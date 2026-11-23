<!--
  BPE 分词实验台 (对应 llm_basic/bpe.py; 这里是字符级, Python 版是 byte-level)。
  只讲一件事: 每一步把语料里最高频的相邻 token 对合并成一个新 token, 词表一条一条长出来。
-->
<template>
  <LabFrame
    title="BPE 分词实验台 — 看着词表一条一条长出来"
    sub="- 模拟什么: 演示 tokenizer 的训练构建流程。语言模型以离散 token 编号序列作为输入; 本实验台从单字符词表出发演示合并过程, Python 实现 bpe.py 则从 256 个基础字节出发, 核心算法完全一致。BPE (Byte Pair Encoding, 字节对编码) 通过循环执行以下三步扩充词表:
  1. 统计当前语料序列中所有相邻 token 对的共现频次, 找出最高频的一对。
  2. 将该相邻对合并为一个新 token。
  3. 将新 token 追加至词表。
- 图上是什么: 中央网格按原文顺序展示当前语料切分得到的 token 序列, 每格对应一个 token:
  - 未着色格子: 尚未参与任何合并的原始单字符。
  - 着色格子: 经多轮合并生成的多字符 token。
  - 黄 (橙) 色高亮格子: 当前「选中的那次合并」所生成的 token 在语料中的全部出现位置 (默认跟随最新一轮合并)。
  - ␣ 和 ⏎: ␣ 表示空格符, ⏎ 表示换行符。
- 可以做什么: 可操作语料文本框、合并轮数滑杆与合并记录列表:
  - 直接编辑顶部文本框中的训练语料, 修改后全部统计读数会实时重算。
  - 拖动「合并次数」设定执行的合并轮数, 取 0 时退化为纯字符级切分。
  - 右侧「合并记录」逐行列出每轮将哪两个相邻 token 合并为何种新 token 及其合并前的共现频次; 点击任意一行即可将中央网格高亮切换至对应 token。
- 应该看到: 默认语料共 145 个字符, 合并 0 次时序列长度等于字符数; 拖至 12 次时缩减为 113 个 token, 拖至 40 次时缩减为 65 个 token。右侧读数与合并记录规律如下:
  - 「压缩率」定义为原始字符数除以当前 token 数, 数值越大代表同等文本编码后的序列越短。
  - 「词表大小」等于语料去重基础字符数 (默认语料为 33 个) 加上已执行的合并次数, 每轮合并新增一个词表项。
  - 合并记录中的「出现 N 次」随轮数递增单调递减: 第 1 次合并 (e 与空格) 共现 5 次, 后期每轮仅共现 1 次, 合并出的子串长度不断增长, 逐渐退化为对当前短语料的逐字记忆。"
    module="llm_basic/bpe.py"
    run="cd llm_basic && python bpe.py"
    :challenge="{
      ask: '在 145 个字符的默认语料上, 将合并次数从 12 拖至 40。请先判断: 单次合并减少的 token 数量会递增还是递减? 最后一轮合并生成的 token 包含多少个字符?',
      answer: '单次合并减少的 token 数逐渐递减; 最后一轮合并生成的是长度为 25 个字符的 token。\n- 单轮收益递减机制: 若一对相邻 token 在语料中共现 N 次, 将其合并可使序列总长度减少 N。由于 BPE 每轮贪心选取当前频次最高的相邻对, 最高共现频次 N 随轮数推进呈单调不增趋势:\n  - 第 1 次合并 (e + 空格) 共现 5 次, 序列缩短 5 个 token。\n  - 第 6 至 32 次合并每轮共现频次仅为 2 次。\n  - 最后 8 次合并每轮共现频次仅为 1 次, 仅将全文唯一相邻出现一次的两个片段拼接在一起。\n- 压缩率变化: 12 次合并时压缩率为 $145 / 113 = 1.28$, 40 次合并时升至 $145 / 65 = 2.23$。\n- 词表过拟合: 第 40 轮合并生成的 token 为「First Citizen:⏎Before we␣」, 长度达 25 个字符, 在整段语料中仅出现 1 次。该片段并非可泛化的词或词缀, 而是对训练文本片段的硬编码记忆, 迁移至新文本时几乎无法命中。\n真实 GPT-2 在规模大数个数量级的语料上执行 5 万次合并, 此时共现成千上万次的高频相邻对才能稳定对应自然语言中的常见词根与词缀; 若训练语料过小而合并轮数过多, 词表便会退化为记忆原文片段。',
    }"
  >
    <template #controls>
      <label :for="boxId" class="corpus-label">语料 (可编辑, 最多 {{ MAX_CHARS }} 字符)</label>
      <textarea :id="boxId" ref="box" v-model="text" rows="3" :maxlength="MAX_CHARS" spellcheck="false" class="mono corpus" @input="fit"></textarea>
      <LabSlider v-model="numMerges" label="合并次数" :min="0" :max="40" />
    </template>

    <div class="token-view">
      <span
        v-for="(t, idx) in tokens" :key="idx"
        class="token mono" :class="{ multi: isMulti(t), hot: t === picked }"
      >{{ disp(t) }}</span>
    </div>

    <template #stats>
      <div class="kv"><span>现在的 token 数 / 语料字符数</span><b>{{ tokens.length }} / {{ charCount }}</b></div>
      <div class="kv"><span>压缩率 (字符数 ÷ token 数, 越大序列越短)</span><b :class="{ good: tokens.length < charCount }">{{ compression }}</b></div>
      <div class="kv"><span>词表大小 (不同字符数 + 合并次数)</span><b>{{ baseVocab }} + {{ merges.length }}</b></div>
      <div class="kv"><span>选中的 token 此刻在语料里出现</span><b>{{ picked === null ? '—' : pickedCount + ' 处' }}</b></div>
      <div class="merge-log">
        <p class="log-title">合并记录 (最新的在最上面; 点一行, 中间高亮它合出来的 token)</p>
        <button
          v-for="m in log" :key="m.idx" type="button" class="log-line mono" :class="{ on: m.idx === pickIdx }"
          :aria-pressed="m.idx === pickIdx" @click="pick = m.idx"
        >{{ m.idx + 1 }}. '{{ disp(m.a) }}'+'{{ disp(m.b) }}' → '{{ disp(m.a + m.b) }}' (出现 {{ m.count }} 次)</button>
        <p v-if="merges.length === 0" class="log-line mono">(拖动滑块开始合并)</p>
      </div>
      <div class="lab-note">
        <p>词表大小决定了序列长度与参数开销之间的工程权衡。词表越大:</p>
        <ul class="pts">
          <li><b>好处:</b> 同一段文本切出的 token 越少, 序列越短, 注意力 (按序列长度的平方计算) 越省。</li>
          <li><b>代价:</b> embedding 表 (每个 token 一行向量) 和输出层 lm_head 也跟着变大, 每个 token 在语料里被见到的次数也更少。</li>
        </ul>
        <p>真实 GPT-2 在规模约大 50 万倍的语料上执行 5 万次合并, 核心算法与此完全一致。Python 实现 bpe.py 采用 byte-level BPE 方案: 基础词表从 256 个原始字节而非字符出发, 因此可无损编码任意 UTF-8 文本, 不会出现未登录字符 (OOV)。</p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, nextTick, onMounted, ref, useId, watch } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'

const DEFAULT_TEXT =
  'First Citizen:\nBefore we proceed any further, hear me speak.\n' +
  'All: Speak, speak.\nFirst Citizen: You are all resolved rather to die than to famish?'
const MAX_CHARS = 2000

const text = ref(DEFAULT_TEXT)
const numMerges = ref(12)
const pick = ref(-1)          // 选中的合并记录下标; -1 = 跟着最新一条走
const boxId = useId()

// 输入框高度跟着内容走, 默认文本全部可见; 超过 max-height 再出滚动条
const box = ref(null)
const fit = () => { const el = box.value; if (!el) return; el.style.height = 'auto'; el.style.height = `${el.scrollHeight + 2}px` }
onMounted(() => nextTick(fit))
// 语料或合并次数一变, 高亮回到最新一条
watch([text, numMerges], () => { pick.value = -1 })

const clipped = computed(() => text.value.slice(0, MAX_CHARS))
const chars = computed(() => Array.from(clipped.value))
const charCount = computed(() => chars.value.length)
const baseVocab = computed(() => new Set(chars.value).size)

// ★ char-level BPE: 每轮统计相邻 token 对频次, 合并最高频的一对 (平手取先出现者)
const bpe = computed(() => {
  let toks = chars.value
  const applied = []
  for (let step = 0; step < numMerges.value; step++) {
    if (toks.length < 2) break
    const counts = new Map()
    for (let k = 0; k < toks.length - 1; k++) {
      const key = toks[k].length + ':' + toks[k] + toks[k + 1]
      const e = counts.get(key)
      if (e) e.count += 1
      else counts.set(key, { a: toks[k], b: toks[k + 1], count: 1 })
    }
    let best = null
    for (const e of counts.values()) if (!best || e.count > best.count) best = e
    if (!best) break
    const next = []
    for (let k = 0; k < toks.length; k++) {
      if (k < toks.length - 1 && toks[k] === best.a && toks[k + 1] === best.b) {
        next.push(best.a + best.b)
        k += 1
      } else {
        next.push(toks[k])
      }
    }
    toks = next
    applied.push({ idx: step, a: best.a, b: best.b, count: best.count })
  }
  return { toks, applied }
})

const tokens = computed(() => bpe.value.toks)
const merges = computed(() => bpe.value.applied)
const log = computed(() => [...merges.value].reverse())
const compression = computed(() =>
  tokens.value.length > 0 ? (charCount.value / tokens.value.length).toFixed(2) : '—'
)
const pickIdx = computed(() => (pick.value >= 0 && pick.value < merges.value.length ? pick.value : merges.value.length - 1))
const picked = computed(() => { const m = merges.value[pickIdx.value]; return m ? m.a + m.b : null })
const pickedCount = computed(() => tokens.value.filter((t) => t === picked.value).length)

const isMulti = (t) => Array.from(t).length > 1
const disp = (t) => t.replace(/ /g, '␣').replace(/\n/g, '⏎')
</script>

<style scoped>
.corpus-label { font-size: 12px; color: var(--text-muted); }
.corpus {
  box-sizing: border-box; width: 100%; max-height: 360px; resize: vertical; font-size: 12px; line-height: 1.6; padding: 8px 10px;
  background: var(--code-bg); color: var(--text);
  border: 1px solid var(--border); border-radius: var(--radius-sm);
}
.corpus:focus { outline: none; border-color: var(--accent); }
.token-view {
  display: flex; flex-wrap: wrap; gap: 3px; align-content: flex-start;
  background: var(--code-bg); border-radius: var(--radius-sm); padding: 10px;
  max-height: 300px; overflow-y: auto;
}
.token {
  font-size: 11px; line-height: 1.5; padding: 1px 4px; border-radius: 3px;
  background: var(--bg-elev); color: var(--text-muted); white-space: pre;
}
.token.multi { background: var(--accent-soft); color: var(--text); }
.token.hot { background: var(--warn); color: var(--code-bg); }
.merge-log {
  max-height: 180px; overflow-y: auto; padding: 8px 10px;
  background: var(--bg-elev); border: 1px solid var(--border); border-radius: var(--radius-sm);
}
.log-title { font-size: 12px; color: var(--text-muted); margin-bottom: 6px; }
.log-line { display: block; width: 100%; min-height: 0; padding: 0 4px; border: 0; border-radius: 3px; background: none; text-align: left; font-size: 11px; line-height: 1.7; color: var(--text-dim); white-space: pre-wrap; }
button.log-line { cursor: pointer; }
.log-line.on { background: color-mix(in srgb, var(--warn) 25%, transparent); color: var(--text); }
</style>
