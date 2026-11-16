<!--
  BPE 分词实验台 (对应 llm_basic/bpe.py; 这里是字符级, Python 版是 byte-level)。
  只讲一件事: 每一步把语料里最高频的相邻 token 对合并成一个新 token, 词表一条一条长出来。
-->
<template>
  <LabFrame
    title="BPE 分词实验台 — 看着词表一条一条长出来"
    sub="- 模拟什么: 分词器 (tokenizer) 的训练过程。模型不直接读字符, 它读的是 token: 文本被切成的一个个小片段, 每个片段在词表里有一个编号。BPE (Byte Pair Encoding, 字节对编码) 造词表的办法只有一步, 反复做: 数一数语料里哪两个相邻的 token 挨在一起出现得最多, 把这一对粘成一个新 token, 加进词表。这里从单个字符出发; Python 版 bpe.py 从 256 个字节出发, 思想一样。
      - 图上是什么: 中间那片格子是语料此刻被切成的 token 序列, 一格一个 token, 顺序就是原文顺序。没上色的格子是还没被合并过的单字符, 上了色的是合并出来的多字符 token, 黄 (橙) 色高亮的是「选中的那次合并」造出来的 token 此刻出现在哪些位置 (默认跟着最新一次合并走)。␣ 表示空格, ⏎ 表示换行。
      - 可以做什么: 上面的文本框是语料, 可以直接改, 改完所有数字立刻重算。拖「合并次数」决定做多少轮合并, 0 就是纯字符级。右侧「合并记录」每一行是一次合并: 哪两个 token 粘成了什么, 粘之前这一对在语料里出现过几次; 点任意一行, 中间的高亮就换成那个 token。
      - 应该看到: 默认语料 145 个字符, 合并 0 次时 token 数等于字符数; 拖到 12 次剩 113 个 token, 40 次剩 65 个。右侧「压缩率」是字符数除以 token 数, 越大说明同样的文本用的 token 越少、序列越短。「词表大小」是语料里不同字符的个数 (默认语料 33 个) 加上合并次数, 每合并一次词表多一条。合并记录里的「出现 N 次」会一路变小: 第 1 次合并的 e + 空格出现 5 次, 到后面每次只出现 1 次, 合出来的 token 也越来越长, 越来越像在背这一段原文。"
    module="llm_basic/bpe.py"
    run="cd llm_basic && python bpe.py"
    :challenge="{
      ask: '默认语料 145 个字符。先猜: 合并次数从 12 拖到 40, 每次合并省下的 token 是越来越多还是越来越少? 最后一条合并出来的 token 有多长?',
      answer: '越来越少; 最后一条合出来的是一个 25 个字符的 token。\n- 每合并一次省多少: 一对相邻 token 在语料里出现 N 次, 把它们粘起来就少 N 个 token。BPE 每次挑的是当前最高频的一对, 所以 N 只会一路往下走: 第 1 次合并 (e + 空格) 出现 5 次, 省 5 个; 第 6 到 32 次每次只出现 2 次; 最后 8 次每次只出现 1 次, 合并它只是把两个本来就只挨过一次的 token 粘在一起。\n- 压缩率: 12 次合并时 145 ÷ 113 = 1.28, 40 次时 145 ÷ 65 = 2.23。\n- 过拟合: 第 40 条合出来的 token 是「First Citizen:⏎Before we␣」, 25 个字符, 整段语料里只出现 1 次。它不是「词」, 只是把这段文字背了下来, 换一段文字根本用不上。\n真实 GPT-2 在大得多的语料上做 5 万次合并, 那时出现成千上万次的高频对才是「词」和「词缀」。语料太小时, 合并次数一多就开始背原文。',
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
        <p>
          词表大小是一个权衡旋钮。词表越大, 同一段文本切出的 token 越少, 序列越短, 注意力 (按序列长度的平方计算) 越省;
          但 embedding 表 (每个 token 一行向量) 和输出层 lm_head 也跟着变大, 每个 token 在语料里被见到的次数也更少。
        </p>
        <p>
          真实 GPT-2 在 50 万倍大的语料上做 5 万次合并, 思想与这里完全一致。Python 版 bpe.py 是 byte-level:
          从 256 个字节而不是字符出发, 所以能处理任意 UTF-8 文本, 永远不会遇到词表外的字符。
        </p>
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
