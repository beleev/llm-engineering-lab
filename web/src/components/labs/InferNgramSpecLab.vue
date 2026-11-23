<!-- 无草稿模型投机实验台 (llm_infer/m36): PromptLookupDrafter.propose + m07 accept_greedy 的 JS 移植, target 是写好参考答案的 oracle。 -->
<template>
  <LabFrame
    title="Prompt lookup — 输出照抄输入时, 草稿直接从 prompt 里查"
    sub="- 模拟什么: 投机解码 (m07) 不一定要草稿模型。取已生成文本最后 n 个 token, 在 prompt 和已生成文本里找它最近一次出现的位置, 把后面的 K 个 token 抄来当草稿, target 一次 forward 验完, 接受最长的一致前缀, 再白送 1 个纠错 token。target 在这里是写好参考答案的 oracle: 它的 greedy 输出就是下面的「参考输出」。
- 图上是什么: 上排是 prompt, 下排是输出。当前这一轮里: 黄色 = 末尾用来查的 n-gram 和它在前文里的匹配, 绿色 = 被接受的草稿, 红色删除线 = 被拒的草稿, 蓝色 = target 白送的纠错或 bonus token。
- 可以做什么: 选负载 (代码编辑 / RAG 摘录 / 开放生成); 拖「草稿长度 K」和「最长 n-gram」; 用播放器一轮一轮走, 或拖时间轴跳到任意一轮。
- 应该看到: 代码编辑和 RAG 摘录里大段输出直接抄自 prompt, 一轮常常接受 4 个以上, 每次 forward 产出远大于 1; 开放生成里几乎查不到匹配, 每轮只产出 1 个, 和逐 token 一样。右侧「草稿模型 forward」恒为 0。"
    module="llm_infer/m36"
    run="python -m llm_infer.m36_draftless_speculation.demo"
    :challenge="{
      ask: '在「开放生成」里, prompt lookup 查不到匹配或猜错时, target 的 forward 次数会比逐 token 解码更多吗?',
      answer: '不会, 最坏就是和逐 token 一样多。\n查不到匹配时不提草稿, 这一轮就是一次普通 decode; 查到了但猜错时, 草稿在同一次 forward 里被验掉, target 照样白送 1 个纠错 token。每轮至少产出 1 个, 所以 forward 次数不超过输出长度。\n代价在别处: 被拒的草稿位置在那次 forward 里白算了。Python demo 的开放生成里 prompt lookup 白验了 1178 个草稿 token。decode 受带宽限制时这几乎不花时间; batch 大到算力受限时就是真实开销。',
    }"
  >
    <template #controls>
      <div class="row">
        <button v-for="(w, i) in WORKLOADS" :key="w.name" type="button" :class="{ active: wl === i }" @click="wl = i">{{ w.name }}</button>
      </div>
      <LabSlider v-model="K" label="草稿长度 K" :min="1" :max="8" />
      <LabSlider v-model="maxN" label="最长 n-gram" :min="1" :max="4" />
      <StepPlayer :stepper="stepper" :label="`第 ${cur.idx + 1} / ${rounds.length} 轮`" />
    </template>

    <div class="seq">
      <p class="cap">prompt</p>
      <div class="toks">
        <span v-for="(t, i) in W.prompt" :key="`p${i}`" class="tok" :class="{ hit: cur.src === 'p' && i >= cur.m0 && i < cur.m0 + cur.n, copy: cur.src === 'p' && i >= cur.m0 + cur.n && i < cur.m0 + cur.n + cur.draft.length }">{{ t }}</span>
      </div>
      <p class="cap">输出 (参考输出共 {{ W.ref.length }} 个 token)</p>
      <div class="toks">
        <span v-for="(t, i) in shown" :key="`o${i}`" class="tok" :class="outCls(i)">{{ t }}</span>
        <span v-for="(t, i) in cur.draft.slice(cur.acc)" :key="`r${i}`" class="tok rej">{{ t }}</span>
      </div>
      <p class="lab-note">{{ cur.note }}</p>
    </div>

    <template #stats>
      <div class="kv"><span>target forward: 逐 token / 投机</span><b>{{ W.ref.length }} / {{ rounds.length + 1 }}</b></div>
      <div class="kv"><span>每次 forward 产出 (输出数 ÷ forward 数)</span><b :class="speed > 2 ? 'good' : speed < 1.2 ? 'bad' : ''">{{ speed.toFixed(2) }}×</b></div>
      <div class="kv"><span>草稿接受率 (接受 ÷ 提出)</span><b :class="acceptRate > 0.5 ? 'good' : acceptRate < 0.1 ? 'bad' : ''">{{ (acceptRate * 100).toFixed(0) }}%</b></div>
      <div class="kv"><span>草稿模型 forward</span><b class="good">0</b></div>
      <div class="lab-note">
        <p>Python demo, K=8, 每类 5 条请求 (实测, 不随左侧变化), 每次 forward 产出:</p>
        <ul class="pts">
          <li><b>代码编辑:</b> 1500 → 265 次 forward, 5.66×</li>
          <li><b>RAG 摘录:</b> 660 → 226 次, 2.92×</li>
          <li><b>开放生成:</b> 660 → 660 次, 1.00×</li>
          <li><b>跨请求模板:</b> 只查当前 prompt 的接受率 0.0%, 记住以前输出的 n-gram 缓存 83.7%</li>
        </ul>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import StepPlayer from '@/components/lab/StepPlayer.vue'
import { useStepper } from '@/composables/useStepper.js'

const sp = (s) => s.split(' ')
const CODE = 'def f ( xs ) : total = 0 for x in xs : total += x return total'
const WORKLOADS = [
  { name: '代码编辑', prompt: sp(`把 total 改名 为 acc : ${CODE}`), ref: sp(CODE.replaceAll('total', 'acc')) },
  { name: 'RAG 摘录', prompt: sp('文档 1 : 光合作用 发生 在 叶绿体 中 , 需要 光照 和 二氧化碳 。 文档 2 : 线粒体 是 细胞 的 能量 工厂 。 问题 : 光合作用 在 哪里 发生 ?'),
    ref: sp('根据 文档 1 , 光合作用 发生 在 叶绿体 中 , 需要 光照 和 二氧化碳 。 文档 2 提到 线粒体 是 细胞 的 能量 工厂 。') },
  { name: '开放生成', prompt: sp('写 一句 关于 秋天 的 话'), ref: sp('枫叶 慢慢 染红 山坡 , 风 里 带着 桂花 香 , 傍晚 天 凉 了 。') },
]

const wl = ref(0), K = ref(6), maxN = ref(3)
const W = computed(() => WORKLOADS[wl.value])

// PromptLookupDrafter.propose: 末尾 n-gram (n 从大到小) 在 prompt + 已生成里最近一次出现, 抄后面 K 个
function propose(ctx) {
  for (let n = Math.min(maxN.value, ctx.length - 1); n >= 1; n--) {
    const tail = ctx.slice(-n).join('\u0001')
    for (let s = ctx.length - n - 1; s >= 0; s--) {                // ★ 从后往前
      if (ctx.slice(s, s + n).join('\u0001') === tail) return { n, s, draft: ctx.slice(s + n, s + n + K.value) }
    }
  }
  return { n: 0, s: -1, draft: [] }
}

// m07 的主循环 + accept_greedy: 第 1 个 token 来自 prefill, 之后每轮 1 次 forward
const rounds = computed(() => {
  const { prompt, ref: tgt } = W.value, P = prompt.length, out = [tgt[0]], list = []
  while (out.length < tgt.length) {
    const { n, s, draft } = propose([...prompt, ...out])
    let acc = 0
    while (acc < draft.length && out.length + acc < tgt.length && draft[acc] === tgt[out.length + acc]) acc++
    const start = out.length
    out.push(...draft.slice(0, acc))
    if (out.length < tgt.length) out.push(tgt[out.length])        // 纠错或 bonus token
    const src = s < 0 ? '' : s < P ? 'p' : 'o'
    list.push({ start, n, src, m0: src === 'p' ? s : s - P, draft, acc, end: out.length,
      note: n === 0 ? '查不到匹配: 不提草稿, 这一轮就是普通 decode, 产出 1 个。'
        : `末尾 ${n} 个 token 在${src === 'p' ? ' prompt ' : '已生成文本'}里找到, 抄来 ${draft.length} 个当草稿, 接受 ${acc} 个, 这一轮产出 ${out.length - start} 个。` })
  }
  return list
})
const stepper = useStepper(() => rounds.value.length, { interval: 1200 })
const cur = computed(() => ({ ...(rounds.value[Math.min(stepper.step.value, rounds.value.length - 1)] || { start: 1, end: 1, n: 0, draft: [], acc: 0, src: '', note: '' }), idx: Math.min(stepper.step.value, rounds.value.length - 1) }))
const shown = computed(() => W.value.ref.slice(0, cur.value.end))
const outCls = (i) => {
  const c = cur.value
  if (i >= c.start + c.acc && i < c.end) return 'bonus'
  if (i >= c.start) return 'acc'
  if (c.src === 'o' && i >= c.m0 && i < c.m0 + c.n) return 'hit'
  return i >= c.start - c.n && c.n > 0 ? 'hit' : ''
}
const speed = computed(() => W.value.ref.length / (rounds.value.length + 1))
const acceptRate = computed(() => {
  const prop = rounds.value.reduce((a, r) => a + r.draft.length, 0)
  return prop ? rounds.value.reduce((a, r) => a + r.acc, 0) / prop : 0
})
</script>

<style scoped>
.seq { max-width: 600px; }
.cap { font-size: 12px; color: var(--text-muted); margin: 8px 0 4px; }
.toks { display: flex; flex-wrap: wrap; gap: 4px; }
.tok { padding: 2px 6px; border: 1px solid var(--border); border-radius: 4px; font-size: 13px; }
.tok.hit { background: color-mix(in srgb, var(--warn) 30%, transparent); border-color: var(--warn); }
.tok.copy { border-color: var(--left); border-style: dashed; }
.tok.acc { background: color-mix(in srgb, var(--left) 25%, transparent); border-color: var(--left); }
.tok.bonus { background: color-mix(in srgb, var(--eye) 25%, transparent); border-color: var(--eye); }
.tok.rej { color: var(--danger); text-decoration: line-through; border-color: var(--danger); }
</style>
