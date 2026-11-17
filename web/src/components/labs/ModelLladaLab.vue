<!--
  LLaDA 迭代去遮实验台 (对应 llm_models/models/language_models/llada.py:LLaDA.sample)。
  只讲一件事: 每步 “全部预测 → 只留最有把握的 → 其余重新遮住”, 剩余 [MASK] 数按线性日程递减。
  玩具模型: 一个位置的置信度 = 自身难度 + 周围已定稿 token 数 (上下文越多越有把握), 采样/重遮逻辑与 Python 一致。
-->
<template>
  <LabFrame
    title="LLaDA 扩散语言模型 — 一次预测全部空位, 只留有把握的, 没把握的遮回去再想"
    sub="- 模拟什么: LLaDA 是一种扩散语言模型: 不像自回归那样从左到右一个一个写, 它从一整行 [MASK] 出发, 一步步去噪, 剩下的 [MASK] 数按线性日程递减: 第 s 步结束后剩 round(T · (1 − s / steps)) 个。这个玩具模型里, 一个位置猜对的概率等于它自身的难度加上「周围 2 格内已定稿的邻居数 × 0.12」(上下文越多越有把握); 采样和重遮的逻辑与 Python 的 LLaDA.sample 一致。每一步去噪做三件事:
        1. 用双向 Transformer 同时预测所有 [MASK] 位置。
        2. 只把最有把握的几个定稿。
        3. 其余重新遮回 [MASK] 留给下一步。
      - 图上是什么: 一共 12 个位置, 目标句子是「今天 天气 很 好 我们 一起 去 公园 散步 然后 喝 咖啡」。每一行是一步去噪之后的样子, 「初始」行全是 ▢ (就是 [MASK])。最下面「自回归」一行是对照组: 同样的串行步数里, 从左到右一次写 1 个的自回归模型能写出多少个 token。格子有四种状态, 见图例:
        - 本步新定稿的。
        - 之前已定稿的。
        - 本步预测过但置信度低被重新遮住的 (划掉的虚线框, 悬停能看到它当时的置信度)。
        - 定稿错了的 (红色, 再也改不回来)。
      - 可以做什么: 按界面顺序:
        - 「低置信度重遮」: LLaDA 的默认策略, 置信度 (模型给自己所选 token 的概率) 最低的 n 个被遮回去。
        - 「随机重遮」: 随机挑 n 个遮回去。
        - 「换一组」: 换每个位置的难度和随机数。
        - 拖「去噪步数 steps」: 改一共分几步 (1 到 12)。
        - 播放控件: 一步步看。
      - 应该看到: 先看三个现象, 再看右侧读数:
        - 低置信度重遮下: 先定下来的是容易的词, 它们成为邻居的上下文之后, 难的词在后面几步也猜对了, 错误数通常是 0。
        - 随机重遮下: 没把握的猜测也会被定稿, 红格子多。
        - 步数越少: 每步定稿的 token 越多, 它们之间互相看不到对方, 错误越多。
        - 「定稿错误数」: 最后一行红格子的个数。
        - 「串行前向次数」: LLaDA 跑了几次模型 (= steps), 对比自回归要跑 12 次。
        - 「token 前向总量」: 每次前向算几个位置加起来。LLaDA 没有 KV cache, 每步都重算全部 12 个, 所以是 steps × 12; 自回归有 cache 一共只算 12 个。
        - 「本步结束应剩 [MASK]」: 就是线性日程给出的数。"
    module="llm_models/models/language_models/llada.py"
    run="python -m llm_models.run_models.language_models.llada.infer_llada"
    :challenge="{
      ask: '步数设为 4, 先用「低置信度重遮」, 再切「随机重遮」, 多点几次「换一组」。先猜: 哪个定稿错误更多? 再把步数拖到 12 和 1, 错误数怎么变?',
      answer: '答案: 随机重遮错得多; 步数拖到 12 错误最少, 拖到 1 错误最多。\n为什么:\n- 随机重遮: 没把握的预测也可能被定稿。定稿的 token 置信度记为 $+\\infty$、永不重遮, 错了就一直错, 而且它还会成为邻居的「上下文」, 把邻居也带偏。\n- 低置信度重遮: 先定「显然」的位置 (置信度高的), 它们又成为别人的上下文, 难的位置留到信息最充分时再定。生成顺序由置信度决定, 而不是从左到右。train 脚本里 15 步低置信度重遮的填空准确率是 1.000, 随机重遮是 0.910。\n步数是质量和速度的旋钮:\n- steps = 12: 每步只定 1 个 token, 每个都在尽量多的上下文下决定, 最稳; 但要 12 次整段前向。双向注意力下任何位置每步都可能变, 没有 KV cache, 每次都重算全部 12 个位置, 总计算量是自回归的 12 倍。\n- steps = 1: 一次定完 12 个, 每个位置都在零上下文下瞎猜, 互相看不到对方的答案。\nLLaDA 的好处是串行步数可以少于 token 数, 代价是每一步比自回归贵。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: remask === 'low' }" @click="remask = 'low'">低置信度重遮 (low_confidence)</button>
        <button type="button" :class="{ active: remask === 'random' }" @click="remask = 'random'">随机重遮 (random)</button>
        <button type="button" @click="seed++">换一组</button>
      </div>
      <LabSlider v-model="steps" label="去噪步数 steps" :min="1" :max="12" />
      <StepPlayer :stepper="stepper" :label="`第 ${stepper.step.value} / ${steps} 步`" />
    </template>

    <div class="cells grid" :style="{ gridTemplateColumns: `52px repeat(${T}, minmax(38px, 1fr))` }">
      <template v-for="(row, s) in frames" :key="s">
        <span class="rl mono" :class="{ dimrow: s > stepper.step.value }">{{ s === 0 ? '初始' : '步 ' + s }}</span>
        <span
          v-for="(c, i) in row" :key="i" class="cell w" :class="[c.cls, { dim: s > stepper.step.value }]"
          :title="c.conf != null ? `置信度 ${c.conf.toFixed(2)}` : ''"
        >{{ c.text }}</span>
      </template>
      <span class="rl mono ar">自回归</span>
      <span v-for="i in T" :key="'ar' + i" class="cell w" :class="i <= arDone ? 'ok' : ''">{{ i <= arDone ? WORDS[i - 1] : '·' }}</span>
    </div>
    <p class="legend">
      <span class="cell w on">词</span> 本步新定稿
      <span class="cell w ok">词</span> 之前已定稿
      <span class="cell w ghost">词</span> 本步预测过, 但置信度低被重新遮住
      <span class="cell w bad">词</span> 定稿错了, 再也改不回来
    </p>

    <template #stats>
      <div class="kv"><span>定稿错误数 (最后一行的红格子)</span><b :class="errors === 0 ? 'good' : 'bad'">{{ errors }} / {{ T }}</b></div>
      <div class="kv"><span>串行前向次数 (LLaDA / 自回归)</span><b :class="steps < T ? 'good' : ''">{{ steps }} / {{ T }}</b></div>
      <div class="kv"><span>token 前向总量 (LLaDA / 自回归)</span><b :class="steps > 1 ? 'bad' : ''">{{ steps * T }} / {{ T }}</b></div>
      <div class="kv"><span>本步结束应剩 [MASK] (线性日程)</span><b>{{ Math.round(T * (1 - stepper.step.value / steps)) }}</b></div>
      <div class="lab-note">
        <p><Tex text="线性日程: 第 $s$ 步结束后剩 $\mathrm{round}(T \cdot (1 - s/\text{steps}))$ 个 [MASK]。" /> 比如 12 个位置分 4 步, 每步后剩 9、6、3、0 个。</p>
        <p>两条前向账:</p>
        <ul class="pts">
          <li><b>自回归:</b> 有 KV cache, 每步只算 1 个新 token, 共 {{ T }} 次 token 前向。</li>
          <li><b>LLaDA:</b> 注意力是双向的, 任何位置每步都可能变, 没法缓存, 每步重算全部 {{ T }} 个, 共 steps × {{ T }} 次。</li>
        </ul>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import Tex from '@/components/Tex.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import StepPlayer from '@/components/lab/StepPlayer.vue'
import { useStepper } from '@/composables/useStepper.js'
import { mulberry32, clamp, range } from '@/utils/labmath.js'

const WORDS = ['今天', '天气', '很', '好', '我们', '一起', '去', '公园', '散步', '然后', '喝', '咖啡']
const WRONG = ['昨天', '心情', '不', '坏', '他们', '单独', '回', '公司', '加班', '但是', '吃', '火锅']
const T = WORDS.length
const remask = ref('low'), steps = ref(4), seed = ref(1)

const frames = computed(() => {
  const r = mulberry32(seed.value * 7919)
  const base = range(T).map(() => 0.45 + r() * 0.35)               // 每个位置的先天难度
  let x = Array(T).fill(null)                                     // null = [MASK]
  const out = [x.map(() => ({ text: '▢', cls: '' }))]
  for (let s = 1; s <= steps.value; s++) {
    const masked = range(T).filter((i) => x[i] === null)
    const pred = {}
    masked.forEach((i) => {
      const ctx = [-2, -1, 1, 2].filter((d) => x[i + d] != null).length // 周围已定稿的 token 数
      const pTrue = clamp(base[i] + 0.12 * ctx, 0.05, 0.98)
      const ok = r() < pTrue
      const conf = ok ? pTrue : pTrue * 0.55                      // 猜错时模型自报的概率通常也偏低, 但不总是最低
      pred[i] = { ok, conf, rank: remask.value === 'low' ? conf : r() }
    })
    // ★ 线性日程: 本步结束后应剩 n 个 [MASK]; 把名次最低的 n 个重新遮住
    const n = Math.round(T * (1 - s / steps.value))
    const again = new Set([...masked].sort((a, b) => pred[a].rank - pred[b].rank).slice(0, n))
    const row = range(T).map((i) => {
      if (x[i] !== null) return { text: x[i].text, cls: x[i].ok ? 'ok' : 'bad' }
      const p = pred[i], text = p.ok ? WORDS[i] : WRONG[i]
      return again.has(i) ? { text, cls: 'ghost', conf: p.conf } : { text, cls: p.ok ? 'on' : 'bad', conf: p.conf }
    })
    x = x.map((v, i) => (v !== null || again.has(i) ? v : { text: pred[i].ok ? WORDS[i] : WRONG[i], ok: pred[i].ok }))
    out.push(row)
  }
  return out
})
const errors = computed(() => frames.value[frames.value.length - 1].filter((c) => c.cls === 'bad').length)

const stepper = useStepper(() => steps.value + 1, { interval: 800 })
stepper.step.value = steps.value
watch(frames, () => { stepper.pause(); stepper.step.value = steps.value })
const arDone = computed(() => Math.min(T, stepper.step.value))
</script>

<style scoped>
.grid { gap: 3px; min-width: 540px; }
.cell.w { min-width: 38px; height: 24px; font-size: 11px; font-family: inherit; }
.cell.ghost { color: var(--text-dim); border-style: dashed; text-decoration: line-through; }
.rl { font-size: 10px; color: var(--text-dim); align-self: center; }
.rl.dimrow { opacity: 0.35; }
.rl.ar { color: var(--eye); margin-top: 8px; }
.rl.ar ~ .cell { margin-top: 8px; }
.legend { display: flex; flex-wrap: wrap; align-items: center; gap: 6px 8px; margin-top: 10px; font-size: 11px; color: var(--text-muted); }
.legend .cell { min-width: 30px; height: 20px; }
</style>
