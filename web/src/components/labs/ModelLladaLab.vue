<!--
  LLaDA 迭代去遮实验台 (对应 llm_models/models/language_models/llada.py:LLaDA.sample)。
  只讲一件事: 每步 “全部预测 → 只留最有把握的 → 其余重新遮住”, 剩余 [MASK] 数按线性日程递减。
  玩具模型: 一个位置的置信度 = 自身难度 + 周围已定稿 token 数 (上下文越多越有把握), 采样/重遮逻辑与 Python 一致。
-->
<template>
  <LabFrame
    title="LLaDA 扩散语言模型 — 一次预测全部空位, 只留有把握的, 没把握的遮回去再想"
    sub="- 模拟什么: LLaDA 是一类掩码扩散语言模型, 它不再像自回归模型那样从左到右逐个生成 token, 而是从全为 [MASK] 的序列出发逐步去噪, 剩余 [MASK] 数量按线性日程递减: 第 s 步结束时保留 $\mathrm{round}(T \cdot (1 - s / \text{steps}))$ 个 [MASK]。在本简化模型中, 单个位置预测正确的概率等于该位置的基础准确率加上「左右 2 格范围内已定稿邻居数 $\times 0.12$」(局部上下文越完整, 预测越准确); 采样与重遮掩逻辑与 Python 实现 LLaDA.sample 一致。每一步去噪依次执行三项操作:
  1. 用双向 Transformer 同时预测当前所有 [MASK] 位置的候选 token。
  2. 仅将置信度最高的一部分位置正式定稿。
  3. 将其余预测位置重新置回 [MASK], 留待后续步骤重算。
- 图上是什么: 序列共 12 个位置, 目标句为「今天 天气 很 好 我们 一起 去 公园 散步 然后 喝 咖啡」。自上而下每一行展示一步去噪后的序列状态, 「初始」行全为 ▢ (代表 [MASK]); 最下方「自回归」行为对照组, 展示在相同串行前向步数下, 从左到右每步生成 1 个 token 的自回归模型所能完成的进度。每个格子有四种状态 (对应图例):
  - 本步新定稿的 token。
  - 此前步骤已定稿的 token。
  - 本步参与预测但因置信度偏低而被重新遮掩的位置 (带删除线的虚线框, 鼠标悬停可查看其当时的预测置信度)。
  - 定稿结果错误的 token (红色高亮, 一旦定稿不再修改)。
- 可以做什么: 按界面控件顺序操作:
  - 「低置信度重遮」: LLaDA 的默认策略, 将当前预测置信度 (模型赋予所选 token 的概率) 最低的 n 个位置重新置为 [MASK]。
  - 「随机重遮」: 从待定位置中随机抽取 n 个重新置为 [MASK]。
  - 「换一组」: 重新随机生成各位置的基础难度与采样随机数。
  - 拖动「去噪步数 steps」: 调节总去噪步数 (范围 1 至 12)。
  - 播放控件: 逐步观察去噪与定稿演化过程。
- 应该看到: 对比不同重遮掩策略与步数下的生成表现及右侧读数:
  - 低置信度重遮下: 易预测的词率先定稿并成为相邻位置的双向上下文, 较难的词在后续步骤获得充足上下文后再定稿, 定稿错误数通常为 0。
  - 随机重遮下: 低置信度的猜测也可能被提前锁定, 红色错误格明显增多。
  - 步数越少: 单步被迫同时定稿的 token 越多, 彼此之间无法互为上下文参考, 定稿错误随之增加。
  - 「定稿错误数」: 最终输出行中红色错误格的数量。
  - 「串行前向次数」: LLaDA 调用模型前向的串行步数 (等于 steps), 而自回归对照组生成 12 个 token 固定需要 12 次。
  - 「token 前向总量」: 各步参与前向计算的 token 位置总数。LLaDA 因采用双向注意力而无法使用标准 KV cache, 每步均需重算全部 12 个位置, 总量为 $\text{steps} \times 12$; 自回归模型配合 KV cache 总计仅需计算 12 个位置。
  - 「本步结束应剩 [MASK]」: 线性去噪日程在当前步规定的剩余掩码数量。"
    module="llm_models/models/language_models/llada.py"
    run="python -m llm_models.run_models.language_models.llada.infer_llada"
    :challenge="{
      ask: '将去噪步数设为 4, 先后选择「低置信度重遮」与「随机重遮」, 并多次点击「换一组」观察: 哪种策略产生的定稿错误更多? 若进一步将步数分别调至 12 和 1, 定稿错误数会如何变化?',
      answer: '答案: 随机重遮产生的定稿错误明显更多; 去噪步数设为 12 时错误最少, 设为 1 时错误最多。\n原因分析:\n- 随机重遮: 缺乏把握的低置信度预测同样可能被提前锁定。已定稿 token 的置信度会被记为 $+\\infty$ 且后续不再重遮, 一旦定错便无法修正, 还会作为错误上下文误导相邻位置的后续预测。\n- 低置信度重遮: 优先锁定高置信度的确定位置, 使其成为周边位置的可靠上下文, 将歧义较大的位置留到上下文最充分的后半程再做决定, 生成顺序由模型置信度自适应决定而非固定从左到右。在 train 脚本实测中, 15 步低置信度重遮的填空准确率为 1.000, 而随机重遮仅为 0.910。\n去噪步数是生成质量与推理延迟之间的权衡旋钮:\n- $\\text{steps} = 12$: 每步仅定稿 1 个 token, 每个位置都在最完整的上下文下决策, 准确率最高, 但需要执行 12 次整段前向计算。由于双向注意力下各位置表示每步都会变化, 无法复用标准 KV cache, 每次前向均需重算全部 12 个位置, 总计算量是带缓存自回归模型的 12 倍。\n- $\\text{steps} = 1$: 单步一次性定稿全部 12 个位置, 各位置在零上下文条件下独立采样, 彼此无法看到对方的采样结果, 错误率最高。\n总体而言, LLaDA 的优势在于可将串行前向步数压缩至小于序列长度, 代价则是单步计算开销高于自回归解码。',
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
        <p><Tex text="线性去噪日程: 第 $s$ 步结束时剩余 $\mathrm{round}(T \cdot (1 - s/\text{steps}))$ 个 [MASK]。" /> 比如 12 个位置分 4 步, 每步后剩 9、6、3、0 个。</p>
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
