<!--
  LoRA 做 RL 实验台: 训练数据最多带来多少 bit, 适配器最多能存多少 bit。
  只讲一件事: SFT 每条样本带回 O(token 数) bit, 策略梯度每个 episode 只带回 O(1) bit (奖励 0/1 时至多 1 bit),
  所以同样的样本数, RL 需要的容量小得多, 秩 1 的 LoRA 可能就够; SFT 数据一多就会超出低秩适配器的容量。
  容量按 "约 2 bit / 参数" (LoRA Without Regret 引用 Allen-Zhu & Li) 估, 是长训练极限下的上界, 不是学习速度。
-->
<template>
  <LabFrame
    title="LoRA 做 RL — 每个 episode 只带回约 1 bit, 需要的容量很小"
    sub="- 模拟什么: 一个数量级的账。SFT 的每条样本按 token 逐个监督, 带回的信息随 token 数增长 (这里按每 token 约 1 bit 估); 策略梯度的每个 episode 只拿到一个优势值, 奖励是 0/1 时至多 1 bit。适配器能存的信息按每个参数约 2 bit 估 (LoRA Without Regret 引用 Allen-Zhu & Li 的结论), LoRA 参数量是 r·(d_in + d_out) 对所有被注入的线性层求和。
- 图上是什么: 上面两根对数刻度的横条。绿条是适配器容量, 橙条是数据最多带来的信息, 橙条比绿条长就表示数据超出容量。下面一排格子是秩 1 到 64, 绿色格子表示这个秩的容量装得下当前数据, 红色表示装不下, 带边框的是当前选中的秩。最下面是 Python 脚本的实测表, 它和上面的账不联动。
- 可以做什么: 用按钮切换训练方式 (SFT / RL) 和模型规模 (本库玩具 / Llama-3.1-8B); 点下面的格子选秩 r; 拖「样本数」和「每条回复 token 数」两个滑杆。
- 应该看到: 拖动 token 数滑杆时, SFT 的橙条跟着变长, RL 的橙条不动。Llama-3.1-8B 上用 MATH 量级的 RL 数据 (1 万题 × 32 次采样), 秩 1 的绿条已经比橙条长十几倍; 换成 SFT, 同样的样本数要 r=64 才装得下。本库玩具的默认设置 (脚本里 60 步 × 256 个 episode) 按这笔账秩 1 装不下, 但实测表里只训 320 个 norm 增益就拿到了最高的 RL pass@1, 说明账是很松的上界。右侧读数:
  - 「LoRA 参数」: 当前秩下所有被注入层的 r·(d_in + d_out) 之和。
  - 「数据 / 容量」: 大于 1 标红, 表示按这笔账适配器装不下。
  - 「装得下的最小秩」: 1 到 64 里第一个容量够用的秩。
  - 实测表不随左侧变化。"
    module="llm_finetune/methods/lora.py"
    run="python -m llm_finetune.run_finetune.lora_rl.train_lora_rl"
    :challenge="{
      ask: '选「本库玩具」和「RL」, 样本数保持 15,360 (脚本里 60 步 × 每步 256 个 episode), 秩选 1。\n1. 这笔账说秩 1 装得下吗?\n2. 对照实测表, RL pass@1 最高的是哪一臂, 它有多少可训参数? 标准 LoRA r=1 追上全参了吗?\n3. 账和实测矛盾吗?',
      answer: '结论: 账上装不下, 实测却是 320 个参数拿到了最高的 pass@1。两件事不矛盾, 因为这笔账算的是上界。\n1. 15,360 个 episode 按每个 1 bit 算是 15,360 bit; 秩 1 的 LoRA 有 2,432 个参数, 按每个 2 bit 算能存 4,864 bit。数据 / 容量 = 3.16, 要到 r=4 才装得下。\n2. 实测 RL pass@1 最高的是只训 norm (0.502), 只有 320 个参数, 账上只能存 640 bit。标准 LoRA r=1 只有 0.193, 全参是 0.324, 没有追上。\n3. 原因有两层:\n- 每个 episode 实际带回的信息远少于 1 bit: 奖励的熵本来不到 1 bit, 组内 8 条结果全相同的题优势为 0, 不同 episode 带回的信息还大量重复。60 步 GRPO 在这里主要学会了一件事, 把分布变尖 (各臂的贪心 EM 都没涨)。\n- 容量只是必要条件: 标准 LoRA r=1 只有 0.193, 加上 320 个 norm 增益才到 0.301。这个模型在 lm_head 之前有 RMSNorm, logits 的整体尺度由 norm 增益决定, 冻结它们的适配器即使容量够, 也很难在 60 步里把分布变尖。\n所以这笔账能说明 RL 需要的容量小, 放开哪些参数还得靠实测来定。',
    }"
  >
    <template #controls>
      <div class="row">
        <button v-for="(m, k) in MODES" :key="k" type="button" :class="{ active: mode === k }" @click="mode = k">{{ m }}</button>
        <button v-for="(p, k) in PRESETS" :key="k" type="button" :class="{ active: preset === k }" @click="choose(k)">{{ p.name }}</button>
      </div>
      <LabSlider v-model="logN" label="样本数 (SFT 条数 / RL episode 数)" :min="2" :max="7" :step="0.05" :format="(v) => fmtNum(Math.round(10 ** v))" />
      <LabSlider v-model="logT" label="每条回复 token 数" :min="0" :max="4" :step="0.05" :format="(v) => fmtNum(Math.round(10 ** v))" />
    </template>

    <svg viewBox="0 0 640 150" role="img" aria-label="适配器容量与数据信息量的对数刻度对比">
      <text v-for="t in ticks" :key="t" :x="X(t)" y="142" text-anchor="middle" class="t">10^{{ t }}</text>
      <line v-for="t in ticks" :key="'g' + t" :x1="X(t)" :x2="X(t)" y1="20" y2="128" stroke="var(--border)" stroke-dasharray="2 4" />
      <rect x="60" y="30" :width="X(Math.log10(capBits)) - 60" height="30" fill="var(--left)" />
      <rect x="60" y="78" :width="X(Math.log10(dataBits)) - 60" height="30" fill="var(--eye)" />
      <text x="56" y="50" text-anchor="end" class="t">容量</text>
      <text x="56" y="98" text-anchor="end" class="t">数据</text>
      <text :x="X(Math.log10(capBits)) + 6" y="50" class="t">{{ fmtNum(capBits) }} bit</text>
      <text :x="X(Math.log10(dataBits)) + 6" y="98" class="t">{{ fmtNum(dataBits) }} bit</text>
    </svg>

    <div class="cells ranks" role="group" aria-label="选择 LoRA 的秩">
      <button
        v-for="(r, i) in RANKS" :key="r" type="button" class="cell"
        :class="[fits(r) ? 'ok' : 'bad', { on: rank === r }]" :aria-pressed="rank === r" @click="rank = r"
      >r={{ r }}</button>
    </div>

    <table class="cmp">
      <caption>实测, 不随左侧变化 (train_lora_rl, sort 任务, 各配置取 lr 网格里最好的一臂, 单种子)</caption>
      <thead><tr><th></th><th>可训参数</th><th>RL 采样 pass@1</th><th>SFT 贪心 EM</th></tr></thead>
      <tbody>
        <tr v-for="row in MEASURED" :key="row[0]"><td>{{ row[0] }}</td><td class="mono">{{ row[1] }}</td><td class="mono">{{ row[2] }}</td><td class="mono">{{ row[3] }}</td></tr>
      </tbody>
    </table>

    <template #stats>
      <div class="kv"><span>LoRA 参数 (r={{ rank }})</span><b>{{ fmtNum(params) }}</b></div>
      <div class="kv"><span>数据 / 容量</span><b :class="ratio > 1 ? 'bad' : 'good'">{{ ratio < 0.01 ? ratio.toExponential(1) : ratio.toFixed(2) }}</b></div>
      <div class="kv"><span>装得下的最小秩</span><b :class="minRank && minRank <= rank ? 'good' : 'bad'">{{ minRank ? 'r=' + minRank : '64 也不够' }}</b></div>
      <div class="lab-note">
        <p>两个 bit 数都是上界: 一个 episode 的奖励熵不到 1 bit, 组内奖励全相同的题连这点信息也带不回来; 2 bit / 参数是长训练极限下的存储量, 不说明学得多快。这笔账只能说明数量级, 不能代替实测。</p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { fmtNum } from '@/utils/labmath.js'

const MODES = { rl: 'RL (每 episode ≤ 1 bit)', sft: 'SFT (每 token 约 1 bit)' }
// 每层被注入的线性层 [d_in, d_out]: q k v o gate up down (LoRA 注入全部 7 种, 含 MLP)
const PRESETS = {
  toy: { name: '本库玩具 (d=64, 2 层)', layers: 2, shapes: [[64, 64], [64, 32], [64, 32], [64, 64], [64, 192], [64, 192], [192, 64]], N: 15360, T: 7 },
  llama: { name: 'Llama-3.1-8B (32 层)', layers: 32, shapes: [[4096, 4096], [4096, 1024], [4096, 1024], [4096, 4096], [4096, 14336], [4096, 14336], [14336, 4096]], N: 320000, T: 1000 },
}
const RANKS = [1, 2, 4, 8, 16, 32, 64]
const BITS_PER_PARAM = 2

const mode = ref('rl'), preset = ref('toy'), rank = ref(1)
const logN = ref(Math.log10(PRESETS.toy.N)), logT = ref(Math.log10(PRESETS.toy.T))
const choose = (k) => { preset.value = k; logN.value = Math.log10(PRESETS[k].N); logT.value = Math.log10(PRESETS[k].T) }

const paramsAt = (r) => PRESETS[preset.value].layers * PRESETS[preset.value].shapes.reduce((s, [a, b]) => s + r * (a + b), 0)
const params = computed(() => paramsAt(rank.value))
const capBits = computed(() => BITS_PER_PARAM * params.value)
// ★ RL 的信息量与回复长度无关: 一个 episode 只有一个优势值
const dataBits = computed(() => 10 ** logN.value * (mode.value === 'rl' ? 1 : 10 ** logT.value))
const ratio = computed(() => dataBits.value / capBits.value)
const fits = (r) => BITS_PER_PARAM * paramsAt(r) >= dataBits.value
const minRank = computed(() => RANKS.find(fits))

const ticks = [2, 4, 6, 8, 10, 12]
const X = (lg) => 60 + (Math.min(Math.max(lg, 1), 12) - 1) / 11 * 560

const MEASURED = [
  ['SFT 起点', '—', '0.186', '0.562'],
  ['全参', '99,648', '0.324', '0.684'],
  ['只训线性层 (norm 冻结)', '98,304', '0.223', '0.672'],
  ['LoRA r=1', '2,432', '0.193', '0.629'],
  ['LoRA r=8', '19,456', '0.221', '0.637'],
  ['LoRA r=1 + norm', '2,752', '0.301', '0.625'],
  ['只训 norm', '320', '0.502', '0.602'],
]
</script>

<style scoped>
.t { font-size: 11px; fill: var(--text-muted); font-family: "SF Mono", Menlo, monospace; }
.ranks { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0 14px; }
.ranks .cell { min-width: 52px; cursor: pointer; }
.ranks .cell.on { outline: 2px solid var(--accent); }
.cmp { width: 100%; border-collapse: collapse; font-size: 12px; }
.cmp caption { text-align: left; color: var(--text-muted); padding-bottom: 4px; }
.cmp th, .cmp td { border-bottom: 1px dashed var(--border); padding: 4px 6px; text-align: right; color: var(--text); }
.cmp th:first-child, .cmp td:first-child { text-align: left; color: var(--text-muted); }
</style>
