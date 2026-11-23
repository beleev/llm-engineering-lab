<!--
  token 级 vs 序列级重要性比率 (GRPO vs GSPO) 实验台。
  只讲一件事: 奖励是给整条序列的, GRPO 却给每个 token 各算一个 ρ_t 各自 clip —— 单个 token 的比率是单样本估计, 噪声很大;
  GSPO 用长度归一化的序列比率 s = exp(mean_t log ρ_t), 整条序列共用一个权重, 要 clip 就整条一起 clip。
-->
<template>
  <LabFrame
    title="GSPO — 重要性比率该按 token 算, 还是按整条序列算"
    sub="- 模拟什么: 标准 GRPO 的奖励信号定义在完整序列粒度上, 但重要性比率 $\rho_t = \pi_{\text{new}}/\pi_{\text{old}}$ 却按每个 token 独立计算并各自执行裁剪。单个位置的比率仅依赖该步采样命中的单一 token, 属于高方差单样本估计; 当局部对数概率波动较大时, 序列内部分 token 会被随机截断归零。GSPO (Group Sequence Policy Optimization) 将重要性采样统一至序列粒度: 对整条回复各位置的对数比率 $\log\rho_t$ 求算术平均得到 $\log s$ (等价于对序列概率比按长度取几何平均), 令整条序列共享同一重要性比率 $s$, 并以整条序列为单位统一判定保留或裁剪。
- 图上是什么: 图中 12 根柱体分别表示同一条回复内 12 个 token 的对数重要性比率 $\log\rho_t$ (取 0 代表新旧策略概率相等, 为正代表概率上升, 为负代表概率下降)。其余辅助参考元素如下:
  - 两条黄色水平线: GRPO 的逐 token 裁剪边界 $\log(1 \pm 0.2)$。
  - 灰色柱体: 沿有利优化方向越过对应黄色边界的 token, 表示在 GRPO 下触发局部裁剪而失去梯度。
  - 绿色水平线: GSPO 计算得到的序列级对数比率 $\log s$, 等于全部 12 根柱体高度的算术均值。
- 可以做什么: 按界面顺序可执行以下操作:
  - 切换「$\hat{A} \gt 0$ / $\hat{A} \lt 0$」按钮: 设定当前回复属于正优势还是负优势样本, 决定哪一侧越界会触发裁剪。
  - 点击「换一组」按钮: 重新采样一组随机逐 token 波动。
  - 拖动「token 级比率噪声」滑杆: 调节各位置对数比率围绕均值的离散程度。
  - 拖动「整体漂移」滑杆: 改变新旧策略在整条序列上的平均偏移量 (即柱体整体均值)。
  - 拖动「GSPO 的 ε」滑杆: 调节序列级重要性比率的裁剪窗口宽度。
  - 上下拖动任意柱顶 (或聚焦后使用方向键): 单独修改指定 token 的对数比率; 注意拖动噪声或整体漂移滑杆会重新生成全部 12 根柱体, 手动微调的单点位置不会保留。
- 应该看到: 增大局部比率噪声时, 越界变灰的柱体数量不断增加, 而代表序列均值的绿色横线基本保持稳定。右侧汇总读数含义如下:
  - GRPO 指标: 「被 clip 的 token」统计当前触发局部裁剪变灰的柱体根数。
  - GSPO 指标: 「序列比率 s」等于 $e^{\log s}$, 在零均值局部噪声下通常紧贴 1; 「整条序列」指示整条回复当前是完整保留梯度还是整体被裁剪。
  - 「不归一的 Πρ_t」: 展示不做长度几何平均而直接连乘全部 12 个 $\rho_t$ 所得的原始序列概率比, 用于观察其偏离 1 的剧烈程度。"
    module="llm_finetune/methods/grpo.py"
    run="python -m llm_finetune.run_finetune.grpo.train_grpo"
    :challenge="{
      ask: '将 token 级比率噪声拖至 0.3 以上, 依次观察并回答:\n1. 在 GRPO 下有多少个 token 触发了局部裁剪?\n2. 此时 GSPO 的序列级比率 s 偏离 1 的幅度有多大?\n3. 对比「不归一的 Πρ_t」读数: 若序列长度扩展至 1000 个 token, 未做长度归一化的连乘比率会出现什么现象?',
      answer: '结论:\n1. 噪声高于 0.3 时, GRPO 下通常有 3 至 4 个 token 越界触发裁剪 (即三四个柱体变灰)。\n2. 此时 GSPO 的序列级比率 $s$ 依然紧贴 1。\n3. 当序列长度达 1000 个 token 时, 未经长度归一化的连乘积 $\\prod\\rho_t$ 会发生指数级发散或下溢归零, 几乎必然触发裁剪。\n机制解析:\n- 逐 token 裁剪的随机撕裂效应: 每个 $\\rho_t$ 仅基于当前位置采样到的单个 token 计算, 单样本重要性权重方差较大。当局部噪声达 0.3 时, 往往有三四根柱体随机越界被截断为零梯度, 其余位置照常更新; 同一条回复共享同一个序列级优势 $\\hat{A}$, 却因随机波动导致部分位置更新而部分位置冻结, 破坏了序列级信用更新的一致性。\n- MoE 路由抖动放大效应: 在混合专家 (MoE) 架构中, 内层更新引起门控路由轻微切换便会导致单 token 概率剧烈跳变; GSPO 原论文指出这正是大规模 MoE 模型使用标准 GRPO 时发生训练崩溃的主因。\n- 序列几何平均的方差压缩性质: 序列级对数比率 $\\log s = \\frac{1}{T}\\sum_t \\log\\rho_t$ 对全序列 $T$ 个位置取均值, 使独立零均值噪声的标准差按 $1/\\sqrt{T}$ 缩放, 因此 $s$ 始终高度集中于 1 附近。相应地, GSPO 的序列级裁剪阈值 $\\varepsilon$ 需设为极小量级 (原论文取 3e-4 至 4e-4 量级, 本仓库 grpo.py 在短序列 gspo 配置中取 0.05), 一旦整体偏移越界则将整条序列统一截断。\n- 长度归一化的必要性: 若不取几何平均而直接使用原始序列似然比 $\\prod\\rho_t = \\exp(\\sum_t\\log\\rho_t)$, 对数项求和的方差随序列长度 $T$ 线性累积, 取指数后在长序列上必然指数发散至无穷大或坍缩至 0, 使裁剪区间完全失效。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: A > 0 }" @click="A = 1">Â &gt; 0</button>
        <button type="button" :class="{ active: A < 0 }" @click="A = -1">Â &lt; 0</button>
        <button type="button" @click="reseed">换一组</button>
      </div>
      <LabSlider v-model="noise" label="token 级比率噪声" :min="0" :max="0.6" :step="0.02" :format="(v) => v.toFixed(2)" />
      <LabSlider v-model="drift" label="整体漂移 (策略真的变了多少)" :min="-0.2" :max="0.2" :step="0.01" :format="(v) => v.toFixed(2)" />
      <LabSlider v-model="epsSeq" label="GSPO 的 ε (序列级)" :min="0.01" :max="0.2" :step="0.01" :format="(v) => v.toFixed(2)" />
    </template>

    <svg ref="svgEl" viewBox="0 0 640 300" role="group" aria-label="各 token 的对数重要性比率">
      <rect x="40" :y="sy(HI)" width="590" :height="sy(LO) - sy(HI)" fill="var(--warn)" opacity="0.08" />
      <line v-for="b in [HI, LO]" :key="b" x1="40" x2="630" :y1="sy(b)" :y2="sy(b)" stroke="var(--warn)" />
      <text x="632" :y="sy(HI) - 4" text-anchor="end" class="t" fill="var(--warn)">log(1+0.2): token 级上界</text>
      <text x="632" :y="sy(LO) + 13" text-anchor="end" class="t" fill="var(--warn)">log(1−0.2): token 级下界</text>
      <line x1="40" x2="630" :y1="sy(0)" :y2="sy(0)" stroke="var(--border-strong)" />
      <text x="4" :y="sy(0) + 4" class="t">ρ=1</text>
      <g v-for="(v, i) in logr" :key="i">
        <rect :x="bx(i)" :y="Math.min(sy(v), sy(0))" width="34" :height="Math.abs(sy(v) - sy(0))" :fill="clipped[i] ? 'var(--text-dim)' : 'var(--accent)'" :opacity="clipped[i] ? 0.45 : 0.85" />
        <!-- ★ 手放在单个 token 的比率上 -->
        <circle
          class="draggable" :cx="bx(i) + 17" :cy="sy(v)" r="8" :fill="clipped[i] ? 'var(--text-dim)' : 'var(--accent)'" stroke="var(--bg-card)" stroke-width="2"
          tabindex="0" role="slider" :aria-label="`token ${i} 的 log ρ`" :aria-valuenow="v" aria-valuemin="-0.9" aria-valuemax="0.9"
          @pointerdown="start($event, { svg: svgEl, onMove: ({ y }) => setTok(i, (150 - y) / SC) })"
          @keydown.up.prevent="setTok(i, v + 0.05)" @keydown.down.prevent="setTok(i, v - 0.05)"
        />
        <text :x="bx(i) + 17" y="292" text-anchor="middle" class="t">t{{ i }}</text>
      </g>
      <line x1="40" x2="630" :y1="sy(logS)" :y2="sy(logS)" stroke="var(--left)" stroke-width="2.5" />
      <text x="44" :y="sy(logS) - 5" class="t" fill="var(--left)">log s = 平均 = {{ logS.toFixed(3) }}</text>
    </svg>

    <template #stats>
      <div class="kv"><span>GRPO: 被 clip 的 token (变灰的柱子)</span><b :class="nClip ? 'bad' : 'good'">{{ nClip }} / {{ N }}</b></div>
      <div class="kv"><span><Tex text="GSPO: 序列比率 $s = e^{\log s}$" /></span><b>{{ Math.exp(logS).toFixed(3) }}</b></div>
      <div class="kv"><span>GSPO: 整条序列 (按 ε 判断)</span><b :class="seqClipped ? 'bad' : 'good'">{{ seqClipped ? '整条被 clip' : '整条保留' }}</b></div>
      <div class="kv"><span><Tex text="不做长度归一的 $\prod_t\rho_t$" /></span><b>{{ Math.exp(logS * N).toFixed(3) }}</b></div>
      <div class="lab-note">
        <p>裁剪机制仅拦截沿优势优化有利方向的越界更新:</p>
        <ul class="pts">
          <li><Tex text="$\hat{A} \gt 0$" /> 时拦 <Tex text="$\rho$" /> 过大 (好 token 涨太多)。</li>
          <li><Tex text="$\hat{A} \lt 0$" /> 时拦 <Tex text="$\rho$" /> 过小 (坏 token 压太狠)。</li>
        </ul>
        <p>零均值局部噪声仅扩大各位置柱体的离散幅度而不改变序列均值, 因此拖动噪声滑杆时绿色均值线基本静止, 而触发逐 token 裁剪的柱体数量却随噪声显著增加。</p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import Tex from '@/components/Tex.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { useDrag } from '@/composables/useDrag.js'
import { clamp, mulberry32, randn, range, sum } from '@/utils/labmath.js'

const N = 12, SC = 150, HI = Math.log(1.2), LO = Math.log(0.8)
const sy = (v) => 150 - clamp(v, -0.93, 0.93) * SC
const bx = (i) => 52 + i * 48

const A = ref(1), noise = ref(0.2), drift = ref(0.02), epsSeq = ref(0.05), seed = ref(1)
const svgEl = ref(null)
const { start } = useDrag()

// 单位噪声去均值: 这样"噪声"滑杆严格不改变平均值, 只改散布
const unit = computed(() => { const rand = mulberry32(seed.value * 1777); const z = range(N).map(() => randn(rand)); const m = sum(z) / N; return z.map((x) => x - m) })
const gen = () => unit.value.map((z) => clamp(drift.value + noise.value * z, -0.9, 0.9))
const logr = ref(gen())
watch([noise, drift, unit], () => { logr.value = gen() })
const reseed = () => { seed.value++ }
const setTok = (i, v) => { logr.value = logr.value.map((x, k) => (k === i ? clamp(Math.round(v * 100) / 100, -0.9, 0.9) : x)) }

// token 级: 每个 ρ_t 各自判断是否越界 (只拦有利方向)
const clipped = computed(() => logr.value.map((v) => (A.value > 0 ? v > HI : v < LO)))
const nClip = computed(() => clipped.value.filter(Boolean).length)
// ★ 序列级: s = (π_new(y)/π_old(y))^(1/|y|) = exp(mean_t log ρ_t)
const logS = computed(() => sum(logr.value) / N)
const seqClipped = computed(() => (A.value > 0 ? logS.value > Math.log(1 + epsSeq.value) : logS.value < Math.log(1 - epsSeq.value)))
</script>

<style scoped>
.t { font-size: 11px; fill: var(--text-muted); font-family: "SF Mono", Menlo, monospace; }
</style>
