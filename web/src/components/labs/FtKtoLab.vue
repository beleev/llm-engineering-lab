<!--
  KTO 实验台。公式与 llm_finetune/methods/kto.py:KTOLoss 相同 (β=0.5, z0 = 0 —— 本例 z0 全程被 clamp 成 0)。
  只讲一件事: 单条 👍 / 👎 也能训, 但好坏不均时, 两类样本的总推力 λ·n 必须拉平。
  曲线和推力是前端实时算的; 右侧表格是 python -m llm_finetune.run_finetune.kto.train_kto 的输出, 不随控件变化。
-->
<template>
  <LabFrame
    title="KTO — 单条 👍/👎 也能训, 但好坏 1:9 时 λ 必须跟着调"
    sub="- 模拟什么: KTO (Kahneman-Tversky Optimization) 无需同题成对偏好数据, 仅依赖单条回复上的二元「好/坏」反馈标签即可执行对齐训练。算法为每条回复计算隐式奖励 $r = \log(\pi/\pi_{\text{ref}})$ (即当前策略 policy 相对参考模型 ref 赋予该回复的对数概率增量); 正向好样本的优化目标是将 $r$ 推高至参考点 $z_0$ 之上, 负向坏样本则是将 $r$ 压低至 $z_0$ 之下 (本实验台取 $z_0 = 0$)。单样本梯度推力由逻辑斯谛 $\sigma$ 效用函数的导数给出, 远离参考点 $z_0$ 的已饱和样本导数趋近于 0 而自动停止推动。正负两类样本分别配有损失权重 $\lambda_D$ (好样本) 与 $\lambda_U$ (坏样本); 当批次内正负样本数量严重失衡时, 必须调节权重使两类的「类别权重乘以样本条数」总推力保持平衡, 否则占多数的一类会主导全局更新方向。
- 图上是什么: 自上而下由四部分组成:
  - 顶部 10 格样本条: 示意单个 batch 内的正负样本构成比例, 绿色 👍 代表好样本, 红色 👎 代表坏样本。
  - 中部梯度幅值曲线: 绘制单条样本在不同隐式奖励 $r$ 处的梯度推力大小 $|\partial v/\partial r| = \lambda\beta\sigma(1-\sigma)$。绿色曲线对应好样本 (固定 $\lambda_D = 1$), 红色曲线对应坏样本 ($\lambda_U$ 由滑杆设定), 温度系数固定为 $\beta = 0.5$ (与本仓库 kto.py 一致)。
  - 紫色竖线: 标示当前考察的隐式奖励位置 $r$。
  - 底部两根对比横条: 展示当前批次的正负两类总推力, 分别等于好样本条数与坏样本条数乘以各自在当前 $r$ 处的单样本推力。
- 可以做什么: 按界面顺序可执行以下交互:
  - 点击预设比例按钮: 在「好:坏 = 1:1 / 1:9」两种正负比例间一键切换。
  - 拖动「好样本条数 n_D」滑杆: 自定义 10 条样本中的正负样本配比。
  - 拖动「坏样本权重 λ_U」滑杆: 调整红色坏样本曲线与总推力高度。
  - 左右拖动紫色竖线 (或聚焦后使用方向键): 改变当前隐式奖励 $r$, 观察不同裕度区间内正负两类推力的相对强弱。
- 应该看到: 在 1:9 失衡配比且保持 $\lambda_U = 1$ 时, 坏样本总推力横条长度达到好样本的 9 倍。右侧读数下方的固定表格展示 train_kto.py 在留出集上的真实训练结果 (不随左侧滑杆变化)。各读数与表格列含义如下:
  - 「好样本 λ_D·n_D」与「坏样本 λ_U·n_U」: 正负两类各自的有效加权样本总量。
  - 「两者之比」: 正负有效总权重比值, 落入原论文推荐的 1 至 1.33 区间时显示为绿色。
  - 「整批净推力」: 指示当前批次合力偏向整体推高对数概率、整体压低还是基本平衡。
  - 表格「准确率」: 留出集上的成对偏好排序准确率。
  - 表格「log π(chosen)」: 模型赋予正确回复的对数似然 (数值越负代表生成该正确回复的概率越低)。
  - 表格「贪心 EM」: 留出集贪心解码的完全匹配率。"
    module="llm_finetune/methods/kto.py"
    run="python -m llm_finetune.run_finetune.kto.train_kto"
    :challenge="{
      ask: '点击「好:坏 = 1:9」并保持 $\\lambda_U$ 初始值不变:\n1. 此时坏样本的总梯度推力是好样本的多少倍?\n2. 在该失衡合力驱动下, 策略模型的输出分布会发生怎样的退化?\n3. 拖动 $\\lambda_U$ 滑杆使正负两类总推力恢复平衡, 对应的 $\\lambda_U$ 取值是多少?',
      answer: '结论:\n1. 坏样本的总梯度推力是好样本的 9 倍。\n2. 策略模型会无差别压低所有候选回复的生成概率, 导致输出分布崩塌。\n3. 将 $\\lambda_U$ 调至 0.11 (约 $1/9$) 时正负两类总推力达到平衡。\n机制解析:\n- 类别样本数的线性放大效应: 由单样本推力公式 $\\lambda\\beta\\sigma(1-\\sigma)$ 可知, 在相同隐式奖励幅度下, 单条样本的梯度幅值正比于类别权重 $\\lambda$。当取 $\\lambda_D = \\lambda_U = 1$ 时单条推力相等, 批次内 9 条坏样本叠加产生的负向压低总推力即为 1 条好样本正向推力的 9 倍。\n- 似然全面塌缩现象: 负向样本的目标是压低 $r = \\log(\\pi/\\pi_{\\text{ref}})$, 在缺乏足够正向拉力制衡时, 优化器最直接的下降路径是将概率质量转移至非正常 token 从而全面压低所有常规回复的对数似然。本仓库实测「KTO 1:9, 未重加权 λ」配置下, 模型完全丧失偏好区分度且贪心解码全部失效:\n  - log π(chosen) 从 -4.03 骤降至 -30.02。\n  - 留出集偏好准确率退化至随机水平 0.492。\n  - 贪心解码完全匹配率 EM 降至 0.000。\n- 重加权平衡条件: 令正负有效总权重满足 $\\lambda_D n_D \\approx \\lambda_U n_U$, 代入 $1\\times 1 = \\lambda_U\\times 9$ 解得 $\\lambda_U \\approx 0.11$。实测启用该重加权配置后, 留出集偏好准确率恢复至 0.988, 贪心 EM 达 0.254; 原论文建议将有效比值 $\\lambda_D n_D / (\\lambda_U n_U)$ 维持在 1 至 $4/3$ 区间内。\n- 单条二元反馈的适用优势: 在这份每个 prompt 仅含单条回复的 1:9 真实日志型数据上, DPO 因无法配对而生成 0 组有效训练对, KTO 则可直接利用单条好坏标签完成对齐训练。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: nD === 5 }" @click="nD = 5">好:坏 = 1:1</button>
        <button type="button" :class="{ active: nD === 1 }" @click="nD = 1">好:坏 = 1:9</button>
      </div>
      <LabSlider v-model="nD" label="好样本条数 n_D" :min="1" :max="5" unit=" / 10" />
      <LabSlider v-model="lamU" label="坏样本权重 λ_U" :min="0.05" :max="1.2" :step="0.01" :format="(v) => v.toFixed(2)" />
    </template>

    <div class="cells batch" style="grid-template-columns: repeat(10, 26px);">
      <span v-for="i in 10" :key="i" class="cell" :class="i <= nD ? 'ok' : 'bad'">{{ i <= nD ? '👍' : '👎' }}</span>
    </div>

    <svg ref="svgEl" viewBox="0 0 560 170" role="group" aria-label="单条样本的推力曲线">
      <line x1="30" y1="140" x2="550" y2="140" class="axis" />
      <line :x1="RX(0)" y1="20" :x2="RX(0)" y2="140" class="axis" />
      <text :x="RX(0) + 4" y="156" class="t">r = z0 = 0</text>
      <path :d="curve(LAM_D)" fill="none" stroke="var(--left)" stroke-width="2" />
      <path :d="curve(lamU)" fill="none" stroke="var(--danger)" stroke-width="2" />
      <text x="34" y="30" class="t" style="fill: var(--left)">好: λ_D·β·σ(1−σ)  (λ_D = 1)</text>
      <text x="34" y="46" class="t" style="fill: var(--danger)">坏: λ_U·β·σ(1−σ)  (λ_U = {{ lamU.toFixed(2) }})</text>
      <!-- 拖竖线 = 改隐式奖励 r -->
      <line :x1="RX(r)" y1="16" :x2="RX(r)" y2="140" stroke="var(--accent)" stroke-width="2" />
      <rect
        class="draggable focus-inset" :x="RX(r) - 8" y="8" width="16" height="136" fill="transparent"
        tabindex="0" role="slider" aria-label="隐式奖励 r" :aria-valuenow="r.toFixed(1)" aria-valuemin="-10" aria-valuemax="10"
        @pointerdown="start($event, { svg: svgEl, onMove: ({ x }) => setR((x - 290) / 26) })"
        @keydown.right.prevent="setR(r + 0.5)" @keydown.left.prevent="setR(r - 0.5)"
      />
      <text :x="RX(r) + 5" y="20" class="t">r = {{ r.toFixed(1) }}</text>
    </svg>

    <div class="push">
      <div v-for="b in pushes" :key="b.id" class="prow">
        <span class="pl">{{ b.name }}</span>
        <i class="pbar" :style="{ width: (b.v / pmax) * 100 + '%', background: b.color }" />
        <span class="mono pv">{{ b.v.toFixed(3) }}</span>
      </div>
    </div>

    <template #stats>
      <div class="kv"><span>好样本总权重 λ_D·n_D</span><b>{{ (LAM_D * nD).toFixed(2) }}</b></div>
      <div class="kv"><span>坏样本总权重 λ_U·n_U</span><b>{{ (lamU * nU).toFixed(2) }}</b></div>
      <div class="kv"><span>两者之比 (论文建议 1 到 1.33)</span><b :class="inBand ? 'good' : 'bad'">{{ balance.toFixed(2) }}</b></div>
      <div class="kv"><span>整批净推力</span><b :class="inBand ? 'good' : 'bad'">{{ balance > 4 / 3 ? '偏向推高' : balance < 0.99 ? '偏向压低' : '大致拉平' }}</b></div>
      <p class="tcap">train_kto.py 留出集 (实测, 不随左侧变化)</p>
      <table class="cmp mono">
        <thead><tr><th>配置</th><th>偏好准确率</th><th>log π(chosen)</th><th>贪心 EM</th></tr></thead>
        <tbody>
          <tr v-for="m in MEASURED" :key="m.name"><td>{{ m.name }}</td><td>{{ m.acc }}</td><td>{{ m.lpc }}</td><td>{{ m.em }}</td></tr>
        </tbody>
      </table>
      <p class="lab-note">计算开销权衡: 在相同的模型前向预算下, DPO 每步可处理 128 条带标签回复, 而 KTO 每步仅能处理 64 条带标签回复, 原因是 KTO 需消耗一半前向算力在随机错配样本上估计 KL 参考点 z0 (本例中 z0 经截断后取值为 0)。</p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { useDrag } from '@/composables/useDrag.js'
import { clamp, range } from '@/utils/labmath.js'

const BETA = 0.5, LAM_D = 1
// train_kto.py 输出表, 原样抄录
const MEASURED = [
  { name: 'SFT 起点', acc: '0.965', lpc: '−4.03', em: '0.332' },
  { name: 'DPO 成对', acc: '0.996', lpc: '−4.25', em: '0.121' },
  { name: 'KTO 1:1', acc: '0.992', lpc: '−3.73', em: '0.324' },
  { name: 'KTO 1:9, λ_U = 1/9', acc: '0.988', lpc: '−4.12', em: '0.254' },
  { name: 'KTO 1:9, 不调 λ', acc: '0.492', lpc: '−30.02', em: '0.000' },
]
const nD = ref(5), lamU = ref(1), r = ref(0)
const svgEl = ref(null)
const { start } = useDrag()
const setR = (v) => { r.value = clamp(Math.round(v * 2) / 2, -10, 10) }

const nU = computed(() => 10 - nD.value)
// ★ 要拉平的量是 λ·n, 不是 λ: λ_D·n_D / (λ_U·n_U)
const balance = computed(() => (LAM_D * nD.value) / (lamU.value * nU.value))
const inBand = computed(() => balance.value >= 0.99 && balance.value <= 4 / 3)   // 0.99: 滑杆步长 0.01 凑不出精确的 1/9

const sig = (x) => 1 / (1 + Math.exp(-x))
// |∂v/∂r|: 好 v = λ_D σ(β(r − z0)), 坏 v = λ_U σ(β(z0 − r)); 两者导数的大小形状一样, 只差 λ
const grad = (lam, x) => { const s = sig(BETA * x); return lam * BETA * s * (1 - s) }
const RX = (x) => 290 + x * 26
const GY = (g) => 140 - g * 800
const curve = (lam) => range(81).map((i) => {
  const x = -10 + i * 0.25
  return `${i ? 'L' : 'M'} ${RX(x).toFixed(1)} ${GY(grad(lam, x)).toFixed(1)}`
}).join(' ')
const pushes = computed(() => [
  { id: 'd', name: `好样本 × ${nD.value}`, v: nD.value * grad(LAM_D, r.value), color: 'var(--left)' },
  { id: 'u', name: `坏样本 × ${nU.value}`, v: nU.value * grad(lamU.value, r.value), color: 'var(--danger)' },
])
const pmax = computed(() => Math.max(...pushes.value.map((b) => b.v), 1e-9))
</script>

<style scoped>
.batch { margin-bottom: 12px; }
.batch .cell { width: 26px; height: 26px; font-size: 13px; }
.t { font-size: 11px; fill: var(--text-muted); font-family: "SF Mono", Menlo, monospace; }
.axis { stroke: var(--border-strong); stroke-dasharray: 3 3; }
.push { display: flex; flex-direction: column; gap: 6px; margin-top: 8px; }
.prow { display: grid; grid-template-columns: 90px 1fr 56px; gap: 8px; align-items: center; font-size: 12px; color: var(--text-muted); }
.pbar { display: block; height: 12px; border-radius: 2px; }
.pv { text-align: right; color: var(--text); }
.tcap { font-size: 11px; color: var(--text); margin-top: 4px; }
.cmp { width: 100%; border-collapse: collapse; font-size: 10px; }
.cmp th, .cmp td { border: 1px solid var(--border); padding: 3px 4px; text-align: left; color: var(--text-muted); }
.cmp th { color: var(--text); background: var(--bg-elev); }
</style>
