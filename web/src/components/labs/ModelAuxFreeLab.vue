<!--
  Aux-loss-free 负载均衡实验台 (对应 llm_models/models/moe/deepseekV3.py:DeepSeekMoE.update_routing_bias)。
  只讲一件事: 给每个专家一个不参与梯度的偏置 b, 只用来 “选谁”, 每步按 b += γ·sign(平均负载 − 自己的负载) 调 ——
  负载被拉平, 而门控权重 (用不带 b 的分数算) 一点没被扭曲。对照组: 传统 aux loss 是直接改路由分数。
-->
<template>
  <LabFrame
    title="Aux-loss-free 均衡 — 一个不收梯度的偏置, 只改「选谁」, 不改「信多少」"
    sub="- 模拟什么: 混合专家 (MoE) 层的训练期负载均衡过程。该层以 8 个并行专家子网络替代常规稠密 FFN, 每个输入 token 仅路由至得分最高的 2 个专家进行计算 (top-2 路由)。实验中设定初始路由器存在系统性倾向, 对编号靠前的专家天然给出偏高的打分, 导致少数专家严重过载而其余专家闲置。DeepSeek-V3 提出的无辅助损失均衡 (aux-loss-free) 机制为每个专家单独维护一个动态偏置项 $b$: 在执行 top-2 排序选拔专家时按「原始路由分 + 偏置 $b$」比较, 但专家输出的加权门控系数依然直接使用未加偏置的原始路由分; 每步训练结束后统计各专家实际负载, 对过载专家将 $b$ 下调固定步长 $\gamma$, 对欠载专家将 $b$ 上调 $\gamma$。对照基线为传统辅助损失 (aux loss): 在主任务损失外额外叠加负载均衡惩罚项, 通过反向传播梯度直接压低热门专家的路由器输出分。
- 图上是什么: 上方为当前步各专家负载柱状图, 下方为跨步演化折线图:
  - 负载柱图: 8 根柱子分别显示当前训练步专家 e0 至 e7 实际分得的 token 数量, 水平虚线标示理想均衡均值 (256 个 token 乘 2 个激活专家再除以 8, 即每专家期望承接 64 个)。
  - 柱体颜色: 超出均值 20% 标红 (过载), 低于均值 20% 标黄 (欠载), 处于合理区间内标绿。
  - 柱底数值: 在偏置模式下显示该专家当前的动态偏置 $b$, 在传统 aux loss 模式下显示惩罚梯度累计修改该专家路由 logit 的偏移量。
  - 演化折线图: 横轴为训练步数 (0 至 120); 粗实线记录全局不均衡度 max/mean (对应左侧纵轴, 水平虚线 1.0 代表绝对均匀), 细实线展示当前选中专家的偏置变化轨迹 (对应右侧纵轴, 零刻度线居中), 竖直虚线标示当前回放步。
- 可以做什么: 按界面从上到下依次操作:
  - 点击顶部三个按钮切换负载均衡方案: 「Aux-loss-free (偏置 b)」「传统 aux loss」与「不做均衡」。
  - 点击「换一批 token」重新采样随机输入分布。
  - 拖动「偏置步长 γ」调整偏置模式下的单步更新幅度 (取值范围 0.001 至 0.1)。
  - 拖动「aux 权重×学习率 η」调整传统辅助损失模式下每步梯度修正路由 logit 的强度。
  - 拖动「路由器偏心程度」调节初始打分倾斜度 (0 为初始完全公平, 2 为初始倾斜最严重)。
  - 点击上方任意一根专家柱, 将下方折线图的细轨迹线切换至对应专家。
  - 使用播放控件逐步观察收敛过程。
- 应该看到: 在「Aux-loss-free (偏置 b)」模式下, 红色过载柱逐轮回落、黄色欠载柱逐轮抬升, 数十步内 8 个专家的负载趋于齐平; 所选专家的偏置轨迹先单调收敛至平衡补偿值, 随后围绕该值小幅波动。右侧读数含义如下:
  - 「不均衡度 max/mean」为最繁忙专家负载与全体均值之比, 理想值为 1.0, 降至 1.2 以内即达良好均衡 (256 个 token 的有限样本抽样本身即存在此量级的统计涨落)。
  - 「首次 &lt; 1.3 的步数」衡量均衡机制收敛起效的速度。
  - 「路由分数被改动量」统计实际乘到专家输出上的门控权重相对于「不做均衡」基线的平均偏移量: 偏置模式下该值恒为 0, 传统 aux loss 模式下则显著大于 0, 直接体现两种机制是否干扰主模型门控语义。
  - 「最后 20 步负载 CV」对应 train_deepseek 脚本输出的变异系数指标 (各专家负载标准差除以均值), 数值越小代表稳态负载越均匀。"
    module="llm_models/models/moe/deepseekV3.py"
    run="python -m llm_models.run_models.moe.deepseek.train_deepseek"
    :challenge="{
      ask: '「Aux-loss-free (偏置 b)」与「传统 aux loss」均能将负载不均衡度 max/mean 压低至 1.2 左右 (接近 256 个 token 有限样本的统计涨落下限)。请先判断右侧「路由分数被改动量」读数在两种方法下分别为多少? 随后再将偏置步长 $\\gamma$ 拖至 0.1 观察收敛行为的变化。',
      answer: '在 Aux-loss-free 偏置模式下「路由分数被改动量」恒为 0, 而在传统 aux loss 模式下显著大于 0; 当将偏置步长 $\\gamma$ 调大至 0.1 时, 各专家负载会出现剧烈往复震荡而无法稳定收敛。\n具体机制原因如下:\n- Aux-loss-free 的解耦设计: 动态偏置 $b$ 仅叠加在用于 top-k 离散选拔的排序得分上 (仅决定路由分配集合), 而最终对所选专家输出做加权组合的门控权重仍直接取自未经修改的原始 sigmoid 亲和度得分。因此主干网络的门控权重偏移恒为 0, 负载均衡约束不会向路由器注入与语言建模目标相冲突的干扰梯度。\n- 传统 aux loss 的梯度耦合代价: 该方法在总训练损失中叠加惩罚项 $\\alpha \\cdot E \\cdot \\sum_e f_e \\cdot P_e$ (其中 $f_e$ 为专家 $e$ 实际分得的 token 比例, $P_e$ 为路由器赋予专家 $e$ 的平均概率), 反向传播梯度直接强行压低热门专家的原始路由得分。虽然分流结果趋于均匀, 但同时也扭曲了 token 与专家之间的真实语义亲和度权重, 形成辅助损失系数 $\\alpha$ 过大损害模型精度、过小则负载失衡的两难权衡。\n- 偏置步长过大的超调震荡: 当设 $\\gamma = 0.1$ 时, 单步偏置修正量已与不同专家原始 sigmoid 得分之间的天然差距 (同属 0.1 量级) 相当, 导致每步反馈均发生严重超调——当前步过载的专家被减去 0.1 后在下一步骤然变为严重欠载, 系统陷入周期性往复震荡。\n由于偏置更新仅依赖负载误差的符号方向 (过载减 $\\gamma$, 欠载加 $\\gamma$), 其更新幅度与训练 batch 规模解耦、超参数迁移稳定, 但步长必须设定得足够精细平滑 (DeepSeek-V3 实际采用 0.001)。',
    }"
  >
    <template #controls>
      <div class="row">
        <button v-for="m in MODES" :key="m.id" type="button" :class="{ active: mode === m.id }" @click="mode = m.id">{{ m.label }}</button>
        <button type="button" @click="seed++">换一批 token</button>
      </div>
      <LabSlider v-if="mode === 'bias'" v-model="gExp" label="偏置步长 γ" :min="-3" :max="-1" :step="0.1" :format="(t) => (10 ** t).toFixed(3)" />
      <LabSlider v-if="mode === 'aux'" v-model="eta" label="aux 权重×学习率 η" :min="0.05" :max="2" :step="0.05" :format="(t) => t.toFixed(2)" />
      <LabSlider v-model="skew" label="路由器偏心程度" :min="0" :max="2" :step="0.1" :format="(t) => t.toFixed(1)" />
      <StepPlayer :stepper="stepper" :label="`训练步 ${stepper.step.value}`" />
    </template>

    <div class="bars" role="group" aria-label="各专家负载, 点击选择">
      <div class="meanline" :style="{ bottom: 34 + (now.mean / loadMax) * 120 + 'px' }"><span class="mono">均值 {{ now.mean }}</span></div>
      <button v-for="(l, e) in now.load" :key="e" type="button" class="ex" :class="{ active: e === selE }" :aria-pressed="e === selE" @click="selE = e">
        <span class="mono n">{{ l }}</span>
        <div class="col" :class="l > now.mean * 1.2 ? 'hotc' : l < now.mean * 0.8 ? 'coldc' : 'okc'" :style="{ height: (l / loadMax) * 120 + 'px' }" />
        <span class="mono">e{{ e }}</span>
        <span class="mono b">{{ now.adj[e] >= 0 ? '+' : '' }}{{ now.adj[e].toFixed(2) }}</span>
      </button>
    </div>
    <p class="cap">柱子 = 本步每个专家接到的 token 数 (红 = 过载, 黄 = 欠载) · 底下的数 = {{ mode === 'aux' ? 'aux loss 累计把该专家的路由 logit 改了多少' : '该专家当前的路由偏置 b' }}</p>

    <!-- 两条线各用一根轴: 左轴读不均衡度, 右轴读偏置 (零线居中, 上下各到 ±最大值) -->
    <svg viewBox="0 0 560 120" role="img" aria-label="不均衡度和所选专家的偏置随训练步变化">
      <line x1="30" x2="515" :y1="iy(1)" :y2="iy(1)" class="ideal" /><text x="26" :y="iy(1) + 3" class="yl">1.0</text>
      <text x="26" :y="iy(imbMax) + 8" class="yl">{{ imbMax.toFixed(1) }}</text>
      <line x1="30" x2="515" :y1="ay(0)" :y2="ay(0)" class="zero" />
      <text v-for="t in [1, 0, -1]" :key="t" x="519" :y="ay(t * adjMax) + 3" class="yr">{{ t ? (t * adjMax).toFixed(adjMax < 0.1 ? 3 : 2) : '0' }}</text>
      <polyline :points="imbLine" class="imb" />
      <polyline :points="adjLine" class="adj" />
      <line :x1="sx(stepper.step.value)" :x2="sx(stepper.step.value)" y1="4" y2="112" class="cursor" />
    </svg>
    <p class="cap">
      <i class="ln imb-c" /> 不均衡度 max/mean (左轴, 粗线) · <i class="ln adj-c" /> e{{ selE }} 的{{ mode === 'aux' ? ' logit 改动量' : '偏置 b' }} (右轴, 细线)
    </p>

    <template #stats>
      <div class="kv"><span>不均衡度 max/mean (最忙专家 / 均值, 1.0 最均衡)</span><b :class="now.imb < 1.3 ? 'good' : 'bad'">{{ now.imb.toFixed(2) }}</b></div>
      <div class="kv"><span>首次 &lt; 1.3 的步数 (均衡起效要几步)</span><b>{{ firstOk < 0 ? '未达到' : firstOk }}</b></div>
      <div class="kv"><span>路由分数被改动量 (和不均衡时比, 每个分数平均改了多少)</span><b :class="now.distort < 1e-9 ? 'good' : 'bad'">{{ now.distort.toFixed(3) }}</b></div>
      <div class="kv"><span>最后 20 步负载 CV (std/mean, 越小越均衡)</span><b :class="cvTail < 0.15 ? 'good' : 'bad'">{{ cvTail.toFixed(3) }}</b></div>
      <p class="lab-note"><Tex :text="MODES.find((m) => m.id === mode).note" /></p>
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
import { mulberry32, randn, range, sum } from '@/utils/labmath.js'

const N = 256, E = 8, K = 2, STEPS = 120
const PREF = [1.5, 1.0, 0.5, 0.1, -0.1, -0.4, -0.7, -1.0] // 路由器对各专家的先天偏好 (logit)
const MODES = [
  { id: 'bias', label: 'Aux-loss-free (偏置 b)', note: '偏置 b 不进计算图、不收梯度, 每步训练之后看负载用符号更新: 过载的专家 $-\\gamma$, 欠载的 $+\\gamma$。选专家时用 $s+b$ 排序, 算门控权重时只用原始分数 $s$, 所以 b 不改门控权重, 也不出现在语言建模的梯度里。' },
  { id: 'aux', label: '传统 aux loss', note: '往总损失里加 $L_{\\text{aux}} = \\alpha \\cdot E \\cdot \\sum f_e \\cdot P_e$, 它的梯度直接压低热门专家的路由 logit。分数被改了, 门控权重跟着变: 均衡是靠扭曲「该信谁」换来的。' },
  { id: 'none', label: '不做均衡', note: '热门专家被选得多, 就学得更好, 于是更被偏爱。真实训练里这是个正反馈, 会滚雪球直到少数专家包揽所有 token (路由坍缩)。这里路由器不更新, 所以只看到偏心停在原地。' },
]
const mode = ref('bias'), gExp = ref(-2), eta = ref(0.5), skew = ref(1), seed = ref(1), selE = ref(0)
const sig = (x) => 1 / (1 + Math.exp(-x))

const frames = computed(() => {
  const gamma = 10 ** gExp.value
  let b = Array(E).fill(0), c = Array(E).fill(0)
  return range(STEPS + 1).map((step) => {
    const r = mulberry32(seed.value * 1000 + step)          // 每步一批新 token
    const load = Array(E).fill(0), gP = Array(E).fill(0)
    let distort = 0
    for (let n = 0; n < N; n++) {
      const raw = PREF.map((p) => randn(r) + skew.value * p)
      const s = raw.map((l, e) => sig(l + c[e]))
      // ★ 选人用 s + b, 但权重只用 s —— b 不碰门控
      const pick = range(E).sort((x, y) => s[y] + b[y] - (s[x] + b[x])).slice(0, K)
      pick.forEach((e) => load[e]++)
      s.forEach((v, e) => { gP[e] += v * (1 - v) / N; distort += Math.abs(v - sig(raw[e])) / (N * E) })
    }
    const mean = (N * K) / E
    const out = { load: [...load], adj: mode.value === 'aux' ? [...c] : [...b], mean, imb: Math.max(...load) / mean, distort }
    if (mode.value === 'bias') b = b.map((v, e) => v + gamma * Math.sign(mean - load[e]))
    if (mode.value === 'aux') {
      const g = c.map((_, e) => E * (load[e] / (N * K)) * gP[e])   // ∂(E·Σ f·P)/∂c_e, f 视为常数
      const gm = sum(g) / E                                        // 去掉公共分量: 整体平移不改变谁被选, 真实路由 (softmax) 对它也不敏感
      c = c.map((v, e) => v - eta.value * (g[e] - gm))
    }
    return out
  })
})

const stepper = useStepper(() => STEPS + 1, { interval: 90 })
stepper.step.value = STEPS
watch([mode, seed], () => { stepper.pause(); stepper.step.value = STEPS })
const now = computed(() => frames.value[stepper.step.value])
const loadMax = computed(() => Math.max(...frames.value.map((f) => Math.max(...f.load))))
const firstOk = computed(() => frames.value.findIndex((f) => f.imb < 1.3))
// 与 train_deepseek 打印的同一个指标: 最后 20 步负载的变异系数 std/mean 取平均
const cv = (load) => { const m = sum(load) / E; return Math.sqrt(sum(load.map((l) => (l - m) ** 2)) / E) / m }
const cvTail = computed(() => sum(frames.value.slice(-20).map((f) => cv(f.load))) / 20)

const imbMax = computed(() => Math.max(1.5, ...frames.value.map((f) => f.imb)))
const sx = (i) => 30 + (i / STEPS) * 485
const iy = (v) => 110 - ((v - 0.9) / (imbMax.value - 0.9)) * 86
const imbLine = computed(() => frames.value.map((f, i) => `${sx(i).toFixed(1)},${iy(f.imb).toFixed(1)}`).join(' '))
// 右轴: 所选专家的偏置, 按它自己的最大绝对值归一, 零线在正中
const adjMax = computed(() => Math.max(0.01, ...frames.value.map((f) => Math.abs(f.adj[selE.value]))))
const ay = (v) => 60 - (v / adjMax.value) * 48
const adjLine = computed(() => frames.value.map((f, i) => `${sx(i).toFixed(1)},${ay(f.adj[selE.value]).toFixed(1)}`).join(' '))
</script>

<style scoped>
.bars { position: relative; display: flex; gap: 6px; align-items: flex-end; height: 190px; padding-top: 10px; }
.ex { flex: 1; min-width: 30px; max-width: 60px; padding: 2px; min-height: 0; display: grid; justify-items: center; gap: 2px; font-size: 10px; background: transparent; color: var(--text-muted); border-color: transparent; }
.ex.active { border-color: var(--accent); background: var(--accent-soft); color: var(--text); }
.col { width: 70%; border-radius: 3px 3px 0 0; min-height: 1px; }
.okc { background: var(--left); } .hotc { background: var(--danger); } .coldc { background: var(--warn); }
.b { color: var(--eye); }
.meanline { position: absolute; left: 0; right: 0; border-top: 1px dashed var(--text-dim); font-size: 10px; color: var(--text-dim); pointer-events: none; }
.meanline span { position: absolute; right: 0; top: -14px; }
.cap { font-size: 11px; color: var(--text-dim); margin: 6px 0 10px; }
.ideal { stroke: var(--left); stroke-dasharray: 4 4; }
.yl { text-anchor: end; font-size: 9px; fill: var(--text-dim); }
.yr { text-anchor: start; font-size: 9px; fill: var(--eye); font-family: "SF Mono", Menlo, monospace; }
.zero { stroke: var(--eye); stroke-dasharray: 2 4; opacity: 0.6; }
.ln { display: inline-block; width: 16px; height: 0; border-top: 2px solid var(--accent); vertical-align: middle; }
.ln.adj-c { border-top: 1px solid var(--eye); }
.imb { fill: none; stroke: var(--accent); stroke-width: 2; }
.adj { fill: none; stroke: var(--eye); stroke-width: 1.2; }
.cursor { stroke: var(--text-muted); stroke-dasharray: 3 3; }
</style>
