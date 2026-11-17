<!--
  Aux-loss-free 负载均衡实验台 (对应 llm_models/models/moe/deepseekV3.py:DeepSeekMoE.update_routing_bias)。
  只讲一件事: 给每个专家一个不参与梯度的偏置 b, 只用来 “选谁”, 每步按 b += γ·sign(平均负载 − 自己的负载) 调 ——
  负载被拉平, 而门控权重 (用不带 b 的分数算) 一点没被扭曲。对照组: 传统 aux loss 是直接改路由分数。
-->
<template>
  <LabFrame
    title="Aux-loss-free 均衡 — 一个不收梯度的偏置, 只改「选谁」, 不改「信多少」"
    sub="- 模拟什么: 一个 MoE 层在训练。MoE 是把 FFN 换成 8 个小专家, 每个 token 只送给路由器打分最高的 2 个专家 (top-2 路由)。这里的路由器天生偏心: 它给编号靠前的专家打分总是偏高, 于是这几个专家接到的 token 太多, 别的专家闲着。aux-loss-free 均衡是 DeepSeek-V3 的办法: 给每个专家配一个偏置 b, 选专家时用「分数 + b」排序, 但乘到专家输出上的权重仍用原始分数; 每步训练后看谁过载就把谁的 b 减一点 (步长 $\gamma$), 谁欠载就加一点。对照组是传统 aux loss: 往总损失里加一项惩罚, 用梯度直接把热门专家的路由分数压低。
      - 图上是什么: 上面是柱图, 下面是折线图:
        - 柱子: 上面 8 根柱子是本步每个专家 (e0 到 e7) 接到的 token 数, 虚线是均值 (256 个 token × 2 个专家 / 8 = 64)。
        - 柱子颜色: 红色表示超过均值 20% (过载), 黄色表示低于均值 20% (欠载), 绿色正常。
        - 柱子底下的数字: 在偏置模式下是这个专家当前的偏置 b, 在 aux loss 模式下是 aux loss 累计把它的路由 logit 改了多少。
        - 折线图: 横轴是训练步 (0 到 120)。粗线是不均衡度 max/mean (左轴, 虚线 1.0 是完全均衡), 细线是你点选的那个专家的偏置轨迹 (右轴, 零线居中), 竖虚线是当前步。
      - 可以做什么: 按界面顺序:
        - 三个按钮切换均衡方法: 「Aux-loss-free (偏置 b)」「传统 aux loss」「不做均衡」。
        - 「换一批 token」: 换一组随机数。
        - 「偏置步长 γ」: 偏置模式下拖它 (0.001 到 0.1)。
        - 「aux 权重×学习率 η」: aux loss 模式下拖它, 它决定每步把路由 logit 改多狠。
        - 「路由器偏心程度」: 拖到 0 是公平的路由器, 拖到 2 偏心最厉害。
        - 点柱子: 点任意一根柱子, 折线图的细线就换成那个专家。
        - 播放控件: 逐步看。
      - 应该看到: 偏置模式下, 红柱子一步步矮下去、黄柱子长上来, 几十步内 8 根柱子齐平; 被点选专家的偏置先单调走到某个值, 然后在附近小幅抖动。右侧读数:
        - 「不均衡度 max/mean」: 最忙专家的负载除以均值, 1.0 是理想, 1.2 以内算好 (256 个 token 的随机抽样就有这么大的噪声)。
        - 「首次 < 1.3 的步数」: 均衡起效要几步。
        - 「路由分数被改动量」: 每个 token 对每个专家的路由分数, 和「不做任何均衡」时相比平均改了多少。偏置模式下恒为 0, aux loss 模式下大于 0, 两种方法的差别就在这一项。
        - 「最后 20 步负载 CV」: train_deepseek 脚本打印的同一个指标, 各专家负载的标准差除以均值, 越小越均衡。"
    module="llm_models/models/moe/deepseekV3.py"
    run="python -m llm_models.run_models.moe.deepseek.train_deepseek"
    :challenge="{
      ask: '「Aux-loss-free」和「传统 aux loss」都能把 max/mean 压到 1.2 左右 (256 个 token 的抽样噪声下限)。先猜: 右侧「路由分数被改动量」这一项, 两种方法分别是多少? 再把 γ 拖到 0.1 看会发生什么。',
      answer: '答案: 偏置模式下改动量恒为 0; aux loss 模式下是一个明显大于 0 的数。γ = 0.1 时负载来回震荡, 压不平。\n为什么:\n- Aux-loss-free: 偏置 b 只加在 top-k 的排序分数上 (决定谁上场), 门控权重 (上场后信多少) 仍取自原始 sigmoid 分数。原始分数没有任何改动, 所以改动量恒为 0, 均衡和语言建模目标互不干扰。\n- Aux loss: 往总损失里加 $\\alpha \\cdot E \\cdot \\sum_e f_e \\cdot P_e$ ($f_e$ 是专家 e 实际接到的 token 占比, $P_e$ 是路由给它的平均概率), 它的梯度直接把热门专家的路由分数往下压。负载是平了, 但「这个 token 本该更信任哪个专家」的信息也被改了。$\\alpha$ 大了伤模型效果, 小了又不均衡。\n- γ = 0.1 时偏置每步跳 0.1, 而 sigmoid 分数之间的差距本来也就 0.1 量级, 于是每一步都矫枉过正: 这一步过载的专家下一步变欠载, 负载来回震荡。\n偏置更新只用符号 (过载 −γ, 欠载 +γ), 步长和 batch 大小无关, 所以好调; 但必须足够小, DeepSeek-V3 用 0.001。',
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
