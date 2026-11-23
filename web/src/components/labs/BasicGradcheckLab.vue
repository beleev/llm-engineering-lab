<!--
  gradcheck ε 实验台 (对应 llm_basic/gradcheck.py:numeric_grad)。
  只讲一件事: 数值梯度的误差是 U 形的 —— ε 太大截断误差, ε 太小舍入误差。
  曲线是浏览器里真算出来的 (JS number = float64; float32 用 Math.fround 逐步舍入模拟)。
-->
<template>
  <LabFrame
    title="gradcheck 的 ε — 检查梯度时, 为什么步长不是越小越好"
    sub="- 模拟什么: 手写反向传播算子的数值梯度校验 (gradcheck) 原理与浮点误差权衡。反向求导代码若存在代数或符号错误通常不会抛出运行期异常, 只会静默导致模型训练发散或精度退化, 因此必须借助独立基准交叉核对。本实验针对 3 分类 softmax 加交叉熵网络中的单个权重元素 W[0][0] 在浏览器端实时重算误差曲线, 算法逻辑与源码 llm_basic/gradcheck.py 完全一致, 分为三步:
  1. 对待测标量权重 $w$ 施加幅度为 $\varepsilon$ 的微小正负扰动, 重新执行前向传播观测损失函数 $f(w)$ 的变化量。
  2. 利用有限差商公式 $(f(w + \varepsilon) - f(w - \varepsilon)) / (2\varepsilon)$ 近似算出仅依赖前向计算的数值梯度。
  3. 将该数值梯度与手写 backward 给出的解析梯度做相对误差比对。
- 图上是什么: 双对数坐标误差曲线图, 横轴为扰动步长 $\varepsilon$ (对数刻度, 由左至右递增), 纵轴为相对误差 $|\text{数值} - \text{解析}| / \max(|\text{数值}|, |\text{解析}|)$ (对数刻度, 数值越低表示吻合度越高), 图中同时标出以下参考元素:
  - 水平虚线: 对应 gradcheck.py 设定的断言通过阈值 1e-6, 相对误差落在该线下方即判定校验通过。
  - 垂直指示线与圆点: 标示当前选定的步长 $\varepsilon$ 及其对应的实时相对误差, 圆点颜色随是否低于通过阈值自动切换。
  - 底部刻度标记: 标出 gradcheck.py 默认采用的扰动步长 1e-5。
- 可以做什么: 操作两组精度与公式切换器、一个错误注入开关及步长滑杆:
  - 切换「中心差分 / 前向差分」: 更改数值梯度近似公式。中心差分在 $w$ 左右两侧对称采样, 泰勒展开截断误差按 $O(\varepsilon^2)$ 二阶收敛; 前向差分仅在右侧单向采样 $(f(w + \varepsilon) - f(w)) / \varepsilon$, 截断误差仅按 $O(\varepsilon)$ 一阶收敛。
  - 切换「float64 / float32」: 改变前向损失计算所使用的浮点数尾数位宽, 观察低精度浮点舍入对极小步长区间的冲击。
  - 点击「注入 bug」: 在反向传播实现中人为引入求导错误 (模拟漏减 $\mathrm{onehot}$ 项), 观察真实代码缺陷在全步长区间上的曲线形态。
  - 拖动图上竖线或底部滑杆: 连续改变扰动步长 $\varepsilon$。
- 应该看到: 正常实现下的相对误差曲线呈现典型的 U 形谷底结构。当右侧 $\varepsilon$ 取值过大时, 割线斜率偏离切线真实导数, 由高阶泰勒余项主导产生截断误差; 当左侧 $\varepsilon$ 取值过小时, $f(w + \varepsilon)$ 与 $f(w - \varepsilon)$ 数值极其接近, 两个相近浮点数相减发生灾难性相消, 有效数字被尾数舍入噪声淹没而产生舍入误差。在默认的 float64 加中心差分配置下, U 形谷底位于 $\varepsilon$ 约为 1e-5 处, 最低相对误差低至 1e-11 量级。开启「注入 bug」后, 曲线整体抬升为一条不随 $\varepsilon$ 缩减而下降的平坦高台: 真正的求导错误无法通过调节 $\varepsilon$ 消除, 而因步长选择不当引发的「假阳性失败」则会随 $\varepsilon$ 移向谷底而消失。右侧各项读数含义如下:
  - 「解析梯度」: 由手写 backward 函数直接算出的导数值。
  - 「数值梯度」: 在当前 $\varepsilon$ 下由有限差商算出的近似导数值。
  - 「相对误差」: 上述两项导数值之间的归一化偏差。
  - 「最佳 $\varepsilon$ / 最低误差」: 当前配置下整条 U 形曲线谷底对应的最优步长与最小相对误差。"
    module="llm_basic/gradcheck.py"
    run="python llm_basic/gradcheck.py"
    :challenge="{
      ask: '先预测: 将浮点精度从 float64 切换至 float32 后, U 形误差曲线的谷底会向哪个方向偏移, 且最低相对误差会退化至什么数量级? 这一现象对编写单元测试中的 gradcheck 有何工程启示?',
      answer: 'U 形曲线的谷底会显著向右偏移 (即必须选用更大的扰动步长 $\\varepsilon$ 才能达到最优近似), 且最低相对误差退化至 1e-6 到 1e-5 量级, 几乎无法稳定低于 1e-6 的断言通过阈值。\n- 误差竞争关系: 中心差分法的总近似误差由截断误差项 $O(\\varepsilon^2)$ (随 $\\varepsilon$ 增大而恶化) 与浮点舍入误差项 $O(\\delta / \\varepsilon)$ (随 $\\varepsilon$ 减小而恶化) 叠加而成, 其中机器精度 $\\delta$ 表示相邻两个可表示浮点数之间的相对间隔。\n- float64 双精度下: 机器精度 $\\delta$ 约为 1e-16, 令两项误差平衡的最优谷底位于 $\\varepsilon \\approx \\delta^{1/3}$ (约为 1e-5), 此时最低相对误差可达 1e-11 量级, 比 1e-6 的通过阈值低出整整 5 个数量级, 安全裕度极为充裕。\n- float32 单精度下: 机器精度 $\\delta$ 劣化至约 6e-8, 最优谷底被迫右移至 $\\varepsilon$ 约为 1e-3 附近, 且谷底最低误差仅能达到 1e-6 至 1e-5 之间; 在该精度下已难以可靠区分「算子求导实现正确但受限于浮点舍入」与「代码中潜伏着微小系数错误」。\n这直接决定了梯度校验的工程规范: 无论模型实际训练时采用何种混合低精度, 单元测试中的 gradcheck.py 都必须强制转换至 float64 双精度、固定采用二阶收敛的中心差分并将步长 $\\varepsilon$ 设为 1e-5。\n开启「注入 bug」开关后, 相对误差曲线变为一条与 $\\varepsilon$ 取值无关的水平高台 (恒定在 1.6 左右, 对应错误解析梯度与真实导数之间的固有偏差); 只要在双精度谷底附近调节 $\\varepsilon$ 仍无法令误差显著下降, 即可确诊反向算子内部存在真实逻辑错误。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: central }" @click="central = true">中心差分 (f(w+ε)−f(w−ε))/2ε</button>
        <button type="button" :class="{ active: !central }" @click="central = false">前向差分 (f(w+ε)−f(w))/ε</button>
      </div>
      <div class="row">
        <button type="button" :class="{ active: !f32 }" @click="f32 = false">float64</button>
        <button type="button" :class="{ active: f32 }" @click="f32 = true">float32</button>
        <button type="button" :class="{ active: bug }" :aria-pressed="bug" @click="bug = !bug">{{ bug ? '✓ 已注入 bug: backward 忘了减 onehot' : '注入 bug: backward 忘了减 onehot' }}</button>
      </div>
      <LabSlider v-model="k" label="步长 ε = 10^k" :min="-12" :max="-1" :step="0.25" :format="(t) => '1e' + t.toFixed(2)" />
    </template>

    <svg ref="svg" viewBox="0 0 560 300" role="group" aria-label="相对误差随 ε 变化的曲线, 竖线可拖动选 ε">
      <g class="grid">
        <template v-for="e in [-16, -12, -8, -4, 0]" :key="'y' + e">
          <line :x1="X0" :x2="X1" :y1="py(e)" :y2="py(e)" /><text :x="X0 - 6" :y="py(e) + 4" class="yl">1e{{ e }}</text>
        </template>
        <template v-for="e in [-12, -10, -8, -6, -4, -2]" :key="'x' + e">
          <line :x1="px(e)" :x2="px(e)" :y1="Y0" :y2="Y1" /><text :x="px(e)" :y="Y1 + 14" class="xl">1e{{ e }}</text>
        </template>
      </g>
      <line :x1="X0" :x2="X1" :y1="py(-6)" :y2="py(-6)" class="pass" />
      <text :x="X1 - 4" :y="py(-6) - 4" class="passl">通过线 1e-6</text>
      <polyline :points="pts" class="curve" :class="{ bad: bug }" />
      <text :x="px(-11.6)" :y="Y0 + 12" class="ann">← 舍入误差 ∝ δ/ε</text>
      <text :x="px(-1.2)" :y="Y0 + 12" class="ann end">截断误差 ∝ ε{{ central ? '²' : '' }} →</text>
      <line :x1="px(-5)" :x2="px(-5)" :y1="Y1 - 6" :y2="Y1" class="tick" /><text :x="px(-5)" :y="Y1 + 28" class="xl tickl">gradcheck.py 默认 1e-5</text>
      <g class="draggable" tabindex="0" role="slider" aria-label="拖动选择 ε" :aria-valuenow="k" aria-valuemin="-12" aria-valuemax="-1" :aria-valuetext="`ε = 1e${k.toFixed(2)}`"
        @pointerdown="start($event, { svg, onMove })" @keydown.left="k = clamp(k - 0.25, -12, -1)" @keydown.right="k = clamp(k + 0.25, -12, -1)">
        <line :x1="px(k)" :x2="px(k)" :y1="Y0" :y2="Y1" class="marker" />
        <rect :x="px(k) - 12" :y="Y0" width="24" :height="Y1 - Y0" fill="transparent" />
        <circle :cx="px(k)" :cy="py(Math.log10(cur.err))" r="7" class="dot" :class="cur.err < 1e-6 ? 'ok' : 'no'" />
      </g>
    </svg>

    <template #stats>
      <div class="kv"><span>解析梯度 (手写 backward 算的) {{ bug ? '(带 bug)' : '' }}</span><b>{{ ana.toFixed(8) }}</b></div>
      <div class="kv"><span><Tex text="数值梯度 (差商算的) @ $\varepsilon$" />={{ eps.toExponential(1) }}</span><b>{{ cur.num.toFixed(8) }}</b></div>
      <div class="kv"><span>相对误差 (通过线 1e-6)</span><b :class="cur.err < 1e-6 ? 'good' : 'bad'">{{ cur.err.toExponential(1) }}</b></div>
      <div class="kv"><span><Tex text="这条曲线的最佳 $\varepsilon$ / 谷底的最低误差" /></span><b>{{ best.eps.toExponential(0) }} / {{ best.err.toExponential(0) }}</b></div>
      <p class="lab-note">
        <Tex :text="bug ? '有 bug 时误差是一条约等于常数的高台: 无论 $\\varepsilon$ 取多少都过不了线, 因为解析梯度是错的, 和用什么步长去量它没关系。$\\varepsilon$ 选得不好造成的「假失败」会随 $\\varepsilon$ 改善, 真 bug 不会。'
          : cur.err < 1e-6 ? '通过: 手写 backward 和差商对上了。注意谷底左右两侧斜率不同: 左边 $\\varepsilon$ 每缩小 10 倍, 误差涨 10 倍 (舍入误差 $\\delta/\\varepsilon$); 右边按 $\\varepsilon$ 的幂次下降 (截断误差)。'
          : '没过线, 但解析梯度是对的, 只是 $\\varepsilon$ 选错了: 要么太大 (截断误差), 要么太小 (舍入误差)。所以 gradcheck 要固定 float64 + 1e-5, 否则分不清是步长选错还是代码写错。'" />
      </p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import Tex from '@/components/Tex.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { useDrag } from '@/composables/useDrag.js'
import { clamp, range } from '@/utils/labmath.js'

const central = ref(true), f32 = ref(false), bug = ref(false), k = ref(-5)
const svg = ref(null)
const { start } = useDrag()

// 被检查的函数: 3 类 softmax + CE, 只扰动 W[0][0]。r = 每步运算后的舍入 (float32 用 fround)
const Wm = [[0.6, -0.4], [-0.3, 0.8], [0.2, 0.1]], xin = [1.3, -0.7], Y = 0
const loss = (w00, r) => {
  const z = Wm.map((row, i) => r(r((i === 0 ? w00 : row[0]) * xin[0]) + r(row[1] * xin[1])))
  const m = Math.max(...z)
  const lse = r(m + r(Math.log(r(z.reduce((s, t) => r(s + r(Math.exp(r(t - m)))), 0)))))
  return r(lse - z[Y])
}
const p0 = (() => { const z = Wm.map((r) => r[0] * xin[0] + r[1] * xin[1]); const e = z.map(Math.exp); return e[0] / (e[0] + e[1] + e[2]) })()
// ★ 正确: (p0 − 1)·x0; bug 版忘了减 onehot
const ana = computed(() => (bug.value ? p0 : p0 - 1) * xin[0])

const measure = (kk) => {
  const r = f32.value ? Math.fround : (t) => t
  const e = r(10 ** kk), w = r(Wm[0][0])
  const num = central.value ? (loss(r(w + e), r) - loss(r(w - e), r)) / (2 * e) : (loss(r(w + e), r) - loss(w, r)) / e
  // ★ 分母与 gradcheck.py:check_op 相同: max(|g_a|, |g_n|, 1e-12)
  const err = Math.abs(num - ana.value) / Math.max(Math.abs(num), Math.abs(ana.value), 1e-12)
  return { num, err: Math.max(err, 1e-16) }   // 恰好为 0 时压到图的地板上
}
const ks = range(45).map((i) => -12 + i * 0.25)
const curve = computed(() => ks.map((kk) => ({ k: kk, ...measure(kk) })))
const cur = computed(() => measure(k.value))
const eps = computed(() => 10 ** k.value)
const best = computed(() => { const b = curve.value.reduce((a, c) => (c.err < a.err ? c : a)); return { eps: 10 ** b.k, err: b.err } })

const X0 = 52, X1 = 548, Y0 = 14, Y1 = 262
const px = (kk) => X0 + ((kk + 12) / 11) * (X1 - X0)
const py = (le) => Y1 - ((clamp(le, -16, 1) + 16) / 17) * (Y1 - Y0)
const pts = computed(() => curve.value.map((c) => `${px(c.k).toFixed(1)},${py(Math.log10(c.err)).toFixed(1)}`).join(' '))
const onMove = ({ x }) => { k.value = clamp(Math.round((((x - X0) / (X1 - X0)) * 11 - 12) * 4) / 4, -12, -1) }
</script>

<style scoped>
.grid line { stroke: var(--border); stroke-width: 1; }
.yl { text-anchor: end; font-size: 10px; fill: var(--text-dim); }
.xl { text-anchor: middle; font-size: 10px; fill: var(--text-dim); }
.tick { stroke: var(--eye); stroke-width: 2; }
.tickl { fill: var(--eye); }
.pass { stroke: var(--left); stroke-dasharray: 5 4; }
.passl { text-anchor: end; font-size: 10px; fill: var(--left); }
.curve { fill: none; stroke: var(--accent); stroke-width: 2; }
.curve.bad { stroke: var(--danger); }
.ann { font-size: 10px; fill: var(--text-muted); }
.ann.end { text-anchor: end; }
.marker { stroke: var(--text-muted); stroke-dasharray: 3 3; }
.dot { stroke: var(--bg); stroke-width: 2; }
.dot.ok { fill: var(--left); }
.dot.no { fill: var(--danger); }
</style>
