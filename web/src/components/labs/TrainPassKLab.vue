<!--
  pass@k 的两个估计 (对应 llm_train/m21_llm_eval:pass_at_k)。
  只讲一件事: 采 n 个、c 个对时, 直接代 1-(1-c/n)^k 会系统性偏低; 无偏估计 1-C(n-c,k)/C(n,k) 的期望恰好等于真值。
  期望在前端按二项分布精确求和 (不是抽样)。数据集级的蒙特卡洛结果取自 m21 demo 第 [3] 段输出。
-->
<template>
  <LabFrame
    title="pass@k 的两种算法 — 把 c/n 直接代进公式, 为什么平均总是算低?"
    sub="- 模拟什么: 代码生成评测普遍采用 pass@k 指标, 定义为模型针对同一道编程题独立生成 k 份候选代码, 其中至少有一份通过全部单元测试的概率。实际评测通常对每道题先采样 n 份候选解 (要求 n 不小于 k), 运行测试集统计其中通过测试的正确样本数 c, 再由 n 与 c 估计该题的 pass@k。本实验台设定单道题目的真实单次通过率 p 已知, 此时正确样本数 c 服从二项分布 (独立采样 n 次且每次以概率 p 通过, c 取 0 至 n 之间各整数的概率均可精确解析求解); 前端直接将每个可能取值 c 的二项概率乘以对应估计量并精确求和, 给出两种估计公式的解析数学期望而非随机抽样均值。对比以下两种估计量:
  - 朴素代入公式: 将经验通过率 c/n 直接视作真实通过率 p 代入解析式 $1-(1-p)^k$。
  - 无偏估计公式: 采用超几何形式 $1-\binom{n-c}{k}/\binom{n}{k}$, 其中组合数 $\binom{n}{k}$ 表示从 n 个样本中选取 k 个的组合总数; 整式精确对应从已采样的 n 个候选解中不放回地随机抽取 k 个、至少抽中一个正确解的概率。「无偏」是指针对同一道题反复独立重采 n 个样本时, 单次算得的估计值虽有随机起伏, 但其长期数学期望严格等于真实 pass@k。
- 图上是什么: 横轴为题目的真实单次通过率 p (范围为 0 至 1), 纵轴为对应的 pass@k 概率值。
  - 绿色实线绘制理论真值曲线 $1-(1-p)^k$, 即已知真实 p 时独立尝试 k 次至少成功一次的准确概率。
  - 红色虚线绘制朴素公式的数学期望曲线, 即在该 p 下反复采样 n 个样本并将 c/n 代入公式所得结果的期望值。
  - 两条曲线之间的淡红色阴影带直观标示朴素公式系统性低估真值的负偏差幅度。
  - 无偏估计的期望曲线与绿色真值线在全区间严格重合, 图上不再重复绘制, 仅在右侧面板实时给出数值对比。
  - 竖直点线标示当前选定的真实通过率 p, 大绿点落在理论真值线上, 小红点落在朴素期望线上。
- 可以做什么: 控件从上到下依次为:
  - 拖动「每题采样数 n」调节单题生成的总样本数 (范围为 1 至 50)。
  - 拖动「k (pass@k 里试几次)」调节目标指标中的尝试次数 k (上限不超过 n; 若将 n 调至小于当前 k, k 会自动同步下调)。
  - 点击「n=20, k=1 / 5 / 10」三个预设按钮一键切换至 m21 离线实验所用的标准评测配置。
  - 在图上左右拖动竖线圆点, 或用 Tab 键聚焦后按左右方向键 (步进为 0.01) 调节真实单次通过率 p。
- 应该看到: 只要 k 大于 1, 红色虚线便在除两端点 (p 为 0 与 p 为 1) 外的整个开区间内严格位于绿线下方; 低估间隙在中小通过率区间最为显著, 随 p 趋近于 1 逐渐收窄。k 越大则低估间隙越宽; 总采样数 n 越大则经验比例 c/n 方差越小, 低估间隙随之收窄 (如当 n 为 50、k 为 10、p 为 0.15 时朴素偏差缩减至 −0.0314)。当 k 取 1 时两条曲线完全重合, 偏差恒为 0。在默认配置 (n 为 20、k 为 10、p 为 0.15) 下: 理论真值为 0.8031, 无偏估计期望为 0.8031, 朴素公式期望仅为 0.7247, 系统性偏差达 −0.0784。右侧读数含义如下:
  - 「真值」: 由当前真实通过率 p 直接解析算得的理论 pass@k。
  - 「无偏估计的期望」: 与真值绝对差小于 0.005 时显示绿色, 在任意合法参数组合下均恒保持绿色一致。
  - 「朴素公式的期望」: 红色虚线在当前 p 处的期望高度。
  - 「朴素偏差」: 等于朴素期望减去理论真值, 负号代表系统性低估, 低于 −0.005 时标红提示。"
    module="llm_train/m21"
    run="python -m llm_train.m21_llm_eval.demo"
    :challenge="{
      ask: '将 $k$ 调至 1 时, 图上两条曲线呈现什么关系? 若保持 $k$ 大于 1 并将总采样数 $n$ 下调至与 $k$ 相等, 曲线间隙与无偏估计期望又会如何变化?',
      answer: '结论:\n- 当 $k=1$ 时两条曲线完全重合, 系统性偏差恒为 0。\n- 当总采样数 $n$ 下调至与 $k$ 相等时, 朴素公式的低估间隙进一步扩大, 但右侧「无偏估计的期望」依然严格等于理论真值。\n机制解析:\n- $k=1$ 的退化情形: pass@1 即单次尝试直接通过的概率, 理论真值为 $p$。此时朴素公式 $1-(1-c/n)^1$ 化简后即为线性比例 $c/n$, 无偏公式 $1-\\binom{n-c}{1}/\\binom{n}{1} = 1-(n-c)/n$ 同样化简为 $c/n$。由于样本均值 $c/n$ 本身是二项参数 $p$ 的无偏估计, 两种算法在 $k=1$ 时均无偏差。m21 实测 $k=1$ 时的理论真值为 0.1883, 无偏估计与朴素估计均为 0.1884, 仅存的 0.0001 尾差源于有限次蒙特卡洛抽样波动。\n- $k \\gt 1$ 时朴素公式系统性偏低的原因: 函数 $1-(1-x)^k$ 在区间 [0, 1] 上是严格凹函数, 在小 x 区域斜率陡峭、随 x 增大斜率逐渐平缓。经验比例 $c/n$ 围绕真值 $p$ 上下对称波动时, 向上偏离带来的函数值增量小于向下等幅偏离带来的函数值跌幅, 取期望后必然低于 $1-(1-p)^k$。这正是概率论中的 Jensen 不等式: 对严格凹函数而言, 先代入随机变量再取期望严格小于先取期望再代入函数。尝试次数 k 越大则曲线弯曲程度越剧烈, 低估偏差随之放大: 在 $n=20$、$p=0.15$ 配置下, $k=5$ 时偏差为 −0.0381, $k=10$ 时偏差扩大至 −0.0784。\n- $n = k$ 的边界情形: 每题仅采样 k 个候选解时, 经验比例 $c/n$ 的估计方差显著增大, 导致朴素公式因凹性损失更多的期望高度。在 $n=k=10$、$p=0.15$ 时朴素期望降至 0.6484, 偏差扩大至 −0.1547, 约为 $n=20$ 时的两倍。此时无偏估计公式退化为指示变量「这 k 个样本中是否至少包含 1 个正确解」(有则取 1, 全错则取 0); 该伯努利指示变量的数学期望恰好定义上等于独立采样 k 次至少成功一次的真实概率 0.8031, 但单题单次评测只能输出非 0 即 1 的二值结果, 单题估计方差极大。\n因此代码评测的标准实践是令每题采样数 n 显著大于目标 k (例如采 $n=20$ 个样本以稳健估计 pass@10) 并配合无偏估计器计算。两种公式的本质差异可归纳为:\n- 无偏公式: 精确计算从已采得的 n 个样本中不放回抽取 k 个、至少命中一个正确解的超几何概率。\n- 朴素公式: 等同于以经验频率 c/n 作有放回独立抽样 k 次, 忽略了有限样本比例 c/n 自身携带的随机估计方差。',
    }"
  >
    <template #controls>
      <LabSlider v-model="n" label="每题采样数 n" :min="1" :max="50" />
      <LabSlider v-model="k" label="k (pass@k 里试几次)" :min="1" :max="n" />
      <div class="row">
        <button v-for="kk in [1, 5, 10]" :key="kk" type="button" :class="{ active: k === kk && n === 20 }" @click="(n = 20), (k = kk)">n=20, k={{ kk }}</button>
      </div>
    </template>

    <svg ref="svg" :viewBox="`0 0 ${W} ${H}`" role="group" aria-label="横轴真实通过率 p, 纵轴 pass@k: 绿色实线是真值, 红色虚线是朴素公式的期望; 竖线上的圆点可左右拖动改 p">
      <line :x1="X0" :x2="W - 10" :y1="py(0)" :y2="py(0)" class="axis" />
      <line :x1="X0" :x2="X0" :y1="py(0)" :y2="py(1)" class="axis" />
      <g v-for="t in [0, 0.25, 0.5, 0.75, 1]" :key="t">
        <text :x="px(t)" :y="py(0) + 14" class="tick" text-anchor="middle">{{ t }}</text>
        <text :x="X0 - 5" :y="py(t) + 3" class="tick" text-anchor="end">{{ t }}</text>
      </g>
      <text :x="W - 10" :y="py(0) - 6" class="tick" text-anchor="end">真实通过率 p →</text>
      <polygon :points="gap" class="gap" />
      <polyline :points="line(truth)" class="truth" />
      <polyline :points="line(naiveMean)" class="naive" />
      <g class="draggable" tabindex="0" role="slider" aria-label="真实通过率 p" :aria-valuenow="p"
        @pointerdown="start($event, { svg, onMove: ({ x }) => (p = clamp(Math.round(pxInv(x) * 100) / 100, 0, 1)) })"
        @keydown.right.prevent.stop="p = clamp(+(p + 0.01).toFixed(2), 0, 1)" @keydown.left.prevent.stop="p = clamp(+(p - 0.01).toFixed(2), 0, 1)">
        <line :x1="px(p)" :x2="px(p)" :y1="py(0)" :y2="py(1)" class="guide" />
        <circle :cx="px(p)" :cy="py(truth(p))" r="7" class="dot" />
        <circle :cx="px(p)" :cy="py(naiveMean(p))" r="5" class="dot2" />
      </g>
    </svg>

    <template #stats>
      <div class="kv"><span>p = {{ p.toFixed(2) }} 时的真值 (知道 p 直接算的 pass@k)</span><b>{{ truth(p).toFixed(4) }}</b></div>
      <div class="kv"><span>无偏估计的期望 (反复重采 n 个样本的平均, 应等于真值)</span><b :class="Math.abs(unbMean(p) - truth(p)) < 0.005 ? 'good' : 'bad'">{{ unbMean(p).toFixed(4) }}</b></div>
      <div class="kv"><span>朴素公式的期望 (把 c/n 当 p 代进去, 再取平均)</span><b :class="bias < -0.005 ? 'bad' : ''">{{ naiveMean(p).toFixed(4) }}</b></div>
      <div class="kv"><span>朴素偏差 (朴素期望减真值, 负数就是算低了)</span><b :class="bias < -0.005 ? 'bad' : 'good'">{{ bias >= 0 ? '+' : '' }}{{ bias.toFixed(4) }}</b></div>
      <div class="lab-note">
        <p>★ 工程实现中无偏公式并不直接计算大整数阶乘组合数, 而是将比值 C(n−c, k) / C(n, k) 约分化简为 c 项连乘形式: 1 − ∏(1 − k/i), 其中下标 i 从 n−c+1 遍历至 n。该写法与直接求组合数在代数上完全恒等, 且在大样本 n 下可避免大数溢出, 与 m21 demo.py 中的 pass_at_k 函数实现完全一致。当错误样本总数不足 k 个 (即 n−c 小于 k) 时, 任意抽取 k 个样本必含正确解, 直接返回 1。</p>
        <p>m21 demo 第 [3] 节数据集级蒙特卡洛实测基准 (固定读数, 不随滑杆变化): 评测集含 300 道题目, 各题真实单次通过率 p 独立抽自 Beta(0.5, 2) 偏态分布 (均值为 0.2, 即高难度题占多数), 每题采样 n 为 20 个候选解, 整套评测独立重复 400 轮取均值, 汇总结果如下:</p>
        <ul class="pts">
          <li><b>k=10:</b> 真值 0.5864, 无偏 0.5861, 朴素 0.5470, 偏低 0.0394。</li>
          <li><b>k=5:</b> 朴素偏低 0.0203。</li>
          <li><b>k=1:</b> 两者相同。</li>
        </ul>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { useDrag } from '@/composables/useDrag.js'
import { clamp, range } from '@/utils/labmath.js'

const n = ref(20), k = ref(10), p = ref(0.15), svg = ref(null)
const { start } = useDrag()
watch(n, (v) => { if (k.value > v) k.value = v })

// 与 llm_train/m21_llm_eval/demo.py:pass_at_k 相同
const passAtK = (nn, c, kk) => (nn - c < kk ? 1 : 1 - range(c).reduce((acc, j) => acc * (1 - kk / (nn - c + 1 + j)), 1))
// 二项分布概率 (对数域, n ≤ 50 足够稳)
const LOGFACT = range(52).reduce((acc, i) => (acc.push(i ? acc[i - 1] + Math.log(i) : 0), acc), []) // log(i!)
const lgamma = (x) => LOGFACT[x - 1] // log((x-1)!), x 为正整数
const binom = (nn, c, q) => {
  if (q <= 0) return c === 0 ? 1 : 0
  if (q >= 1) return c === nn ? 1 : 0
  return Math.exp(lgamma(nn + 1) - lgamma(c + 1) - lgamma(nn - c + 1) + c * Math.log(q) + (nn - c) * Math.log(1 - q))
}
const expect = (f, q) => range(n.value + 1).reduce((s, c) => s + binom(n.value, c, q) * f(c), 0)
const truth = (q) => 1 - (1 - q) ** k.value
const unbMean = (q) => expect((c) => passAtK(n.value, c, k.value), q)
const naiveMean = (q) => expect((c) => 1 - (1 - c / n.value) ** k.value, q) // ★ 凹函数 → 期望偏低
const bias = computed(() => naiveMean(p.value) - truth(p.value))

const W = 560, H = 240, X0 = 34
const px = (v) => X0 + v * (W - X0 - 14)
const pxInv = (x) => (x - X0) / (W - X0 - 14)
const py = (v) => 12 + (1 - v) * (H - 38)
const PS = range(101).map((i) => i / 100)
const line = (f) => PS.map((q) => `${px(q)},${py(f(q))}`).join(' ')
const gap = computed(() => [...PS.map((q) => `${px(q)},${py(truth(q))}`), ...[...PS].reverse().map((q) => `${px(q)},${py(naiveMean(q))}`)].join(' '))
</script>

<style scoped>
.axis { stroke: var(--border-strong); }
.tick { font-size: 10px; fill: var(--text-dim); font-family: "SF Mono", Menlo, monospace; }
.truth { fill: none; stroke: var(--left); stroke-width: 2.5; }
.naive { fill: none; stroke: var(--danger); stroke-width: 2; stroke-dasharray: 6 4; }
.gap { fill: var(--danger); opacity: 0.18; }
.guide { stroke: var(--text-dim); stroke-dasharray: 2 3; }
.dot { fill: var(--left); stroke: var(--bg); stroke-width: 2; }
.dot2 { fill: var(--danger); stroke: var(--bg); stroke-width: 2; }
</style>
