<!--
  forward KL vs reverse KL 实验台 —— off-policy 蒸馏与 on-policy 蒸馏背后的直觉。
  只讲一件事: 同一个"容量不够"的学生 (单峰), 最小化 KL(p‖q) 会摊开盖住老师的所有峰 (mode-covering),
  最小化 KL(q‖p) 会缩进其中一个峰 (mode-seeking)。期望在谁的样本上取, 决定了惩罚落在哪。
  两个 KL 都在网格上数值积分; forward 的最优解用闭式 (矩匹配) 给出, 与数值结果一致。
-->
<template>
  <LabFrame
    title="forward KL vs reverse KL — 容量不够的学生该盖住所有峰, 还是钻进一个峰"
    sub="- 模拟什么: 知识蒸馏中 student 模型容量通常小于 teacher, 难以完整表达 teacher 的多模态输出分布。本实验台将该容量受限拟合问题投影至一维连续空间: teacher 为具有两个高密度峰的双峰混合分布 $p$ (代表同一问题的两种合法解答范式), 而容量受限的 student 仅能取单峰高斯分布 $q$。正向与反向 KL 散度分别对应两类蒸馏采样范式, 期望在哪一分布的样本上求取直接决定了损失惩罚的作用区域:
  - forward KL ($\mathbb{E}_{x\sim p}[\log(p/q)]$): 在 teacher 分布采样下求期望, 对应离线蒸馏与常规 SFT (即在固定的 teacher 生成语料上做极大似然训练)。
  - reverse KL ($\mathbb{E}_{x\sim q}[\log(q/p)]$): 在 student 自身分布采样下求期望, 对应 on-policy 在线蒸馏 (由 student 自回归采样生成序列, 再由 teacher 逐 token 给出对数概率监督)。
- 图上是什么: 上方主图中紫色曲线为双峰 teacher 概率密度 $p$, 绿色曲线为单峰 student 概率密度 $q$, 峰顶绿色圆点支持直接拖拽调节。下方子图绘制当前选定 KL 散度的逐点被积函数曲线, 其与横轴围成的有向面积即等于总 KL 散度值, 局部凸起位置直观指示主要惩罚来源。
- 可以做什么: 按界面顺序可执行以下交互:
  - 切换「看 forward / reverse KL 的被积函数」按钮: 选择下方子图展示哪一种散度的逐点被积函数。
  - 点击「最小化 forward KL」按钮: 将 student 直接移至正向 KL 的全局最优解 (单峰高斯族内具有矩匹配解析解, 即均值与方差严格匹配 $p$)。
  - 点击「最小化 reverse KL」按钮: 通过数值梯度下降搜索反向 KL 的极小值点 (分别从当前参数与左右两峰初值出发优化并取最优解)。
  - 拖动「teacher 两峰间距」与「左峰的质量」滑杆: 调整双峰目标分布的不对称度与模态间隔。
  - 拖动峰顶绿色圆点 (或聚焦后使用方向键): 水平拖动改变高斯均值 $\mu$, 垂直拖动改变标准差 $\sigma$ (峰顶越高分布越窄)。
- 应该看到: 最小化 forward KL 得到横跨双峰的宽方差高斯分布, 将大量概率质量分配在两峰之间的低密度谷底; 最小化 reverse KL 则使高斯分布收缩锁定至其中一个单峰内部。右侧汇总读数含义如下:
  - 前两行: 实时显示正向与反向 KL 散度数值, 当前子图选中的散度项带有 ▸ 标记。
  - 「垃圾样本率」: student 采样落入 teacher 极低密度区域 (密度低于 teacher 峰值的 5%) 的总概率质量, 对应模型自回归生成时产生失真或非法输出的比率。
  - 「漏掉的 teacher 质量」: teacher 概率质量落入 student 极低密度区域的比例, 度量容量受限的 student 主动放弃覆盖合法解答模态的程度。"
    module="llm_finetune/methods/on_policy_distill.py"
    run="python -m llm_finetune.run_finetune.on_policy_distill.train_on_policy_distill"
    :challenge="{
      ask: '依次点击「最小化 forward KL」与「最小化 reverse KL」, 对比观察并回答:\n1. 两种目标函数下的最优 student 分布各自呈现何种几何位置与宽度?\n2. 两者对应的「垃圾样本率」表现有何差异?\n3. 在实际在线生成部署中, 哪一种分布形态更符合小容量模型的质量要求?',
      answer: '结论:\n1. 最小化 forward KL 得到的最优 student 是强行覆盖双峰的宽方差高斯分布; 最小化 reverse KL 得到的最优 student 则收缩锁定在单个主峰内部。\n2. forward KL 最优解的垃圾样本率很高; reverse KL 最优解的垃圾样本率接近 0, 代价是漏掉了另一侧次峰的概率质量。\n3. 面向生成部署时, 容量受限的小模型应优先追求后者 (高准确率单模态输出)。\n机制解析:\n- 正向 KL 的均值覆盖特性 (mode-covering): 正向散度 $\\mathrm{KL}(p\\|q) = \\mathbb{E}_{x\\sim p}[\\log(p/q)]$ 在 teacher 具有非零密度的所有区域检验 student 的概率覆盖度。一旦某处 $p(x) &gt; 0$ 而 $q(x) \\to 0$, 对数比 $\\log(p/q)$ 即趋向正无穷, 迫使单峰高斯拉大方差以同时覆盖左右两峰; 其副作用是将大量概率质量摊铺在两峰之间的低密度空白带, 导致极高的垃圾样本率。基于固定语料的离线蒸馏与 SFT 极大似然训练优化的正是 forward KL。\n- 反向 KL 的寻峰锁定特性 (mode-seeking): 反向散度 $\\mathrm{KL}(q\\|p) = \\mathbb{E}_{x\\sim q}[\\log(q/p)]$ 仅在 student 自身采样到达的区域评估 teacher 的认可度。对于 $q(x) \\approx 0$ 的未覆盖区域被积函数为零而不产生惩罚, 而一旦 $q(x) &gt; 0$ 落入 $p(x) \\approx 0$ 的非法区域则会遭受严厉惩罚, 从而驱动 student 集中收敛至单一高密度峰内, 确保生成的每条样本均符合 teacher 规范, 仅放弃另一套等价表述方式。\n- 在线蒸馏实证对比: on-policy 蒸馏 (由 student 在线自回归采样、teacher 逐 token 提供目标分布) 实质上近似优化 reverse KL, 使小容量模型以牺牲多样性为代价换取高生成可靠性。本仓库实测显示, 将离线蒸馏切换为 on-policy 蒸馏后, 生成样本合格率从 0.059 提升至 0.402, 同时主动放弃了 teacher 中的少数派答法模态。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: view === 'f' }" @click="view = 'f'">看 forward KL(p‖q) 的被积函数</button>
        <button type="button" :class="{ active: view === 'r' }" @click="view = 'r'">看 reverse KL(q‖p) 的被积函数</button>
      </div>
      <div class="row">
        <button type="button" @click="fitForward">最小化 forward KL (闭式: 矩匹配)</button>
        <button type="button" @click="fitReverse">最小化 reverse KL (数值下降, 多起点)</button>
      </div>
      <LabSlider v-model="sep" label="teacher 两峰间距" :min="1" :max="6" :step="0.2" :format="(v) => v.toFixed(1)" />
      <LabSlider v-model="w" label="左峰的质量" :min="0.1" :max="0.9" :step="0.05" :format="(v) => v.toFixed(2)" />
    </template>

    <svg ref="svgEl" viewBox="0 0 640 340" role="group" aria-label="teacher 与 student 的概率密度">
      <path :d="area(pPdf, py)" fill="var(--accent)" opacity="0.18" />
      <path :d="line(pPdf, py)" fill="none" stroke="var(--accent)" stroke-width="2" />
      <path :d="line(qPdf, py)" fill="none" stroke="var(--left)" stroke-width="2" />
      <line x1="20" x2="620" :y1="py(0)" :y2="py(0)" stroke="var(--border-strong)" />
      <text x="24" y="18" class="t" fill="var(--accent)">teacher p (双峰)</text>
      <text x="24" y="34" class="t" fill="var(--left)">student q = N(μ, σ²), μ = {{ mu.toFixed(2) }}, σ = {{ sigma.toFixed(2) }}</text>
      <!-- ★ 手放在学生分布上 -->
      <circle
        class="draggable" :cx="sx(mu)" :cy="py(peak)" r="10" fill="var(--left)" stroke="var(--bg-card)" stroke-width="2"
        tabindex="0" role="slider" aria-label="student 均值, 方向键左右改 μ 上下改 σ" :aria-valuenow="mu" aria-valuemin="-5" aria-valuemax="5"
        @pointerdown="start($event, { svg: svgEl, onMove: drag })"
        @keydown.right.prevent="mu = clamp(mu + 0.2, -5, 5)" @keydown.left.prevent="mu = clamp(mu - 0.2, -5, 5)"
        @keydown.up.prevent="sigma = clamp(sigma - 0.1, SMIN, SMAX)" @keydown.down.prevent="sigma = clamp(sigma + 0.1, SMIN, SMAX)"
      />

      <!-- 被积函数: 正的部分是惩罚 -->
      <text x="24" y="252" class="t">{{ view === 'f' ? 'p(x)·log(p/q) — 在 teacher 有质量的地方结算' : 'q(x)·log(q/p) — 只在 student 自己去的地方结算' }}</text>
      <path :d="area(integrand, iy)" :fill="view === 'f' ? 'var(--accent)' : 'var(--left)'" opacity="0.45" />
      <line x1="20" x2="620" :y1="iy(0)" :y2="iy(0)" stroke="var(--border-strong)" />
    </svg>

    <template #stats>
      <!-- 选中的那一行用 .sel 标出; .good / .bad 留给数值比较 -->
      <div class="kv" :class="{ sel: view === 'f' }"><span><Tex text="forward $\mathrm{KL}(p\,\|\,q)$" /></span><b>{{ kl.f.toFixed(3) }}</b></div>
      <div class="kv" :class="{ sel: view === 'r' }"><span><Tex text="reverse $\mathrm{KL}(q\,\|\,p)$" /></span><b>{{ kl.r.toFixed(3) }}</b></div>
      <div class="kv"><span>垃圾样本率 (学生生成时胡说的比例)</span><b :class="junk > 0.15 ? 'bad' : 'good'">{{ (junk * 100).toFixed(1) }}%</b></div>
      <div class="kv"><span>漏掉的 teacher 质量 (学生不会的答法)</span><b :class="miss > 0.15 ? 'bad' : 'good'">{{ (miss * 100).toFixed(1) }}%</b></div>
      <div class="lab-note">
        <ul class="pts">
          <li><b>垃圾样本率:</b> student 的样本落在 teacher 密度低于其峰值 5% 的区域的概率, 即生成时「胡说」的比例。reverse KL 直接惩罚这一项。</li>
          <li><b>漏掉的质量:</b> teacher 的样本落在 student 密度低于 teacher 峰值 5% 的区域的比例, 即学生「不会」的答法。forward KL 直接惩罚这一项。</li>
        </ul>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import Tex from '@/components/Tex.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { useDrag } from '@/composables/useDrag.js'
import { clamp, range, sum } from '@/utils/labmath.js'

const S = 0.6, SMIN = 0.5, SMAX = 4, DX = 0.05, XS = range(321).map((i) => -8 + i * DX)
const SQ = Math.sqrt(2 * Math.PI)
const sx = (x) => 320 + x * 37.5
const py = (d) => 220 - d / 0.8 * 190          // 密度面板: y∈[30,220]
const iy = (v) => 300 - clamp(v, -0.12, 0.35) / 0.35 * 40
const logN = (x, m, s) => -((x - m) ** 2) / (2 * s * s) - Math.log(s * SQ)

const sep = ref(4), w = ref(0.5), mu = ref(0), sigma = ref(1.2), view = ref('f')
const svgEl = ref(null)
const { start } = useDrag()

// teacher 的 log 密度用 logsumexp, 避免尾部下溢成 log(0)
const logP = computed(() => XS.map((x) => {
  const a = Math.log(w.value) + logN(x, -sep.value / 2, S), b = Math.log(1 - w.value) + logN(x, sep.value / 2, S)
  const m = Math.max(a, b)
  return m + Math.log(Math.exp(a - m) + Math.exp(b - m))
}))
const logQ = computed(() => XS.map((x) => logN(x, mu.value, sigma.value)))
const pPdf = computed(() => logP.value.map(Math.exp))
const qPdf = computed(() => logQ.value.map(Math.exp))
const peak = computed(() => 1 / (sigma.value * SQ))

const klOf = (lq) => {
  const q = lq.map(Math.exp)
  return {
    f: sum(pPdf.value.map((p, i) => p * (logP.value[i] - lq[i]))) * DX,   // ★ 期望在 p 上取
    r: sum(q.map((v, i) => v * (lq[i] - logP.value[i]))) * DX,            // ★ 期望在 q 上取
  }
}
const kl = computed(() => klOf(logQ.value))
const integrand = computed(() => (view.value === 'f'
  ? pPdf.value.map((p, i) => p * (logP.value[i] - logQ.value[i]))
  : qPdf.value.map((q, i) => q * (logQ.value[i] - logP.value[i]))))

const thr = computed(() => 0.05 * Math.max(...pPdf.value))
const junk = computed(() => sum(qPdf.value.filter((_, i) => pPdf.value[i] < thr.value)) * DX)
const miss = computed(() => sum(pPdf.value.filter((_, i) => qPdf.value[i] < thr.value)) * DX)

const drag = ({ x, y }) => {
  mu.value = clamp(Math.round((x - 320) / 37.5 * 20) / 20, -5, 5)
  const h = clamp((220 - y) / 190 * 0.8, 1 / (SMAX * SQ), 1 / (SMIN * SQ))   // 峰高 ↔ σ = 1/(h√2π)
  sigma.value = Math.round(100 / (h * SQ)) / 100
}
// forward KL 在高斯族内的最优解 = 矩匹配 (闭式)
const fitForward = () => {
  const m = (sep.value / 2) * (1 - 2 * w.value)
  mu.value = m
  sigma.value = clamp(Math.sqrt(S * S + (sep.value / 2) ** 2 - m * m), SMIN, SMAX)
  view.value = 'f'
}
// reverse KL 没有闭式, 而且有多个局部极小 (盖住两峰的宽高斯也是一个): 从当前位置和两个峰各下降一次, 取最小的
const fitReverse = () => {
  const f = (mm, ss) => klOf(XS.map((x) => logN(x, mm, ss))).r
  const descend = (m, s) => {
    for (let step = 0.4; step > 0.004; step /= 2) {
      for (let it = 0; it < 12; it++) {
        const cands = [[m, s], [m + step, s], [m - step, s], [m, clamp(s + step, SMIN, SMAX)], [m, clamp(s - step, SMIN, SMAX)]]
        const best = cands.reduce((a, c) => (f(...c) < f(...a) ? c : a))
        if (best[0] === m && best[1] === s) break
        ;[m, s] = best
      }
    }
    return [m, s, f(m, s)]
  }
  const best = [[mu.value, sigma.value], [-sep.value / 2, S], [sep.value / 2, S]].map((st) => descend(...st)).reduce((a, c) => (c[2] < a[2] - 1e-6 ? c : a))
  mu.value = clamp(Math.round(best[0] * 100) / 100, -5, 5); sigma.value = Math.round(best[1] * 100) / 100; view.value = 'r'
}

const pts = (ys, fy) => ys.map((v, i) => `${sx(XS[i]).toFixed(1)},${fy(v).toFixed(1)}`)
const line = (ys, fy) => 'M' + pts(ys, fy).join(' L')
const area = (ys, fy) => `M${sx(-8)},${fy(0)} L` + pts(ys, fy).join(' L') + ` L${sx(8)},${fy(0)} Z`
</script>

<style scoped>
.t { font-size: 11px; fill: var(--text-muted); font-family: "SF Mono", Menlo, monospace; }
.kv.sel { border-bottom: 1px solid var(--accent); }
.kv.sel > span { color: var(--text); }
.kv.sel > span::before { content: '▸ '; color: var(--accent); }
</style>
