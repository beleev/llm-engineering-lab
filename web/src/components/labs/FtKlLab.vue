<!--
  forward KL vs reverse KL 实验台 —— off-policy 蒸馏与 on-policy 蒸馏背后的直觉。
  只讲一件事: 同一个"容量不够"的学生 (单峰), 最小化 KL(p‖q) 会摊开盖住老师的所有峰 (mode-covering),
  最小化 KL(q‖p) 会缩进其中一个峰 (mode-seeking)。期望在谁的样本上取, 决定了惩罚落在哪。
  两个 KL 都在网格上数值积分; forward 的最优解用闭式 (矩匹配) 给出, 与数值结果一致。
-->
<template>
  <LabFrame
    title="forward KL vs reverse KL — 容量不够的学生该盖住所有峰, 还是钻进一个峰"
    sub="- 模拟什么: 蒸馏时 student 常常比 teacher 小, 装不下 teacher 的全部行为。这个台子把这件事压成一维: teacher 是双峰分布 $p$ (两种都对的答法), student 只能是单峰的高斯 $q$ (容量不够)。两种 KL 散度 (衡量两个分布差多远的量) 各代表一种训练方式, 期望在谁的样本上取, 决定了惩罚落在哪:
        - forward KL $= \mathbb{E}_{x\sim p}[\log p/q]$: 在 teacher 的样本上算, 对应离线蒸馏 / SFT (在 teacher 写的数据上做极大似然)。
        - reverse KL $= \mathbb{E}_{x\sim q}[\log q/p]$: 在 student 自己的样本上算, 对应 on-policy 蒸馏 (学生自己生成, teacher 逐 token 打分)。
      - 图上是什么: 上图紫色是 teacher 的密度曲线 (双峰), 绿色是 student 的密度曲线 (单峰), 峰顶的绿点可以拖。下图是当前选中的那种 KL 的逐点被积函数: 曲线下的面积就是 KL 的值, 哪里鼓起来, 惩罚就来自哪里。
      - 可以做什么: 按界面顺序:
        - 「看 forward / reverse KL 的被积函数」: 切换下图画哪一个。
        - 「最小化 forward KL」: 把学生放到 forward KL 最小的位置 (高斯族内有闭式解: 均值和方差匹配)。
        - 「最小化 reverse KL」: 用数值下降找 reverse KL 最小的位置 (从当前位置和两个峰各下降一次, 取最小的)。
        - 拖「teacher 两峰间距」和「左峰的质量」: 改 teacher 的形状。
        - 拖绿点 (或聚焦后按方向键): 左右改均值 $\mu$, 上下改宽度 $\sigma$ (越高越窄)。
      - 应该看到: forward 的最优解是一个摊开盖住两峰的宽高斯, 大量概率放在两峰之间的低谷; reverse 的最优解缩进其中一个峰。右侧读数:
        - 前两行: 两个 KL 的当前值, 选中的那个带 ▸ 标记。
        - 「垃圾样本率」: student 的样本落在 teacher 认为几乎不可能的区域 (密度低于 teacher 峰值的 5%) 的概率, 就是生成时「胡说」的比例。
        - 「漏掉的 teacher 质量」: teacher 的样本落在 student 几乎不会去的区域的比例, 就是学生「不会」的答法。"
    module="llm_finetune/methods/on_policy_distill.py"
    run="python -m llm_finetune.run_finetune.on_policy_distill.train_on_policy_distill"
    :challenge="{
      ask: '先点「最小化 forward KL」, 再点「最小化 reverse KL」, 然后回答:\n1. 两个最优学生分别落在哪?\n2. 各自的「垃圾样本率」是多少?\n3. 哪一个更像你希望小模型在生成时的表现?',
      answer: '答案:\n1. forward 最优的学生是一个盖住两峰的宽高斯, reverse 最优的学生缩进一个峰。\n2. forward 的垃圾样本率很高; reverse 的垃圾样本率接近 0, 但漏掉另一个峰。\n3. 生成时希望小模型像后者。\n为什么:\n- forward KL $= \\mathbb{E}_{x\\sim p}[\\log p/q]$, 这个式子在说: 在 teacher 有质量的每个地方, 检查 student 给了多少。teacher 有质量而 student 没有的地方, $\\log(p/q)$ 会变得极大, 所以学生被迫摊开盖住两个峰。代价是把大量概率放在两峰之间的低谷里, 那里 teacher 认为几乎不可能, 垃圾样本率很高。离线蒸馏 / SFT 优化的是 forward KL: 在 teacher 写的数据上做极大似然。\n- reverse KL $= \\mathbb{E}_{x\\sim q}[\\log q/p]$, 这个式子在说: 在 student 自己去的每个地方, 检查 teacher 认不认可。没去的地方完全不罚, 去了 teacher 不认可的地方重罚, 于是它缩进一个峰。它生成的样本都像 teacher 会写的, 但放弃了另一种答法。\n- on-policy 蒸馏 (学生采样、teacher 逐 token 打分) 优化的是 reverse KL: 容量小的学生因此少会一种答法, 但很少胡说。本仓库实测样本合格率从 0.059 (离线) 到 0.402 (on-policy), 代价是 teacher 的少数派答法被放弃。',
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
