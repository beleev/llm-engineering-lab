<!--
  KV cache 量化分组实验台 (对应 llm_infer/m08_quantization/kv_quant.py:quantize_kv)。
  一件事: 同一个量化器, 只是"谁和谁共用一组 scale"不同 —— K 有固定离群通道所以按通道分组, V 没有所以按 token 分组 (KIVI)。
-->
<template>
  <LabFrame
    title="KV cache 量化 — 谁和谁共用一个 scale, 决定离群值害到谁"
    sub="- 模拟什么: 为降低 KV cache 的显存占用, 可将其中的浮点数压缩为 2~8 bit 整数存储 (即 KV 量化)。基本做法是令同一组数值共享区间 [min, max] 并等分为 $2^{\text{bits}}$ 个量化格点, 将每个数值舍入至最近格点, 相邻格点间距即为缩放因子 scale。一旦组内混入幅值显著偏大的离群值, 整组区间便会被拉宽、scale 变粗, 导致同组其余正常数值的舍入误差同步增大。因此量化分组方式直接决定了重建误差的大小。本实验通过一个 16 个 token × 12 个通道的 K 矩阵对比三种分组粒度。
- 图上是什么: 左侧为原始矩阵视图, 右侧为对应三种分组粒度的误差热力图:
  - 最左侧视图: 原始 K 矩阵, 每一行对应一个 token, 每一列对应一个特征通道 (hidden 维度); 正负值以两种颜色区分, 颜色越深代表绝对值越大。
  - 顶部一排小方块: 矩阵列头, 高亮的列代表「固定离群通道」, 即所有 token 在该通道维度上均呈现大幅值。
  - 右侧三幅视图: 三种分组方式量化还原后的逐元素误差 $|K - \hat{K}|$ 热力图, 颜色越红代表误差越大; 三幅图共享同一色标以便直接对比, 标题右侧的数值为对应分组的整体相对误差。
  - per-tensor: 整张矩阵共享同一个 scale。
  - per-token: 每一行 (每个 token) 独立计算一个 scale。
  - per-channel: 每一列 (每个通道) 独立计算一个 scale。
- 可以做什么: 按界面从上到下依次操作:
  - 点击「K 型」将矩阵设为包含两个固定离群通道 (第 3 列与第 9 列), 模拟真实模型中 K 矩阵的分布特征。
  - 点击「V 型」移除固定离群列, 改为令不同 token 具有不同的整行幅值, 模拟 V 矩阵的分布特征。
  - 点击「换一组」重新采样随机矩阵。
  - 拖动「比特数」修改量化位宽。
  - 拖动「离群通道幅度」调整离群列的幅值倍数 (单位为正常通道标准差 σ 的倍数)。
  - 点击最左侧矩阵顶部的任意列头, 手动将该列切换为离群通道或普通通道。
  - 悬停或点击任意单元格, 图表下方会显示该元素的原始值、三种分组量化反量化后的还原值及各自的 scale。
- 应该看到: 在 K 型分布下, per-channel 的量化误差最小, 因为离群通道仅拉宽其所在列的 scale, 其余正常列不受波及; 而 per-token 分组下每一行均包含离群通道, 导致所有行的 scale 均被拉粗, 热力图整体偏红。切换至 V 型分布后优劣次序反转, per-token 误差最小。右侧读数含义如下:
  - 三项「相对误差」均按 $\|K-\hat{K}\|_F / \|K\|_F$ 计算, 数值越小精度越高, 三者中最小者标绿、最大者标红。
  - 「per-token 误差是 per-channel 的几倍」展示两者的误差比值, 量化比特数越低该比值越大。"
    module="llm_infer/m08"
    run="python -m llm_infer.m08_quantization.demo"
    :challenge="{
      ask: '在 K 型分布与 INT4 量化设置下, 三种分组方式中哪一种相对误差最小? 请先作判断再观察实验结果。',
      answer: '在 K 型分布下 per-channel 分组的误差最小。\n核心原因在于 K 矩阵的离群值表现为「固定通道在所有 token 上均显著偏大」:\n- per-token 分组: 按每一行 (单个 token) 独立统计 min/max。由于每一行都包含这两个离群通道, 所有行的动态范围均被离群值拉宽、scale 变粗, 其余 10 个正常通道的数值只能挤在少数一两个量化格点上, 导致整体误差很大。\n- per-channel 分组: 按每一列 (单个通道) 独立统计 min/max。离群值被隔离在其所在的通道列内, 该列虽然 scale 较大, 但因其自身幅值本就很大, 相对误差可控; 其余正常通道各自使用精细的 scale, 量化误差很小。\n- per-tensor 分组: 整张矩阵共享单一 scale, 全局格点均被离群值拉粗, 精度最差。\n切换至 V 型分布后排名反转: V 矩阵不存在固定离群列, 而是不同 token 之间的整行幅值差异较大, 按行分组恰好能将大幅值 token 与小幅值 token 隔离, 因此 per-token 表现最优。此外 per-token 分组具备天然的流式工程优势: decode 阶段每生成一个新 token 即可立即独立量化写入 cache, 无需等待凑齐多步 token。\nKIVI 论文正是基于这一规律采用非对称策略: K 按通道分组 (per-channel), V 按 token 分组 (per-token), 分别匹配各自的分布特征。\n在 K 型分布下将「比特数」从 4 拖至 2 时差距进一步拉大: 位宽越低可用格点越少, scale 被离群值拉宽造成的精度损失越严重。Python demo 实测的 K 矩阵相对误差如下:\n- INT4: per-channel 为 0.026, per-token 为 0.18。\n- INT2: per-channel 为 0.134, per-token 达 1.02 (误差幅值已超过原始信号本身, 基本不可用)。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: kind === 'K' }" @click="setKind('K')">K 型 (固定离群通道)</button>
        <button type="button" :class="{ active: kind === 'V' }" @click="setKind('V')">V 型 (无固定离群通道)</button>
        <button type="button" @click="seed++">换一组</button>
      </div>
      <LabSlider v-model="bits" label="比特数" :min="2" :max="8" unit=" bit" />
      <LabSlider v-model="mag" label="离群通道幅度" :min="0" :max="20" :step="0.5" unit="×σ" />
    </template>

    <svg viewBox="0 0 572 205" role="group" aria-label="K 矩阵, 以及 per-tensor / per-token / per-channel 三种分组的量化误差热力图" @mouseleave="hover = null">
      <g v-for="(p, k) in PANELS" :key="p.id" :transform="`translate(${8 + k * 142}, 0)`">
        <text x="0" y="10" class="cap" :class="{ best: k > 0 && k - 1 === bestI, worst: k > 0 && k - 1 === worstI }">{{ p.label }}{{ k > 0 ? ' ' + q.errs[k - 1].toFixed(3) : '' }}</text>
        <template v-if="k === 0">
          <rect v-for="c in D" :key="`h${c}`" :x="(c - 1) * CS" y="16" :width="CS - 1" height="9" tabindex="0" role="button"
            :aria-label="`通道 ${c - 1}: 设为或取消离群通道`" :aria-pressed="outs.has(c - 1)" :class="['colhead', { on: outs.has(c - 1) }]"
            @click="toggle(c - 1)" @keydown.enter="toggle(c - 1)" />
        </template>
        <g v-for="t in T" :key="t">
          <rect v-for="c in D" :key="c" :x="(c - 1) * CS" :y="28 + (t - 1) * CS" :width="CS - 1" :height="CS - 1"
            :style="{ fill: k === 0 ? valColor(q.K[t - 1][c - 1]) : heat(Math.abs(q.K[t - 1][c - 1] - q.hats[k - 1][t - 1][c - 1]) / q.eMax, 'var(--danger)') }"
            :class="['c', { hov: hover && hover.t === t - 1 && hover.c === c - 1, clk: k === 0 }]"
            @mouseenter="hover = { t: t - 1, c: c - 1 }" @click="tap(k, t - 1, c - 1)" />
        </g>
      </g>
    </svg>
    <p class="read mono">{{ readout }}</p>

    <template #stats>
      <div v-for="(n, i) in NAMES" :key="n" class="kv">
        <span>{{ n }} 分组的相对误差</span><b :class="i === bestI ? 'good' : i === worstI ? 'bad' : ''">{{ q.errs[i].toFixed(4) }}</b>
      </div>
      <div class="kv"><span>per-token 误差是 per-channel 的几倍</span><b>{{ (q.errs[1] / q.errs[2]).toFixed(1) }}×</b></div>
      <div class="lab-note">
        <p>★ 三种分组只差一件事: 在哪个范围内统计 min/max。per-tensor 看整张矩阵, per-token 看一行, per-channel 看一列。于是 scale 的组数分别是 1 / {{ T }} / {{ D }}: 组越多, 每组的范围越贴合自己的数, 但要多存几个 scale。</p>
        <p><Tex text="三种分组方式均采用同一非对称 RTN (round-to-nearest, 最近邻舍入) 量化器: 量化过程为 $q = \mathrm{round}((x - \text{lo})/\text{scale})$, 反量化还原为 $\hat{x} = q \cdot \text{scale} + \text{lo}$。相对误差定义为 $\|K - \hat{K}\|_F / \|K\|_F$, 即误差矩阵与原矩阵的 Frobenius 范数之比; 0.1 表示误差幅值约占原始信号的 10%。" /></p>
        <p>实验中的 V 型采用简化的玩具分布假设: 不设固定离群列, 各 token 整行的幅值服从对数正态分布以模拟行间幅度差异。真实模型中 V 的分布更为复杂, 但「不存在固定离群通道列」这一核心特征同样成立。</p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import Tex from '@/components/Tex.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { argmax, clamp, heat, mulberry32, randn, range } from '@/utils/labmath.js'

const T = 16, D = 12, CS = 11
const NAMES = ['per-tensor', 'per-token', 'per-channel']
const PANELS = [{ id: 'k', label: 'K 矩阵 (点列头切换离群通道)' }, ...NAMES.map((n) => ({ id: n, label: `${n} 误差` }))]
const kind = ref('K'), bits = ref(4), mag = ref(8), seed = ref(1), hover = ref(null)
const outs = ref(new Set([3, 9]))
const toggle = (c) => { const s = new Set(outs.value); s.has(c) ? s.delete(c) : s.add(c); outs.value = s }
const tap = (k, t, c) => { hover.value = { t, c }; if (k === 0) toggle(c) }   // 点一下也能看 scale, 触屏没有悬停
const setKind = (k) => { kind.value = k; outs.value = new Set(k === 'K' ? [3, 9] : []) }

// 同一个非对称 min/max 量化器, 输入是"共用一组 scale 的那些数"
const quant = (xs, b) => {
  const lo = Math.min(...xs), L = 2 ** b - 1, scale = Math.max((Math.max(...xs) - lo) / L, 1e-4)
  return { scale, hat: xs.map((v) => clamp(Math.round((v - lo) / scale), 0, L) * scale + lo) }
}

const q = computed(() => {
  const r = mulberry32(seed.value * 53 + 11)
  const base = range(T).map(() => range(D).map(() => randn(r)))
  const rowS = range(T).map(() => Math.exp(randn(r) * 0.8)), sign = range(D).map(() => (r() < 0.5 ? -1 : 1))
  // K 型: 离群通道 = 每个 token 在这一维都是 ±mag 附近的大值; V 型: 每个 token 整行乘一个幅度
  const K = base.map((row, t) => row.map((v, c) => (kind.value === 'V' ? v * rowS[t] : v)
    + (outs.value.has(c) ? mag.value * sign[c] * (1 + 0.15 * base[t][(c + 1) % D]) : 0)))
  const all = quant(K.flat(), bits.value)
  const rows = K.map((row) => quant(row, bits.value))
  const cols = range(D).map((c) => quant(K.map((row) => row[c]), bits.value))
  const hats = [
    range(T).map((t) => all.hat.slice(t * D, (t + 1) * D)),   // axis=None
    rows.map((x) => x.hat),                                    // ★ axis=1: 每个 token 一组
    range(T).map((t) => cols.map((x) => x.hat[t])),            // ★ axis=0: 每个通道一组 (KIVI 对 K 的选择)
  ]
  const kf = K.flat(), hf = hats.map((H) => H.flat()), nk = Math.hypot(...kf)
  const errs = hf.map((h) => Math.hypot(...kf.map((v, i) => v - h[i])) / nk)
  const eMax = Math.max(1e-9, ...hf.flatMap((h) => h.map((v, i) => Math.abs(v - kf[i]))))
  const scales = (t, c) => [all.scale, rows[t].scale, cols[c].scale]
  return { K, hats, errs, eMax, scales, vMax: Math.max(...kf.map(Math.abs)) }
})
const bestI = computed(() => argmax(q.value.errs.map((e) => -e))), worstI = computed(() => argmax(q.value.errs))
const valColor = (v) => heat(Math.abs(v) / q.value.vMax, v >= 0 ? 'var(--accent)' : 'var(--right)')
const readout = computed(() => {
  if (!hover.value) return '悬停或点一个格子: 显示它的原值, 以及三种分组下量化再还原的值和各自所在组的 scale (格点间距)。scale 越大, 格点越粗, 还原值离原值越远。'
  const { t, c } = hover.value, s = q.value.scales(t, c)
  return `token ${t}, 通道 ${c}: 原值 ${q.value.K[t][c].toFixed(2)} → ` + NAMES.map((n, i) => `${n} ${q.value.hats[i][t][c].toFixed(2)} (scale ${s[i].toFixed(3)})`).join(' · ')
})
</script>

<style scoped>
/* 窄屏: 保住可读的最小宽度, 由 .lab-viz 横向滚动; 只有 .draggable 把手拦截触摸 */
svg { min-width: 540px; touch-action: pan-x pan-y; }
.cap { font-size: 9.5px; fill: var(--text-muted); }
.cap.best { fill: var(--left); font-weight: 600; }
.cap.worst { fill: var(--danger); }
.c { stroke: var(--border); stroke-width: 0.5; }
.c.clk { cursor: pointer; }
.c.hov { stroke: var(--text); stroke-width: 1.5; }
.colhead { fill: var(--bg-elev); stroke: var(--border-strong); stroke-width: 0.5; cursor: pointer; outline: none; }
.colhead.on { fill: var(--warn); }
.colhead:hover, .colhead:focus-visible { stroke: var(--accent); stroke-width: 1.5; }
.read { font-size: 11px; color: var(--text-muted); margin-top: 8px; line-height: 1.6; min-height: 3.2em; }
</style>
