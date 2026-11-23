<!--
  NF4 量化实验台 (对应 llm_finetune/methods/qlora.py: nf4_quantize)。
  只讲一件事: 权重近似正态分布, 所以 16 个码点应该按正态分位数摆 (中间密、两边疏), 而不是等间距;
  每个 block 用 absmax 缩放, 所以一个 outlier 会把整个 block 的码点一起"拉稀"。
  量化流程与 Python 完全一致: 分 block → 除以 absmax → 最近邻查码本 → 乘回 scale。
-->
<template>
  <LabFrame
    title="NF4 vs INT4 — 4 bit 只有 16 个码点, 该摆在哪"
    sub="- 模拟什么: QLoRA 将冻结的预训练基座权重压缩为 4 bit 离散索引存储: 每个浮点权重仅能就近映射至码本中的 16 个离散「码点」之一, 前向计算时再实时反量化还原为浮点值。均匀量化 INT4 将 16 个码点在区间内等间距排列; NF4 (4-bit NormalFloat) 则按标准正态分布的等概率分位数设置码点, 呈现零点附近密集、两侧尾部稀疏的非均匀布局, 以匹配预训练神经网络权重近似服从零均值正态分布的统计特性。分块量化流程与本仓库 qlora.py 中的 nf4_quantize 完全一致, 每连续 B 个权重划分为一个独立量化 block:
  1. 提取当前组内最大绝对值 absmax 并对组内元素做归一化, 将数值范围映射至 [-1, 1]; 因此组内一旦出现单个幅值悬殊的离群值 outlier, 便会将整组的反量化码点间距同步撑大。
  2. 在 [-1, 1] 标准码本中逐元素查找欧氏距离最近的码点编号。
  3. 存储每个权重的 4 bit 码点索引以及该组共享的浮点缩放因子 absmax。
- 图上是什么: 视图自上而下展示分块量化的统计分布与映射细节:
  - 顶部灰色直方图: 全部 512 个模拟权重的整体经验分布, 横轴以标准差 σ 为单位。
  - 中间水平圆点列: 展示第 0 个 block 内的全部原始权重位置, 其中红色大圆点代表可人为注入拉大的单个离群值 outlier。
  - 上下两排码点刻度: 展示第 0 个 block 反量化回原尺度后的 16 个码点位置, 上排紫色对应 NF4, 下排橙色对应 INT4; 刻度竖线高度正比于当前组内落入该码点的权重数量, 灰色短刻度代表未被任何权重命中的闲置码点。
  - 映射细线: 将第 0 个 block 内的每个原始权重圆点连接至其量化命中的目标码点。
- 可以做什么: 按界面顺序可执行以下操作:
  - 切换「连线看 NF4 / INT4」按钮: 选择映射细线连接至上排 NF4 还是下排 INT4 码点。
  - 点击「换一组权重」按钮: 更换随机种子重新采样 512 个正态分布权重。
  - 拖动红色大圆点 (或拖动「outlier 大小」滑杆): 将 block 0 内的离群权重幅值在 0 至 8σ 之间调节。
  - 拖动「block 大小 B」滑杆: 在 16 至 256 之间调整单个量化块包含的权重数量。
- 应该看到: 当无额外离群值 (outlier 为 0) 时, NF4 的 16 个码点使用率均匀饱满, 而 INT4 两侧边缘码点大量闲置; 将 outlier 拉大后, 整组码点跨度被迫拉宽, 同组其余正常权重全部拥挤坍缩至零点附近的少数几个码点上。右侧汇总读数含义如下:
  - 「量化误差 RMSE」: 全部 512 个权重反量化前后的均方根误差, 数值越低保真度越高, NF4 误差不高于 INT4 时显示为绿色。
  - 「有效比特」: 基于 16 个码点实际命中频率计算的香农信息熵, 理论上限为 4 bit, 越接近 4 代表编码容量利用越充分。
  - 「block 0 用到的码点数」: 第 0 个量化块中至少被一个权重命中的活跃码点个数。
  - 「每参数存储」: 4 bit 索引加上每块一个 32 bit 缩放因子 absmax 分摊至单个参数后的平均存储位数, 块大小 B 越小缩放因子分摊开销越高。"
    module="llm_finetune/methods/qlora.py"
    run="python -m llm_finetune.run_finetune.qlora.train_qlora"
    :challenge="{
      ask: '请按顺序调节参数并观察读数变化:\n1. 将 outlier 幅值拖回 0 (保持默认块大小): NF4 与 INT4 的「有效比特」读数各自约为多少?\n2. 将 outlier 拉至 8σ, 同时将 block 大小 B 调至 256, 观察量化误差变化。\n3. 保持 8σ 离群值不变, 仅将 block 大小 B 缩小至 16: 为什么全局量化误差 RMSE 会显著回落? 缩小分块粒度带来了什么存储代价?',
      answer: '结论:\n- 第 1 步: 在无额外离群值且默认 $B = 64$ 配置下, NF4 的有效比特约为 3.9 bit, INT4 约为 3.4 至 3.5 bit。\n- 第 3 步: 将分块大小缩小至 16 后 RMSE 显著回落, 原因是受单个离群值波及的同组正常权重数量从 255 个缩减至 15 个; 代价是缩放因子分摊开销上升, 相比 $B = 64$ 每个参数需额外多占用 1.5 bit 存储空间。\n机制解析:\n- 分位数编码的信息熵优势: 有效比特定义为 16 个离散码点命中频率的信息熵, 直接度量 4 bit 编码空间的实际利用率:\n  - NF4: 码点严格按标准正态分布等概率分位数排布, 正态权重落入各码点的概率均接近 $1/16$, 实测信息熵达约 3.9 bit, 接近理论信息论上限。\n  - INT4: 均匀网格下正态分布尾部稀疏区占据了大量两端码点, 实际命中频率极低, 信息熵仅为 3.4 至 3.5 bit, 量化 RMSE 相应高出约 20%。\n- 块内离群值污染范围: 分块量化的缩放因子 absmax 按块独立计算, 单个离群值仅污染其所属的量化块——该块的 absmax 被极端值撑大后, 块内其余正常权重归一化后全部压缩至零点邻域, 集中退化为中间少数几个码点。当取 $B = 256$ 且注入 8σ 离群值时, NF4 全局 RMSE 从约 0.10 升至 0.16, INT4 升至 0.26。\n- 分块粒度与元数据开销权衡: 将分块缩小至 $B = 16$ 后, 被该 8σ 离群值波及的同组权重仅剩 15 个, 其余 496 个权重各自使用未受污染的局部 absmax 正常量化, 全局 RMSE 随之大幅回落。相应代价是每块均需额外存储一个 FP32 精度的缩放因子, 单参数平均存储位数为 $4 + 32/B$ bit: 在 $B = 64$ 时为 4.5 bit, 缩至 $B = 16$ 时则增至 6 bit (每参数较默认配置多出 1.5 bit)。\n- 双重量化补偿机制: QLoRA 进一步引入双重量化 (Double Quantization), 对各块的 FP32 缩放因子 absmax 再做一层 8 bit 分块量化, 从而在不牺牲细粒度抗离群能力的前提下显著摊薄元数据存储开销。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: focus === 'nf4' }" @click="focus = 'nf4'">连线看 NF4</button>
        <button type="button" :class="{ active: focus === 'int4' }" @click="focus = 'int4'">连线看 INT4</button>
        <button type="button" @click="seed++">换一组权重</button>
      </div>
      <LabSlider v-model="out" label="outlier 大小" :min="0" :max="8" :step="0.1" unit="σ" :format="(v) => v.toFixed(1)" />
      <LabSlider v-model="logB" label="block 大小 B" :min="4" :max="8" :format="(v) => String(2 ** v)" />
    </template>

    <svg ref="svgEl" viewBox="0 0 640 300" role="group" aria-label="权重直方图与两套量化码点">
      <!-- 全体权重直方图 -->
      <rect v-for="(h, i) in hist" :key="'h' + i" :x="sx(-XMAX) + i * binW + 0.5" :y="92 - h * 80" :width="binW - 1" :height="h * 80" fill="var(--border-strong)" />
      <text x="8" y="16" class="t">全部 {{ N }} 个权重的分布 (单位 σ)</text>

      <!-- 两排码点 -->
      <g v-for="row in rows" :key="row.id" :opacity="focus === row.id ? 1 : 0.45">
        <text x="8" :y="row.y + (row.up ? -38 : 48)" class="t" :fill="row.color">{{ row.label }}</text>
        <line
          v-for="(lv, i) in row.levels" :key="i"
          :x1="sx(lv.v)" :x2="sx(lv.v)" :y1="row.y" :y2="row.y + (row.up ? -1 : 1) * (5 + 28 * lv.n / row.maxN)"
          :stroke="lv.n ? row.color : 'var(--text-dim)'" :stroke-width="lv.n ? 2.5 : 1"
        />
      </g>
      <!-- block 0 的权重 → 最近码点 -->
      <line
        v-for="(p, i) in blockPts" :key="'c' + i"
        :x1="sx(p.w)" :x2="sx(p.q)" y1="190" :y2="focusRow.y" :stroke="focusRow.color" stroke-width="0.6" opacity="0.55"
      />
      <line :x1="sx(-XMAX)" :x2="sx(XMAX)" y1="190" y2="190" stroke="var(--border)" />
      <circle v-for="(p, i) in blockPts" :key="'p' + i" :cx="sx(p.w)" cy="190" r="2.2" fill="var(--text-muted)" />
      <g v-for="t in [-8, -4, 0, 4, 8]" :key="'x' + t">
        <text :x="sx(t)" y="292" text-anchor="middle" class="t">{{ t }}σ</text>
      </g>
      <!-- ★ 可拖的 outlier: 它决定 block 0 的 absmax, 也就决定了 16 个码点铺多宽 -->
      <circle
        class="draggable" :cx="sx(outVal)" cy="190" r="9" fill="var(--danger)" stroke="var(--bg-card)" stroke-width="2"
        tabindex="0" role="slider" aria-label="outlier 大小" :aria-valuenow="out" aria-valuemin="0" aria-valuemax="8"
        @pointerdown="start($event, { svg: svgEl, onMove: ({ x }) => (out = Math.round(clamp((x - sx(0)) / SC, 0, 8) * 10) / 10) })"
        @keydown.right.prevent="out = clamp(out + 0.5, 0, 8)" @keydown.left.prevent="out = clamp(out - 0.5, 0, 8)"
      />
      <text :x="sx(outVal)" y="172" text-anchor="middle" class="t" fill="var(--danger)">{{ out < 0.05 ? '← 拖我' : 'outlier' }}</text>
    </svg>

    <template #stats>
      <div class="kv"><span>量化误差 RMSE (NF4 / INT4)</span><b :class="res.nf4.rmse <= res.int4.rmse ? 'good' : 'bad'">{{ res.nf4.rmse.toFixed(4) }} / {{ res.int4.rmse.toFixed(4) }}</b></div>
      <div class="kv"><span>有效比特, 满分 4 (NF4 / INT4)</span><b>{{ res.nf4.bits.toFixed(2) }} / {{ res.int4.bits.toFixed(2) }}</b></div>
      <div class="kv"><span>block 0 用到的码点数 (NF4 / INT4)</span><b>{{ rows[0].used }} / {{ rows[1].used }}</b></div>
      <div class="kv"><span><Tex text="每参数存储 $4 + 32/B$ bit" /></span><b>{{ (4 + 32 / B).toFixed(2) }} bit</b></div>
      <div class="lab-note">
        <p>有效比特定义为 16 个离散码点实际命中频率的信息熵, 用于度量 4 bit 编码容量的有效利用程度。</p>
        <ul class="pts">
          <li><b>分位数码本 (NF4):</b> 让每个码点被用到的概率都接近 1/16, 熵接近 4 bit。</li>
          <li><b>等间距码本 (INT4):</b> 两端几乎没有权重落进去, 那几个码点等于白占, 熵就低。</li>
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
import { clamp, entropy, mulberry32, randn, range } from '@/utils/labmath.js'

// 与 qlora.py 的 NF4_CODEBOOK 相同 (bitsandbytes 常数): 标准正态的 16 个等概率分位点, 归一化到 [-1, 1]
const NF4 = [-1, -0.6961928, -0.5250731, -0.3949175, -0.2844414, -0.1847734, -0.0910500, 0,
  0.0795803, 0.1609302, 0.2461123, 0.3379152, 0.4407098, 0.5626170, 0.7229568, 1]
const N = 512, XMAX = 8.6, SC = 300 / XMAX, OUT_IDX = 5
const sx = (v) => 320 + v * SC

const out = ref(0), logB = ref(6), seed = ref(1), focus = ref('nf4')
const svgEl = ref(null)
const { start } = useDrag()
const B = computed(() => 2 ** logB.value)

const weights = computed(() => {
  const rand = mulberry32(seed.value * 131)
  const w = range(N).map(() => randn(rand))
  if (out.value >= 0.05) w[OUT_IDX] = out.value
  return w
})
const outVal = computed(() => weights.value[OUT_IDX])

// 两种量化器: 输入已除以 absmax 的 x∈[-1,1], 返回 [码号, 反量化值]
const qNf4 = (x) => { let b = 0; NF4.forEach((c, i) => { if (Math.abs(c - x) < Math.abs(NF4[b] - x)) b = i }); return [b, NF4[b]] }
const qInt4 = (x) => { const q = clamp(Math.round(x * 7), -8, 7); return [q + 8, q / 7] }   // 对称 absmax INT4: 步长 absmax/7

const quantize = (w, qf) => {
  const deq = [], usage = Array(16).fill(0)
  for (let s = 0; s < w.length; s += B.value) {
    const blk = w.slice(s, s + B.value)
    const scale = Math.max(...blk.map(Math.abs), 1e-12)          // ★ 每个 block 一个 absmax scale
    for (const v of blk) { const [code, val] = qf(v / scale); usage[code]++; deq.push(val * scale) }
  }
  const rmse = Math.sqrt(deq.reduce((s, d, i) => s + (d - w[i]) ** 2, 0) / w.length)
  return { deq, rmse, bits: entropy(usage.map((u) => u / w.length)) / Math.LN2 }
}
const res = computed(() => ({ nf4: quantize(weights.value, qNf4), int4: quantize(weights.value, qInt4) }))

const block0 = computed(() => weights.value.slice(0, B.value))
const scale0 = computed(() => Math.max(...block0.value.map(Math.abs)))
const mkRow = (id, label, color, y, up, codes, deq) => {
  const levels = codes.map((c) => ({ v: c * scale0.value, n: 0 }))
  block0.value.forEach((_, i) => { const lv = levels.find((l) => Math.abs(l.v - deq[i]) < 1e-9); if (lv) lv.n++ })
  return { id, label, color, y, up, levels, used: levels.filter((l) => l.n).length, maxN: Math.max(...levels.map((l) => l.n), 1) }
}
const rows = computed(() => [
  mkRow('nf4', 'NF4 码点 (正态分位数)', 'var(--accent)', 150, true, NF4, res.value.nf4.deq),
  mkRow('int4', 'INT4 码点 (等间距)', 'var(--eye)', 230, false, range(16).map((i) => (i - 8) / 7), res.value.int4.deq),
])
const focusRow = computed(() => rows.value.find((r) => r.id === focus.value))
const blockPts = computed(() => block0.value.map((w, i) => ({ w, q: res.value[focus.value].deq[i] })).filter((_, i) => i !== OUT_IDX || out.value < 0.05))

const BINS = 86, binW = (2 * XMAX * SC) / BINS
const hist = computed(() => {
  const h = Array(BINS).fill(0)
  weights.value.forEach((w) => { h[clamp(Math.floor((w + XMAX) / (2 * XMAX) * BINS), 0, BINS - 1)]++ })
  const m = Math.max(...h)
  return h.map((c) => c / m)
})
</script>

<style scoped>
.t { font-size: 11px; fill: var(--text-muted); font-family: "SF Mono", Menlo, monospace; }
</style>
