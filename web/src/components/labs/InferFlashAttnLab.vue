<!-- FlashAttention 实验台 (llm_infer/m11): 按 tile 走 N×N 分数矩阵, 跟着一行 query 看 online softmax 的 m / l 怎么更新。 -->
<template>
  <LabFrame
    title="FlashAttention — 分块算注意力, online softmax 让分块结果能精确拼起来"
    sub="- 模拟什么: 标准注意力机制需先对注意力分数矩阵逐行计算 softmax, 再与值矩阵 V 相乘。朴素实现将完整的 $N \times N$ 分数矩阵 (所有 query 与 key 的点积) 显式计算并写回高带宽显存 HBM, 随后重新读出做归一化与矩阵乘, 访存开销随序列长度呈平方增长。FlashAttention 将 Q 与 K/V 沿序列维度切分为小块 (tile), 每次仅将单个 tile 加载至 GPU 高速片上缓存 SRAM 中完成局部计算与累加, 全程无需在 HBM 中物化 $N \times N$ 中间矩阵。核心难点在于标准 softmax 依赖整行的全局最大值与全局指数和分母, 而分块流式扫描时每次仅能观测到一个局部块。online softmax 通过递推状态解决这一矛盾:
  1. 为每个 query 行仅维护两个标量运行状态: 当前已见元素的运行最大值 $m$ 与运行指数和分母 $l$。
  2. 每扫描一个新的 K/V tile 即在线更新这两个状态: 若新块中出现更大的注意力分数, 先将旧分母 $l$ 乘以重缩放因子 $\exp(m_{\text{old}} - m_{\text{new}})$ 对齐至新最大值基准, 再累加新块内的指数项。
  3. 扫描完该行所有可见 tile 后, 最终结果与一次性在完整行上计算的精确 softmax 严格等价。
- 图上是什么: 面板分为左右两栏, 右下角附有当前跟踪行的逐块递推日志:
  - 左侧分数网格: $N \times N$ 注意力分数矩阵, 每一行对应一个 query (自上而下记为 q0、q1 …), 每一列对应一个 key; 右上角置灰区域代表因果掩码屏蔽区 (当前位置不可关注未来 token)。黄色高亮块表示当前正在片上 SRAM 中处理的活跃 tile, 绿色块表示已扫描归并完毕的历史 tile, 白色块尚未处理; 当前选定跟踪的那一行直接显示整数分数。
  - 右侧权重对比柱状图: 聚焦当前选定的跟踪行。空心轮廓柱表示基于整行完整可见域计算的精确 softmax 真值, 紫色实心柱表示仅基于截至当前步已扫描 key 计算出的局部归一化权重 $\exp(s-m)/l$。
  - 右下角更新记录表: 逐块记录该行每读入一个新 K 块时运行最大值 $m$ 的跃迁以及分母 $l$ 按重缩放因子修正并累加新项的完整算式; 重缩放因子小于 1 的步标注「★重缩放」。
- 可以做什么: 按面板自上而下支持以下交互操作:
  - 拖动「序列长度 N」与「块大小」滑杆调整矩阵规模与分块粒度 (总 tile 数随之改变)。
  - 点击「换一组分数」随机刷新注意力点积矩阵。
  - 点击左侧矩阵行首的 q 行号切换重点追踪的 query 行。
  - 使用播放器逐个 tile 单步演进; 扫描顺序与 FlashAttention-2 保持一致 (外层循环遍历 Q 块, 内层循环遍历 K 块, 整块落入因果掩码区的无效 tile 直接跳过)。
- 应该看到: 仅扫描前几个局部块时紫色实心柱普遍高于空心真值柱 (因局部累积的分母偏小导致已见位置权重被暂时高估); 随着后续 key 块陆续并入分母摊薄权重, 扫描完该行最后一个有效 tile 时实心柱与空心真值柱完全重合。右侧四项读数含义如下:
  - 一个 tile 的元素数 vs 整矩阵: 对比片上 SRAM 单次驻留的分块工作集规模 (块大小的平方) 与完整 $N \times N$ 矩阵元素数, 括号内给出空间缩减倍数。
  - 运行最大值 m / 分母 l: 当前跟踪行在当前扫描步维护的两个在线递推状态值。
  - lse: 全局归一化分母的对数 $m + \ln l$ (log-sum-exp), 实际算子仅需存储该标量即可供反向传播重算与跨块结果归并使用。
  - 与整行 softmax 的最大差: 全部 query 行扫描完毕后与标准整行 softmax 真值之间的最大绝对偏差, 应为精确的 0 或 1e-16 量级的双精度浮点舍入误差。"
    module="llm_infer/m11"
    run="python -m llm_infer.m11_flash_attention.demo"
    :challenge="{
      ask: '单步回放扫描过程并观察右下角的递推更新记录: 在什么条件下旧的运行分母 $l$ 会被乘上一个小于 1 的重缩放系数?',
      answer: '当新扫入的 tile 中出现了严格大于当前运行最大值 $m_{\\text{old}}$ 的注意力分数时。\n运行分母 $l$ 中累加的 $\\sum \\exp(s - m_{\\text{old}})$ 是以旧最大值 $m_{\\text{old}}$ 为指数平移基准计算的; 一旦基准上移更新为更大的 $m_{\\text{new}}$, 原先已累加的每一项都必须统一乘以修正因子 $\\exp(m_{\\text{old}} - m_{\\text{new}})$ (因 $m_{\\text{old}} \\lt m_{\\text{new}}$, 该因子严格小于 1), 才能与以 $m_{\\text{new}}$ 为基准的新块指数项在同一尺度下直接相加。\n两类错误简化均会导致计算失效:\n- 省略重缩放修正: 先前 tile 中旧 key 的指数权重相对新 key 被系统性放大, 归一化结果失真。\n- 完全不减去最大值而直接累加原始 $\\exp(s)$: 在 fp16 半精度下只要分数 $s \\gt 11$ 即发生指数上溢 (overflow), 因此减去运行最大值做数值稳定平移必不可少。\n相较标准两遍扫描 softmax, online softmax 仅增加了这一步在线基准重缩放操作。该递推性质赋予了 softmax 可分块结合律: 任意两段局部结果只需携带各自的 $(m, l)$ 状态二元组即可无损合并, 从而彻底免除了在 HBM 中读写 $N \\times N$ 注意力矩阵的开销, 将中间工作集空间复杂度从 $O(N^2)$ 降至与序列长度 $N$ 无关的常数级 $O(\\text{块大小}^2)$。',
    }"
  >
    <template #controls>
      <LabSlider v-model="N" label="序列长度 N" :min="8" :max="16" />
      <LabSlider v-model="bsIdx" label="块大小" :min="0" :max="3" :format="(v) => 2 ** v" />
      <div class="row"><button type="button" @click="seed++">换一组分数</button></div>
      <StepPlayer :stepper="stepper" :label="`tile ${stepper.step.value + 1} / ${tiles.length}`" />
    </template>

    <div class="wrap">
      <div class="cells" :style="{ gridTemplateColumns: `34px repeat(${N}, 22px)` }">
        <template v-for="i in N" :key="i">
          <button type="button" class="rowbtn mono" :class="{ active: row === i - 1 }" :aria-pressed="row === i - 1" @click="row = i - 1">q{{ i - 1 }}</button>
          <span v-for="j in N" :key="j" class="cell" :class="cellCls(i - 1, j - 1)" :title="j <= i ? `s[${i - 1}][${j - 1}] = ${S[i - 1][j - 1].toFixed(2)}` : 'mask'">
            {{ j <= i && row === i - 1 ? Math.round(S[i - 1][j - 1]) : '' }}
          </span>
        </template>
      </div>
      <div class="side">
        <svg :viewBox="`0 0 ${BW} 110`" role="img" aria-label="跟踪行的部分 softmax 与真实 softmax">
          <g v-for="j in rowLen" :key="j">
            <rect :x="(j - 1) * barW + 1" :y="100 - truth[j - 1] * barH" :width="barW - 2" :height="truth[j - 1] * barH" class="truth" />
            <rect v-if="j <= state.seen" :x="(j - 1) * barW + 3" :y="100 - Math.min(1, partial[j - 1]) * barH" :width="barW - 6" :height="Math.min(1, partial[j - 1]) * barH" class="part" />
          </g>
          <line x1="0" :x2="BW" y1="100" y2="100" class="axis" />
        </svg>
        <p class="legend"><span class="sw truth" />整行 softmax (真值) <span class="sw part" />只看已见 key 的部分 softmax <Tex text="$\exp(s-m)/l$" /></p>
        <ol class="log mono">
          <li v-for="(u, i) in state.log" :key="i" :class="{ hot: u.rescale < 1, cur: i === state.log.length - 1 }">
            K 块 {{ u.b }}: m {{ fmt(u.mOld) }}→{{ fmt(u.m) }} · l = {{ fmt(u.lOld) }}×{{ u.rescale.toFixed(3) }} + {{ fmt(u.add) }} = {{ fmt(u.l) }}{{ u.rescale < 1 ? ' ★重缩放' : '' }}
          </li>
          <li v-if="!state.log.length">q{{ row }} 所在的 Q 块还没轮到, 往后播放。</li>
        </ol>
      </div>
    </div>

    <template #stats>
      <div class="kv"><span>一个 tile 的元素数 vs 整矩阵 (SRAM 工作集)</span><b :class="bs * bs < N * N ? 'good' : ''">{{ bs * bs }} vs {{ N * N }} ({{ (N * N / (bs * bs)).toFixed(0) }}×)</b></div>
      <div class="kv"><span>q{{ row }} 的运行最大值 <Tex text="$m$" /> / 运行分母 <Tex text="$l$" /></span><b>{{ state.log.length ? `${fmt(state.m)} / ${fmt(state.l)}` : '—' }}</b></div>
      <div class="kv"><span>分母的对数 <Tex text="$\mathrm{lse} = m + \ln l$" /></span><b>{{ state.log.length ? fmt(state.m + Math.log(state.l)) : '—' }}</b></div>
      <div class="kv"><span><Tex text="全部行走完后与整行 softmax 的最大差 $\max|\text{online} - \text{整行}|$" /></span><b :class="maxDiff < 1e-6 ? 'good' : 'bad'">{{ maxDiff === 0 ? '0' : maxDiff.toExponential(1) }}</b></div>
      <div class="lab-note">
        <p>每个 tile 只进 SRAM 一次, 共 {{ tiles.length }} 个 (因果: 整块被 mask 的 tile 直接跳过, 连乘法都不做)。</p>
        <p>真实 kernel 除了 <Tex text="$m$" /> 和 <Tex text="$l$" />, 还同步累计输出 O (同样乘 <Tex text="$\exp(m_{\text{old}} - m_{\text{new}})$" /> 重缩放), 并把 lse 存下来:</p>
        <ul class="pts">
          <li>反向传播靠它重算概率矩阵而不用存 <Tex text="$N \times N$" />。</li>
          <li>ring attention 和 chunked prefill 靠它把分段算出的结果合并。</li>
        </ul>
      </div>
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
import { mulberry32, randn, range, softmax } from '@/utils/labmath.js'

const BW = 320
const fmt = (x) => (x === -Infinity ? '−∞' : x.toFixed(2))

const N = ref(12)
const bsIdx = ref(1)
const seed = ref(4)
const row = ref(7)
const bs = computed(() => 2 ** bsIdx.value)
watch(N, (n) => { if (row.value > n - 1) row.value = n - 1 })

// 固定生成 16×16 再裁剪: 拖 N 时已有的分数不变
const S = computed(() => { const rand = mulberry32(seed.value * 31337); return range(16).map(() => range(16).map(() => randn(rand) * 2.5)) })
// tile 顺序同 FlashAttention-2: 外层 Q 块, 内层 K 块; 因果下 K 块起点 > Q 块终点的 tile 整块被 mask, 跳过
const tiles = computed(() => {
  const nb = Math.ceil(N.value / bs.value)
  return range(nb).flatMap((a) => range(a + 1).map((b) => ({ a, b })))
})

// 行 i 在吃掉 K 块 [b·bs, …] 之后的 (m, l): online softmax 的一步
const update = (i, b, m, l) => {
  const keys = range(bs.value).map((j) => b * bs.value + j).filter((j) => j <= i && j < N.value)
  const mNew = Math.max(m, ...keys.map((j) => S.value[i][j]))
  const rescale = m === -Infinity ? 0 : Math.exp(m - mNew) // ★ 基准从 m 换成 mNew: 旧的和先乘 exp(m_old − m_new)
  const add = keys.reduce((a, j) => a + Math.exp(S.value[i][j] - mNew), 0)
  return { b, mOld: m, lOld: l, m: mNew, rescale: m === -Infinity ? 1 : rescale, add, l: l * rescale + add, seen: keys[keys.length - 1] + 1 }
}
const state = computed(() => {
  let m = -Infinity, l = 0, seen = 0
  const log = []
  tiles.value.slice(0, stepper.step.value + 1).filter((t) => t.a === Math.floor(row.value / bs.value)).forEach((t) => {
    const u = update(row.value, t.b, m, l); log.push(u); ({ m, l, seen } = u)
  })
  return { m, l, seen, log }
})
const rowLen = computed(() => row.value + 1)
const barW = computed(() => BW / rowLen.value)
const truth = computed(() => softmax(S.value[row.value].slice(0, rowLen.value)))
const barH = 95 // 纵轴固定 0..1: 部分 softmax 在只见过少数 key 时会高于真值, 看它怎么被后来的 key 稀释
const partial = computed(() => range(state.value.seen).map((j) => Math.exp(S.value[row.value][j] - state.value.m) / state.value.l))
// 所有行走完全部 tile 后, 与整行 softmax 的最大差
const maxDiff = computed(() => Math.max(...range(N.value).map((i) => {
  let m = -Infinity, l = 0
  for (let b = 0; b * bs.value <= i; b++) ({ m, l } = update(i, b, m, l))
  const full = softmax(S.value[i].slice(0, i + 1))
  return Math.max(...full.map((p, j) => Math.abs(p - Math.exp(S.value[i][j] - m) / l)))
})))

const stepper = useStepper(() => tiles.value.length, { interval: 500 })
watch(tiles, (t) => { stepper.pause(); stepper.step.value = Math.floor(t.length / 2) }, { immediate: true })
const cellCls = (i, j) => {
  if (j > i) return 'dim'
  const idx = tiles.value.findIndex((t) => t.a === Math.floor(i / bs.value) && t.b === Math.floor(j / bs.value))
  return [idx === stepper.step.value ? 'hot' : idx < stepper.step.value ? 'on' : '', { follow: i === row.value }]
}
</script>

<style scoped>
.wrap { display: flex; flex-wrap: wrap; gap: 16px; align-items: flex-start; }
.side { flex: 1 1 220px; min-width: 0; }
.side svg { max-width: 360px; }
.rowbtn { min-height: 22px; height: 22px; padding: 0; font-size: 10px; }
.cell.follow { border-color: var(--text); font-size: 9px; }
.truth { fill: none; stroke: var(--text-muted); background: transparent; border: 1px solid var(--text-muted); }
.part { fill: var(--accent); background: var(--accent); }
.axis { stroke: var(--border-strong); }
.legend { font-size: 11px; color: var(--text-dim); margin: 4px 0 8px; }
.sw { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin: 0 4px 0 8px; box-sizing: border-box; }
.log { list-style: none; padding: 0; margin: 0; font-size: 11px; color: var(--text-muted); line-height: 1.7; }
.log .hot { color: var(--warn); }
.log .cur { color: var(--text); font-weight: 600; }
.log .hot.cur { color: var(--warn); }
</style>
