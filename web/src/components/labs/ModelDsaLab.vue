<!--
  DSA 稀疏选择实验台 (对应 llm_models/layers/core/attention.py:LightningIndexer / MultiHeadLatentSparseAttention)。
  只讲一件事: 便宜的 indexer 先给所有 key 打分, 只留 top-k 交给昂贵的主注意力;
  indexer 靠 KL(主注意力 ‖ softmax(indexer 分数)) 学会 “和主注意力排序一致”, 对齐得越好, k 就能越小。
-->
<template>
  <LabFrame
    title="DSA 稀疏注意力 — 先用便宜的 indexer 粗选 top-k 个 key, 再只对它们精算注意力"
    sub="- 模拟什么: 单个查询向量面对 48 个历史 key 时的稀疏筛选机制。标准稠密注意力需对全部 48 个历史 key 逐一计算高维注意力得分、执行 softmax 并加权聚合 value。DSA (DeepSeek Sparse Attention) 在主注意力算子前前置一个极轻量的索引打分器 (indexer): 先以极低开销对全部 48 个历史位置执行粗粒度打分, 仅筛选出得分排名前 $k$ 的历史位置送入开销高昂的主注意力层计算, 其余历史位置直接跳过。由于 indexer 初始化为随机权重, 需借助专门的分布对齐损失驱动其输出分布逼近主注意力的真实概率分布, 使其学会与主注意力保持一致的重要性排序; 对齐精度越高, 达到同等输出质量所需的预算 $k$ 即可压得越小。
- 图上是什么: 上方为两排对齐的概率分布柱状图, 下方为随预算 $k$ 变化的误差曲线:
  - 上排柱图: 48 根柱子展示稠密主注意力对各历史 key 算出的真实归一化概率 $p$, 作为评估稀疏筛选准确度的基准真值。
  - 下排柱图: 轻量级 indexer 输出并经归一化后的近似评分分布 $I$, 推理时仅依据该分布排序选出 top-k 索引。
  - 柱体着色规则: 被 indexer 选入 top-k 集合的位置在上下两排同步着色高亮 (上排标紫色, 下排标橙色); 主注意力赋予显著权重 ($p \gt  0.04$) 却被 indexer 漏选的关键位置在上排标为醒目红色; 其余未选中的低权重位置呈灰色。
  - 底部曲线: 横轴为保留位置数 $k$ (从 1 至 48), 纵轴为仅对所选 top-k 位置计算稀疏注意力所得输出相对于全量稠密输出的相对误差; 绿色水平虚线标示 5% 误差参考阈值, 实心大圆点标示当前选定的 $k$。
- 可以做什么: 按界面从上到下依次操作:
  - 拖动「top-k」滑杆或直接点击底部曲线上的圆点修改保留的历史 key 数量。
  - 拖动「indexer 对齐程度」调节索引器的训练收敛水平: 0 代表未经训练的随机初始化状态 (打分为纯噪声), 1 代表与主注意力 logit 排序完全对齐 (对应训练脚本完成 indexer 预热后的理想状态)。
  - 点击「换一个 query」重新采样一组随机注意力分布。
  - 悬停或点击任意历史位置列, 下方会显示该位置的真实概率 $p$、在 indexer 中的得分排名及是否被选入稀疏集合。
- 应该看到: 当「indexer 对齐程度」较低时, 上排若干高概率主峰常呈红色漏选状态, 底部误差曲线需将 $k$ 放大至接近全长才能回落; 将对齐程度拖至 1 后, 红色漏选柱完全消失, 误差曲线在很小的 $k$ 值下便迅速跌破 5% 参考线。右侧读数含义如下:
  - 「选中 key 覆盖的注意力质量」统计所选 top-k 位置在主注意力下的真实概率 $p$ 之和, 达到 90% 以上表明主要有效上下文已被完整捕获。
  - 「输出相对误差」按 ‖稀疏输出 − 稠密输出‖ / ‖稠密输出‖ 计算, 控制在 5% 以内代表近似精度良好。
  - 「对齐 loss KL」即训练脚本中用于监督 indexer 的 KL 散度损失, 为 0 时代表 indexer 归一化分布与主注意力完全一致。
  - 「主注意力计算量 k/S」表示主注意力层实际参与计算的上下文比例, 该比值越低代表 DSA 节省的主干算力与访存越多。"
    module="llm_models/layers/core/attention.py"
    run="python -m llm_models.run_models.moe.deepseek_v3_2.train_deepseek_v3_2"
    :challenge="{
      ask: '将「indexer 对齐程度」拖至 0 (未训练状态) 并设 $k = 8$, 此时输出相对误差约为多少? 若再将对齐程度拖至 1, 要将输出相对误差控制在 5% 以内, 预算 $k$ 大致只需取多大?',
      answer: '未对齐时输出相对误差接近 100%; 完成对齐后 $k$ 仅需取 8 左右即可将相对误差压低至 5% 以内 (具体临界值随「换一个 query」采样的分布略有浮动)。\n原因分析如下:\n- 未对齐状态 (对齐程度为 0): 此时 indexer 的打分等同于纯随机噪声, 相当于在 48 个历史位置中盲目随机抽取 8 个。主注意力真正依赖的少数高权重位置大概率全部漏选, 导致稀疏输出与稠密基准几乎毫无关联, 相对误差逼近 100%, 直接启用稀疏截断会使模型表征严重失真。\n- 完全对齐状态 (对齐程度为 1): indexer 的得分排序与主注意力一致, 仅保留 top-8 位置即可覆盖 90% 以上的真实注意力质量。其根本原因在于 softmax 注意力具有显著的长尾稀疏特性: 本例 48 个历史位置中仅有约 5 个关键位置拥有显著概率峰值, 其余绝大多数位置的权重均趋近于 0, 跳过它们几乎不损失有效信息。\nDSA 机制成立依赖两项核心前提:\n- 主注意力分布本身具有高度长尾稀疏性。\n- 存在计算代价远低于主注意力的轻量级代理算子能够准确预判高权重位置。\n在训练方法上, 由于离散的 top-k 截断算子不可导, 主干语言模型的交叉熵损失无法直接反向传播至 indexer, 因此必须单独引入 KL 散度对齐损失分两阶段训练:\n1. 稠密预热阶段 (dense warmup): 主注意力层保持全量稠密计算以确保主干表征稳定, 同步以主注意力的真实分布为监督目标单独训练 indexer。\n2. 稀疏训练阶段: 待 indexer 排序充分收敛后再正式切换为 top-k 稀疏注意力。\n在配套 train 脚本实验中, 仅历经 80 步稠密预热, indexer 的 top-8 召回率便从随机抽样基线的 0.450 提升至 0.922。\n在复杂度账面上, 昂贵的主注意力计算复杂度从 $O(T^2)$ 降至线性的 $O(T \\cdot k)$, 额外增加的开销仅为轻量级 indexer 的一次低成本 $O(T^2)$ 扫描 (采用更少的头数、极低的投影维度, 且无需投影聚合 value 向量或执行全长归一化)。',
    }"
  >
    <template #controls>
      <LabSlider v-model="k" label="top-k" :min="1" :max="S" />
      <LabSlider v-model="align" label="indexer 对齐程度" :min="0" :max="1" :step="0.05" :format="(t) => t.toFixed(2)" />
      <div class="row"><button type="button" @click="seed++">换一个 query</button></div>
    </template>

    <!-- 整组只有一个 Tab 停靠点: 悬停临时看, 点击固定, 方向键左右移动固定的那一列 -->
    <div class="rows" tabindex="0" role="group" aria-label="48 个 key 的两排柱子, 方向键左右换一列"
         @mouseleave="hov = -1" @keydown.left.prevent="move(-1)" @keydown.right.prevent="move(1)">
      <p class="cap">上排: 主注意力的真实概率 p (稠密算出来的标准答案)</p>
      <div class="strip">
        <div v-for="j in S" :key="j" class="c" :class="cls(j - 1)" @mouseenter="hov = j - 1" @click="pin = pin === j - 1 ? -1 : j - 1"><div :style="{ height: (sim.p[j - 1] / pMax) * 100 + '%' }" /></div>
      </div>
      <p class="cap">下排: indexer 的打分 softmax(I), 只用来排序选出 top-{{ k }}</p>
      <div class="strip">
        <div v-for="j in S" :key="j" class="c idx" :class="cls(j - 1)" @mouseenter="hov = j - 1" @click="pin = pin === j - 1 ? -1 : j - 1"><div :style="{ height: (sim.qi[j - 1] / qMax) * 100 + '%' }" /></div>
      </div>
      <p class="cap legend">
        <i class="sw p" /> 上排: 被 top-k 选中 <i class="sw i" /> 下排: 被 top-k 选中 <i class="sw m" /> 主注意力看重却被漏掉的 key (p &gt; 0.04) <i class="sw n" /> 其余
      </p>
      <p class="cap mono" aria-live="polite">{{ cur < 0 ? '悬停或点击某一列查看' : `key ${cur}: p = ${sim.p[cur].toFixed(3)}, indexer 排名第 ${sim.rank[cur] + 1}, ${sim.rank[cur] < k ? '选中' : '未选中'}` }}</p>
    </div>

    <svg viewBox="0 0 520 130" tabindex="0" role="slider" aria-label="输出误差随 k 变化; 点击圆点或按方向键设置 k"
         aria-valuemin="1" :aria-valuemax="S" :aria-valuenow="k" :aria-valuetext="`k = ${k}, 误差 ${(sim.curve[k - 1] * 100).toFixed(1)}%`"
         @keydown.left.prevent="k = Math.max(1, k - 1)" @keydown.right.prevent="k = Math.min(S, k + 1)">
      <line x1="34" x2="514" :y1="ey(0.05)" :y2="ey(0.05)" class="ok" /><text x="514" :y="ey(0.05) - 3" class="lg">5% 误差线</text>
      <text x="30" :y="ey(1) + 8" class="yl">100%</text><text x="30" :y="ey(0) + 3" class="yl">0</text>
      <polyline :points="sim.curve.map((e, i) => `${ex(i + 1)},${ey(e).toFixed(1)}`).join(' ')" class="curve" />
      <circle
        v-for="(e, i) in sim.curve" :key="i" :cx="ex(i + 1)" :cy="ey(e)" :r="i + 1 === k ? 6 : 3.5" class="pt" :class="{ now: i + 1 === k }"
        @click="k = i + 1"
      />
      <text x="274" y="126" class="xl">横轴 k (点任意圆点可直接设置) · 纵轴 输出相对误差 ‖o_稀疏 − o_稠密‖ / ‖o_稠密‖</text>
    </svg>

    <template #stats>
      <div class="kv"><span>选中 key 覆盖的注意力质量 (被选中 key 的 p 之和)</span><b :class="sim.mass > 0.9 ? 'good' : 'bad'">{{ (sim.mass * 100).toFixed(1) }}%</b></div>
      <div class="kv"><span>稀疏输出相对稠密输出的误差</span><b :class="sim.curve[k - 1] < 0.05 ? 'good' : 'bad'">{{ (sim.curve[k - 1] * 100).toFixed(1) }}%</b></div>
      <div class="kv"><span><Tex text="对齐 loss $\mathrm{KL}(p \,\|\, \mathrm{softmax}(I))$ (0 = 分布完全一样)" /></span><b>{{ sim.kl.toFixed(3) }}</b></div>
      <div class="kv"><span><Tex text="主注意力只需算的 key 占比 $k / S$" /></span><b>{{ ((k / S) * 100).toFixed(0) }}%</b></div>
      <div class="lab-note">
        <p>KL 散度对齐损失驱动 indexer 的输出分布逼近主注意力分布, 而稀疏推理阶段真正决定近似精度的仅是相对排序——即所选 top-k 集合是否完整囊括了真实概率 $p$ 最高的若干关键位置。因此工程上无需等待 KL 损失严格收敛至 0, 只要头部位置的排序准确, 注意力质量覆盖率即可达到较高水平。</p>
        <p>稀疏输出的算法:</p>
        <ol class="pts">
          <li>把没被选中的 key 的分数填成 −∞。</li>
          <li>只在选中的 k 个 key 上重新做 softmax。</li>
          <li>再加权 value。</li>
        </ol>
        <p>该掩码筛选逻辑与 Python 源码中的 _sparse_mask_from_topk 函数实现完全一致。</p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import Tex from '@/components/Tex.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { mulberry32, randn, softmax, range, sum } from '@/utils/labmath.js'

const S = 48, DV = 4
// 默认对齐程度 0.3: indexer 还没学好, 打开就能看到漏选的 key 和很大的误差
const k = ref(8), align = ref(0.3), seed = ref(2)
const hov = ref(-1), pin = ref(-1)                       // 悬停的列 / 点击固定的列
const cur = computed(() => (hov.value >= 0 ? hov.value : pin.value))
const move = (d) => { pin.value = Math.min(S - 1, Math.max(0, (cur.value < 0 ? 0 : cur.value + d))); hov.value = -1 }

const sim = computed(() => {
  const r = mulberry32(seed.value * 31)
  // 真实注意力 logit: 大部分是噪声, 少数几个 key 明显相关 (注意力天然长尾)
  const logit = range(S).map(() => randn(r) * 0.8)
  range(5).forEach(() => { logit[Math.floor(r() * S)] += 3 + r() * 2 })
  const noise = range(S).map(() => randn(r) * 1.6)
  const vals = range(S).map(() => range(DV).map(() => randn(r)))
  const p = softmax(logit)
  // indexer 分数: align=1 与主注意力 logit 一致 (KL→0), align=0 纯噪声
  const I = logit.map((l, j) => align.value * l + (1 - align.value) * noise[j])
  const qi = softmax(I)
  const order = range(S).sort((a, b) => I[b] - I[a])
  const rank = Array(S); order.forEach((j, i) => { rank[j] = i })
  const out = (w) => range(DV).map((e) => sum(w.map((wj, j) => wj * vals[j][e])))
  const dense = out(p), nd = Math.hypot(...dense)
  // ★ 稀疏输出: 主注意力只在 top-k 上重新 softmax (其余位置 mask 成 -inf)
  const curve = range(S).map((i) => {
    const o = out(softmax(logit.map((l, j) => (rank[j] <= i ? l : -Infinity))))
    return Math.hypot(...o.map((x, e) => x - dense[e])) / nd
  })
  const kl = sum(p.map((pj, j) => (pj > 0 ? pj * Math.log(pj / qi[j]) : 0)))
  return { p, qi, rank, curve, kl, mass: sum(p.filter((_, j) => rank[j] < k.value)) }
})
const pMax = computed(() => Math.max(...sim.value.p))
const qMax = computed(() => Math.max(...sim.value.qi))
const cls = (j) => ({ picked: sim.value.rank[j] < k.value, missed: sim.value.rank[j] >= k.value && sim.value.p[j] > 0.04, hov: cur.value === j })

const ex = (i) => 34 + ((i - 1) / (S - 1)) * 480
const ey = (e) => 108 - Math.min(1, e) * 96
</script>

<style scoped>
.cap { font-size: 11px; color: var(--text-dim); margin: 4px 0; }
.legend { display: flex; flex-wrap: wrap; align-items: center; gap: 4px 6px; }
.sw { width: 10px; height: 10px; border-radius: 2px; background: var(--border-strong); margin-left: 6px; }
.sw.p { background: var(--accent); margin-left: 0; } .sw.i { background: var(--eye); } .sw.m { background: var(--danger); }
.strip { display: flex; gap: 1px; height: 64px; align-items: flex-end; border-bottom: 1px solid var(--border-strong); min-width: 430px; }
.c { flex: 1; height: 100%; display: flex; align-items: flex-end; cursor: pointer; }
.c div { width: 100%; min-height: 2px; background: var(--border-strong); border-radius: 1px 1px 0 0; }
.c.picked div { background: var(--accent); }
.c.idx.picked div { background: var(--eye); }
.c.missed div { background: var(--danger); }
.c.hov { background: var(--accent-soft); }
.ok { stroke: var(--left); stroke-dasharray: 4 4; }
.lg { text-anchor: end; font-size: 9px; fill: var(--left); }
.yl { text-anchor: end; font-size: 9px; fill: var(--text-dim); }
.xl { text-anchor: middle; font-size: 9px; fill: var(--text-dim); }
.curve { fill: none; stroke: var(--accent); stroke-width: 1.5; }
.pt { fill: var(--accent); cursor: pointer; }
.pt.now { fill: var(--eye); stroke: var(--text); stroke-width: 1.5; }
</style>
