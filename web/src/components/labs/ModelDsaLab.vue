<!--
  DSA 稀疏选择实验台 (对应 llm_models/layers/core/attention.py:LightningIndexer / MultiHeadLatentSparseAttention)。
  只讲一件事: 便宜的 indexer 先给所有 key 打分, 只留 top-k 交给昂贵的主注意力;
  indexer 靠 KL(主注意力 ‖ softmax(indexer 分数)) 学会 “和主注意力排序一致”, 对齐得越好, k 就能越小。
-->
<template>
  <LabFrame
    title="DSA 稀疏注意力 — 先用便宜的 indexer 粗选 top-k 个 key, 再只对它们精算注意力"
    sub="- 模拟什么: 注意力里一个 query 面对 48 个历史 key。普通 (稠密) 注意力要对全部 48 个 key 算分数、做 softmax、再加权 value。DSA (DeepSeek Sparse Attention) 在前面加一个很便宜的 indexer: 它先给 48 个 key 粗打一遍分, 只把分数最高的 k 个交给昂贵的主注意力, 其余 key 直接不算。indexer 一开始是随机的, 要靠一个对齐 loss (让它的打分分布去模仿主注意力的分布) 学会「和主注意力的排序一致」。对齐得越好, k 就能取得越小。
      - 图上是什么: 上面两排柱子, 下面一条曲线:
        - 上排: 48 根柱子是主注意力对每个 key 的真实概率 p, 这是稠密算出来的「标准答案」。
        - 下排: indexer 的打分 (softmax 之后的 I), 它只用来排序。
        - 柱子颜色: 被 indexer 选中的 top-k 两排都会标色 (上排紫色, 下排橙色); 红色柱是「主注意力很看重 (p > 0.04) 却被 indexer 漏掉」的 key; 灰色是其余。
        - 曲线: 横轴是 k 从 1 到 48, 纵轴是只保留 top-k 时的输出和稠密输出相比的相对误差。绿色虚线是 5% 误差线, 大圆点是当前 k。
      - 可以做什么: 按界面顺序:
        - 拖「top-k」: 改保留几个 key, 或者直接点曲线上的圆点。
        - 拖「indexer 对齐程度」: 0 表示 indexer 完全没训练过 (打分是纯噪声), 1 表示它和主注意力的 logit 完全一致 (对应 train 脚本里 indexer 预热做完的状态)。
        - 「换一个 query」: 换一组随机分数。
        - 悬停或点击任意一列: 下方会显示这个 key 的 p、它在 indexer 里排第几、有没有被选中。
      - 应该看到: 对齐程度低时, 上排最高的几根柱子常常是红色 (被漏掉), 误差曲线要到很大的 k 才降下来。把对齐程度拖到 1, 红色消失, 曲线在 k 很小时就掉到 5% 线以下。右侧读数:
        - 「选中 key 覆盖的注意力质量」: 被选中 key 的 p 之和, 它表示主注意力关心的 key 有多少被抓住了, 90% 以上算好。
        - 「输出相对误差」: ‖稀疏输出 − 稠密输出‖ / ‖稠密输出‖, 5% 以内算好。
        - 「对齐 loss KL」: train 脚本里训练 indexer 用的那个数, 0 表示 indexer 的分布和主注意力完全一样。
        - 「主注意力计算量 k/S」: 主注意力需要算的 key 占比, 占比越小, DSA 省下的算力越多。"
    module="llm_models/layers/core/attention.py"
    run="python -m llm_models.run_models.moe.deepseek_v3_2.train_deepseek_v3_2"
    :challenge="{
      ask: '把「indexer 对齐程度」拖到 0 (没训练过), k = 8: 输出误差多大? 再拖到 1。先猜: 对齐好之后, 要把误差压到 5% 以内, k 需要多大?',
      answer: '答案: 没对齐时误差接近 100%; 对齐之后 k 取 8 上下就能压到 5% 以内 (具体多少看「换一个 query」出来的那组分数)。\n为什么:\n- 没对齐 (对齐程度 0): indexer 的打分是纯噪声, 等于随机选 8 个 key。主注意力看重的那几个 key 大概率都被漏掉, 稀疏输出和稠密输出几乎没关系, 误差接近 100%。这时候稀疏化会让模型输出严重失真。\n- 对齐之后 (对齐程度 1): indexer 的排序和主注意力一致, top-8 就能盖住 90% 以上的注意力质量。因为 softmax 注意力是长尾的: 这里 48 个 key 里只有 5 个被明显抬高, 其余 key 的概率都接近 0, 算它们纯属浪费。\nDSA 成立靠两个前提:\n- 注意力天然稀疏。\n- 有便宜的办法猜中哪几个 key 重要。\n训练 indexer: top-k 这个「选前 k 个」的操作不可导, 语言模型的 loss 传不到 indexer。所以单独用 KL 对齐 loss 训练它, 分两个阶段:\n1. 先做 dense warmup: 主注意力照旧看全部 key, 只训 indexer。\n2. 等 indexer 学会了再切到稀疏。\ntrain 脚本里预热 80 步, top-8 召回率从 0.450 (随机乱选的期望) 升到 0.922。\n计算量: 主注意力的计算量从 $O(T^2)$ 降到 $O(T \\cdot k)$, 外加 indexer 一个很便宜的 $O(T^2)$ 打分 (head 少、维度小、没有 value、不做 softmax)。',
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
        <p>KL 对齐 loss 只要求 indexer 的分布「形状像」主注意力; 影响结果的是排序: top-k 里有没有包含 p 最大的那几个 key。所以 KL 不用降到 0, 只要排序对了, 覆盖的注意力质量就上去了。</p>
        <p>稀疏输出的算法:</p>
        <ol class="pts">
          <li>把没被选中的 key 的分数填成 −∞。</li>
          <li>只在选中的 k 个 key 上重新做 softmax。</li>
          <li>再加权 value。</li>
        </ol>
        <p>这和 Python 里 _sparse_mask_from_topk 做的事一样。</p>
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
