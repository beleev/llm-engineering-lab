<!-- 稀疏 decode 实验台 (llm_infer/m22): needle_context + quest_upper_bound / mean_score + select_blocks + evaluate 的 JS 移植。 -->
<template>
  <LabFrame
    title="稀疏注意力 decode — 只读打分最高的 k 个 KV block, 看丢了多少注意力"
    sub="- 模拟什么: 自回归 decode 每生成 1 个新 token 都需遍历读取上下文中全部历史 token 的 KV cache, 访存开销随上下文长度线性增长并成为延迟瓶颈; 但经过充分训练的 LLM 注意力分布通常高度稀疏集中, 绝大部分注意力权重落在极少数关键 token 上。本实验构造了长度为 1024 个 token 的上下文 KV 并切分为 64 个物理 block (每块包含 16 个 token), 在其中 4 个块内各植入 2 个与当前查询向量 $q$ 高度对齐的关键 key (称为「针」), 其余位置均为随机噪声 key。分块稀疏 decode 的执行流程如下:
  1. 将历史 KV 按序列位置切分为固定大小的 block, 并为每块常驻维护一份极轻量的元数据摘要。
  2. 每步 decode 先用当前查询向量 $q$ 与各块摘要快速计算块级重要度得分。
  3. 仅加载得分最高的 $k$ 个物理 block 参与注意力计算, 其余块跳过不读。
- 图上是什么: 按序列位置排列的 64 个方格, 每格代表一个 KV block:
  - 方格亮度: 颜色越亮代表该 block 在全量稠密注意力下真实分配到的概率质量越高 (此为事后核算的基准真值, 稀疏推理选块时不可见)。
  - 绿色外框: 当前打分策略选定并实际读取参与计算的 block。
  - 标有「针」的方格: 植入了高相关关键 token 的目标块。
  - 标有 S 与「近」的方格: S 代表第 0 块 (对应序列首部的 attention sink 区域, 常承载大量默认注意力), 「近」代表末尾最后一个局部窗口块; 这两块强制常驻保留并计入预算 $k$ 内。
  - 网格下方文字: 当前点击选中 block 的分项指标明细。
- 可以做什么: 通过三个按钮切换块摘要打分策略, 并配合两个滑杆调整稀疏度与分布集中度:
  - 「Quest 上界」: 每块记录各通道维度的最小值 kmin 与最大值 kmax, 逐维取乘积正上界算出该块内 $q \cdot k$ 可能达到的理论上限, 保证绝不低估块内高分 token。
  - 「均值 key」: 每块仅存储块内所有 key 的均值向量, 按内积 $q \cdot \mathrm{mean}(K)$ 打分。
  - 「随机选块」: 不做摘要打分、随机抽取其余块作为对照基线。
  - 拖动「读取 block 数 k」修改每步实际访存读取的块数预算。
  - 拖动「针的强度」调节关键 key 与 $q$ 的对齐增益 (即叠加至内积 logit 的增量); 拖至 0 表示移除所有关键针、使注意力退化为均匀弥散分布。
  - 点击「换一组」重新随机生成针块位置与背景噪声。
  - 点击任意方格查看该块的真实注意力质量占比、当前策略打分及 Quest 理论上界。
- 应该看到: 在默认「Quest 上界」策略且 $k = 8$ 时, 仅读取 12.5% 的 KV 块即可完整命中全部 4 个含针块, 注意力质量召回率 (recall) 约达 99%、输出相对误差低于 0.01; 若切换为「随机选块」, 同等预算 $k$ 下几乎无法命中含针块, 输出误差超过 1。当将「针的强度」拖至 0 时, 所有摘要打分策略的误差均退化至与随机选块相当。右侧读数含义如下:
  - 「读取的 KV」为实际访存比例 $k / 64$。
  - 「覆盖的注意力质量 (recall)」统计所选 $k$ 个块在全量注意力下的真实概率权重之和, 超过 90% 时标绿。
  - 「输出相对 L2 误差」度量仅用所选 $k$ 块算出的稀疏注意力输出与全量稠密输出之间的相对 L2 偏差, 低于 0.05 时标绿。
  - 「抓到的针」显示 4 个目标含针块中被成功选中的块数。"
    module="llm_infer/m22"
    run="python -m llm_infer.m22_sparse_attention.demo"
    :challenge="{
      ask: '将「针的强度」拖至 0, 再对比「Quest 上界」打分与「随机选块」的输出误差。此时分块稀疏注意力还能发挥作用吗?',
      answer: '几乎完全失效: 三种策略的输出误差均退化至与随机选块相近, 且注意力质量召回率 (recall) 均跌至接近物理块读取比例本身。\n「针的强度」控制着关键 token 沿查询向量 $q$ 方向叠加的内积增量。当将其拖至 0 时, 上下文内全部 1024 个位置均为同分布随机噪声, 注意力权重均匀弥散在所有 token 上, 根本不存在占主导地位的少数关键 token。在此条件下无论采用何种块摘要估算方法, 读取 1/8 的 KV 块都只能捕获约 1/8 的注意力权重 (recall 读数跌至 13% 至 19% 区间), 输出相对 L2 误差远超 1, 与随机抽样无异。\n分块稀疏 decode 成立的根本前提是「注意力分布高度长尾集中」这一经验规律:\n- 经过充分预训练的真实 LLM: 满足该前提, 绝大多数注意力权重集中在极少数历史位置上, 块摘要打分能够以极小开销准确筛选出这些关键块。\n- 未经训练的随机初始化 TinyLM: 不满足该前提。在 Python demo 对照实验中, 当 $k = 8$ 时 Quest 在随机权重模型上的 recall 仅为 15.3%, 与随机选块 (12.0%) 基本持平, 输出重建误差甚至劣于随机基线。\n此外还需注意 Quest 逐维上界本身的松弛性: 该上界在统计上通常是块内真实最大内积的数倍 (实验台右侧注记会实时显示当前松弛倍数), 纯噪声块也可能算出较高的逐维正上界。因此 Quest 上界仅能提供「高真实内积的目标块不被漏选」的保召回性质, 并不能保证低真实相关度的噪声块一定排在末尾。',
    }"
  >
    <template #controls>
      <div class="row">
        <button v-for="m in SCORERS" :key="m.id" type="button" :class="{ active: scorer === m.id }" @click="scorer = m.id">{{ m.name }}</button>
        <button type="button" @click="seed++">换一组</button>
      </div>
      <LabSlider v-model="k" label="读取 block 数 k" :min="2" :max="64" />
      <LabSlider v-model="strength" label="针的强度 (logit 增量)" :min="0" :max="12" :step="0.5" />
    </template>

    <div class="cells grid">
      <button
        v-for="b in NB" :key="b" type="button" class="cell blk" :class="{ picked: res.picked.has(b - 1), cur: sel === b - 1 }"
        :style="{ background: heat(Math.sqrt(ctx.mass[b - 1] / ctx.maxMass)) }"
        :aria-label="`block ${b - 1}${ctx.needles.includes(b - 1) ? ' (埋了针)' : ''}${res.picked.has(b - 1) ? ', 已选中' : ''}`" :aria-pressed="sel === b - 1" @click="sel = b - 1"
      >{{ ctx.needles.includes(b - 1) ? '针' : b === 1 ? 'S' : b === NB ? '近' : '' }}</button>
    </div>
    <p class="lab-note info">
      点中的 block {{ sel }}: 真实注意力质量 {{ (ctx.mass[sel] * 100).toFixed(1) }}% (全量 attention 里分到的权重) · 当前打分 {{ res.scores[sel].toFixed(2) }}
      · Quest 上界 {{ ctx.quest[sel].toFixed(1) }} ≥ 块内真实 <Tex text="$\max q \cdot k$" /> = {{ ctx.trueMax[sel].toFixed(1) }}
      {{ res.picked.has(sel) ? '· 已选中' : '· 未选中' }}{{ sel === 0 || sel === NB - 1 ? ' (首块或末块, 强制保留)' : '' }}
    </p>

    <template #stats>
      <div class="kv"><span>读取的 KV (k ÷ 64 块)</span><b>{{ (k / NB * 100).toFixed(1) }}%</b></div>
      <div class="kv"><span>覆盖的注意力质量 (recall: 选中块的真实注意力之和)</span><b :class="res.recall > 0.9 ? 'good' : res.recall < 0.5 ? 'bad' : ''">{{ (res.recall * 100).toFixed(1) }}%</b></div>
      <div class="kv"><span>输出相对 L2 误差 (只读选中块 vs 全量)</span><b :class="res.err < 0.05 ? 'good' : res.err > 0.5 ? 'bad' : ''">{{ res.err.toFixed(4) }}</b></div>
      <div class="kv"><span>抓到的针 (选中了几个针块)</span><b :class="res.caught === ctx.needles.length ? 'good' : 'bad'">{{ res.caught }} / {{ ctx.needles.length }}</b></div>
      <div class="lab-note">
        <ul class="pts">
          <li><b>S 和 近:</b> S = 第 0 块 (attention sink, 开头的 token 通常吸走大量注意力), 近 = 最后一块 (最近的 token, 对下一个 token 最相关)。永远选中, 占掉 <Tex text="$k$" /> 里的 2 个名额, 所以 <Tex text="$k = 4$" /> 时只剩 2 个名额给 4 根针。</li>
          <li><b>Quest 摘要:</b> 每块只存逐维的 kmin / kmax 两个向量。打分时每一维取 <Tex text="$q_i \cdot \text{kmin}_i$" /> 和 <Tex text="$q_i \cdot \text{kmax}_i$" /> 中较大的再求和, 所以结果一定不小于块内任何一个真实的 <Tex text="$q \cdot k$" />。上界保证不漏针, 但偏松: 这里平均是块内真实最大值的 {{ ctx.loose.toFixed(1) }}×。</li>
          <li><b>均值 key:</b> 每块只存 key 的均值, 打分 <Tex text="$q \cdot \mathrm{mean}(K)$" />。针只占一块里的 2/16, 会被另外 14 个噪声 key 稀释, 但仍比随机强。</li>
        </ul>
        <p>稀疏加速收益建立在注意力高度集中的前提之上, 这一特性仅在训练收敛的 LLM 中涌现; 当针的强度设为 0 时, 各类摘要打分策略的表现均与随机选块无异。</p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import Tex from '@/components/Tex.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { mulberry32, randn, range, softmax, sum, heat } from '@/utils/labmath.js'

const T = 1024, BS = 16, NB = T / BS, DIM = 32, N_NEEDLE = 4, PER = 2
const SCORERS = [{ id: 'quest', name: 'Quest 上界' }, { id: 'mean', name: '均值 key' }, { id: 'random', name: '随机选块' }]

const scorer = ref('quest')
const k = ref(8)
const strength = ref(10)
const seed = ref(1)
const sel = ref(0)

// 只跟 seed 走的部分: q, 噪声 K/V, 针的位置, 随机分数
const base = computed(() => {
  const rand = mulberry32(seed.value * 15485863)
  const vec = (n) => Float64Array.from({ length: n }, () => randn(rand))
  const q = vec(DIM), K = vec(T * DIM), V = vec(T * DIM)
  const pool = range(NB - 2).map((i) => i + 1) // 针不放在首 / 末块
  const needles = range(N_NEEDLE).map(() => pool.splice(Math.floor(rand() * pool.length), 1)[0]).sort((a, b) => a - b)
  const pos = needles.flatMap((b) => { const p = range(BS); return range(PER).map(() => b * BS + p.splice(Math.floor(rand() * p.length), 1)[0]) })
  return { q, K, V, needles, pos, rnd: range(NB).map(() => rand()) }
})

// 加上针之后的上下文 + 所有"真值" (真实 attention, 完整输出) + 两种摘要打分
const ctx = computed(() => {
  const { q, V, needles, pos, rnd } = base.value
  const K = base.value.K.slice(), qq = sum(Array.from(q, (x) => x * x))
  for (const p of pos) for (let i = 0; i < DIM; i++) K[p * DIM + i] += q[i] * (strength.value * Math.sqrt(DIM) / qq) // 使 q·k/√d 增加 strength
  const qk = range(T).map((t) => { let s = 0; for (let i = 0; i < DIM; i++) s += q[i] * K[t * DIM + i]; return s })
  const logits = qk.map((x) => x / Math.sqrt(DIM)), probs = softmax(logits)
  const out = (idx, p) => range(DIM).map((i) => idx.reduce((a, t, j) => a + p[j] * V[t * DIM + i], 0))
  const blocks = range(NB).map((b) => range(BS).map((j) => b * BS + j))
  const mass = blocks.map((ts) => sum(ts.map((t) => probs[t])))
  const trueMax = blocks.map((ts) => Math.max(...ts.map((t) => qk[t])))
  // ★ Quest: 逐维 max(q_i·kmin_i, q_i·kmax_i) 求和 —— q_i 正取 kmax, 负取 kmin, 所以 ≥ 块内任何 q·k
  const quest = blocks.map((ts) => sum(range(DIM).map((i) => {
    const col = ts.map((t) => K[t * DIM + i])
    return Math.max(q[i] * Math.min(...col), q[i] * Math.max(...col))
  })))
  const mean = blocks.map((ts) => sum(ts.map((t) => qk[t])) / BS) // q·mean(K_block)
  const loose = sum(quest.map((u, b) => u / Math.abs(trueMax[b]))) / NB
  return { V, logits, mass, maxMass: Math.max(...mass), trueMax, quest, mean, random: rnd, needles, full: out(range(T), probs), out, loose }
})

const res = computed(() => {
  const c = ctx.value, scores = c[scorer.value]
  const forced = [0, NB - 1] // select_blocks: 首块 (sink) + 末块 (最近) 强制保留, 算在 k 内
  const rest = range(NB).filter((b) => !forced.includes(b)).sort((a, b) => scores[b] - scores[a] || a - b)
  const picked = new Set([...forced, ...rest.slice(0, Math.max(k.value - 2, 0))])
  const idx = range(T).filter((t) => picked.has(Math.floor(t / BS)))
  const sparse = c.out(idx, softmax(idx.map((t) => c.logits[t]))) // 只在选中的 token 上重新 softmax
  const norm = (v) => Math.sqrt(sum(v.map((x) => x * x)))
  return {
    scores, picked, err: norm(sparse.map((x, i) => x - c.full[i])) / norm(c.full),
    recall: sum([...picked].map((b) => c.mass[b])), caught: c.needles.filter((b) => picked.has(b)).length,
  }
})
</script>

<style scoped>
.grid { grid-template-columns: repeat(16, minmax(0, 1fr)); max-width: 560px; }
.blk { min-width: 0; min-height: 0; height: 30px; padding: 0; font-size: 10px; color: var(--text); transition: none; }
.blk.picked { border: 2px solid var(--left); }
.blk.cur { outline: 2px solid var(--accent); outline-offset: 1px; }
.info { margin-top: 10px; min-height: 3.4em; }
</style>
