<!-- 稀疏 decode 实验台 (llm_infer/m22): needle_context + quest_upper_bound / mean_score + select_blocks + evaluate 的 JS 移植。 -->
<template>
  <LabFrame
    title="稀疏注意力 decode — 只读打分最高的 k 个 KV block, 看丢了多少注意力"
    sub="- 模拟什么: decode 每生成 1 个 token, 都要把上下文里全部 token 的 KV 读一遍, 上下文越长越慢, 这是纯访存开销。但训练好的 LLM 的注意力高度集中, 绝大部分权重落在少数 token 上。块稀疏 decode 的做法: 把 KV 按位置切成 block, 每块常驻一份很小的摘要, 每步用当前 query $q$ 给摘要打分, 只读分数最高的 $k$ 块做 attention, 其余跳过。这个台子有 1024 个 token 的 KV, 切成 64 块 (每块 16 个), 其中 4 块各埋了 2 根「针」: 和 $q$ 方向对齐的 key, 模拟真正重要的 token; 其余 key 都是噪声。
      - 图上是什么: 64 个格子按位置排, 一格一个 block。格子越亮, 这个 block 在全量 attention 里真实分到的注意力越多 (这是事后算的真值, decode 时并不知道)。带绿框的是本次被选中、会被读取的块; 写着「针」的是埋了针的块; S 是第 0 块 (attention sink: 开头的 token 通常吸走大量注意力), 「近」是最后一块 (最近的 token), 这两块永远保留, 算在 $k$ 内。格子下面一行是你点中的那个 block 的详情。
      - 可以做什么: 三个按钮切打分方式: 「Quest 上界」每块存逐维的 kmin / kmax, 算出 $q \cdot k$ 在这块里可能达到的最大值, 保证不低估; 「均值 key」每块只存 key 的均值, 算 $q \cdot \mathrm{mean}(K)$; 「随机选块」不打分, 作为对照。拖「读取 block 数 k」改每步读几块; 拖「针的强度」改针和 $q$ 对齐的程度 (加到 logit 上的增量), 拖到 0 就是没有针、注意力均匀弥散。「换一组」换针的位置和随机数。点任意格子看它的真实注意力质量、当前打分和 Quest 上界。
      - 应该看到: 右侧「读取的 KV」是 $k / 64$; 「覆盖的注意力质量 (recall)」是选中块的真实注意力加起来占多少, 超过 90% 标绿; 「输出相对 L2 误差」是只读这 $k$ 块算出的 attention 输出和全量输出的差距, 低于 0.05 标绿; 「抓到的针」是 4 个针块里选中了几个。默认 Quest、$k = 8$ 时, 读 12.5% 的 KV 就能抓到 4 根针、recall 99% 左右、误差不到 0.01; 切到「随机选块」, 同样的 $k$ 几乎抓不到针, 误差超过 1。把「针的强度」拖到 0, 三种打分的误差都和随机差不多。"
    module="llm_infer/m22"
    run="python -m llm_infer.m22_sparse_attention.demo"
    :challenge="{
      ask: '把「针的强度」拖到 0, 再比较 Quest 打分和随机选块的误差。稀疏注意力还有用吗?',
      answer: '几乎没用: 三种打分的误差都接近随机选块, recall 都约等于读取比例。\n「针的强度」是加到针这几个 key 上、让它们的 $q \\cdot k$ 变大的增量。拖到 0 时所有 key 都是噪声, 每个 token 的注意力权重差不多, 均匀弥散在 1024 个 token 上。这时不存在「少数重要 token」, 读 1/8 的 KV 就只能拿到约 1/8 的注意力质量 (recall 读数掉到 13% 到 19%), 输出相对误差远超 1, 怎么打分都和随机差不多。\n稀疏 decode 的收益完全来自「注意力高度集中」这个经验事实, 不是算法自己创造出来的:\n- 训练过的 LLM: 成立, 大部分注意力落在少数 token 上, 摘要打分能把它们找出来。\n- 随机权重的 TinyLM: 不成立。Python demo 里 $k = 8$ 时 Quest 的 recall 只有 15.3%, 和随机 (12.0%) 差不多, 误差甚至不如随机。\n另外注意 Quest 的上界本身很松: 这里它平均是块内真实最大值的好几倍 (右侧说明里有当前的倍数), 噪声块的上界也不小。它保证的是「真正高分的块不会被漏掉」, 不保证低分块一定排在后面。',
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
          <li><b>S 和 近:</b> S = 第 0 块 (attention sink, 开头的 token 常年吸走大量注意力), 近 = 最后一块 (最近的 token, 对下一个 token 最相关)。永远选中, 占掉 <Tex text="$k$" /> 里的 2 个名额, 所以 <Tex text="$k = 4$" /> 时只剩 2 个名额给 4 根针。</li>
          <li><b>Quest 摘要:</b> 每块只存逐维的 kmin / kmax 两个向量。打分时每一维取 <Tex text="$q_i \cdot \text{kmin}_i$" /> 和 <Tex text="$q_i \cdot \text{kmax}_i$" /> 中较大的再求和, 所以结果一定不小于块内任何一个真实的 <Tex text="$q \cdot k$" />。上界保证不漏针, 但偏松: 这里平均是块内真实最大值的 {{ ctx.loose.toFixed(1) }}×。</li>
          <li><b>均值 key:</b> 每块只存 key 的均值, 打分 <Tex text="$q \cdot \mathrm{mean}(K)$" />。针只占一块里的 2/16, 会被另外 14 个噪声 key 稀释, 但仍比随机强。</li>
        </ul>
        <p>收益的前提是注意力集中, 训练过的 LLM 才有。强度 = 0 时, 打分几乎不比随机强。</p>
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
