<template>
  <div>
    <h1 class="page-title">MoE 路由可视化</h1>
    <div class="page-subtitle lead-group">
      <p>想让模型更懂, 最直接的办法是把 FFN 做大。但 FFN 一大, 每个 token 的算力就跟着涨。</p>
      <p>
        MoE 放 E 个小 FFN, 每个 token 只过其中 top-k 个: 参数是单个小 FFN 的 E 倍, 每 token 算力只有 k 倍。
        容量和算力就此解耦。
      </p>
      <p>
        本页把两套路由哲学并排放:
        <strong>Mixtral (softmax + 外挂 aux loss)</strong> 对 <strong>DeepSeek (sigmoid + 共享专家 + aux-free bias)</strong>。
      </p>
    </div>

    <ChapterIntro
      tldr="MoE 就是把 ffn 槽位换成「E 个小 FFN + 一个 router」。Mixtral 和 DeepSeek 不是新旧版本, 是两套不同的路由哲学。"
      question="专家一多, 怎么防止 router 把所有 token 都塞给那几个明星专家? 又怎么不让「负载均衡」这件事污染语言建模的 loss?"
      :goals="[
        '看懂一个 FFN 是怎么被换成 router 加 E 个小 FFN 的',
        '分清 Mixtral 的 softmax top-k 和 DeepSeek 的 sigmoid + 共享专家',
        '说清 aux-free 的偏置怎么在不碰门控权重的前提下把负载拉平',
      ]"
      :codes="[
        { path: 'llm_models/layers/sparse/moe.py', label: 'moe.py · MixtralMoE' },
        { path: 'llm_models/models/moe/deepseekV3.py', label: 'deepseekV3.py · DeepSeekMoE' },
      ]"
      :prereq="prevChapter"
      :next-step="nextChapter"
    />

    <!-- 控制条 -->
    <div class="card" style="margin-bottom: 20px;">
      <div class="controls">
        <div class="form-row">
          <span id="moe-variant-label" class="row-label">变体</span>
          <div class="btn-group" style="grid-column: span 2;" role="group" aria-labelledby="moe-variant-label">
            <button type="button" :class="{ active: variant === 'mixtral' }" :aria-pressed="variant === 'mixtral'" @click="variant = 'mixtral'">Mixtral (softmax)</button>
            <button type="button" :class="{ active: variant === 'deepseek' }" :aria-pressed="variant === 'deepseek'" @click="variant = 'deepseek'">DeepSeek (sigmoid)</button>
          </div>
        </div>
        <div class="form-row">
          <label for="moe-num-experts">专家总数 E</label>
          <input id="moe-num-experts" type="range" min="4" :max="MAX_EXPERTS" step="2" v-model.number="numExperts" />
          <span class="val">{{ numExperts }}</span>
        </div>
        <div class="form-row">
          <label for="moe-top-k">top-k</label>
          <input id="moe-top-k" type="range" :min="1" :max="topKMax" step="1" v-model.number="topK" />
          <span class="val">{{ topK }}</span>
        </div>
        <div v-if="variant === 'deepseek'" class="form-row">
          <label for="moe-num-shared">共享专家</label>
          <input id="moe-num-shared" type="range" min="0" max="4" step="1" v-model.number="numShared" />
          <span class="val">{{ numShared }}</span>
        </div>
        <div class="form-row">
          <label for="moe-num-tokens">Token 数</label>
          <input id="moe-num-tokens" type="range" min="8" :max="MAX_TOKENS" step="2" v-model.number="numTokens" />
          <span class="val">{{ numTokens }}</span>
        </div>
        <div class="form-row">
          <label for="moe-collapse-bias">路由坍缩倾向</label>
          <input id="moe-collapse-bias" type="range" min="0" max="100" step="1" v-model.number="collapseBias" />
          <span class="val">{{ collapseBias }}%</span>
        </div>
      </div>

      <div style="display: flex; gap: 8px; margin-top: 12px;">
        <button type="button" @click="seed++">重新采样 tokens</button>
        <button type="button" @click="animateFlow" :disabled="animating">▷ 播放路由动画</button>
      </div>
      <p class="desc" style="margin-top: 10px;">
        拖滑杆时 token 的底分不变, 只有「重新采样」会换一批。
      </p>
      <p class="desc" style="margin-top: 6px;">
        把「路由坍缩倾向」拉到 100%, 大部分 token 都往前两个专家挤。负载标准差变红, 其余专家白占着显存。
      </p>
      <p class="desc" style="margin-top: 6px;">
        真实训练里这是个正反馈: 被选中得多的专家学得更好, 于是更容易被选中。所以 MoE 必须配一套均衡机制。
      </p>
    </div>

    <!-- 主可视化: 左 tokens → 中 router → 右 experts -->
    <div class="card">
      <svg :viewBox="`0 0 ${W} ${H}`" width="100%" :height="H" role="img"
           :aria-label="`${numTokens} 个 token 经 router 分给 ${numExperts} 个专家, 每个 token 选 top-${topK}。负载标准差 ${loadStd.toFixed(2)}, 最热门的是 E${hottestExpert.idx}`">

        <!-- Tokens -->
        <g>
          <text x="50" y="24" fill="var(--text-muted)" font-size="12" font-family="SF Mono">tokens (batch)</text>
          <g v-for="(t, i) in tokens" :key="'t-'+i">
            <rect :x="20" :y="40 + i * tokenGap"
                  width="80" height="22" rx="4"
                  fill="var(--bg-elev)" stroke="var(--border)" />
            <text :x="60" :y="56 + i * tokenGap"
                  text-anchor="middle" fill="var(--text)" font-size="11" font-family="SF Mono">
              t{{ i }}
            </text>
          </g>
        </g>

        <!-- Router -->
        <g>
          <rect :x="routerX" :y="40" width="100" :height="H - 60" rx="6"
                fill="var(--bg-elev)" stroke="var(--accent)" stroke-opacity="0.6" />
          <text :x="routerX + 50" y="30" text-anchor="middle" fill="var(--accent)" font-size="12" font-weight="600">Router</text>
          <text :x="routerX + 50" :y="H - 10" text-anchor="middle"
                fill="var(--text-dim)" font-size="10" font-family="SF Mono">
            {{ variant === 'mixtral' ? 'softmax' : 'sigmoid' }}
          </text>
        </g>

        <!-- 路由连线 -->
        <g>
          <path v-for="(edge, i) in edges" :key="'e-'+i"
                :d="edge.d"
                :stroke="edge.color"
                :stroke-opacity="edge.opacity"
                :stroke-width="edge.width"
                fill="none"
                :class="{ flowing: animating }" />
        </g>

        <!-- Experts -->
        <g>
          <g v-for="(e, i) in displayedExperts" :key="'exp-'+i">
            <rect :x="expertX" :y="40 + i * expertGap"
                  :width="140" :height="expertGap - 8"
                  rx="4"
                  :style="{ fill: e.shared ? 'color-mix(in srgb, var(--eye) 8%, transparent)' : 'var(--bg-elev)' }"
                  :stroke="e.shared ? 'var(--eye)' : (e.load > 0 ? 'var(--accent)' : 'var(--border)')"
                  :stroke-opacity="e.shared ? 0.8 : (0.4 + Math.min(1, e.load / maxLoad) * 0.6)" />
            <text :x="expertX + 10" :y="55 + i * expertGap"
                  fill="var(--text)" font-size="11" font-family="SF Mono">
              {{ e.shared ? 'Shared' : 'E' + i }}
            </text>
            <!-- load bar -->
            <rect :x="expertX + 60" :y="48 + i * expertGap"
                  :width="60" height="6" rx="2" fill="var(--border)" />
            <rect :x="expertX + 60" :y="48 + i * expertGap"
                  :width="60 * Math.min(1, e.load / maxLoad)" height="6" rx="2"
                  :fill="e.shared ? 'var(--eye)' : e.load > avgLoad * 1.8 ? 'var(--danger)' : 'var(--accent)'" />
            <text :x="expertX + 128" :y="54 + i * expertGap"
                  fill="var(--text-muted)" font-size="10" font-family="SF Mono"
                  text-anchor="end">
              {{ e.load }}
            </text>
          </g>
          <text :x="expertX + 70" y="24" text-anchor="middle" fill="var(--text-muted)" font-size="12" font-family="SF Mono">experts (load)</text>
        </g>
      </svg>

      <!-- 这张图没模拟的东西要写明, 否则读者会以为 DeepSeek 的均衡没用 -->
      <div class="sim-note">
        <p>
          切换 Mixtral / DeepSeek, 这张图只换了打分函数 (softmax 换成 sigmoid), 再加上共享专家。
          两个函数都单调递增, top-k 选出的专家相同, 所以路由专家的负载不变。
        </p>
        <p>
          本图不模拟均衡, 效果见<router-link :to="{ name: 'models-moe-balance' }">「Aux-loss-free 均衡」一章</router-link>。
        </p>
      </div>

      <!-- 负载分析 -->
      <div class="load-analysis" aria-live="polite">
        <div class="stat">
          <div class="k">负载标准差</div>
          <div class="v" :style="{ color: loadStdColor }">{{ loadStd.toFixed(2) }}</div>
          <div class="hint">越小越均衡; 拖「路由坍缩倾向」看它变红</div>
        </div>
        <div class="stat">
          <div class="k">最热门的专家</div>
          <div class="v">E{{ hottestExpert.idx }}</div>
          <div class="hint">一个人扛了 {{ Math.round(hottestExpert.load / totalRoutedAssigns * 100) }}% 的路由</div>
        </div>
        <div class="stat">
          <div class="k">激活比例</div>
          <div class="v accent">{{ (topK / numExperts * 100).toFixed(1) }}%</div>
          <div class="hint">top-k / 总专家 = 每个 token 实际用掉的算力占比</div>
        </div>
      </div>
    </div>

    <!-- 对比表 -->
    <section class="section">
      <h2>Mixtral 和 DeepSeek 的 MoE 差在哪</h2>
      <div class="lead-group">
        <p>每一行都是一个被踩过的坑:</p>
        <ul class="pts">
          <li>专家太大, 就没法分工。</li>
          <li>没有共享专家, 每个专家都要重学一遍通用能力。</li>
          <li>不做均衡, 就会滚雪球滚成路由坍缩。</li>
          <li>用 aux loss 做均衡, 又会跟语言建模抢梯度。</li>
        </ul>
        <p>DeepSeek 的每一列都是对左边那一列的回答。</p>
      </div>
      <div class="card">
        <table class="compare">
          <thead>
            <tr>
              <th></th>
              <th>Mixtral (2024)</th>
              <th>DeepSeek-V3 (2024)</th>
            </tr>
          </thead>
          <tbody>
            <tr><td class="row-label">路由打分</td>
              <td><code class="inline">softmax(logits)</code></td>
              <td><code class="inline">sigmoid(logits)</code></td>
            </tr>
            <tr><td class="row-label">专家归一化</td>
              <td>softmax 本身归一</td>
              <td>top-k 后再 renormalize</td>
            </tr>
            <tr><td class="row-label">共享专家</td>
              <td>❌ 没有</td>
              <td>✅ 每个 token 都过, 专管通用知识, 路由专家才腾得出容量去分化</td>
            </tr>
            <tr><td class="row-label">负载均衡</td>
              <td>Switch 那套外挂 aux loss, 梯度直接压热门专家的路由分数</td>
              <td>aux-loss-free bias: 不收梯度, 只改 top-k 选谁 (demo 实测负载 CV 从 0.273 降到 0.124, LM loss 几乎不动: 2.979 vs 2.982)</td>
            </tr>
            <tr><td class="row-label">专家粒度</td>
              <td>8 个大专家</td>
              <td>64–256 个细粒度专家</td>
            </tr>
            <tr><td class="row-label">代码位置</td>
              <td><RepoLink path="llm_models/layers/sparse/moe.py" label="layers/sparse/moe.py" tiny /></td>
              <td><RepoLink path="llm_models/models/moe/deepseekV3.py" label="models/moe/deepseekV3.py" tiny /></td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <!-- 本章挂载的实验台 (data/labmap/*.js) 与章末自测 (data/quiz/*.js), 没配置时不渲染 -->

    <LabMount />

    <QuizCard />


    <ChapterNav
      :prev="{ ...prevChapter, hint: 'MoE 只是 ffn 槽位的一种填法' }"
      :next="nextChapter"
    />
  </div>
</template>

<script setup>
import LabMount from '@/components/LabMount.vue'
import QuizCard from '@/components/QuizCard.vue'
import { ref, computed, watch } from 'vue'
import { learningPath } from '@/data/models.js'
import { mulberry32 } from '@/utils/labmath.js'
import ChapterIntro from '@/components/ChapterIntro.vue'
import ChapterNav from '@/components/ChapterNav.vue'
import RepoLink from '@/components/RepoLink.vue'

// 上一章 / 下一章从 learningPath 取, 不手写章名和编号 (与 Infer.vue 同一写法)
const at = learningPath.findIndex((x) => x.route === 'moe')
const prevChapter = { name: learningPath[at - 1].route, label: `上一章 · ${learningPath[at - 1].label}` }
const nextChapter = { name: learningPath[at + 1].route, label: `下一章 · ${learningPath[at + 1].label}` }

// 两根滑杆的上限, 底分表按这个尺寸一次生成
const MAX_EXPERTS = 32
const MAX_TOKENS = 32

const variant = ref('deepseek')
const numExperts = ref(8)
const topK = ref(2)
// top-k 滑杆的上限跟着专家数走。专家数调小时把 topK 钳回上限以内, 否则读数会停在滑杆够不到的值上
const topKMax = computed(() => Math.max(2, numExperts.value / 2))
watch(topKMax, (m) => { if (topK.value > m) topK.value = m })
const numShared = ref(2)
const numTokens = ref(16)
const collapseBias = ref(30)  // 0–100: 越大越倾向路由坍缩
const seed = ref(1)           // 只有「重新采样」会改它

const animating = ref(false)

// 画布几何
const W = 900
const tokenGap = 28
const expertGap = 32
const tokenX = 100
const routerX = 280
const expertX = 520

const H = computed(() => Math.max(numTokens.value, numExperts.value + numShared.value) * Math.max(tokenGap, expertGap) + 80)

// --- Tokens 与路由计算 ---
// ★ 每个 (token, 专家) 的底分只由 seed 决定, 按滑杆上限一次生成。
//   拖滑杆只是从这张表里取一块, 所以图不会乱跳, 看得出是哪个量在起作用。
const baseScores = computed(() => {
  const rand = mulberry32(seed.value)
  return Array.from({ length: MAX_TOKENS }, () =>
    Array.from({ length: MAX_EXPERTS }, () => rand() + (rand() - 0.5) * 0.3))
})

// 坍缩倾向越大, 前 2 个专家的分数被抬得越高, 多数 token 都往它们身上挤
const tokens = computed(() => {
  const bias = (collapseBias.value / 100) * 1.8
  return baseScores.value.slice(0, numTokens.value).map((row) => ({
    scores: row.slice(0, numExperts.value).map((base, e) => base + (e < 2 ? bias : 0)),
  }))
})

// --- 路由决策 ---
const routingDecisions = computed(() => {
  return tokens.value.map((t, i) => {
    let probs
    if (variant.value === 'mixtral') {
      // softmax
      const max = Math.max(...t.scores)
      const exp = t.scores.map(s => Math.exp(s - max))
      const sum = exp.reduce((a, b) => a + b, 0)
      probs = exp.map(e => e / sum)
    } else {
      // sigmoid independent
      probs = t.scores.map(s => 1 / (1 + Math.exp(-s)))
    }
    // top-k
    const withIdx = probs.map((p, e) => ({ p, e }))
    const top = withIdx.sort((a, b) => b.p - a.p).slice(0, topK.value)
    // renormalize
    const totalP = top.reduce((a, x) => a + x.p, 0)
    const weights = top.map(x => ({ e: x.e, w: x.p / totalP }))
    return { token: i, weights }
  })
})

// 专家负载
const displayedExperts = computed(() => {
  const E = numExperts.value
  const shared = variant.value === 'deepseek' ? numShared.value : 0
  const result = []
  // routed first
  for (let e = 0; e < E; e++) {
    const load = routingDecisions.value.reduce((acc, r) =>
      acc + r.weights.filter(w => w.e === e).length, 0)
    result.push({ idx: e, load, shared: false })
  }
  // shared
  for (let s = 0; s < shared; s++) {
    result.push({ idx: 's' + s, load: numTokens.value, shared: true })
  }
  return result
})

const maxLoad = computed(() => Math.max(1, ...displayedExperts.value.map(e => e.load)))
const avgLoad = computed(() =>
  (numTokens.value * topK.value) / numExperts.value
)
const loadStd = computed(() => {
  const routed = displayedExperts.value.filter(e => !e.shared)
  const mean = routed.reduce((a, e) => a + e.load, 0) / routed.length
  const variance = routed.reduce((a, e) => a + Math.pow(e.load - mean, 2), 0) / routed.length
  return Math.sqrt(variance)
})
const loadStdColor = computed(() => {
  const v = loadStd.value
  if (v < 1) return 'var(--left)'
  if (v < 3) return 'var(--warn)'
  return 'var(--danger)'
})
const hottestExpert = computed(() => {
  const routed = displayedExperts.value.filter(e => !e.shared)
  return routed.reduce((max, e) => e.load > max.load ? e : max, { load: 0, idx: 0 })
})
const totalRoutedAssigns = computed(() => numTokens.value * topK.value)

// --- 边的绘制 ---
const edges = computed(() => {
  const result = []
  routingDecisions.value.forEach((r, ti) => {
    const tx = tokenX + 0  // right side of token
    const ty = 40 + ti * tokenGap + 11
    const rx1 = routerX
    // Token → router
    result.push({
      d: `M ${tx} ${ty} L ${rx1} ${ty}`,
      color: 'var(--border-strong)',
      opacity: 0.4,
      width: 1,
    })
    // Router → selected experts
    r.weights.forEach(({ e, w }) => {
      const ex = expertX
      const ey = 40 + e * expertGap + 11
      const mid = (routerX + 100 + ex) / 2
      result.push({
        d: `M ${routerX + 100} ${ty} C ${mid} ${ty}, ${mid} ${ey}, ${ex} ${ey}`,
        color: 'var(--accent)',
        opacity: 0.2 + w * 0.6,
        width: 1 + w * 2,
      })
    })
    // Shared expert connections (always on, dashed)
    if (variant.value === 'deepseek') {
      for (let s = 0; s < numShared.value; s++) {
        const ex = expertX
        const ey = 40 + (numExperts.value + s) * expertGap + 11
        const mid = (routerX + 100 + ex) / 2
        result.push({
          d: `M ${routerX + 100} ${ty} C ${mid} ${ty}, ${mid} ${ey}, ${ex} ${ey}`,
          color: 'var(--eye)',
          opacity: 0.15,
          width: 1,
        })
      }
    }
  })
  return result
})

async function animateFlow() {
  animating.value = true
  await new Promise(r => setTimeout(r, 1200))
  animating.value = false
}
</script>

<style scoped>
.row-label { font-size: 13px; color: var(--text-muted); }
.sim-note {
  margin-top: 12px;
  padding-left: 10px;
  border-left: 2px solid var(--accent);
  color: var(--text-muted);
  font-size: 12.5px;
  line-height: 1.7;
}
.sim-note p + p { margin-top: 6px; }

.controls {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px 24px;
}
@media (max-width: 960px) {
  .controls { grid-template-columns: 1fr; }
}

.load-analysis {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 12px;
  margin-top: 16px;
}

table.compare {
  width: 100%;
  border-collapse: collapse;
}
table.compare th, table.compare td {
  padding: 10px 12px;
  text-align: left;
  border-bottom: 1px solid var(--border);
  font-size: 13px;
}
table.compare th {
  color: var(--text-muted);
  font-size: 11px;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.6px;
}
table.compare .row-label {
  color: var(--text-muted);
  font-size: 12px;
  width: 130px;
}

.flowing {
  stroke-dasharray: 4 4;
  animation: flow 1s linear infinite;
}
@keyframes flow {
  to { stroke-dashoffset: -16; }
}
</style>
