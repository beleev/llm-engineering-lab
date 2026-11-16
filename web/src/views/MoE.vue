<template>
  <div>
    <h1 class="page-title">MoE 路由可视化</h1>
    <div class="page-subtitle lead-group">
      <p>
        Transformer 的每一层里都有一个 FFN (feed-forward network, 前馈网络): 几个矩阵乘法, 每个 token 单独过一遍。
        模型记住的「知识」大部分存在 FFN 的参数里, 所以想让模型更懂, 最直接的办法是把 FFN 做大。
      </p>
      <p>
        问题是稠密 FFN 里每个参数对每个 token 都要算一遍: FFN 参数翻倍, 每个 token 的算力也翻倍。
      </p>
      <p>
        MoE (Mixture of Experts, 混合专家) 的做法是把一个大 FFN 换成 E 个小 FFN, 每个小 FFN 叫一个「专家」。
        再加一个叫 router 的小线性层, 给每个 token 对每个专家打一个分, 只把 token 送进分数最高的 k 个专家, 这一步叫 top-k。
        参数量按 E 个专家算, 每个 token 的算力只按 k 个算。容量和算力就这样拆开了。
      </p>
      <p>
        本页把两套路由设计并排放:
        <strong>Mixtral (softmax 打分 + 外挂 aux loss)</strong> 对 <strong>DeepSeek (sigmoid 打分 + 共享专家 + aux-free bias)</strong>。
        这几个名词下面会逐个解释。先说两个: aux loss 是为了让各专家接到的 token 数差不多, 而额外加进总 loss 的一项;
        aux-free 就是不加这一项, 改用一个不走梯度的偏置 bias 去调整「选谁」。
      </p>
    </div>

    <ChapterIntro
      tldr="MoE 把 Transformer 里的 FFN 槽位换成「E 个小 FFN (专家) + 一个 router」, router 给每个 token 挑 k 个专家。参数量按 E 算, 每个 token 的算力按 k 算。
        - Mixtral: softmax 打分, 8 个大专家每个 token 选 2 个; 负载均衡靠额外加进总 loss 的 aux loss。
        - DeepSeek-V3: sigmoid 打分, 64–256 个细粒度专家, 外加每个 token 都必过的共享专家; 负载均衡靠一个不走梯度的 bias。本库 demo 里这个 bias 把负载变异系数从 0.273 压到 0.124, LM loss 几乎不动 (2.979 对 2.982)。
        两者不是新旧版本, 是两套不同的路由哲学。"
      question="专家一多, router 很容易把大部分 token 都塞给那几个「明星专家」, 怎么防? 又怎么让「把负载拉平」这件事不去干扰模型学语言本身的 loss?"
      :goals="[
        '看懂一个 FFN 是怎么被换成 router 加 E 个小 FFN 的, 以及一个 token 从 router 到专家再加回来走了哪几步',
        '分清 Mixtral 的 softmax top-k 和 DeepSeek 的 sigmoid + 共享专家, 说出两种打分各自意味着什么',
        '说清 aux-free 的偏置怎么在不碰门控权重的前提下把负载拉平',
      ]"
      :codes="[
        { path: 'llm_models/layers/sparse/moe.py', label: 'moe.py · MixtralMoE' },
        { path: 'llm_models/models/moe/deepseekV3.py', label: 'deepseekV3.py · DeepSeekMoE' },
      ]"
      :prereq="prevChapter"
      :next-step="nextChapter"
    />

    <EvolutionChain
      title="演进逻辑链 · 容量和算力拆开记账"
      subtitle="根问题: 稠密 FFN 里每个参数对每个 token 都要算一遍, 想多装知识就得加参数, 每个 token 的算力跟着同比上涨"
      :steps="evoSteps"
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
        <strong>模拟什么:</strong> 一层 MoE 里的 router 在给一个 batch 的 token 分配专家。
        每个 token 对每个专家有一个底分 (真实模型里是 router 这个线性层算出来的 logit, 这里用固定随机种子生成),
        打分函数把底分变成概率, 再挑概率最高的 top-k 个专家。
        拖滑杆时 token 的底分不变, 只有「重新采样 tokens」会换一批底分。
      </p>
      <p class="desc" style="margin-top: 6px;">
        <strong>图上是什么:</strong> 下面那张图左边一列是 batch 里的 token (t0, t1, …), 中间的 Router 框是打分函数, 右边一列是专家 E0, E1, …。
        每条从 Router 连到专家的线表示一个 token 被送进了那个专家; 线越粗越实, 这个专家在该 token 的 k 个权重里占比越大。
        每个专家旁边的小横条是它的负载, 就是这一批里有多少个 token 选了它; 负载超过平均值 1.8 倍的横条变红。
        切到 DeepSeek 时最下面多出的 Shared 行是共享专家, 用虚线连到所有 token, 因为每个 token 都必须过它。
      </p>
      <p class="desc" style="margin-top: 6px;">
        <strong>可以做什么:</strong> 「变体」在 Mixtral 的 softmax 和 DeepSeek 的 sigmoid 之间切换打分函数;
        「专家总数 E」改专家有几个; 「top-k」改每个 token 送进几个专家 (上限是 E 的一半);
        「共享专家」只在 DeepSeek 下出现, 改必过专家的个数; 「Token 数」改这一批有几个 token;
        「路由坍缩倾向」把前两个专家 E0、E1 的底分统一抬高, 拉得越大, 越多 token 往它们俩挤。
        「播放路由动画」让连线流动一遍, 方便看清每个 token 去了哪。
      </p>
      <p class="desc" style="margin-top: 6px;">
        <strong>应该看到:</strong> 把「路由坍缩倾向」拉到 100%, 大部分 token 都往 E0、E1 挤, 它们的横条变红, 其余专家几乎没有连线, 却仍然占着显存。
        右下角的「负载标准差」跟着变红: 它是各路由专家负载的标准差, 越小表示 token 分得越平均。
        真实训练里这是个正反馈: 被选中得多的专家学得更好, 于是更容易被选中, 最后少数专家包揽几乎所有 token。
        所以 MoE 必须配一套均衡机制。
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
          这张图模拟的只有「打分 + 选 top-k」这一步。切换 Mixtral / DeepSeek, 变的只有打分函数 (softmax 换成 sigmoid) 和多出来的共享专家。
          softmax 和 sigmoid 都是单调递增函数, 底分高的专家换算后还是高, 所以同一批底分下 top-k 选出的专家完全相同, 路由专家的负载也就不变。
          两者的真正区别是: softmax 让一个 token 对所有专家的分数加起来等于 1, 专家之间互相竞争;
          sigmoid 给每个专家独立打一个 0 到 1 之间的分, 互不影响。这个区别要到训练时才体现出来。
        </p>
        <p>
          本图也不模拟任何均衡机制: 没有 aux loss, 也没有 DeepSeek 的 bias。所以把变体切到 DeepSeek, 坍缩并不会自己消失。
          bias 怎么一步步把负载拉平, 见<router-link :to="{ name: 'models-moe-balance' }">「Aux-loss-free 均衡」一章</router-link>的实验台。
        </p>
      </div>

      <!-- 负载分析 -->
      <div class="load-analysis" aria-live="polite">
        <div class="stat">
          <div class="k">负载标准差</div>
          <div class="v" :style="{ color: loadStdColor }">{{ loadStd.toFixed(2) }}</div>
          <div class="hint">各路由专家负载 (接到的 token 数) 的标准差, 越小越均衡。小于 1 绿色, 1 到 3 黄色, 3 以上红色; 拖「路由坍缩倾向」看它变红</div>
        </div>
        <div class="stat">
          <div class="k">最热门的专家</div>
          <div class="v">E{{ hottestExpert.idx }}</div>
          <div class="hint">它一个接了全部路由次数 (Token 数 × top-k) 的 {{ Math.round(hottestExpert.load / totalRoutedAssigns * 100) }}%</div>
        </div>
        <div class="stat">
          <div class="k">激活比例</div>
          <div class="v accent">{{ (topK / numExperts * 100).toFixed(1) }}%</div>
          <div class="hint">top-k 除以专家总数 E: 每个 token 实际用到的路由专家参数占全部路由专家参数的比例, 也就是省算力的程度 (不含共享专家和注意力)</div>
        </div>
      </div>
    </div>

    <!-- 对比表 -->
    <section class="section">
      <h2>Mixtral 和 DeepSeek 的 MoE 差在哪</h2>
      <div class="lead-group">
        <p>下表每一行都是一个被踩过的坑, 左格是 Mixtral 的做法, 右格是 DeepSeek 针对同一个坑给出的回答:</p>
        <ul class="pts">
          <li>专家个数少、每个都很大, 一个专家什么都得管, 分不出工。</li>
          <li>没有共享专家, 「通用能力」(每个 token 都用得上的那部分知识) 每个专家都得各自重学一遍。</li>
          <li>不做均衡, 热门专家越学越好、越选越多, 滚雪球滚成路由坍缩。</li>
          <li>用 aux loss 做均衡, 这一项的梯度会回传到 router, 跟语言建模 loss 的梯度抢方向。</li>
        </ul>
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
              <td><code class="inline">softmax(logits)</code>: 一个 token 对所有专家的分加起来等于 1, 专家之间互相竞争</td>
              <td><code class="inline">sigmoid(logits)</code>: 每个专家独立得一个 0 到 1 的分, 互不影响</td>
            </tr>
            <tr><td class="row-label">专家归一化</td>
              <td>softmax 值已经是概率; 选出 top-k 后把这 k 个概率再除以它们的和, 让 k 个权重加起来等于 1, 专家输出的幅度才不随「选中专家的概率总和」波动</td>
              <td>sigmoid 的分不是概率, 所以选出 k 个之后必须除以它们的和才能当权重; 权重用的是不带 bias 的原始分</td>
            </tr>
            <tr><td class="row-label">共享专家</td>
              <td>❌ 没有</td>
              <td>✅ 有。共享专家不经过 router, 每个 token 都过它, 输出直接加到路由专家的输出上。它兜住通用知识, 路由专家才腾得出容量去各自专门化</td>
            </tr>
            <tr><td class="row-label">负载均衡</td>
              <td>在总 loss 外挂一项 aux loss (Switch Transformer 的做法): 哪个专家被选得多、平均概率又高, 这一项就大; 它的梯度顺着 router 把热门专家的分数压下去。本库 total = lm_loss + 0.01 · aux</td>
              <td>aux-loss-free bias: 给每个专家一个偏置, 它不走梯度, 只在「选谁」时加到分数上, 算权重时不用。每步训练后, 过载的专家 bias 减一点, 欠载的加一点。本库 demo (router 被故意初始化得偏心) 里, 负载 CV (变异系数, 各专家负载的标准差除以平均值, 越小越均衡) 从 0.273 降到 0.124, LM loss 几乎不动: 2.979 对 2.982</td>
            </tr>
            <tr><td class="row-label">专家粒度</td>
              <td>8 个大专家 (Mixtral 8x7B: E=8, 每个 token 选 2 个)</td>
              <td>64–256 个细粒度专家 (本库 DeepSeekMoE 默认 64 个路由专家每个 token 选 6 个, 外加 2 个共享专家)</td>
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
import EvolutionChain from '@/components/EvolutionChain.vue'
import RepoLink from '@/components/RepoLink.vue'

// 上一章 / 下一章从 learningPath 取, 不手写章名和编号 (与 Infer.vue 同一写法)
const at = learningPath.findIndex((x) => x.route === 'moe')
const prevChapter = { name: learningPath[at - 1].route, label: `上一章 · ${learningPath[at - 1].label}` }
const nextChapter = { name: learningPath[at + 1].route, label: `下一章 · ${learningPath[at + 1].label}` }

// 演进逻辑链: 每步的 pain 是上一步 fix 留下的窟窿
const evoSteps = [
  { name: '稠密 FFN', color: 'var(--text-muted)',
    pain: '(原点) 想让模型多记知识就加宽 FFN, 但每个 token 都要过整个 FFN, 算力跟着涨',
    fix: '每个参数对每个 token 都算一遍: FFN 容量翻倍, 每个 token 的算力也翻倍, 两者绑死' },
  { name: 'Top-k MoE', year: 2017, color: 'var(--accent)',
    pain: '容量和算力绑死, 模型想变大就得同比多花算力',
    fix: '一个大 FFN 换成 E 个小 FFN (专家), router 给每个 token 打分, 只送进分数最高的 k 个: 参数按 E 算, 算力按 k 算' },
  { name: 'Mixtral', year: 2024, color: 'var(--left)',
    pain: '被选得多的专家学得更好, 于是更常被选中, 滚雪球到最后少数专家包揽几乎所有 token (路由坍缩)',
    fix: 'softmax 打分选 top-2, 再往总 loss 里加一项 aux loss 压热门专家的分数; 代价是 8 个大专家各自都得重学一遍通用能力' },
  { name: 'DeepSeekMoE', year: 2024, color: 'var(--right)',
    pain: '专家少而大, 通用能力重复学了好几份, 腾不出容量分工; aux loss 的梯度还会和语言建模抢方向',
    fix: '切成 64–256 个细粒度专家, 加每个 token 必过的共享专家专管通用能力; 均衡交给一个不走梯度的 bias, 只改「选谁」' },
]

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
