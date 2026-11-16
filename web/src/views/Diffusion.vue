<template>
  <div>
    <h1 class="page-title">扩散生成 · DDPM 与 Flow Matching</h1>
    <div class="page-subtitle lead-group">
      <p>
        扩散模型 (diffusion model) 是现在生图、生视频的主流做法。它的出发点是:
        「从一堆随机数直接生成一张图」太难, 但「把一张稍微脏一点的图擦干净一点」很容易。扩散就是把前者拆成几十次后者。
      </p>
      <p>
        训练时往真图里加噪声, 让网络学会从加噪后的图里认出加了什么噪声; 生成时从纯噪声出发, 一步一步把认出的噪声减掉。
      </p>
      <p>
        拖下面的 timestep 滑条, 看原图 <Tex text="$x_0$" /> 一路糊成纯噪声。再切换调度器, 比较
        <strong>DDPM (让网络预测噪声 ε, 用 cosine 噪声调度)</strong> 和
        <strong>Flow Matching (让网络预测速度 v, 噪声和原图之间走直线)</strong> 这两条路的差别。
      </p>
    </div>

    <ChapterIntro
      tldr="扩散把「生成一张图」换成了「学一个去噪函数」: 训练时给真图加噪声, 让网络从加噪后的图里猜出加了什么; 生成时从纯噪声出发, 几十步把噪声减掉。
        - DDPM (2020): 网络预测加进去的噪声 $\varepsilon$。推理用 DDIM 采样器, 要 50 步左右。
        - Flow Matching (2023): 噪声和原图之间走直线, 网络预测这条直线的速度 $v = \varepsilon - x_0$。路径是直的, 用最简单的欧拉法几步就能走完, SD3 推荐 28 步。"
      question="训练目标只换了一个符号 (预测 $\varepsilon$ 换成预测 $v$), 推理步数为什么能从 50 降到 28? 「直线路径」到底直在哪?"
      :goals="[
        '把生成问题翻译成「学去噪」, 说清训练时网络在学什么、采样时每一步在做什么',
        '讲清 $\\varepsilon$-prediction 和 v-prediction 在几何上的差别, 以及为什么直线路径能换来更少的步数',
        '看懂 DiT 用 adaLN-Zero 把时间步和条件注入 Transformer 的那一步, 以及零初始化为什么让每个 block 从恒等映射起步',
      ]"
      :codes="[
        { path: 'llm_models/layers/diffusion/adaln.py', label: 'adaln.py · AdaLNZeroBlock' },
        { path: 'llm_models/training/diffusion.py', label: 'training/diffusion.py' },
      ]"
      :prereq="prevChapter"
      :next-step="nextChapter"
    />

    <EvolutionChain
      title="演进逻辑链 · 一步太难, 就拆成很多小步"
      subtitle="根问题: 从纯噪声一步变出一张图, 一个网络要学的「噪声到图」的映射太陡、太复杂"
      :steps="evoSteps"
    />

    <!-- 控制 -->
    <div class="card" style="margin-bottom: 20px;">
      <div class="controls-grid">
        <div>
          <div id="diffusion-scheduler-label" class="slot-title">调度器</div>
          <div class="btn-group" role="group" aria-labelledby="diffusion-scheduler-label">
            <button type="button" :class="{ active: scheduler === 'ddpm' }" :aria-pressed="scheduler === 'ddpm'" @click="scheduler = 'ddpm'">DDPM (cosine)</button>
            <button type="button" :class="{ active: scheduler === 'fm' }" :aria-pressed="scheduler === 'fm'" @click="scheduler = 'fm'">Flow Matching (linear)</button>
          </div>
        </div>
        <div>
          <div class="slot-title">预测目标</div>
          <div class="form-val mono">
            <Tex :text="scheduler === 'ddpm' ? 'target $= \\varepsilon$ (noise)' : 'target $= v = \\varepsilon - x_0$'" />
          </div>
        </div>
      </div>
      <div class="form-row" style="margin-top: 14px;">
        <label for="diffusion-timestep">timestep t</label>
        <input id="diffusion-timestep" type="range" min="0" max="1" step="0.01" v-model.number="t" />
        <span class="val">{{ t.toFixed(2) }}</span>
      </div>
      <div class="btn-group" style="margin-top: 10px;">
        <button type="button" @click="animate" :disabled="animating">{{ animating ? '播放中…' : '▷ 播放完整去噪' }}</button>
        <button type="button" @click="t = 1">置为纯噪声 (t=1)</button>
        <button type="button" @click="t = 0">置为原图 (t=0)</button>
      </div>
      <p class="desc" style="margin-top: 12px;">
        <strong>模拟什么:</strong> 扩散训练里「正向加噪」这一步。拿一张真图 x_0 和一份固定的高斯噪声 ε, 按时间 t 把两者混在一起得到 x_t:
        t = 0 是原图, t = 1 是纯噪声。本页不跑任何神经网络, 三张图全是按公式算出来的。
      </p>
      <p class="desc" style="margin-top: 6px;">
        <strong>图上是什么:</strong> 左图 x_0 是原图 (教学用的合成环形图案); 中图 x_t 是当前 t 下的加噪图, 它下面两个读数是混合时用的信号系数、噪声系数, 以及由它们算出的信噪比;
        右图是网络训练时要回归的目标: DDPM 下是噪声 ε 本身, Flow Matching 下是速度 v, 等于噪声减原图。
      </p>
      <p class="desc" style="margin-top: 6px;">
        <strong>可以做什么:</strong> 「调度器」在 DDPM 和 Flow Matching 之间切换, 它决定 t 和两个系数怎么对应, 也决定右图是 ε 还是 v;
        拖「timestep t」从 0 到 1 看加噪过程; 「播放完整去噪」让 t 从 1 自动退回 0, 相当于把加噪倒着放一遍; 两个「置为」按钮直接跳到两端。
      </p>
      <p class="desc" style="margin-top: 6px;">
        <strong>应该看到:</strong> 拖 t 时噪声是同一份 (随机种子固定), 只是占比越来越大, 所以中图不是在随机闪烁, 而是原图被同一张噪声逐渐盖住。
        同一个 t 下切换调度器, 两个系数和信噪比都会变, 因为两种调度对「t 对应多少噪声」的定义不同。
        右图不随 t 变: DDPM 的目标就是那一份噪声, Flow Matching 的目标是直线的速度, 处处相同; 它只在切换调度器时变。
      </p>
    </div>

    <!-- 主可视化: 三格 x_0 / x_t / 回归目标。三张图都是按公式算出来的, 本页不跑网络 -->
    <div class="grid grid-3">
      <div class="card panel">
        <h3><Tex text="$x_0$" /> <span class="tag">原图</span></h3>
        <p class="desc">真值原图 (教学用的合成环形图案)</p>
        <canvas ref="canvasX0" width="128" height="128" class="canvas" role="img" aria-label="原图: 合成的环形图案" />
      </div>

      <div class="card panel">
        <h3><Tex text="$x_t$" /> <span class="tag">加噪后</span></h3>
        <p class="desc"><Tex :text="noiseFormula" /></p>
        <p class="desc" style="margin-top: 6px;">这个式子在说: 加噪图 x_t 等于原图乘一个信号系数, 再加上噪声乘一个噪声系数; 两个系数由 t 和调度器决定。</p>
        <canvas ref="canvasXt" width="128" height="128" class="canvas" role="img"
                :aria-label="`t = ${t.toFixed(2)} 时的加噪图, 信噪比 ${snrText}`" />
        <div>
          <div class="stat" style="margin-top: 10px;">
            <div class="k">信号系数 / 噪声系数: <Tex :text="coefLabel" /></div>
            <div class="v mono">{{ coef.a.toFixed(3) }} / {{ coef.b.toFixed(3) }}</div>
          </div>
          <div class="stat" style="margin-top: 8px;">
            <div class="k">信噪比 = (信号系数 / 噪声系数)²</div>
            <div class="v mono">{{ snrText }}</div>
            <div class="hint">大于 1 表示图里原图成分多于噪声, 越接近 0 越接近纯噪声。由当前 t 和调度器算出; t = 0 时没有噪声, 记作 ∞</div>
          </div>
        </div>
      </div>

      <div class="card panel">
        <h3><Tex :text="scheduler === 'ddpm' ? '$\\varepsilon$' : '$v = \\varepsilon - x_0$'" /> <span class="tag">回归目标</span></h3>
        <p class="desc">训练时网络看到的输入是中图 x_t 和时间 t, 要输出的就是这张图。它用的是中图里那同一份噪声 ε, 所以这是「标准答案」, 不是网络的输出。</p>
        <p class="desc" style="margin-top: 6px;">本页不跑网络。显示时幅度除以 2, 大部分像素才不会超出显示范围被截断。</p>
        <canvas ref="canvasPred" width="128" height="128" class="canvas" role="img"
                :aria-label="scheduler === 'ddpm' ? '回归目标: 噪声 ε' : '回归目标: 速度 v, 等于噪声减原图'" />
      </div>
    </div>

    <!-- schedule 对比曲线 -->
    <section class="section">
      <h2>噪声调度曲线</h2>
      <div class="lead-group">
        <p>
          噪声调度 (noise schedule) 回答一个问题: 在时刻 t, 加噪图里原图还剩多少、噪声占多少。
          下面这条曲线画的就是「原图还剩多少」随 t 的变化: 纵轴 1 是完整原图, 0 是一点不剩; 竖线和圆点标的是滑杆当前的 t。
        </p>
        <ul class="pts">
          <li>
            <b>DDPM:</b> 把加噪看成很多小步, 第 t 步加进去的噪声方差记作 <Tex text="$\beta_t$" />; 一路累乘下来, 原图还剩的比例记作 <Tex text="$\bar\alpha_t$" /> (图中实线)。
            本库用 cosine 调度, 也就是直接把 <Tex text="$\bar\alpha_t$" /> 定成一条余弦曲线, 让它在两端都更平缓;
            这是为了避开 linear 调度在 <Tex text="$t \to 1$" /> 时把图糊得过头的问题。
          </li>
          <li>
            <b>Flow Matching:</b> 不走小步累乘, 直接规定 <Tex text="$x_t = (1-t)\cdot x_0 + t\cdot\varepsilon$" /> (图中虚线):
            原图成分按 1 − t 线性减少, 噪声按 t 线性增加。训练公式最短, 采样步数也最少。
          </li>
        </ul>
      </div>
      <div class="card">
        <svg viewBox="0 0 600 240" width="100%" height="240" role="img"
             aria-label="原图系数随 t 的变化: DDPM 的 cosine 曲线两端平缓, Flow Matching 是一条直线">
          <!-- 坐标轴 -->
          <line x1="40" y1="20" x2="40" y2="200" stroke="var(--border)" />
          <line x1="40" y1="200" x2="580" y2="200" stroke="var(--border)" />
          <text x="30" y="24" text-anchor="end" fill="var(--text-dim)" font-size="10" font-family="SF Mono">1</text>
          <text x="30" y="204" text-anchor="end" fill="var(--text-dim)" font-size="10" font-family="SF Mono">0</text>
          <text x="40" y="220" text-anchor="start" fill="var(--text-dim)" font-size="10" font-family="SF Mono">t=0</text>
          <text x="580" y="220" text-anchor="end" fill="var(--text-dim)" font-size="10" font-family="SF Mono">t=1</text>

          <!-- DDPM cosine curve -->
          <path :d="ddpmCurve" fill="none" stroke="var(--accent)" stroke-width="2" />
          <!-- FM linear curve -->
          <path :d="fmCurve" fill="none" stroke="var(--left)" stroke-width="2" stroke-dasharray="4 4" />

          <!-- 当前 t 标记 -->
          <line :x1="40 + t * 540" :x2="40 + t * 540" y1="20" y2="200"
                stroke="var(--warn)" stroke-width="1.5" />
          <circle :cx="40 + t * 540"
                  :cy="200 - 180 * alphaBarCurve(t)"
                  r="5"
                  :fill="scheduler === 'ddpm' ? 'var(--accent)' : 'var(--left)'" />

          <!-- Legend -->
          <g transform="translate(420, 30)">
            <line x1="0" y1="0" x2="20" y2="0" stroke="var(--accent)" stroke-width="2" />
            <text x="26" y="4" fill="var(--text-muted)" font-size="11">DDPM ᾱ_t (cosine)</text>
            <line x1="0" y1="18" x2="20" y2="18" stroke="var(--left)" stroke-width="2" stroke-dasharray="4 4" />
            <text x="26" y="22" fill="var(--text-muted)" font-size="11">FM (1-t) 线性</text>
          </g>
        </svg>
      </div>
    </section>

    <!-- 关键公式 -->
    <section class="section">
      <h2>核心公式速查</h2>
      <div class="grid grid-2">
        <div class="card">
          <h3>DDPM (Ho et al., 2020)</h3>
          <Prose :text="ddpmMath" />
        </div>
        <div class="card">
          <h3>Flow Matching (Lipman et al., 2023)</h3>
          <Prose :text="fmMath" />
        </div>
      </div>
    </section>

    <div class="card" style="margin-top: 20px;">
      <h3>为什么 DiT / MM-DiT / Sora 后来都换成了 Flow Matching?</h3>
      <p class="desc">
        SD3、FLUX、HunyuanVideo、Wan 2.2 这些 2024 年之后的生图/生视频模型, 全部从预测噪声 (<Tex text="$\varepsilon$" />-pred) 切到了预测速度 (v-pred) + Rectified Flow (走直线的 Flow Matching)。
        三个理由:
      </p>
      <ul class="desc" style="padding-left: 20px; list-style: disc; line-height: 1.9; margin-top: 6px;">
        <li><strong>训练更稳</strong>: 直线路径上速度目标 v = ε − x_0 和 t 无关, 量级从头到尾差不多; 预测 <Tex text="$\varepsilon$" /> 时, t 接近 0 那一头目标的方差变化剧烈, 网络要从一张几乎干净的图里猜出很小的噪声</li>
        <li><strong>推理更快</strong>: 路径是直的, 用最简单的欧拉法几步就走完了: SD3 推荐 28 步, DDPM 的少步采样器 DDIM 要 50 步</li>
        <li><strong>可调的东西更少</strong>: 不用再挑噪声调度 (直线本身就是调度), 超参数少一截, 别人复现起来也更容易</li>
      </ul>
    </div>

    <!-- 本章挂载的实验台 (data/labmap/*.js) 与章末自测 (data/quiz/*.js), 没配置时不渲染 -->

    <LabMount />

    <QuizCard />


    <ChapterNav
      :prev="prevChapter"
      :next="nextChapter"
    />
  </div>
</template>

<script setup>
import LabMount from '@/components/LabMount.vue'
import QuizCard from '@/components/QuizCard.vue'
import { ref, computed, watch, onMounted, onBeforeUnmount } from 'vue'
import ChapterIntro from '@/components/ChapterIntro.vue'
import ChapterNav from '@/components/ChapterNav.vue'
import EvolutionChain from '@/components/EvolutionChain.vue'
import Prose from '@/components/Prose.vue'
import Tex from '@/components/Tex.vue'
import { learningPath } from '@/data/models.js'
import { mulberry32, randn } from '@/utils/labmath.js'

// 上一章 / 下一章从 learningPath 取, 不手写章名和编号 (与 Infer.vue 同一写法)
const at = learningPath.findIndex((x) => x.route === 'diffusion')
const prevChapter = { name: learningPath[at - 1].route, label: `上一章 · ${learningPath[at - 1].label}` }
const nextChapter = { name: learningPath[at + 1].route, label: `下一章 · ${learningPath[at + 1].label}` }

// 演进逻辑链: 每步的 pain 是上一步 fix 留下的窟窿
const evoSteps = [
  { name: '一步生成', color: 'var(--text-muted)',
    pain: '(原点) 从纯噪声一步变出整张图, 网络要学的映射又陡又复杂',
    fix: 'GAN 一步出图, 靠一个判别器和它对抗训练: 容易训崩, 也容易只会画少数几种图' },
  { name: 'DDPM', year: 2020, color: 'var(--accent)',
    pain: '一步跨得太远, 对抗训练又不稳',
    fix: '把去噪拆成 T=1000 级小步, 网络只回归每一步加进去的噪声 $\\varepsilon$, loss 就是 MSE; 采样也得一级级走回去' },
  { name: 'Latent + DiT', year: 2023, color: 'var(--left)',
    pain: '采样要走几十步, 每步都在整张像素图上跑一遍去噪网络, 太贵',
    fix: '先用 VAE 把图压成小的 latent (元素数 ÷48), 切成 patch 当 token 过 Transformer; 时间步和条件经 adaLN 注入每一层' },
  { name: 'Flow Matching', year: 2023, color: 'var(--right)',
    pain: '$\\varepsilon$-pred 的去噪轨迹是弯的, 步子一迈大就偏, DDIM (少步采样器) 仍要 50 步',
    fix: '让噪声和数据之间走直线, 网络回归直线的速度 $v=\\varepsilon-x_0$: 欧拉法沿直线几步就到, SD3 推荐 28 步' },
]

const t = ref(0.5)
const scheduler = ref('ddpm')
const animating = ref(false)

const canvasX0 = ref(null)
const canvasXt = ref(null)
const canvasPred = ref(null)

// --- ᾱ_t curves ---
const s = 0.008
function alphaBarCosine(tau) {
  const f = (x) => Math.cos(((x + s) / (1 + s)) * Math.PI / 2) ** 2
  return f(tau) / f(0)
}
function alphaBarCurve(tau) {
  return scheduler.value === 'ddpm' ? alphaBarCosine(tau) : (1 - tau)
}

// ★ x_t = a·x_0 + b·ε 的两个系数, 和 llm_models/training/diffusion.py 里两个 add_noise 同式:
//   DDPM 是 (√ᾱ_t, √(1-ᾱ_t)), Flow Matching 是 (1-t, t)
const coef = computed(() => {
  if (scheduler.value === 'fm') return { a: 1 - t.value, b: t.value }
  const ab = alphaBarCosine(t.value)
  return { a: Math.sqrt(ab), b: Math.sqrt(Math.max(0, 1 - ab)) }
})
const coefLabel = computed(() => scheduler.value === 'ddpm'
  ? '$\\sqrt{\\bar\\alpha_t}\\ /\\ \\sqrt{1-\\bar\\alpha_t}$'
  : '$(1-t)\\ /\\ t$')
// 信噪比: 信号和噪声的功率之比。b = 0 (t = 0) 时是 Infinity
const snr = computed(() => (coef.value.a / coef.value.b) ** 2)
const snrText = computed(() => {
  const v = snr.value
  if (!Number.isFinite(v)) return '∞'
  return v >= 100 ? v.toFixed(0) : v.toPrecision(3)
})

const ddpmCurve = computed(() => {
  const pts = []
  for (let i = 0; i <= 100; i++) {
    const tau = i / 100
    const v = alphaBarCosine(tau)
    pts.push(`${i === 0 ? 'M' : 'L'} ${40 + tau * 540} ${200 - v * 180}`)
  }
  return pts.join(' ')
})
const fmCurve = computed(() => {
  // 这里画 (1-t), 表示 FM 下 x_0 系数
  return `M 40 20 L 580 200`
})

const noiseFormula = computed(() => {
  if (scheduler.value === 'ddpm') return '$x_t = \\sqrt{\\bar\\alpha_t}\\cdot x_0 + \\sqrt{1-\\bar\\alpha_t}\\cdot\\varepsilon$'
  return '$x_t = (1-t)\\cdot x_0 + t\\cdot\\varepsilon$'
})

const ddpmMath = `$$\\begin{aligned} x_t &= \\sqrt{\\bar\\alpha_t}\\cdot x_0 + \\sqrt{1-\\bar\\alpha_t}\\cdot\\varepsilon, \\qquad \\varepsilon \\sim \\mathcal{N}(0, I) \\\\ \\text{target} &= \\varepsilon \\\\ \\text{loss} &= \\mathrm{MSE}(\\mathrm{model}(x_t, t), \\varepsilon) \\end{aligned}$$
这三行在说: 训练时随机挑一个 $t$, 按第一行把原图 $x_0$ 和一份随机噪声 $\\varepsilon$ 混成 $x_t$; 网络拿到 $x_t$ 和 $t$, 要猜出那份 $\\varepsilon$; 猜的和真值差多少 (均方误差 MSE), 就是 loss。
推理用 DDIM (一种确定性的少步采样器), 从纯噪声出发, 每一步做两件事:
$$\\begin{aligned} \\hat x_0 &= \\big(x_t - \\sqrt{1-\\bar\\alpha_t}\\cdot\\hat\\varepsilon\\big) / \\sqrt{\\bar\\alpha_t} \\\\ x_{t-1} &= \\sqrt{\\bar\\alpha_{t-1}}\\cdot\\hat x_0 + \\sqrt{1-\\bar\\alpha_{t-1}}\\cdot\\hat\\varepsilon \\end{aligned}$$
第一行把加噪公式反过来解: 用网络猜出的噪声 $\\hat\\varepsilon$ 从 $x_t$ 里扣掉噪声, 得到对原图的一个估计 $\\hat x_0$。第二行用同一份 $\\hat\\varepsilon$, 按下一步更小的噪声比例重新加噪, 得到 $x_{t-1}$。这样一步步走到 $t=0$, 本库默认 50 步。`

const fmMath = `$$\\begin{aligned} x_t &= (1-t)\\cdot x_0 + t\\cdot\\varepsilon && \\text{线性路径} \\\\ v_t &= \\mathrm{d}x_t/\\mathrm{d}t = \\varepsilon - x_0 && \\text{velocity} \\\\ \\text{target} &= v \\\\ \\text{loss} &= \\mathrm{MSE}(\\mathrm{model}(x_t, t), v) \\end{aligned}$$
这几行在说: 在原图 $x_0$ 和噪声 $\\varepsilon$ 之间连一条直线, $t$ 是直线上的位置 (0 在原图, 1 在噪声); 沿直线走的速度 $v$ 就是终点减起点, 和 $t$ 无关; 网络拿到 $x_t$ 和 $t$, 要猜出这个速度, loss 同样是均方误差。
推理用欧拉法 (最简单的常微分方程求解器), 从纯噪声出发沿直线反着走:
$$\\begin{aligned} \\hat v &= \\mathrm{model}(x_t, t) \\\\ x_{t-\\Delta t} &= x_t - \\Delta t\\cdot\\hat v \\end{aligned}$$
每一步让网络猜一次速度 $\\hat v$, 再把当前点往速度的反方向挪一小段 $\\Delta t$ (速度指向噪声, 我们要往原图走, 所以是减)。路径本来就接近直线, 步子可以迈得大: 本库教学默认 20 步, SD3 推理 28 步左右。`

// --- Canvas 渲染 ---
// 画一个合成的环形/辐射图案作为 x_0, 再按公式算出 x_t 和回归目标

function drawPattern(ctx, w, h) {
  // 同心圆 + 辐射线 + 渐变
  const grad = ctx.createRadialGradient(w / 2, h / 2, 5, w / 2, h / 2, w * 0.6)
  grad.addColorStop(0, '#fef08a')
  grad.addColorStop(0.4, '#f472b6')
  grad.addColorStop(0.8, '#6366f1')
  grad.addColorStop(1, '#0a0e1a')
  ctx.fillStyle = grad
  ctx.fillRect(0, 0, w, h)

  ctx.strokeStyle = 'rgba(255, 255, 255, 0.35)'
  ctx.lineWidth = 1
  for (let i = 0; i < 12; i++) {
    const angle = i * Math.PI / 6
    ctx.beginPath()
    ctx.moveTo(w / 2, h / 2)
    ctx.lineTo(w / 2 + Math.cos(angle) * w * 0.45, h / 2 + Math.sin(angle) * h * 0.45)
    ctx.stroke()
  }
  for (let r = 10; r < w * 0.5; r += 14) {
    ctx.beginPath()
    ctx.arc(w / 2, h / 2, r, 0, Math.PI * 2)
    ctx.strokeStyle = `rgba(255, 255, 255, ${0.15 + (r / w) * 0.3})`
    ctx.stroke()
  }
}

const SIZE = 128
let x0 = null    // 原图像素换算到 [-1, 1], 每个通道一个值
let eps = null   // ε ~ N(0, 1), 每个像素一个值, 三个通道共用。种子固定: 拖 t 时是同一份噪声逐步加进去

function prepare() {
  const ctx = canvasX0.value.getContext('2d')
  drawPattern(ctx, SIZE, SIZE)
  x0 = Float32Array.from(ctx.getImageData(0, 0, SIZE, SIZE).data, (p) => p / 127.5 - 1)
  const rand = mulberry32(42)
  eps = Float32Array.from({ length: SIZE * SIZE }, () => randn(rand))
}

// 把 [-1, 1] 的量画成像素。gain 用来把幅度更大的量 (ε 和 v) 压进显示范围; 越界的由 ImageData 自动截断
function paint(canvas, valueAt, gain = 1) {
  const ctx = canvas.getContext('2d')
  const img = ctx.createImageData(SIZE, SIZE)
  for (let i = 0; i < img.data.length; i += 4) {
    for (let c = 0; c < 3; c++) img.data[i + c] = (valueAt(i + c, i >> 2) * gain + 1) * 127.5
    img.data[i + 3] = 255
  }
  ctx.putImageData(img, 0, 0)
}

function renderAll() {
  if (!canvasX0.value) return
  if (!x0) prepare()
  const { a, b } = coef.value
  paint(canvasXt.value, (k, p) => a * x0[k] + b * eps[p])
  // 回归目标的真值: DDPM 是 ε, Flow Matching 是 v = ε - x_0
  const target = scheduler.value === 'ddpm' ? (k, p) => eps[p] : (k, p) => eps[p] - x0[k]
  paint(canvasPred.value, target, 0.5)
}

watch([t, scheduler], renderAll)
onMounted(renderAll)

let animTimer = null
async function animate() {
  if (animating.value) return
  animating.value = true
  const steps = 40
  let i = 0
  animTimer = setInterval(() => {
    t.value = Math.max(0, 1 - i / steps)
    i++
    if (i > steps) {
      clearInterval(animTimer)
      animating.value = false
    }
  }, 60)
}
onBeforeUnmount(() => { if (animTimer) clearInterval(animTimer) })
</script>

<style scoped>
.controls-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
}
.form-val {
  font-size: 14px;
  color: var(--text);
  padding: 8px 12px;
  background: var(--bg-elev);
  border-radius: 5px;
  border: 1px solid var(--border);
  margin-top: 2px;
}

.panel .canvas {
  width: 100%;
  max-width: 220px;
  height: auto;
  aspect-ratio: 1;
  margin-top: 12px;
  display: block;
  border-radius: 6px;
  image-rendering: pixelated;
  border: 1px solid var(--border);
}

.slot-title {
  font-size: 11px;
  color: var(--text-dim);
  text-transform: uppercase;
  letter-spacing: 0.7px;
  margin-bottom: 6px;
}
</style>
