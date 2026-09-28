<template>
  <div>
    <h1 class="page-title">扩散生成 · DDPM 与 Flow Matching</h1>
    <div class="page-subtitle lead-group">
      <p>"直接生成一张图" 太难, "把一张稍微脏一点的图擦干净" 很容易。扩散就是把前者拆成几十次后者。</p>
      <p>
        拖下面的 timestep 滑条, 看 <Tex text="$x_0$" /> 一路糊成纯噪声。再切换调度器, 比较
        <strong>DDPM (ε-pred, cosine 调度)</strong> 和
        <strong>Flow Matching (v-pred, 直线路径)</strong> 这两条路的形状差别。
      </p>
    </div>

    <ChapterIntro
      tldr="扩散把「生成」换成了「学一个去噪函数」。DDPM 让网络预测噪声 $\varepsilon$, Flow Matching 让它预测速度 $v = \varepsilon - x_0$, 后者把弯路拉成了直线。"
      question="训练目标只换了一个符号 ($\varepsilon \to v$), 推理步数为什么能从 50 降到 28? 「直线路径」直在哪?"
      :goals="[
        '把生成问题翻译成「学去噪」, 并说清训练和采样各在做什么',
        '讲清 $\\varepsilon$-prediction 和 v-prediction 的几何差别, 以及它怎么换成步数',
        '看懂 DiT 用 adaLN-Zero 把扩散嫁接到 Transformer 骨架上的那一步',
      ]"
      :codes="[
        { path: 'llm_models/layers/diffusion/adaln.py', label: 'adaln.py · AdaLNZeroBlock' },
        { path: 'llm_models/training/diffusion.py', label: 'training/diffusion.py' },
      ]"
      :prereq="prevChapter"
      :next-step="nextChapter"
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
            <div class="hint">由当前 t 和调度器算出; t = 0 时没有噪声, 记作 ∞</div>
          </div>
        </div>
      </div>

      <div class="card panel">
        <h3><Tex :text="scheduler === 'ddpm' ? '$\\varepsilon$' : '$v = \\varepsilon - x_0$'" /> <span class="tag">回归目标</span></h3>
        <p class="desc">网络要回归的目标真值, 用的就是中间那张图里的同一份噪声。</p>
        <p class="desc" style="margin-top: 6px;">本页不跑网络, 这张图不是网络输出。显示时幅度除以 2, 大部分像素才不会被截断。</p>
        <canvas ref="canvasPred" width="128" height="128" class="canvas" role="img"
                :aria-label="scheduler === 'ddpm' ? '回归目标: 噪声 ε' : '回归目标: 速度 v, 等于噪声减原图'" />
      </div>
    </div>

    <!-- schedule 对比曲线 -->
    <section class="section">
      <h2>噪声调度曲线</h2>
      <div class="lead-group">
        <p>这条曲线就是 "在时刻 t, 原图还剩多少"。</p>
        <ul class="pts">
          <li>
            <b>DDPM:</b> 用 cosine <Tex text="$\beta$" /> 让 <Tex text="$\bar\alpha_t$" /> 两端更平缓,
            避开 linear 调度在 <Tex text="$t \to 1$" /> 时把图糊得过头。
          </li>
          <li>
            <b>Flow Matching:</b> 干脆走直线 <Tex text="$x_t = (1-t)\cdot x_0 + t\cdot\varepsilon$" />。训练公式最短, 采样步数也最少。
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
        SD3、FLUX、HunyuanVideo、Wan 2.2 这些 2024 年之后的生图/生视频模型, 全部从 <Tex text="$\varepsilon$" />-pred 切到了 v-pred + Rectified Flow。
        三个理由:
      </p>
      <ul class="desc" style="padding-left: 20px; list-style: disc; line-height: 1.9; margin-top: 6px;">
        <li><strong>训练更稳</strong>: 直线路径上速度目标的量级从头到尾差不多, 不像 <Tex text="$\varepsilon$" /> 在 t 接近 0 时方差剧烈变化</li>
        <li><strong>推理更快</strong>: 路径是直的, 欧拉法几步就走完了: SD3 推荐 28 步, DDIM 要 50 步</li>
        <li><strong>可调的东西更少</strong>: 没有 noise schedule 要挑, 超参数少一截, 别人复现起来也更容易</li>
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
import Prose from '@/components/Prose.vue'
import Tex from '@/components/Tex.vue'
import { learningPath } from '@/data/models.js'
import { mulberry32, randn } from '@/utils/labmath.js'

// 上一章 / 下一章从 learningPath 取, 不手写章名和编号 (与 Infer.vue 同一写法)
const at = learningPath.findIndex((x) => x.route === 'diffusion')
const prevChapter = { name: learningPath[at - 1].route, label: `上一章 · ${learningPath[at - 1].label}` }
const nextChapter = { name: learningPath[at + 1].route, label: `下一章 · ${learningPath[at + 1].label}` }

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
训练时网络预测噪声。推理 (DDIM, 确定性少步):
$$\\begin{aligned} \\hat x_0 &= \\big(x_t - \\sqrt{1-\\bar\\alpha_t}\\cdot\\hat\\varepsilon\\big) / \\sqrt{\\bar\\alpha_t} \\\\ x_{t-1} &= \\sqrt{\\bar\\alpha_{t-1}}\\cdot\\hat x_0 + \\sqrt{1-\\bar\\alpha_{t-1}}\\cdot\\hat\\varepsilon \\end{aligned}$$`

const fmMath = `$$\\begin{aligned} x_t &= (1-t)\\cdot x_0 + t\\cdot\\varepsilon && \\text{线性路径} \\\\ v_t &= \\mathrm{d}x_t/\\mathrm{d}t = \\varepsilon - x_0 && \\text{velocity} \\\\ \\text{target} &= v \\\\ \\text{loss} &= \\mathrm{MSE}(\\mathrm{model}(x_t, t), v) \\end{aligned}$$
训练时网络预测速度。推理 (Euler ODE, 极少步), 沿直线反推:
$$\\begin{aligned} \\hat v &= \\mathrm{model}(x_t, t) \\\\ x_{t-\\Delta t} &= x_t - \\Delta t\\cdot\\hat v \\end{aligned}$$`

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
