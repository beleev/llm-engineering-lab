<template>
  <div>
    <h1 class="page-title">扩散生成 · DDPM 与 Flow Matching</h1>
    <div class="page-subtitle lead-group">
      <p>
        扩散模型 (diffusion model) 是当前图像与视频生成的主流路线。直接从纯随机噪声一步映射出清晰图像非常困难, 但把一张略带噪点的图像擦干净一点要容易得多; 扩散模型正是把单步生成拆解为几十步渐进式去噪。
      </p>
      <ul class="pts">
        <li><b>训练阶段:</b> 向真实图像混入已知噪声, 让网络学习从加噪图像中预测噪声或去噪方向。</li>
        <li><b>生成阶段:</b> 从纯随机噪声出发, 分多步迭代扣除预测出的噪声成分。</li>
      </ul>
      <p>下方实验台可以按两步对照观察:</p>
      <ol class="pts">
        <li>拖动 timestep 滑条, 观察原图 <Tex text="$x_0$" /> 如何随时间步逐步退化为纯噪声。</li>
        <li>
          切换调度器, 对比
          <strong>DDPM (预测噪声 ε, 采用 cosine 调度)</strong> 与
          <strong>Flow Matching (预测速度 v, 在原图与噪声间线性插值)</strong> 两条路线的差异。
        </li>
      </ol>
    </div>

    <ChapterIntro
      tldr="扩散生成将复杂的单步采样转化为多步迭代去噪, 主流实现经历了两代演进:
        - DDPM (2020): 按余弦曲线逐步混入高斯噪声, 网络回归所加噪声 $\varepsilon$; 推理时借助 DDIM 等采样器分约 50 步还原图像。
        - Flow Matching (2023): 在原图 $x_0$ 与噪声 $\varepsilon$ 之间构造直线插值路径, 网络回归路径切向速度 $v = \varepsilon - x_0$; 由于轨迹平直, 用欧拉法二十余步即可完成采样 (如 SD3 默认 28 步)。"
      question="训练目标只从预测噪声 $\varepsilon$ 改为预测速度 $v$, 为什么推理步数能从 50 步降到 28 步? 「直线路径」在几何上改变了什么?"
      :goals="[
        '理清扩散模型的训练与采样闭环: 正向按系数混入噪声, 反向沿去噪方向逐步迭代还原',
        '对比 $\\varepsilon$-prediction 与 v-prediction 的几何轨迹差异, 理解平直路径为何允许更大的采样步长',
        '掌握 DiT 通过 adaLN-Zero 注入时间步与条件信息的方式, 以及零初始化如何让残差块从恒等映射平滑起步',
      ]"
      :codes="[
        { path: 'llm_models/layers/diffusion/adaln.py', label: 'adaln.py · AdaLNZeroBlock' },
        { path: 'llm_models/training/diffusion.py', label: 'training/diffusion.py' },
      ]"
      :prereq="prevChapter"
      :next-step="nextChapter"
    />

    <EvolutionChain
      title="演进逻辑链 · 单步映射太难, 拆成多步渐进还原"
      subtitle="根问题: 从高斯噪声一步生成高维图像, 分布跨度大且对抗训练容易坍缩"
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
        <strong>模拟什么:</strong> 扩散训练中的正向加噪过程。固定原图 x_0 与一份高斯噪声 ε, 按时间步 t 将二者线性混合得到 x_t (t = 0 为原图, t = 1 为纯噪声)。三张画布均按解析公式实时计算, 不运行神经网络。
      </p>
      <p class="desc" style="margin-top: 6px;">
        <strong>图上是什么:</strong>
      </p>
      <ul class="desc pts">
        <li><b>左图 x_0:</b> 合成环形测试原图。</li>
        <li><b>中图 x_t:</b> 当前时间步 t 下的加噪图像, 下方展示信号系数、噪声系数与对应的信噪比 (SNR)。</li>
        <li><b>右图:</b> 训练阶段网络需要拟合的回归真值——DDPM 下为噪声 ε, Flow Matching 下为速度 v = ε − x_0。</li>
      </ul>
      <p class="desc" style="margin-top: 6px;">
        <strong>可以做什么:</strong>
      </p>
      <ul class="desc pts">
        <li>切换「调度器」对比 DDPM 与 Flow Matching 的系数变化及回归目标差异。</li>
        <li>拖动「timestep t」从 0 到 1 观察正向加噪过程。</li>
        <li>点击「播放完整去噪」让 t 从 1 平滑退回 0, 直观观察逆向还原过程。</li>
        <li>点击两个「置为」按钮直接跳到起点或终点。</li>
      </ul>
      <p class="desc" style="margin-top: 6px;">
        <strong>应该看到:</strong>
      </p>
      <ul class="desc pts">
        <li>由于随机种子固定, 拖动 t 时混入的是同一份噪声 ε, 中图表现为原图被同一幅噪声逐步覆盖而非随机闪烁。</li>
        <li>在同一 t 下切换调度器, 信号/噪声系数与信噪比会随之改变, 反映两种调度曲线对退化进度的不同定义。</li>
        <li>右图的回归真值不随滑条 t 改变 (DDPM 始终回归固定噪声 ε, Flow Matching 始终回归恒定速度 ε − x_0), 仅在切换调度器时变化。</li>
      </ul>
    </div>

    <!-- 主可视化: 三格 x_0 / x_t / 回归目标。三张图都是按公式算出来的, 本页不跑网络 -->
    <div class="grid grid-3">
      <div class="card panel">
        <h3><Tex text="$x_0$" /> <span class="tag">原图</span></h3>
        <p class="desc">真值原图 (合成环形测试图案)</p>
        <canvas ref="canvasX0" width="128" height="128" class="canvas" role="img" aria-label="原图: 合成的环形图案" />
      </div>

      <div class="card panel">
        <h3><Tex text="$x_t$" /> <span class="tag">加噪后</span></h3>
        <p class="desc"><Tex :text="noiseFormula" /></p>
        <p class="desc" style="margin-top: 6px;">加噪图像 x_t 由原图与高斯噪声按系数加权求和得到, 两组权重由时间步 t 与调度器共同决定。</p>
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
            <div class="hint">大于 1 表示原图能量高于噪声, 越接近 0 越接近纯噪声; t = 0 时无噪声, 记作 ∞</div>
          </div>
        </div>
      </div>

      <div class="card panel">
        <h3><Tex :text="scheduler === 'ddpm' ? '$\\varepsilon$' : '$v = \\varepsilon - x_0$'" /> <span class="tag">回归目标</span></h3>
        <p class="desc">训练时网络接收中图 x_t 与时间步 t, 拟合的监督信号正是这张目标图 (由中图所用的同一份 ε 算出)。</p>
        <p class="desc" style="margin-top: 6px;">为避免较大像素幅值在可视化时越界截断, 画布渲染时将幅值缩放了 0.5 倍。</p>
        <canvas ref="canvasPred" width="128" height="128" class="canvas" role="img"
                :aria-label="scheduler === 'ddpm' ? '回归目标: 噪声 ε' : '回归目标: 速度 v, 等于噪声减原图'" />
      </div>
    </div>

    <!-- schedule 对比曲线 -->
    <section class="section">
      <h2>噪声调度曲线</h2>
      <div class="lead-group">
        <p>
          噪声调度 (noise schedule) 决定了在任意时刻 t 加噪图中保留多少原图成分、混入多少噪声。下方曲线绘制了原图权重随 t 的衰减规律 (纵轴 1 代表完整原图, 0 代表完全消失; 竖线标示当前滑条位置)。
        </p>
        <ul class="pts">
          <li>
            <b>DDPM:</b> 将正向退化拆作多步马尔可夫链, 第 t 步噪声方差记为 <Tex text="$\beta_t$" />, 累积保留的信号比例记为 <Tex text="$\bar\alpha_t$" /> (图中实线)。本库采用两端更平缓的 cosine 调度, 避免早期 linear 调度在 <Tex text="$t \to 1$" /> 之前过早抹除图像结构。
          </li>
          <li>
            <b>Flow Matching:</b> 直接定义线性插值 <Tex text="$x_t = (1-t)\cdot x_0 + t\cdot\varepsilon$" /> (图中虚线), 原图权重按 1 − t 均匀递减, 噪声权重按 t 均匀递增, 训练形式更简洁, 采样所需的离散步数也更少。
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
      <h3>为什么 DiT / MM-DiT 后续模型普遍转向 Flow Matching</h3>
      <p class="desc">
        SD3、FLUX、HunyuanVideo、Wan 2.2 等 2024 年之后的视觉生成模型, 普遍从预测噪声 (<Tex text="$\varepsilon$" />-pred) 切换到预测速度 (v-pred) 与 Rectified Flow, 主要带来三项工程收益:
      </p>
      <ul class="desc" style="padding-left: 20px; list-style: disc; line-height: 1.9; margin-top: 6px;">
        <li><strong>训练目标方差更平稳</strong>: 线性路径上的速度目标 v = ε − x_0 与时间步 t 无关, 全程量级稳定; 而预测 <Tex text="$\varepsilon$" /> 在 t 趋近 0 时需要从几乎干净的图像中剥离极微弱噪声, 梯度方差波动更大</li>
        <li><strong>采样步数更少</strong>: 概率流轨迹接近直线, 使用一阶欧拉离散化即可采用较大步长 (SD3 推荐 28 步, 而基于 DDPM 的 DDIM 通常需要约 50 步)</li>
        <li><strong>超参数更精简</strong>: 线性插值本身即定义了调度关系, 省去了手工调整 <Tex text="$\beta_t$" /> 起止范围与余弦偏移量的开销</li>
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
    pain: '(原点) 从纯噪声一步映射出完整图像, 分布跨度大、单步拟合极难',
    fix: 'GAN 借助判别器对抗实现单步出图, 但博弈训练易失稳且容易发生模式坍缩' },
  { name: 'DDPM', year: 2020, color: 'var(--accent)',
    pain: '单步跨度太大, 对抗训练又不稳定',
    fix: '拆成 T=1000 级微小的加噪与去噪步, 网络只需用 MSE 回归所加噪声 $\\varepsilon$; 代价是采样也得逐级走回' },
  { name: 'Latent + DiT', year: 2023, color: 'var(--left)',
    pain: '采样需几十步迭代, 每步在全分辨率像素图上跑主干网络开销过高',
    fix: '先经 VAE 压缩到低维 latent (元素数 ÷48), 切 patch 送入 Transformer, 时间与条件经 adaLN 注入各层' },
  { name: 'Flow Matching', year: 2023, color: 'var(--right)',
    pain: '$\\varepsilon$-pred 的去噪轨迹带曲率, 步长放大易偏离, DDIM 仍需约 50 步',
    fix: '在原图与噪声间构造直线路径, 回归切向速度 $v=\\varepsilon-x_0$: 欧拉法沿直线大步推进, SD3 仅需约 28 步' },
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
训练阶段每次迭代分为三步:
1. 随机采样时间步 $t$, 按首行公式将原图 $x_0$ 与高斯噪声 $\\varepsilon$ 混成 $x_t$。
2. 网络接收 $x_t$ 与 $t$, 预测混入的那份噪声 $\\hat\\varepsilon$。
3. 计算预测值与真值 $\\varepsilon$ 的均方误差 (MSE) 作为损失。
推理时采用确定性少步采样器 DDIM, 从纯噪声出发逐步反推:
$$\\begin{aligned} \\hat x_0 &= \\big(x_t - \\sqrt{1-\\bar\\alpha_t}\\cdot\\hat\\varepsilon\\big) / \\sqrt{\\bar\\alpha_t} \\\\ x_{t-1} &= \\sqrt{\\bar\\alpha_{t-1}}\\cdot\\hat x_0 + \\sqrt{1-\\bar\\alpha_{t-1}}\\cdot\\hat\\varepsilon \\end{aligned}$$
1. 先将正向加噪式反解: 从 $x_t$ 中扣去网络预测的噪声项, 得到当前对干净原图的估计 $\\hat x_0$。
2. 再用同一份预测噪声 $\\hat\\varepsilon$, 按下一时刻更低的噪声比例重新合成 $x_{t-1}$。
沿时间步递减迭代至 $t=0$ 即可还原图像, 本库默认运行 50 步。`

const fmMath = `$$\\begin{aligned} x_t &= (1-t)\\cdot x_0 + t\\cdot\\varepsilon && \\text{线性路径} \\\\ v_t &= \\mathrm{d}x_t/\\mathrm{d}t = \\varepsilon - x_0 && \\text{velocity} \\\\ \\text{target} &= v \\\\ \\text{loss} &= \\mathrm{MSE}(\\mathrm{model}(x_t, t), v) \\end{aligned}$$
训练目标直接建立在线性插值轨迹之上:
- 在原图 $x_0$ 与噪声 $\\varepsilon$ 之间连一条直线, $t \\in [0, 1]$ 表示当前插值位置。
- 路径对时间求导得到恒定切向速度 $v = \\varepsilon - x_0$, 其幅值与 $t$ 无关。
- 网络接收 $x_t$ 与 $t$ 并通过均方误差拟合速度向量 $v$。
推理时直接用一阶欧拉法求解常微分方程, 从纯噪声 ($t=1$) 反向积分回原图 ($t=0$):
$$\\begin{aligned} \\hat v &= \\mathrm{model}(x_t, t) \\\\ x_{t-\\Delta t} &= x_t - \\Delta t\\cdot\\hat v \\end{aligned}$$
每一步迭代只需两行计算:
1. 调用网络估计当前位置的速度向量 $\\hat v$。
2. 沿速度反方向推进步长 $\\Delta t$ (因 $v$ 指向噪声端, 退回原图端需取减号)。
得益于平直轨迹, 单步跨度可以明显放大: 本库教学示例默认 20 步, SD3 实际推理常取 28 步。`

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
