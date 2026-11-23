<!--
  训练-推理不一致实验台 (对应 llm_finetune/methods/is_correction.py 与 run_finetune/rollout_mismatch)。
  只讲一件事: 同一份权重, 推理引擎的 logits 带一点数值误差, 采样分布 q 就和训练器的 π 不同;
  不校正时策略梯度有偏, TIS 截断权重、MIS 屏蔽离群 token, 阈值 C 决定留下多少偏差。
  梯度偏差在 logits 空间精确求和 (不采样), 口径与 Python 第二部分相同: 单步决策, 采对得 1 分, 组均值作 baseline。
-->
<template>
  <LabFrame
    title="训练-推理不一致: 同一份权重, 两个分布"
    sub="- 模拟什么: RL 的 rollout 由推理引擎采样, 梯度由训练器算。两边权重相同, 但推理引擎用低精度和融合算子, 算出的 logits 带一点误差, 采样分布 q (π_infer) 就和训练器的 π (π_train) 不同。这里给 64 个解码位置各 12 个候选 token, 推理引擎的 logits = 训练器的 logits + 噪声, 噪声大小按 Python 实测的 |log r| 均值标定 (fp16 < bf16 < bf16+int4), 少数 logit 偏得更远, 对应实测里的长尾。r = π_train / π_infer 是每个 token 的概率比。
- 图上是什么: 左图是第 0 个位置的 12 个 token, 按训练器概率排序: 空心框是 π_train, 实心条是 π_infer, 条上的数字是 log r; 红框表示 r 超过阈值 C。带星号的是这一步采对得 1 分的 token。右图是 64 个位置全部 768 个 token: 横轴是 π_infer (对数刻度), 纵轴是 |log r|, 虚线是 log C, 虚线以上的点会被 TIS 截断或被 MIS 丢掉。
- 可以做什么: 三个精度按钮切换推理引擎的数值路径; 三个校正按钮切换不校正 / TIS / MIS; 滑杆调阈值 C 和 logits 的尖锐程度; 点左图任意一个 token, 把它设成得分 token; 换一组随机数。
- 应该看到: 右图的点越往左 (引擎越觉得不可能) 越高, 不一致集中在低概率 token。右侧读数:
  - 「|log r| 均值」按 π_infer 加权, 即采样到的 token 上的平均不一致。
  - 「梯度相对偏差」是 ‖E[ĝ] − g‖ / ‖g‖, g 是真正的同策略梯度。当前校正方式不高于不校正时变绿。
  - 「超过 C 的概率质量」是会被截断或屏蔽的那部分采样。"
    module="llm_finetune/methods/is_correction.py"
    run="python -m llm_finetune.run_finetune.rollout_mismatch.train_rollout_mismatch"
    :challenge="{
      ask: '选 bf16+int4、MIS, 把 C 从 2 拖到 1.1。梯度偏差会变小还是变大? 换成 TIS 再看一遍, 为什么两者走向不同?',
      answer: '结论: MIS 的偏差变大, 超过不校正; TIS 的偏差变大一点, 但仍低于不校正。\n- C=2 时几乎没有 r 超过 2 的 token, TIS 和 MIS 都等于完整的重要性采样, 偏差接近 0。\n- C=1.1 时大量普通 token 的 r 也超过阈值。TIS 只是把它们的权重截到 1.1, 方向还在, 只少了一点; MIS 把它们整个丢掉, 这些 token 对梯度的贡献全没了, 偏差比不校正还大。\n- Python 精确计算给出同样的走向 (bf16+int4, C=1.1): 不校正 0.427, TIS 0.118, MIS 0.829。阈值要放在真正的离群值上, 不能卡在正常波动里。',
    }"
  >
    <template #controls>
      <div class="row">
        <button v-for="(n, k) in NOISE" :key="k" type="button" :class="{ active: prec === k }" @click="prec = k">{{ k }}</button>
        <button type="button" @click="seed++">换一组</button>
      </div>
      <div class="row">
        <button v-for="(n, k) in MODES" :key="k" type="button" :class="{ active: mode === k }" @click="mode = k">{{ n }}</button>
      </div>
      <LabSlider v-model="cap" label="阈值 C" :min="1.05" :max="4" :step="0.05" :format="(v) => v.toFixed(2)" />
      <LabSlider v-model="sharp" label="logits 尖锐程度" :min="1" :max="6" :step="0.5" :format="(v) => v.toFixed(1)" />
    </template>

    <div class="two">
      <svg viewBox="0 0 330 220" role="group" aria-label="第 0 个位置 12 个 token 的两种概率">
        <text x="6" y="14" class="t">第 0 个位置: 空心 π_train, 实心 π_infer</text>
        <g v-for="(b, i) in bars" :key="b.v" role="button" tabindex="0" :aria-label="`把 token ${b.v} 设成得分 token`"
           class="tok" @click="target = b.v" @keydown.enter="target = b.v">
          <rect :x="18 + i * 25" :y="190 - b.q * 160" width="16" :height="b.q * 160" fill="var(--eye)" opacity="0.75" />
          <rect :x="16 + i * 25" :y="190 - b.p * 160" width="20" :height="b.p * 160" fill="none"
                :stroke="b.over ? 'var(--danger)' : 'var(--accent)'" :stroke-width="b.over ? 2 : 1.2" />
          <text :x="26 + i * 25" :y="182 - Math.max(b.p, b.q) * 160" text-anchor="middle" class="t s">{{ b.lr }}</text>
          <text :x="26 + i * 25" y="206" text-anchor="middle" class="t">{{ b.v === positions[0].tgt ? '★' : b.v }}</text>
        </g>
        <line x1="12" x2="320" y1="190" y2="190" stroke="var(--border)" />
      </svg>
      <svg viewBox="0 0 330 220" role="img" aria-label="全部 token 的 π_infer 与 |log r| 散点">
        <text x="6" y="14" class="t">768 个 token: 横轴 π_infer, 纵轴 |log r|</text>
        <line x1="40" x2="320" :y1="yOf(Math.log(cap))" :y2="yOf(Math.log(cap))" stroke="var(--danger)" stroke-dasharray="4 3" />
        <text x="318" :y="yOf(Math.log(cap)) - 4" text-anchor="end" class="t" fill="var(--danger)">log C</text>
        <circle v-for="(d, i) in sim.dots" :key="i" :cx="xOf(d.q)" :cy="yOf(d.lr)" r="1.8"
                :fill="d.lr > Math.log(cap) ? 'var(--danger)' : 'var(--text-muted)'" opacity="0.7" />
        <line x1="40" x2="320" y1="190" y2="190" stroke="var(--border)" />
        <text v-for="e in [-4, -3, -2, -1, 0]" :key="e" :x="xOf(10 ** e)" y="206" text-anchor="middle" class="t">1e{{ e }}</text>
        <text x="36" y="34" text-anchor="end" class="t">{{ yMax.toFixed(2) }}</text>
        <text x="36" y="190" text-anchor="end" class="t">0</text>
      </svg>
    </div>

    <template #stats>
      <div class="kv"><span>|log r| 均值 (按 π_infer 加权)</span><b>{{ sim.meanAbs.toFixed(4) }}</b></div>
      <div class="kv"><span>梯度相对偏差, {{ MODES[mode] }}</span><b :class="sim.bias[mode] <= sim.bias.none + 1e-12 ? 'good' : 'bad'">{{ sim.bias[mode].toFixed(4) }}</b></div>
      <div class="kv"><span>不校正时的偏差</span><b>{{ sim.bias.none.toFixed(4) }}</b></div>
      <div class="kv"><span>超过 C 的概率质量</span><b :class="sim.hit < 0.05 ? 'good' : 'bad'">{{ (sim.hit * 100).toFixed(2) }}%</b></div>
      <div class="lab-note">
        <p>Python 实测 (sort 任务的 2 层模型, 沿正确前缀逐位置精确求和), 实测, 不随左侧变化:</p>
        <ul class="pts">
          <li><b>|log r| 均值:</b> fp16 0.00094, bf16 0.00771, bf16+int4 0.0418; 低概率 token 的 |log r| 是高概率的 37–67 倍。</li>
          <li><b>不校正偏差:</b> fp16 0.0069, bf16 0.052, bf16+int4 0.427。C=2 时 TIS 0.017, MIS 0.058。</li>
          <li><b>GRPO 60 步 pass@1:</b> fp32 0.324, bf16 0.303, bf16+int4 不校正 0.316 / TIS 0.297 / MIS 0.318, 单种子, 差别在噪声内。</li>
        </ul>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { clamp, mulberry32, randn, range, softmax } from '@/utils/labmath.js'

// 每种数值路径的 logit 噪声: 按 Python 实测的 |log r| 均值标定 (默认尖锐程度 3 时量级一致)
const NOISE = { fp16: 0.002, bf16: 0.016, 'bf16+int4': 0.08 }
const MODES = { none: '不校正', tis: 'TIS 截断', mis: 'MIS 屏蔽' }
const V = 12, NPOS = 64

// target = −1: 第 0 个位置的得分 token 按 π_train 抽; 点过某个 token 之后固定为它
const prec = ref('bf16'), mode = ref('tis'), cap = ref(2), sharp = ref(3), seed = ref(1), target = ref(-1)

// ★ 校正权重, 与 is_weights 的 token 级三种相同
const weight = (ratio, m, c) => (m === 'none' ? 1 : m === 'tis' ? Math.min(ratio, c) : ratio <= c ? ratio : 0)

const positions = computed(() => {
  const rand = mulberry32(seed.value * 977)
  return range(NPOS).map((pos) => {
    const z = range(V).map(() => sharp.value * randn(rand))
    // 少数 logit 偏得更远: 实测的不一致有长尾
    const zi = z.map((x) => x + NOISE[prec.value] * randn(rand) * (rand() < 0.05 ? 6 : 1))
    const p = softmax(z), q = softmax(zi)
    let u = rand(), tgt = V - 1
    for (let v = 0, acc = 0; v < V; v++) { acc += p[v]; if (u < acc) { tgt = v; break } }   // 得分 token 按 π_train 抽
    return { p, q, tgt: pos === 0 && target.value >= 0 ? target.value : tgt }
  })
})

const sim = computed(() => {
  const res = { none: 0, tis: 0, mis: 0 }, dots = []
  let den = 0, absSum = 0, hit = 0
  for (const { p, q, tgt } of positions.value) {
    // 真梯度 (对 logits): g_v = π_v (r_v − b_π); 估计量: c_v = q_v w_v (r_v − b_q), ĝ_v = c_v − π_v Σ c
    const gTrue = p.map((pv, v) => pv * ((v === tgt) - p[tgt]))
    den += gTrue.reduce((s, x) => s + x * x, 0)
    for (const m of Object.keys(res)) {
      const c = q.map((qv, v) => qv * weight(p[v] / qv, m, cap.value) * ((v === tgt) - q[tgt]))
      const S = c.reduce((a, b) => a + b, 0)
      res[m] += c.reduce((s, cv, v) => s + (cv - p[v] * S - gTrue[v]) ** 2, 0)
    }
    for (let v = 0; v < V; v++) {
      const lr = Math.abs(Math.log(p[v] / q[v]))
      absSum += q[v] * lr
      if (lr > Math.log(cap.value)) hit += q[v]
      dots.push({ q: q[v], lr })
    }
  }
  return { dots, meanAbs: absSum / NPOS, hit: hit / NPOS,
           bias: Object.fromEntries(Object.entries(res).map(([k, v]) => [k, Math.sqrt(v / den)])) }
})

const bars = computed(() => {
  const { p, q } = positions.value[0]
  return range(V).sort((a, b) => p[b] - p[a]).map((v) => {
    const l = Math.log(p[v] / q[v])
    return { v, p: p[v], q: q[v], over: Math.abs(l) > Math.log(cap.value), lr: (l >= 0 ? '+' : '') + l.toFixed(2) }
  })
})

const yMax = computed(() => Math.max(Math.log(cap.value) * 1.2, ...sim.value.dots.map((d) => d.lr), 0.01))
const xOf = (q) => 40 + (clamp(Math.log10(q), -4, 0) + 4) / 4 * 280
const yOf = (lr) => 190 - clamp(lr / yMax.value, 0, 1) * 160
</script>

<style scoped>
.two { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 8px; }
.t { font-size: 10px; fill: var(--text-muted); font-family: "SF Mono", Menlo, monospace; }
.t.s { font-size: 8px; }
.tok { cursor: pointer; }
.tok:focus-visible rect { stroke: var(--text); }
</style>
