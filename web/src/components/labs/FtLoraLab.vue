<!--
  LoRA 低秩补丁实验台 (对应 llm_finetune/methods/lora.py)。
  只讲一件事: 微调需要的 ΔW 本身就接近低秩, 所以用 r 很小的 B·A 去拟合它, 误差小、参数省;
  训练完 merge 回 W, 推理零开销。
  目标 ΔW* 由已知奇异值谱构造 (U·diag(s)·Vᵀ), 秩 r 的最优近似就是截断 SVD (Eckart–Young), 误差有闭式解。
-->
<template>
  <LabFrame
    title="LoRA — 微调要学的 ΔW 本来就低秩, 所以两个瘦矩阵 B·A 就能拟合它"
    sub="- 模拟什么: 全参微调会把一个 $d\times d$ 的权重矩阵 $W_0$ 改成 $W_0 + \Delta W^*$。LoRA (低秩适配) 不动 $W_0$, 只训两个瘦矩阵 $B$ ($d\times r$) 和 $A$ ($r\times d$), 用它们的乘积 $B\cdot A$ 去逼近 $\Delta W^*$。$r$ 叫「秩」, 就是这两个瘦矩阵窄边的宽度; $r$ 远小于 $d$ 时, 要训的参数从 $d^2$ 降到 $2dr$。这个台子把 $d$ 缩到 12, 让你看 $r$ 取多少才够。
      - 图上是什么: 一排热力图, 每个小格子是矩阵的一个元素, 紫色是正、橙色是负, 越浓绝对值越大。热力图下面的柱状图是 $\Delta W^*$ 的奇异值谱: 每根柱子是矩阵里一个「方向」的强度, 从大到小排, 亮的是秩 $r$ 保留下来的, 暗的是被丢掉的。热力图从左到右:
        1. 冻结的 $W_0$。
        2. 可训的 $B$ 和 $A$ (画的是训练完的样子)。
        3. 它们的乘积 $\Delta W = (\alpha/r)\cdot B\cdot A$。
        4. 最右边是全参微调本来想学到的 $\Delta W^*$, 中间用 ≈ 隔开, 两张图越像说明 LoRA 拟合得越好。
      - 可以做什么: 按界面顺序:
        - 「训练态」: 显示 $W_0$ 和 $B\cdot A$ 分开的样子。
        - 「merge」: 把 $\Delta W$ 加回 $W_0$, 变成一个普通矩阵 $W^\prime$, 推理时不再多算。
        - 「换一组矩阵」: 换随机种子。
        - 拖「秩 r」: 看 $\Delta W$ 和 $\Delta W^*$ 像不像。
        - 拖「ΔW* 谱衰减」: 改 $\Delta W^*$ 有多低秩 (越小衰减越快, 前几个方向就占了几乎全部能量; 越大越接近满秩)。
        - 拖「真实层宽 d」: 只影响右侧的参数量换算, 不改图。
        - 训练态下悬停或点 $\Delta W$ 的任意一格 (聚焦后也能用方向键), 会高亮它由 $B$ 的哪一行和 $A$ 的哪一列相乘得到, 下方给出算式。
      - 应该看到: 谱衰减快时, $r$ 取 2 或 3, $\Delta W$ 已经和 $\Delta W^*$ 几乎一样; 谱很平时要把 $r$ 拖到接近 12 才像。右侧读数:
        - 「可训参数」: $2dr$ 以及它占全参 $d^2$ 的百分比, 按真实层宽算。
        - 「重构相对误差」: $\Delta W$ 和 $\Delta W^*$ 差多少, 低于 10% 变绿。
        - 「缩放 α/r」: 前向里乘在 $B\cdot A$ 前面的系数, $\alpha$ 固定 16。
        - 「推理额外矩阵乘法」: 在训练态是 2 次 (多算 $Ax$ 和 $B(Ax)$), merge 后是 0。"
    module="llm_finetune/methods/lora.py"
    run="python -m llm_finetune.run_finetune.lora.train_lora"
    :challenge="{
      ask: '按顺序做:\n1. 把「谱衰减」拖到 0.95 (ΔW* 几乎满秩)。\n2. 再拖到 0.4 (ΔW* 很低秩)。\n同样 r = 2 时误差差多少? 这说明 LoRA 成立的前提是什么?',
      answer: '答案:\n1. 衰减 0.4 时 r = 2 的重构误差约 16%, 衰减 0.95 时约 86%。\n2. LoRA 成立的前提是「微调带来的权重变化接近低秩」。\n为什么:\n- 误差怎么算: 秩 r 下对 $\\Delta W^*$ 的最优近似就是只保留前 r 个奇异值方向 (截断 SVD, Eckart–Young 定理)。误差 $= \\sqrt{\\sum_{\\text{丢掉}} s_k^2 \\,/\\, \\sum_{\\text{全部}} s_k^2}$, 这个式子在说: 丢掉的方向占了多少能量, 误差就有多大。\n- 谱衰减 0.4: 12 个奇异值按 $0.4^k$ 衰减, 第 3 个起已经很小, 前 2 个方向占了几乎全部能量, 所以 r = 2 够用。\n- 谱衰减 0.95: 12 个奇异值差不多大, 只留 2 个就丢了 10 个方向的能量, 误差自然大。\n- 真实模型里: LoRA 论文和 Aghajanyan 2020 的经验发现是, 预训练模型微调时 $\\Delta W$ 的「内在维度」很低, 所以 d = 4096 的层用 r = 8 就够。任务离预训练分布越远 (例如学一门新语言), $\\Delta W$ 越不低秩, 这时该加大 r 或改用全参。本仓库同样 300 步实测: 全参留出集 EM 0.809, LoRA (r=8) 0.352, LoRA 省的是显存不是步数。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: !merged }" @click="merged = false">训练态: W₀ 与 B·A 分开</button>
        <button type="button" :class="{ active: merged }" @click="merged = true">merge: W′ = W₀ + (α/r)·B·A</button>
        <button type="button" @click="seed++">换一组矩阵</button>
      </div>
      <LabSlider v-model="r" label="秩 r" :min="1" :max="D" />
      <LabSlider v-model="decay" label="ΔW* 谱衰减" :min="0.3" :max="0.95" :step="0.05" :format="(v) => v.toFixed(2)" />
      <LabSlider v-model="logd" label="真实层宽 d (算参数用)" :min="9" :max="13" :format="(v) => String(2 ** v)" />
    </template>

    <div class="mats" @mouseleave="hov = null">
      <template v-for="(m, k) in shown" :key="m.name">
        <span v-if="m.op" class="op mono">{{ m.op }}</span>
        <figure class="mat">
          <!-- 只有 ΔW 可以选格子: 鼠标悬停、触屏点按、键盘方向键三种方式 -->
          <div
            class="cells hm" :style="{ gridTemplateColumns: `repeat(${m.data[0].length}, 12px)` }"
            :tabindex="m.name === 'ΔW' ? 0 : undefined" :role="m.name === 'ΔW' ? 'group' : undefined"
            :aria-label="m.name === 'ΔW' ? `ΔW 矩阵, 方向键选格子。${hovText}` : undefined"
            @focus="m.name === 'ΔW' && moveHov(0, 0)"
            @keydown.up.prevent="moveHov(-1, 0)" @keydown.down.prevent="moveHov(1, 0)"
            @keydown.left.prevent="moveHov(0, -1)" @keydown.right.prevent="moveHov(0, 1)"
          >
            <template v-for="(row, i) in m.data" :key="i">
              <span
                v-for="(v, j) in row" :key="j" class="px"
                :class="{ hl: isHl(m.name, i, j) }"
                :style="{ background: tint(v, m.max) }"
                @mouseenter="m.name === 'ΔW' && (hov = { i, j })"
                @click="m.name === 'ΔW' && (hov = { i, j })"
              />
            </template>
          </div>
          <figcaption class="mono">{{ m.name }} <i>{{ m.shape }}</i><br><em>{{ m.note }}</em></figcaption>
        </figure>
      </template>
    </div>
    <p v-if="!merged" class="lab-note" style="margin-top: 8px;">{{ hovText || '悬停或点 ΔW 的一格: 它等于 B 的那一行和 A 的那一列逐项相乘再求和, 再乘 α/r。' }}</p>
    <div class="spec">
      <span class="mono cap">ΔW* 的奇异值谱, 从大到小 (亮的是秩 r 保留的方向, 暗的是丢掉的)</span>
      <div class="bars">
        <i v-for="(s, k) in sv" :key="k" :class="{ keep: k < r }" :style="{ height: s * 100 + '%' }" :title="`s${k} = ${s.toFixed(3)}`" />
      </div>
    </div>

    <template #stats>
      <div class="kv"><span><Tex text="可训参数 $2dr$ (占全参 $d^2$ 的百分比)" /></span><b>{{ fmtNum(2 * dReal * r) }} ({{ (200 * r / dReal).toFixed(2) }}%)</b></div>
      <div class="kv"><span>重构相对误差 (ΔW 离 ΔW* 多远)</span><b :class="err < 0.1 ? 'good' : err > 0.3 ? 'bad' : ''">{{ (err * 100).toFixed(1) }}%</b></div>
      <div class="kv"><span><Tex text="缩放 $\alpha/r$ ($\alpha = 16$ 固定)" /></span><b>{{ (alpha / r).toFixed(2) }}</b></div>
      <div class="kv"><span>推理时额外的矩阵乘法次数</span><b :class="merged ? 'good' : ''">{{ merged ? 0 : 2 }}</b></div>
      <p class="lab-note">
        <template v-if="merged">
          merge 后 <Tex text="$W^\prime = W_0 + (\alpha/r)\cdot B\cdot A$" /> 就是一个普通的 Linear 层, 推理不多算任何东西。对同一个输入 x, 「分开算」和「merge 后算」的输出差 <Tex text="$\max|\Delta y|$" /> = {{ mergeDiff.toExponential(1) }}, 只是浮点舍入。
        </template>
        <template v-else>
          <Tex text="前向 $y = W_0x + (\alpha/r)\cdot B(Ax)$, 这个式子在说: 原来的输出不动, 另外加一条「先过 $A$ 再过 $B$」的小支路, 所以多两次小矩阵乘法。" /><br />
          <Tex text="系数 $\alpha/r$ 的作用: 换 $r$ 时不必重调学习率。$r$ 翻倍, 每个秩分量的贡献自动减半, 总量不变。" />
        </template>
      </p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import Tex from '@/components/Tex.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { clamp, fmtNum, mulberry32, randn, range } from '@/utils/labmath.js'

const D = 12
const alpha = 16          // 与 LoRALinear 的默认值相同。ΔW 固定时改 α 只会让 B 反向缩放, 图不变, 所以不做成滑杆
const r = ref(2), decay = ref(0.55), logd = ref(12), seed = ref(1)
const merged = ref(false)
const hov = ref(null)
const dReal = computed(() => 2 ** logd.value)

// 随机正交基: 高斯向量做 Gram-Schmidt。cols[k] 是第 k 个单位向量
const orthoBasis = (rand) => {
  const cols = []
  for (let k = 0; k < D; k++) {
    let v = range(D).map(() => randn(rand))
    for (const u of cols) { const p = dot(u, v); v = v.map((x, i) => x - p * u[i]) }
    const n = Math.hypot(...v)
    cols.push(v.map((x) => x / n))
  }
  return cols
}
const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0)

const base = computed(() => {
  const rand = mulberry32(seed.value * 7919)
  return { U: orthoBasis(rand), V: orthoBasis(rand), W0: range(D).map(() => range(D).map(() => randn(rand) * 0.5)) }
})
const sv = computed(() => range(D).map((k) => decay.value ** k))

// 取前 n 个奇异分量求和: Σ s_k · u_k v_kᵀ
const lowRank = (n) => range(D).map((i) => range(D).map((j) =>
  range(n).reduce((s, k) => s + sv.value[k] * base.value.U[k][i] * base.value.V[k][j], 0)))

const target = computed(() => lowRank(D))
const delta = computed(() => lowRank(r.value))            // ★ 秩 r 的最优近似 = 截断 SVD
// 写成 LoRA 的参数形式: ΔW = (α/r)·B·A, 取 A = V_rᵀ, B = U_r·S_r·(r/α)
const A = computed(() => range(r.value).map((k) => base.value.V[k]))
const B = computed(() => range(D).map((i) => range(r.value).map((k) => base.value.U[k][i] * sv.value[k] * r.value / alpha)))
// 闭式误差: sqrt(丢掉的 s² / 全部 s²), 与逐元素相减算出来的 Frobenius 误差一致
const err = computed(() => Math.sqrt(sv.value.slice(r.value).reduce((s, x) => s + x * x, 0) / sv.value.reduce((s, x) => s + x * x, 0)))

const Wm = computed(() => base.value.W0.map((row, i) => row.map((w, j) => w + delta.value[i][j])))
const mergeDiff = computed(() => {
  const x = range(D).map((i) => Math.sin(i + 1)), s = alpha / r.value
  const Ax = A.value.map((a) => dot(a, x))
  const y1 = base.value.W0.map((w, i) => dot(w, x) + s * dot(B.value[i], Ax))
  const y2 = Wm.value.map((w) => dot(w, x))
  return Math.max(...y1.map((v, i) => Math.abs(v - y2[i])))
})

const mk = (name, data, note, op = '') => ({
  name, data, note, op, shape: `${data.length}×${data[0].length}`, max: Math.max(...data.flat().map(Math.abs), 1e-9),
})
const shown = computed(() => (merged.value
  ? [mk('W′', Wm.value, '一个普通矩阵, 补丁消失'), mk('ΔW*', target.value, '想学到的更新', '　')]
  : [
    mk('W₀', base.value.W0, '冻结, 不更新'), mk('B', B.value, '可训 (训练后; 起点全零)', '+'), mk('A', A.value, '可训 (训练后; 起点随机)', '×'),
    mk('ΔW', delta.value.map((row) => row.slice()), '(α/r)·B·A', '='), mk('ΔW*', target.value, '想学到的更新', '≈'),
  ]))

// 悬停 ΔW[i][j] → 高亮 B 的第 i 行、A 的第 j 列
const isHl = (name, i, j) => {
  const h = hov.value
  if (!h) return false
  return (name === 'ΔW' && i === h.i && j === h.j) || (name === 'B' && i === h.i) || (name === 'A' && j === h.j)
}
const moveHov = (di, dj) => {
  const h = hov.value ?? { i: 0, j: 0 }
  hov.value = { i: clamp(h.i + di, 0, D - 1), j: clamp(h.j + dj, 0, D - 1) }
}
const hovText = computed(() => {
  const h = hov.value
  if (!h || merged.value) return ''
  const terms = range(r.value).map((k) => B.value[h.i][k] * A.value[k][h.j])
  return `ΔW[${h.i}][${h.j}] = (α/r) · Σ B[${h.i}][k]·A[k][${h.j}] = ${(alpha / r.value).toFixed(2)} × ${terms.reduce((s, x) => s + x, 0).toFixed(4)} = ${delta.value[h.i][h.j].toFixed(3)}`
})
const tint = (v, max) => `color-mix(in srgb, var(${v >= 0 ? '--accent' : '--right'}) ${Math.round(Math.abs(v) / max * 100)}%, transparent)`
</script>

<style scoped>
.mats { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }
.mat { margin: 0; }
.hm { gap: 1px; width: max-content; }
.px { width: 12px; height: 12px; border-radius: 1px; outline: 1px solid color-mix(in srgb, var(--border) 60%, transparent); outline-offset: -1px; }
.px.hl { outline: 2px solid var(--warn); outline-offset: -1px; }
.hm:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.op { font-size: 18px; color: var(--text-muted); }
figcaption { font-size: 11px; color: var(--text); margin-top: 4px; line-height: 1.4; }
figcaption i { color: var(--text-dim); font-style: normal; }
figcaption em { color: var(--text-dim); font-style: normal; font-size: 10px; }
.spec { margin-top: 14px; }
.cap { font-size: 11px; color: var(--text-dim); }
.bars { display: flex; align-items: flex-end; gap: 3px; height: 48px; margin-top: 4px; max-width: 260px; }
.bars i { flex: 1; background: var(--border-strong); border-radius: 1px; min-height: 1px; }
.bars i.keep { background: var(--accent); }
</style>
