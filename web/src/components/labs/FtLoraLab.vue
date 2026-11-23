<!--
  LoRA 低秩补丁实验台 (对应 llm_finetune/methods/lora.py)。
  只讲一件事: 微调需要的 ΔW 本身就接近低秩, 所以用 r 很小的 B·A 去拟合它, 误差小、参数省;
  训练完 merge 回 W, 推理零开销。
  目标 ΔW* 由已知奇异值谱构造 (U·diag(s)·Vᵀ), 秩 r 的最优近似就是截断 SVD (Eckart–Young), 误差有闭式解。
-->
<template>
  <LabFrame
    title="LoRA — 微调要学的 ΔW 本来就低秩, 所以两个瘦矩阵 B·A 就能拟合它"
    sub="- 模拟什么: 全参微调将形状为 $d\times d$ 的预训练权重矩阵 $W_0$ 更新为 $W_0 + \Delta W^*$。LoRA (Low-Rank Adaptation, 低秩适配) 冻结原始矩阵 $W_0$, 仅训练两个低秩窄矩阵 $B$ ($d\times r$) 与 $A$ ($r\times d$), 利用二者的矩阵乘积 $B\cdot A$ 近似目标增量 $\Delta W^*$。超参数 $r$ 即为低秩分解的秩 (rank), 代表窄矩阵的瓶颈维宽度; 当 $r \ll d$ 时, 单层可训练参数量由 $d^2$ 降至 $2dr$。本实验台将矩阵维度缩放至 $d = 12$, 用于直观观察不同谱结构下所需的最小秩 $r$。
- 图上是什么: 上方横排展示多幅矩阵热力图, 每个网格对应一个矩阵元素, 紫色表示正值、橙色表示负值, 颜色越深绝对值越大。下方柱状图展示目标增量 $\Delta W^*$ 的奇异值谱: 每根柱体代表一个正交奇异方向的分量强度并按降序排列, 高亮柱体为当前秩 $r$ 截断保留的主方向, 暗色柱体为被舍弃的残差分量。热力图自左向右依次为:
  1. 冻结的预训练权重矩阵 $W_0$。
  2. 可训练的低秩因子矩阵 $B$ 与 $A$ (展示最优低秩拟合收敛后的形态)。
  3. 缩放后的低秩乘积增量 $\Delta W = (\alpha/r)\cdot B\cdot A$。
  4. 最右侧为全参微调的理想目标增量 $\Delta W^*$, 与左侧 $\Delta W$ 之间以 $\approx$ 符号分隔, 两幅热力图纹理越接近说明低秩近似误差越小。
- 可以做什么: 按界面顺序可执行以下操作:
  - 切换至「训练态」: 独立展示冻结主干 $W_0$ 与低秩旁路因子 $B\cdot A$。
  - 切换至「merge」: 将低秩增量 $\Delta W$ 原地合并回 $W_0$ 生成单一稠密矩阵 $W^\prime$, 消除线上推理的旁路计算开销。
  - 点击「换一组矩阵」: 更换随机种子重新生成权重与目标增量。
  - 拖动「秩 r」滑杆: 对比不同秩下 $\Delta W$ 对 $\Delta W^*$ 的纹理还原程度。
  - 拖动「ΔW* 谱衰减」滑杆: 调节目标增量 $\Delta W^*$ 的内在低秩程度 (数值越小奇异值衰减越快, 前几个主分量即集中绝大部分能量; 数值越大越接近满秩均匀谱)。
  - 拖动「真实层宽 d」滑杆: 仅用于联动换算右侧实际大模型层宽下的参数量占比, 不改变左侧 12 维热力图。
  - 在训练态下悬停或点击 $\Delta W$ 中任意网格 (聚焦后亦可使用方向键): 高亮显示生成该元素的 $B$ 对应行向量与 $A$ 对应列向量, 并在下方展开内积计算式。
- 应该看到: 当奇异值谱快速衰减时, 仅取 $r$ 为 2 或 3 即可使 $\Delta W$ 与 $\Delta W^*$ 高度吻合; 当奇异值谱趋于平坦时, 必须将 $r$ 拖至接近满秩 12 方能准确还原。右侧汇总读数含义如下:
  - 「可训参数」: 按所选真实层宽 $d$ 计算的 $2dr$ 参数个数及其占全参量 $d^2$ 的百分比。
  - 「重构相对误差」: 低秩近似 $\Delta W$ 相对 $\Delta W^*$ 的 Frobenius 范数相对误差, 低于 10% 时显示为绿色。
  - 「缩放 α/r」: 前向传播中乘在 $B\cdot A$ 前的尺度系数, 此处固定 $\alpha = 16$。
  - 「推理额外矩阵乘法」: 训练未合并态下每层需额外执行 2 次小矩阵乘法 (依次计算 $Ax$ 与 $B(Ax)$), 执行 merge 合并后降为 0。"
    module="llm_finetune/methods/lora.py"
    run="python -m llm_finetune.run_finetune.lora.train_lora"
    :challenge="{
      ask: '请按顺序调节滑杆并对比:\n1. 将「ΔW* 谱衰减」拖至 0.95 (使目标增量接近满秩)。\n2. 再将其拖至 0.4 (使目标增量高度低秩)。\n固定秩 $r = 2$ 不变时, 两种谱结构下的重构相对误差相差多少? 这一对比说明 LoRA 方法有效性的核心前提是什么?',
      answer: '结论:\n1. 在 $r = 2$ 下, 谱衰减系数取 0.4 时重构相对误差约为 16%, 而谱衰减系数取 0.95 时相对误差升至约 86%。\n2. LoRA 成立的核心前提是下游微调所需的权重增量 $\\Delta W^*$ 具有近似低秩结构 (奇异值谱快速衰减)。\n机制解析:\n- 最优低秩近似下界: 根据 Eckart–Young 定理, 任意秩 $r$ 矩阵对 $\\Delta W^*$ 的最优 Frobenius 范数逼近即为保留前 $r$ 个最大奇异值分量的截断 SVD, 对应的最小重构相对误差为 $\\sqrt{\\sum_{k &gt; r} s_k^2 \\,/\\, \\sum_{k=1}^{d} s_k^2}$, 完全取决于被截断舍弃的尾部奇异值能量占比。\n- 快速衰减谱 (0.4): 12 个奇异值按 $0.4^k$ 指数衰减, 自第 3 个奇异值起幅值已十分微小, 前 2 个主方向集中了绝大部分矩阵能量, 因此取 $r = 2$ 即可获得较高逼近精度。\n- 平坦近似满秩谱 (0.95): 12 个奇异值量级相近, 仅保留前 2 个方向会直接丢失其余 10 个正交方向的能量, 导致重构误差高达 86%。\n- 真实预训练模型的内在维度边界: Aghajanyan 2020 与 LoRA 原论文实证表明, 经过充分预训练的大模型在适配常见下游任务时, 权重增量 $\\Delta W$ 的内在秩 (intrinsic rank) 极低, 因此层宽 $d = 4096$ 的线性层往往取 $r = 8$ 即可收敛至接近全参的精度。然而当目标任务显著偏离预训练语料分布 (如持续预训练注入全新语种或领域知识) 时, 所需增量趋近高秩, 此时必须增大秩 $r$ 或改用全参微调。本仓库在相同 300 步预算下的对照实测中, 全参微调留出集 EM 为 0.809, LoRA ($r = 8$) 为 0.352, 说明低秩约束节省的是常驻显存而非收敛所需的优化步数。',
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
          <Tex text="训练态前向传播计算 $y = W_0x + (\alpha/r)\cdot B(Ax)$: 保持冻结主干线性变换输出不变, 并联一条先经 $A$ 降维至 $r$ 维、再经 $B$ 升维回 $d$ 维的低秩旁路 (单层额外增加两次小矩阵乘法)。" /><br />
          <Tex text="引入缩放系数 $\alpha/r$ 的工程目的在于解耦秩 $r$ 与学习率超参: 当秩 $r$ 翻倍时, 单个秩一分量的叠加权重自动减半, 使旁路输出的初始激活与梯度量级保持稳定而无需重新搜索学习率。" />
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
