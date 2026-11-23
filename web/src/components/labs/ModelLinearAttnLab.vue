<!--
  线性注意力 / Gated DeltaNet 实验台 (对应 llm_models/layers/sparse/linear_attention.py:GatedDeltaNet)。
  只讲一件事: softmax 注意力把每个 token 的 K/V 原件都留着 (越存越多), 线性注意力只有一块固定大小的状态矩阵 S —— 省内存, 但容量有限, 写法决定记性。
-->
<template>
  <LabFrame
    title="线性注意力 — 固定大小的状态矩阵 vs 越读越长的 KV cache"
    sub="- 模拟什么: 对比两种历史信息存储机制。标准 softmax 注意力将每个历史 token 的 key 与 value 完整存入 KV cache, 序列有多长就保存多少行; 线性注意力 (此处对应 Qwen3-Next 采用的 Gated DeltaNet) 仅维护一个固定维度的状态矩阵 $S$, 每读入一个 token 就将其 $(k, v)$ 写入 $S$, 后续检索某个历史 token 的 value 时, 直接用其 key 与 $S$ 相乘读出。固定状态显著节省显存, 但存储容量受限, 且具体的写入规则直接决定历史信息的保持精度。
- 图上是什么: 顶部为输入序列 token, 下方分左、中、右三个视图:
  - 顶部序列: t0 至 t9 共 10 个 token, 高亮框表示当前播放位置; 带虚线框的 t1 与 t6 刻意共享同一个 key, 用于模拟同一键名被连续赋值两次的场景。
  - 左侧矩阵: $4 \times 4$ 网格展示状态矩阵 $S$ 的 16 个数值, 颜色深浅代表绝对值大小 (紫色为正、粉色为负), 元素总数恒为 16。
  - 中间列表: 标准 softmax 注意力对照组的 KV cache, 每读入一个 token 新增一行 (含 4 维 k 与 4 维 v)。
  - 右侧柱图: 用各历史 token 的 key 查询 $S$ 时, 实际读回的 value 与目标 value 之间的相对误差; 绿色表示低于 15%, 黄色为 15% 至 50%, 红色为超过 50%。
- 可以做什么: 按界面控件顺序操作:
  - 「delta rule (先擦后写)」: DeltaNet 的更新策略, 先用当前 $k$ 读出 $S$ 在该方向上的已有分量并减去, 再写入新值。
  - 「纯累加」: 基础线性注意力策略, 直接向 $S$ 累加外积项 $k \cdot v^\top$。
  - 「换一组」: 重新生成随机 key 与 value 向量。
  - 拖动「衰减门 α」: 取 1 表示不遗忘, 数值越小历史状态衰减越快。
  - 拖动「写入强度 β」: 取 1 表示完全覆写, 数值减小时仅按比例部分写入。
  - 播放控件: 逐 token 观察状态矩阵 $S$ 的演化过程。
  - 点击顶部任意已读入的 token: 右侧面板展示以该 token 的 key 查询 $S$ 的详细读数。
- 应该看到: 随播放推进观察误差柱图及右侧指标:
  - 前 4 个 token: 写入后读回误差均接近 0 (4 维空间最多可容纳 4 个两两正交的 key)。
  - 从 t4 起: 新增 key 必然与已有 key 存在非零内积, 写入操作会干扰旧方向上的存储值, 导致所有历史 token 的读回误差同步上升。
  - 到达 t6 之后: 点击 t1 时, delta rule 模式下能够准确读回 t6 覆写的新值 (误差小), 而纯累加模式下读回的是 $v_1$ 与 $v_6$ 的叠加值 (误差大)。
  - 「KV cache 大小」: 对照组缓存的标量总数 $(t + 1) \times 8$, 超过 16 后即大于状态矩阵 $S$ 的固定开销。
  - 「查询 tj 的读回误差」: 当前点击选中的历史 token 的相对读回误差。
  - 「所有旧 token 平均误差」: 衡量状态矩阵对全部已读入 token 的整体保持精度。
  - 底部两行向量: 对比该查询下的目标应读回 value 与实际读回 value。"
    module="llm_models/layers/sparse/linear_attention.py"
    run="python -m llm_models.run_models.language_models.qwen3_next.infer_qwen3_next"
    :challenge="{
      ask: '将进度播放至末尾 ($t = 9$), 先在「纯累加」模式下点击 t1, 再切换至「delta rule」模式点击 t1: 哪种模式读回的是 t6 写入的新值? 进一步思考: 当 key 维度仅为 4 时, 写入第 5 个不同的 key 之后, 为什么所有历史 token 的读回误差都会开始上升?',
      answer: '答案: delta rule 模式读回的是 t6 写入的新值, 而纯累加模式读回的是 t1 与 t6 两个 value 的混合结果。\n原因分析:\n- 纯累加更新 $S \\mathrel{+}= k \\cdot v^\\top$ 只做线性叠加。由于 t1 与 t6 共享同一个 key, 以该 key 查询时会同时读出 $v_1 + v_6$ (若开启衰减则附带相应衰减系数), 旧值无法被清除, 导致「应读回」与「实际读回」两行数值明显偏离。\n- delta rule 采用先擦除后写入的更新公式 $S \\leftarrow \\alpha(S - \\beta \\cdot k \\cdot k^\\top S) + \\beta \\cdot k \\cdot v^\\top$: 先用 $k$ 提取 $S$ 在该方向上的当前投影 $k^\\top S$ 并予以扣除, 再写入新值。当 $\\beta = 1$ 时等价于在该方向做精确覆写, 查询结果即为 $v_6$; 若将 $\\beta$ 调小, 旧分量擦除不彻底, 读回误差便会重新出现。\n存储容量瓶颈在两种写入规则下同样存在: 4 维向量空间最多只能容纳 4 个两两正交的 key, 唯有相互正交的 key 才能互不干扰地独立存取。第 5 个不同的 key 必然与部分历史 key 存在非零内积, 写入新值时会连带改动那些历史 key 方向上的投影分量, 因此从 t4 起所有旧 token 的误差都会上升, 且越早写入的 token 经历的串扰次数越多。\n固定大小状态矩阵的本质代价是有损压缩: 令 $\\alpha &lt; 1$ 可使历史状态整体衰减以为新信息腾出容量, 本质是以遗忘远期历史换取近期记忆精度。为此, Qwen3-Next 每 4 层保留 1 层标准全注意力层, 用于兜底精确的远距离信息召回。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: mode === 'delta' }" @click="mode = 'delta'">delta rule (先擦后写)</button>
        <button type="button" :class="{ active: mode === 'add' }" @click="mode = 'add'"><Tex text="纯累加 $S \mathrel{+}= k \cdot v^\top$" /></button>
        <button type="button" @click="seed++">换一组</button>
      </div>
      <LabSlider v-model="alpha" label="衰减门 α" :min="0.6" :max="1" :step="0.01" :format="(t) => t.toFixed(2)" />
      <LabSlider v-model="beta" label="写入强度 β" :min="0.1" :max="1" :step="0.05" :format="(t) => t.toFixed(2)" />
      <StepPlayer :stepper="stepper" :label="`t = ${t}`" />
    </template>

    <div class="cells toks" :style="{ gridTemplateColumns: `repeat(${T}, minmax(24px, 34px))` }" role="group" aria-label="点一个旧 token 作为查询">
      <button
        v-for="j in T" :key="j" type="button" class="cell tok"
        :class="{ dim: j - 1 > t, on: j - 1 === t, active: j - 1 === qj, same: keyOf[j - 1] !== j - 1 || j - 1 === DUP_OF }" :disabled="j - 1 > t"
        @click="qj = j - 1"
      >t{{ j - 1 }}</button>
    </div>

    <div class="wrap">
      <div>
        <p class="cap">线性注意力的全部记忆: 状态 S (4×4 = 16 个数, 永不增长; 紫正粉负)</p>
        <div class="cells" style="grid-template-columns: repeat(4, 34px)">
          <span v-for="(x, i) in now.S.flat()" :key="i" class="cell big mono" :style="{ background: heat(Math.abs(x) / 1.5, x >= 0 ? 'var(--accent)' : 'var(--right)') }">{{ x.toFixed(1) }}</span>
        </div>
      </div>
      <div>
        <p class="cap">对照: softmax 注意力的 KV cache, 每个 token 一行 (4 个 k + 4 个 v): {{ t + 1 }} 行 × 8 = {{ (t + 1) * 8 }} 个数</p>
        <div class="kvlist">
          <div v-for="j in t + 1" :key="j" class="kvrow" :class="{ hit: j - 1 === qj }"><i v-for="c in 4" :key="'k' + c" class="k" /><i v-for="c in 4" :key="'v' + c" class="v" /></div>
        </div>
      </div>
      <div class="errs">
        <p class="cap">用各旧 token 的 key 读 S, 读回的 value 相对误差 (绿 &lt; 15%, 红 &gt; 50%)</p>
        <div class="ebars">
          <div v-for="(e, j) in now.errs" :key="j" class="eb" :class="{ hit: j === qj }">
            <div class="col"><div :class="e > 0.5 ? 'bad' : e > 0.15 ? 'mid' : 'ok'" :style="{ height: Math.min(1, e) * 100 + '%' }" /></div>
            <span class="mono">t{{ j }}</span>
          </div>
        </div>
      </div>
    </div>

    <template #stats>
      <div class="kv"><span>softmax 注意力的 KV cache 存了几个数 (状态 S 恒为 16)</span><b :class="(t + 1) * 8 > 16 ? 'bad' : ''">{{ (t + 1) * 8 }}</b></div>
      <div class="kv"><span>用 t{{ qSafe }} 的 key 读 S 的相对误差</span><b :class="now.errs[qSafe] < 0.15 ? 'good' : 'bad'">{{ (now.errs[qSafe] * 100).toFixed(0) }}%</b></div>
      <div class="kv"><span>所有旧 token 的平均误差 (整体记性)</span><b>{{ (mean(now.errs) * 100).toFixed(0) }}%</b></div>
      <p class="lab-note mono">t{{ qSafe }} 应读回 {{ vec(now.target[qSafe]) }}<br>实际读回 {{ vec(now.read[qSafe]) }}</p>
      <div class="lab-note">
        <p>「应读回」定义为该 key 最近一次写入的 value: 由于 t1 与 t6 共享同一个 key, 在 t6 完成写入后, 点击 t1 对应的目标值也更新为 t6 的 value。</p>
        <p>标准 softmax 注意力因完整保留了所有历史 token 的 KV cache, 检索误差恒为 0; 其代价是缓存体积随序列长度线性增长, 而状态矩阵 S 始终固定为 16 个标量。</p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import Tex from '@/components/Tex.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import StepPlayer from '@/components/lab/StepPlayer.vue'
import { useStepper } from '@/composables/useStepper.js'
import { mulberry32, randn, range, heat, sum } from '@/utils/labmath.js'

const T = 10, Dm = 4, DUP = 6, DUP_OF = 1 // t6 复用 t1 的 key
const mode = ref('delta'), alpha = ref(1), beta = ref(1), seed = ref(1), qj = ref(1)
const keyOf = range(T).map((j) => (j === DUP ? DUP_OF : j))
const mean = (a) => sum(a) / a.length
const vec = (a) => '[' + a.map((x) => x.toFixed(2)).join(', ') + ']'

const data = computed(() => {
  const r = mulberry32(seed.value * 97)
  const unit = () => { const v = range(Dm).map(() => randn(r)); const n = Math.hypot(...v); return v.map((x) => x / n) } // key 做 L2 归一化, 与 Python 一致
  const ks = range(T).map(unit)
  ks[DUP] = ks[DUP_OF]
  return { ks, vs: range(T).map(() => range(Dm).map(() => randn(r))) }
})

const frames = computed(() => {
  const { ks, vs } = data.value, a = alpha.value, b = beta.value
  let S = range(Dm).map(() => Array(Dm).fill(0))
  return range(T).map((t) => {
    const k = ks[t], v = vs[t]
    const kS = range(Dm).map((e) => sum(k.map((kd, d) => kd * S[d][e])))   // kᵀS: k 方向当前存的值
    // ★ delta: S ← α(S − β k (kᵀS)) + β k vᵀ;  纯累加: S ← αS + β k vᵀ
    S = S.map((row, d) => row.map((x, e) => a * (x - (mode.value === 'delta' ? b * k[d] * kS[e] : 0)) + b * k[d] * v[e]))
    const read = range(t + 1).map((j) => range(Dm).map((e) => sum(ks[j].map((kd, d) => kd * S[d][e]))))
    // 目标: 这个 key 最近一次写入的 value (t1 在 t6 之后应读回 v6)
    const target = range(t + 1).map((j) => vs[Math.max(...range(t + 1).filter((i) => keyOf[i] === keyOf[j]))])
    const errs = read.map((o, j) => Math.hypot(...o.map((x, e) => x - target[j][e])) / Math.hypot(...target[j]))
    return { S, read, target, errs }
  })
})

const stepper = useStepper(() => T, { interval: 700 })
stepper.step.value = T - 1
const t = computed(() => stepper.step.value)
const now = computed(() => frames.value[t.value])
const qSafe = computed(() => Math.min(qj.value, t.value))
watch(t, (v) => { if (qj.value > v) qj.value = v })
</script>

<style scoped>
.toks { margin-bottom: 14px; }
.tok { padding: 0; min-height: 26px; height: 26px; cursor: pointer; }
.tok.same { border-style: dashed; border-width: 2px; border-color: var(--warn); }
.wrap { display: flex; flex-wrap: wrap; gap: 20px; align-items: flex-start; }
.cap { font-size: 11px; color: var(--text-dim); margin-bottom: 6px; }
.cell.big { width: 34px; height: 30px; color: var(--text); }
.kvlist { display: grid; gap: 2px; }
.kvrow { display: flex; width: max-content; gap: 1px; padding: 1px; border: 1px solid transparent; border-radius: 2px; }
.kvrow.hit { border-color: var(--accent); }
.kvrow i { width: 9px; height: 7px; border-radius: 1px; }
.kvrow .k { background: color-mix(in srgb, var(--eye) 60%, transparent); }
.kvrow .v { background: color-mix(in srgb, var(--left) 60%, transparent); }
.ebars { display: flex; gap: 4px; }
.eb { display: grid; justify-items: center; gap: 2px; font-size: 9px; color: var(--text-muted); border: 1px solid transparent; border-radius: 3px; padding: 1px; }
.eb.hit { border-color: var(--accent); }
.col { width: 14px; height: 70px; display: flex; align-items: flex-end; border-bottom: 1px solid var(--border-strong); }
.col div { width: 100%; min-height: 1px; border-radius: 2px 2px 0 0; }
.col .ok { background: var(--left); } .col .mid { background: var(--warn); } .col .bad { background: var(--danger); }
</style>
