<!--
  线性注意力 / Gated DeltaNet 实验台 (对应 llm_models/layers/sparse/linear_attention.py:GatedDeltaNet)。
  只讲一件事: softmax 注意力把每个 token 的 K/V 原件都留着 (越存越多), 线性注意力只有一块固定大小的状态矩阵 S —— 省内存, 但容量有限, 写法决定记性。
-->
<template>
  <LabFrame
    title="线性注意力 — 固定大小的状态矩阵 vs 越读越长的 KV cache"
    sub="- 模拟什么: 两种「记住历史」的方式。普通 softmax 注意力把每个 token 的 key 和 value 原件都存进 KV cache, 读多少 token 就存多少行。线性注意力 (这里是 Qwen3-Next 用的 Gated DeltaNet) 只有一块固定大小的状态矩阵 $S$: 每来一个 token, 就把它的 $(k, v)$ 写进 $S$; 以后想取某个 token 的 value, 就用它的 key 去乘 $S$ 读回来。省内存, 但容量有限, 而且「怎么写」决定了记性好不好。
      - 图上是什么: 最上面一排 t0 到 t9 是 10 个 token, 高亮的是播放到的当前 token, 虚线框的 t1 和 t6 故意用了同一个 key (同一个「变量名」被赋了两次值)。左边 4×4 的格子是 $S$ 的 16 个数, 颜色深浅是绝对值大小, 紫色正、粉色负; 它永远是 16 个数。中间是 softmax 注意力对照组的 KV cache, 每读一个 token 就多一行 (4 个 k 加 4 个 v)。右边的柱子是「用每个旧 token 的 key 去读 $S$, 读回来的 value 和它应该读回的 value 差多少」(相对误差): 绿色小于 15%, 黄色 15% 到 50%, 红色超过 50%。
      - 可以做什么: 「delta rule (先擦后写)」是 DeltaNet 的写法: 先用 $k$ 读出 $S$ 里这个方向当前存的值, 减掉, 再写入新值; 「纯累加」是最原始的线性注意力, 只往 $S$ 里加 $k \cdot v^\top$。「换一组」换一组随机的 key 和 value。拖「衰减门 α」(1 是不遗忘, 越小旧内容衰减越快); 拖「写入强度 β」(1 是完全覆写, 小了只写进去一部分)。用播放控件逐 token 看 $S$ 怎么变。点上排任意一个已经读过的 token, 右侧就显示用它的 key 读 $S$ 的结果。
      - 应该看到: 前 4 个 token 写进去时误差都接近 0 (4 维空间里最多放得下 4 个互相正交的 key); 从 t4 开始, 新 key 不可避免地和旧 key 有夹角, 写它就污染旧的槽位, 所有旧 token 的误差一起上升。到 t6 之后: delta rule 下点 t1, 读回的是 t6 的新值 (误差小); 纯累加下读回的是 $v_1$ 和 $v_6$ 的混合 (误差大)。右侧「KV cache 大小」是对照组存的数字个数 (t+1) × 8, 超过 16 就比 $S$ 大了; 「查询 tj 的读回误差」是你点的那个 token 的误差; 「所有旧 token 平均误差」反映整体记性; 下面两行是应该读回的 value 和实际读回的 value。"
    module="llm_models/layers/sparse/linear_attention.py"
    run="python -m llm_models.run_models.language_models.qwen3_next.infer_qwen3_next"
    :challenge="{
      ask: '播放到最后 (t = 9), 先用「纯累加」点 t1, 再切到「delta rule」点 t1。哪个读回的是 t6 写入的新值? 然后想: key 只有 4 维, 写入第 5 个不同的 key 之后, 为什么所有旧 token 的误差都开始上升?',
      answer: '答案: delta rule 读回的是 t6 的新值; 纯累加读回的是 t1 和 t6 两个 value 的混合。\n为什么:\n- 纯累加 $S \\mathrel{+}= k \\cdot v^\\top$ 只会往上叠。t1 和 t6 的 key 相同, 用这个 key 去读, 读回来是 $v_1 + v_6$ (再乘上衰减), 旧值永远擦不掉, 「应读回」和「实际读回」两行对不上。\n- delta rule 写入前先擦: 用 $k$ 读出 $S$ 里这个方向当前存的值 $k^\\top S$, 减掉它, 再写新值。公式是 $S \\leftarrow \\alpha(S - \\beta \\cdot k \\cdot k^\\top S) + \\beta \\cdot k \\cdot v^\\top$; $\\beta = 1$ 时等于精确覆写, 读回的就是 $v_6$。把 β 拖小, 擦得不干净, 误差又回来。\n容量问题两种写法都有: 4 维空间里最多只有 4 个互相正交的 key, 正交的 key 互不干扰。第 5 个 key 必然和某些旧 key 有夹角, 写它的时候就会顺带改掉旧 key 方向上存的值, 所以 t4 之后所有旧 token 的误差一起上升, 而且越早的 token 被污染的次数越多。\n这就是状态大小固定的代价: 有损。$\\alpha < 1$ 让旧内容整体衰减, 给新内容腾地方, 是用「忘旧」换「记新」。所以 Qwen3-Next 每 4 层留 1 层全注意力, 兜底精确召回。',
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
        <p>「应读回」是这个 key 最近一次写入的 value: t1 和 t6 共用 key, 所以 t6 写过之后, 点 t1 也应该读回 t6 的 value。</p>
        <p>softmax 注意力的 KV cache 对应的误差恒为 0: 原件都在, 想读谁就读谁。代价是它随 token 数一直长下去, 而 S 永远是 16 个数。</p>
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
