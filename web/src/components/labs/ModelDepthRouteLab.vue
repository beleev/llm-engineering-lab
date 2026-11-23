<!--
  条件深度实验台 (对应 llm_models/models/language_models/conditional_depth.py: topk_select / MoDLLaMA.forward)。
  只讲一件事: MoD 按整段序列取前 k 名过层, 计算量随容量线性降; 但 "前 k 名" 的门槛依赖后面的 token, 生成时不满足因果,
  换成只看本位置的预测器才行。玩具: 路由分数用随机数, "=" 位置偏高 (Python 训练后 "=" 占入选的一半)。
-->
<template>
  <LabFrame
    title="MoD 路由 — 只让前 k 名过这一层, 可前 k 名要看完整段才知道"
    sub="- 模拟什么: Mixture-of-Depths 在被路由的层前放一个路由器, 给每个 token 打一个分数, 整段序列里分数前 k 名的 token 过这一层 (注意力和 FFN), 其余 token 走残差直接跳过。容量 C = k / T。这里的序列取自 Python 训练任务: 每 4 个 token 一组 [a, b, =, 答案]。
- 图上是什么: 每根柱子是一个 token 的路由分数, 下面写着 token。亮色柱子是入选、过这一层的 token, 灰色是跳过的。虚线是入选门槛: top-k 模式下是第 k 名的分数, 预测器模式下是一个固定的阈值。最后 4 个位置是 '后面的 token', 前 12 个位置里入选状态和改动前不同的, 柱子带红框。
- 可以做什么: 两个按钮在 '整段 top-k (训练时)' 和 '因果预测器 (生成时)' 之间切换。滑杆调容量。点击最后 4 根柱子中的任意一根 (或选中后按回车), 把那个 token 换成一个分数很高的 token, 再点一次换回来。'换一组' 重新抽分数。
- 应该看到: top-k 模式下, 改动后面的 token 会抬高门槛, 前面一些 token 被挤出入选, 右侧 '前 12 个位置翻转' 变成非零; 预测器模式下每个 token 只和固定阈值比, 改动后面的 token 不影响前面, 翻转始终是 0, 代价是入选个数不再固定。每 token FLOPs 按 4 层、第 1 和第 3 层路由来算, 随容量线性下降。"
    module="llm_models/models/language_models/conditional_depth.py"
    run="python -m llm_models.run_models.language_models.conditional_depth.infer_conditional_depth"
    :challenge="{
      ask: '容量设为 25% (k = 4), 留在 top-k 模式, 把最后 4 个 token 都点成高分。前 12 个位置会有几个被挤出入选? 切到预测器模式再做一次呢?',
      answer: '答案: top-k 模式下, 后面 4 个高分 token 正好占满 4 个名额, 前 12 个位置里原来入选的全部被挤出; 预测器模式下翻转是 0。\n原因:\n- top-k 的门槛是整段序列的第 k 名分数。前面 token 自己的分数一点没变 (它们只依赖前文), 变的是门槛, 所以它们是否过这一层取决于后面还没生成的 token。训练时整段都在, 没有问题; 逐个生成时第 i 个 token 要马上决定过不过层, 这时后面的分数还不存在。\n- 预测器只看本位置的激活, 输入只依赖前文, 所以满足因果。它是用 top-k 的决定当标签训练的 (输入 stop-gradient, 不改主模型), Python 训练脚本里与 top-k 的一致率是 0.929, 入选比例从固定的 1/3 变成 0.293。\n- Python 的 infer 脚本里, 只改后 16 个 token, 前 48 个位置的入选决定翻转了 22 次; 换成预测器后翻转 0 次。'
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: mode === 'topk' }" @click="mode = 'topk'">整段 top-k (训练时)</button>
        <button type="button" :class="{ active: mode === 'pred' }" @click="mode = 'pred'">因果预测器 (生成时)</button>
        <button type="button" @click="seed++; edited = []">换一组</button>
      </div>
      <LabSlider v-model="k" label="容量 C = k / 16" :min="1" :max="16" :format="(v) => `${v}/16 = ${Math.round((v / T) * 100)}%`" />
    </template>

    <svg ref="svg" :viewBox="`0 0 ${W} 170`" class="bars" role="group" aria-label="每个 token 的路由分数, 后 4 个可点击改动">
      <line :x1="0" :x2="W" :y1="yOf(thr)" :y2="yOf(thr)" class="thr" />
      <text :x="W - 4" :y="yOf(thr) - 4" text-anchor="end" class="lbl">门槛 {{ thr.toFixed(2) }}</text>
      <g v-for="(s, i) in scores" :key="i" :transform="`translate(${i * 30 + 6}, 0)`">
        <rect :y="yOf(s)" width="22" :height="Math.max(1, 140 - yOf(s))" :class="['bar', sel[i] ? 'on' : 'offbar', { flip: flips.includes(i) }]" />
        <rect v-if="i >= PAST" x="-2" y="0" width="26" height="160" class="hit" :class="{ edited: edited.includes(i) }"
              tabindex="0" role="button" :aria-pressed="edited.includes(i)" :aria-label="`改动位置 ${i} 的 token`"
              @click="toggle(i)" @keydown.enter="toggle(i)" />
        <text x="11" y="156" text-anchor="middle" class="tok mono">{{ edited.includes(i) ? '★' : TOKENS[i] }}</text>
      </g>
      <line :x1="PAST * 30 + 2" :x2="PAST * 30 + 2" y1="0" y2="160" class="sep" />
      <text :x="PAST * 30 + 6" y="10" class="lbl">后面的 token →</text>
    </svg>

    <template #stats>
      <div class="kv"><span>入选 token 数 (top-k 固定为 k)</span><b>{{ selCount }} / {{ T }}</b></div>
      <div class="kv"><span>前 {{ PAST }} 个位置里路由翻转</span><b :class="flips.length === 0 ? 'good' : 'bad'">{{ flips.length }}</b></div>
      <div class="kv"><span>每 token 稠密 FLOPs (相对全容量, 4 层隔层路由)</span><b :class="flopsRel < 1 ? 'good' : ''">{{ flopsRel.toFixed(3) }}</b></div>
      <div class="kv"><span>预测器与 top-k 的一致率 (本序列)</span><b>{{ agree.toFixed(2) }}</b></div>
      <div class="lab-note">
        <p>FLOPs 按 (2 + 2C) / 4 层当量估算: 第 0、2 层全量, 第 1、3 层只算 C 的 token。Python 实测 (T = 64, 含注意力): 容量 12.5% 时是全容量的 0.560。</p>
        <p>Python 实测, 不随左侧变化: 只改后 16 个 token, top-k 路由下前 48 个位置翻转 22 次, 预测器路由下 0 次。</p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { mulberry32 } from '@/utils/labmath.js'

const T = 16, PAST = 12, W = T * 30 + 8
const TOKENS = ['3', '5', '=', '8', '1', '4', '=', '5', '9', '9', '=', '8', '2', '6', '=', '8']
const mode = ref('topk'), k = ref(4), seed = ref(1), edited = ref([])
const yOf = (s) => 140 - s * 45                                    // 分数 0..3 映射到柱高

// 路由分数: "=" 位置偏高, 其余随机; 被改动的位置换成一个高分 token
const base = computed(() => {
  const r = mulberry32(seed.value * 977)
  return TOKENS.map((t) => (t === '=' ? 1.1 : 0.3) + r() * 1.2)
})
const scores = computed(() => base.value.map((s, i) => (edited.value.includes(i) ? 2.9 - 0.05 * (i - PAST) : s)))   // 各不相同, 避免并列
const kth = (arr, n) => [...arr].sort((a, b) => b - a)[n - 1]
// 预测器的阈值: 只由容量决定 (训练时学到的), 不看当前序列; 这里取分数分布的 (1 − C) 分位数
const predThr = computed(() => {
  const r = mulberry32(12345), pool = []
  for (let n = 0; n < 250; n++) TOKENS.forEach((t) => pool.push((t === '=' ? 1.1 : 0.3) + r() * 1.2))
  return kth(pool, Math.max(1, Math.round((pool.length * k.value) / T)))
})
const select = (s) => (mode.value === 'topk' ? s.map((x) => x >= kth(s, k.value)) : s.map((x) => x >= predThr.value)) // ★ 两种门槛
const sel = computed(() => select(scores.value))
const thr = computed(() => (mode.value === 'topk' ? kth(scores.value, k.value) : predThr.value))
const flips = computed(() => {
  const before = select(base.value)
  return sel.value.map((v, i) => (i < PAST && v !== before[i] ? i : -1)).filter((i) => i >= 0)
})
const selCount = computed(() => sel.value.filter(Boolean).length)
const flopsRel = computed(() => (2 + 2 * (k.value / T)) / 4)
const agree = computed(() => {
  const a = scores.value.map((x) => x >= kth(scores.value, k.value)), b = scores.value.map((x) => x >= predThr.value)
  return a.filter((v, i) => v === b[i]).length / T
})
function toggle(i) { edited.value = edited.value.includes(i) ? edited.value.filter((j) => j !== i) : [...edited.value, i] }
</script>

<style scoped>
.bars { width: 100%; min-width: 420px; display: block; }
.bar.on { fill: var(--accent); }
.bar.offbar { fill: var(--border-strong); }
.bar.flip { stroke: var(--danger); stroke-width: 2.5; }
.hit { fill: transparent; cursor: pointer; }
.hit.edited { fill: var(--accent-soft); }
.hit:focus { outline: none; stroke: var(--text); stroke-width: 1; }
.thr { stroke: var(--warn); stroke-width: 1.5; stroke-dasharray: 5 3; }
.sep { stroke: var(--text-dim); stroke-dasharray: 2 3; }
.lbl { font-size: 9px; fill: var(--text-dim); }
.tok { font-size: 11px; fill: var(--text-muted); }
</style>
