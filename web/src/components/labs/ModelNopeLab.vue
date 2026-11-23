<!--
  NoPE 与 iRoPE 实验台 (对应 llm_models/models/language_models/hybrid.py: HybridLM.forward 的三类掩码)。
  只讲一件事: 超出训练长度之后, RoPE 层会遇到训练时没见过的相对距离; 分块把距离压回训练范围内,
  NoPE 层干脆没有位置输入。Llama 4 的 iRoPE 就是让两种层交错。
-->
<template>
  <LabFrame
    title="NoPE 与 iRoPE — 超出训练长度时, 哪些相对距离是没见过的"
    sub="- 模拟什么: 同一个 query 在三种层里能看到哪些 key。RoPE 把位置写成按相对距离的旋转, 训练时只见过不超过训练长度的那些距离; 推理时序列一长, 远处的 key 落在没训练过的旋转角上。分块注意力把每个 query 限制在自己那一块之内, 相对距离因此永远不超过块大小。NoPE 层不加任何位置编码, query 和 key 都不带位置信息。
- 图上是什么: 一排格子是全部 key, 从位置 0 排到 T−1, 带框的那个是当前 query。灰色是这一层看不到的, 蓝色是可见且相对距离在训练长度之内的, 红色是可见但相对距离超出训练长度的。竖线标出训练长度的边界。
- 可以做什么: 三个按钮切换层的类型 (全 RoPE 全注意力 / 分块 RoPE / NoPE 全注意力)。两个滑杆分别调评估长度 T 和分块大小。拖动带框的那个格子 (或选中后按左右方向键) 换 query 的位置。
- 应该看到: 全 RoPE 模式下把 T 调到训练长度的两倍以上, query 放在末尾, 左边一大片 key 变红 —— 这些相对距离模型没训练过。切到分块模式, 可见的 key 只剩块内那几个, 红色全部消失, 代价是跨块的信息读不到。切到 NoPE 模式, 全部 key 都可见而且没有红色: 它根本没有位置输入, 也就没有 '没见过的距离' 这回事。"
    module="llm_models/models/language_models/hybrid.py"
    run="python -m llm_models.run_models.language_models.hybrid.train_hybrid"
    :challenge="{
      ask: 'T 设成 128、训练长度 32, query 放在最后一位。全 RoPE 模式下有多少比例的可见 key 落在没训练过的距离上? 分块模式呢? 既然 NoPE 没有这个问题, 为什么 Llama 4 还要留 RoPE 层?',
      answer: '答案: 全 RoPE 模式下 128 个可见 key 里有 96 个 (75%) 的相对距离超过 32; 分块模式下可见 key 只有 16 个, 红色是 0 个。\\n原因:\\n- RoPE 的位置信息是相对距离决定的旋转角。训练长度 32 时, 模型只拟合过距离 0 到 31 的那段角度; 距离 100 的旋转角在训练中从没出现过, 注意力分数因此不可靠。这就是要做 NTK 或 YaRN 插值的原因。\\n- 分块注意力把距离上限压回块大小, 所以 RoPE 层永远工作在训练过的角度里; 代价是跨块的内容读不到, 得交给别的层。\\n- NoPE 层既没有旋转也没有加性位置编码, 顺序信息只能从因果掩码和别的层带进来。Llama 4 让 NoPE 层与分块 RoPE 层交错, 再在推理时对注意力做温度缩放; Meta 把这套叫 iRoPE, i 是 interleaved, 同时指向支持无限长上下文这个目标。Kimi Linear 与 Kimi K3 走的是另一种搭配: 全注意力层一律 NoPE, 顺序交给中间那几层 KDA 的衰减门隐式携带, 扩长度时不用再调 RoPE 的频率基数或上 YaRN。\\n- Python 实测 (train 脚本, 训练长度 32、4 层): 这三种排布在 T = 128 上的准确率都掉到 T = 32 的七成左右, 玩具规模上分不出高下, 详见页面的要点。'
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: mode === 'rope' }" @click="mode = 'rope'">全 RoPE 全注意力</button>
        <button type="button" :class="{ active: mode === 'chunk' }" @click="mode = 'chunk'">分块 RoPE</button>
        <button type="button" :class="{ active: mode === 'nope' }" @click="mode = 'nope'">NoPE 全注意力</button>
      </div>
      <LabSlider v-model="tExp" label="评估长度 T" :min="5" :max="7" :format="() => String(T)" />
      <LabSlider v-model="chunk" label="分块大小" :min="8" :max="64" :step="8" />
    </template>

    <svg :viewBox="`0 0 400 90`" class="keys" role="group" aria-label="key 的可见性与相对距离, query 位置可拖动">
      <text x="0" y="10" class="lbl">key 位置 0 … {{ T - 1 }} (灰 = 看不到, 蓝 = 距离在训练范围内, 红 = 距离没训练过)</text>
      <rect v-for="s in T" :key="s" :x="(s - 1) * cw" y="18" :width="Math.max(1, cw - 0.5)" height="22"
            :class="['key', state(s - 1)]" />
      <rect :x="qpos * cw - 1" y="16" :width="cw + 2" height="26" class="grip draggable"
            tabindex="0" role="slider" :aria-valuenow="qpos" :aria-valuemin="0" :aria-valuemax="T - 1"
            aria-label="query 的位置" @keydown.left="qpos = Math.max(0, qpos - 1)" @keydown.right="qpos = Math.min(T - 1, qpos + 1)" />
      <line :x1="Math.max(0, qpos - TRAIN + 1) * cw" :x2="Math.max(0, qpos - TRAIN + 1) * cw" y1="14" y2="46" class="edge" />
      <text :x="Math.max(0, qpos - TRAIN + 1) * cw + 3" y="56" class="lbl">↑ 相对距离 {{ TRAIN }} 的边界 (训练长度)</text>
      <text x="0" y="76" class="lbl">query 在位置 {{ qpos }}; 这一层{{ mode === 'nope' ? '没有位置输入' : '用 RoPE' }}</text>
    </svg>

    <template #stats>
      <div class="kv"><span>可见 key 数 (全长是 {{ qpos + 1 }})</span><b :class="visible < qpos + 1 ? 'bad' : 'good'">{{ visible }}</b></div>
      <div class="kv"><span>其中相对距离没训练过的</span><b :class="unseen === 0 ? 'good' : 'bad'">{{ unseen }}</b></div>
      <div class="kv"><span>没训练过的占可见 key 的比例</span><b :class="unseen === 0 ? 'good' : 'bad'">{{ visible ? (100 * unseen / visible).toFixed(1) : '0.0' }}%</b></div>
      <div class="kv"><span>这一层遇到的最大相对距离 (训练上限 {{ TRAIN - 1 }})</span>
        <b :class="maxDist <= TRAIN - 1 ? 'good' : 'bad'">{{ mode === 'nope' ? '没有位置输入' : maxDist }}</b></div>
      <div class="lab-note">
        <p>训练长度固定取 32, 与 Python 训练脚本一致。NoPE 层没有位置输入, 所以 "没训练过的距离" 这一项恒为 0; 它的顺序信息只能来自因果掩码和别的层。</p>
        <p>Python 实测, 不随左侧变化: 植入针的检索任务, 4 层、训练长度 32、1200 步, 答案位准确率按评估长度列出。</p>
        <ul class="pts">
          <li v-for="r in measured" :key="r.name"><b>{{ r.name }}:</b> T=32 {{ r.a32 }} · T=64 {{ r.a64 }} · T=128 {{ r.a128 }} ({{ r.keep }} 倍)</li>
        </ul>
        <p>全 RoPE 在训练长度上满分, 长度翻倍就塌; 不加位置编码的两种排布几乎不掉。iRoPE 那一行在这个玩具上没学会: 分块大小 16 时针和问话几乎总跨块, 只有最后一层能跨块取信息, 和 Llama 4 几十层、块长 8K 的真实配置不是一个量级。</p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { useDrag } from '@/composables/useDrag.js'

const TRAIN = 32                                  // 训练长度, 与 Python 脚本一致
// Python train 脚本的实测表 (seed 0, 4 层, 1200 步), 不随左侧控件变化
const measured = [
  { name: '全 RoPE 全注意力', a32: '1.000', a64: '0.105', a128: '0.062', keep: '0.06' },
  { name: '全 NoPE 全注意力', a32: '1.000', a64: '0.969', a128: '0.824', keep: '0.82' },
  { name: 'Kimi 式 (KDA + NoPE MLA)', a32: '0.996', a64: '0.969', a128: '0.973', keep: '0.98' },
  { name: 'iRoPE (分块 RoPE + NoPE)', a32: '0.086', a64: '0.094', a128: '0.141', keep: '1.64' },
]
const mode = ref('rope'), tExp = ref(7), chunk = ref(16), qpos = ref(127)
const T = computed(() => 1 << tExp.value)
const cw = computed(() => 400 / T.value)

// 可见性: 因果 + (分块模式下) 同一块内
const seen = (s) => s <= qpos.value && (mode.value !== 'chunk'
  || Math.floor(s / chunk.value) === Math.floor(qpos.value / chunk.value))
// ★ 没训练过的距离只存在于带 RoPE 的层: 相对距离超过训练长度
const beyond = (s) => mode.value !== 'nope' && qpos.value - s >= TRAIN
const state = (s) => (!seen(s) ? 'off' : beyond(s) ? 'unseen' : 'ok')

const visible = computed(() => {
  let n = 0
  for (let s = 0; s < T.value; s++) if (seen(s)) n++
  return n
})
// 这一层会遇到的最大相对距离: 全长是 T−1, 分块是 C−1, NoPE 没有位置输入
const maxDist = computed(() => (mode.value === 'chunk' ? Math.min(chunk.value, T.value) - 1 : T.value - 1))
const unseen = computed(() => {
  let n = 0
  for (let s = 0; s < T.value; s++) if (seen(s) && beyond(s)) n++
  return n
})
const { start } = useDrag((p) => { qpos.value = Math.max(0, Math.min(T.value - 1, Math.round(p.x / cw.value - 0.5))) })
</script>

<style scoped>
.keys { width: 100%; min-width: 400px; display: block; }
.key.off { fill: var(--border-strong); }
.key.ok { fill: var(--accent); }
.key.unseen { fill: var(--danger); }
.grip { fill: none; stroke: var(--text); stroke-width: 1.5; cursor: ew-resize; }
.grip:focus { outline: none; stroke: var(--eye); stroke-width: 2.5; }
.edge { stroke: var(--warn); stroke-width: 1.5; stroke-dasharray: 4 3; }
.lbl { font-size: 9px; fill: var(--text-dim); }
</style>
