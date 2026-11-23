<!--
  思考模式与思考预算实验台 (对应 llm_finetune/run_finetune/reasoning/train_reasoning.py)。
  只讲一件事: 写出来的每一步推理都是一份串行计算; 模型能在一次前向里 "默算" 的步数有限,
  所以思考预算砍掉的步数一旦超过这个上限, 正确率就掉回瞎猜。Python 里的 2 层模型默算能力是 0 步。
-->
<template>
  <LabFrame
    title="思考预算: 少写一步, 要在脑子里多算一步"
    sub="- 模拟什么: 推理模型的思考 / 非思考模式与思考预算。题目是 K 步模 10 算术链 (每步 +, −, × 一个数), 思考模式下模型每步写一行 op k 结果, 写完 </think> 再给答案; 非思考模式的思考块是空的; 预算 b 表示只写前 b 步, 剩下 K−b 步要在一次前向里默算。玩具里的模型能默算 c 步: 剩下的步数不超过 c 就答对, 超过就从 10 个数字里瞎猜 (答对概率 0.1)。Python 实验里的 2 层模型实测 c = 0。
- 图上是什么: 上面一行是当前这道题的 token 序列: 灰色是 prompt (含模式 token), 蓝色是写出来的推理步, 黄色是 </think>, 最后一格是答案, 绿色表示答对, 红色表示答错。下面是正确率与输出 token 数的取舍曲线, 横轴是输出 token 数 (3b + 3), 每个点是一档预算, 实心点是当前选中的那一档。
- 可以做什么: 两个模式按钮直接切到写满 (/think) 或不写 (/no_think); 点曲线上任意一个点, 或拖「预算 b」滑杆, 选一档预算; 「链长 K」改题目步数; 「默算步数 c」改模型一次前向能算几步; 换一道题。
- 应该看到: c = 0 时曲线是一个台阶, 只有写满才答对, 这正是 Python 实测的形状; c 调大后台阶左移, 同样的正确率只需要更少的 token。右侧读数:
  - 「输出 token」是当前预算下的回复长度, 「省下的 token」是与写满相比少写的比例。
  - 「期望正确率」是这一档在玩具模型下的正确率, 不低于写满时的 0.9 倍时变绿。
  - 「需要默算的步数」是 K−b, 不超过 c 时变绿。"
    module="llm_finetune/run_finetune/reasoning/train_reasoning.py"
    run="python -m llm_finetune.run_finetune.reasoning.train_reasoning"
    :challenge="{
      ask: 'K=4, c=0 时, 预算从 4 步降到 3 步, 正确率掉多少? 把 c 调到 2, 最少写几步还能保持满分? 这对真实推理模型的长度控制意味着什么?',
      answer: '结论: c=0 时只少写一步, 正确率就从 1 掉到 0.1; c=2 时写 2 步 (9 个 token) 就够, 比写满省 40% 的 token。\n- 每写一步, 模型就多一次前向去算下一个数; 不写的步数只能在回答那一次前向里默算。默算能力是模型的容量决定的, 预算控制改变不了它。\n- Python 里 2 层模型的实测就是 c=0 的台阶: 预算 token 模型 b=0…3 的正确率 0.13–0.17, b=4 为 0.998; 把混训步数从 500 加到 3000, 台阶不变。\n- 真实的推理模型思考链里有大量重复检查和冗余步骤, 大模型的默算能力也强得多, 所以 L1 这类方法能把长度压下来而只掉一点分。长度能省多少, 取决于被省掉的部分是不是冗余的。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: b === K }" @click="b = K">/think 写满</button>
        <button type="button" :class="{ active: b === 0 }" @click="b = 0">/no_think 不写</button>
        <button type="button" @click="seed++">换一道题</button>
      </div>
      <LabSlider v-model="b" label="预算 b (写几步)" :min="0" :max="K" :step="1" />
      <LabSlider v-model="K" label="链长 K" :min="2" :max="6" :step="1" />
      <LabSlider v-model="c" label="默算步数 c" :min="0" :max="3" :step="1" />
    </template>

    <div class="seq" role="list" aria-label="当前题目的 token 序列">
      <span v-for="(t, i) in seq" :key="i" role="listitem" class="chip" :class="t.kind">{{ t.text }}</span>
    </div>
    <svg viewBox="0 0 560 200" role="group" aria-label="正确率与输出 token 数的取舍曲线">
      <line x1="50" x2="540" y1="170" y2="170" stroke="var(--border)" />
      <line x1="50" x2="50" y1="20" y2="170" stroke="var(--border)" />
      <text v-for="y in [0, 0.5, 1]" :key="'y' + y" x="44" :y="yOf(y) + 4" text-anchor="end" class="t">{{ y }}</text>
      <text x="295" y="196" text-anchor="middle" class="t">输出 token 数 (3b + 3)</text>
      <text x="14" y="16" class="t">正确率</text>
      <polyline :points="curve.map((p) => `${xOf(p.tokens)},${yOf(p.acc)}`).join(' ')" fill="none" stroke="var(--accent)" stroke-width="2" />
      <g v-for="p in curve" :key="p.b" role="button" tabindex="0" :aria-label="`预算 ${p.b} 步`" class="pt"
         @click="b = p.b" @keydown.enter="b = p.b">
        <circle :cx="xOf(p.tokens)" :cy="yOf(p.acc)" :r="p.b === b ? 7 : 5"
                :fill="p.b === b ? 'var(--accent)' : 'var(--bg-card)'" stroke="var(--accent)" stroke-width="2" />
        <text :x="xOf(p.tokens)" y="186" text-anchor="middle" class="t">{{ p.tokens }}</text>
      </g>
    </svg>

    <template #stats>
      <div class="kv"><span>输出 token</span><b>{{ cur.tokens }}</b></div>
      <div class="kv"><span>省下的 token (相对写满)</span><b>{{ ((1 - cur.tokens / (3 * K + 3)) * 100).toFixed(0) }}%</b></div>
      <div class="kv"><span>期望正确率</span><b :class="cur.acc >= 0.9 * curve[K].acc ? 'good' : 'bad'">{{ cur.acc.toFixed(2) }}</b></div>
      <div class="kv"><span>需要默算的步数 K−b / 能默算 c</span><b :class="K - b <= c ? 'good' : 'bad'">{{ K - b }} / {{ c }}</b></div>
      <div class="lab-note">
        <p>Python 实测 (K=4, 留出集 512 题, 贪心解码), 实测, 不随左侧变化:</p>
        <ul class="pts">
          <li><b>只答案 vs 长轨迹 SFT:</b> 正确率 0.158 vs 0.996, 输出 3 vs 15 个 token。</li>
          <li><b>模式融合:</b> 同一个模型 /think 1.000 (15 token), /no_think 0.156 (3 token)。</li>
          <li><b>预算 token b=0…4:</b> 0.160, 0.131, 0.145, 0.166, 0.998; 强制截断 b=0…3 为 0.07–0.11。</li>
        </ul>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { mulberry32, range } from '@/utils/labmath.js'

const OPS = ['+', '−', '×']
const K = ref(4), b = ref(4), c = ref(0), seed = ref(1)
watch(K, (k) => { b.value = Math.min(b.value, k) })

// ★ 玩具模型: 剩下要默算的步数 K−b 不超过 c 就答对, 否则在 10 个数字里瞎猜
const accOf = (bb) => (K.value - bb <= c.value ? 1 : 0.1)
const curve = computed(() => range(K.value + 1).map((bb) => ({ b: bb, tokens: 3 * bb + 3, acc: accOf(bb) })))
const cur = computed(() => curve.value[b.value])

const problem = computed(() => {
  const rand = mulberry32(seed.value * 7919)
  const a0 = Math.floor(rand() * 10)
  const steps = range(K.value).map(() => ({ op: Math.floor(rand() * 3), k: 1 + Math.floor(rand() * 9) }))
  let v = a0
  for (const s of steps) { v = [(v + s.k) % 10, (v - s.k + 10) % 10, (v * s.k) % 10][s.op]; s.v = v }
  return { a0, steps, ans: v, guess: Math.floor(rand() * 10) }
})

const seq = computed(() => {
  const { a0, steps, ans, guess } = problem.value
  const mode = b.value === K.value ? '/think' : b.value === 0 ? '/no_think' : `B${b.value}`
  const out = [{ text: a0, kind: 'prompt' }, ...steps.map((s) => ({ text: `${OPS[s.op]}${s.k}`, kind: 'prompt' })),
    { text: mode, kind: 'prompt' }, { text: 'SEP', kind: 'prompt' },
    ...steps.slice(0, b.value).map((s) => ({ text: `${OPS[s.op]}${s.k}=${s.v}`, kind: 'step' })), { text: '</think>', kind: 'end' }]
  const said = K.value - b.value <= c.value ? ans : guess
  return [...out, { text: said, kind: said === ans ? 'ok' : 'bad' }, { text: 'EOS', kind: 'prompt' }]
})

const xOf = (tokens) => 70 + (tokens - 3) / (3 * K.value) * 450
const yOf = (acc) => 170 - acc * 140
</script>

<style scoped>
.seq { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 8px; }
.chip { font-family: "SF Mono", Menlo, monospace; font-size: 12px; padding: 2px 6px; border-radius: 4px;
        border: 1px solid var(--border); color: var(--text-muted); }
.chip.step { border-color: var(--accent); color: var(--accent); }
.chip.end { border-color: var(--warn); color: var(--warn); }
.chip.ok { border-color: var(--left); color: var(--left); font-weight: 600; }
.chip.bad { border-color: var(--danger); color: var(--danger); font-weight: 600; }
.t { font-size: 11px; fill: var(--text-muted); font-family: "SF Mono", Menlo, monospace; }
.pt { cursor: pointer; }
</style>
