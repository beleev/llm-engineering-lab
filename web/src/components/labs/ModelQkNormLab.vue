<!--
  QK-Norm 实验台 (对应 llm_models/layers/core/attention.py:GroupedQueryAttention 的 qk_norm 开关)。
  只讲一件事: logit = q·k/√d 随 ‖q‖‖k‖ 二次增长, softmax 会塌成 one-hot、梯度消失; 对 q/k 各做一次 RMSNorm 就把 logit 钉死在 ±g²√d 以内。
-->
<template>
  <LabFrame
    title="QK-Norm — 把注意力的打分关进笼子里"
    sub="- 模拟什么: attention 给每一对 (query q, key k) 打的分叫 logit, 等于 $q \cdot k / \sqrt{d}$ (点积除以头维度的平方根), 再经 softmax 变成权重。训练中 q、k 的范数 (向量的长度) 会慢慢长大, 点积随之按平方长: 两边各放大 2 倍, logit 放大 4 倍。logit 一大, softmax 就把几乎全部权重压到一个 key 上 (塌成 one-hot), 这个 head 的梯度就传不过去了, 低精度下 logit 还会溢出。QK-Norm 的办法是在算点积之前给 q、k 各做一次 RMSNorm (除以自身的均方根, 再乘一个可学的增益 g), 让 logit 的上界只由 g 决定。对应 llm_models/layers/core/attention.py 的 qk_norm 开关。
      - 图上是什么: 左右两块:
        - 左边: 8×8 的格子是一个 head 的因果注意力图。行是 query, 列是 key, 颜色越深权重越大; 右上半边是未来位置, 被 mask 掉所以画得很淡。
        - 右边: 选中那个 query 对它能看见的每个 key, 上方数字是 logit, 柱子是 softmax 之后的概率。
      - 可以做什么: 按界面顺序:
        - 「无 QK-Norm / QK-Norm」切开关。
        - 「换一组 q / k」: 重抽随机向量。
        - 拖「q、k 范数倍率」: 模拟训练中范数长大的过程 (把 q、k 同时放大这么多倍)。
        - 「可学增益 g」: 开着 QK-Norm 时多出这一根滑杆, 就是 RMSNorm 后面乘的那个系数。
        - 点左图最左边的 q0 到 q7: 换选中的 query。
      - 应该看到: 先关掉 QK-Norm 拖倍率, 再打开它重复一次:
        - 关掉 QK-Norm, 倍率从 1 拖到 4: 右侧「最大 |logit|」从约 2 涨到约 31 (16 倍), 注意力图从每行好几格有颜色变成每行只剩一格最深, 右边的柱子只剩一根。
        - 「注意力熵 / 最大熵」: 选中 query 的权重分布有多散, 熵接近 0 就是全押一个 key。
        - 「梯度通过率 $1 - \sum p^2$」: softmax 反向时能把多少梯度传回 logit, 接近 0 说明饱和, 这个 head 学不动了。
        - 打开 QK-Norm 再拖同一根倍率滑杆: 什么都不变, 因为归一化把倍率整个除掉了。
        - 「理论上界 $g^2 \sqrt{d}$」: logit 最多能有多大 (这里 d = 16, g = 1 时上界 4.0), 只有拖 g 才能改变尖锐度。"
    module="llm_models/layers/core/attention.py"
    run="python -m llm_models.run_models.foundation.attention.train_attention"
    :challenge="{
      ask: '关掉 QK-Norm, 把范数倍率从 1 拖到 4。先猜: 最大 logit 变成几倍? softmax 的「梯度通过率」会怎样?',
      answer: '最大 logit 变成 16 倍 (约 2 涨到约 31), 梯度通过率从 0.79 掉到 0.008。\n- q、k 各放大 4 倍, 点积放大 4 × 4 = 16 倍, 即「二次增长」。logit 之间的差距一拉大, softmax 就塌成 one-hot: 熵趋向 0, 通过率 $1 - \\sum p^2$ 趋向 0。\n- 梯度为什么传不过去: softmax 的雅可比 (输出对输入的导数矩阵) 是 $\\mathrm{diag}(p) - pp^\\top$, p 是 one-hot 时它每一项都是 0, 上游梯度乘上去全是 0, 这个 head 的 q/k 投影就收不到任何更新。\n- 溢出: logit 还会超出 bf16/fp16 能表示的范围。\nQK-Norm 把 q、k 归一到 RMS = 1 再乘可学增益 $g$, 于是 $\\text{logit} = g^2 \\cdot \\sqrt{d} \\cdot \\cos\\theta$, 上界 $g^2\\sqrt{d}$, 与范数倍率完全无关。模型仍然可以通过学 $g$ 来调「尖锐度」, 只是只剩这一个受控的旋钮。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: !qkNorm }" @click="qkNorm = false">无 QK-Norm</button>
        <button type="button" :class="{ active: qkNorm }" @click="qkNorm = true">QK-Norm (q、k 各过一次 RMSNorm)</button>
        <button type="button" @click="seed++">换一组 q / k</button>
      </div>
      <LabSlider v-model="scale" label="q、k 范数倍率 (模拟训练中范数长大)" :min="0.5" :max="6" :step="0.1" unit="×" :format="(t) => t.toFixed(1)" />
      <LabSlider v-if="qkNorm" v-model="gain" label="可学增益 g (RMSNorm 之后乘的系数)" :min="0.5" :max="2" :step="0.05" :format="(t) => t.toFixed(2)" />
    </template>

    <div class="wrap">
      <div>
        <div class="cells" :style="{ gridTemplateColumns: `28px repeat(${T}, 26px)` }" role="group" aria-label="注意力矩阵, 点一行选择 query">
          <template v-for="(row, i) in probs" :key="i">
            <button type="button" class="rl mono" :class="{ active: i === qi }" :aria-pressed="i === qi" @click="qi = i">q{{ i }}</button>
            <span v-for="(p, j) in row" :key="j" class="cell" :class="{ rowsel: i === qi }" :style="j <= i ? { background: heat(p) } : { opacity: 0.15 }" :title="j <= i ? `q${i} 给 k${j} 的注意力概率 p=${p.toFixed(3)}` : '未来位置, 被因果 mask 屏蔽'" />
          </template>
        </div>
        <p class="cap">行 = query (在看的位置), 列 = key (被看的位置), 颜色深浅 = 注意力概率</p>
      </div>
      <div class="bars">
        <div v-for="j in qi + 1" :key="j" class="b">
          <span class="mono lg" :class="{ hi: Math.abs(logits[qi][j - 1]) > 30 }">{{ logits[qi][j - 1].toFixed(1) }}</span>
          <div class="col"><div :style="{ height: probs[qi][j - 1] * 100 + '%' }" /></div>
          <span class="mono">k{{ j - 1 }}</span>
        </div>
        <p class="cap">q{{ qi }} 对它能看见的每个 key: 上方数字 = logit (打的分), 柱高 = softmax 之后的概率</p>
      </div>
    </div>

    <template #stats>
      <div class="kv"><span>最大 |logit| (整张图里打得最重的分)</span><b :class="maxLogit > 30 ? 'bad' : 'good'">{{ maxLogit.toFixed(1) }}</b></div>
      <div class="kv"><span><Tex text="logit 的理论上界 $g^2 \cdot \sqrt{d}$" /></span><b>{{ qkNorm ? (gain * gain * Math.sqrt(D)).toFixed(1) : '无上界 (随倍率的平方长)' }}</b></div>
      <div class="kv"><span>q{{ qi }} 的注意力熵 / 均匀时的最大熵</span><b :class="H / Hmax < 0.15 && qi > 0 ? 'bad' : ''">{{ H.toFixed(2) }} / {{ Hmax.toFixed(2) }}</b></div>
      <div class="kv"><span><Tex text="梯度通过率 $1 - \sum p^2$ (接近 0 = 饱和)" /></span><b :class="pass < 0.05 && qi > 0 ? 'bad' : 'good'">{{ pass.toFixed(3) }}</b></div>
      <p class="lab-note">
        <Tex :text="qkNorm ? '开着 QK-Norm 时拖范数倍率, 图完全不动: 归一化把倍率整个除掉了, 只剩 q、k 的夹角 ($\\cos\\theta$) 和增益 $g$ 决定注意力。拖 $g$ 才能看到图变尖, 而且最大 logit 永远不会超过上界。'
          : '倍率变 2 倍, logit 变 4 倍, 因为 q 和 k 两边都放大了。通过率接近 0 意味着 softmax 饱和: 反向时 $dL/d\\,\\text{logit} \\approx 0$, 这个 head 的 q/k 投影收不到梯度, 学不动了。'" />
      </p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import Tex from '@/components/Tex.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { mulberry32, randn, softmax, entropy, range, heat, sum } from '@/utils/labmath.js'

const T = 8, D = 16
const qkNorm = ref(false), scale = ref(1), gain = ref(1), seed = ref(3), qi = ref(5)

const vecs = computed(() => {
  const r = mulberry32(seed.value)
  const mk = () => range(T).map(() => range(D).map(() => randn(r)))
  return { q: mk(), k: mk() }
})
const rms = (v) => Math.sqrt(sum(v.map((t) => t * t)) / v.length)
const dot = (a, b) => sum(a.map((t, i) => t * b[i]))

const logits = computed(() => {
  // ★ 无 QK-Norm: 范数倍率原样进点积 (二次); 有: RMSNorm 先把它除掉, 再乘增益 g
  const prep = (v) => (qkNorm.value ? v.map((t) => (t / rms(v)) * gain.value) : v.map((t) => t * scale.value))
  const q = vecs.value.q.map(prep), k = vecs.value.k.map(prep)
  return q.map((qv, i) => k.map((kv, j) => (j <= i ? dot(qv, kv) / Math.sqrt(D) : -Infinity)))
})
const probs = computed(() => logits.value.map((row) => softmax(row)))
const maxLogit = computed(() => Math.max(...logits.value.flat().filter(Number.isFinite).map(Math.abs)))
const H = computed(() => entropy(probs.value[qi.value]))
const Hmax = computed(() => Math.log(qi.value + 1))
const pass = computed(() => 1 - sum(probs.value[qi.value].map((p) => p * p)))
</script>

<style scoped>
.wrap { display: flex; flex-wrap: wrap; gap: 20px; align-items: flex-start; }
.cells { gap: 2px; }
.cell { min-width: 0; width: 26px; height: 22px; }
.cell.rowsel { border-color: var(--accent); }
.rl { padding: 0; min-height: 22px; height: 22px; font-size: 10px; border-radius: 3px; }
.cap { font-size: 11px; color: var(--text-dim); margin-top: 6px; flex-basis: 100%; }
.bars { display: flex; flex-wrap: wrap; gap: 4px; flex: 1; min-width: 220px; }
.b { display: grid; justify-items: center; gap: 3px; font-size: 10px; color: var(--text-muted); width: 30px; }
.col { width: 20px; height: 110px; display: flex; align-items: flex-end; border-bottom: 1px solid var(--border-strong); }
.col div { width: 100%; background: var(--accent); border-radius: 2px 2px 0 0; min-height: 1px; }
.lg.hi { color: var(--danger); }
</style>
