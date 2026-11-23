<!--
  QK-Norm 实验台 (对应 llm_models/layers/core/attention.py:GroupedQueryAttention 的 qk_norm 开关)。
  只讲一件事: logit = q·k/√d 随 ‖q‖‖k‖ 二次增长, softmax 会塌成 one-hot、梯度消失; 对 q/k 各做一次 RMSNorm 就把 logit 钉死在 ±g²√d 以内。
-->
<template>
  <LabFrame
    title="QK-Norm — 把注意力的打分关进笼子里"
    sub="- 模拟什么: 注意力层对每一对 $(q, k)$ 计算相关性得分 $\text{logit} = q \cdot k / \sqrt{d}$ (点积除以头维度的平方根), 再经 softmax 归一化为注意力权重。在深层大规模训练中, query 与 key 向量的范数往往会随训练逐步增长, 导致点积按平方倍率放大: 当 $q$ 与 $k$ 的范数各自放大 2 倍时, logit 会放大 4 倍。一旦 logit 幅值过大, softmax 输出就会坍缩为近似 one-hot 分布, 使该注意力 head 因饱和而丧失梯度反传能力, 在低精度浮点下还会引发数值溢出。QK-Norm 在计算点积前分别对 $q$ 和 $k$ 施加 RMSNorm (除以自身均方根后再乘以可学习缩放增益 $g$), 使 logit 的绝对值上界仅由 $g$ 决定。对应实现见 llm_models/layers/core/attention.py 中的 qk_norm 开关。
- 图上是什么: 视图分为左右两部分:
  - 左侧热力图: $8 \times 8$ 网格展示单个注意力 head 的因果注意力权重矩阵, 行为 query、列为 key, 颜色越深代表权重越大; 右上三角为未来位置, 受因果 mask 屏蔽而呈浅淡色。
  - 右侧柱图: 展示当前选中 query 对各可见 key 的计算明细, 柱顶数字为原始 logit, 柱高为经过 softmax 后的注意力概率。
- 可以做什么: 按界面控件顺序操作:
  - 「无 QK-Norm / QK-Norm」: 切换是否启用 QK-Norm。
  - 「换一组 q / k」: 重新采样随机向量。
  - 拖动「q、k 范数倍率」: 将 $q$ 与 $k$ 的范数同步放大指定倍数, 模拟训练过程中表示范数持续膨胀的现象。
  - 「可学增益 g」: 开启 QK-Norm 时显示该滑杆, 对应 RMSNorm 之后乘上的可学习缩放系数。
  - 点击左图左侧的 q0 至 q7: 切换右侧面板观察的目标 query。
- 应该看到: 先在关闭 QK-Norm 下拖动范数倍率, 再开启 QK-Norm 重复相同操作对比:
  - 关闭 QK-Norm 并将倍率从 1 拖至 4: 右侧「最大 |logit|」从约 2 升至约 31 (放大 16 倍), 左侧注意力图从每行分散关注多个位置退化为每行仅剩单格深色, 右侧概率柱也坍缩为单根独大。
  - 「注意力熵 / 最大熵」: 衡量当前选中 query 注意力分布的离散程度, 趋近 0 表示权重几乎全部集中在单一 key 上。
  - 「梯度通过率 $1 - \sum p^2$」: 衡量反向传播经过 softmax 时能够传回 logit 的梯度比例, 趋近 0 说明进入饱和区, 该 head 难以继续更新。
  - 开启 QK-Norm 后再拖动范数倍率滑杆: 各项读数与分布完全保持不变, 因为 RMSNorm 已将输入范数倍率彻底消去。
  - 「理论上界 $g^2 \sqrt{d}$」: 启用 QK-Norm 后 logit 的最大可能幅值 (此处 $d = 16$, 当 $g = 1$ 时上界为 4.0), 此时唯有调节 $g$ 才能受控地改变注意力分布的尖锐度。"
    module="llm_models/layers/core/attention.py"
    run="python -m llm_models.run_models.foundation.attention.train_attention"
    :challenge="{
      ask: '关闭 QK-Norm, 将 q、k 范数倍率从 1 拖至 4。请先思考: 最大 logit 幅值会变为原来的多少倍? softmax 的「梯度通过率」会发生怎样的变化?',
      answer: '答案: 最大 logit 变为原来的 16 倍 (从约 2 增至约 31), softmax 梯度通过率从 0.79 骤降至 0.008。\n- 二次放大效应: 当 $q$ 与 $k$ 各自放大 4 倍时, 其点积放大 $4 \\times 4 = 16$ 倍。logit 之间的分差随之剧烈拉大, 促使 softmax 输出坍缩为近似 one-hot 分布, 注意力熵与梯度通过率 $1 - \\sum p^2$ 双双趋近于 0。\n- 梯度消失机制: softmax 的雅可比矩阵为 $\\mathrm{diag}(p) - pp^\\top$; 当概率向量 $p$ 退化为 one-hot 时, 该矩阵所有元素均趋近于 0, 上游反传梯度左乘该矩阵后几乎全被截断, 导致该 head 的 Q/K 投影矩阵无法获得有效参数更新。\n- 数值溢出风险: 过大的 logit 还会直接超出 bf16 或 fp16 的有限动态范围。\n启用 QK-Norm 后, $q$ 与 $k$ 先被归一化至均方根为 1 再乘以可学习增益 $g$, 此时 $\\text{logit} = g^2 \\cdot \\sqrt{d} \\cdot \\cos\\theta$, 严格受限于理论上界 $g^2\\sqrt{d}$ 而与输入向量的范数倍率无关; 模型仍可通过训练标量参数 $g$ 自主调节注意力尖锐度, 且尺度始终处于受控范围内。',
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
