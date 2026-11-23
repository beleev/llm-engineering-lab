<!--
  MoE 路由实验台 (对应 llm_train/m11_expert_parallel/demo.py)。
  只讲一件事: router 选谁决定了每个专家收多少 token; 超过 capacity 的被丢弃, 最热的卡决定 step 时间。
  均衡用 demo 里 balance_bias 的办法: 选专家时加一个 bias, 过载减、欠载加。
  与 demo 的差别: 步长逐步衰减并在达到均衡时停下。固定步长在 64 个 token 上会绕着均衡点来回跳。
-->
<template>
  <LabFrame
    title="MoE 专家并行路由 — 路由一偏, 哪张卡最忙、哪些 token 被丢"
    sub="- 模拟什么: 混合专家模型 (MoE) 层的专家并行 (EP) 执行过程: 将 E 个专家均匀分布于 D 张计算卡上, 输入批次中的 64 个 token 分别由路由器 (router) 打分并分发给得分最高的 1 个专家 (即 top-1 路由)。由于 token 原本所在的计算卡与其目标专家所在的计算卡往往不一致, 需要执行一次 all-to-all 跨卡通信将各 token 发送至目标卡。每个专家设有处理容量上限 $\text{capacity} = \lceil \text{cf} \times 64 / E \rceil$, 超出容量的溢出 token 将跳过专家计算、仅沿残差分支原样通过 (即发生 token 丢弃), 而全环的同步等待时间由负载最重的那张卡决定。「bias 均衡」对应 DeepSeek-V3 提出的无辅助损失 (aux-loss-free) 负载均衡方案, 仅调整专家选择顺序而不改动乘在专家输出上的门控权重, 具体做法如下:
  1. 为每个专家维护一个可动态调节的路由偏置项, 仅在选取 top-k 专家时叠加到路由得分上。
  2. 每步统计负载后, 对过载专家调低偏置、对欠载专家调高偏置。
  3. 经数百步迭代自适应收敛至各专家负载齐平。
- 图上是什么: 左侧为各专家负载柱状图, 右侧为跨卡通信矩阵:
  - 左侧柱图: 每根柱子代表一个专家 (从 e0 起编号), 柱高表示分配给该专家的 token 数量, 柱体颜色标识该专家驻留的计算卡编号 (图例见右下角)。
  - 黄色水平虚线: 标示单专家容量上限 capacity; 柱体超出虚线的红色区段即为因容量溢出而被丢弃的 token。
  - 柱图底部说明: 点击任意一根专家柱, 此处显示该专家的负载与容量明细。
  - 右侧 all-to-all 发送矩阵: 第 s 行第 t 列的数值表示从卡 s 发送至卡 t 的 token 数量, 颜色越深代表通信量越大, 整张矩阵即构成当步 all-to-all 通信的收发明细表。
- 可以做什么: 按界面控件顺序操作:
  - 「D = 2 卡 / 4 卡」: 切换承载专家的设备卡数。
  - 均衡开关「bias 均衡: 开 / 关」: 切换是否启用动态路由偏置均衡。
  - 「换一组 token」: 重新随机生成一批路由打分输入。
  - 拖动「专家数 E」: 在 4 至 16 之间调节总专家数量。
  - 拖动「容量因子 cf」: 在 1 至 2 之间调节单专家的冗余容量倍率。
  - 拖动「路由倾斜度 skew」: 取 0 代表无偏均匀打分, 数值越大路由器越偏向低编号专家, 用于模拟训练中常见的路由坍缩现象。
- 应该看到: 调大 skew 时, e0、e1 等头部专家的负载柱显著超出 capacity 虚线并出现长段红色溢出, 右侧发送矩阵中对应设备卡所在的接收列颜色明显加深; 开启「bias 均衡」后, 各专家负载柱恢复齐平, 红色溢出段完全消失。右侧读数含义如下:
  - 「不均衡度 max/mean」: 最繁忙专家接收的 token 数与理论均值 $64 / E$ 的比值, 1.00x 代表完全均衡, 示例脚本中倾斜路由器的实测值为 1.62x。
  - 「容量溢出丢弃的 token」: 全层所有专家红色溢出段丢弃的 token 总数。
  - 「最热的卡收到」: 发送矩阵中各列列和的最大值 (理想均衡时应为 $64 / D$), 直接决定本步跨卡通信与专家计算的墙钟耗时。"
    module="llm_train/m11"
    run="python -m llm_train.m11_expert_parallel.demo"
    :challenge="{
      ask: '默认配置下共有 4 个 token 因超出专家容量而被丢弃。请针对以下三项独立操作分别判断丢弃数是否会降至 0 (每次仅改动一个控件): 将 skew 拖至 0; 将容量因子 cf 拖至 2; 开启「bias 均衡」。',
      answer: '答案: 将 skew 调至 0 后仍会发生丢弃; 将 cf 调至 2 可使丢弃归零; 开启「bias 均衡」同样可使丢弃归零。\n原因分析:\n- $\\text{skew} = 0$: 仍有 2 个 token 被丢弃。路由器无倾向性时等价于均匀随机分发, 将 64 个 token 随机分配给 8 个专家, 虽然数学期望为每专家 8 个, 但由多项分布的随机涨落可知, 部分专家仍会收到 12 个 token, 超出容量上限 $\\text{capacity} = \\lceil 1.25 \\times 64 / 8 \\rceil = 10$。统计无偏并不等同于单批次严格均衡。\n- $\\text{cf} = 2$: 单专家容量上限扩大至 16, 溢出丢弃降至 0。但代价是各专家张量缓冲与矩阵计算均需按最坏上限 (每专家 16 个 token) 填充预留 (padding), 显存与算力开销变为 $\\text{cf} = 1$ 时的两倍。\n- 开启「bias 均衡」: 无论 skew 如何倾斜, 动态偏置均能将各专家负载精确调节至 $64 / E$ 个 ($E = 8$ 时恰好每专家 8 个), 溢出丢弃为 0, 负载最重的设备卡也恰好接收 $64 / D$ 个 token ($D = 4$ 时为 16 个)。由于偏置项仅参与 top-1 路由选择而不改动乘在专家输出上的 gate 权重, 不会引入干扰主任务的梯度。\n示例脚本实测结果 (4 卡、每卡 2 个专家、64 个 token、单专家容量 10):\n- 倾斜 router 基线: 不均衡度 1.62x、丢弃 5 个 token、最热卡接收 24 个 token。\n- 引入 aux loss 训练 60 步后: 不均衡度降至 1.12x、丢弃 0 个 token。\n- 采用 bias 均衡后: 不均衡度 1.00x、丢弃 0 个 token、最热卡接收 16 个 token。',
    }"
  >
    <template #controls>
      <div class="row">
        <button v-for="d in [2, 4]" :key="d" type="button" :class="{ active: D === d }" @click="D = d">D = {{ d }} 卡</button>
        <button type="button" :class="{ active: balanced }" :aria-pressed="balanced" @click="balanced = !balanced">
          bias 均衡 (aux-loss-free): {{ balanced ? '开' : '关' }}
        </button>
        <button type="button" @click="seed++">换一组 token</button>
      </div>
      <LabSlider v-model="E" label="专家数 E" :min="4" :max="16" :step="4" />
      <LabSlider v-model="cf" label="容量因子 cf" :min="1" :max="2" :step="0.25" :format="(v) => v.toFixed(2)" />
      <LabSlider v-model="skew" label="路由倾斜度 skew" :min="0" :max="3" :step="0.1" :format="(v) => v.toFixed(1)" />
    </template>

    <div class="chart-wrap">
      <div class="chart">
        <div class="cap-line" :style="{ bottom: hPct(capacity) }">
          <span class="mono">capacity = {{ capacity }}</span>
        </div>
        <button
          v-for="(c, e) in route.counts" :key="e" type="button" class="col" :class="{ sel: e === pick }"
          :aria-label="`专家 ${e}: 收到 ${c} 个 token`" @click="pick = e"
        >
          <span class="seg over" :style="{ height: hPct(Math.max(0, c - capacity)) }" />
          <span class="seg" :style="{ height: hPct(Math.min(c, capacity)), background: devColors[devOf(e)] }" />
        </button>
      </div>
      <div class="xlabs">
        <span v-for="e in E" :key="e" class="mono">e{{ e - 1 }}</span>
      </div>
    </div>
    <p class="lab-note" style="margin-top: 10px;">
      专家 e{{ sel.e }} 在卡 {{ sel.dev }}: 收到 {{ sel.c }} 个 token, 处理 {{ sel.c - sel.over }} 个, 丢弃 {{ sel.over }} 个 (被丢弃的 token 跳过专家, 只走残差连接原样往下传)。
    </p>

    <template #stats>
      <div class="kv"><span>不均衡度 max/mean (最忙专家 / 平均 {{ N / E }} 个, 1.00x 最均衡)</span><b :class="st.imb < 1.2 ? 'good' : 'bad'">{{ st.imb.toFixed(2) }}x</b></div>
      <div class="kv"><span>容量溢出丢弃的 token (所有红段之和)</span><b :class="st.dropped ? 'bad' : 'good'">{{ st.dropped }} / {{ N }}</b></div>
      <div class="kv"><span>最热的卡收到的 token (均匀应为 {{ N / D }}, 它决定这一步多久)</span><b :class="st.hot > st.hotOk ? 'bad' : 'good'">{{ st.hot }}</b></div>
      <div class="a2a">
        <span>all-to-all 发送矩阵 (第 s 行第 t 列 = 从卡 s 寄到卡 t 的 token 数)</span>
        <div class="a2a-grid" :style="{ gridTemplateColumns: `repeat(${D}, 36px)` }">
          <template v-for="(row, s) in dispatch" :key="s">
            <span v-for="(v, t) in row" :key="t" class="a2a-cell mono" :style="{ background: heat((v / st.cellMax) * 0.8) }">{{ v }}</span>
          </template>
        </div>
      </div>
      <div class="legend">
        <span v-for="d in D" :key="d" class="leg mono"><i :style="{ background: devColors[d - 1] }" />卡{{ d - 1 }}</span>
      </div>
      <div class="lab-note">
        <p>动态路由偏置 (bias 均衡) 是 DeepSeek-V3 采用的无辅助损失均衡机制:</p>
        <ol class="pts">
          <li>选专家时用 s_e + b_e 排序 (s_e 是路由分数, b_e 是偏置)。</li>
          <li>过载的专家 b_e 减一点, 欠载的加一点。</li>
          <li>重复几百步直到拉平。</li>
        </ol>
        <p>专家偏置项 b_e 不接入反向传播计算图, 也不参与专家输出加权的 gate 数值计算, 因此不会对语言建模主目标产生梯度干扰。</p>
        <p>另一种经典方案是 Switch Transformer 的辅助负载均衡损失 (aux loss): 在训练总损失中额外加入 $E \cdot \sum f_e \cdot P_e$ 正则项直接训练 router, 促使各专家的路由概率趋于均匀。该方案能够有效缓解倾斜 (示例脚本中不均衡度从 1.62x 降至 1.12x), 但辅助损失产生的梯度会与语言建模主损失存在优化方向冲突。</p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { heat, mulberry32, range, sum } from '@/utils/labmath.js'

const N = 64
const D = ref(4)
const E = ref(8)
const cf = ref(1.25)
const skew = ref(0.2)
const balanced = ref(false)
const seed = ref(1)
const pick = ref(0)
const devColors = ['var(--accent)', 'var(--left)', 'var(--eye)', 'var(--right)']
watch(E, (e) => { if (pick.value >= e) pick.value = 0 })

const devOf = (e) => Math.floor(e / (E.value / D.value))

// 路由打分: score_e = skew·lin_e + noise, lin 从 +1 线性递减到 -1 (专家 0 天然最热)
const scores = computed(() => range(N).map((n) => {
  const rand = mulberry32(seed.value * 7919 + n * 1000 + 7)
  return range(E.value).map((e) => skew.value * (1 - (2 * e) / (E.value - 1)) + (rand() - 0.5) * 2.5)
}))
const choose = (bias) => scores.value.map((row) => row.reduce((b, s, e) => (s + bias[e] > row[b] + bias[b] ? e : b), 0))
const load = (expertOf) => { const c = Array(E.value).fill(0); expertOf.forEach((e) => c[e]++); return c }

// ★ 同 demo 的 balance_bias: b_e += γ·sign(mean − load_e)。bias 只影响「选谁」
const bias = computed(() => {
  const b = Array(E.value).fill(0), mean = N / E.value
  if (!balanced.value) return b
  let best = b.slice(), bestMax = Infinity
  for (let i = 0, gamma = 0.05; i < 1000; i++, gamma *= 0.99) {
    const c = load(choose(b)), mx = Math.max(...c)
    if (mx < bestMax) { bestMax = mx; best = b.slice() }
    if (mx <= Math.ceil(mean)) break
    c.forEach((v, e) => { b[e] += gamma * Math.sign(mean - v) })
  }
  return best
})
const route = computed(() => { const expertOf = choose(bias.value); return { expertOf, counts: load(expertOf) } })

const capacity = computed(() => Math.ceil((N / E.value) * cf.value))
// 发送矩阵: token n 的源卡 = floor(n / (N/D)), 目标卡 = 其专家所在卡
const dispatch = computed(() => {
  const M = range(D.value).map(() => Array(D.value).fill(0))
  route.value.expertOf.forEach((e, n) => { M[Math.floor(n / (N / D.value))][devOf(e)] += 1 })
  return M
})
const st = computed(() => ({
  imb: Math.max(...route.value.counts) / (N / E.value),
  dropped: sum(route.value.counts.map((c) => Math.max(0, c - capacity.value))),
  hot: Math.max(...range(D.value).map((d) => sum(dispatch.value.map((r) => r[d])))),
  hotOk: Math.ceil(N / E.value) * (E.value / D.value),   // 每个专家都不超过 ⌈mean⌉ 时, 一张卡最多收这么多
  cellMax: Math.max(1, ...dispatch.value.flat()),
}))
const sel = computed(() => {
  const c = route.value.counts[pick.value]
  return { e: pick.value, dev: devOf(pick.value), c, over: Math.max(0, c - capacity.value) }
})

const scaleMax = computed(() => Math.max(capacity.value, ...route.value.counts) * 1.15)
const hPct = (v) => `${((v / scaleMax.value) * 100).toFixed(1)}%`
</script>

<style scoped>
.chart-wrap { background: var(--code-bg); border-radius: var(--radius-sm); padding: 10px; }
.chart { position: relative; height: 180px; display: flex; align-items: flex-end; gap: 4px; }
.col { flex: 1; display: flex; flex-direction: column; justify-content: flex-end; height: 100%; min-width: 0; min-height: 0; padding: 0; border: 0; border-radius: 0; background: none; cursor: pointer; }
.col.sel { box-shadow: inset 0 -2px 0 var(--text); }
.seg { display: block; border-radius: 2px 2px 0 0; }
.seg.over { background: var(--danger); }
.cap-line { position: absolute; left: 0; right: 0; border-top: 2px dashed var(--warn); z-index: 1; pointer-events: none; }
.cap-line span { position: absolute; right: 0; top: -16px; font-size: 10px; color: var(--warn); }
.xlabs { display: flex; gap: 4px; margin-top: 4px; }
.xlabs span { flex: 1; text-align: center; font-size: 10px; color: var(--text-dim); }
.a2a { display: flex; flex-direction: column; gap: 6px; }
.a2a-grid { display: grid; gap: 3px; }
.a2a-cell { height: 28px; display: flex; align-items: center; justify-content: center; font-size: 11px; color: var(--text); border: 1px solid var(--border); border-radius: 3px; }
.legend { display: flex; gap: 12px; flex-wrap: wrap; }
.leg { display: inline-flex; align-items: center; gap: 5px; font-size: 11px; color: var(--text-muted); }
.leg i { width: 10px; height: 10px; border-radius: 2px; }
</style>
