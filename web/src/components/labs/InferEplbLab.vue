<!-- EPLB 实验台 (llm_infer/m21): 路由倾斜 → rank 负载不均 → 冗余专家。放置算法与 moe.py:eplb_placement 相同。 -->
<template>
  <LabFrame
    title="MoE 专家并行 — 热点专家把一张卡压满, EPLB 用冗余副本把它拆开"
    sub="- 模拟什么: 混合专家模型 (MoE) 的单层包含 16 个路由专家 (各自为独立的前馈子网络), 每个输入 token 经门控路由器激活得分最高的前 2 个专家参与计算 (top-2 路由)。专家并行 (EP) 将这 16 个专家分布到 4 张设备卡 (rank) 上: 前向时需先通过全对全通信将各 token 分发至目标专家所在卡 (dispatch), 计算完成后再汇总传回原卡 (combine)。由于 combine 存在跨卡同步屏障, 整层的实际耗时完全受最繁忙的那张瓶颈卡制约。当工作负载存在路由偏斜时, 热门专家所在卡严重过载而其余卡处于等待空闲。本实验模拟单批 2048 个 token, 采用叠加 Zipf 先验偏置的随机门控 logits 模拟负载倾斜 (编号靠前的专家先验激活概率更高)。EPLB (专家并行负载均衡器) 通过三步消除木桶效应:
  1. 为高频激活的热门专家分配多个冗余物理槽位 (冗余副本 slot)。
  2. 将发往该逻辑专家的 token 流量均摊至其各个物理副本。
  3. 将全部物理 slot 按预计负载从高到低排序, 依次贪心放置到当前累计负载最低的设备卡上。
- 图上是什么: 四根堆叠柱分别代表四张设备卡 (rank 0 至 rank 3):
  - 柱内堆叠色块: 该卡承载的各个物理专家 slot, 色块高度正比于其分担的 token 数量, 内部标注逻辑专家编号 (如 E3 ×2 表示专家 E3 共部署了 2 个物理副本), 带虚线边框的色块标识拥有多副本的热门专家。
  - 柱底标注: 当前设备卡承受的总 token 负载及其相对全卡均值的负载倍率, 全场负载最高的瓶颈卡标红显示。
  - 水平基准虚线: 理想均衡状态下的单卡平均负载 ($2048 \times 2 \div 4 = 1024$)。
- 可以做什么: 按面板从上至下支持以下交互调节:
  - 点击「连续放置」(将 E0 至 E3 固定放于 rank 0、E4 至 E7 放于 rank 1, 依此类推且每个专家仅占 1 个物理 slot) 或「EPLB 贪心放置」切换专家部署策略。
  - 点击「换一组 token」重新采样一批路由激活分布。
  - 拖动「路由倾斜度 s」滑杆调节门控分布的偏斜程度 (0 对应均匀路由, 数值越大流量越向头部少数专家集中)。
  - 拖动「冗余副本数」滑杆调节 EPLB 额外增配的物理副本槽位数 (在连续放置模式下不生效, 滑杆自动置灰)。
  - 点击或悬停任意专家色块, 跨卡高亮该逻辑专家的全部物理副本并在右侧展示其流量明细。
- 应该看到: 当路由倾斜度设为 1 时, 连续放置下聚集了热门专家 E0 与 E1 的 rank 0 负载远高于其余各卡; 切换至 EPLB 但将副本数设为 0 时, 虽经重排但四柱依然无法对齐, 因为最热的单个逻辑专家自身流量已超出单卡平均负载基准; 增配 4 个冗余副本后, 热门专家被切分摊薄, 四张卡的柱高趋于齐平。右侧四项读数含义如下:
  - max / mean rank 负载: 最忙瓶颈卡负载与全卡平均负载的比值, 单层 EP 步耗时正比于该倍率, 越接近 1 越均衡 (低于 1.1 标绿, 高于 1.5 标红)。
  - 平均利用率: 上述负载倍率的倒数, 反映全集群算力的有效平均繁忙度。
  - 最热专家 ÷ 平均 rank 负载: 单个最繁忙逻辑专家的流量与单卡理想均值之比; 一旦大于 1, 说明仅靠跨卡挪动位置绝无可能实现完全均衡, 必须引入副本拆分。
  - 物理 slot 数: 基础专家数 16 加上额外配置的冗余副本总数。"
    module="llm_infer/m21"
    run="python -m llm_infer.m21_moe_serving.demo"
    :challenge="{
      ask: '当路由倾斜度 $s = 1$ 时, 仅开启「EPLB 贪心放置」但将冗余副本数设为 0, 能否将 max/mean 负载倍率压低至 1.1 以下? 请结合右侧「最热专家 ÷ 平均 rank 负载」读数加以分析。',
      answer: '无法做到。当路由倾斜度 $s = 1$ 时, 最热门的单个专家 E0 独自接收的 token 数量已达单卡理想平均负载的约 1.4 倍 (右侧「最热专家 ÷ 平均 rank 负载」读数约为 1.4)。在不允许拆分副本的前提下, 单个逻辑专家是不可分割的最小放置单元: 无论贪心算法将 E0 安置到哪一张设备卡上, 该卡仅凭这一个专家就至少超载 40%, 因此全卡 max/mean 倍率的理论下界不可能低于 1.4 (本节 demo 脚本实测纯贪心重排结果为 1.48×)。\n单纯调整放置映射无法化解「单个热点专家流量超过单卡平均容量」的结构性失衡; 唯有通过 EPLB 为热点专家创建多个冗余物理 slot 并将发往它的 token 分流摊薄, 才能突破单专家粒度上限。\n由于同一逻辑专家的各个物理副本持有完全相同的权重参数, 任意 token 被分发至哪一个副本执行前向所得结果均严格一致, 模型最终输出逐位不变。引入冗余副本的系统开销主要体现在两方面:\n- 每张设备卡需额外预留若干个专家权重 slot 的静态显存空间。\n- 动态调整部署拓扑时需跨卡搬运专家权重张量, 因此生产系统通常每隔数分钟才根据滑动窗口历史统计重排一次。',
    }"
  >
    <template #controls>
      <div class="row">
        <button v-for="m in MODES" :key="m.id" type="button" :class="{ active: mode === m.id }" @click="mode = m.id">{{ m.name }}</button>
        <button type="button" @click="seed++">换一组 token</button>
      </div>
      <LabSlider v-model="skew" label="路由倾斜度 s (Zipf)" :min="0" :max="2" :step="0.1" />
      <!-- 连续放置下每个专家固定 1 个 slot, 副本数不起作用: 置灰, 不让拖 -->
      <div :inert="mode === 'contig' || null" :class="{ off: mode === 'contig' }">
        <LabSlider v-model="redIdx" label="冗余副本数 (EPLB)" :min="0" :max="2" :format="(v) => RED[v]" />
      </div>
      <p v-if="mode === 'contig'" class="tip">连续放置不复制专家, 副本数滑杆不起作用。先切到「EPLB 贪心放置」。</p>
    </template>

    <svg :viewBox="`0 0 ${W} ${H + 30}`" role="group" aria-label="每张卡上的专家 slot 负载">
      <g v-for="(col, r) in view.cols" :key="r">
        <g
          v-for="s in col.slots" :key="s.slot" role="button" tabindex="0" :aria-label="`专家 ${s.e}, ${s.load} token`"
          class="slot" @click="sel = s.e" @keydown.enter="sel = s.e" @mouseenter="sel = s.e"
        >
          <rect
            :x="r * CW + 14" :y="H - (s.y0 + s.load) * view.sy" :width="CW - 28" :height="Math.max(0.5, s.load * view.sy)"
            :fill="fill(s.e)" :class="{ dim: sel >= 0 && sel !== s.e, rep: s.nRep > 1 }"
          />
          <text v-if="s.load * view.sy > 11" :x="r * CW + CW / 2" :y="H - (s.y0 + s.load / 2) * view.sy + 4" text-anchor="middle" class="lbl in">
            E{{ s.e }}{{ s.nRep > 1 ? ` ×${s.nRep}` : '' }}
          </text>
        </g>
        <text :x="r * CW + CW / 2" :y="H + 14" text-anchor="middle" class="lbl">rank {{ r }} · {{ col.load }}</text>
        <text :x="r * CW + CW / 2" :y="H + 26" text-anchor="middle" class="lbl" :class="{ bad: col.load === view.max }">{{ (col.load / view.mean).toFixed(2) }}×</text>
      </g>
      <line x1="0" :x2="W" :y1="H - view.mean * view.sy" :y2="H - view.mean * view.sy" class="mean" />
      <text :x="W - 2" :y="H - view.mean * view.sy - 4" text-anchor="end" class="lbl">平均 {{ view.mean }}</text>
    </svg>

    <template #stats>
      <div class="kv"><span>max / mean rank 负载 (最忙的卡 ÷ 平均, 一步耗时正比于它)</span><b :class="view.ratio > 1.5 ? 'bad' : view.ratio < 1.1 ? 'good' : ''">{{ view.ratio.toFixed(2) }}×</b></div>
      <div class="kv"><span>平均利用率 (mean/max)</span><b :class="view.ratio > 1.5 ? 'bad' : view.ratio < 1.1 ? 'good' : ''">{{ (100 / view.ratio).toFixed(0) }}%</b></div>
      <div class="kv"><span>最热专家 ÷ 平均 rank 负载 (大于 1 就必须复制)</span><b :class="view.hot > 1 ? 'bad' : ''">{{ view.hot.toFixed(2) }}</b></div>
      <div class="kv"><span>物理 slot 数 (16 个专家 + 副本)</span><b>{{ view.nSlot }}</b></div>
      <p class="lab-note">
        <template v-if="sel >= 0">专家 E{{ sel }}: 这一批收到 {{ counts[sel] }} 个 token, 有 {{ view.nRep[sel] }} 个副本, 每个副本分到约 {{ Math.round(counts[sel] / view.nRep[sel]) }} 个。</template>
        <template v-else>点或悬停柱子里的一段, 高亮这个专家的所有副本。</template>
      </p>
      <div class="lab-note">
        <p>性能评估关注峰值负载 max 而非均值 mean, 原因在于专家并行采用同步集合通信: 每层 combine 阶段必须等待所有 rank 完成本地专家计算后才能汇合返回, 负载较轻的设备卡只能停顿等待负载最高的那张瓶颈卡。</p>
        <p>与 Python demo 一致, 本实验依据前一批历史 token 的激活统计规划专家放置方案, 并在新到达的独立批次上评测实际负载倍率: 在真实推理系统中, 变更放置方案涉及跨卡权重迁移, 必须在当前批次开始执行前预先固化部署拓扑。</p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { mulberry32, randn, range, argmax, sum } from '@/utils/labmath.js'

const E = 16, R = 4, K = 2, T = 2048, W = 480, H = 240, CW = W / R
const RED = [0, 4, 8]
const MODES = [{ id: 'contig', name: '连续放置 (E0–3 在 rank 0 …)' }, { id: 'eplb', name: 'EPLB 贪心放置' }]
const PALETTE = ['var(--accent)', 'var(--left)', 'var(--eye)', 'var(--right)', 'var(--warn)', 'var(--danger)']
const fill = (e) => `color-mix(in srgb, ${PALETTE[e % 6]} ${e < 6 ? 75 : e < 12 ? 50 : 30}%, transparent)`

const mode = ref('contig')
const skew = ref(1)
const redIdx = ref(1)
const seed = ref(1)
const sel = ref(-1)

// 路由 logits 的噪声部分 ~ N(0,1), 只跟 seed 走; 两批: [0] 历史 (用来定放置), [1] 新 batch (用来评测)
const noise = computed(() => {
  const rand = mulberry32(seed.value * 104729)
  return range(2).map(() => Float32Array.from({ length: T * E }, () => randn(rand)))
})
const route = (z) => {
  const bias = range(E).map((e) => -skew.value * Math.log(e + 1)) // 专家 e 的先验 ∝ (e+1)^-s, 同 make_layer
  const cnt = Array(E).fill(0)
  for (let t = 0; t < T; t++) {
    const l = range(E).map((e) => z[t * E + e] + bias[e])
    for (let k = 0; k < K; k++) { const e = argmax(l); cnt[e]++; l[e] = -Infinity }
  }
  return cnt
}
const hist = computed(() => route(noise.value[0]))
const counts = computed(() => route(noise.value[1]))

// 与 moe.py:eplb_placement 相同的两步贪心
const placement = computed(() => {
  if (mode.value === 'contig') return { slotExpert: range(E), slotRank: range(E).map((e) => Math.floor(e / (E / R))), nRep: Array(E).fill(1) }
  const load = hist.value, nRep = Array(E).fill(1)
  // ★ ① 副本数: 反复给「每副本负载」最大的专家加一个副本
  for (let i = 0; i < RED[redIdx.value]; i++) nRep[argmax(load.map((l, e) => l / nRep[e]))]++
  const slotExpert = range(E).flatMap((e) => Array(nRep[e]).fill(e))
  const slotLoad = slotExpert.map((e) => load[e] / nRep[e])
  const cap = slotExpert.length / R, rankLoad = Array(R).fill(0), rankFree = Array(R).fill(cap), slotRank = []
  // ② 放置: slot 从重到轻, 依次放到当前最轻且还有空位的 rank
  for (const s of range(slotExpert.length).sort((a, b) => slotLoad[b] - slotLoad[a] || a - b)) {
    const r = argmax(rankLoad.map((l, i) => (rankFree[i] > 0 ? -l : -Infinity)))
    slotRank[s] = r; rankLoad[r] += slotLoad[s]; rankFree[r]--
  }
  return { slotExpert, slotRank, nRep }
})

const view = computed(() => {
  const { slotExpert, slotRank, nRep } = placement.value
  const seen = Array(E).fill(0)
  const cols = range(R).map(() => ({ load: 0, slots: [] }))
  slotExpert.forEach((e, slot) => {
    const j = seen[e]++, c = counts.value[e]
    const load = Math.floor(c / nRep[e]) + (j < c % nRep[e] ? 1 : 0) // 副本间 round-robin 拆 token, 同 ep_forward 的 slot_of
    const col = cols[slotRank[slot]]
    col.slots.push({ slot, e, load, y0: col.load, nRep: nRep[e] })
    col.load += load
  })
  const mean = (T * K) / R, max = Math.max(...cols.map((c) => c.load))
  return { cols, nRep, mean, max, ratio: max / mean, hot: Math.max(...counts.value) / mean, sy: (H - 10) / max, nSlot: slotExpert.length, total: sum(counts.value) }
})
</script>

<style scoped>
.off { opacity: 0.45; }
.tip { font-size: 11px; color: var(--text-dim); }
.slot { cursor: pointer; outline: none; }
.slot rect { stroke: var(--bg-card); stroke-width: 1; }
.slot rect.rep { stroke: var(--text); stroke-dasharray: 3 2; }
.slot rect.dim { opacity: 0.25; }
.slot:focus-visible rect { stroke: var(--accent); stroke-width: 2; }
.mean { stroke: var(--text-muted); stroke-dasharray: 5 4; }
.lbl { font-size: 10px; fill: var(--text-muted); }
.lbl.in { fill: var(--text); pointer-events: none; }
.lbl.bad { fill: var(--danger); }
</style>
