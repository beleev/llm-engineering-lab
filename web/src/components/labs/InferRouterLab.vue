<!--
  多副本路由实验台 (对应 llm_infer/m27_multi_replica_routing/router.py:route / simulate)。
  一件事: 送到有前缀的副本能省 prefill, 但热点会把一个副本压垮 —— 负载阈值在两者之间取舍。
-->
<template>
  <LabFrame
    title="多副本路由 — 送到有前缀的副本能省 prefill, 但别把一个副本压垮"
    sub="- 模拟什么: 集群中同一模型部署了 8 个独立的推理引擎副本, 每个副本各自维护容量为 24k token 的本地 RadixCache, 副本之间不共享 KV cache。路由层需要决定将每条新请求分发至哪个副本。多轮对话第 k 轮的 prompt 由「system prompt + 此前全部轮次历史」组成, 若路由至上一轮访问过的副本即可直接命中整段历史前缀, 若发往其余副本则需对整段历史重新执行 prefill。前缀感知路由先查询各副本本地缓存与当前请求的最长公共前缀并优先选择匹配最长的副本; 但当大量请求共享同一个热门 system prompt 时, 流量会持续集中涌向最先缓存该前缀的少数副本, 造成严重的队列拥塞。为此需引入负载阈值兜底机制: 一旦前缀最佳副本的排队积压超出最空闲副本达设定阈值以上, 便将请求改发至最空闲副本 (此处「积压」定义为该副本队列中待处理任务折算成的预计执行秒数)。
- 图上是什么: 上下两幅视图及底部决策说明:
  - 图 ①: Python demo 完整回放 1084 条请求 (源自 200 段多轮对话, 共享 4 种 system prompt) 后, 各副本 R0 至 R7 承接请求量的占比分布 (百分比), 水平虚线标示理想绝对均衡时的 12.5%; 该分布随路由策略与负载阈值联动变化。
  - 图 ②: 前端实时推演的单次路由决策快照。当某段对话第 3 轮的请求到达时, 每个副本上方并列展示两根柱子: 紫色柱表示该副本本地缓存可命中的前缀 token 数 (R1 缓存了整段对话历史共 1900 token, R0 与 R5 仅缓存了 512 token 的公共 system prompt, 其余副本命中为 0), 黄色柱表示该副本当前的排队积压秒数; 绿色边框标示路由器本次选定的目标副本。
  - 图表下方文字: 说明本次路由选中该副本的具体判定依据。
- 可以做什么: 通过三个按钮切换路由策略, 并调节负载阈值滑杆或直接拖动图 ② 中的积压手柄:
  - 「轮询」: 按请求到达序号依次循环分发, 既不检查前缀命中也不参考副本负载。
  - 「最少负载」: 仅比较当前排队积压, 始终发往最空闲的副本。
  - 「前缀感知」: 优先选择前缀匹配最长的副本, 并结合差值阈值执行过载兜底; 选中该策略后会出现「负载阈值」滑杆, 调节范围从 ∞ (纯前缀匹配、永不分流) 到 0.05 s (轻微差距即分流, 接近最少负载策略)。
  - 在图 ② 中上下拖动任意副本黄色积压柱顶部的手柄实时改变其队列积压, 观察选中绿框在何时从 R1 切换至其余空闲副本。
- 应该看到: 在纯前缀模式 (阈值为 ∞) 下, 图 ① 仅有 3 个副本承担全部流量, 其余 5 个副本完全闲置; 收紧负载阈值后各副本流量柱逐步趋于均衡, 但全局缓存命中率有所回落。在图 ② 中将 R1 的积压拖高至超出最空闲副本达阈值以上时, 选中绿框会立即跳转至最空闲副本。右侧读数含义如下:
  - 「命中率」表示全部请求的输入 prompt token 中免除重复 prefill 的比例。
  - 「负载 max / mean」为最繁忙副本承接任务量与集群均值之比, 1 代表完全均衡, 超过 2 时标红。
  - 「TTFT 均值 / p90」为首 token 延迟的均值与分位数 (单位为秒, 由排队代价模型算出而非真机实测), 同时受缓存命中率 (影响 prefill 计算量) 与队列负载均衡度 (影响排队等待时间) 共同制约。"
    module="llm_infer/m27"
    run="python -m llm_infer.m27_multi_replica_routing.demo"
    :challenge="{
      ask: '将「负载阈值」拖至 ∞ (纯前缀感知模式), 此时缓存命中率达到最高 (93.9%)。对应的平均 TTFT 会低于「最少负载」策略 (0.469 s) 吗?',
      answer: '不会更低, 反而恶化至「最少负载」策略的 200 倍以上, 达到 105.954 s。\n最高缓存命中率并不等价于最低延迟, 因为前缀命中仅能节省 prefill 计算时间, 而请求的实际 TTFT 还包含副本队列中的排队等待时间。在纯前缀感知策略下会出现严重的热点倾斜:\n- 新对话的第 1 轮仅携带公共 system prompt, 只会在最先缓存了该 system prompt 的副本上命中 512 token, 而在其余副本上命中均为 0, 导致同类新会话被永久吸附至该单一副本; 后续轮次生成的对话历史也仅驻留在该副本上, 进一步锁定后续流量。\n- 数据集中共有 4 种公共 system prompt, 至多只能激活 4 个副本, 本组负载中实际仅有 3 个副本承接了全部流量 (占比分别为 52% / 32% / 16%), 导致「负载 max / mean」高达 4.18。热点副本到达率超过服务率后队列持续积压, 105.954 s 是排队代价模型进入过载发散区的表现, 具体数值本身无稳态物理意义, 仅表明等待队列正在无限增长。\n逐步收紧负载阈值时, 平均 TTFT 呈先降后升趋势:\n- 阈值为 1 s: 命中率为 83.6%, 平均 TTFT 降至 0.314 s, 比「最少负载」基线 (0.469 s) 降低 33%。\n- 阈值为 0.5 s: 达到延迟最优点, 平均 TTFT 为 0.284 s。\n- 阈值为 0.05 s: 轻微负载差异即触发强制分流, 命中率跌至 72.9%, 平均 TTFT 回升至 0.308 s, 行为逐步退化向「最少负载」策略 (后者命中率仅为 59.3%, 多达 41% 的 prompt token 需重复计算)。\n开启合理的负载阈值还会促使公共热门前缀自动跨副本扩散: 被分流至空闲副本的请求执行完毕后, 会将该公共 system prompt 同步写入新副本的本地缓存; 此后新到达的同前缀会话在多个副本上均可命中 512 token, 前缀长度平手时路由器便可自然按最小负载择优分发。',
    }"
  >
    <template #controls>
      <div class="row">
        <button v-for="(name, p) in POLICY" :key="p" type="button" :class="{ active: policy === p }" :aria-pressed="policy === p" @click="policy = p">{{ name }}</button>
      </div>
      <LabSlider v-if="policy === 'prefix'" v-model="thi" label="负载阈值" :min="0" :max="6" :format="(i) => THR_LABEL[i]" />
    </template>

    <svg ref="svg" viewBox="0 0 560 330" role="group" aria-label="各副本负载与单次路由决策">
      <text x="4" y="12" class="cap">① demo 跑完 1084 个请求后各副本分到的活 (占比), 虚线 = 完全均匀的 12.5%; 不随图 ② 的拖动变化</text>
      <g v-for="(s, i) in cfg.share" :key="`s${i}`">
        <rect :x="bx(i)" :y="140 - s * 2.2" width="44" :height="s * 2.2" class="share" />
        <text :x="bx(i) + 22" :y="136 - s * 2.2" class="tick">{{ s }}%</text>
        <text :x="bx(i) + 22" y="152" class="tick">R{{ i }}</text>
      </g>
      <line x1="10" x2="550" :y1="140 - 12.5 * 2.2" :y2="140 - 12.5 * 2.2" class="mean" />

      <text x="4" y="176" class="cap">② 一次路由决策: 这个请求在各副本上能命中的前缀 (token) 与各副本的积压 (秒, 拖把手可改)</text>
      <g v-for="(r, i) in REPS" :key="`r${i}`">
        <rect :x="bx(i) - 3" y="182" width="50" height="136" class="pick" :class="{ on: i === choice }" />
        <rect :x="bx(i)" :y="310 - r.hit / 20" width="20" :height="r.hit / 20" class="hitb" />
        <rect :x="bx(i) + 24" :y="310 - backlog[i] * 40" width="20" :height="backlog[i] * 40" class="load" />
        <g class="draggable knob" tabindex="0" role="slider" :aria-label="`副本 R${i} 积压 (秒)`" aria-valuemin="0" aria-valuemax="3" :aria-valuenow="backlog[i]"
          @pointerdown="start($event, { svg, onMove: (e) => drag(i, e) })"
          @keydown.up.prevent="nudge(i, 0.1)" @keydown.down.prevent="nudge(i, -0.1)">
          <rect :x="bx(i) + 21" y="190" width="26" height="120" class="grab" />
          <rect :x="bx(i) + 22" :y="307 - backlog[i] * 40" width="24" height="6" rx="3" class="handle" />
        </g>
        <text :x="bx(i) + 22" y="326" class="tick">{{ r.hit }} · {{ backlog[i].toFixed(1) }}s</text>
      </g>
    </svg>
    <p class="legend"><span class="sw hitb" /> 前缀命中 (这个副本缓存里有多少 token 不用重算) <span class="sw load" /> 积压 (队列里还没做完的活, 秒; 拖柱顶的把手) · 绿框 = 路由器选中的副本</p>
    <p class="lab-note why">{{ why }}</p>

    <template #stats>
      <div class="kv"><span>{{ cfg.name }}: 前缀命中率 (prompt token 里不用重算的比例)</span><b :class="cfg.hit > 80 ? 'good' : ''">{{ cfg.hit }}%</b></div>
      <div class="kv"><span>负载 max / mean (最忙副本是平均的几倍, 1 = 完全均匀)</span><b :class="cfg.bal > 2 ? 'bad' : cfg.bal < 1.2 ? 'good' : ''">{{ cfg.bal.toFixed(2) }}</b></div>
      <div class="kv"><span>TTFT 首 token 延迟: 均值 / p90 (秒, 代价模型估算)</span><b :class="cfg.ttft > 1 ? 'bad' : cfg.ttft < 0.35 ? 'good' : ''">{{ cfg.ttft.toFixed(3) }} / {{ cfg.p90.toFixed(3) }}</b></div>
      <div class="kv"><span>② 这个请求发往</span><b>R{{ choice }} (在那里命中 {{ REPS[choice].hit }} token)</b></div>
      <div class="lab-note">
        <p>图 ① 的命中率、max/mean 与 TTFT 读数来自 Python demo 的 [1] 与 [3] 实验段; 各副本的流量占比 share 由同一 simulate() 函数针对各组阈值配置运行得出 (demo 控制台仅打印了纯前缀策略的一行分布: 52 / 32 / 0 / 0 / 0 / 16 / 0 / 0)。</p>
        <p>TTFT 由单队列 FIFO 排队代价模型计算得出:</p>
        <ul class="pts">
          <li>每个副本是一条先到先做的队列, prefill 1000 tok/s, decode 100 tok/s。</li>
          <li>TTFT = 到达时的积压 + 没命中部分的 prefill 时间。</li>
        </ul>
        <p>缓存命中率则是在真实执行 RadixCache 前缀树匹配与驱逐逻辑后统计得出。</p>
        <p>图 ② 的路由判定逻辑与 Python 代码中的 route() 函数完全一致:</p>
        <ol class="pts">
          <li>前缀感知先选命中最长的 (平手选积压低的)。</li>
          <li>它比最闲副本多积压超过阈值, 就改发最闲的。</li>
        </ol>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { useDrag } from '@/composables/useDrag.js'
import { argmax, clamp, range } from '@/utils/labmath.js'

// 数据: python -m llm_infer.m27_multi_replica_routing.demo 的 [1] [3] 段;
// share = router.simulate(chat_stream(), policy, threshold_s=thr)['share'] (四舍五入到整数 %)
const POLICY = { round_robin: '轮询', least_load: '最少负载', prefix: '前缀感知' }
const RR = { name: '轮询', hit: 59.3, bal: 1.03, ttft: 0.507, p90: 1.09, share: [13, 12, 13, 13, 12, 12, 12, 12] }
const LL = { name: '最少负载', hit: 59.3, bal: 1.02, ttft: 0.469, p90: 0.999, share: [13, 13, 12, 13, 12, 12, 12, 13] }
const THR = [Infinity, 4, 2, 1, 0.5, 0.2, 0.05]
const THR_LABEL = ['∞ (纯前缀)', '4 s', '2 s', '1 s', '0.5 s', '0.2 s', '0.05 s']
const PX = [
  { hit: 93.9, bal: 4.18, ttft: 105.954, p90: 300.605, share: [52, 32, 0, 0, 0, 16, 0, 0] },
  { hit: 93.4, bal: 1.45, ttft: 0.689, p90: 2.102, share: [15, 15, 11, 8, 9, 18, 13, 11] },
  { hit: 90.5, bal: 1.15, ttft: 0.461, p90: 1.486, share: [14, 14, 11, 12, 12, 14, 12, 11] },
  { hit: 83.6, bal: 1.19, ttft: 0.314, p90: 0.912, share: [14, 15, 10, 11, 12, 13, 12, 13] },
  { hit: 77.9, bal: 1.12, ttft: 0.284, p90: 0.823, share: [13, 14, 12, 12, 11, 13, 12, 13] },
  { hit: 75.7, bal: 1.07, ttft: 0.285, p90: 0.83, share: [12, 13, 12, 12, 13, 12, 12, 13] },
  { hit: 72.9, bal: 1.12, ttft: 0.308, p90: 0.908, share: [12, 14, 12, 13, 12, 13, 11, 13] },
]
// ② 示意: 一段对话的第 3 轮, R1 缓存了它的整段历史, R0/R5 只缓存了同一个 system prompt (512)
const REPS = [512, 1900, 0, 0, 0, 512, 0, 0].map((hit) => ({ hit }))

const policy = ref('prefix'), thi = ref(3)
const backlog = ref([1.2, 2.4, 0.6, 0.3, 0.8, 0.9, 0.4, 0.5])
const svg = ref(null)
const { start } = useDrag()
const thr = computed(() => (policy.value === 'prefix' ? THR[thi.value] : Infinity))

const cfg = computed(() => {
  if (policy.value === 'round_robin') return RR
  if (policy.value === 'least_load') return LL
  return { name: `前缀感知, 阈值 ${THR_LABEL[thi.value]}`, ...PX[thi.value] }
})

// ★ 与 route() 相同: 命中最长者优先 (平手选积压低的); 它比最闲副本多积压超过阈值, 就改走最闲的
const least = computed(() => argmax(backlog.value.map((b) => -b)))
const best = computed(() => argmax(range(8).map((i) => REPS[i].hit * 1e3 - backlog.value[i])))
const gap = computed(() => backlog.value[best.value] - backlog.value[least.value])
const rr = 5   // round_robin 的请求序号 k: 这个请求是第 k 个, 发往 k mod 8
const choice = computed(() => {
  if (policy.value === 'round_robin') return rr % 8
  if (policy.value === 'least_load') return least.value
  return gap.value > thr.value ? least.value : best.value
})

// 这一次为什么选它: 三种策略各说各的理由
const why = computed(() => {
  if (policy.value === 'round_robin') return `轮询: 这是第 ${rr} 个请求, 发往 ${rr} mod 8 = R${choice.value}。不看命中, 也不看积压。`
  if (policy.value === 'least_load') return `最少负载: 只看积压, 选最闲的 R${least.value} (${backlog.value[least.value].toFixed(1)} s)。命中多少不管。`
  const g = `前缀最长的 R${best.value} 比最闲的 R${least.value} 多积压 ${gap.value.toFixed(1)} s`
  return gap.value > thr.value ? `${g}, 超过阈值 ${THR_LABEL[thi.value]} → 让位给 R${least.value}。` : `${g}, 没超过阈值 ${THR_LABEL[thi.value]} → 仍发往 R${best.value}。`
})

const bx = (i) => 14 + i * 67
const drag = (i, { y }) => { backlog.value[i] = Math.round(clamp((310 - y) / 40, 0, 3) * 10) / 10 }
const nudge = (i, d) => { backlog.value[i] = Math.round(clamp(backlog.value[i] + d, 0, 3) * 10) / 10 }
</script>

<style scoped>
svg { min-width: 540px; touch-action: pan-x pan-y; }
.cap { font-size: 10px; fill: var(--text-muted); }
.tick { font-size: 9px; fill: var(--text-dim); text-anchor: middle; }
.share { fill: var(--accent); opacity: 0.8; }
.mean { stroke: var(--text-muted); stroke-dasharray: 5 4; }
.pick { fill: transparent; stroke: var(--border); rx: 4; }
.pick.on { fill: var(--accent-soft); stroke: var(--left); stroke-width: 2; }
.hitb { fill: var(--accent); }
.load { fill: var(--warn); }
.grab { fill: transparent; }
.knob { cursor: ns-resize; outline: none; }
.handle { fill: var(--text); stroke: var(--bg-card); }
.knob:focus-visible .handle { stroke: var(--accent); stroke-width: 2; }
.legend { font-size: 11px; color: var(--text-dim); margin-top: 6px; }
.sw { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin: 0 4px 0 10px; }
.sw.hitb { background: var(--accent); }
.sw.load { background: var(--warn); }
.why { margin-top: 8px; min-height: 1.7em; }
</style>
