<!-- 分层 KV offload 实验台 (对应 llm_infer/m20_kv_offload): TieredKVCache 的 JS 移植, 时间全部来自代价模型。 -->
<template>
  <LabFrame
    title="分层 KV offload — GPU 装不下的历史 KV 降级到 CPU 和磁盘, 再决定搬回来还是重算"
    sub="- 模拟什么: 多轮对话中每一轮请求的 prompt 均由「共享 system prompt + 当前用户历史对话 + 新消息」拼接而成。前缀缓存可避免重复计算历史部分的 KV, 但前提是对应数据仍驻留在 GPU 显存中; 当并发用户增多导致 GPU 容量不足时, 最久未使用的 block (KV 按每块 16 个 token 存储) 会被驱逐。分层卸载 (offload) 的机制是不直接丢弃被驱逐的 block, 而是将其逐级降级至容量更大但速度更慢的存储层 (从 GPU 显存降至 CPU 内存, 再降至磁盘), 后续再次命中时按需回迁。由于跨层回迁受各层带宽与单次加载固定延迟限制, 慢速链路下回迁耗时可能超过直接重算, 因此每层还需在「加载」与「重算」之间做代价择优。实验中的所有耗时均由代价模型计算 (重算吞吐率 8000 token/s, 每 token KV 占 128 KiB), 并非真机实测。
- 图上是什么: 每根柱子代表一个请求 (按对话轮次从左向右排列, 每轮包含所有用户各一条请求), 柱高表示该请求 prompt 的 token 总数, 并按颜色分为四段; 柱状图上方叠加一条折线, 下方附有一行明细:
  - 绿色: 在 GPU 显存命中的 token。
  - 紫色: 在 CPU 内存命中的 token。
  - 黄色: 在磁盘命中的 token。
  - 半透明红色: 所有存储层均未命中、必须重新计算的 token (miss)。
  - 黑色折线: 每个请求的 TTFT (首 token 延迟, 此处等于重算耗时与数据搬运耗时之和), 对应右侧纵轴刻度。
  - 柱状图下方文字: 当前选中请求的分项明细。
- 可以做什么: 按界面从上到下依次操作:
  - 点击「加载 / 重算取小」(每层对比数据搬运与重算耗时并取较小者) 或「命中就加载」(只要命中即执行跨层搬运) 切换回迁策略。
  - 点击「换一组」重新采样随机消息长度, 点击「恢复默认」还原 Python demo 的默认参数。
  - 拖动「用户数」修改并发用户数量 (每位用户固定进行 6 轮对话)。
  - 拖动三个存储层的「容量」滑杆修改每层可容纳的 block 数。
  - 拖动「磁盘带宽」与「磁盘每次加载固定延迟」调整磁盘层的读取性能。
  - 点击或悬停任一柱子 (或聚焦图表后按左右方向键) 选中对应请求查看明细。
- 应该看到: 第 1 轮请求全部呈红色 (尚无历史对话可供命中); 随后的轮次中绿色占比减少、紫色与黄色占比增加, 原因是 GPU 默认仅能容纳 64 个 block, 较早的历史块被逐级降级。右侧读数含义如下:
  - 前两行为全部请求 token 中各存储层命中率与 miss 比例 (miss 超过一半时标红)。
  - 「平均 TTFT」及其括号内相对于「无缓存」(全部重算) 的耗时对比, 若加速幅度不足 10% 则标红。
  - 「磁盘交叉点 n*」表示使磁盘回迁优于重算所需的最少命中块数; 当单块搬运耗时本身已超过单块重算耗时 (2 ms) 时显示「永不」。"
    module="llm_infer/m20"
    run="python -m llm_infer.m20_kv_offload.demo"
    :challenge="{
      ask: '将「磁盘带宽」拖至 0.5 GB/s, 并切换至「命中就加载」策略。此时增加磁盘缓存层后, 平均 TTFT 会缩短还是恶化?',
      answer: '平均 TTFT 会恶化。当磁盘带宽仅为 0.5 GB/s 时, 回迁单个 block (16 个 token, 体积 2 MiB) 按 2 MiB 除以 0.5 GB/s 计算耗时约为 4.2 ms, 而直接重算该 block 仅需 2 ms (16 除以 8000 token/s)。在此条件下即使磁盘命中也不应执行回迁:\n- 「命中就加载」策略: 无条件从磁盘读取命中块, 导致平均 TTFT 比完全不设磁盘层更慢 (Python demo 中慢速链路下平均 TTFT 从 34.81 ms 升至 68.91 ms)。\n- 「加载 / 重算取小」策略: 在每一层对比加载耗时 $\\text{load} = \\text{固定延迟} + \\text{字节}/\\text{带宽}$ 与重算耗时, 若加载更慢则直接重算, 因此最坏情况下也不会劣于无该缓存层的基线。\n即便存储层带宽足够高, 仍存在第二道收益门槛: 单次加载的固定延迟需要足够多的连续命中块数来摊薄。盈亏交叉点为 $n^* = \\text{延迟} / (\\text{每块重算} - \\text{每块加载})$, 当命中块数低于 $n^*$ 时直接重算依然更快。在默认磁盘带宽 3 GB/s、固定延迟 2 ms 下, $n^* = 2 / (2.00 - 0.699) \\approx 1.54$ 块。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: !alwaysLoad }" @click="alwaysLoad = false">加载 / 重算取小</button>
        <button type="button" :class="{ active: alwaysLoad }" @click="alwaysLoad = true">命中就加载</button>
        <button type="button" @click="seed++">换一组</button>
        <button type="button" @click="reset">恢复默认</button>
      </div>
      <LabSlider v-model="users" label="用户数 (各 6 轮)" :min="2" :max="12" />
      <LabSlider v-model="capGpu" label="GPU 层容量" :min="0" :max="256" :step="8" unit=" blk" />
      <LabSlider v-model="capCpu" label="CPU 层容量 (25 GB/s)" :min="0" :max="512" :step="16" unit=" blk" />
      <LabSlider v-model="capDisk" label="磁盘层容量" :min="0" :max="4096" :step="128" unit=" blk" />
      <LabSlider v-model="diskBw" label="磁盘带宽" :min="0.5" :max="7" :step="0.5" unit=" GB/s" />
      <LabSlider v-model="diskLat" label="磁盘每次加载固定延迟" :min="0" :max="20" unit=" ms" />
    </template>

    <svg :viewBox="`0 0 ${W} ${H + 16}`" role="group" aria-label="每个请求的命中构成与 TTFT">
      <!-- 整排柱子只有一个 Tab 停靠点, 左右方向键换请求 -->
      <g
        class="bars" tabindex="0" role="slider" aria-label="选中的请求" aria-valuemin="1" :aria-valuemax="sim.reqs.length" :aria-valuenow="pickI + 1"
        :aria-valuetext="`请求 ${pickI + 1}, TTFT ${cur.ttft.toFixed(1)} ms`"
        @keydown.left.prevent="pick = Math.max(0, pickI - 1)" @keydown.right.prevent="pick = Math.min(sim.reqs.length - 1, pickI + 1)"
      >
        <g v-for="(r, i) in sim.reqs" :key="i" @mouseenter="pick = i" @click="pick = i">
          <rect :x="i * bw" y="0" :width="bw" :height="H" class="bg" />
          <rect
            v-for="s in r.segs" :key="s.k" :x="i * bw + 0.5" :y="H - (s.y0 + s.n) * ky"
            :width="Math.max(1, bw - 1)" :height="s.n * ky" :fill="COLORS[s.k]" :opacity="s.k === 3 ? 0.35 : 0.85"
          />
          <rect v-if="i === pickI" :x="i * bw" y="0" :width="bw" :height="H" class="pick" />
        </g>
      </g>
      <!-- 折线的纵轴: 右侧标 TTFT 刻度 -->
      <g v-for="f in [0.5, 1]" :key="f" class="yt">
        <line x1="0" :x2="W" :y1="H - f * (H - 4)" :y2="H - f * (H - 4)" />
        <text :x="W - 2" :y="H - f * (H - 4) + 11" text-anchor="end">TTFT {{ (sim.maxTtft * f).toFixed(0) }} ms</text>
      </g>
      <polyline :points="sim.reqs.map((r, i) => `${(i + 0.5) * bw},${H - (r.ttft / sim.maxTtft) * (H - 4)}`).join(' ')" class="ttft" />
      <text x="0" :y="H + 13" class="ax">请求 1 (第 1 轮)</text>
      <text :x="W" :y="H + 13" class="ax" text-anchor="end">请求 {{ sim.reqs.length }} (第 6 轮) · 柱高 = prompt token 数, 折线 = TTFT (右侧刻度)</text>
    </svg>
    <p class="legend">
      <span v-for="(n, k) in NAMES" :key="k"><i :style="{ background: COLORS[k], opacity: k === 3 ? 0.35 : 0.85 }" />{{ n }}</span>
    </p>
    <p class="lab-note detail">
      请求 {{ pickI + 1 }}: 用户 {{ cur.user }} 的第 {{ cur.turn }} 轮, prompt {{ cur.len }} 个 token, 其中
      GPU 命中 {{ cur.hit[0] }}、CPU 命中 {{ cur.hit[1] }}、磁盘命中 {{ cur.hit[2] }}、必须重算 {{ cur.miss }};
      TTFT <b>{{ cur.ttft.toFixed(1) }} ms</b> (全部重算要 {{ (cur.len / 8).toFixed(1) }} ms){{ cur.skipped ? `; 有 ${cur.skipped} 个命中的 token 因为加载比重算慢而选择了重算` : '' }}
    </p>

    <template #stats>
      <div class="kv"><span>命中在 GPU / CPU / 磁盘的 token 比例</span><b>{{ pct(sim.hit[0]) }} / {{ pct(sim.hit[1]) }} / {{ pct(sim.hit[2]) }}</b></div>
      <div class="kv"><span>miss (哪层都没有, 必须重算)</span><b :class="sim.miss > 0.5 ? 'bad' : 'good'">{{ pct(sim.miss) }}</b></div>
      <div class="kv">
        <span>平均 TTFT (无缓存全部重算是 {{ sim.noCache.toFixed(1) }} ms)</span>
        <b :class="sim.ttft > sim.noCache * 0.9 ? 'bad' : 'good'">{{ sim.ttft.toFixed(1) }} ms</b>
      </div>
      <div class="kv"><span><Tex text="磁盘交叉点 $n^*$ (至少命中几块才值得搬)" /></span><b :class="nStar === null ? 'bad' : ''">{{ nStar === null ? '永不' : nStar.toFixed(2) + ' 块' }}</b></div>
      <div class="lab-note">
        <p>每块 (16 tok = 2 MiB): 重算 2.00 ms · CPU 加载 0.084 ms · 磁盘加载 {{ diskPerBlock.toFixed(3) }} ms (另加每次 {{ diskLat }} ms 固定延迟)。</p>
        <p><Tex text="$n^* = \text{延迟} \div (\text{每块重算} - \text{每块加载})$:" /></p>
        <ul class="pts">
          <li>命中块数少于它: 固定延迟摊不薄, 重算更快。</li>
          <li>每块加载 ≥ 重算: 永远不该加载, 显示「永不」。</li>
        </ul>
        <p>缓存查找沿 block 的哈希链顺序推进 (每个块的哈希值均串联了其前方所有块的内容), 一旦遇到首个在所有存储层均未命中的块便立即停止: 由于自回归注意力的 KV 严格依赖完整前缀, 后续块即便仍在缓存中也无法直接复用。</p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import Tex from '@/components/Tex.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { mulberry32, range, sum } from '@/utils/labmath.js'

const BS = 16, BLOCK_BYTES = BS * 131072, RECOMP_MS = (n) => n / 8        // 8000 tok/s
const W = 640, H = 170
const COLORS = ['var(--left)', 'var(--accent)', 'var(--warn)', 'var(--danger)']
const NAMES = ['GPU 命中', 'CPU 命中', '磁盘命中', 'miss → 重算']

const users = ref(8), capGpu = ref(64), capCpu = ref(192), capDisk = ref(4096)
const diskBw = ref(3), diskLat = ref(2), alwaysLoad = ref(false), seed = ref(0), pick = ref(0)

const reset = () => { users.value = 8; capGpu.value = 64; capCpu.value = 192; capDisk.value = 4096; diskBw.value = 3; diskLat.value = 2; alwaysLoad.value = false }

const diskPerBlock = computed(() => (BLOCK_BYTES / (diskBw.value * 1e9)) * 1e3)
const nStar = computed(() => (diskPerBlock.value >= 2 ? null : diskLat.value / (2 - diskPerBlock.value)))

const sim = computed(() => {
  const tiers = [
    { cap: capGpu.value, bw: Infinity, lat: 0 }, { cap: capCpu.value, bw: 25, lat: 0.2 },
    { cap: capDisk.value, bw: diskBw.value, lat: diskLat.value },
  ]
  const lru = tiers.map(() => new Map())                                  // Map 的插入序 = LRU 序, 队首最旧
  const insert = (lv, h) => {
    if (lv >= lru.length) return
    lru[lv].delete(h); lru[lv].set(h, 1)
    while (lru[lv].size > tiers[lv].cap) { const old = lru[lv].keys().next().value; lru[lv].delete(old); insert(lv + 1, old) } // 溢出 → 降级
  }
  // 链式 hash 的替身: 前 4 块是共享 system prompt, 之后的块由 (用户, 块号) 唯一确定 (历史只追加)
  const ids = (u, len) => range(Math.floor(len / BS)).map((i) => (i < 4 ? `S${i}` : `u${u}:${i}`))
  const store = (hs) => [...hs].reverse().forEach((h) => { lru.slice(1).forEach((m) => m.delete(h)); insert(0, h) })
  const loadMs = (t, nb) => (nb === 0 || t.bw === Infinity ? 0 : t.lat + (nb * BLOCK_BYTES) / (t.bw * 1e9) * 1e3)

  const rand = mulberry32(seed.value + 11), msg = () => 40 + Math.floor(rand() * 60)
  const hist = Array(users.value).fill(64), reqs = []
  for (let turn = 1; turn <= 6; turn++) for (let u = 0; u < users.value; u++) {
    const len = (hist[u] += msg()), hs = ids(u, len)
    const where = []
    for (const h of hs) { const lv = lru.findIndex((m) => m.has(h)); if (lv < 0) break; where.push(lv) } // 首个 miss 处停
    const nb = tiers.map((_, lv) => where.filter((w) => w === lv).length)
    const miss = len - where.length * BS
    let ttft = RECOMP_MS(miss), skipped = 0
    tiers.forEach((t, lv) => {
      const load = loadMs(t, nb[lv]), re = RECOMP_MS(nb[lv] * BS)
      const useLoad = alwaysLoad.value || load <= re                      // ★ 逐层判断: 加载 vs 重算
      ttft += useLoad ? load : re
      if (!useLoad) skipped += nb[lv] * BS
    })
    store(hs)
    store(ids(u, (hist[u] += msg())))                                     // decode 生成的回复 KV 也登记进缓存
    let y0 = 0
    const hit = nb.map((n) => n * BS)
    const segs = [...hit, miss].map((n, k) => { const s = { k, n, y0 }; y0 += n; return s })
    reqs.push({ user: u, turn, len, hit, miss, ttft, skipped, segs })
  }
  const total = sum(reqs.map((r) => r.len))
  return {
    reqs, hit: [0, 1, 2].map((k) => sum(reqs.map((r) => r.hit[k])) / total), miss: sum(reqs.map((r) => r.miss)) / total,
    ttft: sum(reqs.map((r) => r.ttft)) / reqs.length, noCache: sum(reqs.map((r) => RECOMP_MS(r.len))) / reqs.length,
    maxLen: Math.max(...reqs.map((r) => r.len)), maxTtft: Math.max(...reqs.map((r) => r.ttft), 1),
  }
})
const bw = computed(() => W / sim.value.reqs.length)
const ky = computed(() => H / sim.value.maxLen)
const pickI = computed(() => Math.min(pick.value, sim.value.reqs.length - 1)) // 用户数变少时夹回合法范围
const cur = computed(() => sim.value.reqs[pickI.value])
const pct = (x) => (x * 100).toFixed(1) + '%'
</script>

<style scoped>
svg { min-width: 520px; } /* 窄屏: 图保持可读, 由 .lab-viz 横向滚动 */
.bars { outline: none; cursor: pointer; }
.bg { fill: transparent; }
.bars:focus-visible .pick { stroke: var(--accent); stroke-width: 2.5; }
.yt line { stroke: var(--border-strong); stroke-dasharray: 2 4; pointer-events: none; }
.yt text { font-size: 10px; fill: var(--text-muted); pointer-events: none; }
.pick { fill: none; stroke: var(--text); stroke-width: 1.5; pointer-events: none; }
.ttft { fill: none; stroke: var(--text); stroke-width: 1.5; pointer-events: none; }
.ax { font-size: 11px; fill: var(--text-dim); }
.legend { display: flex; flex-wrap: wrap; gap: 12px; font-size: 11px; color: var(--text-muted); margin-top: 6px; }
.legend i { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 4px; }
.detail { margin-top: 10px; }
.detail b { color: var(--text); }
</style>
