<!-- 分层 KV offload 实验台 (对应 llm_infer/m20_kv_offload): TieredKVCache 的 JS 移植, 时间全部来自代价模型。 -->
<template>
  <LabFrame
    title="分层 KV offload — GPU 装不下的历史 KV 降级到 CPU 和磁盘, 再决定搬回来还是重算"
    sub="- 模拟什么: 多轮对话里每一轮的 prompt 都是「共享的 system prompt + 这个用户的全部历史 + 新消息」。前缀缓存能让历史部分的 KV 免重算, 前提是它还在 GPU 上; 用户一多, GPU 放不下所有人的历史, 最久没用的 block (KV 按 16 个 token 一块存) 会被挤掉。分层 offload 的想法: 被挤掉的 block 不丢, 降级到更大更慢的一层 (GPU 到 CPU 内存, 再到磁盘), 再命中时搬回来。但搬也要时间: 每层有带宽和每次加载的固定延迟, 链路慢的时候搬回来可能比重算还慢, 所以每层还要判断一次「加载还是重算」。所有时间来自代价模型 (重算 8000 token/s, 每 token KV 128 KiB), 不是实测。
      - 图上是什么: 每根柱子是一个请求 (从左到右按轮次排, 每轮所有用户各一条), 柱高是它的 prompt 有多少 token, 按颜色分成四段: 绿色是在 GPU 命中的 token, 紫色是在 CPU 命中的, 黄色是在磁盘命中的, 半透明红色是哪一层都没有、必须重算的 (miss)。黑色折线是每个请求的 TTFT (首 token 延迟, 这里等于重算时间加搬运时间), 刻度在右侧。柱子下面一行字是选中请求的明细。
      - 可以做什么: 点「加载 / 重算取小」(每层比较搬运和重算哪个快, 取快的) 或「命中就加载」(命中了就老实去搬) 切换策略; 「换一组」换一组随机的消息长度, 「恢复默认」回到 demo 的参数; 拖「用户数」改并发用户 (每人 6 轮); 拖三个「容量」改每层能放多少 block; 拖「磁盘带宽」和「磁盘每次加载固定延迟」改磁盘有多慢; 点或悬停一根柱子 (或聚焦图后按左右方向键) 选中一个请求。
      - 应该看到: 第 1 轮全是红色 (没有历史可以命中), 之后几轮绿色越来越少、紫色和黄色越来越多, 因为 GPU 只放得下 64 块, 老历史都被降级了。右侧读数: 前两行是全部请求的 token 里各层命中和 miss 的比例 (miss 超过一半标红); 「平均 TTFT」和括号里「无缓存」(全部重算) 的时间比, 没比无缓存快 10% 以上就标红; 「磁盘交叉点 n*」是命中块数至少要多少, 搬磁盘才比重算划算, 每块搬运时间超过重算时间 (2 ms) 时显示「永不」。"
    module="llm_infer/m20"
    run="python -m llm_infer.m20_kv_offload.demo"
    :challenge="{
      ask: '把磁盘带宽拖到 0.5 GB/s, 再切到「命中就加载」。多了一层缓存, 平均 TTFT 会变好还是变差?',
      answer: '变差。磁盘带宽 0.5 GB/s 时, 搬一个 block (16 个 token, 2 MiB) 要 2 MiB ÷ 0.5 GB/s ≈ 4.2 ms, 重算它只要 2 ms (16 ÷ 8000 token/s)。这时命中了也不该搬。\n- 「命中就加载」: 老老实实去搬, 平均 TTFT 比没有磁盘这一层还差 (demo 里慢链路从 34.81 ms 涨到 68.91 ms)。\n- 「加载 / 重算取小」: 每层比较 $\\text{load} = \\text{固定延迟} + \\text{字节}/\\text{带宽}$ 和重算时间, 慢就直接重算。最坏也不比没有这层差。\n带宽够快时还有第二个门槛: 每次加载的固定延迟要靠命中块数摊薄。交叉点 $n^* = \\text{延迟} / (\\text{每块重算} - \\text{每块加载})$, 命中块数少于它时重算更快。默认磁盘 3 GB/s、2 ms 延迟时 $n^* = 2 / (2.00 - 0.699) \\approx 1.54$ 块。',
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
        <p>查找沿 block 的 hash 链走 (每块的 hash 包含它前面所有块的内容), 遇到第一个哪层都没有的块就停: 后面的块即使还在也用不上, 因为 KV 依赖完整前缀。</p>
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
