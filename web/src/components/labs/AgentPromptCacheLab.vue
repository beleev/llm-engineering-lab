<!--
  Prompt caching 回放 (对应 llm_agent/m19_prompt_caching/demo.py 的 PromptCache.request / bill / replay / session)。
  只讲一件事: 缓存按前缀字节匹配 + TTL。时间戳放错位置一次也命中不了; 两轮之间停太久, 缓存就过期了。
-->
<template>
  <LabFrame
    title="Prompt caching — 同一段会话 12 次模型调用的账单, 缓存命中和没命中差多少"
    sub="- 模拟什么: prompt caching (提示缓存)。agent 每一轮都把整个上下文重发给模型, 其中 system prompt 和工具定义 (这里约 5300 token) 每次都一模一样。这里回放一段 6 轮、12 次模型调用的 agent 会话: 每轮 2 次调用, 相隔 15 秒; 轮和轮之间停一段时间。缓存就是让服务端把「到某个断点为止的前缀」算好存下来:
  - 读: 下一次请求的前缀如果逐字节相同就直接读, 读的价钱是全价的 0.1 倍。
  - 写: 第一次写入缓存要付 1.25 倍。
  - 过期: 缓存有 TTL, 5 分钟没被用到就过期。
  - 缓存只认逐字节相同的前缀: 在 system prompt 开头拼一个当前时间戳, 前缀每次都变, 一次也命中不了, 还要每次付写入价。
- 图上是什么: 每一行是一次模型调用, 从上到下按时间排, 左边标着第几轮:
  - 中间的条: 这次请求的 token 数, 按计费方式分色。绿色是从缓存读的部分 (×0.1), 黄色是这次写进缓存的部分 (×1.25), 灰色是不走缓存的全价部分 (×1)。
  - 行末的数字: 这次调用的费用, 单位是「全价 token」, 1 个全价 token 算 1。
  - 点任意一行: 下方显示这次调用距离上次写入过了多少秒、TTL 过没过期, 以及它的 read / write / input 各多少、费用多少、首 token 延迟 TTFT 的估算。
- 可以做什么: 第一排切换三种做法, 第三种多一根滑杆:
  - 「不缓存」: 每次全价。
  - 「缓存 · 时间戳在 system 开头」: 每次在 system prompt 最前面拼当前时间, 前缀每次都变。
  - 「缓存 · 时间戳在用户消息末尾」: 时间戳跟在每轮用户消息后面, 进入历史后就不再变。
  - 滑杆「轮间停顿」: 只有第三种有, 拖它改每轮之间停多少秒, 看 TTL 什么时候过期; 前两种停多久账单都一样, 所以滑杆隐藏。
- 应该看到: 两种缓存做法的对照和右侧读数:
  - 「时间戳在用户消息末尾」: 只有第 1 次调用是冷的 (整条黄), 之后每次大部分是绿的, 总成本 15267, 是不缓存 (73826) 的 0.21 倍, 命中 11/12。
  - 「时间戳在 system 开头」: 每一行整条黄, 一次都没读到, 总成本反而是不缓存的 1.25 倍。缓存是用写入溢价换后续的读取折扣, 前缀不复用就只剩溢价。
  - 右侧「总成本」: 12 次相加。
  - 「同批请求不缓存」: 同一批请求按全价算的费用。
  - 「倍数」: 两者之比, 小于 1 变绿, 大于 1 变红。
  - 「命中」: 12 次里有几次读到了缓存。
  - 「平均 TTFT」: 首 token 延迟的示意值, 读缓存的部分不用重新 prefill, 所以更快。"
    module="llm_agent/m19"
    run="python -m llm_agent.m19_prompt_caching.demo"
    :challenge="{
      ask: '「时间戳在用户消息末尾」模式下, 把「轮间停顿」从 30 秒慢慢往右拖。先猜: 拖到多少秒时账单突然跳涨? 跳到多少?',
      answer: '拖到 285 秒, 总成本从 15267 跳到 50380, 命中从 11/12 掉到 6/12。\n- 上一轮的最后一次调用写入缓存, 过期时刻 = 那一刻 + 300 秒 (TTL)。\n- 下一轮的第一次调用在 15 + 停顿 秒之后 (每轮两次调用相隔 15 秒, 然后停顿)。停顿到 285 秒时, 15 + 285 = 300, 正好等于 TTL, 按过期处理, 整段重新写入。\n- 还能命中的只剩每轮内的第 2 次调用: 它在第 1 次写入之后 15 秒, 没过期。6 轮各 1 次, 就是 6/12。\n停顿 270 秒还命中 11/12, 285 秒就只剩 6/12, 中间没有过渡。点任意一行, 下方会显示它距上次写入多少秒、过没过期。\n长任务里有两种办法:\n- 让两次请求的间隔落在 TTL 内 (缩短间隔, 或者定时发一个只为刷新缓存的请求)。\n- 换 1 小时的长 TTL, 代价是写入价更高。',
    }"
  >
    <template #controls>
      <div class="row">
        <button v-for="m in MODES" :key="m.id" type="button" :class="{ active: mode === m.id }" @click="mode = m.id">{{ m.label }}</button>
      </div>
      <LabSlider v-if="mode === 'user'" v-model="pause" label="轮间停顿 (每轮之间等多久)" :min="30" :max="600" :step="15" unit=" s" />
      <p v-else class="tip">「轮间停顿」已隐藏: {{ mode === 'none' ? '不缓存时每次都按全价算' : '前缀每次都变, 一次也命中不了' }}, 停多久账单都一样。</p>
    </template>

    <div class="calls">
      <button v-for="(c, k) in calls" :key="k" type="button" class="call" :class="{ sel: sel === k }" @click="sel = k">
        <span class="mono t">{{ k % 2 ? '' : `轮 ${k / 2 + 1}` }}</span>
        <span class="bar">
          <i class="rd" :style="{ width: c.read / MAXTOK * 100 + '%' }" />
          <i class="wr" :style="{ width: c.write / MAXTOK * 100 + '%' }" />
          <i class="in" :style="{ width: c.input / MAXTOK * 100 + '%' }" />
        </span>
        <span class="mono cost">{{ Math.round(c.cost) }}</span>
      </button>
    </div>
    <p v-if="mode === 'user'" class="detail mono" :class="{ expired: calls[sel].gap >= TTL }">
      {{ sel ? `距上次写入 ${calls[sel].gap}s / TTL ${TTL}s → ${calls[sel].gap < TTL ? '还没过期, 能读缓存' : '已过期, 整段重新写入'}` : `第 1 次调用: 还没有缓存可读。之后每次写入或命中, 过期时刻都推到那一刻 + ${TTL}s` }}
    </p>
    <p class="detail mono">调用 {{ sel + 1 }} · t={{ calls[sel].now }}s · read {{ calls[sel].read }} · write {{ calls[sel].write }} · input {{ calls[sel].input }} → cost {{ Math.round(calls[sel].cost) }}, TTFT ≈ {{ Math.round(calls[sel].ms) }}ms</p>

    <template #stats>
      <div class="kv"><span>总成本 (12 次相加, 单位: 全价 token)</span><b :class="tot.cost < tot.base ? 'good' : tot.cost > tot.base ? 'bad' : ''">{{ Math.round(tot.cost) }}</b></div>
      <div class="kv"><span>同批请求不缓存的费用</span><b>{{ tot.base }}</b></div>
      <div class="kv"><span>倍数 (总成本 / 不缓存)</span><b :class="tot.cost < tot.base ? 'good' : tot.cost > tot.base ? 'bad' : ''">{{ (tot.cost / tot.base).toFixed(2) }}x</b></div>
      <div class="kv"><span>命中 (读到了缓存的调用数)</span><b>{{ tot.hits }}/12</b></div>
      <div class="kv"><span>平均 TTFT (首 token 延迟, 示意)</span><b>{{ Math.round(tot.ms) }}ms</b></div>
      <div class="lab-note">
        <p>「时间戳在 system 开头」那种会话, 用户消息里没有时间戳, 所以总 token (69116) 比另外两种 (73826) 少, 它的「同批请求不缓存」也跟着不同; 比较时只看倍数。</p>
        <p>价格倍率 (读 ×0.1, 写 ×1.25) 和延迟都是示意值, 真实定价以官方为准; 1 小时 TTL 的写入价更高。</p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'

const MODES = [{ id: 'none', label: '不缓存' }, { id: 'system', label: '缓存 · 时间戳在 system 开头' }, { id: 'user', label: '缓存 · 时间戳在用户消息末尾' }]
const mode = ref('user')
const pause = ref(30)
const sel = ref(2)

const PRICE = { input: 1, write: 1.25, read: 0.1 }
const TTL = 300, LOOKBACK = 20, SYS = 2 // 前 2 个 block 是工具, 下标 2 是 system (第一个断点)
// 来自 m19 demo 的 session("user"): 最后一次调用的每个 block 的 estimate_tokens; 每次调用的请求都是它的前缀 (已在 Python 里核对)
const TOK = [62, 56, 5302, 25, 29, 41, 56, 35, 69, 24, 29, 40, 56, 33, 67, 24, 29, 40, 56, 32, 66, 25, 29, 41, 56, 35, 69, 24, 29, 40, 56, 33, 67, 24, 29, 40, 56, 32]
const NB = [4, 8, 10, 14, 16, 20, 22, 26, 28, 32, 34, 38] // 12 次调用各自的 block 数
// session("system") 在 system 开头现拼时间戳后, 每次调用的总 token; 前缀每次都变, 永远不命中
const SYS_TOTAL = [5446, 5513, 5562, 5626, 5672, 5734, 5781, 5848, 5897, 5961, 6007, 6069]
const MAXTOK = 6100
const sumTok = (a, b) => TOK.slice(a, b).reduce((x, y) => x + y, 0)
const timeOf = (k) => Math.floor(k / 2) * (30 + pause.value) + (k % 2) * 15 // 每次调用 15s, 每轮后停 pause 秒 (RecordingLLM)

const calls = computed(() => {
  const entries = new Map() // 前缀长度 → 过期时刻。前缀是逐次追加的, 所以"前 i+1 个 block"就能当哈希的身份
  return NB.map((n, k) => {
    const now = timeOf(k)
    let u
    if (mode.value === 'none') u = { read: 0, write: 0, input: sumTok(0, n) }
    else if (mode.value === 'system') u = { read: 0, write: SYS_TOTAL[k], input: 0 }
    else {
      let hit = 0
      for (const bp of [SYS, n - 1]) {
        for (let i = bp; i > Math.max(-1, bp - LOOKBACK); i--) {
          if ((entries.get(i) ?? -1) > now) { hit = Math.max(hit, i + 1); break } // ★ 往回找最长的未过期前缀
        }
      }
      if (hit) entries.set(hit - 1, now + TTL) // 命中刷新 TTL
      for (const bp of [SYS, n - 1]) if (bp + 1 > hit) entries.set(bp, now + TTL)
      u = { read: sumTok(0, hit), write: sumTok(hit, n), input: 0 }
    }
    const cost = u.input * PRICE.input + u.write * PRICE.write + u.read * PRICE.read
    const ms = 300 + (u.input + u.write) * 0.2 + u.read * 0.02
    return { ...u, now, gap: k ? now - timeOf(k - 1) : 0, cost, ms }
  })
})
const tot = computed(() => {
  const c = calls.value
  const base = mode.value === 'system' ? SYS_TOTAL.reduce((a, b) => a + b, 0) : NB.reduce((a, n) => a + sumTok(0, n), 0)
  return { cost: c.reduce((a, x) => a + x.cost, 0), base, hits: c.filter((x) => x.read > 0).length, ms: c.reduce((a, x) => a + x.ms, 0) / c.length }
})
</script>

<style scoped>
.calls { display: flex; flex-direction: column; gap: 3px; }
.call { display: flex; align-items: center; gap: 8px; min-height: 0; padding: 2px 6px; border: 1px solid transparent; border-radius: var(--radius-sm); background: none; color: var(--text); }
.call.sel { border-color: var(--accent); background: var(--accent-soft); }
.t { width: 40px; flex-shrink: 0; font-size: 11px; color: var(--text-dim); text-align: left; }
.bar { flex: 1; display: flex; height: 14px; background: var(--bg); border-radius: 2px; overflow: hidden; }
.bar i { display: block; height: 100%; }
.rd { background: var(--left); }
.wr { background: var(--warn); }
.in { background: var(--text-dim); }
.cost { width: 48px; flex-shrink: 0; font-size: 11px; text-align: right; }
.detail { margin-top: 8px; font-size: 11px; color: var(--text-muted); line-height: 1.6; }
.detail.expired { color: var(--danger); }
.tip { font-size: 12px; color: var(--text-muted); }
</style>
