<!--
  Prompt caching 回放 (对应 llm_agent/m19_prompt_caching/demo.py 的 PromptCache.request / bill / replay / session)。
  只讲一件事: 缓存按前缀字节匹配 + TTL。时间戳放错位置一次也命中不了; 两轮之间停太久, 缓存就过期了。
-->
<template>
  <LabFrame
    title="Prompt caching — 12 次调用的账单"
    sub="一段 6 轮、12 次模型调用的 agent 会话 (每轮 2 次调用, 间隔 15 秒)。
      每一行是一次调用, 条长是这次请求的 token 数。绿色按 ×0.1 读缓存, 橙色按 ×1.25 写缓存, 灰色按 ×1 全价。
      切换三种做法, 拖轮间停顿, 点任意一行看这次调用的账。"
    module="llm_agent/m19"
    run="python -m llm_agent.m19_prompt_caching.demo"
    :challenge="{
      ask: '「时间戳在用户消息末尾」模式下, 把轮间停顿从 30 秒慢慢往右拖。先猜: 拖到多少秒时账单突然跳涨? 跳到多少?',
      answer: '拖到 285 秒, cost 从 15267 跳到 50380, 命中从 11/12 掉到 6/12。\n- 过期时刻: 上一轮最后一次调用写入缓存, 过期时刻 = 那一刻 + 300 秒。\n- 下一轮开头: 第一次调用在 15 + 停顿 秒之后。停顿 ≥ 285 秒, 就晚于过期时刻。\n- 还能命中的: 只剩每轮内的第 2 次调用 (15 秒后)。\n它是个悬崖, 不是斜坡。长任务里有两条路:\n- 缩短间隔: 让轮次间隔落在 TTL 内。\n- 换长 TTL: 付 1 小时 TTL 更高的写入价。',
    }"
  >
    <template #controls>
      <div class="row">
        <button v-for="m in MODES" :key="m.id" type="button" :class="{ active: mode === m.id }" @click="mode = m.id">{{ m.label }}</button>
      </div>
      <LabSlider v-model="pause" label="轮间停顿" :min="30" :max="600" :step="15" unit=" s" />
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
    <p class="detail mono">调用 {{ sel + 1 }} · t={{ calls[sel].now }}s · read {{ calls[sel].read }} · write {{ calls[sel].write }} · input {{ calls[sel].input }} → cost {{ Math.round(calls[sel].cost) }}, TTFT ≈ {{ Math.round(calls[sel].ms) }}ms</p>

    <template #stats>
      <div class="kv"><span>总成本</span><b :class="tot.cost < tot.base ? 'good' : tot.cost > tot.base ? 'bad' : ''">{{ Math.round(tot.cost) }}</b></div>
      <div class="kv"><span>同批请求不缓存</span><b>{{ tot.base }}</b></div>
      <div class="kv"><span>倍数</span><b :class="tot.cost < tot.base ? 'good' : tot.cost > tot.base ? 'bad' : ''">{{ (tot.cost / tot.base).toFixed(2) }}x</b></div>
      <div class="kv"><span>命中</span><b>{{ tot.hits }}/12</b></div>
      <div class="kv"><span>平均 TTFT</span><b>{{ Math.round(tot.ms) }}ms</b></div>
      <div class="lab-note">
        <p>「时间戳在 system 开头」那种会话, 用户消息里没有时间戳, 所以总 token (69116) 比另外两种 (73826) 少。</p>
        <p>价格和延迟倍率是示意值。</p>
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
    return { ...u, now, cost, ms }
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
</style>
