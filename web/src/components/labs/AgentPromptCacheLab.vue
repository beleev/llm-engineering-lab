<!--
  Prompt caching 回放 (对应 llm_agent/m19_prompt_caching/demo.py 的 PromptCache.request / bill / replay / session)。
  只讲一件事: 缓存按前缀字节匹配 + TTL。时间戳放错位置一次也命中不了; 两轮之间停太久, 缓存就过期了。
-->
<template>
  <LabFrame
    title="Prompt caching — 同一段会话 12 次模型调用的账单, 缓存命中和没命中差多少"
    sub="- 模拟什么: prompt caching (提示缓存) 的前缀匹配与过期计费机制。agent 每轮推理均需向服务端重发完整上下文, 其中开头的 system prompt 与工具定义 (本例约为 5300 token) 在整个会话期间保持固定。本实验回放一段包含 6 轮交互、共计 12 次模型调用的会话: 每轮内部包含 2 次相隔 15 秒的模型调用, 轮与轮之间存在一段停顿间隔。提示缓存允许服务端将截至指定断点的上下文前缀预计算并保存下来:
  - 缓存读取: 若后续请求的前缀与已缓存内容逐字节完全一致, 则直接复用 KV 状态, 读取单价仅为基准全价的 0.1 倍。
  - 缓存写入: 首次构建或刷新缓存条目时需支付基准全价 1.25 倍的写入单价。
  - 生存期过期: 缓存条目受 TTL (生存时间) 限制, 若连续 5 分钟未被任何新请求命中则自动失效。
  - 严格前缀匹配: 缓存仅识别逐字节一致的前缀序列; 若在 system prompt 起始处动态拼接当前秒级时间戳, 每次请求的首部字节均发生变化, 将导致缓存命中率归零且每次都白付写入溢价。
- 图上是什么: 每行对应一次模型请求, 从上到下按发送时间排列, 左侧标注所属轮次:
  - 中间分段横条: 表示单次请求的输入 token 构成, 按计费档位着色。绿色代表命中缓存的读取部分 (单价 ×0.1), 黄色代表本次新写入缓存的部分 (单价 ×1.25), 灰色代表未纳入缓存的常规全价部分 (单价 ×1)。
  - 行末数值: 本次调用折算后的等效费用, 以「全价 token」为计价单位 (即 1 个常规全价输入 token 计为 1)。
  - 点击任意一行: 下方面板展示该次调用距上一次缓存写入或刷新的间隔秒数、TTL 是否已过期, 以及具体的 read、write、input 分项 token 数、折算费用与首 token 延迟 (TTFT) 估算值。
- 可以做什么: 第一排切换三种上下文组织与缓存策略, 第三种策略额外提供停顿时长滑杆:
  - 「不缓存」: 禁用提示缓存, 所有输入一律按基准全价计费。
  - 「缓存 · 时间戳在 system 开头」: 每次请求均在 system prompt 最前端注入实时时间戳, 破坏前缀稳定性。
  - 「缓存 · 时间戳在用户消息末尾」: 将实时时间戳附加在每轮新增的用户消息末尾, 写入对话历史后即固定不变, 保持前序前缀稳定。
  - 滑杆「轮间停顿」: 仅在第三种策略下显示, 拖动可调整相邻两轮之间的用户停顿秒数以观察 TTL 过期临界点; 前两种策略的总费用与停顿时长无关, 故隐藏该滑杆。
- 应该看到: 两种时间戳放置位置的成本反差与右侧汇总读数:
  - 「缓存 · 时间戳在用户消息末尾」: 仅第 1 次调用经历冷启动写入 (整条呈黄色), 后续各次调用的绝大部分前缀均呈绿色读取状态, 累计总成本降至 15267, 仅为不缓存基线 (73826) 的 0.21 倍, 缓存命中次数达 11/12。
  - 「缓存 · 时间戳在 system 开头」: 全部 12 行均呈整条黄色写入状态, 读取命中次数为 0, 累计总成本反而升至不缓存基线的 1.25 倍。提示缓存本质上是用首次写入溢价换取后续多次读取折扣, 一旦前缀无法复用便只剩额外溢价。
  - 右侧「总成本」: 全部 12 次模型调用的等效费用总和。
  - 「同批请求不缓存」: 同一批请求在禁用缓存时按基准全价计算的费用。
  - 「倍数」: 当前总成本相对于同批请求不缓存费用的比值, 小于 1 标绿, 大于 1 标红。
  - 「命中」: 12 次请求中成功读取到有效缓存的次数占比。
  - 「平均 TTFT」: 首 token 延迟的估算均值; 命中缓存的前缀免去了重复 prefill 计算, 因此响应显著加快。"
    module="llm_agent/m19"
    run="python -m llm_agent.m19_prompt_caching.demo"
    :challenge="{
      ask: '在「缓存 · 时间戳在用户消息末尾」模式下, 将「轮间停顿」滑杆从 30 秒逐步向右拖动。先预测: 停顿增至多少秒时总成本会发生阶跃式上涨? 上涨后的总成本是多少?',
      answer: '当「轮间停顿」拖至 285 秒时发生阶跃变化, 总成本从 15267 直接跃升至 50380, 缓存命中次数从 11/12 跌落至 6/12。\n- 过期时间计算: 缓存条目的过期时刻等于上一次访问或写入时刻加上 300 秒 (即 TTL)。每轮内部包含两次相隔 15 秒的调用, 若以每轮首次写入为计时起点, 下一轮首次调用到达时的累计间隔为「轮内 15 秒 + 轮间停顿秒数」。\n- 临界点触发: 当轮间停顿增至 285 秒时, 总间隔达到 $15 + 285 = 300$ 秒, 恰好触及 TTL 过期阈值, 服务端缓存失效, 导致每一轮的首次调用都必须全量重写前缀。\n- 残余命中来源: 此时唯一仍能命中缓存的仅剩每轮内部的第 2 次调用, 因为它距同轮第 1 次写入仅相隔 15 秒, 远在 TTL 之内; 6 轮合计命中 6 次, 对应命中率 6/12。\n因此, 停顿为 270 秒时仍能保持 11/12 命中, 一旦跨过 285 秒便直接跌至 6/12, 中间不存在平滑过渡。点击任意请求行可在下方查看该次请求距上次缓存写入的确切秒数及过期状态。\n在长周期任务中维持缓存命中通常有两种工程手段:\n- 控制相邻请求间隔不超过 TTL 上限 (加快工具流转节奏, 或在空闲期定时发送轻量请求刷新缓存生存期)。\n- 切换为支持 1 小时留存的扩展 TTL 档位, 但需承担更高的首次写入单价。',
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
        <p>在「缓存 · 时间戳在 system 开头」模式下, 仅在顶层 system prompt 维护一份时间戳而未向每轮用户消息追加时间后缀, 因此其原始输入总 token 数 (69116) 略低于其余两种模式 (73826), 对应的「同批请求不缓存」基线也随之不同; 横向对比缓存收益时应以「倍数」为准。</p>
        <p>实验中的计费倍率 (缓存读取 ×0.1、缓存写入 ×1.25) 与 TTFT 延迟均为教学模型设定值, 实际费率以各厂商最新官方定价为准; 选用 1 小时扩展 TTL 档位时写入溢价通常更高。</p>
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
