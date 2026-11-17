<!-- PagedAttention 实验台 (对应 llm_infer/m02_paged_attention): 事件列表 → computed 重放, 拖滑杆时结果确定。 -->
<template>
  <LabFrame
    title="PagedAttention — 把 KV 显存切成 block 按需分配, 看浪费少了多少"
    sub="- 模拟什么: 一张显卡上放 KV cache 的显存池。推理引擎把池子切成固定大小的 block, 好几条请求同时在里面写各自的 KV (每生成一个 token 就多写一格)。
      - 图上是什么: 上下两部分:
        - 上面一排格子: 池子里的每一个物理 block, 一共能放 384 个 token 的 KV。格子的颜色表示它属于哪条请求, 没颜色的是空闲; 格子里的填充高度是这个 block 已经写进了多少 token。
        - 下面每一行: 一条正在生成的请求和它的页表 block_table。页表里第 i 个数字, 就是这条请求的第 i 页 KV 实际放在哪个物理 block。
      - 可以做什么: 三个按钮管请求, 三个滑杆改参数:
        - 点「新请求」接一条新 prompt。
        - 点某条请求的「decode +1 / +8」让它多生成几个 token。
        - 点「结束」把它占的 block 还回池子。
        - 拖「block 大小」改每个 block 放几个 token。
        - 拖「连续预留的 max_len」改对照组给每条请求预留多长。
        - 拖「查地址」选一个逻辑位置, 右侧会显示它被翻译到哪个物理块、第几个槽。
      - 应该看到: 几条请求交替 decode 时, 各自新拿到的 block 在池子里是打散的, 不连续也没关系, 页表都记着。右侧读数:
        - 「浪费」一栏把两种做法放在一起算, 分页只浪费每条请求最后一个 block 没填满的部分; 「按 max_len 连续预留」要为每条请求押上整段 max_len, 没写到的全算浪费。
        - 「连续预留装得下 N 条」是同样的池子按整段预留只能放几条请求。"
    module="llm_infer/m02"
    run="python -m llm_infer.m02_paged_attention.demo"
    :challenge="{
      ask: '分页之后, attention 本身算得更快了吗?',
      answer: '没有更快, 反而略慢一点。\n读一个 token 的 KV 时, 要先查页表 block_table 找到它在哪个物理 block, 再去那个 block 里取, 比连续数组多一次间接寻址。这点开销不大, 但确实多了一步。\n分页换来的是显存利用率:\n- 分页: 每条请求只有最后一个 block 可能没写满, 最多浪费 bs−1 个 token 的位置, 平均大约 $\\text{bs}/2$ 个。\n- 连续预留: 事先不知道请求会写多长, 只能按 max_len 整段先占着, 没写到的位置全部浪费。vLLM 论文测得这种做法只有 20~40% 的 KV 显存装着真数据。\n显存利用率高之后:\n1. 同一张卡就能同时放更多条请求。\n2. 每步前向的 batch 更大。\n3. 吞吐 (每秒产出的 token 数) 跟着上去。\n把「block 大小」从 4 拖到 32, 右侧「浪费: 分页」会跟着涨: block 越大, 每条请求末尾那个没写满的 block 平均空得越多。block 越小浪费越少, 但页表越长, GPU kernel 要跳着读的碎片也越多。vLLM 默认取 16, 是在这两头之间折中。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" class="active" @click="push({ type: 'new', len: nextLen })">新请求 (prompt {{ nextLen }} token)</button>
        <button type="button" @click="reset(seed)">重置</button>
        <button type="button" @click="reset(seed + 1)">换一组</button>
        <span class="msg" :class="{ bad: st.bad }">{{ st.msg }}</span>
      </div>
      <LabSlider v-model="bsExp" label="block 大小" :min="2" :max="5" :format="(v) => 2 ** v" unit=" tok" />
      <LabSlider v-model="maxLen" label="连续预留的 max_len" :min="64" :max="256" :step="16" unit=" tok" />
      <LabSlider v-model="pos" label="查地址: 逻辑位置 pos" :min="0" :max="Math.max(0, (sel?.len || 1) - 1)" />
    </template>

    <div class="pool">
      <span
        v-for="(b, i) in st.blocks" :key="i" class="cell blk"
        :class="{ dim: focus !== null && b.owner !== null && b.owner !== focus, hit: i === addr.block }"
        :style="blockStyle(b)" :title="b.owner === null ? `物理块 ${i}: 空闲, 没有请求在用` : `物理块 ${i}: 属于请求 ${b.owner}, 已写入 ${b.fill}/${bs} 个 token 的 KV`"
      >{{ i }}</span>
    </div>

    <div class="reqs">
      <div
        v-for="r in st.live" :key="r.id" class="req" :class="{ sel: r.id === selId }"
        @mouseenter="hover = r.id" @mouseleave="hover = null"
      >
        <button type="button" class="name" :style="{ borderColor: color(r.id) }" @click="selId = r.id">
          请求 {{ r.id }} · {{ r.len }} tok
        </button>
        <span class="table mono">
          [<button
            v-for="(b, i) in r.table" :key="i" type="button" class="tb"
            :class="{ on: r.id === selId && i === addr.page }" :title="`这条请求的第 ${i} 页 KV 放在物理块 ${b}`"
            @click="selId = r.id; pos = i * bs"
          >{{ b }}</button>]
        </span>
        <span class="ops">
          <button type="button" @click="push({ type: 'dec', id: r.id, n: 1 })">decode +1</button>
          <button type="button" @click="push({ type: 'dec', id: r.id, n: 8 })">+8</button>
          <button type="button" @click="push({ type: 'fin', id: r.id })">结束</button>
        </span>
      </div>
      <p v-if="!st.live.length" class="lab-note">池子里没有请求了。点「新请求」接一条。</p>
    </div>

    <template #stats>
      <div class="kv"><span>浪费的 KV 位置: 分页 / 按 max_len 连续预留</span><b :class="wasteCls(st.pagedWaste, st.contWaste)">{{ pct(st.pagedWaste) }} / {{ pct(st.contWaste) }}</b></div>
      <div class="kv">
        <span>同一个池子按 max_len 连续预留, 这 {{ st.live.length }} 条请求装得下</span>
        <b :class="st.contFit < st.live.length ? 'bad' : ''">{{ st.contFit }} 条</b>
      </div>
      <div class="kv"><span>空闲物理块 / 总数</span><b>{{ st.free }} / {{ st.blocks.length }}</b></div>
      <div v-if="sel" class="kv">
        <span>请求 {{ sel.id }} 的逻辑位置 pos {{ addr.pos }} 实际在</span>
        <b>物理块 {{ addr.block }} · 槽 {{ addr.slot }}</b>
      </div>
      <div class="lab-note">
        <p>浪费率的算法:</p>
        <ul class="pts">
          <li><b>分页:</b> (已分配的 block 容量 − 实际写入的 token 数) ÷ 已分配容量。</li>
          <li><b>连续预留:</b> (每条请求各占 max_len − 实际写入) ÷ 总预留量。</li>
        </ul>
        <p>分页更省就标绿, 更费就标红。</p>
        <p>地址翻译只有一行: <code class="inline">block_table[pos // bs], pos % bs</code>。</p>
        <ul class="pts">
          <li>pos 除以 block 大小的整数部分是第几页, 查页表得到物理块。</li>
          <li>余数是块内第几个槽。</li>
        </ul>
        <p>例: bs=8, pos=37, 页表 [9,4,6,1,3]:</p>
        <ol class="pts">
          <li>37 // 8 = 4, 页表第 4 项是 3, 所以在物理块 3。</li>
          <li>37 % 8 = 5, 槽 5。</li>
        </ol>
        <ul class="pts">
          <li>几条请求交替 decode, 各自新要的块就交错排开, 一条请求的 KV 散落在池子各处。</li>
          <li>结束的请求把块还回空闲链表尾部 (与 m02 的 free_list 一致), 池子转过一圈后, 新请求拿到的全是零散块。</li>
        </ul>
        <p>有了页表, 逻辑上连续的 KV 不必在物理上连续, 所以池子里不会留下用不上的空洞。</p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { mulberry32, range, sum, clamp } from '@/utils/labmath.js'

const POOL = 384, CAP = 64 // 池容量 (token) 与单请求长度上限
const PALETTE = ['var(--accent)', 'var(--left)', 'var(--right)', 'var(--warn)', 'var(--danger)', 'var(--eye)']
const color = (id) => PALETTE[id % PALETTE.length]

const bsExp = ref(3)
const bs = computed(() => 2 ** bsExp.value)
const maxLen = ref(128)
const seed = ref(1)
const events = ref([])
const selId = ref(0)
const hover = ref(null)
const pos = ref(0)

// 第 i 个新请求的 prompt 长度只由 (seed, i) 决定, 重放时不会变
const lenOf = (i) => 5 + Math.floor(mulberry32(seed.value * 97 + i)() * 28)
const nextLen = computed(() => lenOf(events.value.filter((e) => e.type === 'new').length))
const push = (e) => { events.value = [...events.value, e]; if (e.type === 'new') selId.value = st.value.reqs.length - 1 }
const reset = (s) => {
  seed.value = s
  // 预置一段剧情: 接 4 条 → 交替 decode (新块交错) → 结束 1 号 → 再接 1 条
  const ev = range(4).map((i) => ({ type: 'new', len: lenOf(i) }))
  ev.push({ type: 'dec', id: 0, n: 8 }, { type: 'dec', id: 2, n: 8 }, { type: 'dec', id: 3, n: 8 }, { type: 'dec', id: 0, n: 8 })
  ev.push({ type: 'fin', id: 1 }, { type: 'new', len: lenOf(4) })
  events.value = ev; selId.value = 4; pos.value = 0
}
reset(1)

const st = computed(() => {
  const B = bs.value, nb = POOL / B, need = (len) => Math.ceil(len / B)
  const free = range(nb) // 空闲链表: 从头取, 还回来的接在尾部
  const reqs = []
  let msg = '', bad = false
  for (const e of events.value) {
    if (e.type === 'new') {
      const r = { id: reqs.length, len: e.len, table: [], live: false }
      reqs.push(r)
      bad = need(e.len) > free.length
      if (bad) { msg = `请求 ${r.id} 被拒: 需要 ${need(e.len)} 个 block, 只剩 ${free.length} 个`; continue }
      r.table = free.splice(0, need(e.len)); r.live = true
      msg = `请求 ${r.id} 拿到物理块 [${r.table.join(', ')}]`
      continue
    }
    const r = reqs[e.id]
    if (!r?.live) continue
    if (e.type === 'fin') { free.push(...r.table); msg = `请求 ${r.id} 结束, 归还 ${r.table.length} 个 block`; bad = false; r.table = []; r.live = false; continue }
    const len = Math.min(CAP, r.len + e.n), extra = need(len) - r.table.length
    bad = extra > free.length || len === r.len
    if (bad) { msg = len === r.len ? `请求 ${r.id} 已到长度上限 ${CAP}` : `请求 ${r.id} decode 失败: 没有空闲 block (真实系统此时会抢占)`; continue }
    // ★ 按需分配: 只有跨过 block 边界时才多要一个物理块
    r.table.push(...free.splice(0, extra)); r.len = len
    msg = extra ? `请求 ${r.id} 跨过 block 边界, 新增物理块 ${r.table.at(-1)}` : `请求 ${r.id} 写进最后一个 block 的空槽, 不需要新块`
  }
  const live = reqs.filter((r) => r.live)
  const blocks = range(nb).map(() => ({ owner: null, fill: 0 }))
  live.forEach((r) => r.table.forEach((b, i) => { blocks[b] = { owner: r.id, fill: Math.min(B, r.len - i * B) } }))
  const used = sum(live.map((r) => r.len)), alloc = sum(live.map((r) => r.table.length)) * B
  const reserved = live.length * maxLen.value
  return {
    reqs, live, blocks, msg, bad, free: free.length,
    pagedWaste: alloc ? (alloc - used) / alloc : 0,
    contWaste: reserved ? (reserved - sum(live.map((r) => Math.min(r.len, maxLen.value)))) / reserved : 0,
    contFit: Math.min(live.length, Math.floor(POOL / maxLen.value)), // 最好情况: 不算外部碎片
  }
})

// 两种分配方式互相比: 分页浪费得少就绿, 多就红, 一样就不上色
const wasteCls = (paged, cont) => (paged < cont ? 'good' : paged > cont ? 'bad' : '')
const sel = computed(() => st.value.live.find((r) => r.id === selId.value) || st.value.live[0] || null)
const focus = computed(() => hover.value ?? sel.value?.id ?? null)
watch(sel, (r) => { if (r) pos.value = clamp(pos.value, 0, r.len - 1) })
const addr = computed(() => {
  const r = sel.value
  if (!r) return { pos: 0, page: -1, block: -1, slot: 0 }
  const p = clamp(pos.value, 0, r.len - 1), page = Math.floor(p / bs.value)
  return { pos: p, page, block: r.table[page], slot: p % bs.value }
})

const pct = (x) => (x * 100).toFixed(1) + '%'
const blockStyle = (b) => {
  if (b.owner === null) return {}
  const c = color(b.owner), f = Math.round((b.fill / bs.value) * 100)
  return {
    borderColor: c,
    background: `linear-gradient(to top, color-mix(in srgb, ${c} 55%, transparent) ${f}%, color-mix(in srgb, ${c} 12%, transparent) ${f}%)`,
  }
}
</script>

<style scoped>
.pool { display: flex; flex-wrap: wrap; gap: 3px; margin-bottom: 14px; }
.blk { width: 26px; height: 26px; color: var(--text); }
.blk.hit { outline: 2px solid var(--text); outline-offset: 1px; }
.reqs { display: flex; flex-direction: column; gap: 6px; }
.req { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; padding: 4px 6px; border: 1px solid transparent; border-radius: var(--radius-sm); }
.req.sel { border-color: var(--border-strong); background: var(--bg-elev); }
.req button { font-size: 11px; min-height: 26px; padding: 2px 8px; }
.name { border-left-width: 4px; min-width: 116px; text-align: left; }
.table { font-size: 12px; color: var(--text-muted); display: flex; flex-wrap: wrap; gap: 2px; align-items: center; }
.tb { min-width: 26px; padding: 2px 4px !important; font-family: inherit; }
.tb.on { border-color: var(--accent); color: var(--accent); }
.ops { display: flex; gap: 4px; margin-left: auto; }
.msg { font-size: 12px; color: var(--text-muted); }
.msg.bad { color: var(--danger); }
</style>
