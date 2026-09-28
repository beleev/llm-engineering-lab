<!--
  布局位移下的点击 (对应 llm_agent/m17_computer_use/demo.py 的 Browser.layout / resolve 与 BrowserPolicy)。
  只讲一件事: 按 ref 点的是"那个元素", 按坐标点的是"那个位置"; 观察之后页面一挪, 坐标就点到别的东西上, 而且不报错。
-->
<template>
  <LabFrame
    title="Computer use — 看完再点, 中间页面挪了"
    sub="左边是 agent 观察到的快照, 右边是点下去那一刻的页面。右边顶部多了一条异步加载的促销横幅。
      拖横幅高度, 切换按 ref / 按坐标点击, 看点击落在哪。点左边任意一行订单, 换一个要取消的目标。"
    module="llm_agent/m17"
    run="python -m llm_agent.m17_computer_use.demo"
    :challenge="{
      ask: '按坐标点, 横幅 80px, 目标 #1004。agent 最后会说什么? 一个只看最终回答的 grader 会判它通过吗?',
      answer: '它会说「订单 #1004 已取消。」, 和点对时一字不差, grader 判通过。\n- 实际发生的: (200, 190) 落在 #1002 那一行。agent 进了 #1002 的详情页, 翻页、取消、确认, 把 #1002 取消了。回到列表拿到新快照, 才点对 #1004。demo 实测被取消的是 [1002, 1004]。\n- 不会报错: 那个位置上确实有一个可点的元素。\n两条对策:\n- 按 ref 点: 浏览器按元素身份找目标。\n- 按终态判分: 评测时数一数到底取消了几单。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: by === 'ref' }" @click="by = 'ref'">按 ref 点 browser_click</button>
        <button type="button" :class="{ active: by === 'xy' }" @click="by = 'xy'">按坐标点 browser_click_xy</button>
      </div>
      <LabSlider v-model="shift" label="横幅高度 (观察之后插入)" :min="0" :max="200" :step="10" unit=" px" />
    </template>

    <svg :viewBox="`0 0 400 ${H}`" class="page" role="group" aria-label="观察到的布局与点击时的布局">
      <text x="0" y="12" class="hd">观察到的快照 (scrollY=0)</text>
      <text x="210" y="12" class="hd">点下去那一刻 · <tspan class="aim-t">{{ by === 'xy' ? `click_xy(200, ${clickY})` : `click(ref=${refOf(target)})` }}</tspan></text>
      <g v-for="(n, i) in seen" :key="'s' + i" :transform="`translate(0, ${T + n.top})`"
        :class="['el', n.kind, { target: n.no === target, pick: n.no }]"
        :tabindex="n.no ? 0 : -1" :role="n.no ? 'button' : null" @click="n.no && (target = n.no)" @keydown.enter="n.no && (target = n.no)">
        <rect x="0" y="1" width="190" :height="n.h - 2" rx="3" />
        <text x="8" :y="n.h / 2 + 4">{{ n.label }}</text>
      </g>
      <g v-for="(n, i) in live" :key="'l' + i" :transform="`translate(210, ${T + n.top})`" :class="['el', n.kind, { landed: n === landed, ok: n === landed && n.no === target }]">
        <rect x="0" y="1" width="190" :height="n.h - 2" rx="3" />
        <text x="8" :y="n.h / 2 + 4">{{ n.label }}</text>
      </g>
      <line x1="0" x2="400" :y1="T + VIEW" :y2="T + VIEW" class="fold" />
      <text x="4" :y="T + VIEW + 12" class="fold-t">视口底 y=360</text>
      <line v-if="by === 'xy'" x1="0" x2="400" :y1="T + clickY" :y2="T + clickY" class="aim" />
      <circle :cx="210 + 100" :cy="T + (by === 'xy' ? clickY : landed.top + landed.h / 2)" r="5" class="dot" />
    </svg>

    <template #stats>
      <div class="kv"><span>实际点中</span><b :class="landed.no === target ? 'good' : 'bad'">{{ landed.label }}</b></div>
      <div class="kv"><span>被取消的订单</span><b :class="outcome.cancelled.length === 1 ? 'good' : 'bad'">{{ outcome.cancelled.length ? '[' + outcome.cancelled.join(', ') + ']' : '—' }}</b></div>
      <div class="kv"><span>agent 最终回答</span><b class="small">{{ outcome.final }}</b></div>
      <div class="kv"><span>只看回答的 grader</span><b :class="outcome.textPass ? '' : 'bad'">{{ outcome.textPass ? '通过' : '不通过' }}</b></div>
      <div class="kv"><span>按环境终态判分</span><b :class="outcome.statePass ? 'good' : 'bad'">{{ outcome.statePass ? '通过' : '失败' }}</b></div>
      <p class="lab-note">{{ outcome.note }}</p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'

// 与 Browser.nodes() 的 /orders 页相同: 标题 50px, 每行订单 40px, 最后是 Delete account; 视口高 360
const ORDERS = [[1001, 'shipped'], [1002, 'processing'], [1003, 'delivered'], [1004, 'processing'], [1005, 'shipped'], [1006, 'delivered']]
const VIEW = 360, T = 22
const by = ref('xy')
const shift = ref(80)
const target = ref(1004)

// 竖直堆叠: 前面多一个元素, 后面全部下移 (Browser.layout)
function stack(banner) {
  const els = []
  if (banner > 0) els.push({ kind: 'promo', label: 'Holiday sale: 20% off', h: banner })
  els.push({ kind: 'heading', label: 'Your orders', h: 50 })
  for (const [no, st] of ORDERS) els.push({ kind: 'order', no, label: `Order #${no} · ${st}`, h: 40 })
  els.push({ kind: 'danger', label: 'Delete account', h: 40 })
  let top = 0
  for (const e of els) { e.top = top; top += e.h }
  return els
}
const seen = stack(0)
const live = computed(() => stack(shift.value))
const H = computed(() => { const l = live.value[live.value.length - 1]; return Math.max(T + VIEW + 18, T + l.top + l.h + 6) })
const refOf = (no) => `e${ORDERS.findIndex((o) => o[0] === no) + 6}` // demo 里订单行的 ref 是 e6..e11
const clickY = computed(() => { const n = seen.find((e) => e.no === target.value); return n.top + n.h / 2 })

// ★ ref 找的是元素身份; 坐标找的是此刻 [top, top+h) 覆盖该 y 的元素 (Browser.resolve)
const landed = computed(() => by.value === 'ref'
  ? live.value.find((e) => e.no === target.value)
  : live.value.find((e) => e.top <= clickY.value && clickY.value < e.top + e.h))

const outcome = computed(() => {
  const l = landed.value, t = target.value, final = `订单 #${t} 已取消。`
  if (l.no === t) return { cancelled: [t], final, textPass: true, statePass: true, note: by.value === 'ref' ? 'ref 绑在元素身份上, 横幅多高都点得中。demo [2]: ref + 位移, 只取消 [1004]。' : '横幅还没把目标挤走, 坐标碰巧仍在目标行里。' }
  if (l.kind === 'order') {
    const cancelled = [l.no, t].sort((a, b) => a - b)
    return { cancelled, final, textPass: true, statePass: false, note: `落在 #${l.no} 那一行: agent 进了它的详情页, 翻页、取消、确认。回到列表拿到新快照才点对 #${t}。demo [2] 在 80px、目标 #1004 时实测为 [1002, 1004]。` }
  }
  if (l.kind === 'heading') return { cancelled: [t], final, textPass: true, statePass: true, note: '点在标题上, 什么也没发生。下一张快照是新的, agent 重新点对, 只是多花一步。' }
  return { cancelled: [], final: '(偏航, 未完成)', textPass: false, statePass: false, note: '点进了促销横幅, 页面跳到 /promo, 任务偏离。demo 没有演示这一支, 这里只标出落点。' }
})
</script>

<style scoped>
.page { width: 100%; max-width: 520px; }
.hd { font-size: 11px; fill: var(--text-muted); }
.el rect { fill: var(--bg-elev); stroke: var(--border); }
.el text { font-size: 11px; fill: var(--text); font-family: "SF Mono", Menlo, monospace; pointer-events: none; }
.el.heading text { font-weight: 600; }
.el.promo rect { fill: color-mix(in srgb, var(--warn) 25%, transparent); stroke: var(--warn); }
.el.danger text { fill: var(--danger); }
.el.pick { cursor: pointer; }
.el.pick:focus-visible rect { stroke: var(--accent); stroke-width: 2; }
.el.target rect { stroke: var(--accent); stroke-width: 2; }
.el.landed rect { stroke: var(--danger); stroke-width: 2; fill: color-mix(in srgb, var(--danger) 15%, transparent); }
.el.landed.ok rect { stroke: var(--left); fill: color-mix(in srgb, var(--left) 15%, transparent); }
.fold { stroke: var(--text-dim); stroke-dasharray: 4 3; }
.fold-t { font-size: 10px; fill: var(--text-dim); }
.aim { stroke: var(--accent); stroke-width: 1.5; }
.aim-t { font-size: 10px; fill: var(--accent); font-family: "SF Mono", Menlo, monospace; }
.dot { fill: var(--accent); }
.small { font-size: 12px !important; text-align: right; }
</style>
