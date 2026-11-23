<!--
  布局位移下的点击 (对应 llm_agent/m17_computer_use/demo.py 的 Browser.layout / resolve 与 BrowserPolicy)。
  只讲一件事: 按 ref 点的是"那个元素", 按坐标点的是"那个位置"; 观察之后页面一挪, 坐标就点到别的东西上, 而且不报错。
-->
<template>
  <LabFrame
    title="Computer use — 看完再点, 中间页面挪了, 坐标会点到谁"
    sub="- 模拟什么: 操作图形界面的浏览器 Agent (computer use)。Agent 按「先观察快照、再下发点击动作」的两步循环取消指定订单 #1004, 但观察与执行之间存在异步布局偏移风险: demo 在 Agent 获取初始页面快照后, 于页面顶部异步插入一条促销横幅, 导致下方全部 DOM 元素整体下移。实验对比两种目标定位方式:
  - 按元素引用 ref 点击: 快照为每个可交互节点分配唯一引用标识 (如 e9), 浏览器在执行时按节点身份定位目标元素。
  - 按屏幕坐标 (x, y) 点击: 直接向视口内的绝对像素坐标派发点击事件。
- 图上是什么: 左右并排展示观察快照与执行时刻的真实页面:
  - 左栏 (观察快照): Agent 决策时看到的初始页面, 尚未加载顶部横幅, 包含 50px 高的标题栏、六行各 40px 高的订单条目及底部的 Delete account 按钮; 紫色边框标出当前待取消的目标订单。
  - 右栏 (执行时刻真实页面): 点击指令落盘时的实际页面, 顶部插入了黄色促销横幅, 下方元素发生垂直偏移; 右栏顶部小字显示当前发出的工具调用 click(ref=e9) 或 click_xy(200, 190)。
  - 坐标参考线与落点圆点: 采用坐标点击时, 水平横线指示点击纵坐标 y, 圆点标示实际落点; 被击中的元素若与目标不符则标红, 命中目标则标绿。
  - 虚线「视口底 y=360」: 标示当前屏幕可视区域下界, 超出该线的元素需滚动页面后方可见。
- 可以做什么: 提供三项交互操作:
  - 切换「按 ref 点」与「按坐标点」: 分别对应底层工具 browser_click 与 browser_click_xy。
  - 拖动「横幅高度」滑杆: 调节观察快照后顶部异步插入横幅的像素高度, 设为 0 表示页面未发生偏移。
  - 点击左栏任意订单行: 切换本次任务要求取消的目标订单。
- 应该看到: 对比两种定位方式的容错表现与右侧读数:
  - 「按 ref 点」模式: 无论横幅高度如何变化均能准确命中目标订单, 因为 ref 绑定的是 DOM 节点身份而非几何坐标。
  - 切换至「按坐标点」且横幅高度为 80px、目标为 #1004: 原快照中 y=190 的中心位置在页面下移后变为订单 #1002, Agent 误入 #1002 详情页将其取消, 返回列表刷新快照后才在第二轮取消 #1004。
  - 右侧「实际点中的元素」: 显示首轮点击指令在偏移后落中的真实页面节点。
  - 「被取消的订单」: 任务结束时环境中实际处于已取消状态的订单列表, 出现目标外订单即表明发生了误操作副作用。
  - 「agent 最终回答」: Agent 在结束时输出的自然语言汇报。
  - 「只看回答文字的 grader」与「按环境终态判分」: 前者仅凭回复文本判分, 后者核查数据库中的实际取消集合; 两者结论相悖即构成「静默失败」——运行无异常报错且回复看似完全正常, 但外部环境已被错误修改。"
    module="llm_agent/m17"
    run="python -m llm_agent.m17_computer_use.demo"
    :challenge="{
      ask: '默认「按 ref 点」模式下仅取消了目标订单 [1004]。保持横幅高度 80px 与目标 #1004 不变, 切换为「按坐标点」。请先预测: Agent 最终会输出什么回复? 仅检查最终回复文本的 grader 会判定其通过吗?',
      answer: 'Agent 最终仍会输出「订单 #1004 已取消。」, 与正常执行时完全一致, 因此仅检查回复文本的 grader 会误判为通过。\n- 误操作发生过程: 像素坐标 (200, 190) 基于无横幅的初始快照计算得出 (对应 #1004 行中心); 执行点击时顶部已插入 80px 横幅, 原坐标位置恰好被下移后的 #1002 占据。Agent 随之进入 #1002 详情页并完成取消与确认操作; 待返回列表获取新快照后才重新点中 #1004, 导致 demo 实测环境中实际被取消的订单变为 [1002, 1004]。\n- 静默无报错原因: 偏移后的坐标位置上恰好存在另一个合法可点击的订单行, 浏览器底层事件派发完全成功, 无法感知该坐标已偏离 Agent 的预期目标。\n- 两项工程防御对策:\n  - 优先按元素引用 ref 操作: 浏览器按节点身份寻址, 布局平移不影响命中目标; 若节点已被卸载则会显式抛出 stale ref 错误而非误点邻近元素。\n  - 按环境终态判分: 评测器直接核查后端订单状态集合, 而非采信模型的自然语言自述。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: by === 'ref' }" @click="by = 'ref'">按 ref 点 browser_click</button>
        <button type="button" :class="{ active: by === 'xy' }" @click="by = 'xy'">按坐标点 browser_click_xy</button>
      </div>
      <LabSlider v-model="shift" label="横幅高度 (观察之后插入)" :min="0" :max="MAX_SHIFT" :step="10" unit=" px" />
    </template>

    <svg :viewBox="`0 0 400 ${H}`" class="page" role="group" aria-label="观察到的布局与点击时的布局">
      <text x="0" y="12" class="hd">观察到的快照 (scrollY=0)</text>
      <text x="210" y="12" class="hd">点下去那一刻 · <tspan class="aim-t">{{ by === 'xy' ? `click_xy(200, ${clickY})` : `click(ref=${refOf(target)})` }}</tspan></text>
      <g v-for="(n, i) in seen" :key="'s' + i" :transform="`translate(0, ${T + n.top})`"
        :class="['el', 'focus-inset', n.kind, { target: n.no === target, pick: n.no }]"
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
      <div class="kv"><span>实际点中的元素</span><b :class="landed.no === target ? 'good' : 'bad'">{{ landed.label }}</b></div>
      <div class="kv"><span>被取消的订单 (环境终态)</span><b :class="outcome.cancelled.length === 1 ? 'good' : 'bad'">{{ outcome.cancelled.length ? '[' + outcome.cancelled.join(', ') + ']' : '—' }}</b></div>
      <div class="kv"><span>agent 最终回答</span><b class="small">{{ outcome.final }}</b></div>
      <div class="kv"><span>只看回答文字的 grader</span><b :class="outcome.textPass ? '' : 'bad'">{{ outcome.textPass ? '通过' : '不通过' }}</b></div>
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
const by = ref('ref') // 默认是对照: ref 点得中; 切到坐标才出事
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
const MAX_SHIFT = 200
// 画布高度按最高的横幅定死: 拖滑杆时只有右栏的元素下移, 整张图不缩放
const H = (() => { const l = stack(MAX_SHIFT).at(-1); return T + l.top + l.h + 6 })()
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
