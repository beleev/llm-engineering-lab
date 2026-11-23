<!--
  跨窗口接力的确定性模拟 (对应 llm_agent/m60_long_horizon/demo.py 的 session / single_long / run)。
  只讲一件事: 同样的总上下文预算下, 把状态写到磁盘上比让一个长会话靠压缩硬撑做完更多功能。
  常量、FLAKY / MOD 规则和会话循环与 Python 一致, 所以默认设置下四行读数逐项对得上。
-->
<template>
  <LabFrame
    title="跨窗口接力 — 任务比上下文窗口大的时候, 状态放在哪"
    sub="- 模拟什么: 一个有 N 个功能的项目, 脚本化的模型每个会话只装得下有限的上下文。功能 i 改动共享模块 i % 4, 实现它需要知道同模块已通过功能的约束; 约束的来源只有两处, 本会话自己做过 (在上下文里) 或清单里记过 (在磁盘上), 两处都没有就把那些功能改坏。i % 3 == 2 的功能第一次写出来是半成品, 只有端到端跑一遍测试才会发现。不用随机数。
- 图上是什么: 一排方块是 N 个功能, 按编号从左到右:
  - 绿色: 真正通过测试。
  - 红色: 记成完成但其实不通过 (虚报)。
  - 灰色: 预算用完还没做。
  下方一行是会话时间轴, 每一格是一个会话, 格子高度正比于这个会话花掉的上下文。
- 可以做什么: 按界面顺序:
  - 四个策略按钮: 单个长会话只靠压缩 / 新会话但没有交接状态 / 新会话加进度文件但不跑测试 / 新会话加进度文件加清单加测试。
  - 「功能数」滑杆: 改变项目规模。
  - 「会话容量」滑杆: 一个会话装得下多少上下文。
  - 点击任意一个功能方块: 下方说明它属于哪个共享模块、是不是半成品、最终为什么是这个状态。
- 应该看到: 默认设置 (12 个功能, 容量 8000, 预算 40000) 与 m60 demo [1] 的输出一致:
  - 接力做法 12 个全绿, 虚报 0, 回归 0, 用掉 17400。
  - 单个长会话只有 6 个绿, 重做 16 次、回归 19 次, 预算几乎用光。
  - 不跑测试那一档全部标成完成, 却有 7 个红: 半成品之外, 建在坏状态上的功能也跟着红。
  - 把会话容量拖到 1600, 接力做法仍然全绿, 但会话数升到和功能数相同, 用掉的上下文从 17400 升到 21000。"
    module="llm_agent/m60"
    run="python -m llm_agent.m60_long_horizon.demo"
    :challenge="{
      ask: '「新会话, 没有交接状态」和「单个长会话, 只靠压缩」在默认设置下通过的功能数一样多。先预测: 把会话容量调大, 哪一条会先追上接力做法?',
      answer: '两条都会, 但原因不同, 而且都只是因为任务被塞进了一个窗口。\n- 没有交接状态的那一条, 进度上限就是一个会话装得下的功能数: 容量够大时第一个会话一口气做完全部功能, 后面的会话只是重做一遍。它从来没有真的接力过。\n- 只靠压缩的那一条, 容量够大时根本不触发压缩, 已完成清单和约束都还在上下文里, 自然也不会重做和改坏。\n- 所以这个实验台量到的不是「压缩不好」, 而是「窗口装不下任务时, 能接着干的信息必须在上下文之外」。容量 1600 时接力做法仍然全绿, 另外两条掉到一个会话的量。\n代价也要照看: 容量越小会话越多, 每个会话重新读一遍进度文件的开销要重复付, 总上下文从 16600 升到 21000。',
    }"
  >
    <template #controls>
      <div class="row">
        <span class="tip">策略:</span>
        <button v-for="m in MODES" :key="m.id" type="button" :class="{ active: mode === m.id }" @click="mode = m.id">{{ m.label }}</button>
      </div>
      <LabSlider v-model="n" label="功能数" :min="6" :max="20" :step="2" unit=" 个" />
      <LabSlider v-model="cap" label="会话容量" :min="1600" :max="20000" :step="400" />
    </template>

    <div class="cells">
      <button v-for="i in n" :key="i" type="button" class="cell" :class="cls(i - 1)" @click="sel = i - 1">{{ i - 1 }}</button>
    </div>
    <p class="hint mono">会话时间轴 ({{ res.sessions.length }} 个会话, 格子高度正比于该会话花掉的上下文)</p>
    <div class="bars">
      <i v-for="(s, k) in res.sessions" :key="k" :style="{ height: Math.max(4, s / maxSession * 34) + 'px' }" />
    </div>
    <p class="hint mono">{{ detail }}</p>

    <template #stats>
      <div class="kv"><span>真正通过的功能</span><b :class="res.passing.size === n ? 'good' : 'bad'">{{ res.passing.size }} / {{ n }}</b></div>
      <div class="kv"><span>虚报完成</span><b :class="res.broken.size === 0 ? 'good' : 'bad'">{{ res.broken.size }}</b></div>
      <div class="kv"><span>回归 / 重做</span><b :class="res.regress + res.redo === 0 ? 'good' : 'bad'">{{ res.regress }} / {{ res.redo }}</b></div>
      <div class="kv"><span>用掉的上下文</span><b :class="res.used < BUDGET ? 'good' : 'bad'">{{ res.used }} / {{ BUDGET }}</b></div>
      <p class="lab-note">确定性模拟, 不是实测: 实现与测试的开销是常数, 回归只在同一个共享模块内扩散, 压缩按「只留最近 2 条记录」建模。</p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'

// 与 m60 demo 的常量相同
const READ = 400, IMPL = 900, TEST = 300, COMPACT = 500, BUDGET = 40000
const MOD = (i) => i % 4
const FLAKY = (i) => i % 3 === 2
const MODES = [
  { id: 'single', label: '单个长会话' },
  { id: 'bare', label: '新会话, 无交接' },
  { id: 'untested', label: '进度文件, 不测' },
  { id: 'handoff', label: '进度 + 清单 + 测试' },
]

const mode = ref('handoff'), n = ref(12), cap = ref(8000), sel = ref(2)

function simulate(mode, N, CAP) {
  const p = { passing: new Set(), broken: new Set(), notes: new Set(), regress: 0, redo: 0, used: 0, sessions: [] }
  if (mode === 'single') {
    let done = [], used = 0
    while (p.used < BUDGET && p.passing.size < N) {
      const todo = [...Array(N).keys()].filter((i) => !done.includes(i))
      if (!todo.length) { done = done.slice(-2); continue }
      const i = todo[0]
      if (used + IMPL + TEST > CAP) { used = Math.floor(CAP / 4); done = done.slice(-2); p.used += COMPACT; continue }
      if (p.passing.has(i)) p.redo++
      // ★ 压缩之后上下文里只剩最近 2 条, 同模块的约束丢了, 那些已通过的功能被改坏
      const hit = [...p.passing].filter((j) => MOD(j) === MOD(i) && !done.includes(j))
      const step = IMPL + TEST + (FLAKY(i) ? IMPL / 2 : 0) + (IMPL / 2) * hit.length
      if (p.used + step > BUDGET) break
      p.regress += hit.length
      used += step; p.used += step
      p.passing.add(i); done.push(i)
    }
    p.sessions = [p.used]
    return p
  }
  const notes = mode !== 'bare', tests = mode !== 'untested'
  while (p.passing.size + p.broken.size < N) {
    const limit = Math.min(CAP, BUDGET - p.used)
    if (limit < IMPL) break // 一个功能也放不进去, 再开会话只是白读一遍进度文件
    let used = notes ? READ : 0
    const self = new Set()
    while (used + IMPL + (tests ? TEST : 0) <= limit) {
      const seen = new Set([...(notes ? p.notes : []), ...self])
      const todo = [...Array(N).keys()].filter((i) => !seen.has(i))
      if (!todo.length) break
      const i = todo[0]
      if (p.passing.has(i)) p.redo++
      used += IMPL
      const known = new Set([...self, ...(notes ? p.notes : [])])
      const hit = [...p.passing].filter((j) => MOD(j) === MOD(i) && !known.has(j))
      p.regress += hit.length
      if (tests) {
        used += TEST + (FLAKY(i) ? IMPL / 2 : 0) + (IMPL / 2) * hit.length
        p.passing.add(i); p.broken.delete(i)
      } else {
        hit.forEach((j) => { p.passing.delete(j); p.broken.add(j) })
        if (FLAKY(i) || [...p.broken].some((j) => MOD(j) === MOD(i))) p.broken.add(i)
        else p.passing.add(i)
      }
      self.add(i)
      if (notes) p.notes.add(i)
    }
    if (used === 0 || (notes && used === READ)) break
    p.used += used; p.sessions.push(used)
  }
  return p
}

const res = computed(() => simulate(mode.value, n.value, cap.value))
const maxSession = computed(() => Math.max(...res.value.sessions, 1))
const cls = (i) => (res.value.passing.has(i) ? 'ok' : res.value.broken.has(i) ? 'bad' : 'dim')
const detail = computed(() => {
  const i = Math.min(sel.value, n.value - 1)
  const where = res.value.passing.has(i) ? '真正通过' : res.value.broken.has(i) ? '记成完成但不通过' : '预算用完还没做'
  const why = res.value.broken.has(i)
    ? (FLAKY(i) ? '它是半成品, 没有端到端测过' : `同模块 ${MOD(i)} 里有记成完成的坏状态, 它建在上面`)
    : res.value.passing.has(i) ? '端到端测过, 并且同模块的约束拿得到' : '前面的功能把预算花掉了'
  return `功能 ${i}: 共享模块 ${MOD(i)}, ${FLAKY(i) ? '第一次写出来是半成品' : '一次就能写对'}; ${where} — ${why}`
})
</script>

<style scoped>
.cells { display: flex; flex-wrap: wrap; gap: 4px; }
.cell { width: 30px; height: 30px; border-radius: var(--radius-sm); border: 1px solid var(--border); background: var(--bg); color: var(--text-muted); font-size: 11px; min-height: 0; padding: 0; }
.cell.ok { background: var(--left); border-color: var(--left); color: var(--bg); }
.cell.bad { background: var(--danger); border-color: var(--danger); color: var(--bg); }
.cell.dim { color: var(--text-dim); }
.bars { display: flex; align-items: flex-end; gap: 3px; height: 38px; margin-bottom: 4px; }
.bars i { width: 12px; background: var(--accent); border-radius: 2px 2px 0 0; }
.hint { font-size: 11px; color: var(--text-muted); margin: 8px 0 2px; line-height: 1.6; }
.tip { font-size: 12px; color: var(--text-muted); }
</style>
