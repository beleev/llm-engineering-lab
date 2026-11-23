<!-- Radix 前缀缓存实验台 (对应 llm_infer/m05_radix_cache): JS 里老老实实建一棵基数树, 逐条请求重放。 -->
<template>
  <LabFrame
    title="Radix cache — 公共前缀的 KV 只算一次, 任意长度都能命中"
    sub="- 模拟什么: 实际线上流量中大量请求共享相同的前缀序列 (如同一段 system prompt、同一组 few-shot 示例或多轮对话中的历史上下文)。由于自回归模型中任一位置 token 的 KV 仅依赖其前序 token, 相同前缀对应的 KV 完全一致, 只需计算一次即可跨请求复用, 新请求仅需对后续差异部分执行 prefill。SGLang 采用基数树 (radix tree) 组织已缓存的 token 序列, 每条树边存储一段连续的 token 子序列而非单个 token; 当缓存容量耗尽时, 按 LRU (最近最少使用) 策略从叶子节点向上逐级驱逐。新请求到达时的匹配流程如下:
  1. 从根节点出发沿树边向下逐段比对, 直至无法继续匹配为止。
  2. 沿途匹配成功的路径即为最长公共前缀, 其对应的 KV cache 直接复用。
- 图上是什么: 上方为当前请求的 token 序列, 下方按行展示基数树的各个节点:
  - 顶部 token 行: 展示当前请求的完整输入序列; 绿色底色标示在树中命中的前缀 (直接复用 KV, 免除前向计算), 紫色底色标示未命中并新插入树中的后缀部分。
  - 下方树节点列表: 每一行对应树中的一个节点, 缩进层级越深代表距根节点越远, 行内展示从父节点指向该节点的边所存储的 token 片段; 行末标注该边的 token 长度、最近一次访问它的请求序号以及是否为叶子节点。
  - 带底框高亮的行: 当前请求匹配路径所经过的节点。
  - 黄色底色的行: 标示刚刚发生边分裂的节点, 即当前请求的公共前缀恰好终止于某条树边中途时, 系统将该边切分为前后两段以使公共前缀准确落在树节点边界上。
- 可以做什么: 通过三个预设按钮切换请求负载模式, 并配合两个滑杆与回放控件观察树结构演化:
  - 「共享 system prompt」: 6 条请求均以同一段 system prompt 开头。
  - 「few-shot」: 多条请求携带相同的上下文示例前缀。
  - 「多轮对话」: 两位用户交替发起多轮提问, 每一轮请求均拼接该用户此前全部轮次的对话历史。
  - 拖动「缓存容量」修改基数树最多可驻留的 KV token 总数, 调小容量即可触发 LRU 驱逐; 拖动「对比: block hash 的 block」修改对照方案的分块大小。
  - 使用「下一步」与「时间轴」逐条回放请求流 (初始默认停留在最后一条请求, 点击播放可从头观察基数树的构建与分裂过程)。
  - 点击任意树节点行, 下方会展开从根到该节点串联而成的完整前缀序列。
- 应该看到: 在「多轮对话」预设下, 同一用户第 2、3 轮请求的历史前缀几乎全部命中; 将「缓存容量」调小至 8 左右时, 可见驱逐严格从叶子节点开始执行, 且当前活跃请求正在引用的路径节点不会被驱逐。右侧读数含义如下:
  - 「本条命中 / 长度」表示当前请求可免除 prefill 计算的命中 token 数与总长度。
  - 「同样前缀, block hash 只能命中」展示按整块计算哈希值方案 (如 vLLM 的前缀缓存机制) 的命中 token 数; 由于该方案只能以完整 block 为粒度命中, 当前缀长度非块大小的整数倍时会损失尾部不足一块的部分, 命中数低于基数树时标红。
  - 「累计命中率」为截至当前请求全部输入 token 中的累计命中占比, 等同于累计节省的 prefill 计算比例。
  - 「树里 token / 容量 · 累计驱逐」展示当前驻留缓存量、总容量上限及累计被驱逐的 token 总数。"
    module="llm_infer/m05"
    run="python -m llm_infer.m05_radix_cache.demo"
    :challenge="{
      ask: '当缓存容量不足时, 如果直接驱逐树中某个最近最少使用的内部中间节点, 会产生什么后果?',
      answer: '该中间节点的所有后代子树都会随之失效, 因此缓存驱逐只能从叶子节点执行。\n根本原因在于因果注意力中 KV 的前缀依赖性: 任一位置 token 的 KV 严格依赖其前方完整的 token 序列。内部节点对应树边上的 token 片段构成了其所有后代节点共享前缀的一部分; 一旦将该中间节点的 KV 释放, 即便后代节点的 KV 仍驻留在显存中, 后续新请求从根节点向下匹配至该断开处便无法继续推进, 所有后代节点的 KV 都会沦为永远无法被命中的孤立无效数据。因此合法的驱逐顺序只能按如下两步推进:\n1. 优先在当前所有叶子节点中按 LRU 驱逐最久未使用的叶子。\n2. 当某内部节点的全部子节点均被驱逐后, 该节点退化为新的叶子节点, 才允许参与后续驱逐。\n此外系统还设有引用计数保护机制: 当前正在执行推理的活跃请求所经过的路径节点会被加锁 (ref_count 大于 0), 期间禁止被驱逐, 以免请求尚未完成生成时其前缀 KV 被回收覆写。\n右侧「block hash 只能命中」读数展示了与分块哈希前缀缓存方案的粒度对比:\n- 基数树 (radix tree): 匹配与复用粒度精确到单个 token, 只要公共前缀长度为 $n$ 个 token 即可完整复用 $n$ 个。\n- 分块哈希 (block hash, 如 vLLM 采用的方案): 以固定大小的物理 block 为单位计算前缀哈希, 仅能复用 $\\lfloor n/\\text{bs} \\rfloor \\cdot \\text{bs}$ 个 token。例如公共前缀长 100 个 token、块大小 $\\text{bs} = 16$ 时, 仅能命中前 96 个 token, 尾部剩余 4 个 token 仍需重新计算。分块越大尾部截断损失越多; 在 Python demo 的合成请求负载下, 基数树的命中率为 44.8%, 而块大小设为 4 时的分块哈希命中率仅为 33.5%。',
    }"
  >
    <template #controls>
      <div class="row">
        <button v-for="(p, k) in PRESETS" :key="k" type="button" :class="{ active: preset === k }" @click="preset = k">{{ p.name }}</button>
      </div>
      <LabSlider v-model="cap" label="缓存容量" :min="8" :max="64" unit=" tok" />
      <LabSlider v-model="bsExp" label="对比: block hash 的 block" :min="2" :max="4" :format="(v) => 2 ** v" unit=" tok" />
      <StepPlayer :stepper="stepper" :label="`请求 ${stepper.step.value + 1} / ${reqs.length}`" />
    </template>

    <p class="cap">当前请求 (第 {{ stepper.step.value + 1 }} 条) 的 token:</p>
    <div class="strip">
      <span v-for="(tk, i) in cur.ids" :key="i" class="cell tok" :class="i < cur.hit ? 'ok' : 'on'">{{ tk }}</span>
    </div>

    <p class="cap">radix tree ({{ tree.rows.length }} 个节点, 一行一个节点, 缩进 = 深度):</p>
    <div class="tree" role="tree">
      <div v-for="r in tree.rows" :key="r.id" class="node" :style="{ paddingLeft: r.depth * 22 + 'px' }">
        <button
          type="button" class="edge" role="treeitem"
          :class="{ path: selPath.has(r.id), sel: r.id === selId }" @click="selId = r.id === selId ? -1 : r.id"
        >
          <span class="twig">└</span>
          <span v-for="(tk, i) in r.tokens" :key="i" class="cell tok" :class="rowClass(r)">{{ tk }}</span>
          <span class="meta mono">{{ r.tokens.length }} tok · 上次用于 #{{ r.last + 1 }}{{ r.leaf ? ' · 叶' : '' }}</span>
        </button>
      </div>
      <p v-if="!tree.rows.length" class="lab-note">当前设定的缓存容量过小, 树中节点已全部被驱逐; 请将「缓存容量」适当调大。</p>
    </div>
    <p class="legend">
      <span class="cell tok ok">tok</span> 这条请求命中的前缀: KV 直接复用, 不用前向
      <span class="cell tok on">tok</span> 这条请求新插进树里的部分: 要 prefill
      <span class="cell tok hot">tok</span> 刚被从中间劈开的边: 前缀停在了它中间
    </p>
    <p v-if="selNode" class="lab-note sel-note">
      选中节点代表的完整前缀 (从根到它一路上的 token, 共 {{ selPrefix.length }} 个): <span class="mono">{{ selPrefix.join(' ') }}</span>
    </p>

    <template #stats>
      <div class="kv"><span>这条请求: 命中的 token / 总长度</span><b :class="cur.hit ? 'good' : ''">{{ cur.hit }} / {{ cur.ids.length }}</b></div>
      <div class="kv"><span>同样的前缀, 按整块哈希 (block hash) 只能命中</span><b :class="blockHit < cur.hit ? 'bad' : ''">{{ blockHit }}</b></div>
      <div class="kv"><span>累计命中率 (到目前为止省掉的 prefill 比例)</span><b>{{ (tree.hitRate * 100).toFixed(1) }}%</b></div>
      <div class="kv"><span>树里的 token 数 / 容量 · 累计驱逐的 token 数</span><b :class="tree.evicted ? 'bad' : ''">{{ tree.size }} / {{ cap }} · {{ tree.evicted }}</b></div>
      <div class="lab-note">
        <p>这一条发生了什么: {{ cur.msg }}</p>
        <p>匹配只有三步:</p>
        <ol class="pts">
          <li>在当前节点的孩子里按「子边的第一个 token」找下一条边。</li>
          <li>沿这条边逐个比对, 求最长公共前缀。</li>
          <li>如果在边的中间比对失败, 就把这条边劈成两段 (<code class="inline">_split</code>), 让命中的部分恰好是一个节点。</li>
        </ol>
        <p>缓存驱逐仅在叶子节点中按最近最少使用 (LRU) 顺序执行; 当前活跃请求匹配路径上的节点处于加锁状态 (ref_count 大于 0), 不会被驱逐。</p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import StepPlayer from '@/components/lab/StepPlayer.vue'
import { useStepper } from '@/composables/useStepper.js'

const SYS = ['你', '是', '严谨', '的', '助手', '。']
const SHOT = ['Q', '1+1', 'A', '2', ';', 'Q', '2+3', 'A', '5', ';']
const A1 = ['sys', 'A:你好'], A2 = [...A1, '→嗨!', 'A:讲个', '笑话'], A3 = [...A2, '→从前…', 'A:再', '来', '一个']
const B1 = ['sys', 'B:在吗'], B2 = [...B1, '→在', 'B:帮我', '改', 'bug'], B3 = [...B2, '→贴代码', 'B:如下', '…']
const PRESETS = {
  system: { name: '共享 system prompt', reqs: [
    [...SYS, '翻译', ':', 'hello'], [...SYS, '翻译', ':', 'world'], [...SYS, '总结', ':', '这篇', '文章'],
    [...SYS, '翻译', ':', 'hello', 'world'], [...SYS, '写', '一首', '诗'], [...SYS, '总结', ':', '这段', '代码'] ] },
  fewshot: { name: 'few-shot', reqs: [
    [...SHOT, 'Q', '7+8', 'A'], [...SHOT, 'Q', '3+9', 'A'], [...SHOT.slice(0, 5), 'Q', '4+4', 'A', '8', ';', 'Q', '9+1', 'A'],
    [...SHOT, 'Q', '7+8', 'A'], [...SHOT, 'Q', '6+6', 'A'], [...SHOT.slice(0, 5), 'Q', '4+4', 'A', '8', ';', 'Q', '5+5', 'A'] ] },
  chat: { name: '多轮对话 (两个用户交替)', reqs: [A1, B1, A2, B2, A3, B3] },
}

const preset = ref('system')
const cap = ref(64)
const bsExp = ref(2)
const selId = ref(-1)
const reqs = computed(() => PRESETS[preset.value].reqs)
const stepper = useStepper(() => reqs.value.length, { interval: 1100 })
// 换预设后直接跳到最后一条, 先看到整棵树, 再按播放重看它怎么长出来
watch(preset, () => { stepper.pause(); stepper.step.value = reqs.value.length - 1; selId.value = -1 }, { immediate: true })

// 把第 0..upto 条请求依次放进一棵新树 —— 纯函数, 滑杆一动整棵树重建
const tree = computed(() => {
  let nid = 0
  const mk = (tokens, parent) => ({ id: nid++, tokens, parent, children: new Map(), last: 0 })
  const root = mk([], null)
  let size = 0, evicted = 0, totHit = 0, totLen = 0, cur = null
  for (let c = 0; c <= stepper.step.value; c++) {
    const ids = reqs.value[c], path = []
    let node = root, i = 0, split = -1
    while (i < ids.length && node.children.has(ids[i])) {   // 孩子按「子边首 token」索引
      let child = node.children.get(ids[i]), n = 0
      while (n < child.tokens.length && child.tokens[n] === ids[i + n]) n++
      if (n < child.tokens.length) {
        // ★ 分叉落在边中间: 劈成 上半 (公共部分) + 下半 (原来的后续), KV 槽位跟着 token 一起切
        const upper = mk(child.tokens.slice(0, n), node)
        upper.last = child.last
        node.children.set(ids[i], upper)
        child.tokens = child.tokens.slice(n); child.parent = upper
        upper.children.set(child.tokens[0], child)
        split = upper.id; child = upper
      }
      child.last = c; path.push(child.id); node = child; i += n
    }
    const hit = i
    let leaf = null
    if (i < ids.length) { leaf = mk(ids.slice(i), node); leaf.last = c; node.children.set(ids[i], leaf); size += leaf.tokens.length }
    // LRU 驱逐: 只看叶子, 当前请求路径上的节点锁住
    const locked = new Set([...path, leaf?.id]), gone = []
    while (size > cap.value) {
      const leaves = []
      const walk = (nd) => { nd.children.size ? nd.children.forEach(walk) : nd !== root && !locked.has(nd.id) && leaves.push(nd) }
      walk(root)
      if (!leaves.length) break
      const v = leaves.reduce((a, b) => (b.last < a.last ? b : a))
      v.parent.children.delete(v.tokens[0]); size -= v.tokens.length; evicted += v.tokens.length; gone.push(v.tokens.join(' '))
    }
    totHit += hit; totLen += ids.length
    const msg = [hit ? `沿树命中 ${hit} 个 token` : '没有公共前缀, 整条都要 prefill',
      split >= 0 ? '命中停在一条边的中间 → 劈开这条边' : '', gone.length ? `超容量, 驱逐叶子: 「${gone.join('」「')}」` : ''].filter(Boolean).join('; ')
    cur = { ids, hit, path: new Set(path), leaf: leaf?.id ?? -1, split, msg: msg + '。' }
  }
  const rows = [], byId = new Map()
  const flat = (nd, depth) => nd.children.forEach((ch) => {
    rows.push({ id: ch.id, tokens: ch.tokens, depth, last: ch.last, leaf: !ch.children.size }); byId.set(ch.id, ch); flat(ch, depth + 1)
  })
  flat(root, 0)
  return { rows, byId, size, evicted, hitRate: totHit / totLen, cur }
})

const cur = computed(() => tree.value.cur)
const blockHit = computed(() => Math.floor(cur.value.hit / 2 ** bsExp.value) * 2 ** bsExp.value)
const rowClass = (r) => (r.id === cur.value.split ? 'hot' : cur.value.path.has(r.id) ? 'ok' : r.id === cur.value.leaf ? 'on' : '')

const selNode = computed(() => tree.value.byId.get(selId.value))
const selChain = computed(() => { const out = []; for (let n = selNode.value; n && n.parent; n = n.parent) out.unshift(n); return out })
const selPath = computed(() => new Set(selChain.value.map((n) => n.id)))
const selPrefix = computed(() => selChain.value.flatMap((n) => n.tokens))
</script>

<style scoped>
.cap { font-size: 12px; color: var(--text-muted); margin: 10px 0 6px; }
.cap:first-child { margin-top: 0; }
.strip { display: flex; flex-wrap: wrap; gap: 3px; }
.tok { min-width: 0; padding: 0 6px; font-size: 11px; white-space: nowrap; }
.tree { display: flex; flex-direction: column; gap: 3px; }
.edge { display: flex; flex-wrap: wrap; align-items: center; gap: 3px; width: 100%; min-height: 30px; padding: 3px 6px; text-align: left; background: transparent; border: 1px solid transparent; }
.edge.path { border-color: var(--border-strong); background: var(--bg-elev); }
.edge.sel { border-color: var(--accent); }
.twig { color: var(--text-dim); font-size: 12px; }
.meta { margin-left: auto; font-size: 10px; color: var(--text-dim); }
.legend { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; font-size: 11px; color: var(--text-dim); margin-top: 10px; }
.sel-note { margin-top: 10px; }
</style>
