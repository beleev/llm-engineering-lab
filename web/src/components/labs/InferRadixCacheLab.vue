<!-- Radix 前缀缓存实验台 (对应 llm_infer/m05_radix_cache): JS 里老老实实建一棵基数树, 逐条请求重放。 -->
<template>
  <LabFrame
    title="Radix cache — 公共前缀的 KV 只算一次, 任意长度都能命中"
    sub="- 模拟什么: 很多请求的开头是一样的 (同一段 system prompt、同一组 few-shot 例子、同一段对话历史)。一个 token 的 KV 只依赖它前面的 token, 所以开头相同的那段 KV 完全一样, 算过一次就该复用, 新请求只需要 prefill 后面不同的部分。SGLang 用一棵基数树 (radix tree) 记住所有见过的 token 序列, 每条边上存一段 token (不是一个)。缓存容量有限, 满了按 LRU (最久没用的先走) 从叶子开始驱逐。新请求这样匹配:
        1. 从根往下走, 走到走不动为止。
        2. 走过的部分就是最长公共前缀, 它的 KV 直接复用。
      - 图上是什么: 上面一排 token, 下面一行一个树节点:
        - 最上面一排: 当前这条请求的 token。绿底的是在树上命中的前缀 (KV 直接复用, 不用前向), 紫底的是新插进树里的部分。
        - 下面每一行: 树上的一个节点, 缩进越深离根越远, 行里的 token 串就是「父节点到它」这条边上存的 token; 行末写着这条边有几个 token、上一次被第几条请求用到、是不是叶子。
        - 有底框的行: 被当前请求走过的。
        - 黄底的行: 「刚被从中间劈开的边」, 意思是这条请求的前缀停在了一条边的中间, 树把这条边切成两段, 好让前缀恰好落在一个节点上。
      - 可以做什么: 三个预设切换请求流, 然后是两个滑杆和播放器:
        - 「共享 system prompt」6 条请求都以同一段 system prompt 开头。
        - 「few-shot」请求都带同样的几个例子。
        - 「多轮对话」两个用户交替提问, 每一轮都带着自己之前的全部历史。
        - 拖「缓存容量」改树里最多能放几个 token 的 KV, 往小拖会触发驱逐; 拖「对比: block hash 的 block」改对照方案的块大小。
        - 「下一步」和「时间轴」逐条放请求, 打开时停在最后一条, 按播放从头重看树是怎么长出来的。
        - 点任意一行, 下面会显示这个节点代表的完整前缀。
      - 应该看到: 多轮对话预设下, 同一个用户的第 2、3 轮几乎整条命中; 把容量拖到 8 左右, 能看到驱逐从叶子开始, 而当前请求走过的节点不会被驱逐。右侧读数:
        - 「本条命中 / 长度」是这条请求有多少 token 不用算。
        - 「同样前缀, block hash 只能命中」是对照方案 (vLLM 的按整块哈希) 的命中数, 它只能命中整块, 前缀长度不是块大小整数倍时会少算一截, 比 radix 少就标红。
        - 「累计命中率」是到目前为止所有请求里命中 token 的比例, 等于省掉的 prefill 比例。
        - 「树里 token / 容量 · 累计驱逐」看容量用了多少、驱逐了多少 token。"
    module="llm_infer/m05"
    run="python -m llm_infer.m05_radix_cache.demo"
    :challenge="{
      ask: '容量不够时, 如果直接扔掉「最久没用」的中间节点, 会怎样?',
      answer: '它的所有后代会跟着一起作废, 所以只能驱逐叶子。\n原因在 KV 的性质: 一个 token 的 KV 依赖它前面的全部 token。中间节点这条边上的 token, 是它所有后代前缀的一部分。把它扔了, 后代的 KV 虽然还躺在显存里, 但新请求从根往下走到这里就断了, 再也走不到后代那里, 等于后代的 KV 全部变成没人能命中的垃圾。所以驱逐顺序只能是:\n1. 先驱逐叶子。\n2. 叶子没了, 父节点变成新的叶子, 再轮到它。\n还有一条保护: 当前正在用的请求走过的路径被锁住 (ref_count 大于 0), 不会被驱逐, 否则这条请求还没算完, KV 就被别人抢走了。\n右侧「block hash 只能命中」是和另一种前缀缓存做法的对比:\n- radix tree: 命中粒度是单个 token, 共享前缀有 $n$ 个 token 就复用 $n$ 个。\n- block hash (vLLM 的做法): 以整块为单位算哈希, 只能复用 $\\lfloor n/\\text{bs} \\rfloor \\cdot \\text{bs}$ 个。共享 100 个 token、bs=16 时只能复用 96 个, 剩下 4 个要重算。block 越大丢得越多; Python demo 的合成负载上 radix 命中 44.8%, block hash (B=4) 只有 33.5%。',
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
      <p v-if="!tree.rows.length" class="lab-note">容量太小, 树里的节点全被驱逐了。把「缓存容量」拖大一点。</p>
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
        <p>驱逐时只挑叶子, 按上次使用时间最旧的先走; 当前请求走过的节点被锁住 (ref_count 大于 0), 不会被驱逐。</p>
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
