<!--
  滑窗 / 全注意力交替实验台 (对应 llm_models/models/moe/gpt_oss.py:GPTOSSMini.is_swa; mask 见 utils/masks.py:build_sliding_window_mask)。
  只讲一件事: 单层滑窗只看 W 个 token, 但信息可以跨层接力; 隔一层插一层全注意力, 感受野立刻铺满, KV cache 却只多一半。
-->
<template>
  <LabFrame
    title="滑窗与全注意力交替 — 远处的信息怎么一层一层接力传过来"
    sub="- 模拟什么: 一个 L 层的 Transformer 处理长度为 40 个 token 的输入序列。滑动窗口注意力 (SWA) 限制每层的每个位置仅关注自身及前方 $W - 1$ 个邻近位置, 更早的历史位置直接通过 mask 屏蔽, 从而将该层的 KV cache 容量上限固定为 W 条。尽管单层视野受限, 历史信息仍可跨层逐级传递: 第 2 层所关注的邻近位置, 在第 1 层已预先聚合了更早一轮邻居的信息。GPT-OSS 进一步采用滑窗层与全注意力层隔层交替的混合排布。本实验定量展示选定输出位置最终能够聚合到的原始输入范围 (即有效感受野), 以及全模型各层合计所需的 KV cache 条目数。
- 图上是什么: 自上而下每一行代表一层网络, 最上方一行为最后一层输出 (支持点击切换目标位置), 最下方一行标有「→ 输入」, 代表第 0 层读取的原始输入序列; 行首标签「滑窗」与「全」标明该层的注意力类型。每行 40 个格子对应 40 个序列位置, 高亮格子表示当前选中的输出 token 在该层间接依赖到了该位置的信息, 颜色含义如下:
  - 绿色: 经由滑窗层逐层传递覆盖的位置。
  - 黄色: 经由全注意力层单步直达覆盖的位置。
  - 灰色: 在当前层数与窗口宽度下未能覆盖到的前序位置。
  - 半透明浅格: 位于选中位置右侧的未来 token (受因果 mask 屏蔽而不可见)。
- 可以做什么: 通过顶部三个按钮切换层间排布方案, 并用三根滑杆调节结构参数:
  - 「全滑窗」: 所有层均采用滑动窗口注意力 (对应 Mistral 的基础做法)。
  - 「滑窗 / 全 交替」: 偶数层采用滑窗注意力、奇数层采用全注意力 (对应 GPT-OSS 的做法)。
  - 「全注意力」: 所有层均关注完整前缀历史 (对应标准 LLaMA 的做法)。
  - 拖动「滑动窗口 W」: 调节每个滑窗层的单层视野宽度。
  - 拖动「层数」: 调节模型的堆叠总层数。
  - 拖动「看哪个输出 token」或直接点击最上方输出行的任意格子: 切换所追踪的目标输出位置。
- 应该看到: 在「全滑窗」模式下, 从顶层向下每经过一层, 高亮覆盖区段向左额外延伸 $W - 1$ 格, 到达最底行输入端时往往仍未铺满全序列, 最早的若干输入 token 始终处于感受野之外; 而只要网络中包含至少一层全注意力, 从该行向下的全部历史位置便立即全亮。右侧读数含义如下:
  - 「token q 的感受野」: 最底行输入端亮起的格子总数, 斜杠后的 $q + 1$ 为该位置左侧 (含自身) 的理论可见总长, 两者相等即代表实现完整覆盖。
  - 「KV cache 条目」: 全模型各层需缓存的 K/V 条数之和, 其中滑窗层按 $\min(T, W)$ 计、全注意力层按 $T = 40$ 计。
  - 「相对全注意力」: 当前总缓存条目数占全注意力基线 (每层均存 40 条) 的百分比, 数值越小越节省显存。
  - 「纯滑窗看全 q+1 个需要的层数」: 按 $\lceil q / (W - 1) \rceil$ 计算的理论最少层数, 与当前层数对比即可看出还差多少层才能覆盖全序列。"
    module="llm_models/models/moe/gpt_oss.py"
    run="python -m llm_models.run_models.moe.gpt_oss.infer_gpt_oss"
    :challenge="{
      ask: '设序列长度 $T = 40$、窗口宽度 $W = 8$、模型共 4 层 (与 infer_gpt_oss 中的 mini 模型设定一致)。采用「全滑窗」时, 末尾 token 的感受野仅覆盖 29 个输入位置。请先思考: 若切换为「滑窗 / 全 交替」, 末尾 token 能看到多少个输入位置? 全模型 KV cache 总量会从 32 条增加至多少条?',
      answer: '答案: 切换为交替排布后, 感受野覆盖全部 40 个输入位置; 全模型 KV cache 从 32 条增至 96 条。\n感受野计算过程:\n- 全滑窗排布: 每层最多向左额外扩展 $W - 1 = 7$ 个历史位置 (窗口宽度 8 中包含当前位置自身), 4 层跨层接力合计覆盖 $4 \\times 7 + 1 = 29$ 个位置, 序列开头的 11 个 token 完全无法触及。若仅靠纯滑窗覆盖全部 40 个位置, 至少需要 $\\lceil 39 / 7 \\rceil = 6$ 层, 且远距离信息历经多次跨层中转后容易衰减混杂。\n- 交替排布: 只要网络中存在一层全注意力, 该层的任意位置即可单步直接读取全部前缀历史, 有效感受野立即扩展至完整的 40 个位置。\nKV cache 总量计算过程 (累加各层所需保存的 K/V 条目数):\n- 全滑窗排布: 每层仅保留最近 $W = 8$ 条, 4 层合计 32 条。\n- 交替排布: 4 层缓存长度依次为 [8, 40, 8, 40], 合计 96 条, 相比全注意力节省 40% 显存, 与 Python 示例脚本打印的各层 cache 长度完全一致。\n- 全注意力排布: 每层均保存完整 40 条, $4 \\times 40 = 160$ 条。\n滑窗层直接裁剪掉窗口外的早期 K/V 不会改变计算结果, 因为那些位置在带状局部 mask 下本就会被置为 $-\\infty$, 对应注意力权重严格为 0。\n在真实长文本尺度下显存收益更为显著: GPT-OSS 采用 $W = 128$、最大上下文 131K, 此时滑窗层的固定缓存开销几乎可以忽略不计, 整网 KV cache 总量约为同规模全注意力模型的一半。',
    }"
  >
    <template #controls>
      <div class="row">
        <button v-for="p in PATTERNS" :key="p.id" type="button" :class="{ active: pattern === p.id }" @click="pattern = p.id">{{ p.label }}</button>
      </div>
      <LabSlider v-model="W" label="滑动窗口 W" :min="2" :max="8" />
      <LabSlider v-model="L" label="层数" :min="1" :max="8" />
      <LabSlider v-model="q" label="看哪个输出 token" :min="0" :max="T - 1" />
    </template>

    <div class="cells grid" :style="{ gridTemplateColumns: `74px repeat(${T}, 14px)` }">
      <template v-for="(row, r) in rows" :key="r">
        <span class="rl mono" :class="row.kind">{{ row.label }}</span>
        <template v-for="(on, i) in row.reach" :key="i">
          <!-- 整列都能点; 键盘只停在选中的那一格, 方向键左右移动 -->
          <button v-if="r === 0" type="button" class="cell pick" :class="{ on: i === q }" :aria-label="`选择 token ${i}`" :aria-pressed="i === q"
                  :tabindex="i === q ? 0 : -1" @click="q = i"
                  @keydown.left.prevent="q = Math.max(0, q - 1)" @keydown.right.prevent="q = Math.min(T - 1, q + 1)" />
          <span v-else class="cell col" :class="{ [row.kind === 'full' ? 'hot' : 'ok']: on, dim: !on && i <= q, fut: i > q }" @click="q = i" />
        </template>
      </template>
    </div>
    <p class="cap">
      <span class="cell ok" /> 经滑窗层传来 <span class="cell hot" /> 经全注意力层传来 · 最底一行 = 这个输出 token 在输入上的感受野
    </p>

    <template #stats>
      <div class="kv"><span>token {{ q }} 的感受野 (够到的输入数 / 它左边的位置数)</span><b :class="field === q + 1 ? 'good' : 'bad'">{{ field }} / {{ q + 1 }}</b></div>
      <div class="kv"><span>KV cache 条目 (所有层合计)</span><b>{{ kv }}</b></div>
      <div class="kv"><span>相对全注意力 (每层存 {{ T }} 条, {{ L }} 层共 {{ L * T }})</span><b :class="kv < L * T ? 'good' : ''">{{ ((kv / (L * T)) * 100).toFixed(0) }}%</b></div>
      <div class="kv"><span>纯滑窗看全 {{ q + 1 }} 个需要的层数</span><b>{{ needLayers }}</b></div>
      <div class="lab-note">
        <p>滑窗层的 KV cache 采用固定长度滚动缓冲区实现, 仅保留最近 W 条记录, 无论上下文如何拉长都不会继续膨胀; 全注意力层的 KV cache 则随上下文长度线性增长。</p>
        <p>隔层交替排布令半数网络层通过滑动窗口压缩缓存开销, 另半数网络层保留单步直达任意远端位置的全局通路, 以保留一半全注意力层的 KV cache 为代价换取无损的全局感受野。</p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { range, sum } from '@/utils/labmath.js'

const T = 40
const PATTERNS = [
  { id: 'swa', label: '全滑窗 (Mistral)' },
  { id: 'alt', label: '滑窗 / 全 交替 (GPT-OSS)' },
  { id: 'full', label: '全注意力 (LLaMA)' },
]
// 默认全滑窗: 打开先看到最后一个 token 只够得着 29 / 40 个输入
const pattern = ref('swa'), W = ref(8), L = ref(4), q = ref(T - 1)

// 第 l 层 (0 起) 的类型: 交替排布里偶数层滑窗、奇数层全注意力
const kinds = computed(() => range(L.value).map((l) => (pattern.value === 'full' || (pattern.value === 'alt' && l % 2 === 1) ? 'full' : 'swa')))

const rows = computed(() => {
  let reach = range(T).map((i) => i === q.value)
  const out = [{ label: '输出 (点选)', kind: 'pick', reach }]
  for (let l = L.value - 1; l >= 0; l--) {
    const kind = kinds.value[l], next = Array(T).fill(false)
    // ★ 位置 i 在这一层能看到: 滑窗 [i−W+1, i], 全注意力 [0, i]
    reach.forEach((on, i) => { if (on) for (let j = kind === 'full' ? 0 : Math.max(0, i - W.value + 1); j <= i; j++) next[j] = true })
    reach = next
    out.push({ label: `L${l} · ${kind === 'full' ? '全' : '滑窗'}${l === 0 ? ' → 输入' : ''}`, kind, reach })
  }
  return out
})
const field = computed(() => rows.value[rows.value.length - 1].reach.filter(Boolean).length)
const kv = computed(() => sum(kinds.value.map((k) => (k === 'full' ? T : Math.min(T, W.value)))))
const needLayers = computed(() => Math.max(1, Math.ceil(q.value / (W.value - 1))))
</script>

<style scoped>
.grid { gap: 2px; min-width: 720px; }
.cell { min-width: 0; width: 14px; height: 18px; }
.cell.fut { opacity: 0.12; }
.pick { padding: 0; min-height: 0; height: 26px; cursor: pointer; border-radius: 3px; }
.col { cursor: pointer; }
.pick.on { background: var(--accent); border-color: var(--accent); }
.rl { font-size: 10px; color: var(--text-dim); align-self: center; white-space: nowrap; }
.rl.full { color: var(--warn); } .rl.swa { color: var(--left); }
.cap { display: flex; flex-wrap: wrap; align-items: center; gap: 4px 6px; font-size: 11px; color: var(--text-dim); margin-top: 8px; }
.cap .cell { width: 12px; height: 12px; }
</style>
