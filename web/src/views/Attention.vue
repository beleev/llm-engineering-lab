<template>
  <div>
    <h1 class="page-title">注意力的四代演进</h1>
    <div class="page-subtitle lead-group">
      <p>
        模型生成文字是一个 token 一个 token 往外吐的。每吐一个新 token, 注意力层都要回头看一遍前面所有 token:
        每个旧 token 在每一层都留下一个 K 向量 (key, 它能被谁问到) 和一个 V 向量 (value, 被问到时交出什么)。
        这些向量以后不会再变, 所以算一次存起来, 这份存货就叫 KV cache。
      </p>
      <p>
        问题是它太大: 上下文拉到 128K 个 token 时, 这份 cache 比模型权重还占显存, 而且每生成一个 token 都要把它从头读一遍。
        从 MHA 到 MQA、GQA、MLA, 再到 DSA, 这一路都在做同一件事:
        <strong>在不掉点 (效果不变差) 的前提下, 把每层每个 token 要缓存的东西再变小一点。</strong>
      </p>
      <p>真实模型上, 一个 token 在全部层上的 KV 加起来:</p>
      <ul class="pts">
        <li><b>LLaMA-2-7B:</b> 512 KiB (每层 K 和 V 各 4096 个数, fp16 每个数 2 字节, 一层 16 KiB, 32 层)。</li>
        <li><b>LLaMA-3-8B:</b> 128 KiB (K/V 只留 8 个头, 每头 128 维)。</li>
        <li><b>DeepSeek-V3:</b> 68.6 KiB (每层只存 576 个数, 61 层)。</li>
      </ul>
    </div>

    <ChapterIntro
      tldr="- 核心公式不变: 八年来注意力的主干计算始终是 $\mathrm{softmax}(QK^\top / \sqrt{d_k})V$ —— 用当前位置的 Q 与历史位置的 K 做点积打分, 归一化成权重后再对 V 加权求和。
- 演进焦点在于怎么存 K/V、怎么省计算:
  - MHA: 每个查询头独立保留一套 K/V, 表达力最足但缓存最大。
  - GQA: 多个查询头共享一组 K/V, 直接按组数成倍压缩缓存。
  - MLA: 不存完整 K/V, 改存低维压缩向量 $c_{kv}$ 与共享旋转位置向量, 用时再投影还原。
  - DSA: 在 MLA 缓存之上先用轻量打分器筛出 top-k 个候选位置, 把超长上下文的注意力计算从 $O(T^2)$ 降到 $O(T\cdot k)$。
- 直观量级: 同样宽度 $d_{\text{model}}=4096$ 的单层里, 每新增 1 个 token 的缓存可从 MHA 的 16 KiB 一路压到 MLA 的约 1.1 KiB。"
      question="模型每生成一个 token 都要翻一遍前面所有 token 的 K 和 V。每层每个 token 存的这份 K、V 还能不能再小, 同时效果不变差?"
      :goals="[
        '说清 MHA / GQA / MLA / DSA 各自省的是哪一块',
        '算出「头数 / KV 头数 / 压缩维 / 稀疏 top-k」变化时 cache 怎么变',
        '对着源码认出每一种注意力的 forward 差在哪几行',
      ]"
      :codes="[{ path: 'llm_models/layers/core/attention.py' }]"
      :prereq="prevChapter"
      :next-step="nextChapter"
    />

    <EvolutionChain
      title="演进逻辑链 · 每一代都在补上一代的窟窿"
      subtitle="每一格先写上一代的痛点, 再写这一代的解法; 解法留下的新问题, 就是下一格的开头。"
      :steps="evoSteps"
    />

    <!-- 先建立实验任务, 再让读者替换内部机制。 -->
    <section class="concept-workbench card" aria-labelledby="attention-workbench-title">
      <div class="workbench-heading">
        <span class="eyebrow">动手建立直觉</span>
        <div>
          <h2 id="attention-workbench-title">先把任务钉死, 再换「KV 怎么存」</h2>
          <p>
            这是一次对照实验, 模拟的是一层注意力在推理时要为整段上下文存多少 KV cache。
            上下文长度 T (要记住多少个 token) 和模型宽度 d_model (每个 token 的向量有多少维) 都钉死不动, 只换 Attention 方案。
            这样下面柱图的差别只可能来自「K/V 怎么存」这一件事。
          </p>
        </div>
      </div>

      <ol class="learning-steps" aria-label="交互阅读顺序">
        <li><span>1</span><strong>选上下文</strong><small>要记住多少个 token</small></li>
        <li><span>2</span><strong>换 Attention</strong><small>看 K/V 换了种存法</small></li>
        <li><span>3</span><strong>读结果</strong><small>说出显存和算力各差多少</small></li>
      </ol>

      <fieldset class="context-presets">
        <legend>第一步 · 挑一个真实的阅读场景</legend>
        <div class="preset-grid">
          <button
            v-for="preset in contextPresets"
            :key="preset.id"
            type="button"
            :class="{ active: activePresetId === preset.id }"
            :aria-pressed="activePresetId === preset.id"
            @click="applyContextPreset(preset)"
          >
            <span>{{ preset.label }}</span>
            <small>{{ preset.hint }}</small>
          </button>
        </div>
      </fieldset>

      <div class="preset-readout" aria-live="polite" aria-atomic="true">
        <span class="readout-kicker">当前实验</span>
        <strong>{{ activePresetLabel }}</strong>
        <span>
          上下文 {{ formatT(p.T) }} 个 token, 模型宽度 d_model = {{ p.d_model }}。
          下一步只换 Attention 方案, 看存法换了之后 cache 的数字怎么走。
        </span>
      </div>
    </section>

    <!-- 顶部 variant 切换 + paper cite -->
    <div class="variant-tabs" role="group" aria-label="第二步 · 选择 Attention 方案">
      <button
        v-for="(v, index) in variants"
        :key="v.id"
        type="button"
        :class="{ active: current.id === v.id }"
        :aria-pressed="current.id === v.id"
        @click="current = v"
      >
        <span class="variant-index" aria-hidden="true">{{ index + 1 }}</span>
        <span class="variant-name">{{ v.name }}</span>
        <span class="yr">{{ v.year }}</span>
      </button>
    </div>

    <section
      class="concept-bridge"
      :style="{ '--variant-color': current.color }"
      aria-labelledby="current-concept-title"
    >
      <div class="bridge-heading">
        <span class="eyebrow">第二步 · 现在用的是哪种</span>
        <h2 id="current-concept-title">{{ current.fullName }}</h2>
        <Prose :text="current.description" />
      </div>
      <div class="causal-flow" aria-live="polite" aria-atomic="true">
        <article>
          <span class="causal-label">输入</span>
          <strong>{{ conceptStory.input }}</strong>
          <p>{{ conceptStory.inputNote }}</p>
        </article>
        <span class="causal-arrow" aria-hidden="true">→</span>
        <article class="mechanism">
          <span class="causal-label">内部机制</span>
          <strong>{{ conceptStory.mechanism }}</strong>
          <p>{{ conceptStory.mechanismNote }}</p>
        </article>
        <span class="causal-arrow" aria-hidden="true">→</span>
        <article class="outcome">
          <span class="causal-label">可观察结果</span>
          <strong>{{ conceptStory.outcome }}</strong>
          <p>{{ conceptStory.outcomeNote }}</p>
        </article>
      </div>
      <div class="concept-takeaway">
        <span>一句话带走</span>
        <strong>{{ conceptStory.takeaway }}</strong>
      </div>
    </section>

    <!-- 主布局: 左参数, 中公式/数据, 右可视化 -->
    <div class="grid grid-2" style="gap: 20px;">
      <!-- 左: 参数 + 公式 -->
      <div class="card">
        <h3>参数调节 <span class="tag">可拖动</span></h3>
        <div class="form-row">
          <label for="attention-seq-length">序列长度 T</label>
          <input id="attention-seq-length" type="range" min="128" max="131072" step="128" v-model.number="p.T" />
          <span class="val">{{ formatT(p.T) }}</span>
        </div>
        <div class="form-row">
          <label for="attention-model-width">模型宽度 d_model</label>
          <input id="attention-model-width" type="range" min="512" max="8192" step="128" v-model.number="p.d_model" />
          <span class="val">{{ p.d_model }}</span>
        </div>
        <div class="form-row">
          <label for="attention-query-heads">查询头数 n_heads</label>
          <input id="attention-query-heads" type="range" min="4" max="64" step="2" v-model.number="p.n_heads" />
          <span class="val">{{ p.n_heads }}</span>
        </div>
        <div v-if="current.id === 'gqa'" class="form-row">
          <label for="attention-kv-heads">KV 头数 num_kv_heads</label>
          <input id="attention-kv-heads" type="range" :min="1" :max="p.n_heads" step="1" v-model.number="p.num_kv_heads" />
          <span class="val">{{ p.num_kv_heads }}</span>
        </div>
        <div v-if="current.id === 'mla' || current.id === 'dsa'" class="form-row">
          <label for="attention-kv-rank">KV 压缩维 kv_lora_rank</label>
          <input id="attention-kv-rank" type="range" min="64" max="1024" step="32" v-model.number="p.kv_lora_rank" />
          <span class="val">{{ p.kv_lora_rank }}</span>
        </div>
        <div v-if="current.id === 'mla' || current.id === 'dsa'" class="form-row">
          <label for="attention-rope-dim">位置维 qk_rope_head_dim</label>
          <input id="attention-rope-dim" type="range" min="16" max="128" step="8" v-model.number="p.qk_rope_head_dim" />
          <span class="val">{{ p.qk_rope_head_dim }}</span>
        </div>
        <div v-if="current.id === 'dsa'" class="form-row">
          <label for="attention-sparse-top-k">稀疏候选 sparse_top_k</label>
          <input id="attention-sparse-top-k" type="range" :min="32" :max="Math.min(2048, p.T)" step="32" v-model.number="p.sparse_top_k" />
          <span class="val">{{ p.sparse_top_k }}</span>
        </div>

        <div class="formula-box">
          <Tex :text="current.formula" />
        </div>

        <p class="parameter-feedback" aria-live="polite" aria-atomic="true">
          每来一个新 token, 本层往 cache 里再写
          <strong class="mono">{{ formatBytes(cacheBytesPerToken(current)) }}</strong> (fp16, 每个数 2 字节);
          当前整段上下文已经占了
          <strong class="mono">{{ formatBytes(cacheBytes(current)) }}</strong>。
        </p>

        <div class="trade" style="margin-top: 16px;">
          <div class="tr-item">
            <span class="ok">✓</span>
            <span>{{ current.pros }}</span>
          </div>
          <div class="tr-item">
            <span class="no">✗</span>
            <span>{{ current.cons }}</span>
          </div>
        </div>

        <div style="margin-top: 12px; font-size: 11px; color: var(--text-dim);">
          <span class="mono">{{ current.paper }}</span>
          · 使用模型: <span class="mono">{{ current.usedIn.join(', ') }}</span>
        </div>
      </div>

      <!-- 右: KV cache 横向对比柱图 -->
      <div class="card">
        <h3>整段 KV cache 对比 <span class="tag">单层 · B=1 · fp16</span></h3>
        <ul class="desc pts" style="margin-bottom: 16px;">
          <li><b>图上是什么:</b> 四根柱子, 一根对应一种 Attention 方案, 长度是这一层为<strong>整段上下文</strong>存下的 KV cache 字节数, 右边的百分比是相对 MHA 的比例。</li>
          <li>
            <b>算法:</b> 「每个 token 要存几个数」乘上 token 数 <span class="mono">T={{ formatT(p.T) }}</span>, 再乘 2 字节 (fp16)。
            当前 <span class="mono">d_model={{ p.d_model }}</span>、<span class="mono">n_heads={{ p.n_heads }}</span>; 只算一层、一条序列 (B=1)。
          </li>
          <li><b>怎么读:</b> 柱越短, 解码时每一步要重读的字节就越少。拖左边的滑条, 柱子跟着动。</li>
        </ul>
        <div class="bars" aria-label="四种 Attention 的单层 KV cache 对比">
          <div v-for="v in variants" :key="v.id" class="bar-row" :class="{ current: v.id === current.id }">
            <div class="bar-label">
              <span class="bar-name">{{ v.name }}</span>
              <span class="bar-sub mono">{{ formatBytes(cacheBytes(v)) }}</span>
            </div>
            <div class="bar-track">
              <div class="bar-fill"
                   :style="{ width: barPct(v) + '%', background: v.color }"></div>
            </div>
            <div class="bar-rel mono">{{ relToMHA(v) }}</div>
          </div>
        </div>

        <div class="hero-stat">
          <div class="stat">
            <div class="k">新增 1 token / 层</div>
            <div class="v accent">{{ formatBytes(cacheBytesPerToken(current)) }}</div>
            <div class="hint">每生成一个新 token, 这一层要往 cache 里追加这么多</div>
          </div>
          <div class="stat">
            <div class="k">当前上下文 / 层</div>
            <div class="v">{{ formatBytes(cacheBytes(current) * 1) }}</div>
            <div class="hint">相对 MHA: {{ relToMHA(current) }}</div>
          </div>
          <div class="stat">
            <div class="k">{{ LAYERS }} 层合计</div>
            <div class="v">{{ formatBytes(cacheBytes(current) * LAYERS) }}</div>
            <div class="hint">按 LLaMA-2-7B 的 {{ LAYERS }} 层算; 只算 KV, 没算 batch 和其他激活</div>
          </div>
          <div class="stat">
            <div class="k">整段要算多少对 (q, k)</div>
            <div class="v">{{ formatFlops(flopsPerQuery) }}</div>
            <div class="hint"><Tex text="每个 query 要和几个 key 算分数, $T$ 个 query 加起来的对数。拉长 $T$ 看它是按 $T^2$ 还是按 $T\cdot k$ 涨" /></div>
          </div>
        </div>
      </div>
    </div>

    <!-- Attention 矩阵可视化 -->
    <section class="section">
      <h2>Attention 矩阵可视化</h2>
      <div class="lead-group">
        <ul class="pts">
          <li>
            <b>模拟什么:</b> 一层注意力里, 每个 token 在看哪些历史 token。图里的权重是按「离得近的分高, 再撒几个热点」生成的示意, 不是真实模型算出来的。
          </li>
          <li>
            <b>图上是什么:</b>
            <ul class="pts">
              <li>一行是一个 query (正在算的 token), 一列是一个 key (被看的 token), 一共 24 个 token。</li>
              <li>格子颜色越深, 这个 query 分给那个 key 的权重越大; 每一行的权重加起来是 1。</li>
              <li>右上三角被因果 mask 屏蔽 (因果 mask: 语言模型只许看前面的 token, 后面位置的分数直接填成负无穷), 所以谁也看不见未来。</li>
              <li v-if="current.id === 'dsa'">DSA 只在 indexer 挑出的 top-k 个位置上有权重, 所以左下三角也大片是空的。</li>
            </ul>
          </li>
          <li>
            <b>可以做什么:</b> 在上面切换方案。
            <ul class="pts">
              <li><b>MHA / GQA / MLA:</b> 只改 K/V 怎么存, 每个 query 看的 key 一个不少, 所以矩阵长得一样。</li>
              <li><b>DSA:</b> 切到 DSA 再拖「稀疏候选 sparse_top_k」, 每行只留分最高的几格。</li>
            </ul>
          </li>
        </ul>
      </div>
      <div class="card">
        <div class="matrix-grid">
          <svg
            :viewBox="`0 0 ${matrixSize} ${matrixSize}`"
            width="100%"
            :height="matrixSize"
            style="max-width: 480px; border: 1px solid var(--border);"
            role="img"
            :aria-label="matrixAriaLabel"
          >
            <rect v-for="cell in matrixCells" :key="cell.i + ':' + cell.j"
                  :x="cell.j * cellSize" :y="cell.i * cellSize"
                  :width="cellSize - 0.5" :height="cellSize - 0.5"
                  :style="{ fill: cell.color }" />
          </svg>
          <div class="legend-attn">
            <div class="legend-strip">
              <span v-for="i in 10" :key="i"
                    :style="{ background: colorFor(i / 10) }"></span>
            </div>
            <div class="legend-labels mono">
              <span>0</span><span>权重</span><span>1</span>
            </div>
            <p class="desc" style="margin-top: 16px; max-width: 280px;">
              N = <span class="mono">{{ N }}</span> 个 token 的因果注意力矩阵。
              <span v-if="current.id === 'dsa'">滑条上的 sparse_top_k 对这张小图封顶成 <span class="mono">{{ Math.min(p.sparse_top_k, N) }}</span>; 画格子时另按 1/32 缩小, 每行最多亮 8 格, 滑条拖到 256 以下亮的格子才会变少。</span>
            </p>
            <p class="matrix-reading">{{ matrixReading }}</p>
          </div>
        </div>
      </div>
    </section>

    <!-- mask 一族: 投影压缩之外的另一条降本路线 -->
    <section class="section">
      <h2>另外两条路线: 改 mask, 或者干脆不存 KV</h2>
      <div class="lead-group">
        <p>本章主线是「压 KV 投影」。要省 cache 和算力, 还有两条路和它互不冲突, 可以叠着用。</p>
        <ul class="pts">
          <li><b>压 KV 投影 (本章, MHA 到 GQA 再到 MLA):</b> 每个 token 要存的 K/V 变小; 每个 query 看的 key 一个不少。</li>
          <li>
            <b>改 mask (Mistral / StreamingLLM / DSA):</b> mask 是一张「谁能看见谁」的表。
            <ul class="pts">
              <li><b>Mistral:</b> 滑动窗口只让每个 token 看最近 W 个, 窗口外的 KV 直接丢掉。</li>
              <li><b>StreamingLLM:</b> 在窗口之外再永久留住开头几个 token (叫 sink)。</li>
              <li><b>DSA:</b> 只看 indexer 挑出的 top-k 个。</li>
            </ul>
            看的 key 少了, 省的是算力; 前两种因为丢掉了窗口外的 KV, 还顺手省了 cache。
          </li>
          <li>
            <b>线性注意力</b> (Gated DeltaNet, <RepoLink path="llm_models/layers/sparse/linear_attention.py" label="linear_attention.py" tiny />):
            省得最多的一条: 它不再按 token 存 K/V, 而是把全部历史压进一个固定大小的状态矩阵, 序列再长它也不变大。代价是固定大小的状态记不精确。
            Qwen3-Next 拿它换掉了 75% 的层, 剩下 25% 留全注意力, 给需要精确召回的地方兜底。
          </li>
        </ul>
        <p>下面的实验台画的是第二条路: 拖窗口大小和 sink 数, 看每个 token 能看见的格子怎么变。细节在「SWA · MTP · 混合线性」那一章。</p>
      </div>
      <AttnMaskLab />
    </section>

    <!-- 源码速览 -->
    <section class="section">
      <h2>核心代码</h2>
      <p class="lead">你现在选中的这一档, 在 <RepoLink path="llm_models/layers/core/attention.py" label="llm_models/layers/core/attention.py" tiny /> 里就是 <span class="mono">{{ classFor(current.id) }}</span> 这个类。四个类在同一个文件里, 可以直接对着看差在哪几行。下面的片段是精简版, 只留了决定「cache 里存什么」的那几行。</p>
      <CodeBlock :code="codeSnippet" />
    </section>

    <!-- 本章挂载的实验台 (data/labmap/*.js) 与章末自测 (data/quiz/*.js), 没配置时不渲染 -->

    <LabMount :names="['ResearchCompletionLab']" />
    <TopicSupplement v-for="section in topicPages.attention?.sections || []" :key="section.id" :section="section" />

    <QuizCard />


    <ChapterNav
      :prev="prevChapter"
      :next="{ ...nextChapter, hint: '看完 attention 后, 再看 RoPE 是怎么注入 Q/K 的' }"
    />
  </div>
</template>

<script setup>
import TopicSupplement from '@/components/TopicSupplement.vue'
import { topicPages } from '@/data/models.js'
import Tex from '@/components/Tex.vue'
import LabMount from '@/components/LabMount.vue'
import QuizCard from '@/components/QuizCard.vue'
import Prose from '@/components/Prose.vue'
import { ref, reactive, computed, watch } from 'vue'
import { variants } from '@/data/attention.js'
import { learningPath } from '@/data/models.js'
import { clamp } from '@/utils/labmath.js'
import CodeBlock from '@/components/CodeBlock.vue'
import ChapterIntro from '@/components/ChapterIntro.vue'
import ChapterNav from '@/components/ChapterNav.vue'
import EvolutionChain from '@/components/EvolutionChain.vue'
import RepoLink from '@/components/RepoLink.vue'
import AttnMaskLab from '@/components/labs/AttnMaskLab.vue'

// 上一章 / 下一章从 learningPath 取, 不手写章名和编号 (与 Infer.vue 同一写法)
const at = learningPath.findIndex((x) => x.route === 'attention')
const prevChapter = { name: learningPath[at - 1].route, label: `上一章 · ${learningPath[at - 1].label}` }
const nextChapter = { name: learningPath[at + 1].route, label: `下一章 · ${learningPath[at + 1].label}` }

// 「N 层合计」用的层数: LLaMA-2-7B 是 32 层, 正文里的 512 KiB = 16 KiB × 32
const LAYERS = 32

const evoSteps = [
  { name: 'MHA', year: 2017, color: 'var(--text-muted)',
    pain: '(原点) 注意力分成很多个头各看各的, 每个头都要存自己的一对 K/V',
    fix: '每个头独立投影出 K、V, 表达力最满。代价是 cache 也最大: 当年序列短没人在意, 这笔账留给了八年后的长上下文推理' },
  { name: 'MQA / GQA', year: '2019 / 2023', color: 'var(--accent)',
    pain: '一层的 cache 是 $2\\cdot T\\cdot d_{\\text{model}}$ 个数 (T 个 token, K 和 V 各 d_model 个): 上下文翻倍, 显存就翻倍',
    fix: '让好几个 Q 头共用同一对 K/V, 只存 num_kv_heads 份: cache 除以每组的头数, 效果几乎不掉。本仓库的教学配置下每 token 每层从 1024 个数降到 256 个' },
  { name: 'MLA', year: 2024, color: 'var(--left)',
    pain: 'KV 头数砍到 1 就到底了, 存的仍是整条 K 和 V 向量, 再想小没有旋钮可拧',
    fix: '不存 K 和 V, 存一个能还原出它们的低维压缩件 c_kv, 外加一小段所有头共享的位置向量 k_rope; 要用时再乘回 K、V。教学配置下从 256 个数降到 96 个' },
  { name: 'DSA', year: 2025, color: 'var(--right)',
    pain: 'cache 压下去了, 但每个 query 仍要和全部 T 个 key 算分数: 上下文 128K 时算力按 $T^2$ 涨',
    fix: '先用一个很小的打分器 Lightning Indexer 给每个位置粗打分, 只留分最高的 k 个, 昂贵的注意力只在这 k 个位置上算: 算力从 $O(T^2)$ 降到 $O(T\\cdot k)$' },
]

const contextPresets = [
  { id: 'chat', label: '日常对话 · 4K', hint: '约 10–20 页文本', T: 4096 },
  { id: 'document', label: '长文档 · 32K', hint: '整篇论文或代码库片段', T: 32768 },
  { id: 'long-context', label: '超长上下文 · 128K', hint: '显存与计算差异被放大', T: 131072 },
]

const current = ref(variants[0])

const p = reactive({
  T: 4096,
  d_model: 4096,
  n_heads: 32,
  num_kv_heads: 8,
  kv_lora_rank: 512,
  qk_rope_head_dim: 64,
  sparse_top_k: 512,
})

const applyContextPreset = (preset) => {
  p.T = preset.T
}

const activePresetId = computed(() =>
  contextPresets.find((preset) => preset.T === p.T)?.id || 'custom'
)
const activePresetLabel = computed(() =>
  contextPresets.find((preset) => preset.id === activePresetId.value)?.label || '自定义上下文'
)

// 保证 num_kv_heads ≤ n_heads 且可整除
watch(() => p.n_heads, () => {
  if (p.num_kv_heads > p.n_heads) p.num_kv_heads = p.n_heads
  // 简单兼容: 找到最近的约数
  while (p.n_heads % p.num_kv_heads !== 0 && p.num_kv_heads > 1) p.num_kv_heads--
})

const cacheBytes = (v) => v.cache({
  T: p.T,
  d_model: p.d_model,
  n_heads: p.n_heads,
  num_kv_heads: v.id === 'gqa' ? p.num_kv_heads : v.id === 'mha' ? p.n_heads : 1,
  kv_lora_rank: p.kv_lora_rank,
  qk_rope_head_dim: p.qk_rope_head_dim,
}) * 2  // fp16

const cacheBytesPerToken = (v) => v.cache({
  T: 1,
  d_model: p.d_model,
  n_heads: p.n_heads,
  num_kv_heads: v.id === 'gqa' ? p.num_kv_heads : v.id === 'mha' ? p.n_heads : 1,
  kv_lora_rank: p.kv_lora_rank,
  qk_rope_head_dim: p.qk_rope_head_dim,
}) * 2

const mhaBytes = computed(() => variants[0].cache({ T: p.T, d_model: p.d_model, n_heads: p.n_heads }) * 2)
const maxBytes = computed(() => Math.max(...variants.map(cacheBytes)))
const barPct = (v) => (cacheBytes(v) / maxBytes.value) * 100
const relToMHA = (v) => {
  const ratio = cacheBytes(v) / mhaBytes.value
  if (ratio < 0.001) return (ratio * 1000).toFixed(2) + '‰'
  return (ratio * 100).toFixed(1) + '%'
}

const flopsPerQuery = computed(() => {
  if (current.value.id === 'dsa') return p.T * Math.min(p.sparse_top_k, p.T)
  return p.T * p.T
})

const conceptStory = computed(() => {
  const shared = {
    input: `${formatT(p.T)} tokens × ${p.d_model} 维`,
    inputNote: `每个 token 是一个 ${p.d_model} 维向量; 本层有 ${p.n_heads} 个 Q 头, 每个头各拿一段维度算一次注意力。换方案时上下文和模型宽度都不动。`,
    outcome: `${formatBytes(cacheBytes(current.value))} / 层`,
    outcomeNote: `每多生成 1 个 token, 本层就再写入 ${formatBytes(cacheBytesPerToken(current.value))}; 整段加起来相对 MHA 是 ${relToMHA(current.value)}。`,
  }

  const stories = {
    mha: {
      mechanism: `${p.n_heads} 个 Q 头各存一组 K/V`,
      mechanismNote: '每个头的 K、V 都是自己投影出来的, 不共享也不压缩。后面三种方案的百分比都拿它当 100%。',
      takeaway: 'MHA 表达力最满, 但上下文翻一倍, KV cache 就跟着翻一倍。',
    },
    gqa: {
      mechanism: `${p.n_heads} 个 Q 头共用 ${p.num_kv_heads} 组 K/V`,
      mechanismNote: `每 ${Math.max(1, p.n_heads / p.num_kv_heads).toFixed(0)} 个 Q 头排成一组, 一组共用同一对 K/V, 所以只要存 ${p.num_kv_heads} 对; Q 头一个没减, 注意力照样是 ${p.n_heads} 个头在算。`,
      takeaway: 'GQA 砍掉的是「重复存了好多份的 K/V 头」, 上下文没变短, 注意力公式也一行没改。',
    },
    mla: {
      mechanism: `K/V 先压成 ${p.kv_lora_rank} 维的 latent`,
      mechanismNote: `不存 K 和 V 本身, 只存一个 ${p.kv_lora_rank} 维的压缩向量 c_kv (叫 latent), 外加一段 ${p.qk_rope_head_dim} 维、所有头共用的位置向量 k_rope; 要用时再把 c_kv 乘上升维矩阵, 还原出各头的 K 和 V。`,
      takeaway: 'MLA 把「存完整 K/V」换成「存一个能还原出 K/V 的压缩件」, 多算一点, 换 cache 小一截。',
    },
    dsa: {
      mechanism: `MLA 的压缩照旧 + 每个 query 只挑 ${Math.min(p.sparse_top_k, p.T)} 个位置算`,
      mechanismNote: 'cache 和 MLA 一模一样, 每个旧位置的压缩件都得留着, 因为将来某个 query 可能选中它; 省的是每个 query 要和多少个位置算分数: 从全部 T 个变成 k 个。',
      takeaway: 'DSA 分两刀砍: MLA 那一刀砍显存, top-k 这一刀只砍超长上下文的算力, 不省 cache。',
    },
  }

  return { ...shared, ...stories[current.value.id] }
})

const formatT = (t) => t >= 1024 ? (t / 1024).toFixed(t % 1024 === 0 ? 0 : 1) + 'K' : String(t)
const formatBytes = (b) => {
  if (b < 1024) return b.toFixed(0) + ' B'
  if (b < 1024 * 1024) return (b / 1024).toFixed(1) + ' KiB'
  if (b < 1024 * 1024 * 1024) return (b / 1024 / 1024).toFixed(2) + ' MiB'
  return (b / 1024 / 1024 / 1024).toFixed(2) + ' GiB'
}
const formatFlops = (f) => {
  if (f < 1e6) return f.toFixed(0)
  if (f < 1e9) return (f / 1e6).toFixed(1) + 'M'
  return (f / 1e9).toFixed(2) + 'G'
}

// --- Attention 矩阵可视化 ---
const N = 24  // 显示用的小矩阵
const matrixSize = 320
const cellSize = matrixSize / N

const matrixCells = computed(() => {
  const cells = []
  for (let i = 0; i < N; i++) {
    // 每行的分数: 近距离 + 一些随机热点, 因果 mask 使 j > i 为 0
    const scores = new Array(N).fill(0)
    for (let j = 0; j <= i; j++) {
      const dist = i - j
      // 模拟真实 attention: 相邻衰减 + 几个高分
      scores[j] = Math.exp(-dist * 0.12) + 0.25 * Math.sin(j * 0.7 + i * 0.3) + 0.2
      scores[j] = Math.max(0, scores[j])
    }
    // 对当前 variant 做处理
    if (current.value.id === 'dsa') {
      const k = Math.min(p.sparse_top_k >> 5, Math.min(8, i + 1))  // 缩放到小矩阵
      // 保留 top-k 最大的
      const top = [...scores.map((s, idx) => ({ s, idx }))].sort((a, b) => b.s - a.s).slice(0, k)
      const keep = new Set(top.map(x => x.idx))
      for (let j = 0; j < N; j++) if (!keep.has(j)) scores[j] = 0
    }
    // 归一化 (softmax 简化)
    const sum = scores.reduce((a, b) => a + b, 0) || 1
    for (let j = 0; j < N; j++) {
      const w = scores[j] / sum
      cells.push({ i, j, color: colorFor(w * 2.5) })  // 放大可视度
    }
  }
  return cells
})

const matrixReading = computed(() => current.value.id === 'dsa'
  ? '怎么读: 右上角空着, 是因为看不到未来。左下三角里只剩零星几个色块, 那就是 indexer 挑出来的 top-k 个位置; 其余位置的分数被直接丢掉, 不参与 softmax。'
  : '怎么读: 右上角空着, 是因为看不到未来。左下三角某格颜色越深, 说明这个 token 越依赖那个历史位置; 对角线附近最深, 是因为示意数据让离得近的 token 分数更高。'
)

const matrixAriaLabel = computed(() =>
  `${current.value.name} 的 ${N}×${N} 因果注意力矩阵。${matrixReading.value}`
)

// 权重 0 (含被 mask 的格子) 用卡片底色, 越大越接近强调色。两个端点都是主题变量, 明暗主题都不刺眼
function colorFor(w) {
  return `color-mix(in srgb, var(--accent) ${Math.round(clamp(w, 0, 1) * 100)}%, var(--bg-card))`
}

// --- 代码片段 ---
const classFor = (id) => ({
  mha: 'MultiHeadAttention', gqa: 'GroupedQueryAttention',
  mla: 'MultiHeadLatentAttention', dsa: 'MultiHeadLatentSparseAttention',
}[id])

const codeSnippet = computed(() => {
  const snippets = {
    mha: `class MultiHeadAttention(nn.Module):
    """原始 MHA (Vaswani et al., 2017)"""
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_head = d_model // num_heads
        self.heads = nn.ModuleList([
            SingleHeadSelfAttention(d_model, self.d_head)
            for _ in range(num_heads)
        ])
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, q, k=None, v=None, mask=None):
        # KV cache: num_heads · d_head = d_model (满)
        outs = [h(q, k, v, mask) for h in self.heads]
        return self.w_o(torch.cat(outs, dim=-1))`,
    gqa: `class GroupedQueryAttention(nn.Module):
    """GQA — Ainslie et al., 2023 (LLaMA-2 70B)"""
    def __init__(self, d_model, num_heads, num_kv_heads=None):
        super().__init__()
        self.num_groups = num_heads // num_kv_heads  # 每组 Q 共享 1 对 KV
        self.w_q = nn.Linear(d_model, num_heads    * head_dim)
        self.w_k = nn.Linear(d_model, num_kv_heads * head_dim)  # ← 更小
        self.w_v = nn.Linear(d_model, num_kv_heads * head_dim)

    def forward(self, q, k=None, v=None, mask=None, rope=None):
        Q = self.w_q(q).view(B, T, num_heads,    head_dim)
        K = self.w_k(k).view(B, T, num_kv_heads, head_dim)
        # 复制 num_groups 份, 对齐 Q
        K = K.repeat_interleave(self.num_groups, dim=1)
        # ... 标准 QK^T · V`,
    mla: `class MultiHeadLatentAttention(nn.Module):
    """MLA — DeepSeek-V2/V3, KV 低秩 + 解耦 RoPE"""
    def __init__(self, d_model, num_heads, kv_lora_rank=512,
                 qk_nope_head_dim=64, qk_rope_head_dim=32):
        # KV 一次压成低秩 c_kv + 共享 k_rope
        self.kv_down = nn.Linear(d_model, kv_lora_rank + qk_rope_head_dim)
        self.k_up = nn.Linear(kv_lora_rank, num_heads * qk_nope_head_dim)
        self.v_up = nn.Linear(kv_lora_rank, num_heads * v_head_dim)
        # Q 也可低秩 (DeepSeek-V3 省训练显存)

    def forward(self, x, rope=None):
        kv_mix = self.kv_down(x)                    # [B, T, lora + rope]
        c_kv, k_rope = torch.split(kv_mix, [lora, rope], dim=-1)
        # nope 段走 latent, rope 段独立旋转
        # 推理: 只缓存 c_kv + k_rope, 约 MHA 的 7%`,
    dsa: `class MultiHeadLatentSparseAttention(nn.Module):
    """DSA — DeepSeek-V3.2 (2025), MLA + Lightning Indexer"""
    def __init__(self, d_model, num_heads, kv_lora_rank=512,
                 indexer_heads=4, sparse_top_k=128):
        self.mla = MultiHeadLatentAttention(...)
        self.indexer = LightningIndexer(d_model, indexer_heads)
        self.sparse_top_k = sparse_top_k

    def forward(self, q, mask=None, rope=None):
        # 1) 廉价 indexer 打分
        idx_scores = self.indexer(q, mask=mask)       # [B, T, S]
        # 2) 先按 mask 屏蔽不可见, 再取 top-k (否则可能泄漏未来)
        sparse_mask = top_k_mask(idx_scores, mask, self.sparse_top_k)
        # 3) MLA 只在 top-k 位置算 softmax
        return self.mla(q, mask=sparse_mask, rope=rope)`
  }
  return snippets[current.value.id]
})

</script>

<style scoped>
.concept-workbench {
  margin-bottom: 18px;
  border-color: color-mix(in srgb, var(--accent) 42%, var(--border));
  background:
    linear-gradient(135deg, color-mix(in srgb, var(--accent) 11%, transparent), transparent 48%),
    var(--bg-card);
  box-shadow: 0 14px 36px color-mix(in srgb, var(--accent) 10%, transparent);
}
.workbench-heading {
  display: grid;
  grid-template-columns: 150px minmax(0, 1fr);
  gap: 18px;
  align-items: start;
}
.eyebrow {
  color: var(--accent);
  font-family: "SF Mono", Menlo, monospace;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 1.1px;
}
.workbench-heading h2,
.bridge-heading h2 {
  margin: 0;
  font-size: 20px;
  line-height: 1.35;
  text-wrap: balance;
}
.workbench-heading p,
.bridge-heading p,
.bridge-heading .prose {
  margin-top: 6px;
  color: var(--text-muted);
  font-size: 13px;
  line-height: 1.7;
  text-wrap: pretty;
}
.learning-steps {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
  margin: 18px 0;
  list-style: none;
}
.learning-steps li {
  display: grid;
  grid-template-columns: 28px 1fr;
  gap: 0 10px;
  align-items: center;
  padding: 10px 12px;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: color-mix(in srgb, var(--bg-elev) 82%, transparent);
}
.learning-steps li > span {
  grid-row: 1 / 3;
  width: 28px;
  height: 28px;
  display: grid;
  place-items: center;
  border-radius: 50%;
  background: var(--accent-soft);
  color: var(--accent);
  font-family: "SF Mono", Menlo, monospace;
  font-size: 11px;
  font-weight: 700;
}
.learning-steps strong { font-size: 12.5px; }
.learning-steps small { color: var(--text-muted); font-size: 10.5px; }
.context-presets { border: 0; }
.context-presets legend {
  margin-bottom: 8px;
  color: var(--text-muted);
  font-size: 12px;
  font-weight: 600;
}
.preset-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 8px;
}
.preset-grid button {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  min-height: 58px;
  padding: 9px 12px;
  text-align: left;
}
.preset-grid button span { font-size: 12.5px; font-weight: 600; }
.preset-grid button small { margin-top: 2px; color: var(--text-muted); font-size: 10.5px; }
.preset-grid button.active small { color: rgba(255, 255, 255, 0.76); }
.preset-readout {
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  gap: 6px 10px;
  margin-top: 12px;
  padding: 10px 12px;
  border-left: 3px solid var(--accent);
  border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
  background: var(--code-bg);
  color: var(--text-muted);
  font-size: 12px;
}
.preset-readout strong { color: var(--text); }
.readout-kicker {
  color: var(--accent);
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.8px;
}

.variant-tabs {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 8px;
  margin-bottom: 12px;
  padding-bottom: 4px;
}
.variant-tabs button {
  display: grid;
  grid-template-columns: 24px 1fr auto;
  gap: 8px;
  align-items: center;
  min-height: 48px;
  padding: 8px 10px;
  text-align: left;
}
.variant-index {
  width: 24px;
  height: 24px;
  display: grid;
  place-items: center;
  border-radius: 50%;
  background: var(--bg-card);
  color: var(--text-muted);
  font-family: "SF Mono", Menlo, monospace;
  font-size: 10px;
}
.variant-name { font-weight: 650; }
.variant-tabs button .yr {
  font-size: 10px;
  color: var(--text-muted);
  font-family: "SF Mono", Menlo, monospace;
}
.variant-tabs button.active .yr { color: rgba(255,255,255,0.75); }
.variant-tabs button.active .variant-index { background: rgba(255, 255, 255, 0.18); color: #fff; }

.concept-bridge {
  --variant-color: var(--accent);
  margin-bottom: 20px;
  padding: 18px 20px;
  border: 1px solid color-mix(in srgb, var(--variant-color) 44%, var(--border));
  border-top: 3px solid var(--variant-color);
  border-radius: var(--radius);
  background: linear-gradient(
    135deg,
    color-mix(in srgb, var(--variant-color) 8%, var(--bg-card)),
    var(--bg-card) 55%
  );
}
.bridge-heading {
  display: grid;
  grid-template-columns: 150px minmax(0, 1fr);
  gap: 4px 18px;
}
.bridge-heading .eyebrow { grid-row: 1 / 3; color: var(--variant-color); }
.causal-flow {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr) auto minmax(0, 1fr);
  gap: 10px;
  align-items: stretch;
  margin-top: 16px;
}
.causal-flow article {
  padding: 12px 14px;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--bg-elev);
}
.causal-flow article.mechanism { border-color: color-mix(in srgb, var(--variant-color) 52%, var(--border)); }
.causal-flow article.outcome { background: color-mix(in srgb, var(--variant-color) 7%, var(--bg-elev)); }
.causal-label {
  display: block;
  margin-bottom: 4px;
  color: var(--text-muted);
  font-family: "SF Mono", Menlo, monospace;
  font-size: 9.5px;
  font-weight: 700;
  letter-spacing: 0.8px;
}
.causal-flow strong { display: block; font-size: 13px; text-wrap: balance; }
.causal-flow p { margin-top: 5px; color: var(--text-muted); font-size: 11.5px; line-height: 1.6; text-wrap: pretty; }
.causal-arrow { align-self: center; color: var(--variant-color); font-size: 18px; }
.concept-takeaway {
  display: grid;
  grid-template-columns: auto minmax(0, 1fr);
  gap: 12px;
  align-items: baseline;
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px dashed var(--border);
}
.concept-takeaway span { color: var(--variant-color); font-size: 10px; font-weight: 700; letter-spacing: 0.7px; }
.concept-takeaway strong { font-size: 12.5px; line-height: 1.7; text-wrap: pretty; }

.formula-box {
  margin-top: 16px;
  padding: 12px;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--code-bg);
  color: var(--code-text);
  font-size: 13.5px;
  line-height: 1.6;
  overflow-x: auto;
}
.parameter-feedback {
  margin-top: 10px;
  padding: 10px 12px;
  border-radius: var(--radius-sm);
  background: var(--accent-soft);
  color: var(--text-muted);
  font-size: 11.5px;
  line-height: 1.65;
}
.parameter-feedback strong { color: var(--text); font-variant-numeric: tabular-nums; }

.trade { font-size: 13px; color: var(--text-muted); }
.tr-item { display: flex; gap: 8px; align-items: flex-start; margin-top: 4px; }
.tr-item .ok { color: var(--left); }
.tr-item .no { color: var(--danger); }

.bars { display: flex; flex-direction: column; gap: 12px; }
.bar-row {
  display: grid;
  grid-template-columns: 110px 1fr 70px;
  gap: 12px;
  align-items: center;
  transition-property: margin, padding, background-color;
  transition-duration: 200ms;
  transition-timing-function: ease-out;
}
.bar-row.current { background: var(--bg-elev); margin: -6px -10px; padding: 6px 10px; border-radius: 6px; }
.bar-label { display: flex; flex-direction: column; gap: 2px; }
.bar-label .bar-name { font-size: 13px; font-weight: 500; }
.bar-label .bar-sub { font-size: 11px; color: var(--text-muted); }
.bar-track {
  height: 22px;
  background: var(--bg-elev);
  border-radius: 4px;
  overflow: hidden;
}
.bar-fill {
  height: 100%;
  transition: width 0.3s ease;
  border-radius: 4px;
}
.bar-rel { font-size: 12px; color: var(--text-muted); text-align: right; font-variant-numeric: tabular-nums; }

.hero-stat {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
  margin-top: 20px;
}

.matrix-grid {
  display: flex;
  gap: 24px;
  align-items: flex-start;
  flex-wrap: wrap;
}
.legend-attn { max-width: 320px; }
.legend-strip {
  display: flex;
  height: 14px;
  border-radius: 3px;
  overflow: hidden;
}
.legend-strip span { flex: 1; }
.legend-labels {
  display: flex;
  justify-content: space-between;
  font-size: 11px;
  color: var(--text-dim);
  margin-top: 4px;
}
.matrix-reading {
  margin-top: 12px;
  padding: 10px 12px;
  border-left: 3px solid var(--accent);
  background: var(--accent-soft);
  color: var(--text-muted);
  font-size: 11.5px;
  line-height: 1.65;
}

@media (max-width: 960px) {
  .workbench-heading,
  .bridge-heading { grid-template-columns: 1fr; }
  .bridge-heading .eyebrow { grid-row: auto; }
  .learning-steps,
  .preset-grid { grid-template-columns: 1fr; }
  .causal-flow {
    grid-template-columns: 1fr;
  }
  .causal-arrow { justify-self: center; transform: rotate(90deg); }
}

@media (max-width: 720px) {
  .variant-tabs { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .concept-workbench,
  .concept-bridge { padding: 16px; }
  .concept-takeaway { grid-template-columns: 1fr; gap: 4px; }
  .bar-row { grid-template-columns: 88px minmax(90px, 1fr) 58px; gap: 8px; }
  .hero-stat { grid-template-columns: 1fr; }
}
</style>
