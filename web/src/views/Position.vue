<template>
  <div>
    <h1 class="page-title">位置编码 · 从 Sinusoidal 到 RoPE 再到 M-RoPE</h1>
    <div class="page-subtitle lead-group">
      <p>
        注意力只看内容, 不看顺序: 一个 token 的 Q 和另一个 token 的 K 做内积得到分数, 分数里没有「谁在前谁在后」。
        所以不给位置编码, attention 就是一袋 token: 把句子里的词打乱顺序, 每个词算出来的结果一模一样 (本仓库的 attention demo 断言了这一点, 误差 8.9e-08)。
      </p>
      <p>
        <strong>RoPE 把 Q 和 K 各转一个角之后, <Tex text="$Q\cdot K$" /> 内积 (也就是注意力分数) 就只剩下「隔了多远」, 绝对位置自动消失</strong>。
        本章拖一根滑条就能看到这一点。
      </p>
    </div>

    <ChapterIntro
      tldr="- 位置信息的注入点一路往后挪, 越挪越靠近 attention 用它的地方。
- Sinusoidal / Learnable (2017 / 2018): 给每个位置配一个向量, 加到 token 的 embedding 上。编的是绝对位置: 第 5 个就是第 5 个。
- RoPE (2021): 不加向量, 改成把 Q 和 K 按位置各转一个角。两个转过的向量做内积, 角度相减, 只剩「隔了多远」。零参数, 位置超出训练长度也能转。
- M-RoPE (2024): 图像 patch 有行有列, 把旋转的维度切成三段, 分别按 (时间, 行, 列) 转。文本三个位置号相同, 结果和普通 RoPE 逐位相等 (demo 实测差 0.0e+00)。"
      question="RoPE 之后, 一个 query 和一个 key 的分数只看它们隔了多远 $(m-n)$, 跟各自是第几个 ($m$、$n$) 无关。那模型剩下要学的是什么? 丢掉绝对位置, 会不会有问题?"
      :goals="[
        '分清绝对位置 / 可学位置 / RoPE / M-RoPE 各自解决什么',
        '自己推一遍 RoPE 为什么让内积只剩相对距离',
        '说清多模态为什么要把位置按三个轴拆开',
      ]"
      :codes="[{ path: 'llm_models/layers/core/position_encoding.py' }]"
      :prereq="prevChapter"
      :next-step="nextChapter"
    />

    <EvolutionChain
      title="演进逻辑链 · 位置信息怎么 &quot;挪&quot; 进 attention"
      :steps="evoSteps"
    />

    <!-- variant 切换 -->
    <div class="btn-group" style="margin-bottom: 20px;" role="group" aria-label="选择位置编码方案">
      <button v-for="v in variants" :key="v.id" type="button"
              :class="{ active: variant === v.id }"
              :aria-pressed="variant === v.id"
              @click="variant = v.id">{{ v.label }}</button>
    </div>

    <div v-if="variant === 'sin'">
      <!-- Sinusoidal -->
      <div class="grid grid-2">
        <div class="card">
          <h3>Sinusoidal PE <span class="tag">加在 embedding 上</span></h3>
          <p class="desc">
            每个位置算一个固定向量, 直接加到 token 的 embedding 上 (embedding: 查表得到的、代表这个 token 本身的向量)。
            下面代码里的 pos 是第几个位置, 2i 是第几个维度: 维度 0 的正弦随 pos 抖得最快, 越往后的维度分母里 10000 的指数越大, 抖得越慢。
            不同维度用不同频率, 合起来每个位置都得到一组独一无二的数, 就是这个位置的「指纹」。
          </p>
          <p class="desc" style="margin: 6px 0 12px;">
            它编的是绝对位置: 第 5 个位置的指纹永远是那一组数。模型只在训练长度内见过这些指纹, 更远的位置虽然也能算出向量, 模型却没学过怎么用, 所以推不出去。
          </p>
          <CodeBlock :code="sinCode" />
        </div>
        <div class="card">
          <h3>频率谱 <span class="tag">d_model=64</span></h3>
          <ul class="desc pts" style="margin-bottom: 8px;">
            <li><b>图上是什么:</b> 画的是 d_model=64 的编码里前 16 个维度, 一条线是一个维度, 横轴是位置 0 到 79, 纵轴是这个维度的值 (正弦或余弦, 在 -1 到 1 之间)。</li>
            <li>
              <b>应该看到:</b>
              <ul class="pts">
                <li>靠前的维度波长短、抖得快, 位置挪一步值就变很多。</li>
                <li>靠后的维度波长长, 在 80 个位置内几乎是一条斜线。</li>
              </ul>
            </li>
          </ul>
          <svg viewBox="0 0 480 320" width="100%" height="320" role="img" aria-label="Sinusoidal 位置编码的 16 个维度随位置变化的曲线">
            <path v-for="(line, idx) in sinLines" :key="idx"
                  :d="line" fill="none"
                  :stroke="sinColor(idx)" stroke-width="1" stroke-opacity="0.7" />
          </svg>
        </div>
      </div>
    </div>

    <div v-else-if="variant === 'rope'">
      <!-- RoPE -->
      <div class="grid grid-2">
        <div class="card">
          <h3>RoPE 旋转可视化 <span class="tag">拖动滑条体验</span></h3>
          <p class="desc">RoPE 不往 embedding 里加东西, 它直接改 Q 和 K:</p>
          <ol class="desc pts">
            <li>把向量的维度两两配对。</li>
            <li>每一对 <Tex text="$(a, b)$" /> 当成一个复数 <Tex text="$z = a + ib$" />。</li>
            <li>在位置 <Tex text="$m$" /> 处乘上 <Tex text="$e^{im\theta}$" />。</li>
          </ol>
          <p class="desc">
            复数乘 <Tex text="$e^{i\phi}$" /> 就是在平面上把这个点转 <Tex text="$\phi$" /> 角、长度不变,
            所以这一步等于把每一对转了 <Tex text="$m\theta$" /> 这么大的角: 位置越靠后, 转得越多。
          </p>
          <ul class="desc pts" style="margin: 6px 0 16px;">
            <li><b>图上是什么:</b> 圆上的 4 个点是 4 个维度对 (d0、d2、d4、d6), 每对有自己的转速 <Tex text="$\theta_i$" />。图里的转速是示意值, 真实 RoPE 用 <Tex text="$\theta_i = 10000^{-2i/d}$" />, 越靠后的对越慢。</li>
            <li><b>可以做什么:</b> 拖「位置 m」从 0 到 64。</li>
            <li>
              <b>应该看到:</b>
              <ul class="pts">
                <li>m=0 时四个点都在正右方重合。</li>
                <li>m 变大后 d0 转得最快, 到 64 已经转了 5 圈多, d6 还没转过四分之一圈。</li>
              </ul>
            </li>
          </ul>

          <div class="form-row">
            <label for="position-rope-m">位置 m</label>
            <input id="position-rope-m" type="range" min="0" max="64" step="1" v-model.number="m" />
            <span class="val">{{ m }}</span>
          </div>

          <svg viewBox="-160 -160 320 320" width="100%" height="320" style="max-width: 360px; margin: 10px auto; display: block;"
               role="img" :aria-label="`位置 m = ${m} 时, 4 个维度对在复平面上各自转过的角度`">
            <!-- 网格 -->
            <circle cx="0" cy="0" r="100" fill="none" stroke="var(--border)" stroke-dasharray="2 4" />
            <line x1="-140" y1="0" x2="140" y2="0" stroke="var(--border)" />
            <line x1="0" y1="-140" x2="0" y2="140" stroke="var(--border)" />
            <text x="140" y="-6" font-size="10" fill="var(--text-dim)" font-family="SF Mono, monospace">Re</text>
            <text x="6" y="-140" font-size="10" fill="var(--text-dim)" font-family="SF Mono, monospace">Im</text>
            <!-- 4 个维度对, 不同频率, 初始在 (1,0) -->
            <g v-for="(d, i) in 4" :key="i">
              <line x1="0" y1="0"
                    :x2="100 * Math.cos(m * freq(i))"
                    :y2="-100 * Math.sin(m * freq(i))"
                    :stroke="dimColor(i)" stroke-width="2" />
              <circle :cx="100 * Math.cos(m * freq(i))"
                      :cy="-100 * Math.sin(m * freq(i))"
                      r="5" :fill="dimColor(i)" />
              <text :x="110 * Math.cos(m * freq(i))"
                    :y="-110 * Math.sin(m * freq(i))"
                    font-size="10" fill="var(--text-muted)"
                    font-family="SF Mono, monospace">d{{ i * 2 }}</text>
            </g>
          </svg>

          <ul class="desc pts" style="font-size: 12px;">
            <li><b>低维 (d0):</b> 频率高、转得快, 相邻两个 token 的角度差很大, 分得清谁先谁后。像秒针。</li>
            <li><b>高维 (d6):</b> 频率低、转得慢, 隔了很远角度才差出一点, 管的是远距离。像时针。</li>
          </ul>
          <p class="desc" style="font-size: 12px; margin-top: 6px;">
            长度外推 (让模型处理比训练时更长的序列) 出问题的是「时针」那一端: 低频的对在训练长度内连一圈都没转完, 位置一超出训练长度, 指针就指向从未见过的角度。
            修法叫 YaRN, 分两步:
          </p>
          <ol class="desc pts" style="font-size: 12px;">
            <li>
              调各对的转速:
              <ul class="pts">
                <li><b>转得慢的对:</b> 再放慢 s 倍 (s 是上下文要扩几倍)。</li>
                <li><b>转得快的对:</b> 原样不动。</li>
                <li><b>中间的:</b> 线性过渡。</li>
              </ul>
            </li>
            <li>再把注意力分数乘一个 <Tex text="$(0.1\ln s + 1)^2$" /> 的系数, 补偿长序列 softmax 变平。</li>
          </ol>
          <p class="desc" style="font-size: 12px;">本仓库 demo 用 s=4, 把越界 4 倍的低频维全拉回 1.00 倍。</p>
        </div>

        <div class="card">
          <h3>相对位置的关键证明 <span class="tag">核心直觉</span></h3>
          <p class="desc" style="margin-bottom: 12px;">
            把「转 <Tex text="$m\theta$" /> 角」写成矩阵 <Tex text="$R(m\theta)$" />。query 在位置 <Tex text="$m$" />, key 在位置 <Tex text="$n$" />, 各自转完再做内积, 绝对位置会被约掉, 只剩位置差 <Tex text="$(m-n)$" />:
          </p>
          <Prose text="$$\begin{aligned} Q_m &= R(m\theta)\cdot q \\ K_n &= R(n\theta)\cdot k \\ \langle Q_m, K_n \rangle &= \langle R(m\theta)\cdot q,\ R(n\theta)\cdot k \rangle \\ &= q^\top \cdot R(m\theta)^\top \cdot R(n\theta)\cdot k \\ &= q^\top \cdot R((n-m)\theta)\cdot k && \leftarrow \text{只依赖 } (n-m) \end{aligned}$$" />

          <p class="desc" style="margin-top: 12px;">
            这个式子在说: 先把 q 转 <Tex text="$m\theta$" />, 再把 k 转 <Tex text="$n\theta$" />, 两个点之间的夹角只差 <Tex text="$(n-m)\theta$" />。内积只和两个向量的长度、夹角有关, 旋转不改长度, 所以结果等于「不转 q, 只把 k 转 <Tex text="$(n-m)\theta$" />」。
            <strong>转过的 Q 和 K 一做内积, 相对距离就自己冒出来了</strong>, 绝对位置 <Tex text="$m$" />、<Tex text="$n$" /> 被约掉。
          </p>
          <p class="desc" style="margin-top: 6px;">
            整个过程没有一个可学参数, 位置往外推也不会遇到「没训过的参数」(会遇到的是上面说的没见过的角度), 所以现代 LLM 几乎全换成了 RoPE。
          </p>
          <ul class="desc pts" style="margin-top: 6px;">
            <li><b>可以做什么:</b> 下面两根滑条随便拖。读数是 d0 这一对的模拟相似度: 假设 q 和 k 都是 (1, 0), 各自转完再做内积, 就是 <Tex text="$\cos((n-m)\theta_0)$" />。</li>
            <li>
              <b>应该看到:</b>
              <ul class="pts">
                <li>两根一起挪, 只要 <Tex text="$|m-n|$" /> 不变, 读数就不动。</li>
                <li>只挪一根, 读数才变。</li>
              </ul>
            </li>
          </ul>

          <div class="form-row" style="margin-top: 16px;">
            <label for="position-q-pos">Q 位置 m</label>
            <input id="position-q-pos" type="range" min="0" max="32" step="1" v-model.number="qPos" />
            <span class="val">{{ qPos }}</span>
          </div>
          <div class="form-row">
            <label for="position-k-pos">K 位置 n</label>
            <input id="position-k-pos" type="range" min="0" max="32" step="1" v-model.number="kPos" />
            <span class="val">{{ kPos }}</span>
          </div>

          <div class="stat" style="margin-top: 10px;" aria-live="polite" aria-atomic="true">
            <div class="k">模拟相似度 (只看 d0 这一对)</div>
            <div class="v accent">{{ cosSim.toFixed(3) }}</div>
            <div class="hint">相对距离 <Tex text="$|m-n|$" /> = {{ Math.abs(qPos - kPos) }}; 与绝对位置 <Tex text="$m, n$" /> 无关</div>
          </div>
        </div>
      </div>
    </div>

    <div v-else>
      <!-- M-RoPE -->
      <div class="grid grid-2">
        <div class="card">
          <h3>M-RoPE: 三轴位置 <span class="tag">Qwen2-VL · 2024</span></h3>
          <p class="desc" style="margin-bottom: 16px;">
            图片进模型前会切成小方块 (patch), 每个 patch 当一个 token; 视频再多一个帧号。所以视觉 token 天生是个网格 (时间 × 高 × 宽), 一根一维的位置轴装不下。
            M-RoPE 把每个头的维度 head_dim 切成三段, 每段用对应轴的位置号各转各的, 于是同一个 attention 能同时喂两种输入:
          </p>
          <ul class="desc" style="padding-left: 16px; line-height: 1.9; list-style: disc;">
            <li><strong>文本 token</strong>: 三个位置号都填同一个 t, 写成 (t, t, t), 结果和普通一维 RoPE 逐位相等</li>
            <li><strong>视觉 patch</strong>: 三个号分别填 (帧号, 行号, 列号), 同一行的 patch 行号相同, 二维结构保住了</li>
          </ul>

          <div class="trade" style="margin-top: 16px;">
            <div class="tr-item">
              <span class="ok">✓</span>
              <span>图和文共用一个 decoder, 不用再加一套 cross-attention (让文本去查另一个视觉编码器输出的额外注意力层)</span>
            </div>
            <div class="tr-item">
              <span class="ok">✓</span>
              <span>动态分辨率随便来: patch 网格多大都是同一套编码, 行号列号接着数就行</span>
            </div>
          </div>

          <p class="desc" style="margin-top: 14px; font-size: 12px;">
            「纯文本下退化成 1-D RoPE」有个前提: 三轴共用同一条频率轴。RoPE 一共有 head_dim/2 个转速, 从快到慢排成一列; M-RoPE 是把这一列按段切开分给 T/H/W, 而不是每段各建一列。
          </p>
          <ul class="desc pts" style="font-size: 12px; margin-top: 6px;">
            <li><b>共用频率轴:</b> 三个号相同时, 每一对拿到的角度和一维 RoPE 完全一样。demo 实测纯文本下与 1-D RoPE 数值差 0.0e+00, 写进了断言。</li>
            <li><b>不共用:</b> 每段都从最快的转速起算, 文本的角度就对不上。纯文本下差 3.3e-3。</li>
          </ul>
          <p class="desc" style="font-size: 12px; margin-top: 6px;">
            代价是 H/W 分到的是这一列里靠后、较慢的转速。4×4 的小网格上, 未训练时对调行列坐标, 输出只变约 1e-5~1e-4 (demo 里是 5.3e-05), 不是零, 但很小。
          </p>
        </div>

        <div class="card">
          <h3>位置索引示意</h3>
          <p class="desc" style="margin-bottom: 12px;">文本「Hi」后面接一张 3×3 的 patch 网格, 再接文本「there」。每个格子下面是这个 token 拿到的三个位置号 (t, h, w):</p>
          <div class="mrope-grid mono">
            <!-- 文本 "Hi" -->
            <div v-for="(tok, i) in ['H', 'i']" :key="'t1-'+i" class="tok text">
              <div class="ch">{{ tok }}</div>
              <div class="ids">({{ i }}, {{ i }}, {{ i }})</div>
            </div>
            <!-- 3x3 视觉 grid -->
            <div v-for="r in 3" :key="'row-'+r" class="viz-row">
              <div v-for="c in 3" :key="'cell-'+r+c" class="tok viz">
                <div class="ch">■</div>
                <div class="ids">(2, {{ r - 1 }}, {{ c - 1 }})</div>
              </div>
            </div>
            <!-- 后续文本 "there" -->
            <div v-for="(tok, i) in ['t', 'h', 'e', 'r', 'e']" :key="'t2-'+i" class="tok text">
              <div class="ch">{{ tok }}</div>
              <div class="ids">({{ 3 + i }}, {{ 3 + i }}, {{ 3 + i }})</div>
            </div>
          </div>
          <ul class="desc pts" style="font-size: 12px; margin-top: 10px;">
            <li><b>H、i:</b> 第 0、1 个 token, 三个号都填自己的序号。</li>
            <li><b>整张图:</b> 算时间轴上的一个位置 2, 所以 9 个 patch 的 t 都是 2, h 和 w 按它在网格里的行、列填 0 到 2。</li>
            <li><b>后面的文本:</b> 从 3 接着数, 三个号又回到相同。</li>
          </ul>
        </div>
      </div>
    </div>

    <section class="section">
      <h2>代码路径</h2>
      <p class="lead">这三个类都在 <RepoLink path="llm_models/layers/core/position_encoding.py" label="llm_models/layers/core/position_encoding.py" tiny /> 里, 在 <RepoLink path="llm_models/layers/core/" label="core/" tiny /> 子包下。任何一个 Transformer 都得从中挑一个用。</p>
      <div class="grid grid-3">
        <div class="card"><h3 style="font-size: 13px;">SinPositionalEncoding</h3><p class="desc">原始 Transformer 那一版: 按公式算好一张位置表, 加在 embedding 上。编的是绝对位置, 推不到训练长度之外。</p></div>
        <div class="card"><h3 style="font-size: 13px;">RotaryPositionalEncoding</h3><p class="desc">按位置旋转 Q 和 K, 内积只剩相对距离。零可学参数, 长度能往外推; 配上面说的 YaRN 缩放 (同一个类的 scaling 参数) 能把上下文扩到训练长度的 4 到 32 倍。</p></div>
        <div class="card"><h3 style="font-size: 13px;">MultimodalRotaryEmbedding</h3><p class="desc">把 head_dim 切成 T/H/W 三段各转各的, 内部复用上面的 RoPE 类和它的频率表; 文本三轴位置号相同时, 结果和标准 RoPE 逐位相等 (文件末尾的自检断言了这一点)。</p></div>
      </div>
    </section>

    <!-- 本章挂载的实验台 (data/labmap/*.js) 与章末自测 (data/quiz/*.js), 没配置时不渲染 -->

    <LabMount :names="[]" />
    <TopicSupplement v-for="section in topicPages.position?.sections || []" :key="section.id" :section="section" />

    <QuizCard />


    <ChapterNav
      :prev="{ ...prevChapter, hint: 'KV cache 的瓶颈与解法' }"
      :next="{ ...nextChapter, hint: '把 attention + ffn + norm + pos 拼起来, 数模型差异' }"
    />
  </div>
</template>

<script setup>
import TopicSupplement from '@/components/TopicSupplement.vue'
import { topicPages } from '@/data/models.js'
import LabMount from '@/components/LabMount.vue'
import QuizCard from '@/components/QuizCard.vue'
import { ref, computed } from 'vue'
import ChapterIntro from '@/components/ChapterIntro.vue'
import ChapterNav from '@/components/ChapterNav.vue'
import EvolutionChain from '@/components/EvolutionChain.vue'
import RepoLink from '@/components/RepoLink.vue'
import Prose from '@/components/Prose.vue'
import Tex from '@/components/Tex.vue'
import CodeBlock from '@/components/CodeBlock.vue'
import { learningPath } from '@/data/models.js'

// 上一章 / 下一章从 learningPath 取, 不手写章名和编号 (与 Infer.vue 同一写法)
const at = learningPath.findIndex((x) => x.route === 'position')
const prevChapter = { name: learningPath[at - 1].route, label: `上一章 · ${learningPath[at - 1].label}` }
const nextChapter = { name: learningPath[at + 1].route, label: `下一章 · ${learningPath[at + 1].label}` }

const variants = [
  { id: 'sin',   label: 'Sinusoidal (2017)' },
  { id: 'rope',  label: 'RoPE (2021)' },
  { id: 'mrope', label: 'M-RoPE (2024)' },
]
const variant = ref('rope')

const evoSteps = [
  { name: 'Sinusoidal', year: 2017, color: 'var(--text-muted)',
    pain: '(原点) attention 只看内容不看顺序: 不给位置信息, 把句子打乱, 每个 token 算出来的结果一模一样',
    fix: '给每个位置算一个固定向量 (不同维度用不同频率的正弦、余弦), 加到 token 的 embedding 上。编的是绝对位置: 第 5 个就是第 5 个' },
  { name: 'Learnable', year: 2018, color: 'var(--accent)',
    pain: 'Sin 的向量是写死的公式算出来的, 不一定贴合数据',
    fix: '每个位置配一个可以训练的向量, 像查词表一样查位置 (BERT / ViT)。但表只有训练长度那么长, 之外的位置从没被训过' },
  { name: 'RoPE', year: 2021, color: 'var(--left)',
    pain: '训练长度之外推不出去; 而且位置加在 embedding 上, 会一路带进 V, 可 V 装的是内容, 并不需要位置',
    fix: '不加向量, 改成旋转: 把 Q、K 每两维当一个复数, 位置 $m$ 处转 $m\\theta$ 角。两个转过的向量做内积, 角度相减, 只剩距离 $(m-n)$。零参数, 能往外推' },
  { name: 'M-RoPE', year: 2024, color: 'var(--eye)',
    pain: '图像切成的 patch 排成二维网格, 有行有列, 一根一维的位置轴装不下',
    fix: '把每个头的维度切成三段, 分别按 (时间, 行, 列) 三个位置号旋转。文本 token 三个号相同, 结果和普通 RoPE 逐位相等, 于是图和文能共用一个 decoder' },
]

// --- Sinusoidal 曲线 ---
const sinLines = computed(() => {
  const lines = []
  const W = 480, H = 320, dModel = 16, positions = 80
  for (let i = 0; i < dModel; i++) {
    const freq = 1 / Math.pow(10000, (2 * Math.floor(i / 2)) / 64)
    const pts = []
    for (let p = 0; p < positions; p++) {
      const v = i % 2 === 0 ? Math.sin(p * freq) : Math.cos(p * freq)
      const x = (p / (positions - 1)) * W
      const y = H / 2 + (H / 2 - 16) * -v * 0.85 * (1 - i * 0.04)
      pts.push(`${p === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`)
    }
    lines.push(pts.join(' '))
  }
  return lines
})
const sinColor = (idx) => {
  const hues = [175, 195, 220, 250, 280, 310, 340, 10]
  return `hsl(${hues[idx % hues.length]}, 70%, 65%)`
}

// --- RoPE 旋转 ---
const m = ref(12)
const qPos = ref(5)
const kPos = ref(9)
// 4 个维度对的频率 (对数衰减)
const freq = (i) => 0.5 * Math.pow(0.3, i) // i=0 快, i=3 慢
const dimColor = (i) => ['var(--accent)', 'var(--left)', 'var(--eye)', 'var(--right)'][i]

// 模拟相似度: 假设 q = k = (1, 0) 向量, 旋转后内积 = cos(差角)
const cosSim = computed(() => {
  const diff = kPos.value - qPos.value
  return Math.cos(diff * freq(0))
})

// --- 代码 ---
const sinCode = `def forward(x):
    # pe[pos, 2i  ] = sin(pos / 10000^(2i/d))
    # pe[pos, 2i+1] = cos(pos / 10000^(2i/d))
    return x + pe[:, :x.size(1)]`

</script>

<style scoped>
.mrope-grid {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
  padding: 14px;
  background: var(--bg-elev);
  border-radius: 6px;
}
.viz-row {
  display: inline-flex;
  gap: 6px;
  flex-direction: row;
}
.tok {
  min-width: 50px;
  padding: 6px 8px;
  border-radius: 5px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  text-align: center;
  font-size: 11px;
}
.tok.viz { border-color: color-mix(in srgb, var(--eye) 50%, var(--border)); }
.tok.text { border-color: color-mix(in srgb, var(--accent) 40%, var(--border)); }
.tok .ch { font-size: 14px; color: var(--text); margin-bottom: 3px; }
.tok .ids { font-size: 10px; color: var(--text-muted); }
</style>
