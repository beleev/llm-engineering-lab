<template>
  <div>
    <h1 class="page-title">位置编码 · 从 Sinusoidal 到 RoPE 再到 M-RoPE</h1>
    <div class="page-subtitle lead-group">
      <p>
        注意力本身只看内容、不看顺序: Q 和 K 做内积算出相关性分数, 式子里没有「谁在前谁在后」。
        如果不注入位置信息, 把句子里的词打乱顺序, 每个词算出的注意力输出完全不变。
      </p>
      <p>
        <strong>RoPE 的做法是按位置把 Q 和 K 各转一个角度: 做完 <Tex text="$Q\cdot K$" /> 内积后绝对位置自动抵消, 分数只取决于两个 token 隔了多远</strong>。
        拖动下方滑条就能直观看到这一几何性质。
      </p>
    </div>

    <ChapterIntro
      tldr="- 演进主线: 位置信息的注入点逐步后移, 从输入层的 token embedding 直接推进至注意力层内部的 Q/K 内积运算。
- Sinusoidal / Learnable (2017 / 2018): 为每个绝对序号分配固定或可训练的位置向量并直接叠加至 token embedding; 编码的是绝对位置索引, 难以自然外推至训练长度之外。
- RoPE (2021): 舍弃加性位置向量, 改为在复平面上按位置索引对 Q 与 K 施加旋转变换; 旋转后的 Q 与 K 做内积时绝对位置角自动抵消, 注意力分数仅依赖相对距离, 且无额外可学习参数。
- M-RoPE (2024): 面向视觉二维网格与视频时间轴, 将单头特征维度切分为三段并分别按 (时间, 行, 列) 坐标旋转; 纯文本输入下三轴坐标一致, 输出与一维 RoPE 严格逐位相等 (本仓库 demo 实测最大数值误差为 0.0e+00)。"
      question="引入 RoPE 后, query 与 key 的内积分数仅取决于相对距离 $(m-n)$, 而与各自的绝对位置 $m$、$n$ 无关。在此机制下注意力参数主要学习哪类特征交互? 消除显式绝对位置是否会带来表达局限?"
      :goals="[
        '厘清正弦绝对位置编码、可学习位置编码、RoPE 与 M-RoPE 各自针对的核心问题',
        '独立推导 RoPE 旋转矩阵使 Q/K 内积仅依赖相对距离 $(m-n)$ 的代数性质',
        '阐明多模态模型将旋转维度按时间、行、列三轴拆分并共享频率表的工程原因',
      ]"
      :codes="[{ path: 'llm_models/layers/core/position_encoding.py' }]"
      :prereq="prevChapter"
      :next-step="nextChapter"
    />

    <EvolutionChain
      title="演进逻辑链 · 位置信息注入点如何逐步移入注意力层"
      subtitle="根问题: 自注意力内积对输入顺序具备置换不变性, 模型需在兼顾长度外推与多维拓扑的前提下编码位置关系。"
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
          <h3>Sinusoidal PE <span class="tag">叠加于 embedding</span></h3>
          <p class="desc">
            正弦位置编码为每个位置索引生成固定的确定性向量, 直接与输入层的 token embedding 相加。
            下方代码中, pos 表示序列位置序号, 2i 与 2i+1 表示特征维度索引: 维度 0 对应的正弦波随 pos 变化频率最高; 维度索引越大, 分母中基数 10000 的幂次越高, 波长随之拉长、数值变化趋于平缓。
            多组几何级数频率共同构成每个位置唯一的编码特征。
          </p>
          <p class="desc" style="margin: 6px 0 12px;">
            该方案直接编码绝对位置序号 (例如位置 5 始终对应同一组固定向量)。由于模型参数仅在训练序列长度范围内拟合过这些绝对位置特征, 推理时即便能够按公式算出更远位置的编码向量, 模型也难以可靠泛化至训练长度之外。
          </p>
          <CodeBlock :code="sinCode" />
        </div>
        <div class="card">
          <h3>频率谱 <span class="tag">d_model 为 64</span></h3>
          <ul class="desc pts" style="margin-bottom: 8px;">
            <li><b>图上是什么:</b> 绘制 d_model 为 64 时前 16 个维度的编码曲线, 每条曲线对应一个特征维度; 横轴为位置索引 0 至 79, 纵轴为对应维度的正弦或余弦取值 (介于 -1 到 1 之间)。</li>
            <li>
              <b>应该看到:</b>
              <ul class="pts">
                <li>低索引维度波长短、频率高, 相邻位置之间的编码值差异显著。</li>
                <li>高索引维度波长显著拉长, 在 80 个位置的观测窗口内近似呈平缓单调变化。</li>
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
          <p class="desc">RoPE 不向 token embedding 叠加位置向量, 而是直接在注意力层内对 Q 与 K 施加旋转变换:</p>
          <ol class="desc pts">
            <li>将向量的相邻特征维度两两配对。</li>
            <li>将每一对二维分量 <Tex text="$(a, b)$" /> 视作复平面上的复数 <Tex text="$z = a + ib$" />。</li>
            <li>在序列位置 <Tex text="$m$" /> 处乘以单位复数 <Tex text="$e^{im\theta}$" />。</li>
          </ol>
          <p class="desc">
            复数乘以 <Tex text="$e^{i\phi}$" /> 等价于在二维平面上保持模长不变并逆时针旋转 <Tex text="$\phi$" /> 角,
            因此该操作将每对维度按位置旋转 <Tex text="$m\theta$" /> 角: 位置索引越靠后, 累积旋转角越大。
          </p>
          <ul class="desc pts" style="margin: 6px 0 16px;">
            <li><b>图上是什么:</b> 单位圆上的 4 个点分别对应 4 个维度对 (d0、d2、d4、d6), 各自具备不同的角速度 <Tex text="$\theta_i$" />。图中角速度为便于观察设定的示意值; 标准 RoPE 采用 <Tex text="$\theta_i = 10000^{-2i/d}$" />, 维度对索引越靠后角速度越慢。</li>
            <li><b>可以做什么:</b> 在 0 至 64 范围内拖动「位置 m」滑杆。</li>
            <li>
              <b>应该看到:</b>
              <ul class="pts">
                <li>位置 m 为 0 时, 四个维度对均在单位圆正右方重合。</li>
                <li>随 m 增大, d0 旋转最快, 到达位置 64 时已累计旋转超过 5 圈, 而 d6 尚未转满四分之一圈。</li>
              </ul>
            </li>
          </ul>

          <div class="form-row">
            <label for="position-rope-m">位置 m</label>
            <input id="position-rope-m" type="range" min="0" max="64" step="1" v-model.number="m" />
            <span class="val">{{ m }}</span>
          </div>

          <svg viewBox="-160 -160 320 320" width="100%" height="320" style="max-width: 360px; margin: 10px auto; display: block;"
               role="img" :aria-label="`位置 m 为 ${m} 时, 4 个维度对在复平面上各自转过的角度`">
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
            <li><b>低维分量 (d0):</b> 频率高、角速度大, 相邻 token 之间即可拉开显著相位差, 主要分辨局部先后顺序 (类似秒针)。</li>
            <li><b>高维分量 (d6):</b> 频率低、角速度小, 相隔较远才积累出明显夹角, 主要编码长程距离关系 (类似时针)。</li>
          </ul>
          <p class="desc" style="font-size: 12px; margin-top: 6px;">
            长度外推 (推理序列超出训练最大长度) 的失效根源在于低频分量: 这部分维度对在训练长度内尚未旋转满一整圈, 一旦位置越界便会进入模型从未见过的相位区间。
            YaRN 通过以下两步修正实现长度外推:
          </p>
          <ol class="desc pts" style="font-size: 12px;">
            <li>
              按频率分段调节各维度对的角速度:
              <ul class="pts">
                <li><b>低频维度对:</b> 将角速度按上下文扩展倍数 s 放慢 (除以 s)。</li>
                <li><b>高频维度对:</b> 保持原始角速度不变。</li>
                <li><b>中频维度对:</b> 在两者之间做平滑线性插值过渡。</li>
              </ul>
            </li>
            <li>将注意力分数乘以系数 <Tex text="$(0.1\ln s + 1)^2$" />, 补偿长序列下 softmax 分布趋于平坦的效应。</li>
          </ol>
          <p class="desc" style="font-size: 12px;">本仓库 demo 在扩展倍数 s 为 4 的设定下, 将原本越界达 4 倍的低频维度相位范围全部拉回 1.00 倍以内。</p>
        </div>

        <div class="card">
          <h3>相对位置性质的代数推导 <span class="tag">核心直觉</span></h3>
          <p class="desc" style="margin-bottom: 12px;">
            将旋转 <Tex text="$m\theta$" /> 角表示为正交旋转矩阵 <Tex text="$R(m\theta)$" />。设 query 位于位置 <Tex text="$m$" />, key 位于位置 <Tex text="$n$" />, 各自旋转后再计算内积, 正交矩阵转置相乘使绝对位置抵消, 仅保留相对位置差 <Tex text="$(n-m)$" />:
          </p>
          <Prose text="$$\begin{aligned} Q_m &= R(m\theta)\cdot q \\ K_n &= R(n\theta)\cdot k \\ \langle Q_m, K_n \rangle &= \langle R(m\theta)\cdot q,\ R(n\theta)\cdot k \rangle \\ &= q^\top \cdot R(m\theta)^\top \cdot R(n\theta)\cdot k \\ &= q^\top \cdot R((n-m)\theta)\cdot k && \leftarrow \text{只依赖 } (n-m) \end{aligned}$$" />

          <p class="desc" style="margin-top: 12px;">
            从几何上看, 先把 q 转 <Tex text="$m\theta$" />, 再把 k 转 <Tex text="$n\theta$" />, 两者的夹角只差 <Tex text="$(n-m)\theta$" />。内积只取决于向量的长度与夹角, 而旋转不改变长度, 所以结果等价于「保持 q 不动, 只把 k 转 <Tex text="$(n-m)\theta$" />」。
            <strong>旋转后的 Q 和 K 一做内积, 绝对位置 <Tex text="$m$" />、<Tex text="$n$" /> 自动抵消, 只留下相对距离</strong>。
          </p>
          <p class="desc" style="margin-top: 6px;">
            整个旋转变换不含任何可学习参数, 序列加长时不会遭遇未经过训练的位置参数 (仅需借助上述 YaRN 等方法处理低频未见相位角), 因此现代主流大语言模型普遍采用 RoPE。
          </p>
          <ul class="desc pts" style="margin-top: 6px;">
            <li><b>可以做什么:</b> 任意拖动下方两根位置滑杆。面板读数为 d0 维度对的模拟内积相似度: 固定初始向量 q 与 k 均为 (1, 0), 各自旋转后做内积即得 <Tex text="$\cos((n-m)\theta_0)$" />。</li>
            <li>
              <b>应该看到:</b>
              <ul class="pts">
                <li>同步平移两根滑杆, 只要相对距离 <Tex text="$|m-n|$" /> 保持不变, 相似度读数恒定不变。</li>
                <li>单独移动其中一根滑杆使 <Tex text="$|m-n|$" /> 改变时, 相似度读数才随之变化。</li>
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
            <div class="k">模拟相似度 (仅取 d0 维度对)</div>
            <div class="v accent">{{ cosSim.toFixed(3) }}</div>
            <div class="hint">相对距离 <Tex text="$|m-n|$" /> 为 {{ Math.abs(qPos - kPos) }}; 与绝对位置 <Tex text="$m, n$" /> 无关</div>
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
            图像输入模型前会被切分为二维局部块 (patch), 每个 patch 作为一个视觉 token; 视频输入在此基础上还包含帧序号。因此视觉 token 天然排列在三维网格 (时间 × 高 × 宽) 上, 单一的一维位置轴无法保留其空间拓扑结构。
            M-RoPE 将每个注意力头的特征维度 head_dim 切分为三段, 各段分别使用对应坐标轴的位置索引独立旋转, 从而令同一套自注意力层兼容两类模态输入:
          </p>
          <ul class="desc" style="padding-left: 16px; line-height: 1.9; list-style: disc;">
            <li><strong>文本 token</strong>: 三轴坐标填入相同序号 t, 记为 (t, t, t), 计算结果与标准一维 RoPE 严格逐位相等。</li>
            <li><strong>视觉 patch</strong>: 三轴坐标分别填入 (帧号, 行号, 列号), 同行 patch 共享相同行号, 完整保留二维网格邻接结构。</li>
          </ul>

          <div class="trade" style="margin-top: 16px;">
            <div class="tr-item">
              <span class="ok">✓</span>
              <span>图像与文本共享同一个 decoder, 无需额外引入专用的跨模态交叉注意力层 (cross-attention)</span>
            </div>
            <div class="tr-item">
              <span class="ok">✓</span>
              <span>原生支持动态分辨率输入: 任意尺寸的 patch 网格均复用同一套编码规则, 行列坐标按实际尺寸顺延即可</span>
            </div>
          </div>

          <p class="desc" style="margin-top: 14px; font-size: 12px;">
            「纯文本输入精确退化为 1-D RoPE」依赖一个关键实现前提: 三轴必须共享同一条频率序列。标准 RoPE 共含 head_dim/2 个由快到慢排列的旋转频率; M-RoPE 将该完整频率序列按段切分后分配给 T、H、W 三轴, 而非为每段各自重建独立的频率序列。
          </p>
          <ul class="desc pts" style="font-size: 12px; margin-top: 6px;">
            <li><b>共享频率序列:</b> 当三轴位置索引相同时, 每个维度对获得的旋转角与一维 RoPE 完全一致; 本仓库 demo 实测纯文本下与 1-D RoPE 的数值差异为 0.0e+00, 并已写入断言。</li>
            <li><b>独立重建频率序列:</b> 若每段均从最高频率重新起算, 纯文本各维度的旋转角将偏离一维 RoPE, 实测输出差异达 3.3e-3。</li>
          </ul>
          <p class="desc" style="font-size: 12px; margin-top: 6px;">
            这一切分方式的代价在于 H 与 W 轴分配到的是频率序列后段较低的角速度: 在 4×4 的小尺寸网格上且模型未经训练时, 对调行列坐标仅引起约 1e-5 至 1e-4 量级的输出变化 (demo 实测为 5.3e-05), 虽非严格为零但初始幅度较小。
          </p>
        </div>

        <div class="card">
          <h3>位置索引示意</h3>
          <p class="desc" style="margin-bottom: 12px;">以下序列依次包含文本「Hi」、一张 3×3 的视觉 patch 网格与后续文本「there」。每个方格下方标注该 token 分配到的三轴位置索引 (t, h, w):</p>
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
            <li><b>H、i:</b> 位于序列第 0、1 位, 三轴位置索引均填入各自的一维序号。</li>
            <li><b>整张图像:</b> 在时间轴上占据单一位置 2, 因此全部 9 个视觉 patch 的 t 坐标均为 2, h 与 w 则按网格内的行、列索引依次取 0 至 2。</li>
            <li><b>后续文本:</b> 从位置 3 起继续顺序递增, 三轴位置索引恢复一致。</li>
          </ul>
        </div>
      </div>
    </div>

    <section class="section">
      <h2>代码路径</h2>
      <p class="lead">以下三个位置编码实现类均位于 <RepoLink path="llm_models/layers/core/position_encoding.py" label="llm_models/layers/core/position_encoding.py" tiny /> 中 (隶属 <RepoLink path="llm_models/layers/core/" label="core/" tiny /> 子包), 构成仓库内各代 Transformer 模型的位置编码底座。</p>
      <div class="grid grid-3">
        <div class="card"><h3 style="font-size: 13px;">SinPositionalEncoding</h3><p class="desc">原始 Transformer 方案: 预先按正余弦公式构建静态位置编码表并叠加至输入 embedding; 编码绝对位置序号, 难以直接外推至训练长度之外。</p></div>
        <div class="card"><h3 style="font-size: 13px;">RotaryPositionalEncoding</h3><p class="desc">按位置索引对 Q 与 K 施加复平面旋转, 使内积仅依赖相对距离。无额外可学习参数且具备长度外推基础; 配合同类的 scaling 参数启用 YaRN 缩放后, 可将有效上下文扩展至训练长度的 4 至 32 倍。</p></div>
        <div class="card"><h3 style="font-size: 13px;">MultimodalRotaryEmbedding</h3><p class="desc">将 head_dim 切分为 T/H/W 三段分别施加旋转, 内部直接复用标准 RoPE 类及其统一频率表; 当纯文本三轴位置索引一致时, 输出与标准一维 RoPE 严格逐位相等 (已由文件末尾的自检断言验证)。</p></div>
      </div>
    </section>

    <!-- 本章挂载的实验台 (data/labmap/*.js) 与章末自测 (data/quiz/*.js), 没配置时不渲染 -->

    <LabMount :names="[]" />
    <TopicSupplement v-for="section in topicPages.position?.sections || []" :key="section.id" :section="section" />

    <QuizCard />


    <ChapterNav
      :prev="{ ...prevChapter, hint: '回顾注意力机制与 KV cache 压缩路线' }"
      :next="{ ...nextChapter, hint: '将注意力、FFN、归一化与位置编码组装为完整架构, 对比主流模型差异' }"
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
    pain: '(原点) 自注意力内积仅度量内容相似度而不感知顺序; 若缺失位置信息, 打乱词序后各 token 的注意力输出完全不变。',
    fix: '按不同几何频率的正弦与余弦函数为每个位置生成固定编码向量, 直接叠加至 token embedding; 编码的是绝对位置序号。' },
  { name: 'Learnable', year: 2018, color: 'var(--accent)',
    pain: '正弦位置编码由手工设计的固定函数生成, 无法根据下游任务与语料分布自适应调整。',
    fix: '为每个位置分配可训练参数向量并按位置索引查表叠加 (如 BERT 与 ViT); 但位置表受限于最大训练长度, 超出范围的位置索引未经过训练, 无法直接外推。' },
  { name: 'RoPE', year: 2021, color: 'var(--left)',
    pain: '加性绝对位置编码难以向训练长度外推, 且叠加在 embedding 上的位置信号会混入承载语义内容的 V 向量。',
    fix: '将 Q、K 相邻维度两两配对视作复数, 在位置 $m$ 处旋转 $m\\theta$ 角; 旋转后的 Q、K 内积使角度相减, 仅保留相对距离 $(m-n)$, 零可学习参数且支持长度外推。' },
  { name: 'M-RoPE', year: 2024, color: 'var(--eye)',
    pain: '图像切分后的视觉 patch 呈二维行列网格 (视频还包含帧轴), 单一的一维线性位置索引无法忠实表达空间邻接关系。',
    fix: '将单头维度切分为三段, 分别按 (时间, 行, 列) 坐标独立旋转; 纯文本三轴坐标相同时精确退化为一维 RoPE, 使图像、视频与文本得以共享同一套 decoder。' },
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
