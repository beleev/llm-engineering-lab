<template>
  <div>
    <h1 class="page-title">MoE 路由可视化</h1>
    <div class="page-subtitle lead-group">
      <p>
        Transformer 层里的前馈网络 (FFN) 承载了模型绝大部分参数容量。但在传统稠密结构里, 每个 token 都必须算遍全部 FFN 参数, 想靠扩参数提升容量, 单 token 的计算成本也会同比例增加。
      </p>
      <p>
        混合专家模型 (MoE) 把单个大 FFN 拆成 E 个独立的专家网络, 再用一个轻量级路由器 (router) 为每个 token 挑出得分最高的 k 个专家参与计算, 从而把「总参数容量」与「单步激活算力」解耦。本页并排对比工业界最主流的两条路由路线: <strong>Mixtral (大粒度专家 + softmax 竞争 + 辅助均衡损失)</strong> 与 <strong>DeepSeek-V3 (细粒度专家 + 共享专家 + 无辅助损失偏置均衡)</strong>。
      </p>
    </div>

    <ChapterIntro
      tldr="MoE 用 E 个专家撑起总参数量, 但每个 token 只激活 top-k 个专家来省算力。为了防止少数热门专家包揽流量 (路由坍缩), Mixtral 在训练目标上外挂辅助损失 (aux loss) 强行拉平负载, 而 DeepSeek-V3 改用细粒度专家加常驻共享专家, 并通过一个不参与梯度反传的动态偏置 (aux-free bias) 调节路由选择, 在不干扰语言建模损失的前提下维持负载均衡。"
      question="当专家数量扩展后, 如何防止路由器将绝大多数 token 集中分配给少数热门专家而引发路由坍缩? 又该如何在维持负载均衡的同时, 避免均衡机制干扰语言建模主损失的梯度优化?"
      :goals="[
        '掌握稀疏 MoE 层以路由器加 E 个独立专家替换单个稠密 FFN 的结构, 以及单 token 路由分发与加权聚合的完整前向流程',
        '辨析 Mixtral 的 softmax top-k 竞争打分与 DeepSeek-V3 的 sigmoid 独立打分加常驻共享专家在架构设计上的本质差异',
        '理解无辅助损失 (aux-loss-free) 动态偏置如何在不干扰门控权重与梯度反传的前提下实现负载均衡',
      ]"
      :codes="[
        { path: 'llm_models/layers/sparse/moe.py', label: 'moe.py · MixtralMoE' },
        { path: 'llm_models/models/moe/deepseekV3.py', label: 'deepseekV3.py · DeepSeekMoE' },
      ]"
      :prereq="prevChapter"
      :next-step="nextChapter"
    />

    <EvolutionChain
      title="演进逻辑链 · 参数容量与激活算力解耦"
      subtitle="根问题: 稠密 FFN 中全部参数对每个 token 均参与运算, 扩充参数容量必然导致单 token 计算开销同比例增长"
      :steps="evoSteps"
    />

    <!-- 控制条 -->
    <div class="card" style="margin-bottom: 20px;">
      <div class="controls">
        <div class="form-row">
          <span id="moe-variant-label" class="row-label">变体</span>
          <div class="btn-group" style="grid-column: span 2;" role="group" aria-labelledby="moe-variant-label">
            <button type="button" :class="{ active: variant === 'mixtral' }" :aria-pressed="variant === 'mixtral'" @click="variant = 'mixtral'">Mixtral (softmax)</button>
            <button type="button" :class="{ active: variant === 'deepseek' }" :aria-pressed="variant === 'deepseek'" @click="variant = 'deepseek'">DeepSeek (sigmoid)</button>
          </div>
        </div>
        <div class="form-row">
          <label for="moe-num-experts">专家总数 E</label>
          <input id="moe-num-experts" type="range" min="4" :max="MAX_EXPERTS" step="2" v-model.number="numExperts" />
          <span class="val">{{ numExperts }}</span>
        </div>
        <div class="form-row">
          <label for="moe-top-k">top-k</label>
          <input id="moe-top-k" type="range" :min="1" :max="topKMax" step="1" v-model.number="topK" />
          <span class="val">{{ topK }}</span>
        </div>
        <div v-if="variant === 'deepseek'" class="form-row">
          <label for="moe-num-shared">共享专家</label>
          <input id="moe-num-shared" type="range" min="0" max="4" step="1" v-model.number="numShared" />
          <span class="val">{{ numShared }}</span>
        </div>
        <div class="form-row">
          <label for="moe-num-tokens">Token 数</label>
          <input id="moe-num-tokens" type="range" min="8" :max="MAX_TOKENS" step="2" v-model.number="numTokens" />
          <span class="val">{{ numTokens }}</span>
        </div>
        <div class="form-row">
          <label for="moe-collapse-bias">路由坍缩倾向</label>
          <input id="moe-collapse-bias" type="range" min="0" max="100" step="1" v-model.number="collapseBias" />
          <span class="val">{{ collapseBias }}%</span>
        </div>
      </div>

      <div style="display: flex; gap: 8px; margin-top: 12px;">
        <button type="button" @click="seed++">重新采样 tokens</button>
        <button type="button" @click="animateFlow" :disabled="animating">▷ 播放路由动画</button>
      </div>
      <p class="desc" style="margin-top: 10px;">
        <strong>模拟什么:</strong> 单层 MoE 中的路由器 (router) 为一个 batch 的输入 token 分配专家的前向过程:
      </p>
      <ol class="desc pts">
        <li>每个 token 对每个专家具有一个初始得分 (实际模型中为 router 线性投影层输出的 logits, 此处由固定随机种子生成)。</li>
        <li>门控打分函数将初始 logits 映射为门控分数。</li>
        <li>选取门控分数最高的 top-k 个专家并对选中权重做归一化。</li>
      </ol>
      <p class="desc">拖动参数滑杆时底层随机采样表保持固定, 仅点击「重新采样 tokens」才会更新随机种子重新生成初始得分。</p>
      <p class="desc" style="margin-top: 6px;">
        <strong>图上是什么:</strong>
      </p>
      <ul class="desc pts">
        <li>画布左侧列为当前 batch 内的输入 token (t0, t1, …), 中间 Router 模块执行门控打分与筛选, 右侧列为候选专家网络 (E0, E1, …)。</li>
        <li>从 Router 连向专家的每条曲线代表一个 token 被分发至对应专家; 连线越粗、不透明度越高, 表示该专家在该 token 的 top-k 归一化权重中占比越大。</li>
        <li>每个专家节点内的横向进度条表示当前负载 (即本 batch 内选中该专家的 token 数量); 当负载超过平均负载的 1.8 倍时进度条变为红色预警。</li>
        <li>切换至 DeepSeek 变体时, 右侧底部新增的 Shared 节点代表常驻共享专家, 与所有 token 建立连接, 表示每个 token 均固定经过共享专家计算。</li>
      </ul>
      <p class="desc" style="margin-top: 6px;">
        <strong>可以做什么:</strong>
      </p>
      <ul class="desc pts">
        <li>切换「变体」, 在 Mixtral 的 softmax 与 DeepSeek 的 sigmoid 门控函数之间对比。</li>
        <li>调节「专家总数 E」, 改变路由专家的总规模。</li>
        <li>调节「top-k」, 改变每个 token 激活的路由专家数量 (上限为专家总数 E 的一半)。</li>
        <li>调节「共享专家」(仅在 DeepSeek 变体下显示), 改变常驻激活专家的数量。</li>
        <li>调节「Token 数」, 改变当前 batch 的输入规模。</li>
        <li>调节「路由坍缩倾向」, 人为抬高前两个专家 E0 与 E1 的初始底分, 数值越大则流量向头部专家集中的程度越显著。</li>
        <li>点击「播放路由动画」, 以动态流向辅助观察各 token 的专家分发路径。</li>
      </ul>
      <p class="desc" style="margin-top: 6px;">
        <strong>应该看到:</strong> 将「路由坍缩倾向」拉至 100% 时:
      </p>
      <ul class="desc pts">
        <li>绝大多数 token 集中涌向 E0 与 E1, 两者负载条变为红色过载状态; 其余专家几乎未被激活, 但其权重参数依然占用显存。</li>
        <li>画布下方统计栏的「负载标准差」随之变红: 该指标度量各路由专家接收 token 数的离散程度, 数值越小代表负载分布越均衡。</li>
      </ul>
      <p class="desc">
        在真实训练过程中, 这种失衡会形成正反馈循环: 高频被选中的专家获得更多梯度更新而拟合得更好, 进而在后续训练步中获得更高路由得分, 最终导致少数专家包揽全部流量。因此 MoE 架构必须配套专门的负载均衡机制。
      </p>
    </div>

    <!-- 主可视化: 左 tokens → 中 router → 右 experts -->
    <div class="card">
      <svg :viewBox="`0 0 ${W} ${H}`" width="100%" :height="H" role="img"
           :aria-label="`${numTokens} 个 token 经 router 分发至 ${numExperts} 个专家, 每个 token 选取 top-${topK}。负载标准差 ${loadStd.toFixed(2)}, 最热门专家为 E${hottestExpert.idx}`">

        <!-- Tokens -->
        <g>
          <text x="50" y="24" fill="var(--text-muted)" font-size="12" font-family="SF Mono">tokens (batch)</text>
          <g v-for="(t, i) in tokens" :key="'t-'+i">
            <rect :x="20" :y="40 + i * tokenGap"
                  width="80" height="22" rx="4"
                  fill="var(--bg-elev)" stroke="var(--border)" />
            <text :x="60" :y="56 + i * tokenGap"
                  text-anchor="middle" fill="var(--text)" font-size="11" font-family="SF Mono">
              t{{ i }}
            </text>
          </g>
        </g>

        <!-- Router -->
        <g>
          <rect :x="routerX" :y="40" width="100" :height="H - 60" rx="6"
                fill="var(--bg-elev)" stroke="var(--accent)" stroke-opacity="0.6" />
          <text :x="routerX + 50" y="30" text-anchor="middle" fill="var(--accent)" font-size="12" font-weight="600">Router</text>
          <text :x="routerX + 50" :y="H - 10" text-anchor="middle"
                fill="var(--text-dim)" font-size="10" font-family="SF Mono">
            {{ variant === 'mixtral' ? 'softmax' : 'sigmoid' }}
          </text>
        </g>

        <!-- 路由连线 -->
        <g>
          <path v-for="(edge, i) in edges" :key="'e-'+i"
                :d="edge.d"
                :stroke="edge.color"
                :stroke-opacity="edge.opacity"
                :stroke-width="edge.width"
                fill="none"
                :class="{ flowing: animating }" />
        </g>

        <!-- Experts -->
        <g>
          <g v-for="(e, i) in displayedExperts" :key="'exp-'+i">
            <rect :x="expertX" :y="40 + i * expertGap"
                  :width="140" :height="expertGap - 8"
                  rx="4"
                  :style="{ fill: e.shared ? 'color-mix(in srgb, var(--eye) 8%, transparent)' : 'var(--bg-elev)' }"
                  :stroke="e.shared ? 'var(--eye)' : (e.load > 0 ? 'var(--accent)' : 'var(--border)')"
                  :stroke-opacity="e.shared ? 0.8 : (0.4 + Math.min(1, e.load / maxLoad) * 0.6)" />
            <text :x="expertX + 10" :y="55 + i * expertGap"
                  fill="var(--text)" font-size="11" font-family="SF Mono">
              {{ e.shared ? 'Shared' : 'E' + i }}
            </text>
            <!-- load bar -->
            <rect :x="expertX + 60" :y="48 + i * expertGap"
                  :width="60" height="6" rx="2" fill="var(--border)" />
            <rect :x="expertX + 60" :y="48 + i * expertGap"
                  :width="60 * Math.min(1, e.load / maxLoad)" height="6" rx="2"
                  :fill="e.shared ? 'var(--eye)' : e.load > avgLoad * 1.8 ? 'var(--danger)' : 'var(--accent)'" />
            <text :x="expertX + 128" :y="54 + i * expertGap"
                  fill="var(--text-muted)" font-size="10" font-family="SF Mono"
                  text-anchor="end">
              {{ e.load }}
            </text>
          </g>
          <text :x="expertX + 70" y="24" text-anchor="middle" fill="var(--text-muted)" font-size="12" font-family="SF Mono">experts (load)</text>
        </g>
      </svg>

      <!-- 这张图没模拟的东西要写明, 否则读者会以为 DeepSeek 的均衡没用 -->
      <div class="sim-note">
        <p>
          本可视化仅模拟单步前向中的「门控打分与 top-k 筛选」环节。在此切换 Mixtral 与 DeepSeek 变体, 实际改变的仅为门控函数 (由 softmax 切换为 sigmoid) 以及是否启用常驻共享专家。
          由于 softmax 与 sigmoid 均为严格单调递增函数, 初始底分更高的专家在映射后依然保持相同名次, 因此在同一组初始底分下 top-k 选出的路由专家集合完全一致, 各路由专家的负载读数也保持不变。
          两种门控函数的核心差异在于:
        </p>
        <ul class="pts">
          <li><b>softmax:</b> 强制单个 token 对所有专家的门控概率之和等于 1, 专家之间形成零和竞争关系。</li>
          <li><b>sigmoid:</b> 为每个专家独立计算 0 到 1 之间的亲和度得分, 各专家得分互不耦合。</li>
        </ul>
        <p>这一梯度与竞争特性的差异需要在多步训练优化过程中才会显现。</p>
        <p>
          此外, 本图属于静态单步路由演示, 未接入多步训练中的负载均衡闭环 (既不包含 Mixtral 的 aux loss, 也不包含 DeepSeek 的动态 bias 更新), 因此仅将变体切换为 DeepSeek 并不会自动消除路由坍缩。
          关于动态偏置如何在多步迭代中逐步拉平专家负载, 请参阅<router-link :to="{ name: 'models-moe-balance' }">「Aux-loss-free 均衡」一章</router-link>的交互实验台。
        </p>
      </div>

      <!-- 负载分析 -->
      <div class="load-analysis" aria-live="polite">
        <div class="stat">
          <div class="k">负载标准差</div>
          <div class="v" :style="{ color: loadStdColor }">{{ loadStd.toFixed(2) }}</div>
          <div class="hint">各路由专家接收 token 数的标准差, 数值越小越均衡。小于 1 显示绿色, 1 到 3 显示黄色, 3 及以上显示红色; 可拖动「路由坍缩倾向」观察指标变化</div>
        </div>
        <div class="stat">
          <div class="k">最热门专家</div>
          <div class="v">E{{ hottestExpert.idx }}</div>
          <div class="hint">单专家承接了总路由分发次数 (Token 数 × top-k) 的 {{ Math.round(hottestExpert.load / totalRoutedAssigns * 100) }}%</div>
        </div>
        <div class="stat">
          <div class="k">激活比例</div>
          <div class="v accent">{{ (topK / numExperts * 100).toFixed(1) }}%</div>
          <div class="hint">等于 top-k 除以专家总数 E, 表示单 token 实际激活的路由专家参数占全部路由专家参数的比例 (不含共享专家与注意力层)</div>
        </div>
      </div>
    </div>

    <!-- 对比表 -->
    <section class="section">
      <h2>Mixtral 与 DeepSeek-V3 的路由机制对比</h2>
      <div class="lead-group">
        <p>稀疏 MoE 在工程落地中面临四类典型问题, 下表逐项对比了 Mixtral 与 DeepSeek-V3 针对同一设计瓶颈给出的不同方案:</p>
        <ul class="pts">
          <li>粗粒度大专家数量有限, 单个专家需兼顾多种语义模式, 难以形成精细的专业化分工。</li>
          <li>缺乏常驻共享通道时, 每个 token 均需用到的通用语言知识被迫在各个路由专家中重复存储, 浪费参数容量。</li>
          <li>缺乏均衡约束时, 高频专家因训练更充分而持续获得更高路由得分, 正反馈累积引发路由坍缩。</li>
          <li>依赖外挂辅助损失 (aux loss) 维持均衡时, 辅助损失的反向梯度会直接作用于 router, 与语言建模主损失的优化方向产生干扰。</li>
        </ul>
      </div>
      <div class="card">
        <table class="compare">
          <thead>
            <tr>
              <th></th>
              <th>Mixtral (2024)</th>
              <th>DeepSeek-V3 (2024)</th>
            </tr>
          </thead>
          <tbody>
            <tr><td class="row-label">路由打分</td>
              <td><code class="inline">softmax(logits)</code>: 单 token 对全体专家的门控分数之和等于 1, 专家之间相互竞争</td>
              <td><code class="inline">sigmoid(logits)</code>: 每个专家独立输出 0 到 1 之间的亲和度得分, 彼此互不干扰</td>
            </tr>
            <tr><td class="row-label">专家归一化</td>
              <td>softmax 输出本身已是概率分布; 选出 top-k 后需对这 k 个概率重新除以它们的总和做重归一化, 确保聚合权重之和等于 1, 避免输出幅值随选中概率之和波动</td>
              <td>sigmoid 得分并非归一概率分布, 选出 top-k 后必须除以这 k 个得分之和完成归一化; 聚合权重基于不带偏置的原始亲和度得分计算</td>
            </tr>
            <tr><td class="row-label">共享专家</td>
              <td>无共享专家</td>
              <td>配置常驻共享专家。共享专家绕过 router, 对每个输入 token 全量激活, 输出直接与路由专家的加权输出相加, 由共享专家承载共性特征以释放路由专家的专用容量</td>
            </tr>
            <tr><td class="row-label">负载均衡</td>
              <td>沿用 Switch Transformer 的辅助损失方案, 在总目标中外挂 aux loss: 同时被高频选中且平均门控概率偏高的专家会推高该惩罚项, 通过反向梯度压低热门专家得分。本仓库总损失定义为语言建模损失加 0.01 倍辅助损失 (<code class="inline">lm_loss + 0.01 · aux</code>)</td>
              <td>采用无辅助损失的动态偏置 (aux-loss-free bias): 为每个路由专家维护一个不参与梯度反传的偏置项, 仅在 top-k 筛选阶段叠加至得分上, 计算专家聚合权重时仍采用原始得分。每步训练后调低过载专家偏置、调高欠载专家偏置。在本仓库有偏初始化 router 的对照实验中, 负载变异系数 CV (各专家负载标准差除以均值, 越小越均衡) 从 0.273 降至 0.124, 而 LM loss 基本不变 (2.979 对比 2.982)</td>
            </tr>
            <tr><td class="row-label">专家粒度</td>
              <td>8 个大粒度专家 (Mixtral 8x7B 配置专家总数 E 为 8, 单 token 激活 top-2)</td>
              <td>64 至 256 个细粒度专家 (本仓库 DeepSeekMoE 默认配置 64 个路由专家、单 token 激活 top-6, 外加 2 个常驻共享专家)</td>
            </tr>
            <tr><td class="row-label">代码位置</td>
              <td><RepoLink path="llm_models/layers/sparse/moe.py" label="layers/sparse/moe.py" tiny /></td>
              <td><RepoLink path="llm_models/models/moe/deepseekV3.py" label="models/moe/deepseekV3.py" tiny /></td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <!-- 本章挂载的实验台 (data/labmap/*.js) 与章末自测 (data/quiz/*.js), 没配置时不渲染 -->

    <LabMount />

    <QuizCard />


    <ChapterNav
      :prev="{ ...prevChapter, hint: 'MoE 本质上是 PreLNBlock 中 ffn 槽位的稀疏化实现' }"
      :next="nextChapter"
    />
  </div>
</template>

<script setup>
import LabMount from '@/components/LabMount.vue'
import QuizCard from '@/components/QuizCard.vue'
import { ref, computed, watch } from 'vue'
import { learningPath } from '@/data/models.js'
import { mulberry32 } from '@/utils/labmath.js'
import ChapterIntro from '@/components/ChapterIntro.vue'
import ChapterNav from '@/components/ChapterNav.vue'
import EvolutionChain from '@/components/EvolutionChain.vue'
import RepoLink from '@/components/RepoLink.vue'

// 上一章 / 下一章从 learningPath 取, 不手写章名和编号 (与 Infer.vue 同一写法)
const at = learningPath.findIndex((x) => x.route === 'moe')
const prevChapter = { name: learningPath[at - 1].route, label: `上一章 · ${learningPath[at - 1].label}` }
const nextChapter = { name: learningPath[at + 1].route, label: `下一章 · ${learningPath[at + 1].label}` }

// 演进逻辑链: 每步的 pain 是上一步 fix 留下的窟窿
const evoSteps = [
  { name: '稠密 FFN', color: 'var(--text-muted)',
    pain: '(原点) 提升模型知识容量需要加宽 FFN, 但每个 token 都必须遍历全部 FFN 参数, 导致计算开销同步增长',
    fix: '所有参数对每个输入 token 全量激活: FFN 参数规模翻倍时, 单 token 的浮点计算量也随之翻倍, 容量与算力严格耦合' },
  { name: 'Top-k MoE', year: 2017, color: 'var(--accent)',
    pain: '参数容量与单步算力严格耦合, 限制了模型参数规模在固定算力预算下的进一步扩展',
    fix: '将单个大 FFN 拆分为 E 个独立专家网络, 由轻量级 router 对每个 token 打分并仅激活得分最高的 k 个专家: 总容量按 E 扩展, 单步算力按 k 结算' },
  { name: 'Mixtral', year: 2024, color: 'var(--left)',
    pain: '高频被选中的专家得到更充分训练, 进而获得更高路由得分, 正反馈累积导致少数专家垄断流量的路由坍缩',
    fix: '采用 softmax 竞争打分选取 top-2 专家, 并在总损失中外挂辅助均衡损失 (aux loss) 抑制过载专家得分; 但 8 个大专家仍需各自重复学习通用特征' },
  { name: 'DeepSeekMoE', year: 2024, color: 'var(--right)',
    pain: '粗粒度专家导致通用知识在多专家间冗余存储, 削弱专业化分工; 同时 aux loss 的反向梯度会干扰语言建模主目标',
    fix: '拆分为 64 至 256 个细粒度专家并增设常驻共享专家承载通用知识; 通过不参与梯度反传的动态偏置 (bias) 仅调节专家选取以维持负载均衡' },
]

// 两根滑杆的上限, 底分表按这个尺寸一次生成
const MAX_EXPERTS = 32
const MAX_TOKENS = 32

const variant = ref('deepseek')
const numExperts = ref(8)
const topK = ref(2)
// top-k 滑杆的上限跟着专家数走。专家数调小时把 topK 钳回上限以内, 否则读数会停在滑杆够不到的值上
const topKMax = computed(() => Math.max(2, numExperts.value / 2))
watch(topKMax, (m) => { if (topK.value > m) topK.value = m })
const numShared = ref(2)
const numTokens = ref(16)
const collapseBias = ref(30)  // 0–100: 越大越倾向路由坍缩
const seed = ref(1)           // 只有「重新采样」会改它

const animating = ref(false)

// 画布几何
const W = 900
const tokenGap = 28
const expertGap = 32
const tokenX = 100
const routerX = 280
const expertX = 520

const H = computed(() => Math.max(numTokens.value, numExperts.value + numShared.value) * Math.max(tokenGap, expertGap) + 80)

// --- Tokens 与路由计算 ---
// ★ 每个 (token, 专家) 的底分只由 seed 决定, 按滑杆上限一次生成。
//   拖滑杆只是从这张表里取一块, 所以图不会乱跳, 看得出是哪个量在起作用。
const baseScores = computed(() => {
  const rand = mulberry32(seed.value)
  return Array.from({ length: MAX_TOKENS }, () =>
    Array.from({ length: MAX_EXPERTS }, () => rand() + (rand() - 0.5) * 0.3))
})

// 坍缩倾向越大, 前 2 个专家的分数被抬得越高, 多数 token 都往它们身上挤
const tokens = computed(() => {
  const bias = (collapseBias.value / 100) * 1.8
  return baseScores.value.slice(0, numTokens.value).map((row) => ({
    scores: row.slice(0, numExperts.value).map((base, e) => base + (e < 2 ? bias : 0)),
  }))
})

// --- 路由决策 ---
const routingDecisions = computed(() => {
  return tokens.value.map((t, i) => {
    let probs
    if (variant.value === 'mixtral') {
      // softmax
      const max = Math.max(...t.scores)
      const exp = t.scores.map(s => Math.exp(s - max))
      const sum = exp.reduce((a, b) => a + b, 0)
      probs = exp.map(e => e / sum)
    } else {
      // sigmoid independent
      probs = t.scores.map(s => 1 / (1 + Math.exp(-s)))
    }
    // top-k
    const withIdx = probs.map((p, e) => ({ p, e }))
    const top = withIdx.sort((a, b) => b.p - a.p).slice(0, topK.value)
    // renormalize
    const totalP = top.reduce((a, x) => a + x.p, 0)
    const weights = top.map(x => ({ e: x.e, w: x.p / totalP }))
    return { token: i, weights }
  })
})

// 专家负载
const displayedExperts = computed(() => {
  const E = numExperts.value
  const shared = variant.value === 'deepseek' ? numShared.value : 0
  const result = []
  // routed first
  for (let e = 0; e < E; e++) {
    const load = routingDecisions.value.reduce((acc, r) =>
      acc + r.weights.filter(w => w.e === e).length, 0)
    result.push({ idx: e, load, shared: false })
  }
  // shared
  for (let s = 0; s < shared; s++) {
    result.push({ idx: 's' + s, load: numTokens.value, shared: true })
  }
  return result
})

const maxLoad = computed(() => Math.max(1, ...displayedExperts.value.map(e => e.load)))
const avgLoad = computed(() =>
  (numTokens.value * topK.value) / numExperts.value
)
const loadStd = computed(() => {
  const routed = displayedExperts.value.filter(e => !e.shared)
  const mean = routed.reduce((a, e) => a + e.load, 0) / routed.length
  const variance = routed.reduce((a, e) => a + Math.pow(e.load - mean, 2), 0) / routed.length
  return Math.sqrt(variance)
})
const loadStdColor = computed(() => {
  const v = loadStd.value
  if (v < 1) return 'var(--left)'
  if (v < 3) return 'var(--warn)'
  return 'var(--danger)'
})
const hottestExpert = computed(() => {
  const routed = displayedExperts.value.filter(e => !e.shared)
  return routed.reduce((max, e) => e.load > max.load ? e : max, { load: 0, idx: 0 })
})
const totalRoutedAssigns = computed(() => numTokens.value * topK.value)

// --- 边的绘制 ---
const edges = computed(() => {
  const result = []
  routingDecisions.value.forEach((r, ti) => {
    const tx = tokenX + 0  // right side of token
    const ty = 40 + ti * tokenGap + 11
    const rx1 = routerX
    // Token → router
    result.push({
      d: `M ${tx} ${ty} L ${rx1} ${ty}`,
      color: 'var(--border-strong)',
      opacity: 0.4,
      width: 1,
    })
    // Router → selected experts
    r.weights.forEach(({ e, w }) => {
      const ex = expertX
      const ey = 40 + e * expertGap + 11
      const mid = (routerX + 100 + ex) / 2
      result.push({
        d: `M ${routerX + 100} ${ty} C ${mid} ${ty}, ${mid} ${ey}, ${ex} ${ey}`,
        color: 'var(--accent)',
        opacity: 0.2 + w * 0.6,
        width: 1 + w * 2,
      })
    })
    // Shared expert connections (always on, dashed)
    if (variant.value === 'deepseek') {
      for (let s = 0; s < numShared.value; s++) {
        const ex = expertX
        const ey = 40 + (numExperts.value + s) * expertGap + 11
        const mid = (routerX + 100 + ex) / 2
        result.push({
          d: `M ${routerX + 100} ${ty} C ${mid} ${ty}, ${mid} ${ey}, ${ex} ${ey}`,
          color: 'var(--eye)',
          opacity: 0.15,
          width: 1,
        })
      }
    }
  })
  return result
})

async function animateFlow() {
  animating.value = true
  await new Promise(r => setTimeout(r, 1200))
  animating.value = false
}
</script>

<style scoped>
.row-label { font-size: 13px; color: var(--text-muted); }
.sim-note {
  margin-top: 12px;
  padding-left: 10px;
  border-left: 2px solid var(--accent);
  color: var(--text-muted);
  font-size: 12.5px;
  line-height: 1.7;
}
.sim-note p + p { margin-top: 6px; }

.controls {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px 24px;
}
@media (max-width: 960px) {
  .controls { grid-template-columns: 1fr; }
}

.load-analysis {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 12px;
  margin-top: 16px;
}

table.compare {
  width: 100%;
  border-collapse: collapse;
}
table.compare th, table.compare td {
  padding: 10px 12px;
  text-align: left;
  border-bottom: 1px solid var(--border);
  font-size: 13px;
}
table.compare th {
  color: var(--text-muted);
  font-size: 11px;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.6px;
}
table.compare .row-label {
  color: var(--text-muted);
  font-size: 12px;
  width: 130px;
}

.flowing {
  stroke-dasharray: 4 4;
  animation: flow 1s linear infinite;
}
@keyframes flow {
  to { stroke-dashoffset: -16; }
}
</style>
