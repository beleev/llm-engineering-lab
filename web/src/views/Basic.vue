<template>
  <div>
    <h1 class="page-title">最小可跑闭环 · 用 numpy 把 Transformer 写穿</h1>
    <div class="page-subtitle lead-group">
      <p>
        <RepoLink path="llm_basic/" label="llm_basic/" tiny /> 不依赖 PyTorch, 只用 numpy 数组把前向计算 (forward)、反向求导 (backward)、Adam 参数更新和自回归采样完整写出来。
      </p>
      <p>
        先把梯度在模型里的流动路径看清楚, 后续读 <RepoLink path="llm_models/" label="llm_models/" tiny /> 的 PyTorch 实现时, 就能明白 <code class="inline">autograd</code> 在底层替我们完成了哪些工作。
      </p>
    </div>

    <ChapterIntro
      tldr="语言模型的训练主循环可以浓缩成五步: 取数据 (get_batch)、前向预测 (forward)、计算损失 (loss)、反向求梯度 (backward) 和更新参数 (adam_step)。本阶段用纯 numpy 手写这五步并用数值差分逐项校验, 把 Transformer 的训练闭环完整拆开。"
      question="脱离 PyTorch 自动求导, 仅凭链式法则如何准确推导 RMSNorm、softmax 与残差连接的反向传播公式, 又如何定位手写梯度中的隐蔽错误?"
      :goals="[
        '不依赖深度学习框架, 用纯 numpy 在约 20 秒内将一个 45,568 参数的微型 GPT 训练至 val_loss 1.99',
        '厘清从输入 ids [B, T] 到输出 logits [B, T, V] 各层的张量形状变化, 以及前向需存入 cache 供反向使用的中间变量',
        '掌握 gradcheck 利用中心差分数值梯度逐元素核查解析梯度并定位反向实现错误的原理',
      ]"
      :codes="[
        { path: 'llm_basic/model.py' },
        { path: 'llm_basic/optim.py' },
        { path: 'llm_basic/train.py' },
        { path: 'llm_basic/sample.py' },
        { path: 'llm_basic/gradcheck.py' },
      ]"
      :prereq="prevChapter"
      :next-step="nextChapter"
    />

    <StageContents stage-id="basic" />

    <EvolutionChain
      title="演进逻辑链 · 从二元计数表到可训练的微型 GPT"
      subtitle="根问题: 给定前文序列, 如何预测下一个字符的概率分布, 并让模型参数直接从原始语料中学习这一映射关系。"
      :steps="evoSteps"
    />

    <!-- ── 1. 整体闭环 ─────────────────────────────────────────── -->
    <section class="section">
      <h2>1. 一张图看完整闭环</h2>
      <div class="lead-group">
        <p>
          左表是数据准备、训练、采样与梯度检查四条入口脚本, 右表是支撑它们的 7 对核心函数。每个算子都按 <code class="inline">forward</code> 与 <code class="inline">backward</code> 成对编写: 前向计算时把反向所需的中间张量存入 <code class="inline">cache</code>, 反向传播时再按相反顺序取出并链式回传梯度。
        </p>
      </div>

      <div class="grid grid-2" style="gap: 16px;">
        <div class="card">
          <h3>四条入口脚本 <span class="tag">主入口</span></h3>
          <table class="pipeline">
            <tbody>
              <tr>
                <td class="pp-name mono"><RepoLink path="llm_basic/prepare.py" label="prepare.py" tiny /></td>
                <td class="pp-desc">下载 Tiny Shakespeare 语料并完成字符级编码, 导出 <RepoLink path="llm_basic/train.bin" label="train.bin" tiny />、<RepoLink path="llm_basic/val.bin" label="val.bin" tiny /> 与词表元信息 <RepoLink path="llm_basic/meta.npz" label="meta.npz" tiny />。</td>
              </tr>
              <tr>
                <td class="pp-name mono"><RepoLink path="llm_basic/train.py" label="train.py" tiny /></td>
                <td class="pp-desc">执行 5 行核心训练循环, 依次调用 get_batch、forward、loss、backward 与 adam_step; 训练 2000 步耗时约 20 秒, val_loss 从 4.15 降至 1.99。</td>
              </tr>
              <tr>
                <td class="pp-name mono"><RepoLink path="llm_basic/sample.py" label="sample.py" tiny /></td>
                <td class="pp-desc">加载训练保存的 ckpt 权重, 结合 temperature 缩放与 top-k 截断做自回归文本生成。</td>
              </tr>
              <tr>
                <td class="pp-name mono"><RepoLink path="llm_basic/gradcheck.py" label="gradcheck.py" tiny /></td>
                <td class="pp-desc">以中心差分数值梯度为基准, 对 7 个手写算子的解析梯度做逐元素核对, 全程耗时不到 1 秒。</td>
              </tr>
            </tbody>
          </table>
        </div>

        <div class="card">
          <h3>7 个函数对 <span class="tag"><RepoLink path="llm_basic/model.py" label="model.py" tiny /></span></h3>
          <table class="pairs">
            <tbody>
              <tr v-for="p in pairs" :key="p.name">
                <td class="pair-name mono">{{ p.name }}</td>
                <td class="pair-role"><Tex :text="p.role" /></td>
                <td class="pair-key mono"><Tex :text="p.key" /></td>
              </tr>
            </tbody>
          </table>
          <p class="hint">右列标出了各算子反向传播中最易出错的关键环节, 第 3 节将挑选典型算子逐一展开推导。</p>
        </div>
      </div>
    </section>

    <!-- ── 2. 模型结构 + 数据流 ───────────────────────────────────── -->
    <section class="section">
      <h2>2. 模型结构 · 单层单头配置下的完整 Pre-LN 骨架</h2>
      <p class="lead">
        模型规模刻意收敛到最小配置: 默认单层 (<code class="inline">--n-layer</code> 可配置多层, 堆叠仅需一层循环)、单头因果注意力、ReLU 激活的两层 MLP 与可学习位置编码。Pre-LN 残差主干与现代 LLaMA 架构完全同构, <strong>差异仅在于各子层内部组件的具体选型</strong>。
      </p>

      <div class="card">
        <DagView v-bind="modelDag" />
        <p class="hint">
          默认超参数 <code class="inline">V=65, D=64, H=128, T=64, n_layer=1</code> 下模型共计 <strong>45,568</strong> 个参数;
          将层数增至 2 层时总参数量为 78,656 (脚本直接根据参数字典中的 <code class="inline">block_{i}_*</code> 键名推断层数)。
        </p>
        <p class="hint">
          形状记号约定为: <code class="inline">B</code> 表示 batch 大小, <code class="inline">T</code> 表示序列长度 <code class="inline">seq_len</code>, <code class="inline">D</code> 表示隐藏维 <code class="inline">dim</code>, <code class="inline">H</code> 表示 MLP 中间维, <code class="inline">V</code> 表示词表大小 <code class="inline">vocab</code>。
          完整实现见 <RepoLink path="llm_basic/model.py:transformer_forward" label="llm_basic/model.py:transformer_forward" tiny />;
          如需逐步跟踪各层张量形状与 <code class="inline">cache</code> 内容, 可前往 <router-link :to="{ name: 'basic-forward' }" class="dt-link">{{ chapterNo('basic-forward') }} 的形状流水线实验台</router-link>。
        </p>
      </div>
    </section>

    <!-- ── 3. 反向传播的 4 个最容易写错的点 ─────────────────────── -->
    <section class="section">
      <h2>3. 反向传播四个易错点 · 解析梯度的关键细节</h2>
      <p class="lead">
        自动求导框架在底层透明处理了索引累加、归一化耦合、矩阵雅可比化简与残差分支汇合。亲手推导这四处反向算子, 才能理解反向传播的真实计算过程, 并明白为何梯度实现出错时往往不会触发形状异常、只会表现为收敛变慢。
      </p>

      <div class="grid grid-2" style="gap: 16px;">
        <div v-for="g in gotchas" :key="g.title" class="card">
          <h3>{{ g.title }} <span class="tag">{{ g.tag }}</span></h3>
          <Prose class="desc" style="margin-bottom: 10px;" :text="g.intuition" />
          <pre class="code">{{ g.formula }}</pre>
          <p class="hint" style="margin-top: 8px;">
            <strong>代码位置:</strong> <RepoLink :path="`llm_basic/${g.where}`" :label="g.where" tiny />
          </p>
        </div>
      </div>
    </section>

    <!-- ── 4. 训练循环 + Adam ───────────────────────────────────── -->
    <section class="section">
      <h2>4. 训练循环 · 5 行核心调度与 60 行手写 Adam</h2>
      <p class="lead">
        PyTorch 中封装在 <code class="inline">loss.backward()</code> 与 <code class="inline">opt.step()</code> 两句调用背后的梯度回传和动量更新, 在本阶段直接展开为显式的纯函数调用。
      </p>

      <div class="grid grid-2" style="gap: 16px;">
        <div class="card">
          <h3><RepoLink path="llm_basic/train.py" label="train.py" tiny /> 主循环 <span class="tag">5 行核心逻辑</span></h3>
          <pre class="code">{{ trainLoopCode }}</pre>
          <ul class="hint pts">
            <li><b>无需 <code class="inline">zero_grad()</code>:</b> <code class="inline">transformer_backward</code> 每步构造并返回全新的梯度字典, 不做原地累加, 旧梯度不会跨步残留。</li>
            <li><b>无需 <code class="inline">requires_grad</code>:</b> 前向缓存与反向计算图由手写函数显式串联, 无需在张量对象上挂载动态计算图标记。</li>
          </ul>
          <p class="hint">主流训练框架的底层执行语义正是这 5 行纯函数调度的工程化封装。</p>
        </div>

        <div class="card">
          <h3>Adam · 矩估计、偏置修正与自适应步长 <span class="tag"><RepoLink path="llm_basic/optim.py" label="optim.py" tiny /></span></h3>
          <Prose :text="adamMath" />
          <div class="adam-why">
            <p><strong>为何不使用朴素 SGD</strong></p>
            <p class="hint">
              Transformer 不同组件的参数梯度幅值往往相差数个数量级 (例如输出层 <code class="inline">lm_head</code> 的梯度远大于输入层 <code class="inline">embedding</code>), 单一全局学习率难以兼顾: 若学习率足以推动平坦方向收敛, 陡峭方向便极易震荡发散。
            </p>
            <p class="hint">
              Adam 通过除以二阶矩平方根 <Tex text="$\sqrt{\hat v}$" /> 对每个参数坐标做尺度归一化, 使单步有效更新量级近似等于学习率 <code class="inline">lr</code> 而不再受局部梯度幅值支配。完整实现见 <RepoLink path="llm_basic/optim.py:adam_step" label="adam_step" tiny />。
            </p>
          </div>
        </div>
      </div>
    </section>

    <!-- ── 5. gradcheck ───────────────────────────────────────── -->
    <section class="section">
      <h2>5. gradcheck · 用数值梯度反查解析梯度</h2>
      <div class="lead-group">
        <p>
          手写反向传播最棘手的问题在于<strong>验证解析梯度的正确性</strong>: 矩阵转置颠倒或沿错误维度求和往往仍能凑出合法的张量形状, 程序既不抛出异常, 训练损失甚至仍能缓慢下降。
        </p>
        <p>
          <code class="inline">gradcheck</code> 通过中心有限差分独立计算每个参数元素的数值梯度并与解析梯度逐项比对, 使隐蔽的推导错误直接暴露。例如删去 <code class="inline">rmsnorm_backward</code> 中的归一化耦合项后, 检验脚本会立即检出 <code class="inline">dx</code> 的相对误差升至 3.11e-01。
        </p>
      </div>

      <div class="card">
        <pre class="code">{{ gradcheckCode }}</pre>
        <p class="hint">
          判定阈值采用混合容差 <code class="inline">atol + rtol · max(|g_a|, |g_n|)</code>: 纯绝对容差对大梯度太松, 纯相对容差又会对接近 0 的梯度 (例如当前 batch 未出现的 <code class="inline">tok_emb</code> 行, 量级约 1e-10) 误报。
        </p>
        <p class="hint">
          步长 <code class="inline">eps = 1e-5</code> 处在 float64 下 U 形误差曲线的谷底 —— 步长再小会被浮点舍入误差淹没, 再大则会引入高阶截断误差 (<router-link :to="{ name: 'basic-backward' }" class="dt-link">{{ chapterNo('basic-backward') }} 的实验台</router-link>可交互观察这条曲线)。
        </p>
        <p class="hint">
          <code class="inline">gradcheck.py</code> 分两级检验: 先逐个算子做全元素核对 (7 个算子相对误差均在 1e-11 ~ 2e-10), 再对 <code class="inline">n_layer = 1</code> 和 <code class="inline">2</code> 做端到端抽样校验。
        </p>
      </div>
    </section>

    <!-- ── 6. 简化 vs 现代 ─────────────────────────────────────── -->
    <section class="section">
      <h2>6. 最小实现的简化项 · 衔接阶段 2 现代架构</h2>
      <p class="lead">
        为了凸显反向传播主干, <code class="inline">llm_basic</code> 将各子层收敛为最简形态。下表列出的每一项「现代模型方案」都对应 <RepoLink path="llm_models/" label="llm_models/" tiny /> 中的一个独立实现, 阶段 2 各章将逐项替换并做控制变量对比。
      </p>
      <div class="card" style="padding: 0; overflow-x: auto;">
        <table class="diff-table">
          <thead>
            <tr>
              <th>模块维度</th>
              <th>llm_basic 最小方案</th>
              <th>现代模型方案 (llm_models)</th>
              <th>对应章节</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="d in diffs" :key="d.topic">
              <td class="dt-topic">{{ d.topic }}</td>
              <td class="mono small">{{ d.basic }}</td>
              <td class="mono small">{{ d.modern }}</td>
              <td>
                <router-link v-if="d.route" :to="{ name: d.route }" class="dt-link">
                  {{ chapterNo(d.route) }}
                </router-link>
                <RepoLink v-else-if="d.file" :path="d.file" :label="d.routeLabel" tiny />
                <span v-else class="muted small">{{ d.routeLabel }}</span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <!-- 本章挂载的实验台 (data/labmap/*.js) 与章末自测 (data/quiz/*.js), 没配置时不渲染 -->

    <LabMount />

    <QuizCard />


    <ChapterNav
      :prev="{ ...prevChapter, hint: '回到六阶段全局学习地图' }"
      :next="{ ...nextChapter, hint: '先将原始文本语料编码为可复现的训练张量' }"
    />
  </div>
</template>

<script setup>
import StageContents from '@/components/StageContents.vue'
import LabMount from '@/components/LabMount.vue'
import QuizCard from '@/components/QuizCard.vue'
import Prose from '@/components/Prose.vue'
import Tex from '@/components/Tex.vue'
import ChapterIntro from '@/components/ChapterIntro.vue'
import ChapterNav from '@/components/ChapterNav.vue'
import EvolutionChain from '@/components/EvolutionChain.vue'
import RepoLink from '@/components/RepoLink.vue'
import DagView from '@/components/dag/DagView.vue'
import { learningPath, stages, chapterByRoute } from '@/data/models.js'

// 上一章 / 下一章从 learningPath 取, 不手写章名和编号 (与 Infer.vue 同一写法)
const at = learningPath.findIndex((x) => x.route === 'basic')
const prevChapter = { name: learningPath[at - 1].route, label: `上一章 · ${learningPath[at - 1].label}` }
const nextChapter = { name: learningPath[at + 1].route, label: `下一章 · ${learningPath[at + 1].label}` }

// 章节编号与三级目录共用数据, 跨阶段引用也跟随归组更新。
const chapterNo = (route) => {
  for (const s of stages) {
    if (s.route === route) return `阶段 ${s.idx}`
    if (chapterByRoute[route]) return `章节 ${chapterByRoute[route].number}`
  }
  return ''
}

const evoSteps = [
  { name: '计数表', color: 'var(--text-muted)',
    pain: '(原点) 预测下一个字符最直接的做法是统计语料中「字符 a 后紧跟字符 b」的共现频次。',
    fix: '构造 $V\\times V$ 频次矩阵并按行归一化即可得到条件概率分布, 但上下文视野受限于前 1 个字符。' },
  { name: '神经网络 LM', color: 'var(--accent)',
    pain: '若直接将统计窗口扩展至前 $n$ 个字符, 计数表规模将按 $V^n$ 指数增长且绝大部分条目在语料中为零。',
    fix: '改用参数化的 Transformer 将前文编码为稠密向量并映射为候选词表上的 logits, 通过最小化交叉熵损失迭代优化权重。' },
  { name: '手写 backward', color: 'var(--left)',
    pain: '全模型 45,568 个参数每一步更新都依赖精确梯度, 而纯 numpy 环境不提供自动求导机制。',
    fix: '前向计算时将中间张量存入 cache, 反向传播按链式法则逐层回传梯度, 并通过 gradcheck 中心差分数值梯度逐元素校验。' },
  { name: 'Adam', color: 'var(--right)',
    pain: '求得解析梯度后, 不同网络层的梯度幅值相差数个数量级, 单一全局学习率难以兼顾收敛速度与稳定性。',
    fix: '引入 Adam 维护梯度的一阶与二阶矩估计并按坐标自适应缩放步长, 2000 步内将验证损失从初始约 4.15 ($\\ln 65 \\approx 4.17$) 降至 1.99。' },
  { name: '自回归采样', color: 'var(--eye)',
    pain: '训练收敛的模型单次前向仅输出下一位置的条件概率分布, 尚不能直接产出连贯文本。',
    fix: '每次取序列末尾位置的 logits, 经 temperature 缩放与 top-k 截断后采样新字符追加回输入序列, 循环迭代生成文本。' },
]

const pairs = [
  { name: 'embedding',   role: 'token 与位置向量查表映射',              key: '重复索引须用 np.add.at 按位累加梯度' },
  { name: 'linear',      role: '仿射变换 $y = xW + b$',                 key: '展平前导维后计算 $dW = x^\\top dy$' },
  { name: 'rmsnorm',     role: '按行均方根归一化并乘可学习缩放因子',    key: '反向传播须减去均方根分母引入的耦合项' },
  { name: 'attention',   role: '单头因果注意力 $\\mathrm{softmax}(QK^\\top)\\cdot V$', key: 'softmax 反向化简为 $a\\cdot(da - \\sum a\\cdot da)$' },
  { name: 'mlp',         role: '投影升维至 $H$、经 ReLU 激活后降回 $D$', key: '前向须缓存激活前符号掩码供 ReLU 反向门控' },
  { name: 'block',       role: 'Pre-LN 注意力子层、MLP 子层与双残差连接', key: '残差汇合处须将子层梯度与恒等分支梯度相加' },
  { name: 'transformer', role: '串联嵌入层、多层 Block 与输出头计算 logits', key: 'CE 合并求导得 $d\\mathrm{logits} = (p - \\mathrm{onehot})/N$' },
]

// 模型骨架。Block 里是 Pre-LN 的两个子层, 主干只有加法 —— 和阶段 2 那张接线图是同一个结构。
// Block 标成一组, 可以整个折起来只看主干。
const modelDag = {
  dir: 'TB',
  nodes: [
    { id: 'ids', label: 'ids', sub: '(B, T)', kind: 'io' },
    { id: 'emb', label: 'tok_emb + pos_emb', sub: '[V,D] + [T_max,D] → (B, T, D)', note: '词嵌入与位置嵌入逐元素相加' },
    { id: 'n1', label: 'RMSNorm', kind: 'norm', group: 'block' },
    { id: 'attn', label: 'Attn', sub: '单头, causal', kind: 'attn', group: 'block' },
    { id: 'add1', label: '+', kind: 'trunk', note: '残差 1', group: 'block' },
    { id: 'n2', label: 'RMSNorm', kind: 'norm', group: 'block' },
    { id: 'mlp', label: 'MLP', sub: 'D → H → D, ReLU', kind: 'ffn', group: 'block' },
    { id: 'add2', label: '+', kind: 'trunk', note: '残差 2', group: 'block' },
    { id: 'fnorm', label: 'RMSNorm', kind: 'norm' },
    { id: 'head', label: 'lm_head', sub: '[D,V] → logits (B, T, V)' },
    { id: 'loss', label: 'softmax + cross-entropy', sub: '→ loss', kind: 'ok' },
  ],
  edges: [
    { from: 'ids', to: 'emb', kind: 'main' },
    { from: 'emb', to: 'add1', kind: 'main', label: '残差主干' },
    { from: 'emb', to: 'n1' }, { from: 'n1', to: 'attn' }, { from: 'attn', to: 'add1' },
    { from: 'add1', to: 'add2', kind: 'main', label: '残差主干' },
    { from: 'add1', to: 'n2' }, { from: 'n2', to: 'mlp' }, { from: 'mlp', to: 'add2' },
    { from: 'add2', to: 'fnorm', kind: 'main' },
    { from: 'fnorm', to: 'head', kind: 'main' },
    { from: 'head', to: 'loss', kind: 'main' },
  ],
  groups: [{ id: 'block', label: 'Block × n_layer' }],
}

const trainLoopCode = `for step in range(MAX_ITERS):
    x, y = get_batch(train_data, BATCH_SIZE, SEQ_LEN, rng)

    # 1. forward: 顺便把所有 cache 串起来
    logits, cache = transformer_forward(W, x)

    # 2. loss + dlogits (fused: dlogits=(probs-onehot)/N)
    loss, dlogits = cross_entropy_forward_backward(logits, y)

    # 3. backward: 沿 cache 倒推所有参数的梯度
    grads = transformer_backward(dlogits, cache)

    # 4. Adam: 拿 grads 更新 W, 不就地修改 (返回新 dict)
    W, opt_state = adam_step(W, grads, opt_state, lr=LR)`

const adamMath = `每一步迭代先递增时间步 $t$, 并用指数滑动平均分别更新梯度的一阶矩估计 $m$ 与二阶矩估计 $v$:
$$\\begin{aligned} t &\\leftarrow t + 1 \\\\ m &= \\beta_1 \\cdot m + (1-\\beta_1)\\cdot g \\\\ v &= \\beta_2 \\cdot v + (1-\\beta_2)\\cdot g^2 \\end{aligned}$$
由于 $m$ 与 $v$ 从全零初始化起步, 训练初期的矩估计会向零偏置, 因此需按当前步数 $t$ 做无偏修正:
$$\\begin{aligned} \\hat m &= m / (1 - \\beta_1^t) \\\\ \\hat v &= v / (1 - \\beta_2^t) \\end{aligned}$$
最后用修正后的一阶矩除以二阶矩平方根完成参数更新, 其中数值稳定项 $\\mathrm{eps}$ 用于防止分母为零:
$$W \\leftarrow W - \\mathrm{lr} \\cdot \\hat m / (\\sqrt{\\hat v} + \\mathrm{eps})$$`

const gradcheckCode = `# 对每个参数 W[k] 的每个元素 i:
g_analytic = grads[k].flat[i]

# 数值梯度: 中心差分
W[k].flat[i] += eps;  loss_plus,  _ = forward_loss()
W[k].flat[i] -= 2·eps; loss_minus, _ = forward_loss()
W[k].flat[i] += eps   # 还原
g_numeric = (loss_plus - loss_minus) / (2·eps)

# 判定: |g_a - g_n| < atol + rtol · max(|g_a|, |g_n|)`

const gotchas = [
  {
    title: 'embedding 的反向',
    tag: 'np.add.at',
    intuition: '同一个 token id 在单个 batch 内往往重复出现多次, 每次出现都会向词表对应行贡献一份上游梯度。\n若直接写成 dW[ids] += dout, numpy 花式索引只会保留同索引最后一次赋值的结果而丢失其余位置的梯度累加, 导致高频 token 的真实梯度被静默截断且不触发任何运行时异常。',
    formula: 'dW = zeros((V, D))\n# 重复索引必须累加, 不是覆盖\nnp.add.at(dW, ids, dout)',
    where: 'model.py: embedding_backward',
  },
  {
    title: 'RMSNorm 的耦合项',
    tag: '分母耦合项',
    intuition: 'RMSNorm 将每行向量除以其均方根 $\\mathrm{rms}$, 输入分量 $x_i$ 同时出现在分子与分母中: 增大 $x_i$ 会同步抬高整行的 $\\mathrm{rms}$, 进而拉低同位置及其余分量的归一化输出。因此对输入求导时, $dx$ 除直接缩放项 $dy \\cdot g / \\mathrm{rms}$ 外, 还必须减去通过分母传导的耦合项。\n若删去该耦合项运行 gradcheck, 相对误差会直接从 1e-10 升至 3.11e-01。',
    formula: 'dx_i = (g_i / rms) · dy_i\n     - x_i / (D · rms³) · Σ_j (dy_j · g_j · x_j)\n              ↑ 删掉这一项, 误差 3.11e-01',
    where: 'model.py: rmsnorm_backward',
  },
  {
    title: 'softmax 的 Jacobian',
    tag: '雅可比矩阵化简',
    intuition: 'softmax 归一化分母包含整行指数和, 每个输出分量 $a_i$ 都依赖同行全部输入 $s_j$, 其局部导数是一个稠密雅可比矩阵而非逐元素向量。将其与上游梯度 $da$ 相乘后可代数化简为向量运算, 且表达式仅依赖前向输出概率 $a$, 因此前向 cache 只需保存归一化后的注意力权重 attn 而无需保留原始打分矩阵 scores。',
    formula: '设 a = softmax(s),  da 是上游梯度\nds_i = a_i · (da_i - Σ_j a_j · da_j)\n            ↑ 减去整行的加权和',
    where: 'model.py: attention_backward',
  },
  {
    title: '残差连接的反向两路累加',
    tag: '恒等路径直传',
    intuition: '前向分支相加在反向传播中对应梯度的等值分发与汇合: 上游梯度 dout 一路沿恒等分支无损直传, 另一路穿过归一化与注意力或 MLP 子层回传, 回到分叉点 $x$ 时须将两路梯度逐元素相加。恒等分支提供的无衰减通路避免深层网络梯度在连乘中消失。',
    formula: 'h = x + Attn(RMSNorm(x))\nout = h + MLP(RMSNorm(h))\n\ndh = dout + 穿过 MLP 回来的那份\ndx = dh   + 穿过 Attn 回来的那份',
    where: 'model.py: block_backward',
  },
]

const diffs = [
  { topic: '位置编码',  basic: '可学习绝对位置嵌入 pos_emb, 上下文长度受限于 T_max (64)', modern: 'RoPE 旋转位置编码, 直接作用于 Q 与 K 向量', route: 'position' },
  { topic: '注意力',    basic: '单头因果注意力, 单头维度等于隐藏维 D', modern: 'MHA / GQA / MLA / DSA 多头与稀疏变体', route: 'attention' },
  { topic: 'FFN 激活',  basic: 'ReLU 激活配合两层线性映射', modern: '从 GELU 演进至 SwiGLU 门控三线性层结构', route: 'blocks' },
  { topic: '层数',      basic: '默认 1 层 (可通过 --n-layer 参数扩展)', modern: '数十层深度堆叠及多种归一化与残差 Block 变体', route: 'blocks' },
  { topic: '分词',      basic: '字符级编码, 词表大小 65 (bpe.py 独立演示子词合并)', modern: 'BPE 或 SentencePiece 子词分词, 词表规模数万至十余万', route: 'basic-data' },
  { topic: 'FFN 形态',  basic: '全参数激活的稠密 MLP', modern: 'MoE 混合专家结构: 多个子专家网络配合 router top-k 稀疏路由', route: 'moe' },
  { topic: '推理',      basic: '每生成一个字符均重新执行整段序列的完整前向计算', modern: 'KV cache 增量解码: 仅计算新 token 的 Q 并复用历史 K/V', route: 'infer-kv-memory' },
  { topic: '优化器',    basic: '基础 Adam (权重衰减、学习率调度与梯度裁剪开关默认关闭)', modern: 'AdamW 配合 warmup、余弦退火调度与全局范数梯度裁剪', routeLabel: '参考 trainer.py', file: 'llm_models/training/trainer.py' },
  { topic: '精度',      basic: '全程使用 float64 双精度以满足 gradcheck 数值差分精度要求', modern: 'bf16 / fp16 混合精度训练配合 loss scaling 与 fp32 主权重', route: 'train' },
]
</script>

<style scoped>
table.pipeline,
table.pairs,
table.diff-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}
table.pipeline td,
table.pairs td {
  padding: 8px 0;
  border-bottom: 1px dashed var(--border);
  vertical-align: top;
}
table.pipeline tr:last-child td,
table.pairs tr:last-child td { border-bottom: none; }

.pp-name { width: 130px; color: var(--accent); font-size: 12.5px; }
.pp-desc { color: var(--text-muted); font-size: 12.5px; line-height: 1.6; }
.pair-name { width: 110px; color: var(--left); font-size: 12.5px; }
.pair-role { color: var(--text); font-size: 12.5px; padding-right: 12px; }
.pair-key  { color: var(--text-dim); font-size: 11.5px; text-align: right; min-width: 220px; }

.hint {
  margin-top: 10px;
  font-size: 12px;
  color: var(--text-dim);
  line-height: 1.6;
}
.adam-why { margin-top: 14px; padding-top: 12px; border-top: 1px dashed var(--border); }
.adam-why p { margin-bottom: 4px; font-size: 13px; }

table.diff-table th {
  text-align: left;
  padding: 12px 14px;
  background: var(--bg-elev);
  color: var(--text-muted);
  font-size: 11px;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.7px;
  border-bottom: 1px solid var(--border-strong);
}
table.diff-table td {
  padding: 10px 14px;
  border-bottom: 1px solid var(--border);
}
table.diff-table .dt-topic { color: var(--text); font-weight: 500; width: 110px; }
table.diff-table .small { font-size: 12px; }
table.diff-table .muted { color: var(--text-dim); }
.dt-link {
  font-size: 12px;
  color: var(--accent);
  border-bottom: 1px dashed var(--accent);
}
.dt-link:hover { text-decoration: none; border-bottom-style: solid; }
</style>
