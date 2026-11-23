<template>
  <div>
    <h1 class="page-title">规模化训练 · 把单机训练循环拆解为分布式系统</h1>
    <div class="page-subtitle lead-group">
      <p>
        <RepoLink path="llm_train/" label="llm_train/" tiny /> 使用 numpy 小张量复现分布式训练框架中的核心机制。这里用 Python 列表中的单个元素模拟一个 rank (分布式训练中的单个计算进程与设备), 使跨卡通信与数据分片的每一步都在内存中直接可见。
      </p>
      <p>
        每个演示脚本末尾都通过 <code class="inline">assert</code> 做数值对齐校验: 无论如何切分 batch、模型层、优化器状态或降低计算精度, 从 <code class="inline">loss</code>、<code class="inline">grad</code> 到 <code class="inline">update</code> 的计算结果都必须与单卡基线严格一致。
      </p>
    </div>

    <ChapterIntro
      tldr="单张显卡的显存容量与算力吞吐都不足以独立完成大模型训练, 规模化训练的核心是将单次训练任务沿四个维度拆分到多张设备上协同执行:
        - 切分 batch: 各卡并行处理不同样本子集, 同步聚合梯度。
        - 切分模型参数: 按层内矩阵维度或层间阶段拆分权重, 分摊模型本体与激活显存。
        - 切分优化器状态: 将参数、梯度与动量分片存储在不同设备上, 消除冗余副本。
        - 切分序列长度: 将长上下文沿序列维拆分到多卡协同计算。
        本阶段对每一种切分策略均通过 assert 校验其计算得到的梯度与参数更新同单卡基线严格等价。"
      question="面对显存溢出或吞吐不足时, 应当如何选择切分 batch、切分权重矩阵、切分模型层或切分优化器状态?"
      :goals="[
        '理清数据并行 (DDP) 切分 batch、张量并行 (TP) 切分矩阵、流水线并行 (PP) 切分层与 ZeRO 切分优化器状态各自解决的显存或吞吐瓶颈',
        '阐明混合精度训练为何必须配合 loss scaling、fp32 主权重 (master weights) 与梯度裁剪 (grad clip)',
        `将 ${moduleSpan} 各子模块定位到 full_loop 单步训练流程中的具体环节`,
      ]"
      :codes="[
        { path: 'llm_train/core/' },
        { path: 'llm_train/full_loop/' },
        { path: 'llm_train/m01_gradient_accumulation/' },
        { path: 'llm_train/m05_zero_fsdp/' },
      ]"
      :prereq="prevChapter"
      :next-step="nextChapter"
    />

    <StageContents stage-id="train" />

    <section class="section">
      <h2>1. 训练技术的依赖链</h2>
      <div class="lead-group">
        <p>分布式训练技术按硬件瓶颈出现的先后次序逐层叠加, 只有当前一层的手段触及上限时才引入下一层切分:</p>
        <ul class="pts">
          <li>单卡显存受限于大 batch 激活值时, 先通过梯度累积拆分为多个 micro-batch, 再通过数据并行扩展到多卡提升吞吐。</li>
          <li>单卡显存无法容纳完整模型权重时, 再沿层内矩阵宽度做张量并行或沿层间深度做流水线并行。</li>
          <li>多卡数据并行下参数、梯度与优化器状态的冗余副本占满显存时, 引入 ZeRO 分片消除重复存储。</li>
        </ul>
        <p>混合精度计算、激活重计算、断点续训 (checkpoint) 与数值稳定性保护则正交作用于上述所有并行维度。</p>
      </div>
      <EvolutionChain
        title="从单机循环到 full_loop"
        subtitle="根问题: 单卡显存装不下模型与激活、单机算力跑不完海量数据, 如何在多卡切分与低精度加速下严格保持单机训练的数学等价性。"
        :steps="trainChain"
      />
    </section>

    <section class="section">
      <h2>2. {{ axes.length }} 个切分维度</h2>
      <p class="lead">
        阅读分布式训练框架源码时, 核心线索是确认每个模块切分的对象与对应的集合通信原语 (如 all-reduce、reduce-scatter 与 all-gather)。下表汇总了八个维度各自解决的工程瓶颈与数值等价关系。
      </p>
      <div class="card" style="padding: 0; overflow-x: auto;">
        <table class="train-table">
          <thead>
            <tr>
              <th>维度</th>
              <th>解决的问题</th>
              <th>核心等价关系</th>
              <th>原始代码</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="a in axes" :key="a.name">
              <td class="axis">{{ a.name }}</td>
              <td>{{ a.problem }}</td>
              <td>{{ a.equiv }}</td>
              <td class="mono small"><RepoLink :path="`llm_train/${a.file}`" :label="a.file" tiny /></td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <section class="section">
      <h2>3. 四个关键等价 · 对照原始代码</h2>
      <div class="lead-group">
        <p>本阶段所有并行演示脚本均遵循统一的三步验证结构:</p>
        <ol class="pts">
          <li>在单卡稠密张量上运行前向与反向传播, 记录参考基线。</li>
          <li>按目标并行策略切分数据、权重或优化器状态, 执行带集合通信的分布式版本。</li>
          <li>通过 <code class="inline">assert</code> 逐元素比对两者的输出、梯度与更新后权重。</li>
        </ol>
        <p>数学等价性是分布式并行策略的首要验收标准; 若切分或通信缩放引入偏差, 模型收敛轨迹将偏离单机基线。</p>
      </div>

      <div class="grid grid-2" style="gap: 16px;">
        <div v-for="c in codeCards" :key="c.title" class="card">
          <h3>{{ c.title }} <span class="tag">{{ c.tag }}</span></h3>
          <p class="desc">{{ c.desc }}</p>
          <pre class="code">{{ c.code }}</pre>
          <p class="hint"><strong>代码位置:</strong> <CodeRef :value="c.file" base="llm_train/" tiny /></p>
        </div>
      </div>
    </section>

    <section class="section">
      <h2>4. 从 llm_basic 到 full_loop</h2>
      <p class="lead">
        <RepoLink path="llm_basic/train.py" label="llm_basic/train.py" tiny /> 的单机主循环仅包含 <code class="inline">get_batch</code>、<code class="inline">forward</code>、<code class="inline">loss</code>、<code class="inline">backward</code> 与 <code class="inline">adam_step</code> 五步。<RepoLink path="llm_train/full_loop/demo.py" label="llm_train/full_loop/demo.py" tiny /> 完整保留了这一优化语义, 只是将每个单机步骤展开为多卡切分、梯度缩放、集合通信与状态保护等工程操作。
      </p>
      <div class="grid grid-2" style="gap: 16px;">
        <div class="card">
          <h3>单机最小循环 <span class="tag">llm_basic</span></h3>
          <pre class="code">{{ basicLoop }}</pre>
        </div>
        <div class="card">
          <h3>分布式训练一步 <span class="tag">full_loop</span></h3>
          <pre class="code">{{ distributedLoop }}</pre>
        </div>
      </div>
      <div class="card compare-card">
        <table class="train-table">
          <thead>
            <tr>
              <th>单机动作</th>
              <th>full_loop 中的展开</th>
              <th>工程必要性</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="r in loopMap" :key="r.simple">
              <td class="axis">{{ r.simple }}</td>
              <td>{{ r.full }}</td>
              <td>{{ r.why }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <section class="section">
      <h2>5. 模块索引 · 运行与核对指南</h2>
      <div class="lead-group">
        <p>每个模块的演示脚本都会依次输出实验现象、量化指标与验证结论。推荐按以下顺序执行:</p>
        <ol class="pts">
          <li>单独运行目标子模块, 观察对应切分维度或稳定性机制的前后对比数据。</li>
          <li>
            执行 <code class="inline">python -m llm_train.run_all</code> 按依赖顺序串联运行全部模块
            (<RepoLink path="llm_train/run_all.py" label="llm_train/run_all.py" tiny />)。
          </li>
        </ol>
      </div>
      <div class="card" style="padding: 0; overflow-x: auto;">
        <table class="train-table">
          <thead>
            <tr>
              <th>模块</th>
              <th>要点</th>
              <th>与前置章节的衔接</th>
              <th>原始代码</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="m in trainModules" :key="m.id">
              <td class="axis">{{ m.name }}</td>
              <td>{{ m.concept }}</td>
              <td>{{ m.link }}</td>
              <td class="mono small"><RepoLink :path="m.file" tiny /></td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <!-- 本章挂载的实验台 (data/labmap/*.js) 与章末自测 (data/quiz/*.js), 没配置时不渲染 -->

    <LabMount />

    <QuizCard />


    <ChapterNav
      :prev="{ ...prevChapter, hint: '模型架构组件已拆解完毕, 下一步是将训练扩展到多卡规模' }"
      :next="{ ...nextChapter, hint: '先拆分 micro-batch 控制峰值显存, 再同步多卡梯度' }"
    />
  </div>
</template>

<script setup>
import StageContents from '@/components/StageContents.vue'
import LabMount from '@/components/LabMount.vue'
import QuizCard from '@/components/QuizCard.vue'
import ChapterIntro from '@/components/ChapterIntro.vue'
import ChapterNav from '@/components/ChapterNav.vue'
import EvolutionChain from '@/components/EvolutionChain.vue'
import CodeRef from '@/components/CodeRef.vue'
import RepoLink from '@/components/RepoLink.vue'
import { learningPath } from '@/data/models.js'
import { trainModules } from '@/data/modules.js'

// 上一章 / 下一章从 learningPath 取, 不手写章名和编号 (与 Infer.vue 同一写法)
const at = learningPath.findIndex((x) => x.route === 'train')
const prevChapter = { name: learningPath[at - 1].route, label: `上一章 · ${learningPath[at - 1].label}` }
const nextChapter = { name: learningPath[at + 1].route, label: `下一章 · ${learningPath[at + 1].label}` }

// 模块范围从数据里数, 不写死 (trainModules 里除了 mNN 还有一行 full)
const moduleCount = trainModules.filter((m) => /^m\d+$/.test(m.id)).length
const moduleSpan = `m01..m${String(moduleCount).padStart(2, '0')}`

const trainChain = [
  {
    name: 'Accum',
    year: 'm01',
    pain: '增大全局 batch size 时, 前向传播为反向求导保存的中间激活值会率先耗尽单卡显存。',
    fix: '将全局 batch 拆分为 K 个 micro-batch 串行执行前向与反向, 梯度按样本占比加权累加, 使激活显存峰值降至原来的 1/K 且累积梯度与整批计算严格一致。',
    color: 'var(--left)',
  },
  {
    name: 'DDP',
    year: 'm02',
    pain: '单卡算力吞吐有限, 串行累加 micro-batch 无法缩短海量语料的总训练时长。',
    fix: '各 rank 独立计算本地样本分片的梯度, 通过 all-reduce 求均值得到一致的全局梯度并同步更新权重, 各卡参数全程逐位对齐。',
    color: 'var(--accent)',
  },
  {
    name: 'TP / PP',
    year: 'm03-m04',
    pain: '单层隐藏维过宽或网络层数过深时, 单张显卡无法容纳完整模型权重。',
    fix: '张量并行 (TP) 沿矩阵行列切分单层权重并在层内集合通信, 流水线并行 (PP) 按层划分 stage 并仅在相邻边界传输激活与梯度。',
    color: 'var(--eye)',
  },
  {
    name: 'ZeRO',
    year: 'm05',
    pain: '标准 DDP 在每张卡上完整复制参数、梯度与 Adam 状态, 混合精度下每参数仍需占用 16 字节显存。',
    fix: '利用 reduce-scatter 聚合梯度分片, 每张卡仅维护并更新本地分片的优化器状态与参数, 将单卡状态显存从 $16\\Psi$ ($\\Psi$ 为参数量) 降至 $16\\Psi/N$。',
    color: 'var(--right)',
  },
  {
    name: 'AMP + Guard',
    year: 'm06-m10',
    pain: '低精度计算可提升吞吐并节省显存, 但易引发微小梯度下溢、大激活值上溢以及异常样本污染权重。',
    fix: '组合使用 loss scaling、fp32 主权重副本、全局范数梯度裁剪、warmup 与 WSD 学习率调度及 NaN 检测跳过机制保障数值稳定。',
    color: 'var(--warn)',
  },
  {
    name: 'Full Loop',
    year: 'full',
    pain: '孤立的并行与数值技巧尚不能构成支持故障恢复的完整分布式训练闭环。',
    fix: '将数据切分、梯度累积、跨卡同步、梯度裁剪、分片更新与分片 checkpoint 串联为统一训练步, 实现中断续训后轨迹逐位一致。',
    color: 'var(--left)',
  },
]

const axes = [
  { name: 'batch', problem: '单卡激活显存受限下扩大有效 batch size', equiv: '各 micro-batch 梯度的样本数加权和等于完整 batch 梯度', file: 'm01_gradient_accumulation/demo.py' },
  { name: 'replica', problem: '多卡并行处理不同样本分片以扩展训练吞吐', equiv: '各卡本地梯度经 all_reduce_mean 聚合后等于单机全量梯度', file: 'm02_data_parallel/demo.py' },
  { name: 'matrix', problem: '单层参数矩阵过宽超出单卡显存容量', equiv: '按列切分 W1、按行切分 W2 后经 all-reduce 求和等价于稠密 MLP', file: 'm03_tensor_parallel/demo.py' },
  { name: 'layers', problem: '网络层数过深且串行切分层导致设备空闲等待', equiv: '按层划分 stage 并通过 micro-batch 流水排程压缩空闲气泡 (bubble) 占比', file: 'm04_pipeline_parallel/demo.py' },
  { name: 'states', problem: '多卡重复存储参数、梯度与优化器状态导致显存冗余', equiv: '各卡对本地分片执行更新后再经 all-gather 收集与全量更新结果一致', file: 'm05_zero_fsdp/demo.py' },
  { name: 'precision', problem: '全精度计算占用显存与带宽高、难以利用低精度算力', equiv: '放大损失求导后再除以缩放因子 scale, 保护低精度微小梯度不下溢', file: 'm06_mixed_precision/demo.py' },
  { name: 'activation', problem: '深层网络前向缓存的全部中间激活值占满显存', equiv: '前向仅保留边界输入、反向传播前重算局部前向, 所得梯度与全缓存严格一致', file: 'm07_activation_checkpointing/demo.py' },
  { name: 'recovery', problem: '长周期训练遭遇硬件中断或异常更新步需要回滚', equiv: '完整恢复模型权重、优化器状态、数据游标与随机数生成器 (RNG) 状态后后续轨迹逐位一致', file: 'm08_checkpoint_resume/demo.py' },
]

const codeCards = [
  {
    title: '梯度累积',
    tag: 'mean reduction',
    desc: '每个 micro-batch 的损失在本地已按自身样本数取均值, 累加梯度时需乘以 len(xb) / len(x) 的样本占比权重, 才能还原完整 batch 的均值梯度。',
    file: 'llm_train/m01_gradient_accumulation/demo.py',
    code: `accum_grads = zeros_like(full_grads)
for xb, yb in micros:
    loss, grads = model.loss_and_grads(xb, yb)
    add_inplace(accum_grads, grads, scale=len(xb) / len(x))

assert max_abs_diff(full_grads, accum_grads) < 1e-6`,
  },
  {
    title: 'DDP 梯度同步',
    tag: 'all-reduce',
    desc: '每个 replica 仅使用分配到本地的样本分片计算梯度, 经 all_reduce_mean 跨卡同步取平均后, 所得全局梯度等价于在整批样本上直接反向求导。',
    file: 'llm_train/m02_data_parallel/demo.py',
    code: `local_grads = []
for rank, replica in enumerate(replicas):
    xb = split(x, world_size)[rank]
    _, grads = replica.loss_and_grads(xb, yb)
    local_grads.append(grads)

synced = all_reduce_mean(local_grads)
replica.apply_grads(synced, lr=0.1)`,
  },
  {
    title: '张量并行',
    tag: 'Megatron pattern',
    desc: 'MLP 第一层权重 W1 按列切分、激活函数在各卡本地独立执行, 第二层权重 W2 按行切分; 前向通过 all_reduce_sum 汇总各卡部分和并加上偏置 b2, 反向则将各分片梯度还原对齐至稠密权重形状。',
    file: 'llm_train/m03_tensor_parallel/demo.py',
    code: `W1_shards = split(W1, world, axis=1)
W2_shards = split(W2, world, axis=0)

h_shards = [relu(x @ W1_s) for W1_s in W1_shards]
partial  = [h @ W2_s for h, W2_s in zip(h_shards, W2_shards)]
tp_out   = all_reduce_sum(partial)[0] + b2`,
  },
  {
    title: 'ZeRO / FSDP',
    tag: 'state sharding',
    desc: '各卡梯度经 reduce_scatter_sum 聚合切分后, 每个 rank 仅对自身负责的参数分片执行更新, 需要完整权重时再通过 all_gather 收集拼接, 结果与单机稠密更新一致。',
    file: 'llm_train/m05_zero_fsdp/demo.py',
    code: `grad_shards  = reduce_scatter_sum(grad_per_rank, axis=0)
param_shards = split(param, world, axis=0)
updated      = [p - lr * g for p, g in zip(param_shards, grad_shards)]

gathered = all_gather(updated, axis=0)[0]
assert allclose(dense_updated, gathered)`,
  },
]

const basicLoop = `x, y = get_batch(...)
logits, cache = transformer_forward(W, x)
loss, dlogits = cross_entropy_forward_backward(logits, y)
grads = transformer_backward(dlogits, cache)
W, opt_state = adam_step(W, grads, opt_state, lr=LR)`

const distributedLoop = `x_ranks = split(x, world)
for rank in ranks:
    for xb, yb in micro_batches(rank):
        loss, grads = replica.loss_and_grads(xb, yb)
        accum += unscale(scale(grads)) * micro_weight

synced = all_reduce_mean(local_grads)
clipped = clip_by_global_norm(synced)
zero_style_sgd_step(replica0, clipped)
broadcast_params_to_replicas()`

const loopMap = [
  { simple: 'get_batch', full: '全局 batch 先按数据并行 rank 切分样本子集, 单个 rank 内再进一步拆分为多个 micro-batch', why: '兼顾多卡并行吞吐与单卡前向激活峰值显存控制' },
  { simple: 'backward', full: '各 micro-batch 反向求导并在本地加权累积后, 执行跨卡 all-reduce 求均值或 reduce-scatter 聚合分片', why: '在减少跨卡通信频次的同时保证所有设备获得一致的全局更新方向' },
  { simple: 'optimizer.step', full: '依次执行梯度反缩放 (unscale)、全局范数裁剪 (clip) 与 ZeRO 分片参数更新并广播同步', why: '消除损失缩放因子、抑制梯度爆炸并分摊优化器状态显存开销' },
  { simple: 'checkpoint', full: '同步持久化当前训练步数 step、模型权重、优化器动量、数据读取游标与随机数生成器 (RNG) 状态', why: '保证长周期训练发生中断后可精确恢复且后续训练轨迹与未中断基线一致' },
]
</script>

<style scoped>
table.train-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}
table.train-table th {
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
table.train-table td {
  padding: 10px 14px;
  border-bottom: 1px solid var(--border);
  vertical-align: top;
}
table.train-table .axis {
  color: var(--text);
  font-weight: 600;
  white-space: nowrap;
}
table.train-table .small {
  color: var(--text-muted);
  font-size: 12px;
}
.hint {
  margin-top: 10px;
  font-size: 12px;
  color: var(--text-dim);
  line-height: 1.6;
}
.compare-card {
  margin-top: 16px;
  padding: 0;
  overflow-x: auto;
}
</style>
