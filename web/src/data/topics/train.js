import { canonicalizeRelated } from '../../utils/consolidate.js'
import { chapterLabels } from '../chapterLabels.js'
import { mergePage } from '../../utils/consolidate.js'
// 阶段 3 · llm_train 的全部章节页 (完整定义)。
// 数字全部取自对应 demo 的实际输出, 改 Python 后请重新核对。
// points 里 key: true 的那一条 = 这一章只让人记住一句话时, 记这句。
const topic = {
  stage: 'train',
  chapters: [
    { route: 'train-pipeline-schedules', label: '1F1B 与交错调度', hint: '流水线为什么空等、怎么排能少等' },
    { route: 'train-lr-schedule', label: 'WSD 学习率调度', hint: '训到一半想加训, 学习率曲线怎么办' },
    { route: 'train-low-precision', label: 'FP8 与 FP4 微缩放', hint: '砍到 4 bit 为什么要靠一小块共用一个 scale' },
    { route: 'train-muon', label: 'Muon 优化器', hint: '让更新在每个方向上步子一样大的优化器' },
    { route: 'train-ulysses', label: 'Ulysses 序列并行', hint: '长序列切给多卡的另一种办法: 切头' },
    { route: 'train-data-packing', label: '数据流水线与 packing', hint: '改过几个字的转载怎么抓, 拼行怎么不串文档' },
    { route: 'train-scaling', label: '规模: scaling law 与 μP', hint: '钱定了, 模型多大、学习率多少怎么定' },
    { route: 'train-eval', label: '评测: PPL、污染、pass@k、裁判', hint: '每个评测数字都有口径' },
    { route: 'train-fault-tolerance', label: '容错与低通信训练', hint: '挂掉是常态: 多久存一次盘, 同步能少到什么程度' },
  ],
  pages: {
    // ---------------------------------------------------------------- 切 batch
    'train-batch-ddp': {
      title: 'batch 与 DDP · 同一个梯度的三种算法',
      subtitle: '单卡训练受限于激活显存与单卡算力。梯度累积在时间上分批累加, DDP 在多卡间并行求平均, 两者都在不改变全局梯度的前提下突破单卡限制。',
      tldr: '单步更新的目标是算出全局 $N$ 个样本的平均梯度, 按硬件条件有三种等价做法:\n- 单卡整批: 一次前向与反向算完整个 batch, 受单卡显存和算力限制。\n- 梯度累积: 在单卡上拆成 $K$ 个 micro-batch 串行计算并累加梯度, 用 $K$ 倍时间换取 $1/K$ 的激活显存峰值。\n- DDP (分布式数据并行): $P$ 张卡各持一份完整模型、各算 $1/P$ 数据, 再通过 all-reduce 同步平均梯度, 用通信换取近 $P$ 倍的吞吐提升。\n三者算出的梯度相等, 前提是每一份的样本数相同; 份数不等长时, 累加必须按样本数加权。',
      question: '梯度累积和 DDP 都在把一个 batch 拆成多份计算, 为什么前者无需通信, 后者却每步都必须在卡间同步梯度?',
      evolution: {
        title: '怎么切 batch 都得是同一个梯度',
        subtitle: "根问题: 稳定更新需要平均足够多样本的梯度, 但单张卡的显存容量与算力上限无法同时承载大 batch。",
        steps: [
          { name: '单卡整 batch', pain: "(原点) 梯度需要对大量样本求平均以降低方差, 但前向中间激活显存随 batch 大小线性增长", fix: "单卡一次前反向算完整个 batch; batch 稍大即触发显存溢出 (OOM), 吞吐也受限于单卡算力上限" },
          { name: '梯度累积', pain: "单张卡的显存无法容纳整个 batch 的前向中间激活", fix: "将全局批次拆成多个 micro-batch 串行计算并累加梯度; 因各份内部已取均值, 须乘样本占比 $n_k/N$ 加权才与整批等价" },
          { name: '异步参数服务器', year: 2012, pain: "单卡串行计算 $K$ 个 micro-batch 耗时过长, 需要多张卡并行分担计算量", fix: "各卡算完本地梯度立即推送至参数服务器并拉取新权重, 彼此无需同步等待; 代价是梯度常在陈旧参数上算出" },
          { name: '同步 DDP', pain: "陈旧梯度使单步更新无法对应确定的数据 batch, 收敛稳定性受损且训练结果难以复现", fix: "每步通过 all-reduce 求全局平均梯度, 所有副本用同一份梯度同步更新; 各卡有效样本数相等时与整批等价" },
        ],
      },
      code: 'llm_train/m01_gradient_accumulation/demo.py · llm_train/m02_data_parallel/demo.py',
      points: [
        {
          title: '核心概念: DDP、rank 与 all-reduce',
          body: '数据并行围绕三个基础概念展开:\n- DDP (Distributed Data Parallel, 分布式数据并行): $P$ 张计算卡各保存一份完整模型副本, 将全局 batch 切成 $P$ 片分别计算本地梯度; 代码中每张卡称为一个 rank, 编号从 0 开始。\n- all-reduce: 多卡集合通信原语。每张卡交出本地梯度张量, 通信完成后每张卡都拿到全体卡的梯度总和, 再除以卡数即得全局平均梯度。\n- micro-batch (微批次): 单张卡在显存受限时将一个 batch 拆成若干小份串行执行前向与反向, 梯度累加齐后再调用一次优化器更新。',
        },
        {
          title: '切分方式不同, 算出的全局平均梯度严格等价',
          key: true,
          body: '无论在时间上分多轮累加还是在空间上分多卡并行, 只要最终对同一批 $N$ 个样本取等权平均, 算出的梯度在数学上完全相同。因为全批次梯度定义为各样本梯度的算术平均, 改变分组求和顺序不会改变结果, 仅在浮点末位留下微小的舍入差异:\n- m01 (梯度累积): 将 8 个样本拆成 4 份各 2 个串行累积, 与整批一次算出的梯度最大绝对误差仅 2.98e-08。\n- m02 (DDP): 4 张卡各算 4 个样本, 经 all-reduce 平均后各自更新一步, 与单卡一次训 16 个样本的参数最大差仅 3.73e-09。\n两个实验均通过 assert 校验这一数值等价性。',
        },
        {
          title: '梯度累积省激活显存, DDP 省训练墙钟时间',
          body: '两种切分针对的是不同的硬件瓶颈:\n- 梯度累积降低激活显存峰值: 前向中间激活随单次送入的样本数线性增长。拆成 $K$ 个 micro-batch 后, 单次激活峰值降为 $1/K$ (m01 中从 8 个样本压到 2 个), 代价是串行执行耗时变为 $K$ 倍, 总计算量不变。\n- DDP 缩短单步耗时: $P$ 张卡同时处理各自分片, 计算耗时近似降为 $1/P$; 代价是每步需同步一次全模型梯度。采用环形 (ring) 算法时, 每张卡每步发送 $2(P-1)/P$ 份梯度字节 (8 卡为 1.75 份, 64 卡为 1.97 份), 上界恒为 2 份, 几乎不随卡数膨胀。\n需要注意 DDP 并不节省模型状态显存: 每张卡依然独立保存完整的参数、梯度和 Adam 优化器状态, 切分这部分冗余要靠后续章节的 ZeRO。',
        },
        {
          title: '数学等价的前提是按实际样本数加权',
          body: '每个 micro-batch 内部计算 loss (m01 中为 MSE) 时已经除以了自身的样本数 $n_k$。要把 $K$ 份局部平均还原成全局 $N$ 个样本的平均, 累加时每份梯度必须乘上样本占比 $n_k/N$ (对应代码 accumulate 中的 scale=n / len(x))。\n各份样本数恰好相等时 $n_k/N = 1/K$; 一旦漏乘缩放系数或遇到不等长切分就会引入隐蔽偏差:\n- m01 对照组 (weighted=False): 直接将 4 份梯度相加, 梯度幅值恰好偏大 4.00 倍, 在 SGD 下相当于把学习率放大了 4 倍。\n- 不等长切分 [3, 3, 2]: 简单除以份数 $1/K$ 会高估末尾短批次的权重, 只有按 $n_k/N$ 加权才能与整批梯度对齐 (误差 5.96e-08)。\n跨卡 DDP 同理: all_reduce_mean 对各卡局部均值做等权平均, 前提正是每张卡分到的有效样本数相同。',
        },
      ],
      links: [
        { from: 'llm_basic train.py', to: '梯度累积', body: '基础单卡循环每算一个 batch 立刻更新参数; 梯度累积只把参数更新推迟到累加满 K 个 micro-batch 之后, 其余前向与反向逻辑保持不变。' },
        { from: 'local grads', to: 'all_reduce_mean', body: '每张卡基于本地数据片算出的梯度只反映全局 batch 的一部分, 经 all_reduce_mean 求和并除以卡数后才得到一致的全局梯度; 若漏掉同步, 各卡权重将从该步起分叉。' },
        { from: 'DDP', to: 'ZeRO', body: 'DDP 在每张卡上完整复制参数、梯度和 Adam 状态, 单卡放不下模型状态时需进一步引入 ZeRO 按卡分片存储。' },
        { from: '2(N−1)/N', to: 'ring all-reduce', body: 'DDP 每卡每步发送 $2(N-1)/N$ 份梯度数据的系数来自环形接力算法 ring all-reduce, 具体分块传输过程见「通信与 full_loop」一章。' },
      ],
      sourceRows: [
        { concept: 'micro 权重', code: 'm01_gradient_accumulation/demo.py:accumulate', takeaway: '每份 micro-batch 梯度乘上样本占比 n / len(x) 后再累加; weighted=False 对照组将权重固定为 1, 用于展示漏除引发的尺度偏差。' },
        { concept: '不等长 micro', code: 'sizes=[3, 3, 2]', takeaway: '将 8 个样本切成 3、3、2 三份时, 简单按 1/K 平均与按样本数加权结果不同, 只有后者与整批训练等价。' },
        { concept: '梯度同步', code: 'core/collectives.py:all_reduce_mean', takeaway: '先对所有卡的梯度求和再除以卡数 world, 确保每张卡拿到逐位一致的更新量。' },
        { concept: '副本一致性', code: 'assert max_abs_diff(replicas[0], r) == 0', takeaway: '断言同步更新后所有 rank 的参数与 rank 0 严格逐位相等, 防止副本静默漂移。' },
        { concept: '通信账', code: 'core/collectives.py:CommCounter', takeaway: '统计每张卡每步发出的字节数: m02 中 56 B 梯度在 4 卡 DDP 下每卡发送 84 B, 精确对应 $2(N-1)/N$ 公式。' },
      ],
      snippetTitle: '累积的一步 vs DDP 的一步',
      snippet: `# 累积: 时间上串行, 每个 micro 只占 n/N 的激活
accum = zeros_like(params)
for xb, yb in micro_batches:                 # K 次前向 + 反向
    _, g = model.loss_and_grads(xb, yb)      # 这个 loss 已经是 xb 内部的均值
    add_inplace(accum, g, scale=len(xb) / N) # ★ 权重是样本数占比, 不是 1/K

# DDP: 空间上并行, 每卡只吃 N/P 个样本
local = [rep.loss_and_grads(xs, ys)[1] for rep, xs, ys in zip(replicas, x_shards, y_shards)]
synced = all_reduce_mean(local)              # 之后所有副本逐位相同
for rep in replicas:
    rep.apply_grads(synced, lr)`,
      source: ['llm_train/m01_gradient_accumulation/demo.py:accumulate', 'llm_train/core/collectives.py:all_reduce_sum'],
      run: 'python -m llm_train.m02_data_parallel.demo',
    },

    // ---------------------------------------------------------------- 切模型
    'train-model-parallel': {
      title: 'TP / PP 切模型 · 层内切矩阵, 层间切流水线',
      subtitle: '当模型本身超出单卡显存时, 张量并行 (TP) 在层内拆分矩阵乘法, 流水线并行 (PP) 按深度将连续网络层分配到不同设备。',
      tldr: '模型放不进单张卡时, 有两条正交的模型并行路线:\n- 张量并行 (TP): 针对过宽的单层, 将 MLP 的第一个矩阵 $W_1$ 按列切、第二个矩阵 $W_2$ 按行切, 中间的逐元素激活本地完成, 整个块前向与反向各只需一次 all-reduce。\n- 流水线并行 (PP): 针对过深的层数, 将连续若干层作为一个 stage 放到一张卡上, 卡间只传层边界激活; 通过把一个 batch 拆成 $M$ 个 micro-batch 连续送入流水线来压缩空闲等待 (气泡)。',
      question: '为什么 TP 每层内部都要通信, 而 PP 只在 stage 边界传一次激活? TP 前向已经做过一次 all-reduce, 为什么反向还要再做一次?',
      evolution: {
        title: '先切层, 再切矩阵',
        subtitle: "根问题: 模型自身体积超出单卡显存容量, 要求每张卡持有完整模型副本的 DDP 无法装载。",
        steps: [
          { name: '按层切', pain: "(原点) 模型参数规模超出单卡显存上限, 而 DDP 仍要求每张卡独立保存完整模型副本", fix: "将连续若干层划分为一个 stage 放入单卡, 沿层间依次传递边界激活; 代价是任意时刻仅有一张卡处于计算状态" },
          { name: 'GPipe', year: 2018, pain: "单批次串行流转时, 流水线上其余 $\\mathrm{PP}-1$ 张卡全程处于空闲等待状态", fix: "将全局 batch 拆成 $M$ 个 micro-batch 连续送入流水线, 空闲等待收敛至首尾气泡, 气泡占比随 $M$ 增大而下降" },
          { name: '朴素切矩阵', pain: "单个宽层的权重与激活本身已超出单卡显存, 仅按层间切分无法继续拆分", fix: "将单层权重矩阵拆分至多卡; 若先对 $W_1$ 按行切分, 进入非线性激活函数前就必须先跨卡同步一次部分和" },
          { name: 'Megatron TP', year: 2019, pain: "每次进入非线性激活函数前都需跨卡归约同步, 导致层内集合通信次数翻倍", fix: "改为 $W_1$ 按列切分使各卡在本地独立完成激活函数, $W_2$ 按行切分产出部分和, 直至块尾才执行一次 all-reduce" },
        ],
      },
      code: 'llm_train/m03_tensor_parallel/demo.py · llm_train/m04_pipeline_parallel/demo.py',
      points: [
        {
          title: 'Megatron TP: 先列切再行切, 避开激活函数前的同步',
          key: true,
          body: '两层 MLP 块的计算为 $\\mathrm{ReLU}(XW_1)W_2$。Megatron-LM 将两张权重矩阵按互补方向切给 $N$ 张卡, 把通信压到最少:\n- $W_1$ 按列切分 ([D, H] -> N × [D, H/N]): 每张卡读取完整输入 $X$, 独立算出隐藏层中属于自己的 $H/N$ 列。由于 ReLU 或 GeLU 都是逐元素运算, 各卡直接在本地分片上完成非线性变换, 无需跨卡通信。\n- $W_2$ 按行切分 ([H, O] -> N × [H/N, O]): 每张卡用刚算出的 $H/N$ 列激活乘对应的 $W_2$ 行分片, 得到输出张量的一个部分和, 最后在块尾做一次 all-reduce 求和即可拿到完整输出。\n若把顺序倒过来先对 $W_1$ 按行切, 算出部分和后必须先做一次 all-reduce 才能进激活函数, 通信次数直接翻倍。此外输出偏置 $b_2$ 必须在 all-reduce 求和之后只加一次, 否则会被重复累加 $N$ 遍。m03 实测 TP 输出与单卡完整矩阵乘误差为 0.00e+00。',
        },
        {
          title: '前向广播输入, 反向就必须汇总输入梯度',
          body: 'TP 的 MLP 块在前向末尾做了一次 all-reduce, 反向回传到块入口时还必须对输入梯度 $dX$ 再做一次 all-reduce。\n原因来自链式法则的分叉累加: 前向时同一个输入 $X$ 同时送给了全部 $N$ 张卡去乘各自的 $W_1$ 列分片; 反向时第 $r$ 张卡算出的局部 $dX_r$ 只包含它那 $H/N$ 列隐藏单元回传的梯度, 必须将 $N$ 张卡的局部 $dX_r$ 求和才能还原完整的上游梯度。m03 中若省去这次反向同步, 单卡局部 $dX$ 与真值偏差达 1.8e-2, 求和后误差降至 1.86e-09。\n注意力层按注意力头切分 $W_Q, W_K, W_V$ (列切) 并行切输出投影 $W_O$, 通信对称, 因此标准 Transformer 每层前向与反向合计需要 4 次 all-reduce。',
        },
        {
          title: 'PP: 气泡与在途激活',
          body: 'PP 将模型按连续层切成 PP 段, 每段作为一个 stage 独占一张卡, 卡间只在边界传递单层激活。代价是流水线首尾存在空闲气泡, 且未完成反向的在途激活需要驻留显存。',
        },
      ],
      links: [
        { from: 'llm_models Block', to: 'TP', body: 'TP 只切分 Transformer 块内的注意力投影与 MLP 大矩阵; LayerNorm、激活函数和残差相加均在本地独立完成。' },
        { from: 'num_layers', to: 'PP', body: '模型层数越深越适合按连续层划分为多个 stage, 每个 stage 内部保持单卡计算逻辑, 仅在段间点对点收发激活与梯度。' },
        { from: 'micro-batch 数 M', to: '气泡占比', body: '增大 micro-batch 数量 M 可以稀释首尾气泡占比, 但同时驻留的在途激活份数也受单卡显存容量制约。' },
        { from: 'PP', to: '1F1B 与交错调度', body: 'GPipe、1F1B 与交错 1F1B 三种排程如何在气泡比例与在途激活峰值之间做取舍, 将在下一章展开。' },
      ],
      sourceRows: [
        { concept: 'column parallel', code: 'W1_s = np.split(W1, world, axis=1)', takeaway: '将 [D, H] 的 $W_1$ 沿列切成 $N$ 份 [D, H/N], 每张卡用完整输入算出对应的隐藏层分片。' },
        { concept: 'row parallel + g 算子', code: 'tp_out = all_reduce_sum(partial)[0] + b2', takeaway: '将各卡行切矩阵乘得到的部分和通过 all_reduce_sum 相加, 偏置 b2 在归约后只加一次。' },
        { concept: 'f 算子的反向', code: 'tp_dx = all_reduce_sum(dx_partial)[0]', takeaway: '前向恒等传递完整输入 $X$, 反向调用 all_reduce_sum 汇总各卡回传的局部输入梯度。' },
        { concept: '流水线排程', code: 'm04_pipeline_parallel/demo.py:simulate', takeaway: '按「当前设备空闲且上游依赖就绪」的最早时刻推进离散事件模拟, 统计总时长与气泡比例。' },
      ],
      snippetTitle: 'TP MLP 的前向与反向',
      snippet: `z_s = [x @ w + b for w, b in zip(W1_s, b1_s)]     # 列切: N × [B, H/N]
h_s = [relu(z) for z in z_s]                      # 逐元素, 无需通信
partial = [h @ w for h, w in zip(h_s, W2_s)]      # 行切: 每份只是部分和
tp_out = all_reduce_sum(partial)[0] + b2          # 前向唯一一次通信

dx_partial = [dz @ w.T for dz, w in zip(d_z_s, W1_s)]
tp_dx = all_reduce_sum(dx_partial)[0]             # 反向唯一一次通信`,
      source: ['llm_train/m03_tensor_parallel/demo.py:main'],
      run: 'python -m llm_train.m03_tensor_parallel.demo',
    },

    // ---------------------------------------------------------------- 切状态 / 省显存 / 可恢复
    'train-memory': {
      title: '状态与显存 · ZeRO、激活重算与可恢复',
      subtitle: '训练显存由模型状态与中间激活两部分组成。ZeRO 消除多卡间的模型状态冗余, 激活重算以额外前向换取更低的激活峰值, 完整断点保存则保证长程训练随时可无损续跑。',
      tldr: '- 混合精度 Adam 训练中, 每个参数占用 16 字节模型状态 (2 字节 fp16 参数 + 2 字节 fp16 梯度 + 12 字节 fp32 主权重与一二阶矩)。ZeRO-1/2/3 依次将优化器状态、梯度、参数按卡数 $N$ 切片, 被摊薄的部分分别为 12 / 14 / 16 字节。\n- 前向中间激活随层数 $L$ 线性增长; 激活重算只保留分段入口并在反向前局部重算, 用约 25%~33% 的额外计算把激活峰值压到 $O(\\sqrt{L})$。\n- 断点续训要做到逐位无损恢复, 检查点必须同时保存参数、优化器状态、数据游标与随机数生成器 (RNG) 状态。',
      question: '7B 模型在 64 张卡上开启 ZeRO-1 时, 单卡模型状态还要占约多少 GB? 为什么继续加卡也压不下去?',
      evolution: {
        title: '把 16 字节一份份切走',
        subtitle: "根问题: 单卡显存容量固定, 模型参数、梯度、优化器状态与前向中间激活共同争用同一份显存预算。",
        steps: [
          { name: 'DDP 全复制', pain: "(原点) 混合精度 Adam 训练中每个参数占用 16 字节静态状态, 其中 12 字节来自优化器状态", fix: "每张卡完整保存一份模型与优化器状态, 实现最直接; 但无论扩充多少张卡, 单卡状态显存恒为 $16\\Psi$" },
          { name: 'ZeRO-1/2', year: 2019, pain: "$N$ 张卡重复存储着 $N$ 份完全相同的优化器状态与梯度", fix: "将优化器状态与梯度按卡数切成 $N$ 片, 每张卡仅更新自身负责的分片; ZeRO-2 的单步通信量与 DDP 持平" },
          { name: 'ZeRO-3 / FSDP', pain: "计算权重仍由每张卡整份持有, 这 $2\\Psi$ 字节的静态显存下限无法随加卡摊薄", fix: "将参数也按卡切片存储, 逐层计算前临时 all-gather 拼回完整权重并在算完后立即释放; 总通信量升至 DDP 的约 1.5×" },
          { name: '激活重算', year: 2016, pain: "静态模型状态均已分片, 但不在 16 字节账本内的前向中间激活仍随网络层数 $L$ 线性增长", fix: "前向仅保留分段入口激活, 反向求导前局部重跑段内前向: 激活显存峰值压至 $O(\\sqrt{L})$, 浮点计算量增加约 1/3" },
          { name: '可恢复 checkpoint', pain: "显存瓶颈缓解后训练规模扩至更多设备与更长周期, 硬件故障或任务中断成为常态", fix: "检查点同时保存参数、优化器状态、数据游标与 RNG 状态四项; 漏存任一项虽不报错, 续训参数轨迹却会发生偏离" },
        ],
      },
      code: 'llm_train/m05_zero_fsdp/demo.py · llm_train/m07_activation_checkpointing/demo.py · llm_train/m08_checkpoint_resume/demo.py',
      points: [
        {
          title: '2 + 2 + 12 的显存账本与三档 ZeRO 切分',
          key: true,
          body: '混合精度 Adam 训练下, 设模型参数量为 $\\Psi$、卡数为 $N$, 每个参数的静态显存由三块构成: fp16 计算权重 (2 字节)、fp16 梯度 (2 字节) 以及 fp32 优化器状态 (主权重 4 字节 + 一阶矩 $m$ 4 字节 + 二阶矩 $v$ 4 字节 = 12 字节), 合计 $16\\Psi$。由于 Adam 逐坐标独立更新, 多卡无需重复保存全部状态:\n- DDP: $16\\Psi$ (全量复制)\n- ZeRO-1 (切优化器状态): $4\\Psi + 12\\Psi/N$\n- ZeRO-2 (再切梯度): $2\\Psi + 14\\Psi/N$\n- ZeRO-3 / FSDP (连参数一起切): $16\\Psi/N$\nm05 中 $\\Psi=192, N=4$, 统计 rank 0 状态字节数依次为 3072 / 1344 / 1056 / 768 B, 与理论公式逐字节吻合, 且三档更新后的参数与 DDP 差值均为 0.0e+00。代入 7B 模型、64 卡 ZeRO-1 可得 $7\\times 10^9 \\times (4 + 12/64) \\approx 29\\text{ GB}$, 其中未分片的 $4\\Psi$ (28 GB) 是继续加卡也除不掉的下限。',
        },
        {
          title: '通信代价: ZeRO-2 与 DDP 持平, ZeRO-3 约 1.5 倍',
          body: '把状态切片后, 计算时需要靠集合通信按需拼回:\n- ZeRO-2 不增加通信量: 一次标准 all-reduce 本就可拆成 reduce-scatter (归约求和并各留 $1/N$ 分片) 与 all-gather (各片拼回整份) 两步。ZeRO-2 只是在 reduce-scatter 拿到梯度分片后原地完成 Adam 更新, 再用 all-gather 广播更新后的参数分片, 单步总通信量仍为 DDP 的 1.00×。\n- ZeRO-3 / FSDP 通信约为 1.5×: 由于参数本身也分片存储, 前向和反向每算一层前都需临时 all-gather 该层完整 fp16 参数, 算完立即释放 (demo 中瞬时峰值仅多出单层 256 B), 多出的一次前向 all-gather 使总通信升至 DDP 的 1.5 倍。(注: m05 中 ZeRO-1 简化用了 all-reduce 加 all-gather 故计为 1.5×, 工业实现改用 reduce-scatter 后同样为 1.0×。)',
        },
        {
          title: '激活重算: 多花约三成计算把激活峰值压到 √L',
          body: '静态模型状态之外, 反向求导还需要前向各层的输入激活, 全存时空间复杂度为 $O(L)$。激活重算每隔 $k$ 层设一个检查点, 前向只保留段入口激活, 反向退到某一段时再从段入口重跑一次段内前向。\n任意时刻驻留的激活份数约为 $L/k + k$ ($L/k$ 个段边界加上当前段重算出的 $k$ 层瞬时激活), 取 $k = \\sqrt{L}$ 时达到极小值 $O(\\sqrt{L})$。m07 在 $L=16$ 层模型上实测: 全存 ($k=1$) 峰值占 17 份, 取 $k=4$ 降至 8 份, 算出的梯度与全存基线逐位相同 (0.0e+00); 代价是重跑了一次段内前向, 浮点计算量增加 25% (理论上界约 33%)。',
        },
        {
          title: '无损断点续训依赖四类状态同时落盘',
          body: '长周期训练要求中断恢复后的参数轨迹与不中断训练逐位一致, 为此检查点必须收齐下一步计算依赖的四项状态: 模型权重、优化器动量状态、数据流读取游标 (cursor) 与随机数生成器 (RNG) 状态。\nm08 在第 3 步存盘并在新进程更换初始种子后恢复训练至第 5 步: 四项齐全时参数误差为 0.0e+00; 若单独漏存 RNG 状态 (dropout 掩码改变)、优化器动量或数据游标, 程序均不报错, 但两步后参数立刻产生 3.1e-2、5.8e-2 与 1.5e-1 的偏差。',
        },
      ],
      links: [
        { from: 'DDP 每卡全量', to: 'ZeRO 分片', body: '将全量复制改为分片存储换取显存空间: ZeRO-2 通信开销与 DDP 相同, ZeRO-3 则用额外一次前向 all-gather 换取参数分片。' },
        { from: 'llm_basic 的 cache', to: '激活重算', body: '阶段 1 手写反向把全部前向 cache 留在内存中; 深层大模型通过分段只存边界并在反向前局部重跑来控制显存峰值。' },
        { from: 'ToyDataStream.cursor', to: 'resume', body: '数据流游标记录当前消费到了第几个 batch, 漏存会导致续训重复读取旧样本或跳过样本。' },
        { from: 'reduce-scatter', to: '通信原语', body: '将 all-reduce 拆解为 reduce-scatter 与 all-gather 两半, 正是理解 ZeRO-2 零额外通信开销的关键, 详见「通信与 full_loop」。' },
      ],
      sourceRows: [
        { concept: '状态怎么切', code: 'm05_zero_fsdp/demo.py:init_rank_state', takeaway: '按所选 stage 决定 p16、g16、master、m、v 中哪些存整份、哪些只切本卡负责的 $1/N$ 分片。' },
        { concept: '梯度同步的分水岭', code: 'reduce_scatter_sum(layer_g)', takeaway: 'stage 0/1 使用 all_reduce, stage 2/3 改用 reduce_scatter 让每张卡只保留自身分片的梯度。' },
        { concept: 'FSDP gather → compute → free', code: 'full = all_gather([rk["p16"][i] for rk in ranks])', takeaway: 'ZeRO-3 逐层临时拼回完整 fp16 权重完成前向或反向, 计算结束立刻 del full 释放内存。' },
        { concept: '激活账本', code: 'm07_activation_checkpointing/demo.py:run', takeaway: '同时追踪常驻段边界与反向重算期间的段内瞬时激活, 取全过程最大值作为真实激活峰值。' },
        { concept: 'resume payload', code: 'm08_checkpoint_resume/demo.py:snapshot', takeaway: '将步数、权重、优化器状态、数据流种子与游标、NumPy RNG 状态统一打包序列化。' },
      ],
      snippetTitle: 'ZeRO 一步 (stage 2/3)',
      snippet: `# 每 rank 常驻: p16 (stage 3 只存 1/N), master/m/v (stage>=1 只存 1/N)
for i in range(L):                                  # FSDP 前向: 逐层 gather → 算 → free
    full = all_gather([rk.p16[i] for rk in ranks])
    h = layer_fwd(h, full)
    del full

g_shard = reduce_scatter_sum(layer_grads) / world   # 每 rank 只留自己那片梯度
adam_update(master_shard, g_shard, m_shard, v_shard, t, lr)
p16_shard = master_shard.astype(float16)            # 下一步再按需 gather`,
      source: ['llm_train/m05_zero_fsdp/demo.py:init_rank_state', 'llm_train/m07_activation_checkpointing/demo.py:run'],
      run: 'python -m llm_train.m05_zero_fsdp.demo',
    },

    // ---------------------------------------------------------------- 数值格式
    'train-precision-stability': {
      title: '精度与稳定性 · FP16 / BF16 / FP8 与坏 step 防护',
      subtitle: '缩减浮点位宽能成倍降低带宽与算力开销, 但也压缩了可表示的数值范围与尾数精度。现代训练通过 BF16 动态范围、fp32 主权重与溢出跳步机制守住收敛稳定性。',
      tldr: '本章采用的混合精度配方是「低精度做矩阵乘, 高精度存主权重 (fp32 master)」; 这不是所有实现的必要条件, 也可以用 Kahan 补偿或随机舍入代替 fp32 master:\n- FP16 指数位窄, 极小梯度易下溢归零, 需配合动态 loss scaling 先放大损失再还原, 遇到 Inf 溢出则跳过该步并将缩放倍数减半。\n- BF16 保留与 FP32 相同的 8 位指数位, 以尾数精度换取动态范围, 省去了动态 loss scaling。\n- 进一步降到 FP8 / FP4 时, 格式自身范围已不足以直接容纳张量分布, 必须按张量或按小块配备缩放系数 scale; 本章同时保留 fp32 主权重承接细微更新 (m13 中去掉主权重会导致验证损失恶化 584 倍)。',
      question: '为什么混合精度训练不能直接把所有权重和梯度转成 float16? BF16、FP8 和 FP4 各自靠什么补齐位宽缩减带来的损失?',
      evolution: {
        title: '位宽每砍一刀, 就补一道保险',
        subtitle: "根问题: 缩减浮点位宽能成倍节省显存、访存带宽与矩阵乘算力, 但同时压缩了可表示的动态范围与尾数精度。",
        steps: [
          { name: '全程 FP32', pain: "(原点) 单精度每个元素占 4 字节, 显存占用、访存带宽与乘加算力开销均随位宽成倍消耗", fix: "动态范围与尾数精度充裕、数值收敛最稳定; 代价是每次张量搬运与矩阵乘法都承担 32 位全量开销" },
          { name: 'FP16 混合精度', year: 2017, pain: "直接全量转为 FP16 时动态范围过窄: 1e-8 量级的梯度下溢为 0, 70000 的激活上溢为 Inf", fix: "反向前放大 loss 并在更新前除回, 参数更新累加在 FP32 主权重 (master) 上; 检出溢出则跳过该步并将 scale 减半" },
          { name: 'BF16', pain: "loss scale 需全程动态试探调节, 一旦触发上溢仍要作废并跳过当前训练步", fix: "保留与 FP32 相同的 8 位指数位, 上下界均不易溢出而省去 loss scaling; 代价是尾数精度缩减至 7 位" },
          { name: 'FP8', year: 2022, pain: "位宽再减半至 8 位后, 格式自身的动态范围已无法直接容纳训练张量的数值分布", fix: "E4M3 (4 位指数 3 位尾数) 与 E5M2 须配备量化 scale 将张量映射入量程; 更新仍依赖 FP32 master, 移除后验证 loss 恶化 584×" },
        ],
      },
      code: 'llm_train/m06_mixed_precision/demo.py · llm_train/m10_training_stability/demo.py · llm_train/m13_fp8_training/demo.py · llm_train/core/numerics.py',
      points: [
        {
          title: '同样 16 位: FP16 保尾数精度, BF16 保指数范围',
          body: '浮点数由符号位、指数位 (决定动态范围) 与尾数位 (决定相邻格点间距) 组成。同样 16 位预算下, 两种格式做了不同取舍:\n- FP16 (5 位指数 + 10 位尾数): 最大可表示正数为 65504, 最小非零正数约 6e-8; 因此 1e-8 量级的梯度会直接下溢成 0, 70000 的激活会直接上溢成 Inf。\n- BF16 (8 位指数 + 7 位尾数): 指数位宽与 FP32 完全一致, 1e-8 与 70000 都能正常落入范围; 代价是在 1.0 附近最小步长只有 $2^{-7}$ (FP16 为 $2^{-10}$), 例如 $1+2^{-9}$ 在 BF16 下会被直接舍入回 1。\n对深度网络训练而言, 上下溢会导致梯度断流或整步报废, 远比尾数粗糙几位的危害大, 因此现代大模型预训练普遍默认采用 BF16。',
        },
        {
          title: 'loss scaling 防梯度下溢, fp32 master 防微小更新丢失',
          key: true,
          body: 'FP16 混合精度训练需要两道保护机制协同工作:\n- 动态 loss scaling: 反向传播前先将标量损失乘上放大因子 $S$, 使整条链路的梯度同步放大 $S$ 倍从而避开 FP16 下溢区, 优化器更新前再转回 fp32 除以 $S$。由于最优 $S$ 随训练动态变化, 若某步检出 Inf/NaN 则跳过参数更新并将 $S$ 减半, 连续若干步无溢出再将 $S$ 翻倍 (m06 从 $2^{20}$ 起步, 24 步内自动跳过 5 次真实 fp16 溢出, 损失仍从 1.6477 顺利收敛至 0.0229)。\n- fp32 主权重 (master weights): 在权重约 1.0 附近, FP16 相邻可表示值相差约 0.001, 而单步 lr * grad 往往在 1e-5 量级。若直接在 fp16 权重上累加, 100 步后数值仍停在 1.0 原地不动; 保留一份 fp32 主权重精确累积每步微小更新, 每步前向时再转出 fp16 计算副本, 才能让微小步长持续生效。',
        },
        {
          title: 'FP8 的格式分工与缩放粒度',
          body: '位宽继续减半到 8 位时, 常见 E4M3 (4 位指数 + 3 位尾数, 上限 448, 格点更密) 与 E5M2 (5 位指数 + 2 位尾数, 上限 57344, 范围更宽) 两种格式。量化前需将一组元素除以缩放因子 scale = amax / max_val 映射进格点区间, 业界主流有两套配方:\n- Transformer Engine 方案: 前向用更准的 E4M3, 反向梯度用范围更宽的 E5M2, 整张量共用一个缩放因子 (per-tensor)。\n- DeepSeek-V3 方案: 前反向统一用 E4M3, 缩放到每 128 个元素独立算一个缩放因子 (block scaling)。\n在 m13 的无离群值回归上两者验证损失打平 (约 4.6e-4); 当在 3 个 token 中注入达到正常值 1e5 倍的极端离群值 (outlier) 时, per-tensor 因整张量共享过大 scale 导致 17.8% 的正常元素下溢归零, 其余 token 输出相对误差升至 0.126, 而分块缩放将影响限制在局部块内, 误差保持在 0.026。',
        },
        {
          title: 'FP8 消融中保留 fp32 主权重最为关键',
          body: 'm13 在贴近真实大模型尺度 (权重标准差 0.004, 梯度量级 1e-5) 的任务上做了逐项消融, 对比 200 步后的验证损失:\n- 去掉张量缩放 (scaling): 损失恶化 29 倍, 小梯度成片落入欠表示区间。\n- 去掉 fp32 主权重 (直接在 FP8 格点上更新权重): 损失恶化高达 584 倍, 因为小于 FP8 格点间距的更新在每步舍入中被直接抹除。\n另需注意, FP8 最好的几组停在约 4.6e-4, 而 FP32 基线是 1.02e-08: 这个玩具回归问题没有噪声, 量化误差直接露了出来; 真实 LLM 的 loss 噪声远大于它, 这个差距不能外推。',
        },
        {
          title: '2026 年的现状: 全栈 FP4 连优化器状态和注意力一起量化',
          body: '早期的 NVFP4 预训练只把 Transformer 的线性投影量化到 4 位, 优化器状态、优化器自身的计算和注意力的前反向路径仍是高精度。Full-Stack FP4 (arXiv 2607.04422) 把这三块也量化了, 四份配方各解决一个问题:\n- 线性投影: LoRA-SVD 把一个紧凑的投影子空间留在 BF16, 其余保持全形状 NVFP4 计算, 只量化线性层的 loss 差距从 1.40% 降到 0.61%。\n- 优化器状态: 先开平方、再按 tile 取均值、再做 Hadamard 变换的顺序化流水线, 让 AdamW 的动量能稳定地存成 NVFP4。\n- 优化器计算: 正交化迭代 (论文称 Root, 就是 Muon 那一步) 直接在 NVFP4 上做, 靠形状相关的系数和裁剪稳住。\n- 注意力: 混合精度, 对 softmax 敏感的算子留在 BF16, 不是全部 FP4。\n3B 模型 64B token 预训练的结果是 BF16 的 loss 2.267 对 FP4 的 2.286, 差 0.838%; 平均 zero-shot 困惑度 26.675 对 26.665, 准确率平均低 0.10 个百分点。单张 RTX 5090 上的四块实测: Root 比优化过的 BF16 快 2.50 到 2.83 倍, AdamW 峰值显存低 37.9% 到 42.5% —— 这两个数字是单算子加速和单项显存, 不是端到端训练的加速比。\n这和上一条消融里 "去掉 fp32 主权重损失恶化 584 倍" 不矛盾: 被量化的是 AdamW 的动量, 主权重仍然是高精度那一份。',
        },
      ],
      links: [
        { from: 'Adam state', to: 'fp32 master', body: '无论计算层采用 FP16、BF16 还是 FP8, 单步微小更新都需要高精度容器累加; 本库采用 fp32 主权重与高精度 Adam 一二阶矩 (工业界也有结合随机舍入或 Kahan 补偿的省显存变体)。' },
        { from: 'global grad norm', to: 'DDP/FSDP', body: '全局梯度范数裁剪必须排在跨卡梯度同步与除以 loss scale (unscale) 之后, 否则测到的是残缺分片或被放大后的伪范数。' },
        { from: 'LossScaler.scale', to: 'checkpoint', body: '动态缩放器的当前 scale 与连续正常步数 good_steps 均属于训练状态, 断点存盘时需一并写入。' },
        { from: 'FP8 block scaling', to: 'FP4 微缩放', body: 'FP8 可按 128 个元素分块共享 scale; 当位宽再降到动态范围仅 12 倍的 FP4 时, 分块粒度必须进一步缩至 16~32 个元素, 详见下方进阶小节。' },
      ],
      sourceRows: [
        { concept: '动态 loss scale', code: 'm06_mixed_precision/demo.py:LossScaler', takeaway: '检出溢出时将 scale 减半并跳过本步更新, 连续 growth_interval 步正常后翻倍, 试探 scale 的上限。' },
        { concept: '浮点格式三要素', code: 'core/numerics.py:fake_quant_float', takeaway: '统一根据尾数位数、最大值与最小正规数指数模拟 BF16 / FP8 / FP4 的格点舍入与饱和截断。' },
        { concept: 'NaN guard + clip', code: 'm10_training_stability/demo.py:guarded_step', takeaway: '先拦截 Inf/NaN 坏步直接返回 False, 通过后再做全局范数裁剪与参数更新 (若顺序反过来, NaN 会污染整个范数)。' },
        { concept: '裁剪不改方向', code: 'g * min(1, max_norm / ‖g‖)', takeaway: '对全体梯度统一乘缩放比例, 将总范数从 111.83 压至 5.00 的同时保持更新方向余弦恒为 1.000000。' },
        { concept: 'FP8 消融的旋钮', code: 'm13_fp8_training/demo.py:make_quantizer', takeaway: 'A~E 五组对照每次只改动一个变量 (开关缩放、切换反向格式、per-tensor 改 block、移除 fp32 master), 隔离每项设计的独立贡献。' },
        { concept: 'E4M3 max = 448', code: 'FLOAT_FORMATS["e4m3"]', takeaway: 'E4M3 保留最高尾数编码表示 NaN, 最大有限值为 448 而非 480, 超界尖峰会被饱和截断至 448。' },
      ],
      snippetTitle: 'AMP 的一步 (含跳步)',
      snippet: `half = model.astype(float16)                     # fp16 计算副本
loss, g16 = half.loss_and_grads(x, y, loss_scale=scaler.scale)

if has_overflow(g16):                            # Inf 来自 fp16 乘法本身超过 65504
    skipped += 1                                 # 跳过: master 不能被 Inf 污染
else:
    g32 = {k: g.astype(float32) / scaler.scale for k, g in g16.items()}
    clipped, _, _ = clip_by_global_norm(g32, max_norm)
    master.apply_grads(clipped, lr)              # 更新发生在 fp32 master 上
scaler.update(overflow)                          # 溢出减半, 连续好 step 翻倍`,
      source: ['llm_train/m06_mixed_precision/demo.py:LossScaler', 'llm_train/core/numerics.py:fake_quant_float'],
      run: 'python -m llm_train.m06_mixed_precision.demo',
    },

    // ---------------------------------------------------------------- 切专家 / 切序列
    'train-moe-seq': {
      widgets: ['MoeRouteLab', 'RingAttnLab'],
      title: 'EP 与序列并行 · 切专家, 切序列',
      subtitle: '混合专家模型 (MoE) 需要把不同专家分布到多张卡上动态路由 token, 超长上下文则需要把单条序列拆到多张卡上协同计算注意力。',
      tldr: '- 专家并行 (EP): 将多个专家子网络分布到不同设备上, 每步通过两次 all-to-all 把 token 发往目标专家所在卡并在算完后寄回。与通信量恒定的 DDP 不同, EP 每卡收发多少数据取决于当步路由结果, 负载失衡会导致慢卡拖尾或溢出丢 token。\n- Ring Attention (环状序列并行): 将长序列切成 $D$ 段分给各卡, Q 留在本地, K/V 分块沿环传递并用 online softmax 增量合并, 计算结果与单卡全量注意力精确一致; 在因果掩码下配合首尾配对的 zigzag 切分才能让各卡计算负载均衡。',
      question: '为什么 MoE 跨卡通信的流量由当步输入数据动态决定? Ring Attention 分块流式计算注意力, 为什么能与单卡一次性 softmax 精确相等?',
      evolution: {
        title: '参数和算力脱钩之后',
        subtitle: "根问题: 扩充参数规模能提升模型容量, 但稠密架构每增加一份参数, 每个 token 都必须多承担一份浮点计算。",
        steps: [
          { name: '稠密 FFN', pain: "(原点) 稠密网络中每个输入 token 都必须参与全部模型参数的矩阵运算", fix: "总参数量与单 token 激活计算量严格绑定: 参数规模翻倍时, 每步计算开销也随之翻倍" },
          { name: '稀疏门控 MoE', year: 2017, pain: "希望继续扩大模型参数容量, 却无法承受同比例增长的单步算力开销", fix: "将 FFN 替换为一组并行专家子网络, 由路由器为每个 token 仅激活其中少数几个; 总参数扩充而单 token 计算量保持不变" },
          { name: 'GShard 专家并行', year: 2020, pain: "专家总数增加后, 单张卡显存已无法容纳全部专家权重", fix: "将专家均分至多张卡, 用 all-to-all (各卡向其余每卡发送专属数据块) 把 token 发往目标卡并在算完后寄回; 流量取决于当步路由分布" },
          { name: 'Switch aux loss', year: 2021, pain: "门控路由向热门专家倾斜: 最忙设备负载超出均值 50%, 超出专家容量上限的 token 被直接丢弃", fix: "在损失中加入辅助项 $E\\sum f_e P_e$, 接收 token 越多的专家其路由概率受到的下压梯度越强; 最忙与平均负载比 (max/mean) 从 1.62× 降至 1.12×" },
          { name: '无辅助损失 bias', year: 2024, pain: "辅助损失直接并入训练目标求导, 在推平路由分布的同时会干扰主语言建模损失的优化方向", fix: "为每个专家维护仅用于选人排序的偏置 bias, 过载下调、空闲上调; 全程不进入计算图也不产生额外梯度, max/mean 降至 1.00×" },
        ],
      },
      code: 'llm_train/m11_expert_parallel/demo.py · llm_train/m12_sequence_parallel/demo.py',
      points: [
        {
          title: 'EP 的跨卡流量由当步门控路由动态决定',
          key: true,
          body: 'MoE 将一层内的 FFN 替换为 $E$ 个并行的专家子网络, 每个 token 先由轻量路由器 (router) 打分并选出 top-k 个专家 (m11 采用 top-1)。当专家总数超出单卡容量时, 专家并行 (EP) 把 $E$ 个专家均分到 $D$ 张卡上 (demo 为 4 卡共 8 个专家, 每卡 2 个)。由于 token 所在卡往往不是目标专家所在卡, 前向需经历两趟 all-to-all 通信:\n- dispatch (分发): 各卡按每个 token 选中的专家整理目标设备, 将 token 发往对应卡。\n- combine (回收): 专家计算完成后将输出原路寄回源卡, 源卡按最初记录的本地行号 (send_idx) 归位并乘上门控概率。\n(demo 中 dispatch 将隐状态与专家编号分两次调用通信接口, 计数显示 3 次 all-to-all, 逻辑上仍是一去一回。) 与每步搬运固定字节数的 DDP 不同, EP 每张卡每步收发多少 token 完全取决于路由器对该批数据的分配结果。',
        },
        {
          title: '路由倾斜会引发木桶等待与容量溢出丢弃',
          body: '门控路由一旦向少数热门专家扎堆, 会同时带来两方面损耗:\n- 慢卡拖慢全局同步: m11 中 64 个 token 分往 4 张卡, 倾斜路由下各卡实际收到 [11, 18, 11, 24] 个 token (理想均分为 16 个), 最忙的卡比均值多算 50%, 其余三张卡只能空等。\n- 超出容量上限直接丢弃: 为固定显存与算力上界, 静态实现常设置单专家容量上限 $\\text{capacity} = \\lceil \\text{cf} \\cdot N_{\\text{tok}} / E \\rceil$, 即平均负载乘上余量系数 cf (capacity factor)。m11 中 64 个 token、8 个专家、cf=1.25 对应单专家容量 10, 倾斜路由下各专家分到 [1, 10, 11, 7, 4, 7, 11, 13] 个 token, 超出 10 的 5 个 token 直接跳过专家层仅走残差连接。盲目调大 cf 又会成倍浪费预留显存与算力, 因此必须在训练中主动约束路由均衡。',
        },
        {
          title: '负载均衡的两代方案: 辅助损失 vs 无梯度偏置调节',
          body: '用最忙专家负载与平均负载之比 (max/mean, 理想值 1.00×) 衡量均衡度, 上述倾斜路由初始高达 1.62×:\n- 辅助损失 (Switch / Mixtral): 在总损失中加入 $L_{\\text{aux}} = E \\cdot \\sum_e f_e P_e$, 其中 $f_e$ 为专家 $e$ 实际接收的 token 比例 (由不可导的 argmax 统计, 视作常数), $P_e$ 为路由器给出的平均概率。越拥挤的专家 $f_e$ 越大, 其路由概率 $P_e$ 受到的下压梯度越强。训练 60 步后 max/mean 降至 1.12×, 溢出丢弃清零; 但辅助损失梯度会与主语言建模损失相互干扰。\n- 无辅助损失偏置 (DeepSeek-V3 aux-loss-free): 为每个专家维护一个不参与反向传播的标量 bias, 仅在选择 top-k 时按 概率 + bias 排序, 前向加权仍乘原始概率; 每步结束后将过载专家的 bias 下调 $\\gamma$、空闲专家上调 $\\gamma$。m11 迭代 200 步后 max/mean 精准降至 1.00× (8 个专家各收 8 个 token), 且全程不向计算图注入额外梯度。',
        },
        {
          title: 'Ring Attention 靠 online softmax 流式合并分块注意力',
          body: '超长序列单卡装不下完整 K/V 时, Ring Attention 将序列切成 $D$ 段分放各卡, 本地固定持有本段 Q, 而 K/V 分块沿环依次右移传递 $D-1$ 轮。为了在未见完全部 K 之前就开始算 softmax, 每行 query 维护三个流式统计量: 历史最大分数 $m$、指数和分母 $\\text{den}$ 与加权输出累加器 $\\text{acc}$。每收到一块新 K/V:\n- 更新全局最大值 $m^\\prime = \\max(m, m_{\\text{block}})$。\n- 将旧统计量统一乘重缩放因子 $e^{m - m^\\prime}$ 对齐到新基准, 再累入当前块的指数和与加权值。\n走完一圈输出 $\\text{acc} / \\text{den}$, 这是代数恒等变形而非近似: m12 实测与单卡全量注意力最大差仅 4.4e-16, 且任意时刻单卡只需常驻 $1/D$ 的 K/V。',
        },
        {
          title: '因果掩码下需用首尾配对的 zigzag 切分消除负载倾斜',
          body: '引入因果掩码后, 整块落在当前 query 未来的 K/V 块可以直接跳过, 总计算量省去约一半 (m12 中 $T=32$ 的有效点积对数从 1024 降至 528)。但若按自然顺序连续切分, 排在序列末尾的最后一张卡需要看全部前文, 每轮都满载计算, 其余卡省下的算力全花在等它上 (模拟墙钟从无掩码的 256 仅微降至 228)。\nZigzag 切分将序列切成 $2D$ 小段, 让第 $r$ 张卡同时持有靠前的第 $r$ 段和靠后的第 $2D-1-r$ 段 (首尾对称配对)。这样除首轮外每张卡每轮的有效计算量完全相等 (均为 32 对), 模拟墙钟直接降至 132, 相比连续切分提速 1.73 倍。',
        },
      ],
      links: [
        { from: 'gating top-k', to: 'all-to-all dispatch', body: '门控网络为每个 token 挑出目标专家后, 由 all-to-all 将跨卡 token 重排发往目标设备, 计算完毕后再逆向排回原位。' },
        { from: 'capacity factor', to: 'token dropping', body: '容量系数 cf 设定单专家每步可接纳的最大 token 数, 超额到达的 token 跳过专家计算直接走残差支路。' },
        { from: 'KV 环传递', to: 'online softmax (m, l, acc)', body: '环状传递每送来一块 K/V, 本地就用指数重缩放将它并入 (m, l, acc) 三个统计量, 单卡无需拼出完整序列的 K/V。' },
        { from: '因果 mask', to: 'zigzag 切分', body: '因果注意力越靠后的位置计算越重; 首尾配对的 zigzag 分片把前后计算量均匀摊给每张卡, 避免末卡成为同步瓶颈。' },
      ],
      sourceRows: [
        { concept: 'dispatch 发送矩阵', code: 'm11_expert_parallel/demo.py:moe_forward_ep', takeaway: '构造 [D, D] 计数表记录源卡到目标卡的发送数量, 按列求和即得各卡实际负载 [11, 18, 11, 24]。' },
        { concept: 'aux loss 的梯度', code: 'm11_expert_parallel/demo.py:train_router_aux', takeaway: '对 $L = E\\sum f_e P_e$ 手写路由梯度: 将实际占比 $f_e$ 视为常数向预测概率 $P_e$ 施压, 抑制热门专家扎堆。' },
        { concept: 'aux-loss-free bias', code: 'm11_expert_parallel/demo.py:balance_bias', takeaway: '根据每步实际负载上下微调选人偏置 bias, 不污染主损失梯度且保持门控加权值不变。' },
        { concept: '环上的一步', code: 'm12_sequence_parallel/demo.py:ring_attention', takeaway: '第 $s$ 轮处理来自 (r - s) % D 的 K/V 分块, 跳过纯未来块后更新本地累加器并向下一邻居传递。' },
        { concept: '增量合并', code: 'm_new / scale', takeaway: '利用 $e^{m - m_{\\text{new}}}$ 将历史分子分母重缩放到最新最大值基准, 与单卡全量 softmax 误差仅 4.4e-16。' },
        { concept: 'zigzag 切分', code: 'm12_sequence_parallel/demo.py:shard_positions', takeaway: '将序列均分为 $2D$ 段后首尾配对分配, 使 4 张卡总点积工作量全部拉齐为 132 对 (连续切分为 [36, 100, 164, 228])。' },
      ],
      snippetTitle: 'EP 与 Ring Attention 的控制流',
      snippet: `# EP: 两次 all-to-all 夹一段本地专家计算
recv = all_to_all(tokens_by_dst)      # dispatch: token 去找自己的专家
out_local = expert(recv)              # 只算自己持有的那几个专家
out = all_to_all(out_local)           # combine: 结果送回原卡

# Ring: D 轮之后每张卡都见过完整序列
for step in range(D):
    src = (rank - step) % D           # 本轮处理谁的 KV 块
    m, l, acc = online_merge(Q_local @ K[src].T, V[src])
    send_to_next(K[src], V[src])      # 环传, 可与计算重叠`,
      source: ['llm_train/m12_sequence_parallel/demo.py:shard_positions'],
      run: 'python -m llm_train.m11_expert_parallel.demo',
    },

    // ---------------------------------------------------------------- 通信原语与主循环
    'train-collectives-loop': {
      title: '通信与 full_loop · 从四个原语到完整主循环',
      subtitle: '各类分布式并行策略底层的跨卡交换都归结为四个集合通信原语。理清它们的字节账本后, 即可将混合精度、分片优化器与断点恢复组装成完整的训练主循环。',
      tldr: '- 多卡训练的所有跨卡同步都建立在四个集合通信原语之上: all-reduce、reduce-scatter、all-gather 与 all-to-all, 核心区别只在于「是否对数据做归约求和」以及「结束时每卡拿整份还是分片」。\n- 采用环形算法 (ring all-reduce) 时, 每张卡每步发送字节数为 $2(N-1)/N \\cdot S$, 无论集群扩到多少张卡都不会超过张量大小的两倍 $2S$。\n- full_loop 将数据切分、micro-batch 累积、AMP 动态缩放跳步、跨分片全局范数裁剪、ZeRO 分片 Adam 与断点续训整合进同一个训练步中验证端到端闭环。',
      question: '面对一个陌生的分布式训练框架, 顺着哪几条通信与状态线索拆解最快能看清它的并行全貌?',
      evolution: {
        title: '从一个中心到一个环',
        subtitle: "根问题: 单卡网卡带宽固定, 而分布式训练每一步都必须跨设备归约同步全量梯度。",
        steps: [
          { name: '汇总到 0 号卡', pain: "(原点) 同步更新要求每张卡在反向传播后都取得全体设备梯度的总和", fix: "所有卡将梯度发往 0 号卡集中求和后再广播回各卡; 0 号卡独自收发 $2(N-1)\\cdot S$ 字节, 流量随卡数线性增长" },
          { name: 'ring all-reduce', pain: "0 号卡成为单点带宽瓶颈, 其余设备的互连带宽却处于闲置状态", fix: "将张量均分为 $N$ 块沿逻辑环接力传递, 各卡收发均衡且单卡发送量上界恒为 $2S$ ($N=4$ 时中心化 0 号卡收发量是环形单卡发送量的 4 倍)" },
          { name: 'reduce-scatter + all-gather', pain: "ZeRO 分片下每张卡只需更新自身负责的参数分片, 完整 all-reduce 却将全量梯度和广播给了每张卡", fix: "将 all-reduce 拆为 reduce-scatter 与 all-gather 两阶段, 在中间直接完成分片参数更新; 单步总通信字节数与 all-reduce 相同" },
          { name: 'full_loop', pain: "梯度按卡分片后, 单张卡无法独立求取全局梯度范数或单独裁定溢出跳步", fix: "通过一次标量 all-reduce 汇总分片平方和求全局范数; 任一卡检出溢出时全体协同跳过本步更新 (实测 40 步内拦截跳过 4 步)" },
        ],
      },
      code: 'llm_train/core/collectives.py · llm_train/m09_collectives/demo.py · llm_train/full_loop/demo.py',
      points: [
        {
          title: '四个集合通信原语的语义与对应并行策略',
          body: '四个原语可以按「是否归约求和」与「输出形态是整份还是分片」直接区分:\n- all-reduce (求和 + 拿整份): 每卡输入一份同形张量, 结束后每卡都持有全体之和; DDP 用它同步全模型梯度, TP 用它归约部分和。\n- reduce-scatter (求和 + 拿分片): 同样对全体输入求和, 但每张卡只拿回自己负责的那 $1/N$ 分片; ZeRO-2/3 用它归约并切分梯度。\n- all-gather (不求和 + 拼整份): 每卡交出 $1/N$ 分片, 结束后每卡拿到拼接完整的张量; ZeRO/FSDP 用它按需收集完整参数。\n- all-to-all (不求和 + 个性化互换): 卡 src 发往卡 dst 的专属数据块落到 out[dst][src]; MoE 专家并行分发 token 与 Ulysses 切换序列/注意力头均依赖它。\n先执行一次 reduce-scatter 再紧接一次 all-gather, 在结果与通信字节数上都等价于一次完整的 all-reduce。',
        },
        {
          title: '环形接力消除了单点带宽瓶颈, 单卡流量恒有 2S 上界',
          key: true,
          body: '不同并行策略算出相同的数学结果, 工程性能差异主要落在跨卡通信账上。设待同步梯度大小为 $S$ 字节、总卡数为 $N$:\n- 中心化汇总: 所有卡发往 0 号卡求和后再广播回来, 0 号卡独自收发 $2(N-1)S$ 字节而其余网口闲置, 通信耗时随卡数线性恶化。\n- Ring all-reduce: 将张量均分为 $N$ 块, $N$ 张卡组成逻辑环。第一阶段走 $N-1$ 步边右移边累加 (即 reduce-scatter, 结束时每卡持有一块的全局和); 第二阶段再走 $N-1$ 步将完整和块传遍全环 (即 all-gather)。每张卡全程均匀发出 $2(N-1)/N \\cdot S$ 字节, 上限封顶在 $2S$。m09 中 $N=4, S=32\\text{ B}$ 时单卡实测发出 48 B, 恰好由 reduce-scatter 的 24 B 与 all-gather 的 24 B 两半拼成。',
        },
        {
          title: 'full_loop: 把分布式、混合精度与容错组件拼进同一步',
          body: 'full_loop 将前面各章独立验证的组件组合成完整的 train_step 闭环, 单步依次串联六件事:\n- 数据切分与累积: 全局 batch 分给 world 张卡, 每卡再拆成 micro 份串行计算, 各份在 fp16 下算出的梯度转成 fp32 后累加。\n- 全局一致性跳步: 只要任意一个 rank 的本地梯度检出 Inf/NaN, 所有 rank 必须协同跳过本步更新并同步下调 loss_scale, 避免副本状态分叉。\n- 跨分片全局裁剪: 梯度经 reduce-scatter 分片后单卡看不到全貌, 需各卡先算本片平方和, 再做一次标量 all-reduce 开方得到全局范数。\n- 分片优化与存盘: 每张卡仅更新本片 fp32 主权重与 Adam 动量, 存盘时各自写入本地分片并附一份全局元数据。\ndemo 实测 40 步内自动拦截 4 次坏步 (3 次真实 fp16 溢出 + 1 次注入 NaN), 第 20 步存盘重启续训后的主权重与连续训练误差为 0.0e+00。',
        },
        {
          title: '拆解工业级分布式框架的七个抓手',
          body: '阅读 Megatron-LM、DeepSpeed 或 TorchTitan 等大型框架时, 不必一头扎进类继承树, 先顺着七个问题定位主干即可:\n- 批次怎么切: 哪些进程组共享模型、分担数据 (DDP 与梯度累积)。\n- 网络层怎么切: 层内矩阵走 TP 列行切分, 还是层间走 PP 流水线。\n- 模型状态怎么放: 参数、梯度与优化器状态哪些复制、哪些分片 (ZeRO/FSDP)。\n- 长序列怎么拆: 注意力走环状传递 K/V (Ring) 还是重排注意力头 (Ulysses)。\n- 数值用什么精度: 前反向矩阵乘用 BF16/FP8, 主权重如何驻留更新。\n- 集合通信落在哪: 每步在何处触发上述四种通信原语。\n- 异常与中断怎么兜底: 溢出跳步如何对齐全局状态, 分片检查点保存了哪些游标与种子。',
        },
      ],
      links: [
        { from: 'all-reduce', to: 'DDP', body: 'DDP 每步通过一次 all-reduce 同步全模型平均梯度, 底层即由 reduce-scatter 与 all-gather 两阶段组成。' },
        { from: 'reduce-scatter + all-gather', to: 'ZeRO / FSDP', body: 'ZeRO/FSDP 在 reduce-scatter 得到分片梯度后直接完成本地参数更新, 下一步计算前再由 all-gather 拼回完整权重。' },
        { from: 'all-to-all', to: 'MoE / Ulysses', body: '无论是 MoE 把 token 发往专家卡, 还是 Ulysses 在按序列切与按头切之间翻转布局, 调用的都是同一个矩阵转置式通信原语 out[dst][src] = in[src][dst]。' },
        { from: 'full_loop', to: 'llm_finetune', body: '训练主循环的分布式同步、精度防护与状态管理到此定型; 下一阶段微调对齐主要替换数据构造与损失函数形态。' },
      ],
      sourceRows: [
        { concept: 'ring all-reduce', code: 'core/collectives.py:ring_all_reduce_sum', takeaway: '按环形拓扑分两阶段各走 $N-1$ 步传递数据块, 断言实测字节数严格等于 $2(N-1)/N \\cdot S$。' },
        { concept: '通信计数器', code: 'core/collectives.py:CommCounter', takeaway: '在每次原语调用时累加单卡平均发送字节数, 为各章并行实验提供统一的通信开销度量。' },
        { concept: '完整一步', code: 'full_loop/demo.py:train_step', takeaway: '串联分片前向、micro-batch 累积、全局溢出跳步、跨分片范数裁剪与分片 Adam 更新; 传入 world=1, micro=1, amp=False 即退化为单卡基准。' },
        { concept: '跨分片求范数', code: 'all_reduce_sum([sum(s ** 2) for s in shards])', takeaway: '各卡先求本地梯度分片的平方和, 经一次标量归约相加后再开方, 还原出与未分片完全一致的全局二范数。' },
        { concept: '分片 checkpoint', code: 'full_loop/demo.py:save_sharded', takeaway: '各 rank 独立落盘本地主权重与动量分片, 配合一份全局元数据记录步数、缩放因子与数据游标, 实现逐位一致的断点续训。' },
      ],
      snippetTitle: 'full_loop 的一步',
      snippet: `flat_lp = all_gather([rk["master"].astype(float16) for rk in ranks])  # ① 算副本, 通信的是 fp16
for r in range(world):                                 # ② 每 rank 累积自己的 micro-batch
    for xb, yb in micro_batches(r):
        acc[r] += grads(xb, yb, loss_scale=scaler.scale) / micro

if has_overflow(local):                                # ③ 任一 rank 坏了, 所有 rank 一起跳过
    scaler.update(True); return

shards = [g / (scale * world) for g in reduce_scatter_sum(local)]   # ④ 每 rank 只留自己那片
sq = all_reduce_sum([sum(s ** 2) for s in shards])     # ⑤ 全局范数要跨分片求和
clip = min(1.0, MAX_NORM / sqrt(sq))
for rk, g in zip(ranks, shards):                       # ⑥ 分片 Adam, 只碰自己的 master/m/v
    adam_update(rk["master"], g * clip, rk["m"], rk["v"], opt_steps, lr)`,
      source: ['llm_train/core/collectives.py:ring_all_reduce_sum', 'llm_train/full_loop/demo.py:train_step'],
      run: 'python -m llm_train.full_loop.demo',
    },

    // ---------------------------------------------------------------- 新章节
    'train-pipeline-schedules': {
      title: '1F1B 与交错调度 · 气泡和在途激活的账',
      subtitle: '流水线并行同时面临设备空等 (气泡) 与未释放的在途激活两笔开销。对比 GPipe、1F1B 与交错 1F1B 的时间表, 看清不同排程如何压低显存峰值与空闲比例。',
      tldr: '流水线并行将模型按层切成 $\\mathrm{PP}$ 个 stage, 核心开销有两项:\n- 气泡占比: 设备等待上下游依赖的空闲时间比例, GPipe 与标准 1F1B 均为 $(\\mathrm{PP}-1)/(M+\\mathrm{PP}-1)$ (PP=4, M=8 时为 27.3%)。\n- 在途激活峰值: 已完成前向但尚未执行反向的 micro-batch 必须驻留显存。1F1B 通过尽早穿插反向将各卡峰值从 [8,8,8,8] 压到 [4,3,2,1], 腾出显存开大 $M$ 来间接压气泡; 交错 1F1B 则通过每卡托管 $v$ 个不相邻小段直接把气泡压低到 $(\\mathrm{PP}-1)/(vM+\\mathrm{PP}-1)$。',
      question: '标准 1F1B 的理论气泡公式和 GPipe 完全一样, 为什么实际训练里大家普遍用 1F1B 替代 GPipe?',
      evolution: {
        title: '气泡和在途激活的拉锯',
        subtitle: "根问题: 模型按层切分为多个 stage 后, 各设备前向需等待上游激活、反向需等待下游梯度, 空闲气泡与在途激活彼此制约。",
        steps: [
          { name: 'GPipe', year: 2018, pain: "(原点) 模型按连续层切分后, 整批单次流转时任意时刻仅有一个 stage 处于计算状态", fix: "将全局 batch 拆成 $M$ 个 micro-batch, 先跑完全部前向再统一执行全部反向; 每个 stage 峰值需同时驻留 $M$ 份在途激活" },
          { name: '1F1B', pain: "压缩流水线气泡需要增大 $M$, 但 GPipe 的在途激活显存随 $M$ 同步线性增长", fix: "热身若干个前向后进入一前一反稳态交替以尽早释放激活, stage $s$ 在途峰值上界固定为 $\\mathrm{PP}-s$ 份: 从 [8,8,8,8] 降至 [4,3,2,1]" },
          { name: '1F1B + 大 M', pain: "仅调整前反向穿插顺序并未改变理论气泡公式: $\\mathrm{PP}=4, M=8$ 时空闲气泡占比仍为 27.3%", fix: "利用在途激活与 $M$ 解耦省下的显存空间大幅调大 $M$: $\\mathrm{PP}=8, M=64$ 时气泡占比压低至 9.9%" },
          { name: '交错 1F1B', year: 2021, pain: "当 $M$ 受显存或全局批大小上限制约无法继续增大时, 流水线首尾仍残留可观气泡", fix: "每张卡托管 $v$ 个不相邻的虚拟小 stage, 使气泡公式分母中的 $M$ 放大为 $vM$ ($v=2$ 时气泡从 27.3% 降至 15.8%); 代价是跨卡传输次数增加" },
        ],
      },
      code: 'llm_train/m04_pipeline_parallel/demo.py',
      points: [
        {
          title: 'GPipe 与 1F1B 计算完全相同, 只调整了前反向穿插顺序',
          body: '设模型按连续层切成 $\\mathrm{PP}$ 个 stage, 全局 batch 拆成 $M$ 个 micro-batch, 每个 micro-batch 在每个 stage 上都要跑一次前向 (F, 耗时 1 格) 和一次反向 (B, 耗时约 2 格):\n- GPipe: 每张卡先把全部 $M$ 个 micro-batch 的前向一口气算完, 再统一执行全部 $M$ 个反向。\n- 1F1B (One-Forward-One-Backward): 第 $s$ 张卡开头只做 $\\mathrm{PP}-1-s$ 个前向热身, 一旦首个 micro-batch 的反向梯度从下游传回, 立刻进入「做一个 F 紧跟一个 B」的稳态交替, 及时把算完反向的激活释放掉。',
        },
        {
          title: '1F1B 直接省的是在途激活显存, 再借更大的 M 间接压低气泡',
          key: true,
          body: '固定相同的 $\\mathrm{PP}=4, M=8$ 时, GPipe 与 1F1B 的空闲气泡占比完全相等, 都是 27.3%; 真正的差别落在每张卡同时驻留的在途激活份数上:\n- GPipe: 前向全跑完才开始释放, 四张卡峰值全为 [8, 8, 8, 8], 随 $M$ 线性膨胀。\n- 1F1B: 第 $s$ 张卡在途份数上界锁死在 $\\mathrm{PP}-s$ (即 [4, 3, 2, 1]), 与 $M$ 开多大脱钩。\n正因为激活显存不再随 $M$ 上涨, 训练时才能放心把 $M$ 开大 (例如 $\\mathrm{PP}=8, M=64$), 让气泡 $(\\mathrm{PP}-1)/(M+\\mathrm{PP}-1)$ 顺势压低到 9.9%。',
        },
        {
          title: '交错 1F1B: 每卡托管多个虚拟小段, 把气泡分母放大 v 倍',
          body: '当全局批大小受限、无法继续增大 $M$ 时, 交错调度 (interleaved 1F1B) 将模型切得更细: 拆成 $\\mathrm{PP}\\cdot v$ 个虚拟 stage, 让卡 $s$ 负责不相邻的第 $s, s+\\mathrm{PP}, \\dots$ 段。例如 $\\mathrm{PP}=4, v=2$ 时共 8 段, 卡 0 拿第 0 段和第 4 段, 数据沿 4 张卡绕两圈流完前向。\n由于单段计算耗时缩为原来的 $1/v$, 首尾流水线填充与排空的等待时间也同步缩减为 $1/v$, 气泡公式变为 $(\\mathrm{PP}-1)/(vM+\\mathrm{PP}-1)$。m04 事件模拟实测 PP=4, M=8, v=2 时气泡从 27.3% 直接降到 15.8%。这个模拟把跨卡 P2P 传输时间记为 0, 真实收益还要扣掉交错多出来的传输。',
        },
        {
          title: '交错调度的工程代价: 跨卡通信翻 v 倍且在途激活增多',
          body: '交错 1F1B 压缩气泡并非免费 (PP=4, M=8, v=2 实测):\n- 跨卡点对点传输增加 $v$ 倍: 单个 micro-batch 前反向跨越卡边界的次数从 $2(\\mathrm{PP}-1)$ 增至 $2(\\mathrm{PP}\\cdot v - 1)$。\n- 在途激活高于标准 1F1B: 交错排程需要更长的热身步数, 卡 0 同时持有的半段激活峰值为 11 份, 折合 $11 \\times 0.5 = 5.5$ 个完整 stage (高于标准 1F1B 的 4 个, 仍远低于 GPipe 的 8 个)。\n- 整除约束: 要求 $M$ 必须是 $\\mathrm{PP}$ 的整数倍。',
        },
      ],
      links: [
        { from: 'train-model-parallel', to: 'device_order', body: '上一章介绍了按层切分的流水线概念, 本章由 device_order 排出各卡具体的 F/B 执行序列并交由离散事件模拟器核算开销。' },
        { from: 'warmup 个 F', to: '在途激活上限', body: '热身阶段连做 warmup 个前向后再接 1 个前向才迎来首个反向释放, 因此 warmup + 1 直接决定了该卡在途激活的峰值份数。' },
        { from: '在途激活', to: '激活重算', body: '流水线并行常与激活重算叠加使用: 每个在途 micro-batch 只保留 stage 入口张量, 进一步把首卡显存压力打下来。' },
      ],
      sourceRows: [
        { concept: '每卡的执行顺序', code: 'm04_pipeline_parallel/demo.py:device_order', takeaway: '生成单卡的操作序列: GPipe 为全 F 后全 B, 1F1B 为 warmup 个 F 后严格一前一反交替。' },
        { concept: '交错的 warmup', code: '(PP - 1 - s) * 2 + (v - 1) * PP', takeaway: 'Megatron-LM 交错调度的热身前向数公式, 比标准 1F1B 更长, 解释了交错模式下在途激活份数的增加。' },
        { concept: '虚拟 stage', code: 'k = c * PP + s', takeaway: '卡 $s$ 上的第 $c$ 个虚拟段对应全局流水线的第 $k$ 段, 满足 k % PP == s 轮转映射。' },
        { concept: '事件模拟', code: 'm04_pipeline_parallel/demo.py:simulate', takeaway: '按设备空闲与上游完成时刻推进时间线, 最终由 1 - 有效计算时长 / 总墙钟时长 算出真实空闲气泡占比。' },
      ],
      snippetTitle: '三种调度只差一个 warmup',
      snippet: `def device_order(kind, s, PP, M, v=1):
    fwd = [F(m, chunk) ...]              # 每 PP 个 micro-batch 一组, 组内先 chunk 0 再 chunk 1
    bwd = [B(m, chunk) ...]              # chunk 倒序
    if kind == "gpipe":
        return fwd + bwd                 # 在途 = M
    warmup = PP-1-s if v == 1 else (PP-1-s)*2 + (v-1)*PP
    order = fwd[:warmup]
    while bwd 没做完:                     # 稳态: 一个 F 一个 B
        order += [next(fwd), next(bwd)]
    return order

bubble = (PP - 1) / (v * M + PP - 1)`,
      source: ['llm_train/m04_pipeline_parallel/demo.py:device_order', 'llm_train/m04_pipeline_parallel/demo.py:simulate'],
      run: 'python -m llm_train.m04_pipeline_parallel.demo',
    },

    'train-lr-schedule': {
      title: 'WSD 学习率调度 · 不用提前承诺总步数',
      subtitle: '余弦退火需要事先定死总训练步数, 中途追加语料十分被动。WSD (预热-恒定-退火) 将主干保持在恒定学习率, 允许在任意节点分叉退火产出模型。',
      tldr: '学习率调度曲线决定了训练能不能灵活续跑:\n- 余弦调度 (cosine): 每一步学习率都依赖进度比 step / total, 总步数 total 一改整条历史曲线都会变形 (m10 中 total 从 100 改 200 时前 90 步最大相差 5.96e-4)。\n- WSD 调度 (Warmup-Stable-Decay): 预热后一直保持恒定峰值学习率推进主干, 不含 total 参数; 需要评估或发版时, 从任一稳定段 checkpoint 分叉跑一小段快速退火 (如末 10% 步数) 即可收敛。',
      question: '模型训完预定步数发现 loss 还在稳步下降, 想追加一倍语料继续训: 余弦调度和 WSD 各自会遇到什么局面?',
      evolution: {
        title: '从一个常数到三段式',
        subtitle: "根问题: 学习率偏大时前期下降快却难以落入损失谷底, 偏小时收敛平稳但推进过慢, 单一常数无法兼顾训练全程。",
        steps: [
          { name: '常数 lr', pain: "(原点) 学习率 lr 偏大时下降迅速却在谷底附近震荡, 偏小时更新平稳但收敛过慢", fix: "训练全程固定单一标量学习率, 只能在前期推进速度与末期收敛精度之间被动折中" },
          { name: 'warmup', pain: "训练起步即施加大学习率 lr 时, Adam 二阶矩统计尚未稳定, 极易引发参数剧烈跳变甚至发散", fix: "前几百步将 lr 从极小值线性拉升至峰值以平稳度过冷启动; 但此后长期维持峰值仍会在损失谷底附近持续震荡" },
          { name: 'cosine 衰减', year: 2016, pain: "预热结束后 lr 始终停留在峰值, 步长过大而无法落入精细损失谷底", fix: "预热后 lr 按进度比 step / total 沿余弦曲线平滑衰减至 0; 代价是总步数 total 耦合进每一步计算, 调整预算会改变整条曲线" },
          { name: 'WSD', year: 2024, pain: "训满预定步数后若需追加语料续训: 要么重新升温 (re-warmup) 导致 loss 剧烈反弹, 要么只能按新 total 从头重训", fix: "稳定段维持恒定峰值且不依赖 total; 任意稳定段 checkpoint 只需分叉接一段短退火 (m10 中占末 10% 步数) 即可收敛产出成品" },
        ],
      },
      code: 'llm_train/m10_training_stability/demo.py',
      points: [
        {
          title: '初期线性预热 (warmup) 用于平稳度过二阶矩冷启动',
          body: '大模型训练开头的几百到几千步通常会把学习率从接近 0 线性拉升到峰值。这主要是为了配合 Adam 优化器: 训练刚起步时二阶矩滑动平均 $v$ 只见过极少数样本, 方差估计很不稳定, 自适应步长容易剧烈跳变; 先用极小学习率热身能让 $v$ 积累可靠统计量后再放开步伐。',
        },
        {
          title: '余弦退火把总步数耦合进了中间每一步',
          key: true,
          body: '余弦调度的进度比例写作 $(\\text{step} - \\text{warmup}) / (\\text{total} - \\text{warmup})$, 分母直接绑定了总训练步数 total。一旦预算调整改变 total, warmup 结束后每一步本该使用的学习率就全变了。m10 对比了将总步数从 100 扩到 200 时前 90 步的学习率差异:\n- 余弦调度: 最大偏移达 5.96e-4 (超过峰值学习率 1e-3 的一半)。\n- WSD 调度: 稳定段偏差为 0.00e+00。\n因此在余弦调度下若训到终点想追加数据, 要么从接近 0 的学习率重新拉高 (引发 loss 剧烈反弹), 要么只能按新预算从头重训。',
        },
        {
          title: 'WSD 支持主干持续续训与按需分叉发版',
          body: 'MiniCPM、DeepSeek-V3 与 Kimi 等模型广泛采用 WSD 类三段式调度, 把持续预训练与阶段性交付解耦:\n- 持续主干: 预热结束后长期维持峰值学习率稳定训练, 随时可以无缝接上新一批语料。\n- 按需分叉退火: 每当需要对比不同 token 预算的 scaling 点或导出正式发布版时, 直接从稳定段最新 checkpoint 拉出一条分支, 用约 10% 的步数把学习率快速退火至接近 0 即可落入损失谷底 (工业实践常配合 1 - sqrt 或指数衰减曲线)。',
        },
      ],
      links: [
        { from: 'train-precision-stability', to: 'warmup', body: '预热防护训练初期 Adam 二阶矩尚未估准的风险, 梯度裁剪与 NaN 拦截防护训练中途的异常尖峰, 三者共同构成稳定性防线。' },
        { from: 'decay_start', to: 'checkpoint', body: '退火起点前夕保存的稳定段 checkpoint 是最宝贵的公共基座: 后续追加预训练与各版本退火分支都从它接续。' },
        { from: 'WSD 退火', to: 'Muon / AdamW', body: '学习率调度只输出当前步的标量步长, 与底层选用 AdamW 还是 Muon 正交, 实际训练如 Kimi K2 常将 Muon 与 WSD 搭配使用。' },
      ],
      sourceRows: [
        { concept: 'warmup + cosine', code: 'm10_training_stability/demo.py:warmup_cosine_lr', takeaway: '通过 (step - warmup) / (total - warmup) 计算余弦相位, warmup 结束后的每一步都受总步数 total 影响。' },
        { concept: 'WSD', code: 'm10_training_stability/demo.py:wsd_lr', takeaway: '稳定段直接返回 base_lr, 完全不读取 total_steps, 确保主干轨迹与总预算无关。' },
        { concept: '退火起点', code: 'decay_start = total_steps - int(total_steps * decay_frac)', takeaway: '仅在末尾 decay_frac (默认 10%) 步进入线性退火段, 将总步数的影响收束在收尾阶段。' },
        { concept: '两种调度的实测差', code: 'total=100 → 200, 前 90 步 lr 最大变化', takeaway: '扩容总步数时前 90 步学习率最大漂移: 余弦调度为 5.96e-04, WSD 严格保持 0.00e+00。' },
      ],
      snippetTitle: 'WSD 三段式',
      snippet: `def wsd_lr(step, total_steps, base_lr, warmup_steps, decay_frac=0.1, min_ratio=0.0):
    decay_start = total_steps - int(total_steps * decay_frac)
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    if step < decay_start:
        return base_lr                              # 稳定段: 与 total_steps 无关
    progress = (step - decay_start) / max(1, total_steps - decay_start)
    return base_lr * (1 - (1 - min_ratio) * progress)`,
      source: ['llm_train/m10_training_stability/demo.py:wsd_lr', 'llm_train/m10_training_stability/demo.py:warmup_cosine_lr'],
      run: 'python -m llm_train.m10_training_stability.demo',
    },

    'train-low-precision': {
      title: 'FP8 与 FP4 微缩放 · scale 的粒度决定能砍到几位',
      subtitle: '4-bit 浮点单元素的非零动态范围只有 12 倍, 必须把缩放因子细化到 16~32 个元素一小块; MXFP4 与 NVFP4 的精度差距则直接取决于缩放因子自身的位宽格式。',
      tldr: '- FP4 (E2M1) 单个元素的非零正数仅覆盖 0.5 到 6, 必须采用微缩放 (microscaling): 每 16~32 个相邻元素共享一个局部缩放系数 scale, 把离群值的影响锁死在极小块内。\n- MXFP4 与 NVFP4 都使用 E2M1 存元素, 核心差异在于 scale 本身的编码精度: MXFP4 用无尾数的 E8M0 (只能取 2 的整数次幂), NVFP4 改用带 3 位尾数的 E4M3 搭配整张量二级缩放, 显著减少了块最大值截断误差。',
      question: '同样采用 4-bit E2M1 元素加分块缩放, 为什么 NVFP4 的量化误差明显低于 MXFP4? 什么分布下均匀网格的 INT4 反而比 FP4 更准?',
      evolution: {
        title: 'scale 越切越细, 位宽越砍越少',
        subtitle: "根问题: 浮点位宽越低, 单张量化网格所能覆盖的动态范围越窄, 张量内幅值悬殊的元素无法同时落入有效区间。",
        steps: [
          { name: 'FP8 per-tensor', year: 2022, pain: "(原点) FP8 可表示的动态范围有限, 量化前必须先将张量数值缩放映射至格式量程之内", fix: "整张量按全局最大绝对值 amax 共用一个缩放因子 scale; 单个高幅值离群值 (outlier) 便会迫使大量正常元素下溢趋近于 0" },
          { name: 'block scale', year: 2024, pain: "当 outlier 幅值达到正常元素的 1e5 倍时, 其余 token 的输出相对误差从 0.026 恶化至 0.126", fix: "改为每 128 个连续元素独立计算一个 scale, 将 outlier 的波及范围限制在同块之内: 其余 token 相对误差保持在 0.026" },
          { name: 'MXFP4', year: 2023, pain: "位宽进一步缩减至 4 bit 后, 单个非零元素的动态范围仅剩 12 倍", fix: "每 32 个元素共享一个纯指数 E8M0 格式的块 scale; 因 scale 只能取 2 的整数次幂, 42% 的 block 最大值遭到饱和截断" },
          { name: 'NVFP4', pain: "块 scale 受限于 2 的整数次幂, 无法将块内最大值 amax 精确对齐至网格上限 6", fix: "将块 scale 升级为带 3 位尾数的 E4M3 格式并将 block 缩至 16 个元素: 相对量化误差从 0.113 降至 0.095" },
        ],
      },
      code: 'llm_train/m13_fp8_training/demo.py · llm_train/m15_fp4_microscaling/demo.py · llm_train/core/numerics.py',
      points: [
        {
          title: '分块越小, 离群值连累的邻居越少',
          key: true,
          body: '分块量化按组内最大绝对值 amax 计算缩放系数 scale = amax / max_val。当张量出现个别幅值极高的离群值 (outlier) 时, 共用同一个 scale 的同块正常元素会被迫除以一个大数而挤向 0 附近。缩小分块尺寸能直接限制离群值的波及半径; 尤其降到 FP4 (E2M1) 后, 非零格点只剩 $\\pm\\{0.5, 1, 1.5, 2, 3, 4, 6\\}$, 动态范围仅 12 倍, 必须把块缩至 16~32 个元素才能正常表示张量。',
        },
        {
          title: 'MXFP4 与 NVFP4 的精度差距全部来自 scale 自身的尾数位',
          body: '两种微缩放标准的元素都是 E2M1, 区别在于如何编码每块的缩放因子:\n- MXFP4 (每 32 个元素一块): 块 scale 采用 8 位纯指数格式 E8M0 (0 位尾数), 只能取 $2^k$。因此 amax / scale 落在 $[4, 8)$ 区间内全看运气, 超过 6 的部分直接被饱和截断回 6 (m15 中多达 42% 的块最大值遭截断)。\n- NVFP4 (每 16 个元素一块): 块 scale 改用带 3 位尾数的 E4M3 (辅以整张量 1 个 FP32 二级缩放补齐动态范围), 能把块内 amax 紧贴映射到网格顶端 6。\nm15 在高斯分布权重上测量相对量化误差 $\\|Q(x)-x\\|/\\|x\\|$: 默认配置下 MXFP4 为 0.113, NVFP4 降至 0.095; 若将两者分块大小统一固定为 16、仅对比 E8M0 与 E4M3 缩放格式, 误差依然是 0.1138 对 0.0947, 证实收益主要来自 scale 的尾数精度。',
        },
        {
          title: '均匀网格 INT4 与对数网格 FP4 各擅胜场',
          body: '同为 4-bit 分块量化, INT4 使用 $-7 \\dots 7$ 的等间距格点, 而 FP4 (E2M1) 格点越往外越稀疏。谁误差更低取决于张量分布形态:\n- 集中型高斯分布 (如常规权重): 数值密集分布在中小幅值区, 等间距的 INT4 相对误差为 0.096, 优于 MXFP4 的 0.113。\n- 重尾分布 (如含尖峰的激活或梯度): 偶发大值拉宽量程后, 近似对数分布的 MXFP4 误差为 0.142, 优于 INT4 的 0.147。(计入 scale 开销后, MXFP4 平均每元素占 4.25 bit, INT4 占 4.50 bit。)',
        },
      ],
      links: [
        { from: 'train-precision-stability', to: 'FP8 block scaling', body: 'BF16、FP8 与 FP4 均复用同一个格点模拟函数 fake_quant_float, 切换数据格式只需替换尾数位宽、最大值与最小正规指数三项参数。' },
        { from: 'quant_blockwise', to: 'mxfp4 / nvfp4', body: 'quant_blockwise 实现理想浮点 scale 下的分块量化, mxfp4 与 nvfp4 在此基础上进一步对 scale 本身做 E8M0 或 E4M3 量化。' },
        { from: 'FP4 权重', to: 'llm_infer 量化', body: '本节评估单次张量量化误差; 全流程 FP4 预训练往往还需搭配随机舍入与 Hadamard 正交旋转打散离群值。' },
      ],
      sourceRows: [
        { concept: 'block scaling', code: 'core/numerics.py:quant_blockwise', takeaway: '将张量重排为每行 block 个元素, 按行求 amax / max_val 完成缩放、舍入与反量化还原。' },
        { concept: 'MXFP4 的 scale', code: 'scale = 2.0 ** (np.floor(np.log2(amax)) - 2)', takeaway: '将块缩放限制为 2 的整数次幂 (E8M0), 使 amax / scale 落入 [4, 8), 超过 E2M1 上限 6 的部分发生截断。' },
        { concept: 'NVFP4 的 scale', code: 'm15_fp4_microscaling/demo.py:nvfp4', takeaway: '先借整张量 FP32 系数将各块 amax / 6 纳入 E4M3 量程 (上限 448), 量化为带 3 位尾数的 E4M3 scale。' },
        { concept: 'FP8 消融', code: 'm13_fp8_training/demo.py:train', takeaway: '逐项对比训练 200 步后的验证损失: 干净数据上 per-tensor 与 block 打平 (约 4.6e-4), 移除 fp32 主权重则恶化 584 倍。' },
      ],
      snippetTitle: 'MXFP4 与 NVFP4 只差 scale 一行',
      snippet: `def mxfp4(x, block=32):
    b = x.reshape(-1, block)
    amax = abs(b).max(axis=1, keepdims=True)
    scale = 2.0 ** (floor(log2(amax)) - 2)            # E8M0: 只能是 2 的幂
    return quant(b / scale, "e2m1") * scale

def nvfp4(x, block=16):
    b = x.reshape(-1, block)
    amax = abs(b).max(axis=1, keepdims=True)
    tensor_scale = abs(x).max() / (448 * 6)           # 整张量 1 个 FP32
    scale = quant(amax / 6 / tensor_scale, "e4m3") * tensor_scale
    return quant(b / scale, "e2m1") * scale`,
      source: ['llm_train/m15_fp4_microscaling/demo.py:mxfp4', 'llm_train/m15_fp4_microscaling/demo.py:nvfp4', 'llm_train/core/numerics.py:quant_blockwise'],
      run: 'python -m llm_train.m15_fp4_microscaling.demo',
    },

    'train-muon': {
      title: 'Muon 优化器 · 把更新矩阵正交化',
      subtitle: 'Adam 逐个参数缩放步长, 遇到斜向耦合的病态曲率依然受限。Muon 用 5 步矩阵乘迭代将二维动量矩阵近似正交化, 让各个奇异方向以均等幅度推进。',
      tldr: '- Adam 按坐标轴独立维护二阶矩来调整步长; Muon 专门针对二维权重矩阵, 将动量矩阵的所有奇异值推平到 1 附近 (正交化), 使小奇异值方向不再被大奇异值方向淹没。\n- 正交化无需运行昂贵的 SVD, 改用 5 步纯矩阵乘法的 Newton–Schulz 多项式迭代即可在 GPU 上高效近似。\n- 当损失曲面的病态方向不与坐标轴对齐时 Muon 优势明显 (m14 中损失比 Adam 低 9.2 倍); 若病态恰好沿单个坐标轴分布, 逐元素缩放的 Adam 反而低 1.9 倍。',
      question: 'Adam 已经给每个参数单独自适应步长了, 为什么遇到旋转后的病态输入仍会变慢? 为什么嵌入层与一维参数不能直接套用 Muon?',
      evolution: {
        title: '从逐元素缩放到逐奇异方向缩放',
        subtitle: "根问题: 损失曲面在不同方向上的曲率相差数个数量级, 单一全局学习率无法兼顾所有方向。",
        steps: [
          { name: 'SGD', pain: "(原点) 损失曲面在各方向上的曲率相差数个数量级, 单一学习率 lr 难以兼顾全局", fix: "所有参数方向共用同一学习率更新: 高曲率方向易发散震荡, 低曲率方向则收敛极慢" },
          { name: 'Adam', year: 2014, pain: "高曲率方向需要小步长抑制震荡, 低曲率方向需要大步长加速推进", fix: "按每个参数维护梯度二阶矩以自适应缩放步长; 但仅能校正与坐标轴对齐的病态曲率" },
          { name: 'Muon', year: 2024, pain: "当病态曲率方向是多个参数坐标的斜向线性组合时, 逐元素对角缩放无法解耦", fix: "用 5 步纯矩阵乘法的 Newton–Schulz 迭代将动量矩阵奇异值推平至 1 附近, 使各奇异方向等步长更新; 旋转病态问题上损失低 9.2 倍" },
          { name: 'MuonClip', year: 2025, pain: "满秩正交更新在大规模训练中易推高注意力 logits, 导致 softmax 饱和为独热分布", fix: "当最大注意力分数超过阈值 $\\tau$ 时, 将 $W_q$ 与 $W_k$ 各乘 $\\sqrt{\\tau/S_{\\max}}$, 把最大 logit 从 752.1 压回 100.0" },
        ],
      },
      code: 'llm_train/m14_muon_optimizer/demo.py',
      points: [
        {
          title: '把动量矩阵的奇异值拉平, 消除弱方向饥饿',
          body: '对任意二维梯度或动量矩阵做奇异值分解 $G = U\\Sigma V^\\top$, 可以将其看作在一组正交方向 ($U, V$ 的列向量) 上按奇异值 $\\Sigma$ 各自缩放。神经网络训练中梯度矩阵的奇异值往往相差数个数量级, 直接更新时少数主方向占据绝大部分步长, 其余微弱奇异方向几乎停滞。\nMuon 将对角阵 $\\Sigma$ 替换为单位阵, 取正交极因子 $UV^\\top$ 作为更新方向 (等价于谱范数下的最陡下降), 让所有奇异方向在单步更新中获得同等大小的推力, 且优化器只需存一份一阶动量缓冲区, 状态显存比存 m、v 两份的 Adam 更省。',
        },
        {
          title: '5 步 Newton–Schulz 多项式迭代替代精确 SVD',
          body: '直接调库求 SVD 在 GPU 上并行度差、耗时长。Muon 先将动量矩阵除以自身的 Frobenius 范数确保全部奇异值落在 $[0, 1]$ 内, 随后重复 5 步纯矩阵乘迭代 $X \\leftarrow aX + (bA + cA^2)X$ (其中 $A = XX^\\top$, 系数取 $(3.4445, -4.7750, 2.0315)$)。\n每次迭代保持左右奇异向量不变, 仅把每个奇异值 $\\sigma$ 映射为奇数次多项式 $a\\sigma + b\\sigma^3 + c\\sigma^5$: 接近 0 的小奇异值每步被放大约 3.44 倍快速抬升, 较大的奇异值则被压在 $[0.7, 1.2]$ 区间内来回抖动 (这组系数只求快速推进该区间, 不追求精确收敛到 1; m14 中最大奇异值始终约 1.16)。m14 对条件数高达 $10^4$ 的 $16\\times 32$ 矩阵迭代 5 步后, 最小奇异值从 1e-4 抬升至 0.041, 与精确 $UV^\\top$ 的余弦相似度达 0.880。',
        },
        {
          title: '轴对齐病态归 Adam, 旋转耦合病态归 Muon',
          key: true,
          body: 'Adam 与 Muon 的长板正好互补:\n- Adam 的逐元素除法等价于对角预处理, 只能校正严格沿着单个参数坐标轴的曲率悬殊; 一旦高曲率与低曲率方向混成多个坐标的斜向线性组合, 逐元素缩放便无法解耦。\n- Muon 的正交化具备旋转不变性, 直接在奇异向量张成的子空间上拉平步长。\nm14 在条件数为 $10^4$ 的 32 维矩阵回归上各扫 5 档最优学习率训练 150 步: 加入随机正交旋转 (非轴对齐) 时 Muon 最终损失为 4.93e-04, 比 Adam (4.52e-03) 低 9.2 倍; 不加旋转 (严格轴对齐) 时 Adam 为 5.54e-04, 反而比 Muon (1.08e-03) 低 1.9 倍。\n两边 weight decay 都为 0, 1-D 偏置都用 Adam, 只比缩放方式这一处差别。这是线性回归玩具问题, 「真实网络上 Muon 省约一半 FLOPs」是 Moonlight / Kimi 的报告结论, 本 demo 证明不了。',
        },
        {
          title: '大规模训练的分层边界与 MuonClip 防护',
          body: '将 Muon 用于大规模 Transformer 时有两项关键工程配套:\n- 参数适用范围: Muon 只更新隐藏层内部的二维线性投影矩阵; 一维的归一化增益/偏置仍交由 AdamW 更新; 词嵌入层与语言模型输出头虽是矩阵, README 指出对它们用 Muon 反而更差, 同样交给 AdamW。\n- QK-clip 抑制注意力分数爆炸: 满秩正交更新在超大规模训练中容易让注意力 logit 越涨越大, softmax 饱和成独热分布、梯度消失。m14 只演示裁剪本身, 不演示 logit 怎么涨上去: 手工构造偏大的 $W_Q, W_K$ 时最大 logit 达 752.1, 最大概率均值 1.000。Kimi K2 提出的 MuonClip 在监测到最大 logit $S_{\\max} > \\tau$ 时, 将 $W_Q$ 和 $W_K$ 各乘 $\\sqrt{\\tau / S_{\\max}}$, 把内积峰值精确压回阈值 $\\tau=100.0$。',
        },
        {
          title: '2026 年的现状: 进了最大的开源模型, 并且按 head 切开',
          body: 'Muon 已经是几个最大的开源模型的默认优化器:\n- DeepSeek-V4 (arXiv 2606.19348) 把 Muon 列为三项升级之一, 理由写成收敛更快、训练更稳, 没有给具体数字。两档模型是 1.6T 参数 / 49B 激活和 284B / 13B 激活, 预训练超过 32T token。\n- Kimi K3 (arXiv 2607.24653, 2.8T 参数 / 104B 激活) 用的是 per-head Muon: 注意力的 $W_Q$、$W_K$、$W_V$ 不再整块做 Newton-Schulz, 而是把动量矩阵沿 head 维切块、每个 head 单独正交化。整块正交化把所有 head 当成一个耦合的块, 梯度或动量尺度大的 head 会主导共享的更新方向, 尺度小的 head 得到的归一化不足; 按 head 切开让各 head 的更新幅度一致, 大规模下训练更稳, 顺带省一点开销 (瘦长的 per-head 块做 Newton-Schulz 比整个投影矩阵便宜)。论文没有给量化收益。\nKimi K3 还给了分布式 Muon 那笔工程成本的一种解法: 分布式优化器按数据并行的 rank 切参数, 而 Newton-Schulz 要完整矩阵, 于是每个 rank 只用点对点通信向持有者取自己负责的那一份, 不做整参数的 all-gather。',
        },
        {
          title: '两条还在争的结论: 正交化之后要不要再归一化, 最优优化器是否固定',
          body: 'Muon 的配方和它的适用范围都还在动:\n- Muon+ (arXiv 2602.21545) 指出一个反直觉的现象: 正交化把奇异值拉平到 1, 但实际的 polar 迭代会放大更新的列范数与行范数不均衡, 而这个不均衡会削弱单步下降的保证。修法是在正交化之后插一个归一化步, 不加任何优化器状态; 60M 到 7B 的 GPT 与 LLaMA 预训练上训练和验证困惑度都一致更好, 摘要没给倍数。\n- 大规模受控对比 (arXiv 2607.20548) 用 update-RMS 对齐保证各优化器之间学习率可比, 在多十亿参数、万亿 token 上测出 SOAP 与 Muon 一致优于 AdamW, batch size 到 1 亿 token 时前两者仍稳而 AdamW 退化。正文另给一条配方: 训练早期加一段 batch size ramp-up (从约 2M token 起每若干步翻倍, 学习率 warmup 绑在同一条曲线上, 大致维持 $\\eta \\propto \\sqrt{B}$), Muon 因此在最终训练 loss 和下游评测上都明显超过 AdamW。\n这篇对比不是 Muon 的独唱: SOAP 是并列主角, 一组固定 batch 的受控实验里 SOAP 的 loss 还略低于 Muon; ramp-up 那一组只用在大 batch 实验上, 作者也写明没有按模型规模调过这条曲线, 同样手法对 AdamW 无效。另一篇 (arXiv 2610.08975) 给出反向证据: 没有哪条 Muon 的缩放规则在各种训练设置下都成立, 最优优化器本身会随 batch size 改变。m14 实现的是最初那一版 (整块 Newton-Schulz 加 QK-clip), 以上几条都没有实现。',
        },
      ],
      links: [
        { from: 'llm_basic optim.py', to: 'Muon', body: 'Adam 为每个参数维护一阶矩 m 与二阶矩 v 两份状态; Muon 只保留单份动量缓冲, 靠矩阵正交化替代逐坐标方差归一化。' },
        { from: 'newton_schulz', to: 'GPU 友好', body: 'Newton–Schulz 全程由稠密矩阵乘法组成, 能直接跑满 GPU Tensor Core, 5 步迭代相对整网前反向的额外耗时通常不足 1%。' },
        { from: '0.2·√max(n,m)', to: 'AdamW 学习率', body: 'Moonlight 经验缩放系数 $0.2\\sqrt{\\max(n,m)}$ 将正交矩阵元素的均方根对齐到 AdamW 更新量的典型幅度 (~0.2), 方便直接复用成熟的 AdamW 学习率与权重衰减超参。' },
      ],
      sourceRows: [
        { concept: 'Newton–Schulz', code: 'm14_muon_optimizer/demo.py:newton_schulz', takeaway: '先除以 Frobenius 范数保证奇异值不超过 1, 再做 5 步五次奇多项式矩阵迭代将奇异值推向 1 附近。' },
        { concept: '一步更新', code: 'm14_muon_optimizer/demo.py:Muon', takeaway: '维护系数 0.95 的动量 buf, 对 Nesterov 组合 G + 0.95 * buf 正交化后乘维度缩放因子更新二维权重。' },
        { concept: 'QK-clip', code: 'm14_muon_optimizer/demo.py:qk_clip', takeaway: '当最大注意力分数超出阈值 $\\tau$ 时计算 $\\eta = \\tau / S_{\\max}$, 对 $W_Q$ 与 $W_K$ 各乘 $\\sqrt{\\eta}$ 使点积恰好缩回上限。' },
        { concept: '对照实验', code: 'm14_muon_optimizer/demo.py:run', takeaway: '通过 rotate=True/False 切换耦合病态与轴对齐病态, 并为两种优化器各自独立网格搜索最优学习率以保证公平比较。' },
      ],
      snippetTitle: 'Muon 的核心',
      snippet: `def newton_schulz(G, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (norm(G) + 1e-7)               # 奇异值全部 ≤ 1, 迭代才收敛
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X  # 每个 σ ← aσ + bσ³ + cσ⁵
    return X                               # ≈ U Vᵀ

buf = mu * buf + G                         # 唯一的优化器状态
O = newton_schulz(G + mu * buf)            # Nesterov 动量, 再正交化
W -= lr * 0.2 * sqrt(max(W.shape)) * O`,
      source: ['llm_train/m14_muon_optimizer/demo.py:newton_schulz', 'llm_train/m14_muon_optimizer/demo.py:Muon'],
      run: 'python -m llm_train.m14_muon_optimizer.demo',
    },

    'train-ulysses': {
      title: 'Ulysses 序列并行 · 用 all-to-all 把切序列换成切头',
      subtitle: 'Ring Attention 固定切序列并沿环传 K/V, Ulysses 则在进入注意力层前用 all-to-all 把「分段序列 + 全部头」翻转为「完整序列 + 部分头」, 直接复用标准注意力核。',
      tldr: 'DeepSpeed-Ulysses 利用多头注意力各头相互独立的性质, 在注意力前后做 4 次 all-to-all 重排张量切分维度:\n- 注意力层外按序列切分 [T/P, H, d], 逐位置计算的 Norm 与 MLP 零通信完成。\n- 进入注意力前重排为 [T, H/P, d], 每张卡在完整序列上直接跑本地的 $H/P$ 个头, 算完再换回 [T/P, H, d]。\n其单卡通信量 $4(P-1)/P^2$ 份随卡数 $P$ 增大反而下降, $P$ 较大时近似按 $1/P$ 缩小 (m16 中 $P=8$ 时仅为 Ring Attention 的 1/4), 但受限于头数整除约束与全互联带宽, 实际集群常采用「机内 Ulysses + 机间 Ring」混合部署。',
      question: 'Ulysses 的单卡通信字节数随并行度 P 增大反而减少, 为什么超大规模长上下文训练没有全部只用 Ulysses?',
      evolution: {
        title: '长序列从切段到切头',
        subtitle: "根问题: 中间激活随序列长度增长, 超长上下文下单条序列便会超出单卡显存容量。",
        steps: [
          { name: '整条序列放一卡', pain: "(原点) 前向中间激活随序列长度 T 增长, 超长上下文在单张卡上无法容纳", fix: "单卡独立计算整条序列; 序列长度 T 增大后触发 OOM, 而常规 DP、TP 与 PP 均未切分序列维度" },
          { name: 'Ring Attention', year: 2023, pain: "将长序列切分到多张卡后, 注意力计算仍要求每个位置的 query 访问全序列的 key 与 value", fix: "将序列切成 P 段并沿环传递 K/V 分块; 每收到新块便用 online softmax 按更新后的最大值重缩放并合并统计量, 输出与全量计算精确相等" },
          { name: 'zigzag', pain: "因果 mask 下按顺序连续切分会导致末尾卡计算量最重, 模拟墙钟仅从 256 降至 228", fix: "将序列切成 2P 段并让每张卡首尾对称配对持有两段, 使各卡每轮有效计算量均衡, 模拟墙钟从 228 降至 132" },
          { name: 'Ulysses', year: 2023, pain: "Ring Attention 的单卡通信量随并行度 P 递增, 在 $P=8$ 时达到每 rank 28672 B", fix: "通过 all-to-all 将按序列切分重排为按注意力头切分, 直接复用标准注意力核; 单卡通信量随 P 增大反而下降, $P=8$ 时降至 7168 B" },
        ],
      },
      code: 'llm_train/m16_ulysses_sequence_parallel/demo.py · llm_train/m12_sequence_parallel/demo.py',
      points: [
        {
          title: '只翻转分片维度, 原封不动复用标准注意力核',
          key: true,
          body: '设序列长 $T$、注意力头数 $H$、头维 $d$、并行卡数 $P$。Transformer 中除注意力外的所有算子 (RMSNorm、MLP、残差) 都按位置独立运算, 按序列维切成 [T/P, H, d] 即可无通信并行; 唯独注意力需要跨全序列交互, 但各头之间互不耦合。\nUlysses 在算出本地 Q、K、V 后各做一次 all-to-all, 将数据重排为 [T, H/P, d]: 每张卡拥有完整的 $T$ 个位置、只负责其中 $H/P$ 个注意力头, 直接调用现成的因果 FlashAttention 核即可 (且每卡都是全序列因果三角形, 负载天然均衡, 无需 zigzag 配对), 算完再经第 4 次 all-to-all 换回 [T/P, H, d]。m16 实测 $P=2, 4, 8$ 下输出与单卡全量注意力最大差不超过 4.4e-16。',
        },
        {
          title: '单卡通信量随并行卡数 P 增加反而递减',
          body: '对比单趟前向中每张卡发出的字节数 (以完整 [T, H, d] 张量大小记作 1 份):\n- Ulysses: 共 4 次 all-to-all (Q、K、V 与输出各一次), 每次单卡手头本就只占 $1/P$ 份数据, 其中属于自身负责头的那 $1/P$ 留在本地, 实际发往外卡的只有 $4 \\times (1/P) \\times (P-1)/P = 4(P-1)/P^2$ 份, 随 $P$ 增大而下降, $P$ 较大时近似与 $P$ 成反比。\n- Ring Attention: 每卡需把 K 和 V 各分块传满 $P-1$ 轮, 发出总量为 $2(P-1)/P$ 份 (K、V 各一份 $(P-1)/P$), 随 $P$ 增大单调趋近常数。\nm16 在 T=32, H=8, d=8 (float64) 下实测 $P=2 / 4 / 8$ 的单卡发送字节数: Ulysses 为 16384 / 12288 / 7168 B, Ring 则为 16384 / 24576 / 28672 B ($P=8$ 时 Ring 通信量是 Ulysses 的 4 倍)。',
        },
        {
          title: '头数整除上限与全互联拓扑促成「机内 Ulysses + 机间 Ring」',
          body: 'Ulysses 有两道硬边界阻止它无限扩卡:\n- 并行度受头数卡死: 每张卡必须分到整数个头 (H % P == 0, m16 传入 P=16, H=8 直接触发断言报错); 在 GQA 模型中 KV 头数往往只有 8 个, 不额外复制 KV 时 $P$ 最多只能开到 8。\n- 依赖高对分带宽: all-to-all 要求组内任意两卡同时互发小块数据, 适合单机内高速 NVLink 全互联; 跨机器网卡更容易被全员并发打满, 反而是只跟左右相邻节点点对点通信的 Ring 更易与计算重叠。\n因此工业界超长上下文训练常将两者正交组合: 机内 8 卡走 Ulysses 压低通信量, 跨机 4 节点走 Ring 突破头数上限, 合起来支撑 32 路序列并行。',
        },
      ],
      links: [
        { from: 'train-moe-seq', to: 'Ring Attention', body: 'Ring Attention 固定切序列、沿环流式传 K/V 并重写分块合并核; Ulysses 则靠一次性重排换回标准全序列注意力调用。' },
        { from: 'core/collectives.py:all_to_all', to: 'seq_to_head', body: 'seq_to_head 与 MoE 的专家分发复用同一个 all_to_all 原语, 将各卡本地序列段的头分组转置发往对应目标卡。' },
        { from: '[T/P, H, d]', to: '[T, H/P, d]', body: '从「分段序列、全部头」切换到「完整序列、部分头」再切回原形状, 这一次维度对调就是 Ulysses 的全部核心逻辑。' },
      ],
      sourceRows: [
        { concept: '序列 → 头', code: 'm16_ulysses_sequence_parallel/demo.py:seq_to_head', takeaway: '每卡将本地 [T/P, H, d] 按头维切成 $P$ 组经 all_to_all 互换, 收齐后沿序列维拼接成 [T, H/P, d]。' },
        { concept: '头 → 序列', code: 'm16_ulysses_sequence_parallel/demo.py:head_to_seq', takeaway: '注意力算完后做对称逆变换: 将完整序列切回 $P$ 段互换, 再沿头维拼回 [T/P, H, d]。' },
        { concept: '整除约束', code: 'assert H % P == 0', takeaway: '校验注意力头数可被并行度整除, P=16, H=8 的反例在此处直接抛出 AssertionError。' },
        { concept: '通信账', code: '4 * (P - 1) / P * shard_bytes', takeaway: '单卡本地分片大小为 shard_bytes (整张量的 $1/P$), 4 次重排每次发出其中的 $(P-1)/P$。' },
      ],
      snippetTitle: 'Ulysses 注意力',
      snippet: `def ulysses_attention(q, k, v, P):
    assert H % P == 0                                  # 头数必须能被 P 整除
    # 3 次 all-to-all: P × [T/P, H, d] → P × [T, H/P, d]
    qs, ks, vs = (seq_to_head(split(a, P, axis=0)) for a in (q, k, v))
    # 每卡: 完整序列 × H/P 个头, 普通因果注意力, 核不用改
    outs = [causal_mha(qs[r], ks[r], vs[r]) for r in range(P)]
    # 第 4 次 all-to-all: 换回按序列切
    return concat(head_to_seq(outs), axis=0)`,
      source: ['llm_train/m16_ulysses_sequence_parallel/demo.py:seq_to_head', 'llm_train/m16_ulysses_sequence_parallel/demo.py:ulysses_attention'],
      run: 'python -m llm_train.m16_ulysses_sequence_parallel.demo',
    },

    // ---------------------------------------------------------------- 数据: 去重 / 配比 / packing
    'train-data-packing': {
      title: '数据流水线与 packing · 去重、配比、拼行不串文档',
      subtitle: '训练算力最终转化成什么能力取决于喂进去的语料: MinHash-LSH 筛除改写转载, 温度配比平衡大小语料源, 带文档掩码的序列装箱 (packing) 则挤干无效 padding。',
      tldr: '- 近似去重: 整篇精确哈希漏掉了改过几个词的转载; MinHash-LSH 通过短签名估计 5-gram 集合重合度并用分桶筛选候选对, 将去重召回率从 0.258 提升至 0.967。\n- 数据源配比: 按库存幂次 $p_i \\propto n_i^{1/T}$ 上采样稀缺高质量领域 (如数学/代码), 同时权衡重复轮数带来的背诵风险。\n- 序列装箱 (packing): 用 FFD 把多篇短文档塞进同一条定长序列, 将有效 token 占比从 0.285 拉高到 0.990; 同时补齐分块对角文档 mask、位置重置与跨篇边界 label 屏蔽, 使计算结果与逐篇独立训练严格一致。',
      question: '改了几个词的网页转载怎么高效检出而不用两两全比? 把多篇互不相干的短文拼进同一行训练时, 漏掉哪几项处理会让文档互相串味?',
      evolution: {
        title: '把 pad 一点点挤出去',
        subtitle: "根问题: GPU 批处理要求输入张量形状规整, 但语料文档长短不一, 补齐的 pad 位置同样消耗矩阵乘算力。",
        steps: [
          { name: '每篇一行补 pad', pain: "(原点) 语料文档长度中位数仅 28 token, 而序列窗口定长为 128, 短文档不足部分需补齐", fix: "每篇文档独占一行并补 pad 至定长 L; 真实 token 占比仅为 0.285, 逾七成算力消耗在无效填充位上" },
          { name: '动态 padding', pain: "定长填充下超过七成位置均为无效 pad, 前向与反向计算存在大量算力浪费", fix: "改为在每个 batch 内仅动态补齐至本批最长文档, 有效 token 占比升至 0.406; 但单篇长文档仍会拉长整批填充" },
          { name: '直接拼接', pain: "只要维持单篇文档独占一行的排布, 批内长度差异带来的填充浪费便无法根除", fix: "将多篇文档首尾拼接后按定长切行 (GPT-2/3 做法), 彻底消除 pad; 但同一行内相邻文档在注意力中彼此可见" },
          { name: 'FFD + 文档 mask', pain: "仅用下三角因果 mask 时后一篇文档会读取前一篇内容, 导致前向输出相对独立计算产生 3.0 的最大偏差", fix: "用 FFD 整篇装箱将有效占比推至 0.990, 配合分块对角文档 mask 隔离跨篇注意力, 并重置篇首位置与屏蔽跨篇交界 label" },
        ],
      },
      code: 'llm_train/m17_data_pipeline/demo.py · llm_train/m18_sequence_packing/demo.py',
      points: [
        {
          title: '精确哈希只能拦截一字不差的完全副本',
          body: '互联网语料中充斥着轻度改写的转载文章。m17 构造了 300 篇原件加 120 篇重复件 (30 篇逐字复制 + 90 篇每词约 4% 概率增删改的转载, 与原件的 5-gram Jaccard 相似度均值 0.69):\n- 整篇 CRC32 精确哈希: 改掉一个词整篇哈希值就随之改变, 仅抓出 30 篇完全副本与 1 篇未触发随机改动的转载, 召回率低至 0.258。\n- MinHash-LSH 近似去重: 召回率跃升至 0.967 且误删为 0 (仅 4 篇真实相似度落在 0.48~0.52 临界区被判在线下)。',
        },
        {
          title: 'MinHash 压缩集合求相似度, LSH 分桶避开两两全比',
          body: '直接存储每篇文档的全部 5-gram 并做 $O(n^2)$ 两两交并比无法扩展到十亿级网页, MinHash 与 LSH 分两步化简:\n- MinHash 签名: 对全集施加一次随机哈希置换后, 两个集合最小哈希值相等的概率严格等于它们的 Jaccard 相似度 $s$。独立取 128 组线性哈希系数求最小值, 每篇文档即压成 128 个整数的短签名, 两签名对应位相等的比例即为 $s$ 的无偏估计。\n- LSH (局部敏感哈希) 分桶: 将 128 维签名切成 $b=32$ 段、每段 $r=4$ 行; 只要两篇文档在任意一段上 4 个值完全一致就落入同桶成为候选对。单段全相等概率为 $s^4$, 至少一段命中的概率呈陡峭 S 形曲线 $1 - (1 - s^r)^b$ (拐点约 $(1/b)^{1/r} \\approx 0.42$): 相似度 0.2 的无关文档仅 5% 概率进桶, 相似度 0.6 的转载则有 98.8% 概率被捕获。进桶候选对再按签名相似度 $\\ge 0.5$ 二次确认即可。',
        },
        {
          title: '数据源温度配比与确定性额度调度',
          body: '预训练语料各分域规模极其悬殊: m17 中 5 个数据源共 1000M token, 普通网页占 800M 而数学仅 2M (0.2%)。温度采样按库存幂次设定采样权重 $p_i \\propto n_i^{1/T}$ ($T=1$ 为原始比例, $T$ 越大越趋向均分; 注意它调节的是数据源抽样配比而非解码 softmax 温度):\n- 当总训练预算为 500M token 时, 取 $T=2$ 可将数学占比抬升至 2.8% (重复训练约 6.9 遍); 若盲目拉到 $T=5$, 数学库存会被重复训练 25.4 遍, 很可能已经过拟合。\n确定目标配比后, 若每个样本独立随机抽样 (i.i.d.), 单 batch (64 条) 内的实际配比最大偏差达 0.228; 改用确定性额度调度 (每步为各源累加 $p_i$ 额度, 选最高者出样并扣减 1) 可将单批最坏偏差压低到 0.021。',
        },
        {
          title: 'FFD 序列装箱将有效算力占比从 0.285 推升至 0.990',
          body: 'm18 中模型定长窗口为 $L=128$, 而 400 篇文档长度中位数仅 28 token:\n- 每篇独占一行补 pad 到 128: 真实 token 仅占 0.285, 逾七成矩阵乘浪费在 pad 位置上。\n- Batch 内动态补齐到本批最长篇: 升至 0.406, 仍易被批内单篇长文拖累。\n- FFD (First-Fit-Decreasing) 整篇装箱: 将文档按长度降序排列, 依次塞入第一个剩余空间足够的训练行中, 400 篇文档紧凑装入 115 行, 有效 token 比例达 0.990。',
        },
        {
          title: '拼行不串味的三件套: 文档掩码、位置归零与跨篇标签屏蔽',
          key: true,
          body: '把多篇独立文档塞进同一行后, 若只保留普通下三角因果掩码, 后一篇文档就会在注意力里直接读到前一篇的内容 (m18 中与逐篇独立前向相比, 输出最大偏差在 RoPE 下达 3.0, 绝对位置编码下达 4.3)。要让装箱结果与逐篇训练严格等价, 需补齐三处细节:\n- 分块对角文档 mask: 限定位置 $t$ 只能关注满足 t >= s and doc_id[t] == doc_id[s] 的同篇前文 (工业界直接向 FlashAttention varlen 接口传入各篇起止偏移 cu_seqlens, 连跨篇注意力块都不用算, 本例注意力算量只剩整行因果注意力的 0.466)。\n- 篇首位置 id 归零: 每篇文档开头把 position_id 重置回 0 (绝对位置编码必需; RoPE 在加了文档 mask 后仅依赖篇内相对位移, 重置还能避免 bf16 大位置转角精度退化)。\n- 屏蔽跨篇交界处的预测标签: 前一篇末尾 token 的下一个位置是后一篇开头, 两者毫无因果关联, 必须从交叉熵损失中剔除。三项齐备后整行交叉熵总和为 566.353069, 与逐篇单独计算的总和相同 (漏屏蔽交界标签则会偏高至 577.563075)。',
        },
        {
          title: '规则清洗的盲区与预训练拼接惯例',
          body: '读实验数字时需留意两处真实工程边界:\n- 网页启发式规则会误杀代码: m17 按平均词长、特殊符号占比与停用词过滤合成垃圾文本时, 把 100% 的正常代码片段也判成了垃圾; 真实数据管线必须先按来源/模态分流, 再叠加强分类器过滤 (如 FineWeb / DCLM)。另外 demo 的垃圾页按 5 种模式合成, 规则一眼就能分开, 拦截率 1.000 不代表真实网页上的效果。\n- 预训练与微调对跨篇掩码的取舍: 早期 GPT-2/3 预训练常仅用 <eos> 拼接而不加文档 mask; Llama 3 加了文档 mask, 但其报告说它在标准预训练里影响有限, 在超长序列的继续预训练里才重要。指令微调 (SFT) 样本短且彼此无关, 装箱时一般都要启用文档 mask 与交界标签屏蔽, 避免短样本之间相互污染。',
        },
      ],
      links: [
        { from: 'MinHash 签名', to: 'LSH 分桶', body: 'MinHash 将不定长文本压成 128 个整数以快速估算重合度, LSH 分桶则把 $O(n^2)$ 全量比对收敛到同桶高相似候选对。' },
        { from: 'doc_mask', to: 'FlashAttention varlen', body: '教学代码构造显式 [L, L] 布尔矩阵验证等价性; 实际训练直接把文档边界 cu_seqlens 传给变长 FlashAttention, 既隔离跨篇干扰又省去空白块算力。' },
        { from: 'RoPE', to: '位置重置', body: 'RoPE 内积 $\\langle R(m)q, R(n)k\\rangle$ 仅取决于相对距离 $n-m$, 但实际框架在装箱时仍习惯在篇首重置位置 id 以防大坐标下浮点转角失真。' },
        { from: 'packing', to: 'SFT', body: '监督微调样本普遍短小且彼此独立, 装箱训练时必须配齐文档掩码与边界 -100 屏蔽。' },
      ],
      sourceRows: [
        { concept: 'MinHash 签名', code: 'm17_data_pipeline/demo.py:minhash_signatures', takeaway: '用 128 组线性同余参数 (a * x + b) % p 模拟随机置换并对文档 5-gram 集合求最小值向量。' },
        { concept: 'LSH 分桶 + 校验', code: 'm17_data_pipeline/demo.py:lsh_dedup', takeaway: '签名切为 32 段×4 行分桶产生候选对, 再按签名重合率 $\\ge 0.5$ 校验并用并查集聚簇去重。' },
        { concept: 'crc32 而非 hash()', code: 'zlib.crc32(" ".join(words[i:i + n]).encode())', takeaway: '使用确定性的 zlib.crc32 替代每次启动随机加盐的内置 hash(), 保证跨进程去重结果可复现。' },
        { concept: '温度配比', code: 'm17_data_pipeline/demo.py:temperature_weights', takeaway: '对各数据源 token 总量取 $1/T$ 次幂后归一化, 平滑拉高小体量高价值语料的抽样权重。' },
        { concept: '文档 mask', code: 'm18_sequence_packing/demo.py:doc_mask', takeaway: '一行布尔组合 (t >= s) & (doc[t] == doc[s]), 同时满足因果顺序与同篇隔离。' },
        { concept: '位置重置', code: 'm18_sequence_packing/demo.py:reset_positions', takeaway: '在每篇新文档起始处将位置计数归零 (如长度 3、2、4 拼行后位置序列为 [0,1,2, 0,1, 0,1,2,3])。' },
      ],
      snippetTitle: '去重与 packing 的骨架',
      snippet: `# 近似去重: 签名 → 分段分桶 → 同桶成候选 → 校验
sig = minhash(shingles(doc))                  # [128] 个整数
for band in range(32):                        # 32 段 × 4 行
    buckets[band, tuple(sig[band*4:(band+1)*4])].append(doc_id)
dups = [(i, j) for i, j in same_bucket_pairs
        if mean(sig[i] == sig[j]) >= 0.5]     # ★ 两个阈值分开调

# packing: 一行塞多篇, 补三样才与逐篇等价
mask = causal & (doc[:, None] == doc[None, :])   # 1. 文档 mask
pos = arange(L) - start_of_my_doc                # 2. 位置重置 (绝对位置编码必需)
valid = doc[1:] == doc[:-1]                      # 3. 不跨篇预测`,
      source: ['llm_train/m17_data_pipeline/demo.py:lsh_dedup', 'llm_train/m17_data_pipeline/demo.py:temperature_weights', 'llm_train/m18_sequence_packing/demo.py:doc_mask', 'llm_train/m18_sequence_packing/demo.py:reset_positions'],
      run: 'python -m llm_train.m17_data_pipeline.demo',
    },

    // ---------------------------------------------------------------- 规模: scaling law / μP
    'train-scaling': {
      title: '规模 · scaling law 与 μP',
      subtitle: '大模型往往只训得起一次, 开训前就要定下参数规模、数据量与学习率。Scaling law 用小规模网格外推算力最优配比, μP 则通过按宽度缩放初始化与层学习率实现超参零样本迁移。',
      tldr: '- Scaling law 用幂律 $L(N,D) = E + A/N^\\alpha + B/D^\\beta$ 刻画损失随参数量 $N$ 和训练数据量 $D$ 的下降规律; 在固定算力 $C \\approx 6ND$ 下, 模型过小或数据不足都会推高损失, IsoFLOP 曲线谷底给出算力最优模型规模。\n- 标准参数化 (SP) 下模型加宽 16 倍会导致最优学习率漂移 64 倍; 最大更新参数化 (μP) 将输出层初始化与隐藏/输出层 Adam 学习率按宽度倍数 $m$ 额外除一次, 使宽 32 到 512 的最优学习率稳定停在同一档 $2^{-5}$。',
      question: '总算力预算固定时, 该把预算多分给模型参数还是训练语料? 为什么在标准初始化下小模型调好的学习率搬到宽模型上就失效了?',
      evolution: {
        title: '只训得起一次, 所以先在小模型上算',
        subtitle: "根问题: 大规模预训练通常只容许单次预算投入, 参数量、数据规模与学习率均需在启动训练前确定。",
        steps: [
          { name: '照经验放大', pain: "(原点) 大模型算力预算只够完整训练一次, 参数规模、数据量与学习率必须在训练前选定", fix: "按上一代模型配置经验等比放大; 新增算力如何在参数与语料间分配、学习率如何设定均缺乏定量依据" },
          { name: 'Kaplan scaling law', year: 2020, pain: "算力预算增加后, 缺乏定量规律指导预算应优先投入模型参数还是训练数据", fix: "在小规模模型上拟合验证损失随参数量 N 与数据量 D 变化的幂律并向外推; 所给配比偏向优先扩大模型规模" },
          { name: 'Chinchilla IsoFLOP', year: 2022, pain: "参数规模偏大而训练语料偏少时模型欠训练, 相同算力预算下未能将损失压至最低", fix: "固定总算力 $C\\approx 6ND$ 扫描不同 N 与 D 配比, 定位损失曲线的 U 形谷底, 并拟合 $E+A/N^\\alpha+B/D^\\beta$ 推算最优 N" },
          { name: 'μP', year: 2022, pain: "模型规模选定后无法在大模型上网格搜索 lr; 标准参数化 (SP) 下宽度放大 16 倍会使最优 lr 漂移 64 倍", fix: "将输出层初始化标准差及隐藏层、输出层 lr 按宽度倍数 m 额外除一次: 宽度从 32 扩至 512 时最优 lr 均保持在 $2^{-5}$" },
        ],
      },
      code: 'llm_train/m19_scaling_laws/demo.py · llm_train/m20_mup/demo.py',
      points: [
        {
          title: '损失拆解为不可约噪声、模型容量项与数据有限项',
          body: 'Chinchilla 幂律公式 $L(N,D) = E + A/N^\\alpha + B/D^\\beta$ 将验证损失分解为三项之和: 数据固有的不可约熵底 $E$、受模型规模制约的近似误差 $A/N^\\alpha$, 以及受训练样本数制约的估计误差 $B/D^\\beta$。\nm19 在带方差 0.01 标签噪声的合成师生回归任务上真实训练了 $6\\times 6 = 36$ 个不同宽度与数据量的 MLP (单遍训练并各自完整余弦退火), 拟合得到 $E=0.0080$ (贴近真实噪声底 0.01)、$\\alpha=1.075$、$\\beta=1.125$: 当模型极小 ($N=21$) 时把数据从 16K 加到 65K, 损失仅从 0.0608 微动到 0.0603, 说明此时瓶颈完全卡在容量项 $A/N^\\alpha$ 上。',
        },
        {
          title: '固定算力预算下, IsoFLOP 曲线呈 U 形谷底',
          key: true,
          body: '训练全程的前反向总计算量近似为 $C \\approx 6ND$ 次浮点运算 (每个参数每个 token 前向约 2 次、反向约 4 次): 预算 $C$ 固定时, 参数量 $N$ 翻倍就意味着能训练的 token 数 $D$ 减半。沿着 m19 网格中算力同为 $C \\approx 8\\times 10^6$ 的反对角线观察 6 个模型:\n- 模型偏小 ($N=21$): 数据虽充足但容量见顶, loss 为 0.0603。\n- 配比均衡 ($N=81$): 落入 U 形谷底, 取得最低 loss 0.0285 (闭式解预测最优值为 $N=89$; 这 6 个点本身也参与了拟合, 不是样本外检验)。\n- 模型偏大 ($N=641$): 步数太少欠训练, loss 反弹至 0.0853。\n将 $D = C/(6N)$ 代入幂律求极值可得最优缩放指数 $N_{\\text{opt}} \\propto C^a$, 其中 $a = \\beta / (\\alpha + \\beta)$ 取决于数据项与容量项的衰减速率之比。',
        },
        {
          title: '合成回归指数不能直接套用到真实自然语言模型',
          body: '本实验用于演示完整拟合与极值推导方法, 具体数值与真实 LLM 存在本质区别:\n- 幂指数差异: 合成教师网络第 $k$ 个单元的输出权重按 $1/k$ 衰减, 这决定了 $\\alpha \\approx 1.08, \\beta \\approx 1.13$, 远陡于自然语言的 Chinchilla 经验值 ($0.34 / 0.28$)。$a = \\beta / (\\alpha + \\beta) \\approx 0.51$ 与论文的 $\\sim 0.50$ 接近只是巧合: 本例 $\\alpha$、$\\beta$ 碰巧差不多大, 换一个 teacher 谱, $a$ 就会变。\n- 数据口径与统计: 这里一个「token」是一个回归样本, 算出的 $D/N \\approx 150$ 不能与 Chinchilla 的每参数约 20 个 token 直接比较; 每格只跑 1 个种子, 拟合的相对残差约 10%。\n- 训练最优不等于推理最优: Chinchilla 只最小化一次性预训练算力; 面向海量请求部署的小模型 (如 Llama 3 8B 训满 15T token, $D/N \\approx 1900$) 会刻意大幅「过训练」以换取线上推理阶段更低的延迟与显存成本。',
        },
        {
          title: '标准参数化 (SP) 下加宽网络会让最优学习率大幅左移',
          body: '标准参数化 (SP) 采用 $1/\\sqrt{\\text{fan\\_in}}$ 初始化并对所有层使用同一个学习率。为什么加宽后原学习率会失效? 在 Adam 下每个权重元素单步变化量级约为 $\\text{lr}$; 隐藏层单个神经元输出将 width 个输入的更新同向叠加后, 对下一层激活造成的扰动幅度与 $\\text{lr} \\cdot \\text{width}$ 成正比——网络越宽冲击越剧烈, 被迫大幅调小学习率。\nm20 在 3 种宽度 (32 / 128 / 512) × 10 档学习率网格上训练两层 ReLU MLP: SP 下最优 $\\log_2 \\text{lr}$ 从 -5 一路滑落到 -7 和 -11 (加宽 16 倍最优 lr 缩小了 64 倍); 直接把宽 32 的最优 lr 搬给宽 512 使用, 损失高达 0.0976, 比该宽度真实最优值 (0.0215) 差了 4.5 倍。\n上面的叠加机制只预测 lr 与宽度成反比, 即宽 16 倍时 lr 小 16 倍; 实测的 64 倍比预测多, 这组实验分不清多出的部分从哪来: SP 曲线本身不光滑 (宽 512 在 -9 到 -7 之间来回跳), lr 网格也是 2 倍一格。',
        },
        {
          title: 'μP: 三处按宽度倍数多除一次, 锁住最优学习率位置',
          body: '最大更新参数化 (μP) 定义相对基础宽度的放大倍数 $m = \\text{width} / \\text{base}$ (m20 中 base=32, 宽 512 时 $m=16$), 仅在 SP 基础上修改三处缩放:\n- 输出层权重初始化标准差额外除以 $m$ (让初始输出随宽度变小, 否则训练一开始就被随机的输出层主导)。\n- 隐藏层与输出层的 Adam 学习率各除以 $m$ (恰好抵消 width 个权重更新的叠加放大; 输入层 fan_in 固定为输入特征维 $d_{\\text{in}}$, 学习率保持不变)。\n在基础宽度 $m=1$ 处 μP 与 SP 完全重合; 扩展到宽 32 / 128 / 512 后, 三条曲线的最优学习率全部停在 $2^{-5}$, 且同学习率下模型越宽损失单调越低 (0.0186 -> 0.0101 -> 0.0088), 省去了在大模型上网格重扫学习率的高昂代价。\n这里只到 512 宽、150 步; μTransfer 论文在 Transformer 上验证到约 6.7B 参数的迁移, 本 demo 只说明机制。若换成常数 lr, μP 的最优点会在 -5 / -7 / -8 之间漂, 迁移没这么干净, 所以 demo 用了线性衰减到 0。',
        },
      ],
      links: [
        { from: 'Kaplan (2020)', to: 'Chinchilla (2022)', body: '早期实验为不同数据量共用固定余弦周期导致小预算组未充分退火; Chinchilla 修正为每组训练步数恰好走完各自完整的余弦退火周期, m19 沿用了这一设定。' },
        { from: 'Chinchilla 最优', to: '过训练', body: '算力最优配比只核算训练期成本; 实际开源基座常把 token/参数比推高一到两个数量级, 用一次性训练算力换取长期推理端低成本。' },
        { from: 'm19 共用 lr=0.03', to: 'μP', body: 'm19 所有规模共用 lr = 0.03, 只在 1e-2 / 3e-2 / 1e-1 三个值里粗扫过。真实的 scaling 网格横跨不同模型宽度时, 需要每个规模单独调 lr, 或借助 μP 按宽度自动缩放各层步长, 以免固定单一学习率给宽模型引入次优偏差。' },
        { from: 'μP', to: 'lr 幂律', body: '除修改参数化的 μP 外, 另一条常见工程路线是在若干小规模点上实测最优学习率并直接拟合其随算力的幂律外推公式 (如 DeepSeek LLM)。' },
      ],
      sourceRows: [
        { concept: '拟合', code: 'm19_scaling_laws/demo.py:fit_scaling_law', takeaway: '采用变量投影法: 在 $79\\times 79$ 网格上枚举 $(\\alpha, \\beta)$, 内层通过非负最小二乘直接解出最优线性系数 $(E, A, B)$。' },
        { concept: '闭式最优', code: 'm19_scaling_laws/demo.py:compute_optimal', takeaway: '根据拉格朗日极值公式直接解出 $N_{\\text{opt}} = G(C/6)^a$, 并与一维数值搜索结果交叉校验至 1% 误差内。' },
        { concept: '单遍 + 退火到 D', code: 'm19_scaling_laws/demo.py:train', takeaway: '每个样本只训练一遍且余弦学习率周期严格对齐当次数据量 $D$, 消除未完成退火带来的系统性偏差。' },
        { concept: 'SP vs μP', code: 'm20_mup/demo.py:init_and_lrs', takeaway: '清晰展示两套参数化的唯一代码差异: 输出层初始化标准差与隐藏/输出层学习率各除以宽度倍数 $m$。' },
      ],
      snippetTitle: '算力最优与 μP 的实现',
      snippet: `# 固定 C = 6ND, 把 D = C/(6N) 代入 L(N, D) 求最小
a = beta / (alpha + beta)                     # 本例 0.511; 论文指数给 0.45
G = (alpha * A / (beta * B)) ** (1 / (alpha + beta))
N_opt = G * (C / 6) ** a
D_opt = (C / 6) / N_opt

# μP (Adam): m = width / base_width, m = 1 时与 SP 完全一样
W_out = randn(width, 1) / sqrt(width) / m     # ★ 输出层初始化多除 m
lrs = [lr, lr / m, lr / m]                    # ★ 输入层不缩, 隐藏层和输出层除 m`,
      source: ['llm_train/m19_scaling_laws/demo.py:fit_scaling_law', 'llm_train/m19_scaling_laws/demo.py:compute_optimal', 'llm_train/m20_mup/demo.py:init_and_lrs'],
      run: 'python -m llm_train.m19_scaling_laws.demo',
    },

    // ---------------------------------------------------------------- 评测
    'train-eval': {
      title: '评测 · perplexity、污染、pass@k 与裁判',
      subtitle: '模型榜单数字离开统计口径便失去可比性: 分词粒度会改写困惑度, 改写题能绕过 n-gram 污染筛查, 朴素 pass@k 存在向下偏置, LLM 裁判则自带前后位置偏好。',
      tldr: '四类主流评测指标各自对应明确的口径陷阱与校正方法:\n- 困惑度 (PPL): 数值随分词器粒度剧烈变化 (token 合并减半会让同一模型的 PPL 从 5.501 跳到 30.265), 跨词表对比必须折算为每字符/每字节比特数 (bits-per-char/byte)。\n- 数据污染筛查: n-gram 匹配能抓出原题和微改题, 但对每 3 个词换掉 1 个的重度改写题召回为 0, 过滤后的剩余得分仍可能高于真实干净水平。\n- 代码生成 pass@k: 直接用经验通过率代入 $1-(1-\\hat p)^k$ 会因凹函数性质系统性偏低, 应采用无放回组合无偏估计量。\n- LLM-as-a-Judge: 裁判天然偏爱排在前面的候选回答, 需交换 A/B 顺序各判一次并取一致结论消除位置偏差。',
      question: '评测集得分上涨时, 如何逐项排除词表差异、改写题泄漏、估计器偏置与裁判位置偏心这四类干扰?',
      evolution: {
        title: '题越开放, 判分越难',
        subtitle: "根问题: 评测分数旨在刻画模型的真实能力, 但单一打分方式只能度量特定维度的表现并受制于统计口径。",
        steps: [
          { name: 'perplexity', pain: "(原点) 训练过程仅输出交叉熵 loss, 需要直观的留出集指标度量语言建模好坏", fix: "在留出集上计算困惑度 $e^{\\mathrm{CE}}$; 但更换分词器会导致数值剧烈变化 (从 5.501 变至 30.265), 且无法衡量下游解题能力" },
          { name: '题库 + 去污染', pain: "仅凭困惑度 PPL 只能反映文本似然高低, 无法判断模型在下游任务上的解题能力", fix: "构建固定评测题库按答题准确率打分; 但原题混入训练语料会虚抬得分, 且 n-gram 去污染只能检出未经改写的泄漏题" },
          { name: 'pass@k', year: 2021, pain: "代码生成题存在多种等价写法, 与参考答案逐字比对会将正确实现误判为错误", fix: "通过执行单元测试判定对错, 对每题采样 n 个解并用无偏估计式计算 pass@k; 朴素有偏公式在 $k=10$ 时偏低 0.039" },
          { name: 'LLM 裁判 + 交换', year: 2023, pain: "开放式写作与对话任务无单元测试可跑, 引入 LLM 裁判评判时又会偏爱排在前面的候选回答", fix: "交换 A/B 顺序各评判一次、仅采纳两次一致的结论: 更优回答排在 B 位时的准确率从 0.593 升至 0.991, 代价是调用次数翻倍" },
        ],
      },
      code: 'llm_train/m21_llm_eval/demo.py',
      points: [
        {
          title: '困惑度 (PPL) 度量单步预测的等效候选分支数',
          body: '给定验证集逐 token 平均负对数似然 (即交叉熵 $\\mathrm{CE}$), 困惑度定义为 $\\mathrm{PPL} = e^{\\mathrm{CE}}$, 直观含义是模型在每个位置平均在多少个等概率候选词之间犹豫。m21 在已知真实转移矩阵的 20 词马尔可夫语料上验证:\n- 均匀瞎猜 ($p=1/20$): PPL 恰好为 20.000。\n- 训练集统计出的 bigram 模型: PPL 降至 5.501, 逼近用真实转移概率算出的 PPL 下界 5.448。',
        },
        {
          title: '跨模型对比不能直接比 PPL, 必须换算到字符或字节基准',
          body: '对同一段文本、同一个概率模型, 假设将相邻两个 token 合并为一个复合 token (总 token 数减半, 整篇文本的总对数似然不变):\n- 单 token 平均交叉熵翻倍, 导致 PPL 从 5.501 变为它的平方 30.265。\n- 折算到原始字符长度的 bits-per-char 则保持恒定的 2.4598。\n因此词表大小不同的两个模型绝不能直接比较 PPL; 即使同一词表, 全库按 token 加权平均 (5.501) 与先按篇求 PPL 再算术平均 (5.573) 也会因短文档权重差异给出不同读数。',
        },
        {
          title: 'n-gram 去污染拦得住原题复制, 拦不住重度改写泄漏',
          key: true,
          body: '评测基准混入预训练语料会让背题伪装成推理能力。常规做法是检索测试题与训练语料的连续词串重叠 (如 8-gram 或 13-gram)。m21 在 200 道测试题背后的训练语料里混入 30 道原题、30 道改 2 词的微改题和 30 道每 3 词改 1 词的深度改写题:\n- 原题与微改题: 仍保留完整 8-gram 片段, 检出率 1.00。\n- 深度改写题: 连续 8 词原串已全部打断, 检出率为 0.00。\n在「见过的题答对率 90%、没见过的题四选一瞎猜 (25%)」的设定下, 原始总分为 0.570, 剔除 8-gram 命中题后降为 0.436, 但仍显著高于真正干净子集实测的 0.300——漏网的改写题依然在虚抬分数。\n「见过就 90% 答对」只是用来演示污染如何抬高分数的假设, 不代表真实模型的记忆率。',
        },
        {
          title: '代码评测 pass@k 需用无放回组合估计消除向下偏置',
          body: '评估代码或数学题时常报告 pass@k (生成 $k$ 个候选中至少有一个通过单测的概率)。实践中每题采样 $n \\ge k$ 个解并测出 $c$ 个正确解, 两种估计公式表现迥异:\n- 朴素代入式 $1 - (1 - c/n)^k$: 由于 $1-(1-p)^k$ 在 $k>1$ 时是关于 $p$ 的严格凹函数, 由 Jensen 不等式可知将带采样方差的 $\\hat p = c/n$ 直接代入会导致期望值系统性偏低。\n- Codex 无偏估计量 $1 - \\binom{n-c}{k} / \\binom{n}{k}$: 直接计算从 $n$ 个样本中无放回抽取 $k$ 个全部落入错误样本 ($n-c$ 个) 的补集概率。\nm21 在 $n=20, k=10$ 下模拟 300 题 × 400 轮: 真实期望为 0.5864, 无偏估计给出 0.5861, 而朴素公式仅为 0.5470 (系统性低估近 4 个百分点)。',
        },
        {
          title: '双向交换判分可抵消 LLM 裁判的位置偏心',
          body: '开放式问答与写作常依赖大模型裁判做 A/B 胜负盲评, 但模型对排在第一位的候选普遍存在正向偏置。m21 用给首位固定加 0.8 分的带噪比较器评测 2000 对随机排列的样本:\n- 单向固定顺序 (A, B): 裁判 70.0% 的概率判首位赢; 优质答案排在 A 位时准确率 0.994, 落到 B 位时骤降至 0.593。\n- 双向交换顺序 (A, B) + (B, A): 正反各判一次, 仅当两次都选同一答案时计胜负、前后矛盾记平局, 优质答案在 A 位与 B 位的非平局准确率立即拉平至 0.993 与 0.991。代价是评测 API 开销翻倍, 且质量接近的样本会留出 40.7% 的平局 (可记 0.5 分或引入多裁判投票); 此外交换顺序只消位置偏差, 消不掉偏爱长回答的长度偏差 (length bias) 与偏爱自家模型回答的自我偏好。',
        },
      ],
      links: [
        { from: '训练 loss', to: 'PPL', body: '预训练监控的逐 token 交叉熵取自然指数 $e^{\\text{loss}}$ 即为对应词表下的困惑度 PPL。' },
        { from: '数据去重', to: '去污染', body: '近似去重在训练集内部清理重复网页, 去污染则在训练集与评测集之间排查重叠样本。' },
        { from: '改写题', to: '其他检测法', body: '弥补字面 n-gram 盲区通常要结合稠密向量语义检索、检查模型能否低熵补全题干后半段, 或持续换用模型截断日期后的新题 (如 LiveCodeBench)。' },
        { from: '平局', to: '多裁判', body: '双向交换判出的平局反映了两份回答差距小于位置偏置阈值, 工程上常折算为半胜或交给更高阶裁判复核。' },
      ],
      sourceRows: [
        { concept: '逐 token NLL', code: 'm21_llm_eval/demo.py:token_nll', takeaway: '逐位置计算真实下一词的负对数概率, 求全序列均值后再取指数即得 PPL。' },
        { concept: 'n-gram 污染检测', code: 'm21_llm_eval/demo.py:ngram_flags', takeaway: '将训练语料全部 8-gram 存入哈希集合, 筛查任一片段命中的测试题。' },
        { concept: '无偏 pass@k', code: 'm21_llm_eval/demo.py:pass_at_k', takeaway: '将组合数比值化简为连乘 $1 - \\prod_{i=n-c+1}^n (1 - k/i)$, 避免大整数阶乘溢出。' },
        { concept: '交换判', code: 'm21_llm_eval/demo.py:judge_swap', takeaway: '先后执行 judge(A, B) 与 judge(B, A), 仅两次指向同一方时确认胜负, 否则返回 0 (平局)。' },
      ],
      snippetTitle: 'pass@k 与交换判',
      snippet: `def pass_at_k(n, c, k):
    if n - c < k:                          # 错的不够 k 个: 挑 k 个必然有对的
        return 1.0
    return 1 - prod(1 - k / arange(n - c + 1, n + 1))   # = 1 − C(n−c,k)/C(n,k)

naive = 1 - (1 - c / n) ** k               # ✗ 对 c/n 是凹的 → 期望偏低

def judge_swap(qa, qb):
    a_first = judge(qa, qb)                # True = 选 A (A 在前)
    b_first = judge(qb, qa)                # True = 选 B (B 在前)
    return +1 if a_first and not b_first else -1 if b_first and not a_first else 0`,
      source: ['llm_train/m21_llm_eval/demo.py:pass_at_k', 'llm_train/m21_llm_eval/demo.py:ngram_flags', 'llm_train/m21_llm_eval/demo.py:judge_swap', 'llm_train/m21_llm_eval/demo.py:token_nll'],
      run: 'python -m llm_train.m21_llm_eval.demo',
    },

    'train-fault-tolerance': {
          title: '容错与低通信训练 · checkpoint 间隔、goodput 与 DiLoCo',
          subtitle: '卡数越多整体 MTBF 越短, 训练任务的指标从吞吐变成 goodput (有效训练时间占墙钟时间的比例)。最优 checkpoint 间隔由落盘耗时和整体 MTBF 算出来, 异步快照把阻塞时间压短, 慢节点与静默数据损坏是另两笔账; 再往下一步是让每个 worker 本地多跑几步, 把同步次数直接降到 1/H。',
          tldr: '容错训练要排的是四件事, 每件都有能算的量:\n- 整体 MTBF: $M_{\\text{sys}} = M_{\\text{gpu}} / N$。单卡 10 万小时在 16384 张卡上只剩 6.1 小时。\n- 最优 checkpoint 间隔: 落盘占比 $\\delta/\\tau$ 与重跑占比 $\\tau/(2M)$ 在 $\\tau = \\sqrt{2\\delta M}$ 处相等, 这也是和最小的那一点 (Young 1974)。\n- 异步快照: 阻塞时间从 10 分钟降到 30 秒时, 最优 goodput 从 75.9% 升到 92.2% (m27 模拟)。它写的字节数一点没少, 省的只是阻塞那一段。\n- DiLoCo: 本地跑 H 步再同步一次, 通信次数降到 1/H; H 太小会被外层动量放大, 太大跟不上, m28 的玩具上最好的 H 是 48。',
          question: '一万多张卡的任务平均几小时就会挂一次, 为什么不干脆每一步都存盘? 既然每步同步梯度这么贵, 为什么不让各个 worker 多跑几步再同步?',
          evolution: {
            title: '从拍脑袋存盘到算出来的同步频率',
            subtitle: '根问题: N 张卡各自独立出故障, 任意一张挂掉整个同步任务就得停, 所以整体 MTBF 随卡数反比缩短。',
            steps: [
              { name: '定期落盘', pain: '(原点) 千卡任务的整体 MTBF 以小时计, 没有 checkpoint 一次故障就等于白跑', fix: '把参数、优化器状态、数据游标和 RNG 一起落盘, 挂掉从上一份接着跑 (m08); 间隔多大靠拍脑袋' },
              { name: 'Young / Daly 算间隔', year: 1974, pain: '存太勤时间花在落盘上, 存太稀每次挂掉要重跑很长一段, 拍脑袋两头都可能亏', fix: '$\\tau = \\sqrt{2\\delta M}$ 让落盘与重跑两笔开销相等; 16384 卡、落盘 10 分钟时最优 goodput 75.9%' },
              { name: '异步快照 + 弹性缩容', pain: '同步落盘那十几分钟谁都不能算, 它直接压住 goodput 的上限; 少几张卡就得停工等修', fix: '先把状态拷进内存再后台落盘, 阻塞降到 30 秒、goodput 升到 92.2%; 少几张卡时重切数据, 用梯度累积补回全局 batch' },
              { name: 'DiLoCo', year: 2023, pain: '以上都还要求每一步跨卡 all-reduce 一遍梯度, 跨数据中心的带宽和延迟撑不住', fix: '每个 worker 本地跑 H 步再同步一次参数, 外层用 Nesterov 动量 SGD 走伪梯度, 通信次数降到 1/H' },
            ],
          },
          code: 'llm_train/m27_fault_tolerance/fault_tolerance.py · llm_train/m28_diloco/diloco.py',
          points: [
            {
              title: '最优 checkpoint 间隔: 两笔开销相等的那一点',
              key: true,
              body: '卡越多挂得越勤: N 张卡各自独立出故障时故障率相加, 整体 MTBF 等于单卡 MTBF 除以 N。单卡取 10 万小时, 1024 张卡是 97.7 小时, 16384 张卡只有 6.1 小时。\n设 $\\delta$ 是一次落盘阻塞的时间, $M$ 是整体 MTBF, $\\tau$ 是两次 checkpoint 之间的纯计算时间, 要付的是两笔:\n- 落盘占比 $\\delta / \\tau$: 每跑 $\\tau$ 就额外花 $\\delta$, 存得越勤这笔越大。\n- 重跑占比 $\\tau / (2M)$: 故障平均落在一段的中间, 期望丢掉半段, 存得越稀这笔越大。\n两者相加在 $\\tau = \\sqrt{2\\delta M}$ 处最小, 这是 Young 1974 的一阶近似; 最优点上两笔正好相等 (m27 实测两边都是 0.118)。Daly 把重启耗时 R 算进去再减掉落盘本身占的那段, 给出 $\\tau = \\sqrt{2\\delta(M+R)} - \\delta$, 适用条件是 $(\\tau+\\delta)/M \\lt 1/2$。m27 在整体 MTBF 6 小时、重启 10 分钟、落盘 10 分钟时: Young 给 85 分钟、Daly 给 76 分钟, 蒙特卡洛扫出来的最优点在 90 分钟 (goodput 75.8%), 三者落在同一片平台上 —— 60 分钟到 90 分钟之间 goodput 只差 0.6 个百分点。',
            },
            {
              title: '异步 checkpoint 省的是阻塞时间, 不是字节',
              body: '公式里只有阻塞时间 $\\delta$ 说话, 所以异步 checkpoint 的做法是先把状态拷进主机内存 (这一下很快), 再让后台线程慢慢写进共享存储。写的字节数一点没少。\nm27 的两组参数 (整体 MTBF 6 小时、重启恢复 10 分钟):\n- 同步落盘阻塞 10 分钟: Daly 间隔 76 分钟, 最优 goodput 75.9%。\n- 异步快照阻塞 30 秒: Daly 间隔 19 分钟, 最优 goodput 92.2%。\n$\\delta$ 缩到 1/20 时, 最优间隔和两笔开销都按 $\\sqrt{1/20} \\approx 0.22$ 缩小, 所以存得更勤反而更划算。goodput 的曲线用解析式 $\\tau / (M e^{R/M}(e^{(\\tau+\\delta)/M}-1))$ 算, 和按泊松过程抽样的蒙特卡洛在八组参数上差最多 0.39 个百分点。\n边界: 模拟假设后台落盘总能在下一次故障前写完。真实实现要留着上一份完整的 checkpoint, 以防后台还没写完就又挂了。',
            },
            {
              title: '慢节点和静默数据损坏: 另外两笔账',
              body: '同步训练的每步时间是 N 个 rank 里的最大值, 而不是平均值。m27 让每个 rank 的单步耗时服从 $1.0 + \\mathrm{Exp}(0.1)$, 抽 4000 次:\n- N = 1 / 8 / 64 / 1024 时每步 1.102 / 1.271 / 1.477 / 1.753。\n- 被最慢的那个拖掉的比例依次是 0.0% / 13.6% / 25.5% / 37.2%。\n静默数据损坏 (SDC, silent data corruption) 是算错了但没报错, NaN guard 查不出来, 因为结果是一个有限、量级也正常的数。两条发现它的路子:\n- 副本间对梯度校验和: m27 翻掉一个 fp32 的指数位时最大差 2.39e+37, 1e-6 的阈值一定抓得到; 翻尾数最低位时差只有 7.45e-09, 阈值法抓不到。逐位比较什么都抓得到, 但这要求两个副本的归约顺序完全一致。\n- 确定性重算: 同一个 rank 同一份输入重算一遍, 校验和逐位相同 (m27 实测差 0.0e+00), 于是一旦对不上就能定位到具体那张卡。',
            },
            {
              title: '弹性缩容能接着训, 接不回逐位一致',
              body: '少几张卡时不必停工: 重切数据分片与优化器状态, 用梯度累积把全局 batch 补回原来的大小, 训练就能接着跑。m27 把全局 batch 48 固定住:\n- 8 张卡 × 每卡 1 个 micro-batch × 6 条。\n- 6 张卡 × 每卡累积 2 次 × 4 条。\n两者的全局平均梯度相差 2.24e-08, 都与单卡整批吃 48 条相差 2.61e-08 以内, 也就是在 fp32 噪声内相同。但它们不是逐位相同: 份数变了, 求和顺序跟着变, 而浮点加法不满足结合律。\nm08 的"挂掉重启等于从未挂过"要求的正是逐位相同, 靠的是连求和顺序都不变。缩容之后这条保证就断了: loss 曲线还能接着走, 但不能再靠回滚加重放去复现某一次 spike。真实框架还要做 checkpoint resharding (把 ZeRO 分片和 TP / PP 拓扑重新切一遍), m27 只重切数据和累积步数。',
            },
            {
              title: 'DiLoCo: 本地跑 H 步, 外层拿参数差当梯度',
              body: 'DiLoCo (Douillard et al., arXiv 2311.08105) 让每个 worker 用自己的数据分片连着跑 H 步 AdamW, 只在每 H 步末尾同步一次:\n1. 各 worker 从同一份外层参数 $\\theta^{(t-1)}$ 出发, 各跑 H 步, 得到 $\\theta_i^{(t)}$。内层优化器的状态留在本地, 跨轮次不清空。\n2. 把 $\\theta^{(t-1)} - \\theta_i^{(t)}$ 在 worker 之间取平均, 得到伪梯度 $\\Delta^{(t)}$。方向是旧参数减新参数, 这样它才和真梯度同向。\n3. 外层用带 Nesterov 动量的 SGD 把 $\\Delta^{(t)}$ 当梯度走一步 (论文用学习率 0.7、动量 0.9)。\n同步的东西从梯度换成了参数, 所以通信次数能降到 1/H, 而不只是把梯度攒起来。把 H 设成 1、内层换成朴素 SGD、外层学习率设 1 且无动量时, 第 3 步化简成 $\\theta \\leftarrow \\bar\\theta$, 外层变成恒等映射, 而 $\\bar\\theta$ 正是 DDP 走一步会到的地方: m28 用这组参数和手写的 DDP 对表, 差 1.79e-07, 这个差来自 $\\theta - 1.0 \\cdot (\\theta - \\bar\\theta)$ 在 fp32 下不是逐位的 $\\bar\\theta$。',
            },
            {
              title: 'H 的两头都坏: 太小被动量放大, 太大跟不上',
              body: '外层的有效步长约为 $\\mathrm{lr}/(1-\\mu) = 0.7/0.1 = 7$ 倍, 也就是本地走完的那一段又被放大 7 倍走了一遍。m28 在 16 维回归、4 个 worker、内层 AdamW 学习率 0.02、总共 960 步上扫 H (留出集 MSE, 全 0 预测的参照是 4.46):\n- H = 1 / 4: 1.18 / 1.33, 同步次数和 DDP 一样多却被这 7 倍放大走坏了。\n- H = 16 / 48: 3.6e-10 / 1.4e-11, 通信分别比 DDP 少 16 倍和 48 倍。\n- H = 120 / 240 (只同步 8 次和 4 次): 2.7e-04 / 2.9e-03, worker 之间漂得太远, 平均一下补不回来。\n- DDP 每步同步: 9.9e-14, 在这个玩具上仍然最准。\n所以 H 有一个中间的好区间, 不是越大越省。这里最好的 H 是 48, 这个数字由玩具问题的条件数和学习率决定, 论文在 1.5 亿参数、8 个 worker 上取的是 500。',
            },
            {
              title: 'Streaming DiLoCo 降峰值带宽; 玩具规模的边界',
              body: 'Streaming DiLoCo (arXiv 2501.18512) 把参数分成若干片轮流同步, 每片仍然每 H 步被同步一次, 但同步事件错开, 单次只发其中一片。m28 分 2 份和 4 份时单次字节从 408 降到 204 和 102, 总通信量三档都是 8160 B/rank, 一点没变 —— 省的是峰值带宽, 不是总量。原文按 transformer 的层分片 (默认每 3 层一片, 24 层模型峰值带宽降 8 倍), 还叠加了同步期间继续训练和把外层梯度量化到 4 位 (E3M0), 总比特数才降两个数量级。\n这一章的数字都来自玩具规模, 不能外推:\n- m28 的模型是一层线性回归, 272 字节, 数据严格 IID。论文在 1.5 亿参数、C4 语料、8 个 worker 上取 H = 500, 通信次数是每步同步的 1/500, 困惑度与每步同步持平, 还测了非 IID 分片的鲁棒性。\n- m27 的故障按单一指数分布的泊松过程到达。真实集群有相关故障: 一个机架掉电、一次固件升级会同时带走很多卡, 这时实际 goodput 比模型低。Llama 3 405B 的 54 天预训练快照里有 466 次中断, 其中 47 次计划内、419 次意外, 报告的有效训练时间高于 90%, 靠的是自动化恢复。\n- 单进程模拟, worker 之间不是真的异步, 慢 worker 不阻塞别人这件事没有建模。',
            },
          ],
          links: [
            { from: 'm08 的逐位续训', to: '弹性缩容', body: '参数、优化器状态、数据游标和 RNG 齐全时, 重启等于从未挂过; 一旦卡数变了, 求和顺序跟着变, 这条逐位保证就只剩 fp32 噪声内相同。' },
            { from: 'm01 梯度累积', to: '保持全局 batch', body: '缩容后每张卡多累积几次, 全局 batch 不变, 学习率和调度都不用改。各份样本数相同时累加等价于整批取均值, 不等长时要按样本数加权。' },
            { from: 'm02 的每步 all-reduce', to: 'DiLoCo 的每 H 步一次', body: 'DDP 同步的是梯度, DiLoCo 同步的是参数; 后者把 H 步的位移合成一个伪梯度交给外层优化器, 所以通信次数能降到 1/H 而不只是把梯度攒起来。' },
            { from: 'm10 的坏 step 防护', to: 'SDC', body: 'NaN guard 只拦非有限值。SDC 产生的是一个有限、量级正常的数, 它绕过这道检查, 只能靠副本间对比或确定性重算发现。' },
          ],
          sourceRows: [
            { concept: '整体 MTBF', code: 'm27_fault_tolerance/fault_tolerance.py:cluster_mtbf', takeaway: '单卡故障率 1/M 相加得 N/M, 倒过来就是 M/N。指数分布下这是精确结论, 不是近似。' },
            { concept: 'Young / Daly 间隔', code: 'm27_fault_tolerance/fault_tolerance.py:daly_interval', takeaway: '$\\sqrt{2\\delta(M+R)} - \\delta$; 落盘比 $2(M+R)$ 还长时式子给不出有意义的间隔, 返回无穷。' },
            { concept: 'goodput 解析式', code: 'm27_fault_tolerance/fault_tolerance.py:goodput_model', takeaway: '分母是保住 τ 秒有效进度要花的墙钟时间: 期望重试次数乘上每次重启的代价。' },
            { concept: '缩容后重切', code: 'm27_fault_tolerance/fault_tolerance.py:accumulate_grads', takeaway: 'n_rank 张卡各累积 per_rank/micro 次再跨卡平均; 整除关系不成立就得先改 micro。' },
            { concept: 'DiLoCo 一轮', code: 'm28_diloco/diloco.py:run_diloco', takeaway: '每轮先把外层参数拷给所有 worker, 跑 H 步内层 AdamW (状态不清空), 再平均参数走一步外层。' },
            { concept: '外层那一步', code: 'm28_diloco/diloco.py:outer_step', takeaway: '伪梯度是旧参数减平均后的新参数; Nesterov 把动量再往前推一步, 有效步长约 lr/(1-μ)。' },
          ],
          snippetTitle: '两件事: 间隔怎么算, 同步怎么少',
          snippet: `# m27: 最优 checkpoint 间隔 = 两笔开销相等的那一点
tau = sqrt(2 * delta * mtbf)                 # Young: delta/tau == tau/(2M)
tau = sqrt(2 * delta * (mtbf + restart)) - delta   # Daly: 把重启算进去

goodput = tau / (M * exp(R / M) * (exp((tau + delta) / M) - 1))

# m28: DiLoCo 的一轮
for worker in workers:
    worker.load_params(outer)                # 都从同一份外层参数出发
for _ in range(H):                           # 本地跑 H 步, 期间零通信
    for w, stream in zip(workers, streams):
        inner_adam(w, stream, lr_in)

avg   = all_reduce_mean([w.params for w in workers])   # H 步才一次
delta = outer - avg                          # 伪梯度: 旧参数 - 新平均参数
buf   = mu * buf + delta
outer = outer - lr_out * (delta + mu * buf)  # lr_out=1, mu=0 时退回 DDP`,
          source: [
            'llm_train/m27_fault_tolerance/fault_tolerance.py:daly_interval',
            'llm_train/m27_fault_tolerance/fault_tolerance.py:goodput_model',
            'llm_train/m27_fault_tolerance/fault_tolerance.py:accumulate_grads',
            'llm_train/m28_diloco/diloco.py:run_diloco',
            'llm_train/m28_diloco/diloco.py:outer_step',
          ],
          run: 'python -m llm_train.m27_fault_tolerance.demo',
          related: [
            { route: 'train-collectives-loop', label: '完整训练主循环', reason: '本章的 checkpoint 间隔和弹性缩容都建立在主循环已经把状态存全的前提上; 分片 checkpoint 怎么存、怎么恢复, 见那一章。' },
            { route: 'train-batch-ddp', label: 'DDP 与梯度累积', reason: 'DiLoCo 的对照基线是每步 all-reduce 的 DDP; 缩容时用来补回全局 batch 的也是那一章的梯度累积。' },
            { route: 'train-eval', label: '评测口径', reason: 'goodput 是训练侧的口径, 和模型好不好无关; 两类数字不要混在同一张汇报表里。' },
          ],
    },
  },
}

// 机制实验的原理、运行入口与源码引用。
const supplements = {
  "train-model-parallel": [
    {
      "title": "三种并行要放进同一张预算表",
      "body": "大规模集群训练往往同时组合 DP、TP 与 PP (三路并行度乘积等于总卡数)。单看参数放不放得下无法判断哪套切分跑得最快, 因此 m22 将单卡显存、集合通信与流水线气泡放进同一张代价表里联合估算。\n固定 1B 参数、24 层、全局批大小 64、8 张卡的场景, 对候选 DP×TP×PP 组合依次核算四步: 先校验总卡数匹配与 hidden/TP、层数/PP、batch/(DP×micro) 三项整除约束 (如 DP2×TP3×PP2 需 12 卡会被直接拒绝); 再统计单卡模型状态与在途激活显存; 接着按 消息条数 × 启动延迟 α + 传输字节 / 带宽 计算通信耗时; 最后结合 Roofline 计算耗时、不可隐藏通信与 PP 气泡倍率给出单步墙钟时间:\n- DP2×TP2×PP2: 单卡状态 3.725 GiB, 单步约 708 ms, 关键卡 (代价模型据以计算步耗时的那张卡) 每步发出 786 条通信消息。\n- DP1×TP4×PP2: 单卡状态减半至 1.863 GiB, 单步耗时却反升至 975 ms (消息激增至 4640 条)——TP 跨层高频同步累加了大量固定启动延迟 $\\alpha$, 吃掉了省显存的收益; 若网络延迟 $\\alpha$ 从 8 μs 恶化到 100 μs, 单步耗时更会拉长至 1429 ms。\n需注意这里的步耗时与 MFU (0.694 对 0.504) 均出自解析代价模型, 用于开训前筛除明显失衡的拓扑配置, 而非真机算子实测。",
      "source": "llm_train/m22_parallel_budget/demo.py:estimate"
    }
  ],
  "train-collectives-loop": [
    {"title":"本章核验通信语义等价, 不估算硬件耗时", "body":"本章聚焦集合通信原语的数值语义与张量分片契约: 每次调用后哪些 rank 参与归约、各卡持有的分片如何对齐。m09 在 4 张卡、32 B 张量上逐项断言: reduce_scatter 每卡发 24 B、all_gather 每卡发 24 B, 串联后字节数与求和结果均与 ring_all_reduce_sum (48 B) 严格一致; 至于不同并行拓扑的显存与耗时权衡, 则归入「TP / PP 切模型」一章的联合预算表。", "source":null}
  ]
}
for (const [route, rows] of Object.entries(supplements)) {
  const page = topic.pages[route]
  if (!page) throw new Error(`缺少补充章节: ${route}`)
  for (const { title, body, source } of rows) {
    page.points.push({ title, body })
    if (source) page.source = [...new Set([].concat(page.source || [], source))]
  }
}

// 审查后的主讲职责与前置链接。
topic.pages["train-collectives-loop"].related = [{"route": "train-model-parallel", "label": "三种并行组合: 显存与时间一起算", "reason": "本章聚焦四个集合通信原语的数值语义与端到端训练主循环。至于 DP、TP 与 PP 如何正交组合, 以及不同拓扑下的单卡显存占用与单步耗时估算, 则归入「TP / PP 切模型」一章的并行预算表。"}]
topic.pages["train-model-parallel"].related = [{"route": "infer-compute", "label": "从估算到实测: profiler 怎么用", "reason": "本章的并行预算表由解析代价模型估算得出, 仅用于在开训前筛除明显失衡的拓扑配置。定位真实运行时的性能瓶颈, 仍需借助 profiler 采集每个算子的调用 trace 与耗时分布, 详见该章的实测方法。"}]
export default topic

mergePage(topic, 'train-low-precision', 'train-precision-stability', '进阶 · 分块缩放 (block scale) 与 FP4', ["TrainBlockScaleLab"])

topic.pages['train-precision-stability'].title = '低精度训练 · 稳定性、缩放与 FP4'
const precision = topic.pages['train-precision-stability']
precision.sections[0].points[0].body = '低精度浮点能覆盖的区间极窄, 量化前需按一组共享元素里的最大绝对值 amax 算出缩放系数 scale = amax / max_val, 将数值映射进网格后再就近舍入 (注意这里的张量量化 scale 按数据块现算, 与前文放大标量损失的 loss scaling 不是同一概念)。\n当块内混入个别幅值极高的离群值 (outlier) 时, 它会独自撑大 amax, 迫使同块正常元素除以大数后挤向 0 附近; 分块尺寸切得越细, 离群值波及的邻居就越少。FP8 下离群值达到 1e5 倍时, 整张量共享 scale 的相对误差为 0.126, 按 128 元素分块则压回 0.026; 再降到非零动态范围仅 12 倍的 FP4 ($\pm\\{0.5, 1, 1.5, 2, 3, 4, 6\\}$), 分块粒度更必须收缩到 16~32 个元素才能装下真实权重与激活。'
topic.pages['train-model-parallel'].points.find(p => p.title.startsWith('PP:')).body = '流水线并行 (PP) 将模型按连续层切成 PP 个 stage 各放一张卡, 卡间只在 stage 边界传递单层激活, 跨卡流量远小于每层都要做四次 all-reduce 的 TP。\n它的代价体现在首尾空等与显存驻留两方面: 一个 batch 从首卡流向末卡再反向传回期间, 未排满的空闲占比称为气泡; 将 batch 拆成 M 个 micro-batch 连续送入后可把空闲收束到首尾 (m04 中 PP=4, M=8 时气泡占比为 27.3%)。同时, 已跑完前向但尚未等回反向的 micro-batch 必须将激活留在显存里 (称为在途激活):\n- GPipe 先跑完全部 M 个前向再统一反向, 每张卡峰值都要驻留 M 份激活。\n- 1F1B 尽早穿插前反向, 将第 s 张卡的在途激活峰值压到 PP - s 份, 但气泡比例本身与 GPipe 相同。\n具体的气泡公式、排程时间表与交错调度代价将在下一章「1F1B 与交错调度」中详细展开。'

for (const chapter of topic.chapters) chapter.label = chapterLabels[chapter.route] || chapter.label

canonicalizeRelated(topic)

// 补充段: 进阶 · Zero Bubble 与 DualPipe: 把反向拆成两半
topic.pages['train-pipeline-schedules'].sections = [...(topic.pages['train-pipeline-schedules'].sections || []), {
  id: 'zero-bubble',
  title: '进阶 · Zero Bubble 与 DualPipe: 把反向拆成两半',
  tldr: '完整反向其实是两件事: 算对输入的梯度 (B) 和算对权重的梯度 (W)。B 要立刻传给上一个 stage, W 谁都不等它:\n- ZB-H1: W 往后挪去填本来空着的格子, 每卡空闲从 $(\\mathrm{PP}-1)(t_f+t_b+t_w)$ 降到 $(\\mathrm{PP}-1)(t_f+t_b-t_w)$, 三者耗时相等时正好是 1/3。在途激活峰值仍是 PP 份, 和 1F1B 相同。\n- ZB-H2 / ZB-V: 前者允许峰值升到 $(2\\mathrm{PP}-1)$ 份换零气泡, 后者把模型切成 $2\\mathrm{PP}$ 块按 V 字形分配, 在 1F1B 的显存下做到零气泡。\n- DualPipe: 两端同时喂 micro-batch, 用一对前向与反向块的计算掩盖通信; 代价是每卡存两份参数。',
  question: '1F1B 的气泡来自首尾那两段必须空等的时间, 既然这段时间卡是闲着的, 为什么不拿点活来填? 哪一部分反向计算可以被挪到这里, 哪一部分不行?',
  labNames: ['TrainZeroBubbleLab'],
  evolution: {
    title: '从填不满到填得满',
    subtitle: '根问题: 流水线首尾两段里一张卡要等上游的前向或下游的梯度, 这段时间它手上没有能做的活。',
    steps: [
      { name: '1F1B 的三角形空白', pain: '(原点) 一个 micro-batch 的反向要等下游的反向传回来, 等的时候这张卡只能闲着', fix: '尽早穿插前反向把空闲收束到首尾; 气泡仍是 $(\\mathrm{PP}-1)(t_f+t_b+t_w)$, PP=4、M=8 时占 27.3%' },
      { name: '拆出 W', year: 2024, pain: '完整反向被当成一个不可分的操作, 它里面那部分谁都不等的计算也被绑在关键路径上', fix: 'Qi et al. 把反向拆成算输入梯度的 B 和算权重梯度的 W; B 仍在关键路径上, W 可以任意往后挪' },
      { name: 'ZB-H1', pain: '拆开之后 W 如果紧跟着 B 做, 时间表和原来一样长, 等于白拆', fix: 'W 只在这张卡没别的事可做时插进去, 填掉收尾的空白; 每卡空闲降到 $(\\mathrm{PP}-1)(t_f+t_b-t_w)$, 显存峰值不变' },
      { name: 'ZB-H2 / ZB-V', pain: '首段那 $(\\mathrm{PP}-1)$ 格仍然空着, 因为开头还没有任何 W 可填', fix: '热身多做几个 F 把首段也填上 (ZB-H2, 峰值升到 $(2\\mathrm{PP}-1)$ 份), 或把模型切成 $2\\mathrm{PP}$ 块按 V 字形分配 (ZB-V, 峰值不变)' },
      { name: 'DualPipe', year: 2024, pain: '跨节点专家并行的 all-to-all 让计算与通信的比例接近 1:1, 光排计算填不满', fix: 'DeepSeek-V3 两端同时喂 micro-batch, 用一对前向与反向块的计算掩盖 all-to-all 和 PP 通信; 每卡存两份参数' },
    ],
  },
  points: [
    {
      title: 'B 在关键路径上, W 不在',
      key: true,
      body: '完整反向做的是两件事: 算对这一层输入的梯度, 和算对这一层权重的梯度。前者是上一个 stage 的输入, 挪晚了整条流水线跟着等; 后者只在这一步的优化器更新之前要用到, 谁都不等它。Zero Bubble (Qi et al., ICLR 2024) 把它们分别记作 B 和 W, 前向记作 F。\nm04 的事件模拟保持 1F1B 的 F/B 顺序不变, 只让 W 在这张卡没别的事可做时插进去。F / B / W 各占 1 格、PP=4、M=8 时:\n- 1F1B (完整反向占 2 格): 总时长 33 格, 每卡空闲 9 格, 气泡占比 27.3%。\n- ZB-H1: 总时长 27 格, 每卡空闲 3 格, 气泡占比 11.1%。\n每卡空闲的两条闭式是 $(\\mathrm{PP}-1)(t_f+t_b+t_w)$ 和 $(\\mathrm{PP}-1)(t_f+t_b-t_w)$: $t_w$ 从加号挪到了减号。模拟在 5 组 $(\\mathrm{PP}, M)$ × 4 组耗时上逐一核对过这个式子, 也逐个 micro-batch 核对了依赖合法性 (W 在自己的 B 之后、B 在下一个 stage 的 B 之后、F 在上一个 stage 的 F 之后)。',
    },
    {
      title: '"降到 1/3" 带着一个前提',
      body: '论文表 2 给的是绝对空闲格数而不是比例, 1/3 这个说法来自把 $t_f = t_b = t_w$ 代进去: $(1+1-1)/(1+1+1) = 1/3$。论文在导出这组手工调度时明确说了先假设三者耗时相同。一般情况下比值是 $(t_f+t_b-t_w)/(t_f+t_b+t_w)$, 它对 $t_w$ 很敏感:\n- $t_b$ 占 1 格时 (三者相等): 1F1B 空闲 9 格, ZB-H1 空闲 3 格, 比值 1/3。\n- $t_b$ 占 3 格时: 15 格对 9 格, 比值 3/5。\n论文实测的比例是 $T_W \\lt T_F \\lt T_B$ 且 $T_B + T_W = 2T_F$, 所以真实收益小于 1/3 这个理想值。\n另一个容易读错的地方: 气泡**占比**从 27.3% 降到 11.1%, 比值是 0.41 而不是 0.33, 因为占比的分母是总格数。每卡空闲确实是 9 → 3 格 (正好 1/3), 但总格数也从 33 降到 27。要比的是总格数, 不是占比。',
    },
    {
      title: '显存: 在途份数每卡都满, 峰值却不变',
      body: '1F1B 里第 s 张卡的在途激活峰值是 $\\mathrm{PP}-s$ 份 (PP=4 时 [4,3,2,1]), 因为反向一做完就把激活释放了。ZB-H1 把 W 推迟了, 而激活要等 W 用完才能放, 所以每张卡都攥满 PP 份 (实测 [4,4,4,4])。\n这不是显存涨了 $\\mathrm{PP}/(\\mathrm{PP}-s)$ 倍。论文把每卡显存写成 $(\\mathrm{PP}-i+1)M_B + (i-1)M_W$: 还没反向的那部分要留完整的中间量 $M_B$, 做完 B 只等 W 的那部分只要 $M_W$, 后者更小。卡 0 上 PP 份全是大的那种, 所以峰值仍由卡 0 决定, 等于 1F1B 的 $\\mathrm{PP} \\cdot M_B$ —— 这是 ZB-H1 的设计目标: 在不超过 1F1B 峰值的前提下填气泡。m04 的模拟只数份数, 不区分 $M_B$ 和 $M_W$。\n想再往下压气泡就得放开这个约束。ZB-H2 在热身阶段多做几个 F 把首段也填上, 空闲降到 $(\\mathrm{PP}-1)(t_f+t_b-2t_w)$ (三者相等时为 0), 代价是峰值升到 $(2\\mathrm{PP}-1)M_B$; ZB-V 把模型切成 $2\\mathrm{PP}$ 块、按"第一张卡到最后一张卡再折回来"的 V 字形分配, 在 $\\mathrm{PP} \\cdot M_B$ 的显存下也做到零气泡。',
    },
    {
      title: 'DualPipe: 按报告的代价公式算, 本章不排它的时间表',
      body: 'DeepSeek-V3 技术报告的动机是跨节点专家并行: all-to-all 的开销让计算与通信的比例接近 1:1。DualPipe 把每个块切成注意力、all-to-all dispatch、MLP、all-to-all combine 四段 (反向的注意力和 MLP 再按 B / W 拆开), 重排这些片段并手工分配通信与计算各占多少 SM, 让 all-to-all 和 PP 通信都被掩盖; 同时两端同时喂 micro-batch, 走双向流水线。\n报告表 2 的三行 (F 是一次前向, B 是一次完整反向, W 只算权重梯度, F&B 是一对合并执行的前向与反向块):\n- 1F1B: 空闲 $(\\mathrm{PP}-1)(F+B)$, 参数 1 份, 在途激活 PP。\n- ZB1P: 空闲 $(\\mathrm{PP}-1)(F+B-2W)$, 参数 1 份, 在途激活 PP。\n- DualPipe: 空闲 $(\\mathrm{PP}/2-1)(F\\&B+B-3W)$, 参数 2 份, 在途激活 PP+1。\n两篇论文的 B 不是一回事: 报告的 B 是完整反向, 所以 ZB1P 那一行换成本段的符号就是 $(\\mathrm{PP}-1)(t_f+t_b-t_w)$, 也就是 ZB-H1。m04 按 F=1、B=2、W=1、F&B=2 代入, PP=8 时每卡空闲依次是 21 / 7 / 3 格。这是代价模型不是事件模拟, 双向时间表 m04 没有排。',
    },
    {
      title: '这些数字是确定性模拟, 不是实测',
      body: 'm04 整个模块都是时间表仿真: 不真的算梯度, 跨卡 P2P 传输时间记为 0, 每个 stage 的耗时假设相同。拆出 W 之后多出来三件现实代价:\n- 通信: W 不产生跨卡消息, 但 B 和 W 分开调度意味着优化器更新的同步点变多; ZB-H2 还要去掉优化器步之间的同步。\n- 调度: m04 的 W 插入用的是贪心规则 (没别的事可做, 或者在途满了就补一个 W), 不是论文第 3 节那个按运行时 profile 解整数规划的自动调度器。真实负载下各 stage 耗时不均 (embedding 和输出头), 手工调度未必最优。\n- 实现: 把反向拆成两半要框架支持只算输入梯度或只算权重梯度的反向调用, 这是接入成本最高的一块。DualPipe 另外要求 stage 数和 micro-batch 数都能被 2 整除 (但不要求 micro-batch 数被 stage 数整除), 并且每卡存两份参数。\nZero Bubble 论文自己报的端到端收益是: 同等显存下吞吐比 1F1B 高最多 15%, 放开显存约束后 30% (这是 ICLR 发表版的数字; arXiv 扩展版写的是 23% 和 31%)。ZB-V 只在 arXiv 扩展版里有, ICLR 发表版没有这一节, 后续论文是 arXiv 2405.15362。',
    },
  ],
  sourceRows: [
    { concept: 'W 插空的规则', code: 'm04_pipeline_parallel/demo.py:simulate_zb', takeaway: 'F/B 顺序沿用 device_order("1f1b"); W 在卡空闲或在途满 PP 份时插入, 函数内逐个 micro-batch 断言依赖合法。' },
    { concept: '在途份数的口径', code: 'live[s] += 1 if kind == "F" else -1', takeaway: 'F 加一份、W 减一份 (B 不释放)。1F1B 是 B 就释放, 所以得到 [PP-s]; ZB-H1 每卡都是 PP。' },
    { concept: '显存上限', code: 'full = kind == "F" and live[s] >= cap', takeaway: 'cap 默认取 PP, 也就是 1F1B 里 stage 0 的峰值; 满了就先还一个 W 再收新的 F。' },
    { concept: 'DualPipe 代价公式', code: 'm04_pipeline_parallel/demo.py:dualpipe_cost', takeaway: '直接实现报告表 2 的三行, 返回 (每卡空闲, 参数份数, 在途激活); 不做事件模拟。' },
  ],
  snippet: `# 1F1B 的 F/B 顺序一个字没改, 只是 B 之后不立刻做 W
order = device_order("1f1b", s, PP, M)     # [("F",0), ("F",1), ..., ("B",0), ...]

for t in count():                          # 一格一格往前走
    if free[s] > t: continue               # 这张卡还在忙
    op = next_ready(order, ptr[s])         # F 等上游的 F, B 等下游的 B
    if op is None or (op.kind == "F" and live[s] >= PP):
        op = ("W", pending[s].pop(0))      # ★ 没活干, 或者显存满了 -> 补一个 W
    run(op); free[s] = t + dur[op.kind]
    if op.kind == "B": pending[s].append(op.m)   # B 不释放激活
    if op.kind == "W": live[s] -= 1              # W 用完了才释放

# 每卡空闲: 1F1B (PP-1)(t_f+t_b+t_w)  ->  ZB-H1 (PP-1)(t_f+t_b-t_w)`,
  source: [
    'llm_train/m04_pipeline_parallel/demo.py:simulate_zb',
    'llm_train/m04_pipeline_parallel/demo.py:dualpipe_cost',
  ],
  run: 'python -m llm_train.m04_pipeline_parallel.demo',
  links: [
    { from: 'device_order', to: 'simulate_zb', body: 'ZB-H1 没有换顺序表: 它拿的就是 1F1B 那张 F/B 顺序表, 区别只在 B 之后不立刻做 W, 以及空闲时拿一个欠着的 W 来填。' },
    { from: '交错 1F1B 的气泡 ÷ v', to: 'ZB-H1 的空闲减一个 t_w', body: '交错靠把每格切小来缩短首尾的等待, 代价是 v 倍的 P2P 次数和更多在途激活; 拆 W 不改变流水线形状, 只把一部分计算挪到空闲处, 显存峰值不变。两者可以叠加。' },
    { from: 'm11 专家并行的 all-to-all', to: 'DualPipe', body: '跨节点专家并行让 all-to-all 成为主要开销, 光排计算顺序填不满气泡; DualPipe 要掩盖的正是这部分通信, 所以它和 MoE 一起出现。' },
  ],
  related: [
    { route: 'train-moe-seq', label: '专家并行的 all-to-all' },
    { route: 'train-memory', label: '在途激活与激活重算' },
  ],
}]

// 补充段: 进阶 · z-loss: 交叉熵管不住的那个自由度
topic.pages['train-precision-stability'].sections = [...(topic.pages['train-precision-stability'].sections || []), {
  id: 'z-loss',
  title: '进阶 · z-loss: 交叉熵管不住的那个自由度',
  tldr: 'softmax 给所有 logit 加同一个常数, 输出概率完全不变, 所以交叉熵在"整体高度"这个方向上的梯度是 0。这个方向没人管, $\\log Z$ 就会一路往上漂:\n- 现象: m10 的 16 类分类器在学习率偏大时练 3000 步, 平均 $\\log Z$ 漂到 21.85, 最大 logit 44.31。\n- 修法: 损失里加一项 $c_z \\log^2 Z$ (PaLM 取 $10^{-4}$)。同一组设置下 $\\log Z$ 降到 10.93, 加到 $10^{-3}$ 时降到 1.50。\n- 代价: 交叉熵本身从 0.0018 升到 0.0159, top-1 正确率三档都是 1.000 —— 它只压整体高度, 不改 logit 之间的差。\n- 同一个式子用在 MoE 路由器上就是 ST-MoE 的 router z-loss, 系数取 $10^{-3}$。',
  question: '交叉熵已经在训练输出头了, 为什么还需要另一项损失来管 logits 的大小? 把 logits 压小会不会改变模型选哪个 token?',
  points: [
    {
      title: '整体高度是个自由参数',
      key: true,
      body: 'softmax 的定义是 $p_i = \\exp(z_i) / \\sum_j \\exp(z_j)$。给每个 $z_i$ 都加上同一个常数 $c$, 分子分母同乘 $e^c$, $p$ 一点不变。交叉熵只看 $p$, 所以它对"所有 logit 一起上下移动"这个方向的梯度恰好是 0 —— 这个自由度训练过程完全不管。\n不管的结果不是停着不动, 而是被别的因素推着走。数据可分时交叉熵会一直把胜出的那个 logit 往上顶, 而 $\\log Z = \\log \\sum_j \\exp(z_j)$ 在分布很尖时约等于最大的那个 logit, 于是它跟着一路涨。m10 用一个 16 类的 softmax 分类器 (可分数据, 练 3000 步) 量这件事:\n- 学习率 0.5: 平均 $\\log Z$ 13.36, 最大 logit 26.95。\n- 学习率 8.0 (偏大): 平均 $\\log Z$ 21.85, 最大 logit 44.31。\n学习率越大漂得越远, 而且不会自己停下来。',
    },
    {
      title: 'z-loss 的梯度按 softmax 分摊, 所以不改排序',
      body: 'PaLM (arXiv 2204.02311) 在损失里加一项 $c_z \\log^2 Z$, 系数取 $10^{-4}$。这一项在 $\\log Z = 0$ 处最小, 也就是把 $Z$ 往 1 拉。它的梯度是:\n$$\\frac{\\partial \\log^2 Z}{\\partial z_j} = 2 \\log Z \\cdot \\mathrm{softmax}_j$$\n读法: 每个 logit 受到的下压力正比于它自己的 softmax 概率, 而比例系数 $2 \\log Z$ 对所有 logit 相同。概率大的被压得多、概率小的被压得少, 整体往下挪, logit 之间的差基本不动。\nm10 在学习率 8.0 下的三档对照 (系数 0 / $10^{-4}$ / $10^{-3}$):\n- 平均 $\\log Z$: 21.85 / 10.93 / 1.50。\n- 最大 logit: 44.31 / 23.55 / 10.45。\n- 交叉熵 (不含 z-loss 那一项): 0.0018 / 0.0082 / 0.0159。\n- top-1 正确率: 1.000 / 1.000 / 1.000。\n三个种子重复, $\\log Z$ 从 21.8 / 21.8 / 21.5 降到 10.9 / 11.2 / 11.5, 方向一致。学习率 0.5 时漂移本来就小, z-loss 的效果也小 (13.36 → 12.46 → 8.03)。',
    },
    {
      title: 'router z-loss: 同一个式子, 换个地方用',
      body: 'ST-MoE (arXiv 2202.08906) 把同一项用在 MoE 路由器的 logits 上, 称为 router z-loss, 系数取 $10^{-3}$。路由器的 logits 要过一次 softmax 决定每个 token 送给哪些专家, 它同样有整体高度这个自由度, 也同样会漂。\nm10 用一个 8 个专家的路由器重复这个实验 (学习率 8.0, 3000 步):\n- 不加: 平均 $\\log Z$ 23.06, 最大 logit 47.37。\n- 加 $10^{-3}$: 平均 $\\log Z$ 0.60, 最大 logit 10.98。\n- 两档的 top-1 专家选对比例都是 1.000。\n路由器的系数比输出头大一个数量级, 因为路由决策一旦被数值误差带偏, 整个 token 就被送到错误的专家, 比输出头多一位小数的概率误差要紧得多。\n真实 MoE 训练里 router z-loss 通常和负载均衡损失一起加 (两件不同的事: 一个管 logits 的大小, 一个管 token 在专家间的分布)。',
    },
    {
      title: '为什么在意 log Z: 存不下和算不准',
      body: 'softmax 的实现都会先减掉最大值再取指数, 所以 $Z$ 本身不会溢出。真正的问题在那之前: logits 要先被存下来、被搬运、被归约。\nm10 顺手算了两个阈值: $\\exp$ 在 fp16 下超过 65504 就溢出, 对应 logit 11.1; bf16 的上界是 3.39e38, 对应 logit 88.7。无界增长的 $\\log Z$ 迟早会撞上其中一个, 撞上就是一个 Inf 加一步作废 (这正是上文 NaN guard 要拦的那种坏 step)。PaLM 报告的收益是训练更稳、loss spike 更少。\n边界照写: 把 m10 训练好的 logits 舍入到 bf16 再算 softmax, 三档的最大概率偏差都在 1e-3 量级, 并没有随 $\\log Z$ 一起降下来 —— 加了 z-loss 之后分布更平, 可被扰动的概率质量反而更多。所以在这个 16 类玩具的规模上, 能证明的只有"$\\log Z$ 确实会漂、z-loss 确实拉得住、代价是交叉熵变差"这三条; "更少 spike" 是论文在真实预训练上的结论, 3000 步的分类器看不到。',
    },
    {
      title: '它和这一章其他三件事的分工',
      body: '这一章的四件工具管的是四个不同的东西, 叠加使用:\n- loss scaling: 让 fp16 的小梯度不下溢, 在反向之前放大损失、更新之前除回。\n- fp32 主权重: 让小于低精度格点间距的更新不被舍掉。\n- 全局范数裁剪: 偶发的梯度尖峰只缩长度、不改方向。\n- z-loss: 盯住 logits 的整体高度这一个标量, 不让它无界增长。\n容易混的两组:\n- z-loss 和 weight decay: 后者压的是权重范数, 前者只对 $\\log Z$ 这一个量施压。权重范数小不代表 logits 不漂 (logits 还取决于激活的大小)。\n- z-loss 和 QK-norm / QK-clip (见 Muon 一章): 两者都在防 softmax 饱和, 但位置不同 —— z-loss 在输出头或路由器, QK-clip 在注意力分数。注意力那边压的是 $W_Q$ 和 $W_K$ 的乘积, z-loss 压的是 logits 本身。',
    },
  ],
  sourceRows: [
    { concept: 'z-loss 与它的梯度', code: 'm10_training_stability/demo.py:z_loss', takeaway: '返回 (惩罚值, 对 logits 的梯度); 梯度是 $2 c_z \\log Z \\cdot \\mathrm{softmax}$, 按概率分摊到每个 logit。' },
    { concept: 'log Z 怎么算', code: 'm10_training_stability/demo.py:log_partition', takeaway: '先减每行最大值再取指数求和, 只为防 exp 溢出, 加回最大值后结果不变。' },
    { concept: '三档对照', code: 'm10_training_stability/demo.py:train_softmax', takeaway: '同一个函数既当输出头 (16 类) 又当路由器 (8 个专家), 两种用法只差 n_class 和系数。' },
    { concept: '唯一的改动', code: 'd_logits = d_logits + z_loss(logits, coef)[1]', takeaway: '交叉熵那部分一个字没改, z-loss 只是在对 logits 的梯度上加一项。' },
  ],
  snippet: `def log_partition(logits):
    m = logits.max(axis=-1, keepdims=True)       # 减最大值只为防 exp 溢出
    return (m + log(exp(logits - m).sum(-1, keepdims=True)))[..., 0]

def z_loss(logits, coef):
    lz = log_partition(logits)                   # [batch]
    grad = coef * 2.0 * lz[:, None] / batch * softmax(logits)
    return coef * mean(lz ** 2), grad            # ★ 梯度按 softmax 分摊

# 训练循环里只多一行
d_logits = (probs - onehot) / batch              # 交叉熵, 不变
d_logits = d_logits + z_loss(logits, c_z)[1]     # c_z: 输出头 1e-4, 路由器 1e-3`,
  source: [
    'llm_train/m10_training_stability/demo.py:z_loss',
    'llm_train/m10_training_stability/demo.py:log_partition',
    'llm_train/m10_training_stability/demo.py:train_softmax',
  ],
  run: 'python -m llm_train.m10_training_stability.demo',
  links: [
    { from: 'NaN guard', to: 'z-loss', body: 'guard 是事后拦截: 检出 Inf 就作废这一步。z-loss 是事前约束: 不让 logits 涨到 exp 溢出的那条线上去, 于是坏 step 本身变少。' },
    { from: '全局范数裁剪', to: 'z-loss', body: '裁剪处理的是梯度偶然变得很大, 做法是缩长度不改方向; z-loss 处理的是 logits 系统性地一路变大, 做法是在损失里加一项把它拉回来。前者是补救, 后者是预防。' },
    { from: 'MoE 的负载均衡损失', to: 'router z-loss', body: '两项都加在路由器上但管的不是同一件事: 均衡损失管 token 在专家之间分得匀不匀, router z-loss 管路由 logits 的整体高度。' },
  ],
  related: [
    { route: 'train-muon', label: 'QK-clip: 注意力那一侧的同类防护' },
    { route: 'train-moe-seq', label: 'MoE 路由与负载均衡' },
  ],
}]

// 新章与补充段的前置和延伸链接。
for (const [route, link] of [
  ['train-collectives-loop', { route: 'train-fault-tolerance', label: '容错与 goodput', reason: '主循环把状态存全之后, 下一个问题是多久存一次盘: 最优间隔由落盘耗时和整体 MTBF 算出来。' }],
  ['train-batch-ddp', { route: 'train-fault-tolerance', label: 'DiLoCo: 同步能少到什么程度', reason: 'DDP 每步 all-reduce 一遍梯度; 跨数据中心时可以让各 worker 本地跑 H 步再同步一次参数, 通信次数降到 1/H。' }],
  ['train-moe-seq', { route: 'train-pipeline-schedules', label: 'DualPipe: 用计算掩盖 all-to-all', reason: '跨节点专家并行的 all-to-all 让计算与通信接近 1:1, DeepSeek-V3 的 DualPipe 就是为掩盖这部分通信设计的, 见那一章的 Zero Bubble 段。' }],
]) topic.pages[route].related = [...(topic.pages[route].related || []), link]
