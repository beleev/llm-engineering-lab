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
  ],
  pages: {
    // ---------------------------------------------------------------- 切 batch
    'train-batch-ddp': {
      title: 'batch 与 DDP · 同一个梯度的三种算法',
      subtitle: '读完你能说清 DDP 为什么 "就是一个更大的 batch", 以及这句话在什么条件下会悄悄失效。',
      tldr: '训练一步要的是一个梯度: 把 batch 里 N 个样本各自的梯度取平均。同一个平均值有三种算法:\n- 单卡: 一张卡一次算完整个 batch。\n- 梯度累积: 把 batch 切成 $K$ 小份, 在同一张卡上一份接一份算, 梯度累加后再更新一次。用 $K$ 倍的串行时间, 换 $1/K$ 的激活显存。\n- DDP (分布式数据并行): $P$ 张卡各放一份完整模型, 各算 batch 的 $1/P$, 再把梯度平均。用每步 $2(P-1)/P$ 份梯度大小的通信, 换 $1/P$ 的时间。\n三者算出的梯度相等, 前提是每一份的样本数相同。',
      question: '梯度累积和 DDP 都在把一个 batch 切成几份算, 为什么一个不用通信、另一个每步都要把梯度在卡之间同步一次?',
      evolution: {
        title: '怎么切 batch 都得是同一个梯度',
        subtitle: '根问题: 梯度要平均足够多的样本, 一张卡的显存和算力都撑不住',
        steps: [
          { name: '单卡整 batch', pain: '(原点) 梯度要平均很多样本才稳, 激活显存随 batch 线性涨', fix: '一张卡一次算完; batch 一大就 OOM, 算力也封顶在一张卡' },
          { name: '梯度累积', pain: '整 batch 的激活一张卡放不下', fix: '拆成 micro-batch 串行算再累加; 各份已取过均值, 要乘 $n_k/N$ 才等于整 batch' },
          { name: '异步参数服务器', year: 2012, pain: '串行算 K 份太慢, 想让多张卡同时算', fix: '各卡算完就把梯度推给服务器、拉回新参数, 谁也不等谁; 梯度常基于过时的参数' },
          { name: '同步 DDP', pain: '过时梯度让一步更新对不上任何一个 batch, 结果难复现', fix: '每步 all-reduce(mean), 全体用同一个梯度更新; 每卡样本数相同就等于整 batch' },
        ],
      },
      code: 'llm_train/m01_gradient_accumulation/demo.py · llm_train/m02_data_parallel/demo.py',
      points: [
        {
          title: '三个名词: DDP、rank、all-reduce',
          body: '这一章会反复用三个词, 先说清它们是什么。\n- DDP (Distributed Data Parallel, 分布式数据并行): 有 P 张卡, 每张卡都放一份完整的模型。一个 batch 切成 P 片, 每张卡算自己那一片的梯度。代码里一张卡叫一个 rank, 编号从 0 开始。\n- all-reduce: 一种多卡之间的通信操作。每张卡交出自己的梯度, 操作结束后每张卡都拿到所有卡梯度的和。把这个和再除以卡数, 就是平均梯度。DDP 每步用它把各卡的梯度合成一个。\n- micro-batch: 同一张卡把一个 batch 拆成几小份, 一份接一份串行算, 每份算出的梯度累加起来, 累加完再更新一次参数。这就是梯度累积。',
        },
        {
          title: '切法不同, 梯度相同',
          key: true,
          body: '结论: 不管把 batch 怎么切, 只要最后算的是同一组 N 个样本的平均梯度, 结果就是同一个数。\n为什么: 一个 batch 的梯度, 定义上就是 N 个样本各自梯度的平均。平均这个运算不在乎加法的先后顺序。先算哪个样本、在哪张卡上算、分几批算, 都只是改变了求和的顺序, 不改变和。\ndemo 里的两组数字:\n- m01 (梯度累积): 把 8 个样本切成 4 份各 2 个, 串行累积出来的梯度, 与一次算完 8 个样本的梯度最大差 2.98e-08。\n- m02 (DDP): 4 张卡各算 4 个样本, all-reduce 取平均后各自更新一步, 更新后的参数与单卡一次算 16 个样本的结果最大差 3.73e-09。\n这两个差不是 0, 是因为 float32 做加法时顺序不同会留下最后几位的舍入零头。两个 demo 都用 assert 卡着这个差必须小于一个阈值。\n所以 "DDP 就是一个更大的 batch" 不是比喻, 是一个等式。',
        },
        {
          title: '累积换显存, DDP 换时间',
          body: '两种切法省的东西不一样。\n- 梯度累积省显存: 前向时每一层的中间结果 (激活) 要留到反向用, 激活占的显存和一次送进去的样本数成正比。切成 micro-batch 后, 一次只送一小份, 激活峰值按 micro-batch 算, m01 里从 8 个样本降到 2 个。代价是 K 份要一份接一份算, 时间是 K 倍; 算力和通信一点没省。\n- DDP 省时间: P 张卡同时算, 每卡的样本数除以 P, 一步的时间也大约除以 P。代价是每步要做一次覆盖全部模型梯度的 all-reduce。用 ring (环形) 算法实现时, 每张卡每步要发出 $2(P-1)/P$ 份梯度大小的数据: 8 卡是 1.75 份, 64 卡是 1.97 份, 上限是 2 份, 几乎不随卡数涨。\nDDP 不省显存: 每张卡仍然存一整份参数、一整份梯度、一整份 Adam 优化器状态。要把这些也切开, 就是 ZeRO (把参数、梯度、优化器状态按卡切成片, 每卡只存一片的办法) 要做的事, 下一章讲。',
        },
        {
          title: '等价挂在 "按样本数加权" 上',
          body: '上一条的等式有一个前提: 累加时的权重要按样本数来。\n先看一份 micro-batch 的梯度是什么: 每份的 loss 在算的时候已经对自己的 $n_k$ 个样本取过平均, 所以它的梯度是 "$n_k$ 个样本的平均梯度"。要把 K 份合成 "全部 N 个样本的平均梯度", 每份要乘上它占的样本比例 $n_k/N$ ($n_k$ 是第 k 份的样本数, $N$ 是总样本数), 再相加。代码里就是 accumulate 函数的 scale=n / len(x)。\n每份样本数相同时, $n_k/N$ 就是 $1/K$。这时如果忘了乘, 结果只是整体放大 K 倍, 程序不会报错。\n- m01 的 "直接相加" 对照组 (weighted=False): 梯度的大小恰好是正确值的 4.00 倍, K=4。对 SGD 来说, 这等于偷偷把学习率乘了 4。\n- 换成 [3,3,2] 这种不等长切法: 按 1/K 平均和按样本数加权给出不同的梯度, 错误立刻暴露。按样本数加权时, 与整 batch 的差是 5.96e-08。\n跨卡是同一回事: all-reduce(mean) 做的是 "各卡自己那片的平均梯度" 再做一次简单平均。只有每卡的样本数相同时, 这才等于全局平均。',
        },
      ],
      links: [
        { from: 'llm_basic train.py', to: '梯度累积', body: '左边是最基础的单卡训练循环: 每个 batch 算一次梯度、更新一次参数。右边的梯度累积只改了一处: 参数更新从每个 batch 一次变成每 K 个 micro-batch 一次, 中间的梯度累加起来。其余代码一行没改。' },
        { from: 'local grads', to: 'all_reduce_mean', body: '左边是每张卡用自己那片数据算出的本地梯度, 它只代表全局 batch 的一片。右边的 all_reduce_mean 把各卡的本地梯度加起来再除以卡数。只要漏掉一次同步, 各卡用不同的梯度更新, 参数从那一步起就不再相同。' },
        { from: 'DDP', to: 'ZeRO', body: 'DDP 每张卡都存一整份参数、梯度和 Adam 状态, 显存和单卡一样紧。ZeRO 把这三样按卡切成片, 每卡只存自己那一片。DDP 存不下的时候才轮到它。' },
        { from: '2(N−1)/N', to: 'ring all-reduce', body: '左边是 DDP 每卡每步的通信量系数: 发出 $2(N-1)/N$ 份梯度大小的数据。右边的 ring all-reduce 是产生这个系数的具体算法, 它怎么算出来的见「通信与 full_loop」一章。' },
      ],
      sourceRows: [
        { concept: 'micro 权重', code: 'm01_gradient_accumulation/demo.py:accumulate', takeaway: '每份 micro-batch 的梯度乘上 n/len(x) (它的样本数占总样本数的比例) 再累加, 这一行就是让累积梯度等于整 batch 平均梯度的全部秘密。weighted=False 是故意写错的对照组, 它把权重换成 1。' },
        { concept: '不等长 micro', code: 'sizes=[3, 3, 2]', takeaway: '8 个样本切成 3、3、2 三份, 最后一份不满。这时按样本数加权和按 1/K 平均给出不同的梯度, 只有前者等于整 batch 的梯度。' },
        { concept: '梯度同步', code: 'core/collectives.py:all_reduce_mean', takeaway: '先 all-reduce 求和, 再除以卡数 world。做完之后每张卡手里的梯度逐位相同, 之后各自更新参数也就走同一条路。' },
        { concept: '副本一致性', code: 'assert max_abs_diff(replicas[0], r) == 0', takeaway: '这个断言检查同步后每张卡上的参数副本与 0 号卡逐位相同 (差为 0)。如果不同, 说明训练已经分叉成几条不同的轨迹, 而框架本身不会为此报错。' },
        { concept: '通信账', code: 'core/collectives.py:CommCounter', takeaway: 'CommCounter 记下每张卡每步发出的字节数。DDP 下是 $2(N-1)/N$ 乘以全模型梯度的字节数, m02 里 56 B 的梯度在 4 卡下是 84 B。上限是 2 份, 卡数再多也几乎不变。' },
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
      subtitle: '读完你能判断一个放不进单卡的模型该切矩阵还是切层, 以及各自要在哪里付通信。',
      tldr: '模型一张卡放不下时, 有两种切法:\n- 单层太宽: 把一层的权重矩阵切给几张卡, 叫 TP (张量并行)。MLP 的切法是第一个矩阵 W1 按列切、第二个矩阵 W2 按行切, 每个 MLP 块只要前向 1 次、反向 1 次 all-reduce (每卡交出部分和、拿回总和)。\n- 层数太多: 把连续几层放一张卡, 叫 PP (流水线并行)。卡与卡之间只传层边界的激活, 但一个 batch 流过去时总有卡在空等, 叫气泡。把 batch 切成 M 个 micro-batch 依次灌进去, 气泡就变小。',
      question: 'TP 的通信为什么在层内、PP 的只在 stage 边界? 前向已经 all-reduce 过了, 反向为什么还要再来一次?',
      evolution: {
        title: '先切层, 再切矩阵',
        subtitle: '根问题: 模型本身超过一张卡的显存, 每卡存整份的 DDP 帮不上',
        steps: [
          { name: '按层切', pain: '(原点) 模型一张卡放不下, DDP 每卡还要存整份', fix: '连续几层放一张卡, 激活顺着往下传; 同一时刻只有一张卡在算' },
          { name: 'GPipe', year: 2018, pain: '流水线上其余 PP−1 张卡都在空等', fix: 'batch 拆成 M 个 micro-batch 灌进去, 空等只剩首尾的气泡, M 越大越小' },
          { name: '朴素切矩阵', pain: '一层就比一张卡还宽, 按层切切不下去', fix: '把一层的矩阵分到多卡; W1 先行切的话, 激活函数前就得同步一次部分和' },
          { name: 'Megatron TP', year: 2019, pain: '每个非线性前都要同步, 通信次数翻倍', fix: 'W1 列切让激活函数各卡本地算, W2 行切出部分和, 到块尾才 all-reduce' },
        ],
      },
      code: 'llm_train/m03_tensor_parallel/demo.py · llm_train/m04_pipeline_parallel/demo.py',
      points: [
        {
          title: 'TP: 先列切, 后行切',
          key: true,
          body: '一个 MLP 块是两次矩阵乘夹一个激活函数: 输入 X 乘 W1, 过 ReLU, 再乘 W2。TP 把 W1 和 W2 分给 N 张卡, 关键是切的方向。\n- W1 按列切: W1 的形状是 [D, H], 按列切成 N 份, 每份 [D, H/N]。每张卡拿完整的输入 X, 算出隐藏层 H 维里属于自己的那一段。中间的 ReLU 是逐元素的运算 (每个数只看自己), 各卡在自己那段上本地做, 不用通信。换成 GeLU 同理。\n- W2 按行切: W2 的形状是 [H, O], 按行切成 N 份, 每份 [H/N, O]。每张卡手里的隐藏层那一段正好乘它拿到的那几行, 得到的是最终输出的一个部分和。最后做一次 all-reduce 把 N 个部分和加起来, 每卡都拿到完整输出。\n如果顺序反过来, W1 先按行切, 那么每卡算出的是 X·W1 的部分和, 必须先把它们加起来才能过 ReLU, 激活函数前就得多同步一次。\n偏置 b2 只在 all-reduce 之后加一次。如果每张卡都在自己的部分和上加, 求和后就被加了 N 遍。\nm03 实测: TP 的输出和不切分的 MLP 差 0.00e+00。',
        },
        {
          title: '反向也要 all-reduce',
          body: '前向只在输出处同步了一次, 反向还要再同步一次, 位置在输入 X 的梯度上。\n原因: 前向时输入 X 被每张卡都用过一遍 (每卡用它乘自己那份 W1)。反向时 loss 对 X 的梯度, 是各卡各自贡献的和: 卡 r 算出的 dX 只包含它那 $H/N$ 列隐藏单元的贡献。\n不求和就直接传给上一层, 上一层拿到的梯度是错的。m03 里单张卡的 dX 和真值差 1.8e-2; all-reduce 之后差 1.86e-09。\nall-reduce 次数: 一个 MLP 块前向 1 次 + 反向 1 次。attention 块的切法是按头切 Q、K、V 的矩阵 (列切) 加输出投影 (行切), 通信模式相同, 也是两次。一层 Transformer 共 4 次。',
        },
        {
          title: 'PP: 气泡与在途激活',
          body: 'PP 把模型按连续的层切成 PP 段, 每段叫一个 stage, 各放一张卡。一个 batch 从 stage 0 流到最后一个 stage 做前向, 再倒回来做反向。卡与卡之间只在 stage 边界传一份激活, 比 TP 每层两次 all-reduce 省得多。\n代价是空等: 前向时 batch 还没流到后面的 stage, 后面的卡没事干; 反向时反过来。把 batch 切成 M 个 micro-batch 依次灌进去, 各 stage 就能同时忙起来, 空等只剩首尾。\n- 气泡: 卡空等的时间占总时间的比例, 公式是 $(\\mathrm{PP}-1)/(M+\\mathrm{PP}-1)$, $M$ 是 micro-batch 数。M 越大气泡越小。GPipe 和 1F1B 两种调度的气泡完全相同。\n- 在途激活: 一个 micro-batch 做完前向、还没做反向时, 它的激活要留着给反向用。两种调度的区别在这里。GPipe 先把所有 M 个 micro-batch 的前向做完再做反向, 每个 stage 最多攥着 M 份; 1F1B 让 stage $s$ 最多攥 $\\mathrm{PP}-s$ 份, stage 0 最多。\n细节见「1F1B 与交错调度」。',
        },
      ],
      links: [
        { from: 'llm_models Block', to: 'TP', body: '一个 Transformer Block 里有 attention 和 MLP 两组大矩阵, TP 切的就是它们。LayerNorm、激活函数、残差相加这些逐元素的算子不切, 各卡在自己那片上本地做。' },
        { from: 'num_layers', to: 'PP', body: '模型层数 num_layers 越多, 按连续层切成几个 stage 越自然: 每个 stage 内部和单卡一样算, 只在边界把一份激活传给下一张卡。' },
        { from: 'micro-batch 数 M', to: '气泡占比', body: '气泡公式的分母里有 M, M 越大气泡越小。但 M 不能随便开大: 每个在途的 micro-batch 都要留一份激活, M 能开多大取决于这些激活放不放得下。' },
        { from: 'PP', to: '1F1B 与交错调度', body: '这一章只说了 PP 的两笔成本 (气泡和在途激活)。GPipe、1F1B、交错 1F1B 三种调度各自怎么安排时间表、各省哪一笔, 在下一章摊开。' },
      ],
      sourceRows: [
        { concept: 'column parallel', code: 'W1_s = np.split(W1, world, axis=1)', takeaway: '把 [D, H] 的 W1 沿列切成 N 份, 每份 [D, H/N]。每个 rank 用完整输入乘自己那份, 算出隐藏层的一段。' },
        { concept: 'row parallel + g 算子', code: 'tp_out = all_reduce_sum(partial)[0] + b2', takeaway: '各 rank 的部分和用 all_reduce_sum 加起来, 得到和不切分时相同的输出; 偏置 b2 在求和之后只加一次。' },
        { concept: 'f 算子的反向', code: 'tp_dx = all_reduce_sum(dx_partial)[0]', takeaway: '输入 X 的梯度由各 rank 的部分 dX 求和得到。这个算子前向什么都不做 (每卡拿完整 X), 反向做 all-reduce。省掉它, dX 就错 1.8e-2。' },
        { concept: '流水线排程', code: 'm04_pipeline_parallel/demo.py:simulate', takeaway: '流水线的时间表是模拟出来的: 每张卡按固定顺序执行自己的前向和反向, 每个操作在 "这张卡空闲了, 并且它依赖的上游操作做完了" 的最早时刻开始。' },
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
      subtitle: '读完你能自己算: 一个 7B 模型在 N 张卡上开某一档 ZeRO 之后, 每卡的模型状态占多少显存, 还剩多少留给激活。',
      tldr: '- 用混合精度训练时, 每个参数在显存里占 16 字节: 2 字节 fp16 参数 + 2 字节 fp16 梯度 + 12 字节 fp32 的 master 权重和 Adam 的两个状态 m、v。DDP 下每张卡都存这完整的 16 字节。\n- ZeRO 把这 16 字节按卡切开: ZeRO-1 切优化器状态, ZeRO-2 再切梯度, ZeRO-3 连参数一起切。每参数被除以卡数的部分依次是 12 / 14 / 16 字节。\n- 前向的中间结果 (激活) 不在这 16 字节里, 它随层数 $L$ 线性涨。激活重算用多算约 33% 的办法, 把激活显存从 $O(L)$ 压到 $O(\\sqrt{L})$。',
      question: '7B 模型在 64 张卡上开 ZeRO-1, 每卡还要多少 GB? 为什么再加卡也降不下去?',
      evolution: {
        title: '把 16 字节一份份切走',
        subtitle: '根问题: 一张卡的显存是定数, 参数、梯度、优化器状态和激活都在抢它',
        steps: [
          { name: 'DDP 全复制', pain: '(原点) 混合精度 Adam 每参数 16 字节, 其中 12 字节是优化器状态', fix: '每卡存一整份, 实现最简单; 加多少卡, 每卡都还是 $16\\Psi$' },
          { name: 'ZeRO-1/2', year: 2019, pain: 'N 张卡存着 N 份一模一样的优化器状态和梯度', fix: '按卡切成 N 片, 每卡只更新自己那片; ZeRO-2 的通信和 DDP 持平' },
          { name: 'ZeRO-3 / FSDP', pain: '参数仍是每卡整份, $2\\Psi$ 加卡也除不掉', fix: '参数也切片, 算到哪层临时 all-gather 哪层, 用完就丢; 通信约 1.5×' },
          { name: '激活重算', year: 2016, pain: '状态切完了, 激活不在 16 字节里, 随层数 $L$ 线性涨', fix: '只存段边界, 反向前重跑段内前向: 峰值 $O(\\sqrt{L})$, 计算多约 1/3' },
          { name: '可恢复 checkpoint', pain: '省出显存就能堆更多卡、训更久, 中途断掉成了常态', fix: '参数、优化器状态、数据游标、RNG 四样一起存; 缺一样不报错, 轨迹却偏了' },
        ],
      },
      code: 'llm_train/m05_zero_fsdp/demo.py · llm_train/m07_activation_checkpointing/demo.py · llm_train/m08_checkpoint_resume/demo.py',
      points: [
        {
          title: '2 + 2 + 12',
          key: true,
          body: '先算清一个参数占多少显存。混合精度训练里, 每个参数有三份东西: fp16 的参数副本 (2 字节, 算前向反向用)、fp16 的梯度 (2 字节)、fp32 的优化器状态 (12 字节: master 权重 4 字节, Adam 的 m 和 v 各 4 字节)。一共 16 字节, 其中 12 字节是优化器状态。\n设 $\\Psi$ 是参数个数, $N$ 是卡数。DDP 下每张卡都存一整份, N 张卡存了 N 份一模一样的东西, 纯冗余。ZeRO 把它们切成 $N$ 片, 每卡只存一片, 分三档:\n- DDP: $16\\Psi$\n- ZeRO-1: 只切 12 字节的优化器状态, 变成 $4\\Psi+12\\Psi/N$\n- ZeRO-2: 再切梯度, 变成 $2\\Psi+14\\Psi/N$\n- ZeRO-3/FSDP: 连参数一起切, 变成 $16\\Psi/N$\nm05 里 $\\Psi=192$、$N=4$, 把 rank 0 状态字典里所有数组的字节数加起来, 恰好是 3072 / 1344 / 1056 / 768, 与公式精确相等。\n为什么能切: Adam 的更新是逐元素的, 每个参数的更新只用到它自己的梯度和 m、v。所以每卡只更新自己那一片, 拼起来和整体更新逐位相同 (m05 实测三档与 DDP 的差都是 0.0e+00)。\n代入 7B 模型、64 卡、ZeRO-1: 7e9 × (4 + 12/64) 字节 ≈ 29 GB。那个 $4\\Psi$ (fp16 参数加 fp16 梯度) 是 ZeRO-1 切不到的, 加多少卡都除不掉, 是地板。',
        },
        {
          title: '通信账: ZeRO-2 不比 DDP 贵',
          body: '切片之后用到完整张量时要临时通信, 所以要算通信账。\n- ZeRO-2: DDP 每步做一次 all-reduce 同步梯度。一次 all-reduce 等于两步: 先 reduce-scatter (求和后每卡只留自己那片), 再 all-gather (各片拼回整份)。ZeRO-2 只是在这两步中间插了一步 "各自在自己那片上更新参数", 然后 all-gather 的对象从梯度变成更新后的参数。总字节和 DDP 一样, m05 实测 1.00×。\n- ZeRO-3: 参数也切了片, 每层算前向、反向之前都要先 all-gather 一次这一层的完整参数才能算, 通信约 1.5×。用完立刻释放, 任何时刻只多出一层的完整 fp16 参数, demo 里是 +256 B。\nm05 里 ZeRO-1 也是 1.5×, 因为这个实现用 all-reduce 加 all-gather; DeepSpeed 改用 reduce-scatter 后与 DDP 持平。',
        },
        {
          title: '激活重算: 用 33% 计算换 √L 显存',
          body: '反向传播算第 i 层的梯度时, 需要第 i 层前向时的输入。所以前向要把每一层的中间结果 (激活) 都留着, L 层就是 L 份, 随层数线性涨。\n激活重算的办法: 前向只保存每一段的入口, 每 k 层一段。反向走到哪一段, 就从那段的入口重跑一遍段内的前向, 把激活临时算出来, 用完就扔。\n峰值激活 $\\approx L/k + k$: $L/k$ 份是常驻的段边界, $k$ 份是当前段重算出来的瞬时激活。$k=\\sqrt{L}$ 时最小, 所以是 $O(\\sqrt{L})$。\nm07 里 $L=16$: $k=1$ (全存) 要 17 份, $k=4$ 只要 8 份。k 太大也不行, $k=16$ 只存输入, 段内瞬时激活又回到 17 份。重算的是同一串浮点运算, 梯度与全存逐位相同 (0.0e+00)。\n代价是多跑一次前向。前向算 1 个单位, 反向约 2 个单位, 本来一共 3 个单位, 现在接近 4 个, 上限 +33%。demo 里 $L=16$、$k=4$ 每段出口不重算, 实测 +25%。',
        },
        {
          title: '可恢复 = 四样齐全',
          body: '训练中途断掉是常态, 要能从 checkpoint 接着训。"能恢复" 的标准不是 loss 差不多, 是逐位相同: 续训出来的参数和不中断训出来的一模一样。\n要做到这点, 得把下一步计算会读到的所有状态都存下来, 共四样: 参数、优化器状态 (动量或 Adam 的 m、v)、数据游标 (数据流读到第几个 batch)、RNG 状态 (随机数生成器, dropout 和打乱顺序都从它抽数)。\nm08 从第 3 步存盘, 新进程里把所有对象换一组种子重建, 只靠 checkpoint 拉回来, 再训到第 5 步和不中断的比。然后一样样故意漏掉:\n- 完整恢复: 差 0.0e+00。\n- 漏 RNG: 差 3.1e-2。dropout 的 mask 序列变了。\n- 漏优化器: 差 5.8e-2。动量从 0 重新起步。\n- 漏数据游标: 差 1.5e-1。接下来喂的不是同一批数据。\n三种漏法全都不报错, 训练照跑, 只是轨迹悄悄变成了另一条。\n那个 checkpoint 共 3047 B, 其中 RNG 状态占 2496 B (numpy 的 MT19937 生成器内部有 624 个 32 位的字), 玩具模型的参数只有 40 B。',
        },
      ],
      links: [
        { from: 'DDP 每卡全量', to: 'ZeRO 分片', body: 'DDP 每卡存一整份, ZeRO 每卡只存一片。复制换成分片, 显存下降, 代价是用到完整张量时要临时通信: ZeRO-2 和 DDP 持平, ZeRO-3 约 1.5×。' },
        { from: 'llm_basic 的 cache', to: '激活重算', body: '基础教程里的小模型把前向的中间量 (cache) 全部存下来给反向用。大模型存不下, 只存每段的边界, 其余的在反向之前重跑前向算出来。' },
        { from: 'ToyDataStream.cursor', to: 'resume', body: '数据流的 cursor 记着读到了第几个 batch。它也是训练状态: 不存的话, 续训时喂进去的不是同一批数据, 就接不回同一条轨迹。' },
        { from: 'reduce-scatter', to: '通信原语', body: '把一次 all-reduce 拆成 reduce-scatter 加 all-gather 两半看, 就明白 ZeRO-2 为什么不涨通信: 它只是在两半中间插了一步分片更新。四个原语的细节见「通信与 full_loop」。' },
      ],
      sourceRows: [
        { concept: '状态怎么切', code: 'm05_zero_fsdp/demo.py:init_rank_state', takeaway: '每个 rank 的状态是一个字典: p16 (fp16 参数)、g16 (fp16 梯度)、master、m、v。stage 决定其中哪些数组存整份、哪些只存第 r 片 ($1/N$)。这就是 ZeRO 的全部数据结构, 显存账直接用这些数组的字节数加出来。' },
        { concept: '梯度同步的分水岭', code: 'reduce_scatter_sum(layer_g)', takeaway: '梯度同步这一行是三档的分水岭: stage 0 和 1 用 all-reduce, 每卡留整份梯度; stage 2 和 3 换成 reduce_scatter, 每卡只留自己那片。' },
        { concept: 'FSDP gather → compute → free', code: 'full = all_gather([rk["p16"][i] for rk in ranks])', takeaway: 'stage 3 的前向和反向: 算到哪一层, 就 all_gather 这一层的参数片拼成完整参数, 算完 del full 立刻释放。任何时刻只多出一层的完整参数。' },
        { concept: '激活账本', code: 'm07_activation_checkpointing/demo.py:run', takeaway: 'run 函数里有一本激活账: 每保存或释放一份激活都记一笔, 峰值取历史最大。反向时重算出来的段内激活也走这本账, 所以峰值里包含它们, 不能只数段边界。' },
        { concept: 'resume payload', code: 'm08_checkpoint_resume/demo.py:snapshot', takeaway: 'snapshot 把五样东西收进一个字典: step、模型参数、优化器状态、数据流的 seed 和 cursor、RNG 状态。缺任何一样, 续训的轨迹就对不上。' },
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
      subtitle: '读完你能解释现代训练为什么默认 BF16、本章的 fp32 master 为什么删不得, 以及一个 Inf 冒出来时框架在做什么。',
      tldr: '低精度只负责算, 更新落在 fp32 master (一份用 fp32 存的权重主副本) 上。这是本章的训练配方, 不是所有实现的必要条件; m13 里去掉它, 200 步后的验证 loss 差 584×。\n- FP16: 能表示的范围窄, 很小的梯度会变成 0。要配动态 loss scaling: 先把 loss 放大再做反向, 梯度跟着一起放大; 一旦放大过头溢出成 Inf, 就跳过这一步, 并把放大倍数 scale 减半。\n- BF16: 指数位和 FP32 一样多, 用精度换范围, 省掉 scaling。\n- FP8: 位宽再砍一半, 张量要先除以一个 scale 缩放进格式能表示的范围再存, scale 变成必选项。',
      question: '为什么 FP16 训练不是把所有数组 astype(np.float16) 就完事? BF16 和 FP8 又各改了什么?',
      evolution: {
        title: '位宽每砍一刀, 就补一道保险',
        subtitle: '根问题: 每个数用的位越少, 显存、带宽、算力越省, 能表示的范围和精度也越窄',
        steps: [
          { name: '全程 FP32', pain: '(原点) 每个数 4 字节, 显存、带宽、算力都按位宽付钱', fix: '最稳; 代价是每次搬运、每次乘法都付 32 位的价钱' },
          { name: 'FP16 混合精度', year: 2017, pain: '直接换 FP16: 1e-8 的梯度变 0, 70000 的激活变 Inf', fix: 'loss 先放大再除回, 更新落在 FP32 的权重主副本 (master) 上; 溢出就跳步、scale 减半' },
          { name: 'BF16', pain: 'scale 要一路动态调, 溢出还得白白跳步', fix: '指数位和 FP32 一样宽 (8 位), 两头都不炸, 不用 scaling; 尾数只剩 7 位' },
          { name: 'FP8', year: 2022, pain: '位宽再砍一半, 格式自带的范围装不下训练里的数', fix: 'E4M3 (4 位指数 3 位尾数) / E5M2 必须配 scale, 先缩放进格式的范围; 更新仍靠 FP32 master, 去掉它 loss 差 584×' },
        ],
      },
      code: 'llm_train/m06_mixed_precision/demo.py · llm_train/m10_training_stability/demo.py · llm_train/m13_fp8_training/demo.py · llm_train/core/numerics.py',
      points: [
        {
          title: '范围 vs 精度: 16 位的两种分法',
          body: '同样是 16 位, FP16 和 BF16 把位数分给了不同的用处: FP16 偏精度, BF16 偏范围。\n先说浮点数怎么存。一个浮点数分成符号、指数、尾数三部分。指数位决定能表示多大、多小的数 (范围), 尾数位决定相邻两个能表示的数挨得多近 (精度)。16 位总共就这么多, 给指数多一位, 尾数就少一位。\n- FP16 (5 位指数 + 10 位尾数): 能表示的最大数是 65504, 最小的正数约 6e-8。所以 1e-8 的梯度直接变 0 (下溢), 70000 的激活直接变 Inf (上溢)。\n- BF16 (8 + 7): 指数位和 FP32 一样宽, 范围也和 FP32 一样。1e-8 存成 1.001e-08、70000 存成 70144, 两头都不炸。\nBF16 的代价是尾数少 3 位: 在 1.0 附近, BF16 相邻两个数差 $2^{-7}$, FP16 差 $2^{-10}$。所以 $1+2^{-9}$ 在 BF16 里会被舍回 1, FP16 能存成 1.001953125。\nBF16 精度比 FP16 低, 赢的只有范围。但范围恰好是训练最怕丢的东西: 梯度变 0 就学不到东西, 变 Inf 整步就废了。README 提到现代 LLM 预训练几乎都用 BF16, FP16 加 loss scaling 主要见于老卡。\nnumpy 没有 bfloat16, demo 用「把尾数截到 7 位」来模拟 BF16 的舍入。',
        },
        {
          title: 'loss scaling + fp32 master',
          key: true,
          body: 'FP16 训练要两样保险: loss scaling 救小梯度, fp32 master 救小更新。\nloss scaling 怎么做: 反向传播算出的梯度和 loss 成正比。所以反向前先把 loss 乘一个大数 S, 所有梯度也跟着放大 S 倍, 原本太小的梯度就挪进了 FP16 能表示的范围。更新前再在 fp32 里除回 S。例子: 1e-8 直接转 fp16 是 0; 先乘 32768 再转 fp16, 再除回来, 得到 9.997e-09。\nS 取多大不好事先定, 所以用动态的:\n- 出现 Inf: 说明 S 太大, 梯度放大后溢出了。跳过这一步, 权重一位都不动, 并把 scale 减半。\n- 连续若干个好 step: scale 再翻倍, 试探能不能更大。\nm06 故意从 $2^{20}$ 起步, 24 步里跳了 5 步, scale 在上限附近来回振荡, loss 仍从 1.6477 降到 0.0229。这些 Inf 来自 fp16 矩阵乘本身超过 65504, 不是手工塞进去的。\nfp32 master 怎么回事: 本章的更新落在 fp32 master 上。master 是一份用 fp32 存的权重, 每步计算时从它转出一份 fp16 副本来算。为什么要它: fp16 在 1.0 附近相邻两个数差约 0.001, 比这小得多的更新加上去会被舍回原值。demo 里纯 fp16 权重每步加 1e-5 的更新, 100 步后还是 1.0, 一动没动。fp32 master 记住了每一步, 100 步后是 0.998999; 它的 fp16 副本 1 步后还是 1.0, 累计够了才跟着动, 100 步后是 0.9990234375。',
        },
        {
          title: 'FP8 的两套配方',
          body: 'FP8 只有 8 位, 公开的训练配方有两套, 差别在用哪种格式和 scale 管多大一块。\n格式名直接写位数: E4M3 = 4 位指数 + 3 位尾数, 最大 448, 更准; E5M2 = 5 位指数 + 2 位尾数, 最大 57344, 范围更大。\nscale 是什么: 量化前先把一组数除以 scale, 让这组数里绝对值最大的那个正好落在格式的最大值上, 量化完再乘回来。这组数可以是整个张量, 也可以是一小块。\n- Transformer Engine: 前向 E4M3、反向 E5M2, 整个张量共用一个 scale (per-tensor)。\n- DeepSeek-V3: 全程 E4M3, 每 128 个元素一个 scale (block)。\nE4M3 自带约 $2^{15}$ 的动态范围 (能表示的最大数和最小数之比), 所以 scale 管多大一块, 平时看不出差别。m13 量了两个指标:\n- 验证 loss (训 200 步, 除以 $\\|y\\|^2$), 每次只动一个旋钮: 反向 E5M2 换成 E4M3, 从 4.54e-4 变成 4.64e-4; per-tensor 换成 block, 从 4.64e-4 变成 4.61e-4, 都打平。\n- 其余 token 的输出相对误差: demo 在 3 个 token 里各放一个 outlier (远大于其它元素的值), 看另外 29 个 token 的矩阵乘输出被连累多少。outlier 超过约 1e5 倍才拉开, per-tensor 0.126, block 0.026。原因是 per-tensor 下这个 outlier 决定了整个张量的 scale, 其余正常的数被除得太小, 掉进格式表示不了的区间, 1e5 倍时有 17.8% 直接变成 0。block 下 outlier 只影响它自己那 128 个元素。\ndemo 是假量化: 量化后马上还原成 FP32 再做矩阵乘, 只模拟舍入误差, 不模拟 FP8 的速度。',
        },
        {
          title: '去掉 master 比去掉 scaling 更伤',
          body: '在本章的 FP8 消融里, 去掉 fp32 master 的伤害比去掉 scaling 大一个数量级以上。\nm13 的训练是一个线性回归, 量级刻意取真实 LLM 的水平: 权重的标准差 0.004, 每个元素的梯度约 1e-5。同一个消融, 比的都是 200 步后的验证 loss:\n- 去掉 scaling: 差 29×。权重和梯度都掉进 FP8 表示不了的小数区。0.004 左右的权重在 E4M3 里相邻格点差约 0.002, 一个标准差只跨两个格点; 1e-5 的梯度比 E5M2 最小的步长 (约 1.5e-5) 还小, 只能舍成 0 或一个步长。\n- 去掉 fp32 master: 差 584×。权重只以 FP8 的形式存在, 每步更新后马上舍回 FP8 的格点。小于 FP8 步长的更新全部丢失, 权重几乎学不动。\nFP8 最好的那几臂停在约 4.6e-4, FP32 基线是 1.02e-08。README 解释这是因为玩具问题没有噪声, 量化误差直接露了出来; 真实 LLM 的 loss 噪声远大于它。',
        },
      ],
      links: [
        { from: 'Adam state', to: 'fp32 master', body: '参数更新需要保留足够精度, FP16 / FP8 都一样: 每步的小更新太细, 低精度存不下。本章用 fp32 master 来保证, Adam 的状态 m、v 也一直留在高精度; README 提到也可以用 Kahan 补偿或随机舍入。' },
        { from: 'global grad norm', to: 'DDP/FSDP', body: '梯度裁剪看的是全局范数: 所有参数的梯度拼在一起算一个长度。所以裁剪要排在全局同步之后 (同步前每张卡手上的梯度都不完整), 也要排在 unscale (除掉 loss scale) 之后, 否则量到的是放大过的长度。' },
        { from: 'LossScaler.scale', to: 'checkpoint', body: 'LossScaler 除了当前的 scale, 还记着 good_steps (已经连续多少个好 step)。两者都是训练状态, 要和权重一起写进 checkpoint。' },
        { from: 'FP8 block scaling', to: 'FP4 微缩放', body: 'FP8 每 128 个元素共用一个 scale。位宽再砍一半到 FP4, 每个数只能取 ±{0, 0.5, 1, 1.5, 2, 3, 4, 6} 这几个值, block 必须缩到 16~32 个元素一组, 见「FP8 与 FP4 微缩放」。' },
      ],
      sourceRows: [
        { concept: '动态 loss scale', code: 'm06_mixed_precision/demo.py:LossScaler', takeaway: '溢出就把 scale 减半 (最小减到 1) 并跳过这一步; 连续 growth_interval 个好 step 就翻倍, 试探上限。真实默认 growth_interval 是 2000, demo 用 5 是为了在 24 步里看到完整的涨跌循环。' },
        { concept: '浮点格式三要素', code: 'core/numerics.py:fake_quant_float', takeaway: '一个浮点格式由三个数决定: 尾数位数、最大值、最小 normal 指数 (比它更小的数有效位越来越少, 再小就变 0)。函数按这三个数把每个元素舍入到格式的格点上, 超过最大值就截到最大值。BF16 / FP8 / FP4 共用这一个函数。' },
        { concept: 'NaN guard + clip', code: 'm10_training_stability/demo.py:guarded_step', takeaway: '先查梯度里有没有 Inf / NaN。有就是坏 step, 返回 False, 一个参数都不碰。没有才是好 step, 先按全局范数裁剪再更新。检查必须排在裁剪之前, 因为 NaN 的范数还是 NaN, 裁剪救不了它。' },
        { concept: '裁剪不改方向', code: 'g * min(1, max_norm / ‖g‖)', takeaway: '梯度太长就整体等比缩短到 max_norm, 方向不变。m10 里 $\\|g\\|$ 从 111.83 缩到 5.00, 裁剪前后的方向余弦是 1.000000。' },
        { concept: 'FP8 消融的旋钮', code: 'm13_fp8_training/demo.py:make_quantizer', takeaway: 'make_quantizer 管 scaling (none / tensor / block) 和量化格式, train 再管是否保留 master。五个臂 A 到 E 相邻两个只差一个旋钮: A 到 B 加 scaling, B 到 C 把反向格式从 E5M2 换成 E4M3, C 到 D 把粒度从 per-tensor 换成 block, D 到 E 去掉 master。demo 用断言钉住了这一点。' },
        { concept: 'E4M3 max = 448', code: 'FLOAT_FORMATS["e4m3"]', takeaway: 'E4M3 能表示的最大数是 448, 不是 480: 最高的那个尾数编码留给了 NaN。超过 448 的数会被截成 448, demo 里 3e4 的尖峰就变成了 448。' },
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
      subtitle: '读完你能说清 MoE (混合专家) 模型分到多卡以后, 卡之间要传的数据量为什么每一步都不一样。也能说清一条长到单卡放不下的序列, 怎么切给多张卡还能算出和单卡完全一样的注意力。',
      tldr: '- EP (专家并行): MoE 模型里有很多个「专家」小网络, 每个 token 由路由器挑一个 (或几个) 去算。专家太多一张卡放不下, 就分到多张卡上。token 要先发到它的专家所在的卡, 算完再发回来, 一去一回各是一次 all-to-all (每张卡给每张卡各发一份不同的数据)。发多少由 gating (路由器这一步怎么分 token) 决定, 模型结构定不了它。demo 里路由一偏, 最忙的卡收到 24 个 token, 均匀的话只该收 16 个。\n- Ring Attention: 一条长序列切给 D 张卡, 每卡留着自己那段的 query, K 和 V 按块在卡之间转圈传。每卡收到一块就把它并进结果, 最后和完整注意力只差 4.4e-16。它的通信量是固定的, 和怎么切无关。但在因果 mask (每个 token 只看前面) 下必须用 zigzag 切法 (每卡拿一头一尾两段), 否则计算量省了一半, 墙钟几乎没省。demo 里墙钟从连续切分的 228 降到 132。',
      question: 'MoE 的 all-to-all 每一步要传多少数据, 为什么由数据决定、不由模型结构决定? Ring Attention 把注意力拆到几张卡上一块一块算, 凭什么和完整注意力精确相等?',
      evolution: {
        title: '参数和算力脱钩之后',
        subtitle: '根问题: 参数越多模型越强, 可稠密模型每加一份参数, 每个 token 就多一份计算',
        steps: [
          { name: '稠密 FFN', pain: '(原点) 每个 token 都要乘过全部参数', fix: '参数量和每 token 计算量绑死: 参数翻倍, 算力也翻倍' },
          { name: '稀疏门控 MoE', year: 2017, pain: '想加参数, 又付不起同比例的算力', fix: 'FFN 换成一组专家, 路由器给每个 token 只挑一个或几个; 参数涨, 每 token 计算不涨' },
          { name: 'GShard 专家并行', year: 2020, pain: '专家多到一张卡放不下', fix: '专家分到各卡, token 用 all-to-all (每卡给每卡各发一份数据) 发去专家所在卡再寄回; 流量看路由结果' },
          { name: 'Switch aux loss', year: 2021, pain: '路由扎堆: 最热那张卡多干 50%, 装不下的 token 被丢掉', fix: 'loss 里加 $E\\sum f_e P_e$, 收 token 越多的专家路由概率被压得越狠; 最热专家比平均 (max/mean) 从 1.62× 降到 1.12×' },
          { name: '无辅助损失 bias', year: 2024, pain: '惩罚项掺进 loss, 推平路由的同时也在拉偏语言建模', fix: '每个专家一个 bias, 只在挑专家时加上, 热的调低、冷的调高; 不进计算图, 不产生梯度: max/mean 1.00×' },
        ],
      },
      code: 'llm_train/m11_expert_parallel/demo.py · llm_train/m12_sequence_parallel/demo.py',
      points: [
        {
          title: '路由决定通信量',
          key: true,
          body: 'MoE 的通信量不是模型定的, 是这一步的路由结果定的。\nMoE (mixture of experts, 混合专家) 把 Transformer 里的 FFN 换成 E 个并排的小 FFN, 每个叫一个专家。每个 token 进来时, 一个叫路由器 (router) 的小线性层给它在 E 个专家上打分, 再过 softmax 变成概率。这一步叫 gating。demo 用 top-1 路由: 每个 token 只送去概率最高的那一个专家。\n专家多了一张卡放不下, EP (expert parallel, 专家并行) 就把 E 个专家分给 D 张卡。demo 里是 4 张卡、8 个专家, 每卡放 2 个。麻烦在于 token 所在的卡, 常常不是它选中的专家所在的卡。所以要来回两趟 all-to-all。all-to-all 是一种集合通信, 每张卡给每张卡各发一份不同的数据。\n- dispatch: 每张卡把手上的 token 按目标卡分成 D 堆, 发往专家所在的卡。\n- combine: 专家算完, 结果原路寄回。源卡发货时在本地记下了每个 token 原来的行号 (代码里的 send_idx), 收到结果后按行号放回原位, 再乘上 gate 权重 (路由器给选中专家的那个概率)。\ndemo 的 dispatch 把 token 和专家 id 分两次发, 所以计数器记的是 3 次 all-to-all 调用, 逻辑上还是一去一回两趟。\n通信量由谁定:\n- DDP (数据并行, 每卡存整份模型、每步把各卡梯度加总): 每步搬多少字节由模型大小定死, 每一步都一样。\n- EP: all-to-all 发多少、发给谁, 全看 gating 这一次把 token 路由到了哪。换一批数据, 发送量就变了。\ndemo 只做前向、只用 top-1; 真实模型每个 token 选 2 到 8 个专家, 反向也要同样来回两趟。',
        },
        {
          title: '不均衡会丢 token, 拖慢整步',
          body: '路由一不均衡, 就有两个后果: 最忙的卡拖慢整步, 超出容量的 token 被丢掉。\n- 负载不均: m11 里 64 个 token 分到 4 张卡, 按路由结果每卡实际收到 [11, 18, 11, 24] 个, 均匀的话应该各 16 个。所有卡要等最慢的那张做完才能进下一步。最热那张卡收了 24 个, 比 16 个多干 50% 的活, 其余三张卡就在等它。\n- 装不下: 每个专家有一个容量 capacity, 是它一步最多收的 token 数。公式是 capacity = ⌈cf · N_tok / E⌉, 这个式子在说: 先算每个专家平均该收几个 token, 再乘一个余量倍数 cf (capacity factor)。demo 里 64 个 token、8 个专家、cf=1.25, 容量就是 10。超过容量的 token 先到先得, 后来的被丢弃: 专家不算它, 它只靠残差连接把原输入原样带到下一层。demo 故意给 router 加了倾斜, 8 个专家收到 [1, 10, 11, 7, 4, 7, 11, 13] 个 token, 超过 10 的部分加起来, 直接有 5 个 token 装不下。\n把 cf 调大也不是免费的: README 指出显存和计算要按容量预留, cf=2 就是两倍开销。\n所以均衡不能碰运气, 要写进训练流程。',
        },
        {
          title: '两种压平路由的手段',
          body: '衡量均衡的指标是 max/mean: 收到 token 最多的那个专家, 它的 token 数除以各专家的平均数。1.00× 是完全均衡。上一条的倾斜 router 是 1.62× (最多 13 个, 平均 8 个)。\n- Switch / Mixtral 的做法: 在 loss 里加一项辅助损失 (aux loss) $L = E\\cdot\\sum f_e\\cdot P_e$。$f_e$ 是专家 e 实际收到的 token 占比, $P_e$ 是路由器给它的平均概率。这个式子在说: 一个专家实际收的 token 越多, 它的路由概率就被罚得越重。$f$ 来自 argmax (挑概率最大的那个), argmax 没法求导, 所以把 $f$ 当常数, 只对 $P$ 求导。对 $P_e$ 的梯度和 $f_e$ 成正比, 越热的专家, 概率被往下压得越狠, 梯度就在推平路由概率。m11 只用 aux loss 训路由器 60 步: max/mean 从 1.62× 降到 1.12×, 丢弃的 token 从 5 个降到 0。\n- DeepSeek-V3 的做法 (aux-loss-free, 不加辅助损失): 给每个专家一个可调的 bias, 挑专家时用「概率 + bias」比大小。每步看一遍负载, 热门的 (收得比平均多) 把 bias 减一点, 冷门的加一点, demo 每步调 γ=0.01, 调 200 步。gate 权重仍用原始概率, bias 只改「选谁」, 不往 loss 里掺东西。m11: max/mean 降到 1.00×, 8 个专家各收 8 个 token。\n区别在副作用。aux loss 的梯度会流进路由器, 和语言建模的 loss 抢方向。bias 不进计算图, 反向传播不经过它, 不产生任何梯度。\ndemo 的路由器只用 aux loss 训练, 好单独看它的效果。真实训练里 aux loss 乘一个小系数 (α≈0.01) 加到任务 loss 上, 两者互相拉扯, 这正是 aux-loss-free 的动机。',
        },
        {
          title: '分块算注意力, 结果精确相等',
          body: '注意力可以一块一块地算, 再增量合并, 结果和一次算完完全一样。这个技巧叫 online softmax。\n序列很长时, 一张卡放不下全部的 K 和 V。上下文并行 (context parallel) 就把序列切成 D 段, 每张卡留着自己那段的 Q, K 和 V 则按块在卡之间传。注意力要对一个 query 和所有 key 的分数做 softmax: 每个分数取指数, 再除以所有指数之和。难点是 softmax 的分母要等所有分数都到齐才知道。online softmax 给每一行 query 维护三个累计量: 目前见过的最大分数 m、分母 den、加权和 acc (每个 v 乘上自己的指数权重后加起来)。取指数前先减去最大值, 是为了不让指数溢出。每收到一块新的 K/V:\n- 算这一块的分数, 和旧的 m 比, 得到新的最大值 m\'。\n- 旧的 den 和 acc 是按旧最大值 m 算的指数, 统一乘上 $e^{m-m\'}$。这个系数在说: 新最大值比旧的大多少, 旧的累计量就按比例缩小多少, 这样新旧两部分站在同一个基准上。\n- 把这一块的贡献加进 den 和 acc。\n全部块到齐后, 输出 acc/den。这是恒等变形, 不是近似: m12 实测与完整注意力差 4.4e-16, 只是浮点舍入。\n- 在单卡内这样做: FlashAttention, 块来自同一张卡的显存。\n- 跨卡传块这样做: Ring Attention。D 张卡连成一个环, 每轮每卡把手里的 KV 块传给右边邻居, 传 D−1 轮后每卡都见过所有块。任何时刻每卡只持有 $1/D$ 的 KV ($D$ 是环上的卡数)。demo 里 D=4、T=32 个 token, 每卡常驻 256 个数, 整份是 1024 个。\n通信量是固定的: 每卡传 D−1 轮, 每轮一个 KV 块, demo 里每卡共 6144 字节, 和下一条讲的切法无关。',
        },
        {
          title: '因果 mask 下要 zigzag 切',
          body: '因果 mask 下, 按顺序连续切会让最后一张卡最忙, 省下的计算变不成时间。zigzag 切法把活分平。\n因果 mask 的意思是每个 token 只能看它自己和前面的 token。所以一张卡收到的 KV 块如果整块都在它的 query 后面 (整块落在未来), 就直接跳过不算。总计算量因此省了约一半: demo 里 T=32, 要算的 q·k 对数是 528, 不加 mask 是 1024。\n省下的计算能不能变成时间, 要看怎么切。环上每一轮, 所有卡都要等最慢的那张算完才能一起传下一块。所以 demo 这样估墙钟时间: 每轮取最慢那张卡的 q·k 对数, 再把各轮加起来。\n- 不利用因果性: 4 轮每轮都算满块 (8×8=64 对), 墙钟 256。\n- 连续切分: 卡 r 拿第 r 段, 卡 0 拿最前面 8 个 token, 卡 3 拿最后 8 个。卡 r 有 r+1 个块要算 (它前面的块和它自己那块), 卡 0 只算 1 块。最后一张卡第一轮算自己那块的对角一半 (36 对), 之后每轮都满载 64 对。其余人等它, 墙钟 228, 只省约 10%。\n- zigzag: 序列切 $2D$ 段, 卡 r 拿第 r 段和第 2D−1−r 段, 一头一尾配对。D=4 时切 8 段, 卡 0 拿第 0 和第 7 段, 卡 1 拿第 1 和第 6 段。每张卡都是一段靠前、一段靠后。demo 里除了第一轮 (各算 36 对), 之后每轮每卡都正好算 32 对。每卡每轮工作量相同, 墙钟降到 132, 比连续切分快 1.73×。\ndemo 的墙钟是按 q·k 对数估的代理指标, 没有模拟传输和计算的重叠 (真实系统靠重叠把传输时间藏起来), 也只做了前向。',
        },
      ],
      links: [
        { from: 'gating top-k', to: 'all-to-all dispatch', body: 'gating 给每个 token 选出得分最高的 k 个专家 (demo 里 k=1)。token 所在的卡和它的专家所在的卡往往不是同一张, 所以要用 all-to-all 重排一次发过去, 算完再排回来。' },
        { from: 'capacity factor', to: 'token dropping', body: 'capacity factor 决定每个专家一步最多收多少 token。64 个 token、8 个专家、cf=1.25 时每专家容量是 10, 溢出的 token 被丢掉, 不经过专家, 直接走残差连接。' },
        { from: 'KV 环传递', to: 'online softmax (m, l, acc)', body: 'KV 块沿环一块一块传来, 每卡收到一块, 就用 online softmax 的三个累计量 (最大值 m、分母 l、加权和 acc) 把它增量合并。所以任何时刻每卡只需持有 $1/D$ 的 KV。' },
        { from: '因果 mask', to: 'zigzag 切分', body: '因果 mask 让靠后的卡要算的块更多。连续切分时最后一张卡每轮满载, 大家等它; zigzag 一头一尾配对后每卡每轮工作量相同, 墙钟快 1.73×。' },
      ],
      sourceRows: [
        { concept: 'dispatch 发送矩阵', code: 'm11_expert_parallel/demo.py:moe_forward_ep', takeaway: '一张 [D, D] 的表: 行是源卡, 列是目标卡, 第 s 行第 d 列是卡 s 要发给卡 d 几个 token。这张表就是 all-to-all 的发货单, 按列加起来就是每卡收到的 token 数 [11, 18, 11, 24]。' },
        { concept: 'aux loss 的梯度', code: 'm11_expert_parallel/demo.py:train_router_aux', takeaway: '按 $L = E\\cdot\\sum f_e\\cdot P_e$ 手写梯度。$f$ 来自 argmax 不可导, 视为常数; 对 $P$ 求导, 越热的专家梯度越大, 概率被压得越多, 这就在推平路由。' },
        { concept: 'aux-loss-free bias', code: 'm11_expert_parallel/demo.py:balance_bias', takeaway: '每步数一遍各专家的负载, 热门专家的 bias 往下调 γ、冷门的往上调 γ。bias 只影响挑谁, gate 概率不变 (demo 有断言), 也不往 loss 里掺东西。' },
        { concept: '环上的一步', code: 'm12_sequence_parallel/demo.py:ring_attention', takeaway: '卡 r 在第 s 轮处理来自卡 (r−s)%D 的 KV 块: 按因果 mask 算这一块的分数 (整块在未来就跳过), 并进累计量, 再把块传给右邻居。' },
        { concept: '增量合并', code: 'm_new / scale', takeaway: '用新的最大值 m_new 算出修正系数, 把旧的分母和加权和换到新的指数基准上, 再加上这一块的贡献。结果与一次性 softmax 差 4.4e-16。' },
        { concept: 'zigzag 切分', code: 'm12_sequence_parallel/demo.py:shard_positions', takeaway: '序列切 $2D$ 段, 卡 $r$ 拿第 $r$ 段和第 $(2D-1-r)$ 段, 一头一尾配对。D=4 时 4 张卡的总工作量都是 132 对, 连续切分是 [36, 100, 164, 228]。' },
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
      subtitle: '读完你拿到任何一个训练框架, 都能先把它的通信路径还原成四种基本操作, 再去看它的封装。',
      tldr: '- 多卡训练里所有的并行策略, 最后都落到四种卡与卡之间的基本通信操作 (叫通信原语): all-reduce、reduce-scatter、all-gather、all-to-all。它们的区别只有两点: 要不要把各卡的数据加起来, 以及每卡最后拿回整份还是一片。\n- 用 ring (环形) 算法实现 all-reduce 时, 每张卡每步只发 $2(N-1)/N\\cdot S$ 字节 ($S$ 是张量大小, $N$ 是卡数), 卡数再多也不超过 $2S$。\n- full_loop 把前面各章的模块拼成一个完整的训练主循环: 40 步里跳过了 4 个坏 step, 中途存盘续训后参数逐位相同。',
      question: '拿到一个陌生的训练框架, 从哪里开始读才最快看懂它的并行方式?',
      evolution: {
        title: '从一个中心到一个环',
        subtitle: '根问题: 每张卡的网口带宽是定数, 多卡训练每一步都要把梯度汇总一遍',
        steps: [
          { name: '汇总到 0 号卡', pain: '(原点) 每张卡都要拿到所有卡梯度的和', fix: '全发给 0 号卡求和再广播; 0 号卡收发 $2(N-1)\\cdot S$, 随卡数线性涨' },
          { name: 'ring all-reduce', pain: '0 号卡的网口堵死, 其余卡的带宽闲着', fix: '张量切 N 块沿环接力, 每卡收发一样多, 封顶 $2S$ (N=4 时 0 号卡省 4 倍)' },
          { name: 'reduce-scatter + all-gather', pain: 'ZeRO 每卡只更新自己那片, all-reduce 却把整份和发给每张卡', fix: 'all-reduce 拆成两半, 中间插一步分片更新; 总字节和 all-reduce 一样' },
          { name: 'full_loop', pain: '梯度切成片后, 全局范数和该不该跳步, 没有哪张卡能单独判', fix: '一次标量 all-reduce 求全局范数; 任一卡溢出, 全体一起跳步 (40 步跳 4 步)' },
        ],
      },
      code: 'llm_train/core/collectives.py · llm_train/m09_collectives/demo.py · llm_train/full_loop/demo.py',
      points: [
        {
          title: '四个原语各做什么, 谁在用',
          body: '把 N 张卡想成 N 个人, 每人手里一块数据。四个原语只在两件事上不同: 要不要把大家的数据加起来; 每人最后拿回整份还是只拿一片。\n- all-reduce: 加。每张卡交出自己的一份, 拿回所有卡的和, 整份。DDP 用它同步梯度。\n- reduce-scatter: 加。同样求和, 但每张卡只拿回和的 $1/N$: 自己负责的那一片。\n- all-gather: 不加。每张卡交出自己那一片, 拿回把所有片拼好的完整张量。ZeRO/FSDP (把参数和优化器状态按卡切片存的数据并行) 用 reduce-scatter 和 all-gather 这一对: 梯度求和后各留一片, 参数更新后再拼回整份。\n- all-to-all: 不加, 互换。每张卡给每张卡各准备一块不同的数据, 卡 src 给卡 dst 准备的那块最后到了卡 dst 手里。写成代码是 out[dst][src] = in[src][dst]。MoE (混合专家模型) 和 Ulysses (一种序列并行) 用它。\n先做 reduce-scatter 再做 all-gather, 结果和一次 all-reduce 完全相同。这一点后面反复用到。',
        },
        {
          title: '通信原语是成本中心',
          key: true,
          body: '不同的并行策略算的是同一个数学公式, 结果也一样。它们的差别在每一步要在卡之间搬什么数据、搬多少、什么时候搬。所以看一个策略贵不贵, 先看它的通信。\n以 all-reduce 为例, 设张量大小 $S$ 字节、卡数 $N$:\n- 朴素做法: 所有卡把数据发给 0 号卡, 0 号卡加完再广播给大家。0 号卡要收 $N-1$ 份、再发 $N-1$ 份, 一共收发 $2(N-1)\\cdot S$ 字节, 其他卡的网口却闲着。$N=4$ 时 0 号卡的流量已经是 ring 的 4 倍 (m09 里 192 B 对 48 B), 而且随 $N$ 线性涨。\n- ring 实现: 把张量切成 N 块, N 张卡连成一个环, 每一步每张卡只给右邻居发一块。先绕一圈边传边加 (reduce-scatter 阶段, N−1 步), 结束时每张卡手里有了某一块的全局和; 再绕一圈把加好的块传给大家 (all-gather 阶段, 又 N−1 步)。每张卡一共发 $2(N-1)$ 块, 合计 $2(N-1)/N\\cdot S$ 字节, 上限是 $2S$, 卡数再多也不超过。\nm09 里 N=4、S=32 B: ring 实测每卡发 48 B, 和公式一样; reduce-scatter 24 B 加 all-gather 24 B, 正好是它的两半。',
        },
        {
          title: 'full_loop 是合成章',
          body: '前面每一章的模块都是单独成立的。full_loop 把它们叠进同一个 train_step 函数, 验证叠在一起仍然成立。一步里依次发生:\n- rank 切分: 全局 batch 切给 world 张卡, 每卡算自己那片 (DDP)。\n- micro 累积: 每卡再把自己的数据切成 micro 份串行算, 梯度累加 (梯度累积)。\n- AMP (自动混合精度) 放大与跳步: 用 fp16 算前向反向, loss 先乘一个 scale 放大, 防止小梯度在 fp16 里变成 0。任何一张卡的梯度里出现 Inf 或 NaN, 所有卡一起跳过这一步, scale 减半。\n- 全局裁剪: 梯度已经切成片分在各卡, 全局范数要靠每卡算自己那片的平方和, all-reduce 加起来再开方。超过阈值就按比例缩小。\n- ZeRO 分片 Adam: 每卡只在自己那片 master 权重和 Adam 状态上做更新。\n- 分片 checkpoint: 每卡存自己的分片, 外加一个全局 meta 文件。\ndemo 的数字:\n- 40 步跳了 4 步: 3 次是 fp16 计算真的溢出, 1 次是故意往第 12 步的数据里塞的一个 NaN。最终 scale 稳在 $2^{18}$。\n- 第 20 步存盘、新进程读回来接着训: 训完的 master 权重与不中断的那次差 0.0e+00, 逐位相同。\n同一个函数传 world=1、micro=1、amp=False, 就退化成单卡基线。所以 "分布式结果等于单卡结果" 这个断言, 是同一份代码在检验自己。',
        },
        {
          title: '读真实框架的入口',
          body: '拿到一个真实的训练框架, 别从类名开始读。先对着代码回答七个问题, 每个问题都对应本站的一章:\n- batch 怎么切: 哪些卡拿同一份模型、各算不同数据 (DDP, 梯度累积)。\n- 层怎么切: 一层的矩阵切给几张卡 (TP), 连续几层放一张卡 (PP)。\n- 状态怎么切: 参数、梯度、优化器状态哪些整份存、哪些按卡分片 (ZeRO)。\n- 序列怎么切: 一条长序列怎么分给多卡算注意力 (Ring Attention, Ulysses)。\n- 数用几位存: 哪些算子用 fp16 / bf16 / fp8, master 权重放哪 (混合精度)。\n- 通信怎么走: 每一步在哪里做 all-reduce、reduce-scatter、all-gather、all-to-all。\n- 坏 step 怎么恢复: 溢出跳步、NaN guard、checkpoint 存什么。\n答完这七条, 再去对框架里的类名, 每个类都能归到上面某一条。',
        },
      ],
      links: [
        { from: 'all-reduce', to: 'DDP', body: 'DDP 每步用一次 all-reduce 让每张卡拿到完整的平均梯度。把这次 all-reduce 拆成两半看, 就是先 reduce-scatter 再 all-gather。' },
        { from: 'reduce-scatter + all-gather', to: 'ZeRO / FSDP', body: 'ZeRO / FSDP 的每一步就是在这两个原语之间来回: reduce-scatter 让每卡只留自己那片梯度, 在那片上更新参数; all-gather 再把各卡更新后的参数片拼回完整视图供下一步计算。' },
        { from: 'all-to-all', to: 'MoE / Ulysses', body: 'MoE 把 token 寄去专家所在的卡, Ulysses 把注意力头换到别的卡, 用的是同一个原语。把每张卡要发给每张卡的数据排成一张发货单 in[src][dst], all-to-all 做的就是把它转置: out[dst][src] = in[src][dst]。' },
        { from: 'full_loop', to: 'llm_finetune', body: 'full_loop 的主循环骨架 (切数据、算梯度、同步、更新、存盘) 到这里定型。后面微调阶段换的只是喂进去的数据和算 loss 的方式, 骨架不再变。' },
      ],
      sourceRows: [
        { concept: 'ring all-reduce', code: 'core/collectives.py:ring_all_reduce_sum', takeaway: '真的一块一块沿环传的实现。第 t 步 rank r 把第 (r−t)%N 块发给右邻居。两个阶段各 N−1 步, $N=8$ 时共 14 步; 断言实测发出的字节数等于公式 $2(N-1)/N\\cdot S$。' },
        { concept: '通信计数器', code: 'core/collectives.py:CommCounter', takeaway: '每个原语被调用时, 都把每 rank 平均发送的字节数记到这个计数器上。各章 demo 打印的 "通信量 B/rank" 都来自它。' },
        { concept: '完整一步', code: 'full_loop/demo.py:train_step', takeaway: '一步训练的全部流程写在这一个函数里: 切 batch 给各卡、micro-batch 累积、AMP 溢出跳步、全局范数裁剪、ZeRO 分片 Adam 更新。传 world=1、micro=1、amp=False 就是单卡基线。' },
        { concept: '跨分片求范数', code: 'all_reduce_sum([sum(s ** 2) for s in shards])', takeaway: '梯度已经按卡切成片, 没有哪张卡手里有完整梯度。全局范数的算法: 每 rank 算自己那片的平方和, all-reduce 加成一个数, 开方就是整个梯度的范数。' },
        { concept: '分片 checkpoint', code: 'full_loop/demo.py:save_sharded', takeaway: '每 rank 把自己那片 master 权重和 Adam 状态写进一个文件, 再写一个全局 meta 文件 (step 数、loss scale、数据游标)。恢复后继续训练, 结果与不中断的那次逐位一致。' },
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
      subtitle: '读完你能分清流水线的两笔成本: 卡在空等, 和已前向未反向的激活。也能说出三种调度各自动了哪一笔。',
      tldr: '流水线并行把模型按层切给几张卡, 每段叫一个 stage。它有两笔成本:\n- 气泡: 卡空等的时间占总时间的比例。GPipe 和 1F1B 两种调度完全一样, 都是 $(\\mathrm{PP}-1)/(M+\\mathrm{PP}-1)$, PP 是 stage 数, M 是 micro-batch 数。PP=4、M=8 时是 27.3%。\n- 在途激活: 已经做完前向、还没做反向的 micro-batch, 它的激活必须留在显存里等反向用。1F1B 省的是这一笔, PP=4、M=8 时每卡峰值从 [8,8,8,8] 份降到 [4,3,2,1] 份。\n要真的减气泡只有两条路: 把 M 开大, 或者交错 (每卡放几个不相邻的小 stage)。',
      question: '1F1B 明明不减少气泡, 为什么大家都说它「更快」?',
      evolution: {
        title: '气泡和在途激活的拉锯',
        subtitle: '根问题: 按层切开后, 每个 stage 都要等上游的激活和下游的梯度',
        steps: [
          { name: 'GPipe', year: 2018, pain: '(原点) 按层切后, 同一时刻只有一个 stage 在算', fix: '拆 M 个 micro-batch, 全部前向完再全部反向; 每个 stage 攥着 M 份激活' },
          { name: '1F1B', pain: '想压气泡就得加大 M, 在途激活跟着 M 一起涨', fix: '先做几个前向热身, 之后一前一反交替, stage $s$ 最多攥 $\\mathrm{PP}-s$ 份: [8,8,8,8] 降到 [4,3,2,1]' },
          { name: '1F1B + 大 M', pain: '只改了顺序, 气泡公式没变: PP=4、M=8 仍是 27.3%', fix: '省下的激活显存拿去加大 M: PP=8、M=64 时气泡只剩 9.9%' },
          { name: '交错 1F1B', year: 2021, pain: 'M 加到显存上限, 气泡还剩一截', fix: '每卡拿 v 个不相邻的小 stage, 公式里 M 变成 vM (v=2: 27.3% 降到 15.8%); 跨卡传输变多' },
        ],
      },
      code: 'llm_train/m04_pipeline_parallel/demo.py',
      points: [
        {
          title: '两种调度只差执行顺序',
          body: 'GPipe 和 1F1B 做的计算完全一样, 区别只在每张卡先做哪个、后做哪个。\n先说几个词。流水线并行 (PP) 把模型按连续的层切成 PP 段, 每段叫一个 stage, 一个 stage 放一张卡。一个 batch 再切成 M 个 micro-batch (小批次), 一个接一个流过各个 stage。每个 micro-batch 在每个 stage 上都要做一次前向 (F) 和一次反向 (B)。\n- GPipe: 所有 micro-batch 的前向全做完, 再开始做反向。\n- 1F1B (one forward, one backward): 每张卡先连做几个前向热身 (warmup), 之后严格一个前向、一个反向交替。stage s 的热身个数是 PP−1−s, 正好是它后面还有几个 stage。\ndemo 不真的算梯度, 只排时间表。规则是: 每张卡按固定顺序执行自己的操作, 每个操作在「这张卡空闲了, 并且它依赖的上游操作做完了」的最早时刻开始。反向大约是前向的 2 倍, 所以一个 B 占 2 个时间格, 一个 F 占 1 格。',
        },
        {
          title: '1F1B 省显存, 不省时间',
          key: true,
          body: '1F1B 不减少空等的时间, 它减少的是要留在显存里的激活。\n为什么要留激活: 一个 micro-batch 做完前向后, 反向时还要用到前向算出来的中间结果 (激活)。所以在它的反向做完之前, 这份激活一直占着显存。已前向、未反向的 micro-batch 叫在途的, 它们的激活叫在途激活。\nPP=4、M=8 时, 两种调度的气泡都是 27.3%, 但各 stage 的在途激活峰值不同:\n- GPipe: [8,8,8,8]。前向全做完才开始反向, 每个 stage 最多同时攥着全部 8 份。\n- 1F1B: [4,3,2,1]。stage 0 最多, 越靠后越少。\n规则只有一行: 卡 s 上「已前向未反向」的 micro-batch 不许超过 $\\mathrm{PP}-s$ 份。热身做 PP−1−s 个前向, 再做一个前向就必须等着做一个反向, 所以最多攥 PP−s 份。\n1F1B 对时间的帮助是间接的: 显存省下来, M 就可以开大; 气泡公式的分母里有 M, M 越大气泡越小。PP=8、M=64 时气泡只剩 9.9% (7/71)。',
        },
        {
          title: '交错: 气泡再除以 v',
          body: '交错调度 (interleaved 1F1B) 让每张卡不再拿一段连续的层, 而是拿 v 个不相邻的小 stage。\n举例: PP=4、v=2 时, 模型切成 8 个小 stage (叫虚拟 stage), 编号 0 到 7。卡 0 拿第 0 和第 4 个, 卡 1 拿第 1 和第 5 个, 依此类推。一个 micro-batch 从卡 0 走到卡 3, 再从卡 0 走到卡 3, 绕着卡转 v 圈。\n效果: PP=4、M=8、v=2 时, 气泡从 27.3% 降到 15.8%。公式变成 $(\\mathrm{PP}-1)/(v\\cdot M+\\mathrm{PP}-1)$。\n直觉: 气泡的绝对长度正比于「单个 stage 的计算量」。demo 打印的时间表里, 开头第 s 张卡要空等 s 格, 等前面的 stage 把第一个 micro-batch 的前向一个个做完; 结尾反过来。交错把单个 stage 切小了 $v$ 倍, 总有效工作量没变, 所以公式里的 $M$ 变成 $v\\cdot M$。demo 用事件模拟实测, 气泡和这个公式完全对上。',
        },
        {
          title: '交错的代价',
          body: '交错少等了, 但要付三个代价 (PP=4、M=8、v=2):\n- 在途激活更多: 交错的热身段更长 (见下面「去代码里找什么」), 所以攥着的激活更多。各卡峰值是 [11,9,7,5] 份, 但每份只是半个 stage 的激活。卡 0 最多, 11 × 1/2 = 5.5 个 stage 的激活。普通 1F1B 的卡 0 是 4 个 stage。\n- 跨卡传输更多: 每个 micro-batch 前向加反向要跨卡传的次数, 从 $2(\\mathrm{PP}-1)$ 次变成 $2(\\mathrm{PP}\\cdot v-1)$ 次。因为虚拟 stage 多了, 相邻虚拟 stage 又都在不同卡上。\n- 约束: M 必须是 PP 的整数倍 (demo 有断言)。\n所以它适合 M 已经被显存卡住、加不上去的时候。\ndemo 把跨卡传输的时间记为 0, 真实收益还要扣掉多出来的传输。demo 也没有覆盖 zero-bubble 这类调度: README 提到它们把反向拆成算输入梯度和算权重梯度两部分, 拿来填气泡。',
        },
      ],
      links: [
        { from: 'train-model-parallel', to: 'device_order', body: '上一章只看了时间表长什么样。这一章看时间表是怎么生成的: device_order 给出每张卡的执行顺序, 模拟器按这个顺序和依赖关系排出时间表。' },
        { from: 'warmup 个 F', to: '在途激活上限', body: '一张卡热身时连做 warmup 个前向, 接着再做一个前向才开始第一个反向。所以 warmup+1 就是这张卡上「已 F 未 B」的 micro-batch 数上限。' },
        { from: '在途激活', to: '激活重算', body: '交错多出来的那部分在途激活, 通常用 activation checkpointing (激活重算: 前向时不存中间激活, 反向时再算一遍) 换回显存。' },
      ],
      sourceRows: [
        { concept: '每卡的执行顺序', code: 'm04_pipeline_parallel/demo.py:device_order', takeaway: '生成一张卡的操作清单。GPipe 是全部 F 再全部 B; 1F1B 是先做 warmup 个 F, 之后严格一个 F、一个 B 交替。' },
        { concept: '交错的 warmup', code: '(PP - 1 - s) * 2 + (v - 1) * PP', takeaway: '交错调度下 stage s 的热身前向个数 (Megatron-LM 的公式)。它比普通 1F1B 的 $\\mathrm{PP}-1-s$ 长得多, 交错的在途激活变多就来自这里。' },
        { concept: '虚拟 stage', code: 'k = c * PP + s', takeaway: '卡 s 上的第 c 个小 stage, 在整条流水线里是第 k 个。共 $\\mathrm{PP}\\cdot v$ 个虚拟 stage, 第 k 个住在卡 k % PP 上。' },
        { concept: '事件模拟', code: 'm04_pipeline_parallel/demo.py:simulate', takeaway: '每个操作的开始时刻取「卡空闲的时刻」和「依赖完成的时刻」中晚的那个。跑完后, 气泡 $= 1 - \\text{有效工作量}/\\text{总时长}$, 这个式子在说: 总时长里没在干活的那部分比例。' },
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
      subtitle: '读完你能解释为什么用 WSD 调度训的主干可以在任何时候分叉出一个成品模型, 而 cosine 调度做不到。',
      tldr: '学习率 (lr) 不是一个常数, 训练过程中按一条曲线变化, 这条曲线叫调度。\n- cosine 调度: 每一步的 lr 都写成 $f(\\text{step}/\\text{total})$, 公式里有总步数 total。总步数一改, 整条曲线都变, m10 里前 90 步的 lr 最多变了 5.96e-4。\n- WSD 调度 (warmup 爬升、stable 恒定、decay 退火三段): 稳定段的 lr 就是一个常数, 公式里没有 total。所以稳定段的任何一个 checkpoint 都能接着往下训, 或者分叉出一段短退火做成成品。',
      question: '训到 100% 发现 loss 还在降, 想加训: cosine 和 WSD 各要付出什么?',
      evolution: {
        title: '从一个常数到三段式',
        subtitle: '根问题: 步子大走得快却落不进谷底, 步子小稳却慢, 一个 lr 管不了全程',
        steps: [
          { name: '常数 lr', pain: '(原点) lr 大走得快但落不进谷底, 小了稳却慢', fix: '全程一个值, 只能在快和稳之间挑一头' },
          { name: 'warmup', pain: '一上来就用大 lr, Adam 的二阶矩还没估准, 容易一步走飞', fix: '前几百步从小 lr 线性爬到峰值; 之后停在峰值, 末段在谷底来回震' },
          { name: 'cosine 衰减', year: 2016, pain: 'lr 一直停在峰值, 落不进谷底', fix: 'lr 按 step/total 余弦降到 0; total 写进了每一步, 改总步数整条曲线都变' },
          { name: 'WSD', year: 2024, pain: '训完想加训: 要么 re-warmup 让 loss 反弹, 要么按新 total 重来', fix: '稳定段恒定峰值、不含 total; 任一 checkpoint 接一段短退火 (m10 占 10%) 就是成品' },
        ],
      },
      code: 'llm_train/m10_training_stability/demo.py',
      points: [
        {
          title: 'warmup 在防什么',
          body: 'warmup 是训练开头的几百步, 把 lr 从一个很小的值线性爬到峰值。\n它防的是 Adam 的一个弱点。Adam 给每个参数维护两个滑动平均: 梯度的均值 m 和梯度平方的均值 v, 更新量大致是 m 除以 $\\sqrt{v}$。v 的作用是把步长按每个参数自己的梯度大小归一化。最初几百步 v 还没 "热" 起来, 只平均了很少几步, 估得很不准, 归一化后的步长方差很大。这时配上大 lr, 很容易一步走飞。\n线性 warmup 就是用小步子把这段最危险的路走完, 等 v 估准了再用峰值 lr。',
        },
        {
          title: 'cosine 把总步数写进了每一步',
          key: true,
          body: 'cosine 调度的公式是 $\\text{lr} = f((\\text{step} - \\text{warmup})/(\\text{total} - \\text{warmup}))$。这个式子在说: 先算当前走到了全程的几分之几 (progress), 再把 progress 代进半个余弦, lr 从峰值平滑降到 0。总步数 total 出现在 progress 的分母里, 所以每一步的 lr 都依赖它。total 一变, 整条曲线都跟着变。\nm10 里把 total 从 100 改成 200, 看前 90 步的 lr 变了多少:\n- cosine: 最大差 5.96e-4 (峰值 lr 是 1e-3, 差了一半以上)。\n- WSD: 差 0.00e+00, 一步都没变。\n所以训完想给 cosine 加训, 只剩两条路: 从 lr 已经降到 0 的状态重新 warmup (re-warmup), loss 会先反弹; 或者按新的 total 从头再训一遍。',
        },
        {
          title: 'WSD 的工程价值',
          body: 'WSD 的稳定段 lr 恒定, 不依赖总步数。MiniCPM / DeepSeek-V3 / Kimi 用的都是这一类调度, 工程上这样用:\n- 主干: 一直用峰值 lr 往前训, 不设终点。\n- 要 "成品": 从稳定段的任意一个 checkpoint 分叉出来, 接一段短退火 (lr 快速降到接近 0), 退火完就是一个可以评测、可以发布的模型。m10 里退火段占总步数的最后 10%。\n数据追加、scaling law 实验 (比较不同规模的模型) 都不用重训主干。退火起点前的那个 checkpoint, 是所有分支的公共祖先。\n代价: 要成品或要评测, 都得从稳定段 checkpoint 另跑一段退火, m10 里是总步数的 10%。\n边界: 真实 WSD 的退火形状常用 1−sqrt 或指数, m10 用的是线性。',
        },
      ],
      links: [
        { from: 'train-precision-stability', to: 'warmup', body: '上一章的梯度裁剪 (clip) 和 NaN guard, 和这一章的 warmup 是同一类工具: 都是为了别让某一个坏 step 毁掉整条训练。warmup 防开头, 另两个防中途。' },
        { from: 'decay_start', to: 'checkpoint', body: 'decay_start 是退火开始的那一步。它前面的最后一个稳定段 checkpoint 最值钱: 续训从它出发, 每一个退火分支也从它出发, 它是所有分支的公共祖先。' },
        { from: 'WSD 退火', to: 'Muon / AdamW', body: '调度管 lr 怎么随时间变, 优化器管拿到 lr 之后怎么更新参数, 两者互不干涉, 可以任意搭配。Kimi K2 用的就是 Muon 优化器加 WSD 调度。' },
      ],
      sourceRows: [
        { concept: 'warmup + cosine', code: 'm10_training_stability/demo.py:warmup_cosine_lr', takeaway: '先算 $\\text{progress} = (\\text{step} - \\text{warmup})/(\\text{total} - \\text{warmup})$, 再代进余弦。total 在分母里, 每一步的 lr 都依赖它。' },
        { concept: 'WSD', code: 'm10_training_stability/demo.py:wsd_lr', takeaway: '稳定段的代码只有一行: 直接 return base_lr。这一行里没有 total_steps, 所以稳定段的 lr 和总步数无关。' },
        { concept: '退火起点', code: 'decay_start = total_steps - int(total_steps * decay_frac)', takeaway: '退火起点等于总步数减去退火段长度 (总步数的 decay_frac, 默认 0.1)。整个 WSD 里只有这一个量依赖总步数, 而且只影响最后约 10% 的步。' },
        { concept: '两种调度的实测差', code: 'total=100 → 200, 前 90 步 lr 最大变化', takeaway: '总步数从 100 改到 200, 前 90 步的 lr 最大变化: cosine 5.96e-04, WSD 0.00e+00。' },
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
      subtitle: '读完你能解释一个数只用 4 bit 存时, 为什么必须一小块元素共用一个缩放系数, 以及 MXFP4 和 NVFP4 两种格式的差距全部来自哪一处。',
      tldr: '- 一个数只用 4 bit 存 (FP4) 时, 能表示的最小非零值是 0.5、最大是 6, 动态范围只剩 12 倍。单靠它什么都存不了。\n- 办法是每 16~32 个元素共享一个缩放系数 (scale): 先把这一小块按它的最大值缩放到网格里, 再舍入。块越小, 一个异常大的值连累的邻居越少。\n- MXFP4 和 NVFP4 都是这么做的, 两者的差距来自 scale 自己的精度: MXFP4 的 scale 只能取 2 的幂, NVFP4 的 scale 带 3 位尾数。',
      question: '同样是 4 bit + block scale, 为什么 NVFP4 的误差明显低于 MXFP4? INT4 什么时候反而更好?',
      evolution: {
        title: 'scale 越切越细, 位宽越砍越少',
        subtitle: '根问题: 位宽越少, 一张网格能覆盖的范围越窄, 张量里大小悬殊的数挤不进去',
        steps: [
          { name: 'FP8 per-tensor', year: 2022, pain: '(原点) FP8 范围有限, 张量得先缩放进格式能表示的区间', fix: '整个张量按 amax 共用一个 scale; 一个 outlier 就把其余元素挤向 0' },
          { name: 'block scale', year: 2024, pain: 'outlier 到 1e5 倍时, 其余 token 的相对误差从 0.026 涨到 0.126', fix: '每 128 个元素一个 scale, outlier 只连累同块邻居: 误差留在 0.026' },
          { name: 'MXFP4', year: 2023, pain: '再砍到 4 bit, 单个元素的动态范围只剩 12 倍', fix: '每 32 个元素共用一个 E8M0 scale; scale 只能取 2 的幂, 42% 的 block 最大值被截断' },
          { name: 'NVFP4', pain: 'scale 只能是 2 的幂, amax 贴不准网格顶端的 6', fix: 'scale 换成带 3 位尾数的 E4M3, block 缩到 16: 误差 0.113 → 0.095' },
        ],
      },
      code: 'llm_train/m13_fp8_training/demo.py · llm_train/m15_fp4_microscaling/demo.py · llm_train/core/numerics.py',
      points: [
        {
          title: 'scale 由 block 里最大的那个元素定',
          key: true,
          body: '低精度格式能表示的数值范围很窄, 张量里的数要先缩放进这个范围再存。缩放系数叫 scale, 由一组元素里绝对值最大的那个 (amax) 决定: scale = amax / 格式的最大值。这样最大的那个元素正好落在网格顶端, 其余元素除以 scale 后再舍入到网格上。\n共享 scale 的那一组元素叫一个 block。整个张量共用一个 scale 叫 per-tensor; 每 128 个元素一个叫 block scaling。\n问题出在 outlier (个别异常大的值): 它一个人把 amax 顶得很高, 同 block 的其他元素除以 scale 后都变得很小, 挤在网格靠近 0 的几个格点上, 甚至直接变成 0。block 越小, 被它连累的邻居越少。\n- FP8: m13 往 32 个 token 的激活里放 3 个孤立的 outlier, 看其余 29 个 token 的矩阵乘输出误差。outlier 要大到约 1e5 倍才显形: 这时 per-tensor 的相对误差是 0.126, 每 128 个元素一个 scale 的是 0.026。1e4 倍以内两者打平, 因为 E4M3 自带约 $2^{15}$ 的动态范围。\n- FP4: 元素格式 E2M1 的网格只有 $\\pm\\{0.5,1,1.5,2,3,4,6\\}$ 和 0, 动态范围 12 倍。不切成小 block, 一个张量里稍微悬殊一点的数就装不下, 根本用不了。',
        },
        {
          title: 'scale 自己也有精度',
          body: '格式名直接写位数: E8M0 是 8 位指数、0 位尾数; E4M3 是 4 位指数、3 位尾数; FP4 的元素本身是 E2M1, 2 位指数、1 位尾数。\n两种 FP4 格式的元素都是 E2M1, 区别只在 scale 怎么存:\n- MXFP4: 每 32 个元素一个 E8M0 的 scale。没有尾数, scale 只能取 2 的幂。于是 amax 除以 scale 落在 [4, 8) 的哪个位置全凭运气, 大于 6 的直接饱和截断到 6。m15 里 42% 的 block 的最大值被截断了。\n- NVFP4: 每 16 个元素一个 E4M3 的 scale, 带 3 位尾数, 能把 amax 几乎精确地贴到网格顶端的 6。E4M3 的范围只到 448, 所以还要再乘一个整张量的 FP32 scale 把它搬进这个范围。\nm15 在高斯分布的权重上量相对量化误差 $\\|Q(x)-x\\|/\\|x\\|$ (量化再还原后的值与原值之差的范数, 除以原值的范数):\n- 各用默认 block: MXFP4 (block 32) 0.113, NVFP4 (block 16) 0.095。\n- 同为 block 16, 只换 scale 格式: E8M0 scale 0.1138, E4M3 scale 0.0947。block 一样大时差距还在, 所以差距全部来自 scale 的精度。',
        },
        {
          title: '格式要对着数据分布选',
          body: '同样是 4 bit, 还有整数格式 INT4 可选: 每个 block 一个 scale, 元素量化到 −7 到 7 的整数, 格点是均匀间距的。FP4 的格点 $\\{0.5,1,1.5,2,3,4,6\\}$ 越大越稀, 近似对数间距。哪个更准, 看数据长什么样。比的仍是相对量化误差:\n- 高斯分布的权重 (数值集中在 0 附近, 很少有大值): INT4 0.096, MXFP4 0.113, INT4 低 15%。数据集中时, 均匀网格更划算。\n- 重尾数据 (激活、梯度, 偶尔出现很大的值): MXFP4 0.142, INT4 0.147, 对数间距的浮点网格低约 4%。浮点网格更擅长这种偶尔冒出大值的数据。\n两者的位数也不同: 把 scale 摊到每个元素上, MXFP4 是 4 + 8/32 = 4.25 bit, INT4 是 4.50 bit。',
        },
      ],
      links: [
        { from: 'train-precision-stability', to: 'FP8 block scaling', body: '上一章的 BF16、FP8 模拟和这一章的 FP4 用的是同一个函数 fake_quant_float。一个浮点格式只由三个数决定: 尾数位数、最大值、最小指数。换格式就是换这三个数。' },
        { from: 'quant_blockwise', to: 'mxfp4 / nvfp4', body: 'core 里的 quant_blockwise 是通用的 block 量化: 每块按 amax 算 scale, 缩放、舍入、还原。m15 的 mxfp4 和 nvfp4 两个函数做的是同样的事, 只有算 scale 的那一行不一样。' },
        { from: 'FP4 权重', to: 'llm_infer 量化', body: '本章只量了一次性量化的误差。FP4 目前主要用在推理时的权重和 QAT (量化感知训练)。全程用 FP4 做预训练还要加随机舍入、Hadamard 旋转打散 outlier 等技巧, 本章没做。' },
      ],
      sourceRows: [
        { concept: 'block scaling', code: 'core/numerics.py:quant_blockwise', takeaway: '把张量拍平后每 block 个元素切成一行, 每行算 scale = amax / max_val (这一块的最大绝对值除以格式的最大值), 除以 scale 后舍入, 再乘回来。block 大于等于张量大小时, 整个张量一个 scale, 就是 per-tensor。' },
        { concept: 'MXFP4 的 scale', code: 'scale = 2.0 ** (np.floor(np.log2(amax)) - 2)', takeaway: 'scale 取 amax 的对数向下取整再减 2, 再做 2 的幂, 所以只能是 2 的幂 (E8M0)。减 2 是因为 E2M1 的最大值 6 写成 $1.5\\cdot 2^2$, 最大指数是 2; 这样 amax 除以 scale 落在 [4, 8)。' },
        { concept: 'NVFP4 的 scale', code: 'm15_fp4_microscaling/demo.py:nvfp4', takeaway: '每块的 scale 是 amax / 6, 但它要存成 E4M3, 范围只到 448。所以先用一个整张量的 FP32 scale 把它搬进这个范围, 量化成 E4M3, 再乘回去。' },
        { concept: 'FP8 消融', code: 'm13_fp8_training/demo.py:train', takeaway: 'FP8 的训练消融, 比 200 步后的验证 loss, 相邻两个实验只动一个旋钮。全程 E4M3 时 per-tensor 4.64e-4, block 4.61e-4, 干净数据上打平。去掉 fp32 master 权重后 loss 差 584 倍。' },
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
      subtitle: '读完你能说清 Muon 优化器在什么问题上赢 Adam、在什么问题上输, 以及为什么它只用在 2-D 权重矩阵上。',
      tldr: '- Adam: 逐个元素看梯度, 给每个参数单独定步长。\n- Muon: 只管 2-D 权重矩阵。它把动量矩阵的奇异值 (矩阵在各个方向上的放大倍数) 全部拉成 1 再拿去更新, 这一步叫正交化。结果是每个奇异方向迈同样大的步子。正交化不做 SVD, 用 5 步只含矩阵乘的 Newton–Schulz 迭代近似。\n损失面的病态方向 (特别陡或特别平的方向) 不与坐标轴对齐时 Muon 赢, demo 里 loss 低 9.2 倍; 对齐时 Adam 赢, 好 1.9 倍。',
      question: 'Adam 已经给每个参数单独调步长了, 为什么还会被「病态方向」拖住? Muon 什么时候反而不如 Adam?',
      evolution: {
        title: '从逐元素缩放到逐奇异方向缩放',
        subtitle: '根问题: 损失面在不同方向上弯曲程度差几个数量级, 同一个步长管不住所有方向',
        steps: [
          { name: 'SGD', pain: '(原点) 各方向曲率差几个数量级, 一个 lr 管不住所有方向', fix: '所有方向同一步长: 陡的方向来回震, 平的方向走不动' },
          { name: 'Adam', year: 2014, pain: '陡的方向和平的方向需要不同步长', fix: '每个元素按自己梯度平方的滑动平均 (二阶矩) 缩放步长; 只修得了沿坐标轴的病态' },
          { name: 'Muon', year: 2024, pain: '病态方向是几个坐标的线性组合时, 逐元素缩放够不着', fix: '动量矩阵用 5 步 Newton–Schulz (只含矩阵乘的迭代) 把奇异值都推到 1 附近, 各奇异方向同步长: 旋转问题上 loss 低 9.2×' },
          { name: 'MuonClip', year: 2025, pain: '满秩更新让 attention logit (softmax 前的分数) 越涨越大, softmax 成了 one-hot', fix: 'max logit 超过 $\\tau$ 就把 $W_q$、$W_k$ 各乘 $\\sqrt{\\tau/S_{\\max}}$, 最大 logit 从 752.1 压到 100.0' },
        ],
      },
      code: 'llm_train/m14_muon_optimizer/demo.py',
      points: [
        {
          title: '梯度的奇异值差几个数量级',
          body: 'Muon 的出发点: 一个权重矩阵的梯度, 在不同方向上的大小差得很远, 更新被少数几个大方向主导。\n怎么看「方向」: 任何矩阵都能做 SVD (奇异值分解), 写成 $G = U\\Sigma V^\\top$。这个式子在说: 矩阵 G 的作用可以拆成一组互相垂直的方向 (U 和 V 的列), 每个方向上放大多少倍, 由对角矩阵 $\\Sigma$ 里的一个数给出, 这个数叫奇异值。\n梯度矩阵的奇异值常常差几个数量级。直接拿梯度 (或它的动量) 去更新, 大奇异值的那几个方向步子很大, 小奇异值的方向几乎不动。更新被头几个大方向吃掉了。\nMuon 的做法是把 $\\Sigma$ 里的数全换成 1, 更新量变成 $UV^\\top$。方向还是梯度给出的那些方向, 但每个奇异方向迈的步子一样大。奇异值全是 1 的矩阵叫正交矩阵 (它的行或列两两垂直、长度都是 1), 所以这一步叫正交化。用数学的说法, 这相当于在谱范数 (矩阵最大的那个奇异值) 下做最陡下降。\n实际被正交化的是动量矩阵, 不是单步的梯度, 见下面「去代码里找什么」的「一步更新」。',
        },
        {
          title: '不做 SVD 也能正交化',
          body: '精确的 SVD 在 GPU 上很慢。Muon 改用 Newton–Schulz 迭代, 只用矩阵乘法, 就把奇异值推到 1 附近。\n- 先归一: 把矩阵除以它的 Frobenius 范数 (所有元素平方和再开根号)。这样所有奇异值都 ≤ 1, 后面的多项式迭代才不会发散。\n- 再迭代 5 次: $X \\leftarrow aX + (bA + cA^2)X$, 其中 $A = XX^\\top$, 系数 $(a,b,c) = (3.4445, -4.7750, 2.0315)$。这个式子在说: 每迭代一次, 奇异方向 U、V 不动, 每个奇异值 σ 被换成 $a\\sigma + b\\sigma^3 + c\\sigma^5$。σ 很小时 $a\\sigma$ 那一项占主导, 每步约放大 3.4 倍; 大的 σ 被压在 1 附近。\n这组系数是为「快速把奇异值推进 [0.7, 1.2]」调的, 不追求精确收敛到 1。\nm14 对一个 16×32、条件数 1e4 (最大奇异值是最小的 1 万倍) 的矩阵迭代 5 步:\n- 最小奇异值: 从 1e-4 抬到 0.041 (1 步、3 步后分别是 0.000、0.003)。这么小的方向 5 步来不及到 1, 但也被放大了两个数量级以上。\n- 最大奇异值: 一直压在 1.16 附近。\n- 方向: 与精确 $UV^\\top$ 的方向余弦 0.880 (1 表示完全同向)。',
        },
        {
          title: '病态落在谁的轴上, 谁赢',
          key: true,
          body: 'Adam 和 Muon 谁赢, 要看损失面的病态方向是不是正好沿着坐标轴。\n病态的意思是: 损失面在某些方向特别陡, 在另一些方向特别平, 一个学习率很难同时照顾好。\n- Adam 的缩放是逐元素的: 它给每个参数记一个梯度平方的滑动平均 v, 步长除以 $\\sqrt{v}$。这只能修「对角」的病态, 意思是陡的方向和平的方向恰好各自对应单个参数 (单个坐标轴)。病态方向一旦是多个坐标的线性组合 (斜着的方向), 逐元素缩放就无能为力。\n- Muon 的正交化只看奇异方向, 和坐标轴怎么摆无关, 这叫与基无关。\nm14 的实验是一个 32 维输入的矩阵回归。输入各主轴上的标准差从 1 降到 0.01, 协方差的条件数是 1e4。「非轴对齐」版本再给输入乘一个随机旋转矩阵, 让病态方向不再是单个坐标。两个优化器各跑 150 步, 各扫 5 个 lr (0.03 到 3.0) 取最好, 比最后的 loss:\n- 非轴对齐: Muon 4.93e-04, Adam 4.52e-03, Muon 好 9.2×。\n- 轴对齐: Adam 5.54e-04, Muon 1.08e-03, Adam 好 1.9×。\nAdam 按坐标轴缩放, Muon 按奇异方向缩放。病态落在谁的轴上, 谁就赢。\n两边 weight decay 都是 0, 只比缩放方式这一处差别; 1-D 的偏置两边都用 Adam。这是线性回归玩具问题, 「真实网络上 Muon 省约一半 FLOPs」是 Moonlight / Kimi 的报告结论, 这个 demo 证明不了。',
        },
        {
          title: '大规模要配 QK-clip',
          body: 'Muon 用在大模型上, 还要配一个叫 QK-clip 的保护, 防止注意力分数爆掉。\nattention logit 是 softmax 之前的分数, 由 query 和 key 点乘得到 (q·k)。Muon 的更新是满秩的: 正交化之后所有奇异方向都迈同样大的步子, 没有哪个方向被压着不动。README 指出, 大规模训练时 attention logit 容易因此越涨越大。logit 太大, softmax 就成了 one-hot (几乎全部概率压在一个位置上), 梯度消失, 注意力学不动了。\nm14 只演示裁剪本身, 不演示 logit 怎么涨上去: 手工造一组偏大的 $W_q$、$W_k$ (随机矩阵乘 3), max logit 是 752.1, softmax 最大概率的均值是 1.000。已经是 one-hot, 梯度全没了。\n- Kimi K2 的 MuonClip: 每步更新后算出这个 batch 上最大的 logit $S_{\\max}$。若它超过阈值 $\\tau$, 就把 $W_q$、$W_k$ 各乘 $\\sqrt{\\tau/S_{\\max}}$。为什么各乘一个平方根: logit 里 $W_q$ 和 $W_k$ 各出现一次, 两边各乘一个平方根, 乘起来正好缩小 $\\tau/S_{\\max}$ 倍。demo 取 $\\tau$=100, 裁剪后正好 100.0。真实的 MuonClip 按 head 分别做。\n- embedding、输出头、norm、bias: 仍然用 AdamW。norm 和 bias 是向量, 正交化是矩阵概念, 对向量参数没有意义; embedding 和输出头虽然是矩阵, README 指出用 Muon 反而更差。',
        },
      ],
      links: [
        { from: 'llm_basic optim.py', to: 'Muon', body: 'llm_basic 的 optim.py 里是 Adam, 它给每个参数存 m (梯度的滑动平均) 和 v (梯度平方的滑动平均) 两份状态。Muon 只存动量一份, 优化器显存反而更省。' },
        { from: 'newton_schulz', to: 'GPU 友好', body: '精确的 $UV^\\top$ 要做 SVD, GPU 上很慢。newton_schulz 只用矩阵乘, 正好是 GPU 最擅长的运算, 5 步的开销相对一次前向加反向不到 1%。' },
        { from: '0.2·√max(n,m)', to: 'AdamW 学习率', body: '这是 Moonlight 的缩放: 正交化后的更新再乘 $0.2\\cdot\\sqrt{\\max(n,m)}$ ($n$、$m$ 是权重矩阵的形状), 让 Muon 更新的 RMS (每个元素的典型大小) 接近 AdamW 的约 0.2。这样学习率和 weight decay 可以直接沿用 AdamW 的。' },
      ],
      sourceRows: [
        { concept: 'Newton–Schulz', code: 'm14_muon_optimizer/demo.py:newton_schulz', takeaway: '先除以 Frobenius 范数, 保证所有奇异值 $\\sigma \\le 1$。再迭代 5 次, 每次对每个奇异值作用多项式 $a\\sigma + b\\sigma^3 + c\\sigma^5$, 奇异方向不变。全程只有矩阵乘, 不做 SVD。' },
        { concept: '一步更新', code: 'm14_muon_optimizer/demo.py:Muon', takeaway: '先把梯度累进动量 buf (动量系数 0.95), 再对「当前梯度 + 0.95 × buf」(Nesterov 式动量) 做正交化, 乘 $0.2\\cdot\\sqrt{\\max(n,m)}$, 最后减到权重上。优化器状态只有 buf 一份。' },
        { concept: 'QK-clip', code: 'm14_muon_optimizer/demo.py:qk_clip', takeaway: '$\\eta = \\min(1, \\tau/S_{\\max})$: 最大 logit 没超过阈值时 η=1, 什么都不动。超过时 $W_q$、$W_k$ 各乘 $\\sqrt{\\eta}$, logit 正好缩小 η 倍。' },
        { concept: '对照实验', code: 'm14_muon_optimizer/demo.py:run', takeaway: 'rotate=True/False 对应非轴对齐、轴对齐两种病态。两个优化器各自扫 5 个 lr 取最好的再比, 否则比的是谁的 lr 碰巧调得好。' },
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
      subtitle: '读完你能算出 Ulysses 和 Ring 各自的每卡通信量, 并说清为什么实战里两者要叠着用。',
      tldr: 'Ulysses 是另一种把长序列切给多张卡的办法。它在注意力前后做 all-to-all (每张卡给每张卡各发一份不同的数据), 换一种切法:\n- 换之前: 每卡一段序列、全部头。\n- 换之后: 每卡完整序列、一部分头。\n多头注意力里各个头本来互不相干, 所以每卡对完整序列算自己那几个头就行, 注意力核一行都不用改。它的每卡通信量随卡数 P 增大反而下降: demo 里 P=8 时每卡 7168 字节, Ring Attention 是它的 4 倍。代价是 P 必须整除头数, 跨机器时也不如 Ring 友好。',
      question: 'Ulysses 的每卡通信量随卡数 P 增大反而下降, 为什么大家没有全部换成它?',
      evolution: {
        title: '长序列从切段到切头',
        subtitle: '根问题: 激活随序列长度涨, 长上下文的一条序列一张卡就放不下',
        steps: [
          { name: '整条序列放一卡', pain: '(原点) 激活随序列长度 T 涨, 长上下文一张卡放不下', fix: '整条序列一卡算; T 一长就 OOM (显存不够), DP/TP/PP 都切不到序列这一维' },
          { name: 'Ring Attention', year: 2023, pain: '序列要分到多卡, 可每个 query 都要看全部 key', fix: '序列切 P 段, KV 块沿环传; online softmax 每收到一块就按新的最大值修正旧结果再合并, 结果精确相等' },
          { name: 'zigzag', pain: '因果 mask 下连续切, 末卡最忙: 墙钟 256 只降到 228', fix: '切 2P 段一头一尾配对, 每卡每轮活一样多: 墙钟从 228 降到 132' },
          { name: 'Ulysses', year: 2023, pain: 'Ring 每卡通信随 P 涨, P=8 时 28672 B/rank', fix: 'all-to-all (每卡给每卡各发一份) 把切序列换成切头, 注意力核不改; 通信随 P 降 (P=8 时 7168 B)' },
        ],
      },
      code: 'llm_train/m16_ulysses_sequence_parallel/demo.py · llm_train/m12_sequence_parallel/demo.py',
      points: [
        {
          title: '换切法, 不换数学',
          key: true,
          body: 'Ulysses 只换了数据在卡之间的切法, 注意力的计算本身一点没变。\n先定符号: $P$ 张卡, $T$ 个 token, $H$ 个注意力头, 每头 $d$ 维。多头注意力里每个头各算各的, 头与头之间不交换信息。\n- 注意力之外: 每卡持有 [T/P, H, d], 一段序列、全部头。MLP、norm 这些层对每个 token 单独算, 按序列切不需要任何通信。\n- 进注意力前: 注意力要让每个 token 看到前面整段序列, 按序列切就不够了。于是对 Q、K、V 各做一次 all-to-all, 每卡变成 [T, H/P, d]: 完整序列, 但只有 $H/P$ 个头。每卡直接跑一次普通的因果 FlashAttention, 算完再用一次 all-to-all 换回 [T/P, H, d]。\n举例: P=2、H=8 时, 卡 0 原来拿前一半 token 的全部 8 个头。all-to-all 之后, 卡 0 拿全部 token 的第 0 到 3 个头, 卡 1 拿第 4 到 7 个头。\n头与头之间本来互不相干, 所以结果和完整注意力对得上 (m16: P=2/4 差 0.0e+00, P=8 差 4.4e-16)。\n因果 mask 下每卡都算完整序列, 活天然一样多, 不需要 zigzag (把序列一头一尾配对来平衡负载的切法)。\ndemo 只有前向; 反向是方向相反的同样 4 次 all-to-all。',
        },
        {
          title: '通信量随 P 下降',
          body: 'Ulysses 的每卡通信量随卡数 P 增加而下降, Ring 的则上升。\nUlysses 每卡发 $4(P-1)/P^2$ 份完整张量 (一份完整张量指整个 [T, H, d] 的 Q、K、V 或输出)。这个式子这样来:\n- 一共 4 次 all-to-all: Q、K、V 各一次, 输出换回来一次。\n- 每次, 每卡手上只有完整张量的 $1/P$。其中属于自己那几个头的部分留下, 其余 $(P-1)/P$ 发给别的卡。\n- 乘起来是 $4 \\times (1/P) \\times (P-1)/P$。P 越大, 每卡手上那片越小, 要发的也越少。\nRing 每卡要传 P−1 轮, 每轮把一块 KV (完整 K、V 的 1/P) 交给邻居, 合计是完整 KV 的 $(P-1)/P$。P 越大越接近整份 KV, 只升不降。\nm16 里 T=32、H=8、d=8 (float64, 每个数 8 字节), P=2/4/8 时每 rank (每张卡) 发的字节数:\n- Ulysses: 16384 / 12288 / 7168\n- Ring: 16384 / 24576 / 28672。P=8 时是 Ulysses 的 4.0 倍。\n两列都是 demo 用通信计数器实测的, 和公式逐字节相等; Ring 那一列是逐头调用上一章 m12 的 ring_attention 得到的。真实实现常把 Q、K、V 拼起来一次发完, demo 分成 3 次发。',
        },
        {
          title: '两个硬约束',
          body: 'Ulysses 有两个硬约束, 所以实战里它通常和 Ring 叠着用。\n- P 必须整除头数: 每卡要分到整数个头。m16 里 P=16、H=8, 8 个头切不成 16 份, 直接 AssertionError。GQA/MQA 下限制更紧。GQA (grouped-query attention) 让几个 Q 头共用一组 K 和 V, MQA 是所有 Q 头共用一组, 所以 KV 头比 Q 头少得多。README 举的例子是 64 个 Q 头只有 8 个 KV 头, 这时 KV 要么复制, 要么 P 只能开到 8。\n- 跨机不友好: all-to-all 要每张卡和其它所有卡交换数据, 吃的是对分带宽 (把集群切成两半, 两半之间的总带宽)。Ring 只和环上的邻居点对点传, 不需要所有人互相发。\n所以实战常见的是机内 8 路 Ulysses × 机间 4 路 Ring: 机器内部用 Ulysses, 跨机器那一层交给 Ring, 两者的并行度相乘, 一共切 32 路。',
        },
      ],
      links: [
        { from: 'train-moe-seq', to: 'Ring Attention', body: '上一章的 Ring Attention 让每卡的 Q 留在原地, KV 块沿环一块块流过去。Ulysses 不这样传, 它用 all-to-all 让所有卡一次性交换数据换位。' },
        { from: 'core/collectives.py:all_to_all', to: 'seq_to_head', body: 'seq_to_head 用的 all_to_all 和 MoE 的 dispatch 是同一个通信原语。代码里写成 out[dst][src] = in[src][dst]: 卡 src 准备发给卡 dst 的那份数据, 到了卡 dst 手上的第 src 格。' },
        { from: '[T/P, H, d]', to: '[T, H/P, d]', body: '左边是每卡一段序列、全部头, 右边是每卡完整序列、一部分头。盯住这个 shape 变化, 整个算法就看懂了。' },
      ],
      sourceRows: [
        { concept: '序列 → 头', code: 'm16_ulysses_sequence_parallel/demo.py:seq_to_head', takeaway: '每张卡 (rank s) 把手上的 H 个头分成 P 组, 第 j 组发给 rank j。每卡收齐 P 段序列后, 按顺序沿序列维拼起来, 得到 [T, H/P, d]。' },
        { concept: '头 → 序列', code: 'm16_ulysses_sequence_parallel/demo.py:head_to_seq', takeaway: '完全对称的逆变换: 每卡把完整序列切回 P 段, 第 s 段发给 rank s, 收到后沿头维拼回 [T/P, H, d]。' },
        { concept: '整除约束', code: 'assert H % P == 0', takeaway: '每卡要分到整数个头。8 个头切不成 16 份, demo 里故意用 P=16 触发这个 AssertionError。' },
        { concept: '通信账', code: '4 * (P - 1) / P * shard_bytes', takeaway: 'shard_bytes 是一张卡手上一份 Q (或 K、V) 的字节数。4 次 all-to-all, 每次只发其中不属于自己的 $(P-1)/P$。' },
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
      subtitle: '读完你能说清改过几个字的转载为什么要靠 MinHash-LSH 才抓得到, 以及把多篇文档拼进一行之后要补哪三样才和逐篇训练一样。',
      tldr: '- 去重: 精确哈希只认逐字相同的副本, 改一个词就漏掉。MinHash-LSH 比的是两篇文档有多像, 改过几个字的转载也抓得到: m17 里召回从 0.258 升到 0.967。\n- 配比: 按原始比例采样, 小语料几乎见不到。数据源温度配比 $p_i \\propto n_i^{1/T}$ 把小语料的占比抬上来, 代价是它被重复很多遍。这里的温度管各来源的占比, 和解码时的采样温度不是一回事。\n- packing: 把多篇文档拼进同一行训练, 省掉补位用的 padding, 有效 token 从 0.285 提到 0.990。必须加文档 mask, 否则同一行的文档会互相看见。',
      question: '转载时改了几个字, 去重还抓得到吗? 把几篇文档拼进同一行训练, 它们会互相看见吗?',
      evolution: {
        title: '把 pad 一点点挤出去',
        subtitle: '根问题: GPU 吃定长矩阵, 文档却长短不一, 补进去的 pad 照样花算力',
        steps: [
          { name: '每篇一行补 pad', pain: '(原点) 文档中位数 28 token, 一行 128, 短的要补齐', fix: '每篇占一行补 pad 到 L; 真 token 只占 0.285, 其余算力花在 pad 上' },
          { name: '动态 padding', pain: '大半位置是 pad, 算了白算', fix: 'batch 内只补到最长那篇: 0.406; 一篇长文档还是拖着整批' },
          { name: '直接拼接', pain: '只要一篇占一行, pad 就去不掉', fix: '文档首尾相接切成定长行, 没有 pad (GPT-2/3 风格); 同一行的文档互相看得见' },
          { name: 'FFD + 文档 mask', pain: '只有因果 mask 时后一篇看得见前一篇: 输出差 3.0', fix: 'FFD 整篇装箱 (0.990), 文档 mask 只许看同篇, 再重置位置、屏蔽跨篇 label' },
        ],
      },
      code: 'llm_train/m17_data_pipeline/demo.py · llm_train/m18_sequence_packing/demo.py',
      points: [
        {
          title: '精确哈希只抓逐字副本',
          body: '去重要抓的是转载: 同一篇文章被转了几十次, 每次改几个字。m17 造了一个有答案的语料: 300 篇原件, 外加 120 篇应该删掉的重复, 其中 30 份是逐字副本, 90 份是轻改转载 (每个词约有 4% 的概率被替换、删除或在后面插入一个词)。\n两篇文档像不像, 用 Jaccard 相似度衡量: 把每篇切成所有连续 5 个词的片段 (5-gram), 两篇的 5-gram 集合取交集大小除以并集大小。轻改转载与原件的 Jaccard 均值 0.69。\n召回是这 120 篇里被抓到的比例:\n- 精确哈希 (整篇算一个 crc32, 和见过的相同就删): 召回 0.258。改一个词整篇的哈希就全变了, 所以只抓到 30 份逐字副本, 外加 1 篇恰好一个词都没被改到的转载。\n- MinHash-LSH: 召回 0.967, 误杀 0 篇。漏掉的 4 篇, 真实 Jaccard 在 0.48 到 0.52 之间, 正好卡在 0.5 的校验线上, 签名估计有一点误差就被判到线下。',
        },
        {
          title: 'MinHash 估相似度, LSH 只决定谁被比较',
          body: '两篇两篇直接算 Jaccard 太慢: 每篇要存整个 5-gram 集合, 还要两两全比。MinHash 和 LSH 分别解决这两个问题。\nMinHash 把集合压成一个短签名。它靠一个性质: 把所有 5-gram 的编号随机打乱一次 (随机置换), 两个集合各自取打乱后最小的那个元素, 两者相同的概率恰好等于 Jaccard。打乱 128 次, 每篇文档就得到 128 个最小值, 这就是它的签名。两篇签名里相同位置的比例, 就是 Jaccard 的估计。\nLSH (局部敏感哈希) 决定谁和谁要比。两两全比是 $O(n^2)$。LSH 把 128 位签名切成 $b=32$ 段, 每段 $r=4$ 位; 两篇只要有任何一段 4 位完全相同, 就落进同一个桶, 成为候选对。\n两篇成为候选的概率是一条 S 形曲线 $1-(1-s^r)^b$ ($s$ 是两篇的真实 Jaccard)。这个式子在说: 一段 4 位全相同的概率是 $s^4$, 32 段里至少有一段相同的概率, 就是 1 减去 32 段全都不同的概率。曲线的拐点约在 $(1/b)^{1/r} \\approx 0.42$。$s$ = 0.2 / 0.4 / 0.6 / 0.8 时成候选的概率是 0.05 / 0.564 / 0.988 / 1.0: 不像的几乎不会被比, 像的几乎一定被比。\n成了候选还要校验: 用签名估出的 Jaccard ≥ 0.5 才算重复。所以有两个阈值, 分开调:\n- S 曲线的拐点: 管有多少对会进入比较, 决定召回和算量。\n- 校验线 0.5: 管最终判不判重复。',
        },
        {
          title: '配比: 数据源温度配比',
          body: '数据来自好几个来源, 大小悬殊。m17 里 5 个来源共 1000M token, web 占 800M, math 只有 2M。按原始比例采样, math 只占 0.2%, 模型几乎见不到它。\n数据源温度配比 $p_i \\propto n_i^{1/T}$ 把小语料抬上来。$n_i$ 是来源 i 的库存 token 数, $p_i$ 是它在训练数据里的占比, T 是温度。这个式子在说: 占比不按库存本身定, 按库存开 1/T 次方定。T=1 就是原始比例; T 越大, 各来源的占比越接近均匀。训练预算 500M token 时:\n- T=2: math 的占比变成 0.028, 相对原始比例上采样 13.84 倍, 2M 的库存在预算内被过 6.9 遍。\n- T=5: 上采样 50.89 倍, 过 25.4 遍, 很可能已经在背诵。\n目标配比定了, 还有一个问题: 单个 batch 里的实际占比会偏。指标是一个 batch (64 条) 内, 各来源的实际占比与目标占比的最大偏差, 再取所有 batch 里最坏的那个:\n- iid 抽样 (每条独立按 $p_i$ 随机抽): 0.228。\n- 额度调度: 0.021。\n额度调度是确定性的: 每出一条数据前, 给每个来源的额度加上它的 $p_i$, 选额度最大的来源出一条, 然后把它的额度减 1。偏差不会随时间累积。',
        },
        {
          title: '拼行把有效 token 从 0.285 提到 0.990',
          body: '训练时模型吃的是定长序列, m18 里序列长 L=128。但文档长短不一, 400 篇文档的长度中位数只有 28 个 token。有效 token 比例是真正来自文档的 token 占全部位置的比例, 其余位置是补位用的 pad, 算力照样花在它们身上。\n- 每篇一行, 补 pad 到 128: 0.285。七成算力花在 pad 上。\n- 动态 padding: 每个 batch (8 篇) 内只补到本 batch 最长那篇的长度。0.406。一篇长文档还是会拖着同批的其他文档一起补。\n- FFD packing: 0.990, 400 篇塞进 115 行。\nFFD (First-Fit-Decreasing) 的做法: 把文档从长到短排好, 每篇放进第一个还装得下它的行, 哪行都放不下就新开一行。不切断文档。',
        },
        {
          title: 'packing 必须配文档 mask',
          key: true,
          body: 'packing 的代价: 同一行里的文档会互相看见。第 3 篇的 token 在算注意力时能看到第 1、2 篇的内容, 学到的是一个不存在的上下文。\nm18 拿逐篇单独前向当基准, 比较拼在一行后每篇输出的最大绝对差:\n- 只有普通因果 mask (只挡住未来, 不分文档): RoPE 位置编码下差 3.0, 绝对位置编码下差 4.3。\n- 加上文档 mask: 差 1e-15, 只剩浮点舍入。\n文档 mask 是 block-diagonal (分块对角) 的: 每个 token 只许看同一篇里排在自己前面的 token。代码里就是 "因果 mask 并且属于同一篇" 这一个条件。',
        },
        {
          title: '位置和 label 也要管',
          body: '有了文档 mask 还不够, 还有两样:\n- 位置重置: 每篇文档的位置 id 要从 0 重新数。用绝对位置编码 (位置向量直接加在 embedding 上) 时必须重置, 否则第 2 篇起每个 token 的位置都和单独训练时不同, 输出差 4.6。用 RoPE (把位置转进 q 和 k 的旋转式位置编码) 时, 注意力分数只依赖两个 token 的位置差, 有文档 mask 时同一篇内的位置差没变, 不重置也等价, 差 1.3e-15。\n- 边界 label: 训练目标是预测下一个 token。每篇最后一个 token 的 "下一个" 是下一篇的第一个 token, 两者毫无关系, 这个预测要从 loss 里屏蔽掉。一行 4 篇共 128 token, 有效预测只有 124 个 (128 减去 4 篇各自的最后一个)。\n三样都做到, 这一行的 CE (交叉熵) 总和是 566.353069, 与逐篇单独训练完全相同。漏掉边界屏蔽, 变成 577.563075。',
        },
        {
          title: '这个 demo 没覆盖的',
          body: '- 垃圾是合成的: demo 的质量过滤用几条规则 (长度、平均词长、符号比例、停用词、重复 3-gram) 拦垃圾页。垃圾按 5 种模式合成, 规则一眼就能分开, 拦截率 1.000 没有意义。真实流水线在规则之后还要跑质量分类器 (FineWeb / DCLM)。\n- 代码被误杀: 为网页调的规则把 100% 的代码当成了垃圾, 这是真问题。符号比例、停用词、词长这些规则都不适合代码。实际做法是按来源分流, 代码、表格、数学各走各的过滤。\n- 不加 mask 很常见: 很多预训练 (GPT-2/3 风格) 直接把文档拼接起来, 不加文档 mask。Llama 3 加了, 并说在标准预训练里影响有限, 超长序列续训时才重要。',
        },
      ],
      links: [
        { from: 'MinHash 签名', to: 'LSH 分桶', body: 'MinHash 签名只负责把 "两篇像不像" 压成每篇 128 个整数, 比起来快, 但还是要两两比, $O(n^2)$。LSH 分桶接在后面: 只比落进同一个桶的文档, 全比才变成只比候选。' },
        { from: 'doc_mask', to: 'FlashAttention varlen', body: 'demo 的 doc_mask 是一张 [L, L] 的 bool 表。真实实现不会造这张表, 而是把每篇的起止偏移 (cu_seqlens) 传给 FlashAttention 的 varlen 接口, 它只算每篇自己那个对角块。所以 packing 还顺带省了注意力算量, 本例只剩整行因果注意力的 0.466。' },
        { from: 'RoPE', to: '位置重置', body: 'RoPE 把位置 m 的 q 和位置 n 的 k 各转一个角度, 它们的内积 $\\langle R(m)q, R(n)k\\rangle$ 只依赖 $n-m$。这个式子在说: 分数只看两个 token 隔多远, 不看它们的绝对位置。所以有文档 mask 时, 同一篇内的相对位置没变, 不重置也一样。HF 的实现仍然重置, 因为 bf16 下位置很大时转角有误差。' },
        { from: 'packing', to: 'SFT', body: 'SFT (监督微调) 的样本一般很短, 而且彼此无关。把它们 pack 进一行时, 文档 mask 和边界 label 屏蔽一般都要加, 否则一条样本会看到另一条的内容。' },
      ],
      sourceRows: [
        { concept: 'MinHash 签名', code: 'm17_data_pipeline/demo.py:minhash_signatures', takeaway: '签名的第 j 位是对集合里每个 5-gram 的哈希值 x 算 $(a_j x + b_j) \\bmod p$, 再取最小。一对系数 $(a_j, b_j)$ 就是一次随机置换, 取最小就是取置换后的最小元素。128 对系数给出 128 位签名。' },
        { concept: 'LSH 分桶 + 校验', code: 'm17_data_pipeline/demo.py:lsh_dedup', takeaway: '签名切 32 段分桶, 同一个桶里的文档都和桶里第一篇连成候选对。候选对再用签名估的 Jaccard ≥ 0.5 确认。确认的对用并查集连成簇, 每簇只留下标最小的那篇, 其余删掉。' },
        { concept: 'crc32 而非 hash()', code: 'zlib.crc32(" ".join(words[i:i + n]).encode())', takeaway: '5-gram 的哈希用 zlib.crc32 而不用 Python 自带的 hash()。hash(str) 每启动一个进程就换一个随机盐, 两次运行算出的签名不同, 去重结果就不可复现。' },
        { concept: '温度配比', code: 'm17_data_pipeline/demo.py:temperature_weights', takeaway: '把各来源的 token 数开 1/T 次方再归一化。T=1 时就是原始比例; T 越大各来源越接近均匀, T 趋于无穷时完全均匀。' },
        { concept: '文档 mask', code: 'm18_sequence_packing/demo.py:doc_mask', takeaway: 'mask[t, s] 为真的条件是 t ≥ s (因果: 只看前面) 并且 doc[t] == doc[s] (同一篇)。一行代码。' },
        { concept: '位置重置', code: 'm18_sequence_packing/demo.py:reset_positions', takeaway: '每篇文档开头的位置 id 归零: 三篇长度 3、2、4 的文档拼在一起, 位置 id 是 [0,1,2, 0,1, 0,1,2,3]。' },
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
      subtitle: '读完你能用 loss 公式 $L(N,D)$ 算出固定算力下模型该做多大。也能说清在 SP (标准参数化, 默认初始化加各层统一学习率) 下, 小模型调好的学习率 lr 为什么搬不到大模型。',
      tldr: '- scaling law: 用 $L(N,D)=E+A/N^\\alpha+B/D^\\beta$ 描述 loss 怎么随参数量 N 和训练数据量 D 下降。训练算力约为 $C \\approx 6ND$。算力定死时, 模型大小有一个最优值, 两头都亏: demo 里同样的算力, 参数量 81 的模型 loss 0.0285, 比 21 和 641 的都低。\n- μP (最大更新参数化): 按宽度改初始化和每层的 lr, 小模型上扫出的 lr 直接搬到大模型。demo 里宽 32 到 512 的最优 lr 都是 $2^{-5}$; 不改的话, 宽 16 倍最优 lr 要移 64 倍。\n本章的指数是玩具任务拟合的, 与 Chinchilla 论文的完全不同, 不能拿来推 LLM。',
      question: '钱一定, 模型做大还是数据加多? 小模型上调好的学习率, 为什么搬到大模型就不灵了?',
      evolution: {
        title: '只训得起一次, 所以先在小模型上算',
        subtitle: '根问题: 大模型只训得起一次, 模型多大、数据多少、lr 多少都得开训前定',
        steps: [
          { name: '照经验放大', pain: '(原点) 大模型只训得起一次, 大小、数据量、lr 都得开训前定', fix: '照上一代配置放大; 算力给模型还是给数据、lr 取多大, 全凭感觉' },
          { name: 'Kaplan scaling law', year: 2020, pain: '不知道多出来的算力该加在模型还是数据上', fix: '小模型上拟合 loss 随 N、D 的幂律再外推; 给出的配方偏向加大模型' },
          { name: 'Chinchilla IsoFLOP', year: 2022, pain: '模型偏大、数据偏少, 同样的算力 loss 没压到底', fix: '固定算力 $C\\approx 6ND$, 只改 N 和 D 的分法, 找 loss 的 U 形谷底; 拟合 $E+A/N^\\alpha+B/D^\\beta$ 推最优 N' },
          { name: 'μP', year: 2022, pain: '大小定了, lr 却扫不起; SP (默认参数化) 下宽 16 倍, 最优 lr 移 64 倍', fix: '输出层初始化和隐藏/输出层 lr 按宽度倍数 m 多除一次: 宽 32 到 512 最优 lr 都是 $2^{-5}$' },
        ],
      },
      code: 'llm_train/m19_scaling_laws/demo.py · llm_train/m20_mup/demo.py',
      points: [
        {
          title: 'loss 被三件事卡住',
          body: 'loss 降不下去, 原因可以拆成三份: 数据本身有噪声、模型太小、数据太少。\n先定符号: $N$ 是参数量, $D$ 是训练用的 token 数, $L$ 是 loss。Chinchilla 把 loss 写成 $L(N,D)=E+A/N^\\alpha+B/D^\\beta$。这个式子在说: loss 是三项相加。\n- 数据本身的噪声: $E$。它是常数, 模型再大、数据再多也消不掉。\n- 模型太小: $A/N^\\alpha$。模型越大这项越小, $\\alpha$ 决定小得多快。\n- 数据太少: $B/D^\\beta$。数据越多这项越小, $\\beta$ 决定小得多快。\nm19 真训了 6 个宽度 × 6 个数据量, 共 36 个单隐层 MLP。它们学的是一个固定的 teacher 网络 (生成数据的目标网络), 标签上加了方差 0.01 的噪声。每个模型的数据只看一遍。然后用这 36 个 loss 去拟合 5 个数, 得到 $E / A / \\alpha / B / \\beta$ = 0.0080 / 1.213 / 1.075 / 460.3 / 1.125, 相对残差 0.100 (拟合值和实测值平均差约 10%)。\n- 噪声地板: $E$ = 0.0080, 真实标签噪声方差是 0.01。$E$ 确实是谁也消不掉的那部分。\n- 被模型卡住: N=21 的模型, 数据从 16K 加到 65K, loss 从 0.0608 到 0.0603, 纹丝不动。这时第二项占了大头, 加数据只是在压第三项。',
        },
        {
          title: '固定算力, IsoFLOP 是 U 形',
          key: true,
          body: '算力定死时, 把它多分给模型还是多分给数据, loss 画出来是一条 U 形曲线, 谷底就是最优模型大小。\n先算算力。训练一个模型的浮点运算次数约为 $C \\approx 6ND$: 每个参数每个 token 前向约 2 次浮点运算, 反向约 4 次。所以算力一定时, 模型大一倍, 能看的数据就少一半。\nIsoFLOP (iso 是相等, FLOP 是浮点运算次数) 曲线就是固定 $C$, 只换 $N$ 和 $D$ 的分法, 看 loss 怎么变。m19 取训练网格的反对角线: 宽度每翻一倍, 数据就减半, 6 个点的算力都约等于 $8 \\times 10^6$。\n- 太小: N=21, loss 0.0603。数据很多, 但模型学不下。\n- 中间: N=81, loss 0.0285, 最好。拟合预测的最优是 N=89。\n- 太大: N=641, loss 0.0853。模型够大, 但数据不够把它训出来。\n六个点按 N 从小到大是 0.0603、0.0312、0.0285、0.0338、0.0632、0.0853, 先降后升。\n最优点也可以直接算。把 $D = C/6N$ 代进 loss 公式, 对 $N$ 求最小, 得 $N_{\\text{opt}} \\propto C^a$, $a = \\beta/(\\alpha+\\beta)$。这个式子在说: 算力每涨 10 倍, 最优模型大小涨 $10^a$ 倍, 而 $a$ 只看模型项和数据项下降速度的比。\n本例 $a$ = 0.511: 算力从 1e7 到 1e9, $N_{\\text{opt}}$ 从 100 到 1054。数据那边的指数是 0.489, 模型和数据大约各拿一半指数。',
        },
        {
          title: '这些数字不能拿去推 LLM',
          body: '上面的数字只说明方法, 不能拿去推 LLM:\n- 指数: 这里 $\\alpha \\approx 1.08$、$\\beta \\approx 1.13$, 论文是 0.34 / 0.28。玩具 teacher 的 $1/k$ 谱决定了 $\\alpha$: 生成数据的目标网络里, 第 k 个单元的输出权重按 $1/k$ 衰减。窄的学生网络只学得到排在前面的那些单元, 所以误差随宽度按幂律下降, 这个幂次由 $1/k$ 定, 和语言无关。\n- $a \\approx 0.5$ 是巧合: $a$ 只看 $\\alpha$ 与 $\\beta$ 的比值, 这里两者碰巧接近, 才和 Chinchilla 的 0.5 撞上。换个 teacher 就变。\n- D/N ≈ 150, 不是 20: Chinchilla 的结论是每个参数约 20 个 token。这里一个「token」是一个回归样本, 信息量和语言 token 不可比。\n- 不是样本外检验: IsoFLOP 的 6 个点也参与了拟合。每格只跑 1 个种子。',
        },
        {
          title: 'SP 下最优 lr 随宽度漂',
          body: '用默认做法时, 网络越宽, 最优学习率越小, 小模型上调好的 lr 搬到宽模型会差很多。\nSP (标准参数化) 就是默认做法: 每层初始化的标准差取 1 除以 fan_in 的平方根 (fan_in 是这一层的输入个数), 各层用同一个 lr。\nm20 用一个两层隐藏层的 ReLU MLP 去学一个固定的 teacher, Adam 训 150 步。它扫了 3 个宽度 × 10 个 lr ($2^{-13}$ 到 $2^{-4}$, 每格差 2 倍), 每个宽度挑 loss 最低的 lr:\n- SP 最优 $\\log_2 \\text{lr}$: −5 / −7 / −11 (宽 32 / 128 / 512)。宽 16 倍, lr 左移 6 格, 每格 2 倍, 一共小了 64 倍。\n- 直接搬: 把宽 32 调出的 lr 用到宽 512, loss 0.0976。该宽度最优 0.0215, 差 4.5 倍。',
        },
        {
          title: '为什么会漂: width 个小更新叠在一起',
          body: '最优 lr 会漂, 是因为 Adam 下网络越宽, 同样的 lr 对下一层的冲击越大。\n原因在 Adam: 它把每个元素的梯度除以这个元素梯度平方的滑动平均的平方根, 所以每个元素的更新大小约等于 lr, 与梯度本身多大无关。\n隐藏层的每个输出, 是 width 个输入各乘一个权重再加起来。一步更新里, 这 width 个权重各变了约 lr 那么多, 而且方向相同。它们的影响叠加起来, 下一层激活的变化 $\\propto \\text{lr} \\cdot \\text{width}$。这个式子在说: 宽度翻 4 倍, 同一个 lr 造成的冲击也翻 4 倍, 所以 lr 得小 4 倍才能维持原来的效果。\n按这个机制, 宽 16 倍, 最优 lr 应该小 16 倍 (lr $\\propto 1/\\text{width}$)。实测是 64 倍, 比预测的多。\n这组实验分不清多出来的部分从哪来: SP 的曲线不光滑 (宽 512 在 −9 到 −7 之间来回跳), lr 网格也是 2 倍一格, 都会让读数偏。',
        },
        {
          title: 'μP: 三处多除一个 m',
          body: 'μP 只比 SP 多改三个地方, 每处都是按宽度多除一次, 改完后最优 lr 不再随宽度移动。\nμP (maximal update parametrization, 最大更新参数化) 设宽度倍数 $m = \\text{width}/\\text{base}$, demo 的 base 宽度是 32, 宽 512 时 m=16。三处改动:\n- 输出层初始化: 标准差多除 $m$ (方差除 $m^2$)。源码注释给的理由是, 初始输出也要随宽度变小, 否则一开始就被随机的输出层主导。\n- 隐藏层的 Adam lr: 除以 $m$。抵消上一条说的「width 个更新叠加」。\n- 输出层的 Adam lr: 除以 $m$。\n输入层的 fan_in 固定是 $d_{\\text{in}}$ (输入特征数, 不随宽度变), lr 不缩。\n在 base 宽度上 m=1, μP 和 SP 完全一样, 所以在小模型上调出的 lr 两种参数化通用。\nm20 的结果:\n- 最优 lr: 三个宽度都是 $2^{-5}$。\n- 同一 lr 下的 loss: 越宽越好 (宽 32 / 128 / 512 依次是 0.0186、0.0101、0.0088)。宽 512 直接用宽 32 的 lr, 就是它自己的最优。\nμP 省的是在大模型上重新扫 lr 的成本。',
        },
        {
          title: '玩具规模下的边界',
          body: '- 规模: 只到 512 宽、150 步。README 提到 μTransfer 论文在 Transformer 上验证到约 6.7B 参数, 这里只能说明机制。\n- lr 调度: 如果换成常数 lr, μP 的最优点会在 −5 / −7 / −8 之间漂, 迁移没这么干净。所以 demo 用了线性衰减到 0。',
        },
      ],
      links: [
        { from: 'Kaplan (2020)', to: 'Chinchilla (2022)', body: 'cosine 调度让 lr 按余弦曲线从最大值降到 0, 降完叫退火完。Kaplan 所有规模共用一个 cosine 周期, 小数据的模型还没退火完就被评估, 数据的作用被低估了, 结论偏向加大模型 ($N_{\\text{opt}} \\propto C^{0.73}$)。m19 改成 cosine 周期等于本次的 D。' },
        { from: 'Chinchilla 最优', to: '过训练', body: 'Chinchilla 最优只最小化训练算力, 不管部署。Llama 3 8B 训了 15T token (D/N ≈ 1900), 远超这个最优。原因是模型要服务大量推理, 小模型推理便宜, 多训一些更划算。' },
        { from: 'm19 共用 lr=0.03', to: 'μP', body: 'm19 所有规模共用 lr=0.03, 只是粗扫过。真实的 scaling 研究要每个规模单独调 lr, 或者用 μP 让最优 lr 不随宽度漂。' },
        { from: 'μP', to: 'lr 幂律', body: '还有更省事的替代: 不改参数化, 直接在几个规模上测出最优 lr, 拟合它随规模变化的幂律 (DeepSeek LLM 就这样做)。' },
      ],
      sourceRows: [
        { concept: '拟合', code: 'm19_scaling_laws/demo.py:fit_scaling_law', takeaway: '变量投影: $\\alpha$、$\\beta$ 一固定, loss 公式对 $E, A, B$ 就是线性的。所以 $\\alpha$、$\\beta$ 在 79×79 的网格上逐格试, 每格用线性最小二乘解出 $E, A, B \\ge 0$, 留残差最小的那格。残差按相对误差计。' },
        { concept: '闭式最优', code: 'm19_scaling_laws/demo.py:compute_optimal', takeaway: '直接套公式 $G = (\\alpha A / \\beta B)^{1/(\\alpha+\\beta)}$, $N_{\\text{opt}} = G (C/6)^a$。main 里再用数值搜索找一遍最小点, 两者核对到 1% 以内。' },
        { concept: '单遍 + 退火到 D', code: 'm19_scaling_laws/demo.py:train', takeaway: '每步取新样本, 一共见 D 个, 每个只看一次。cosine 周期等于本次数据量, 每个模型都正好退火完。这正是 Chinchilla 对 Kaplan 的修正。' },
        { concept: 'SP vs μP', code: 'm20_mup/demo.py:init_and_lrs', takeaway: '两种参数化的全部差别是三处 /m ($m$ = 宽度 / 32): 输出层初始化的标准差, 隐藏层的 lr, 输出层的 lr。' },
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
      subtitle: '读完你能看出一个评测数字背后的四个坑: 分词器不同、测试题混进了训练集、pass@k 算法偏低、裁判偏心第一个位置。',
      tldr: '每个评测数字都有口径, 不说口径的数字没法比:\n- PPL (perplexity, 困惑度) 跟着分词器变: 同一个模型同一段文本, token 数减半, PPL 从 5.501 变成 30.265。跨模型要比按字符或字节摊的 bits (本章实测的是 bits-per-char)。\n- 测试题混进训练集叫污染。n-gram 查得到原题, 查不到改写过的题。\n- pass@k (生成 k 个答案至少一个对的概率) 用朴素公式算会系统性偏低, 要用无偏估计。\n- 让 LLM 当裁判, 它偏爱先出现的回答, 要交换 A/B 各判一次。',
      question: '分数高是模型真会, 还是背过题、换了分词器、或者裁判偏心?',
      evolution: {
        title: '题越开放, 判分越难',
        subtitle: '根问题: 分数要说明模型真会什么, 可每种打分只量得到它能量的那一部分',
        steps: [
          { name: 'perplexity', pain: '(原点) 训练只看得到 loss, 要一个数说模型多好', fix: '留出集上算 $e^{\\mathrm{CE}}$; 分词器一换就变 (5.501 → 30.265), 也量不出会不会做题' },
          { name: '题库 + 去污染', pain: '想知道会不会做题, 光看 PPL 答不了', fix: '固定题库按答对比例打分; 原题混进训练语料会抬分, n-gram 只查得到没改写的' },
          { name: 'pass@k', year: 2021, pain: '代码题写法千万种, 和标准答案逐字比会把对的判错', fix: '跑单元测试判对错, 每题采 n 个用无偏式估 pass@k; 朴素公式 k=10 时偏低 0.039' },
          { name: 'LLM 裁判 + 交换', year: 2023, pain: '写作、对话没有测试可跑, 只能请裁判; LLM 裁判偏爱先出现的那份', fix: 'A/B 交换各判一次、两次一致才算: 好的在 B 位时准确率 0.593 → 0.991, 调用翻倍' },
        ],
      },
      code: 'llm_train/m21_llm_eval/demo.py',
      points: [
        {
          title: 'PPL 是等效分支数',
          body: 'PPL (perplexity, 困惑度) 是最常用的语言模型指标。先说 CE (交叉熵): 模型给每个 token 的正确答案一个概率, 取负对数, 再对所有 token 取平均。$\\mathrm{PPL} = e^{\\mathrm{CE}}$。这个式子在说: 把平均的负对数概率还原回去, 得到一个 "模型平均在几个词里犹豫" 的数。\nm21 用一条 20 个词的马尔可夫链 (下一个词的概率只由前一个词决定) 生成语料, 所以真实分布是已知的:\n- 均匀分布 (每个词概率 1/20): PPL 恰好 20.000, 等于词表大小。完全瞎猜就是在 20 个词里犹豫。\n- bigram 计数 (在训练集上数每个词后面跟什么): 5.501。\n- 真实分布: 5.448, 这是能达到的下限。',
        },
        {
          title: 'PPL 不能跨分词器比',
          body: '同一段文本、同一个模型, 只换分词器。假设新的分词器把两个字符并成一个 token, token 数减半:\n- PPL: 从 5.501 变成 30.265。\n- bits-per-char (每个字符平均多少比特): 都是 2.4598。\n原因: 整段文本的总信息量 (所有 token 负对数概率之和) 没变, 只是摊到了更少的 token 上, 每个 token 分到的就多了。所以跨分词器比模型, 要用每字符或每字节的 bits。\n平均的口径也要对齐:\n- 按 token 平均 (所有文档的 token 拼在一起算 CE 再取指数): 5.501。\n- 按文档平均 (每篇先算自己的 PPL, 再对文档取平均): 5.573, 被短文档拉高了。',
        },
        {
          title: '去掉被标记的题还不够',
          key: true,
          body: '污染指测试题混进了训练语料, 模型见过答案, 分数是背出来的。常用的检测是 n-gram 重叠: 把训练集里所有连续 8 个词的片段 (8-gram) 建一个集合, 测试题只要有任何一个 8-gram 在里面出现过, 就判为脏题。GPT-3 用的是 13-gram。\nm21 往训练语料里混了 30 道原题、30 道轻改的题 (换 2 个词)、30 道重度改写的题 (每 3 个词换 1 个):\n- 原题、轻改: 被标记的比例 1.00。换 2 个词, 30 个词的题里仍有完整的 8-gram 幸存。\n- 重度改写: 0.00。每 3 个词就换 1 个, 没有任何连续 8 个词是原样的, 检测完全失效。\n分数怎么被抬高: demo 假设模型见过的题 90% 答对, 没见过的 4 选 1 瞎猜。全部 200 题的报告分 0.570; 去掉被标记的题后 0.436; 只算真正干净的题, 真实能力 0.300。中间那个 0.436 仍然虚高, 因为漏网的重度改写题还在里面。',
        },
        {
          title: '朴素 pass@k 偏低',
          body: 'pass@k 是代码题的指标: 每题让模型生成 k 个答案, 至少一个通过单元测试的概率。\n实际做法是每题采 n 个样本 (n 比 k 大), 数出 c 个对的。怎么从 (n, c) 估 pass@k, 有两种算法:\n- 朴素: 把 c/n 当成这题的通过率 p, 代进 $1-(1-p)^k$。它会偏低。原因是 $1-(1-p)^k$ 对 p 是凹函数 (越往上越平), 由 Jensen 不等式, 对一个有随机误差的估计值取这个函数再求期望, 会比对真值取函数小。\n- 无偏: $1-\\binom{n-c}{k}/\\binom{n}{k}$。这个式子在说: 从 n 个样本里无放回地挑 k 个, 挑到的全是错的概率是 $\\binom{n-c}{k}/\\binom{n}{k}$, 1 减去它就是至少一个对的概率。\nm21 的实验 ($n=20$, 300 题, 重复 400 次取平均), $k=10$ 时:\n- 真值 (用每题真实的通过率算): 0.5864。\n- 无偏: 0.5861。\n- 朴素: 0.5470, 偏低 0.0394。\n$k=1$ 时两种算法都退化成 c/n, 结果相同。',
        },
        {
          title: '裁判要交换顺序',
          body: '写作、对话这类题没有测试可跑, 常让一个 LLM 当裁判比较两个回答 A 和 B。LLM 裁判有位置偏差: 偏爱先出现的那个。\nm21 用一个规则裁判模拟这件事: 它看两个回答的质量差, 但给排在第一个位置的回答加 0.8 分, 再加一点噪声。2000 对回答, A 和 B 谁更好是随机的 (A 更好的占 49.8%):\n- 单次判 (只按 A 在前、B 在后判一次): 70.0% 的情况选了第一个。好的回答在 A 位时准确率 0.994, 在 B 位时只有 0.593。\n- 交换判 (A、B 各当一次第一个, 判两次, 两次结论一致才算数, 不一致记平局): 好的在 A 位和在 B 位的准确率变成 0.993 和 0.991。两边一样, 位置偏差被抵消了。\n代价:\n- 调用次数翻倍。\n- 平局 40.7%: 都是两个回答质量接近、0.8 的位置加分压过了真实差距的那些对。',
        },
        {
          title: '合成语料下的边界',
          body: '- 语料是合成的: 随机文本里两篇恰好撞上同一个 8-gram 的概率很低, 所以干净题一道都没被误标。真实语料里的版权声明、代码模板这类常见短语会造成误标, 实际检测一般会忽略在训练集里出现太多次的高频 n-gram。\n- 记忆率是假设: "见过就 90% 答对" 只是用来演示污染怎么抬高分数, 不代表真实模型的记忆率。\n- 交换只消位置偏差: 真实 LLM 裁判还有长度偏差 (偏爱长回答) 和自我偏好 (偏爱自家模型的回答), 交换顺序对这两种没用。',
        },
      ],
      links: [
        { from: '训练 loss', to: 'PPL', body: '训练时打印的 loss 就是 CE (逐 token 负对数概率的平均)。对它取指数 $e^{\\text{loss}}$ 就是 PPL。两者一一对应, 看 loss 曲线和看 PPL 曲线是同一件事。' },
        { from: '数据去重', to: '去污染', body: '去重和去污染用的是同一类工具 (哈希、n-gram 重叠), 只是找的地方不同: 去重在训练集内部找近似副本, 去污染在训练集和测试集之间找。' },
        { from: '改写题', to: '其他检测法', body: 'n-gram 抓不到改写过的题, 实际还有三种办法: 用 embedding 相似度检索语义上像的题; 让模型补全题目的后半段, 看它能不能逐字复现; 只用模型发布日期之后才出现的新题。' },
        { from: '平局', to: '多裁判', body: '交换判留下 40.7% 的平局, 不能就这么扔掉。真实做法是把平局记 0.5 分, 或者再加几个裁判投票。' },
      ],
      sourceRows: [
        { concept: '逐 token NLL', code: 'm21_llm_eval/demo.py:token_nll', takeaway: '对每个 token 算正确答案概率的负对数, 第一个 token 按均匀分布计。这些数的均值就是 CE, $e^{\\mathrm{CE}}$ 就是 PPL。' },
        { concept: 'n-gram 污染检测', code: 'm21_llm_eval/demo.py:ngram_flags', takeaway: '把训练集里所有的 8-gram 放进一个 set。对每道测试题, 它的任何一个 8-gram 在 set 里出现过就判为污染。' },
        { concept: '无偏 pass@k', code: 'm21_llm_eval/demo.py:pass_at_k', takeaway: 'Codex 论文的写法: 把 $\\binom{n-c}{k}/\\binom{n}{k}$ 约分成 c 项连乘, 算 $1-\\prod_{i=n-c+1}^{n}(1-k/i)$。不用算大的组合数, n 很大也不溢出。' },
        { concept: '交换判', code: 'm21_llm_eval/demo.py:judge_swap', takeaway: '判两次, 第一次 A 在前, 第二次 B 在前。两次都选 A 才算 A 赢, 两次都选 B 才算 B 赢, 两次不一致记平局。' },
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
  },
}

// 机制实验的原理、运行入口与源码引用。
const supplements = {
  "train-model-parallel": [
    {
      "title": "三种并行要放进同一张预算表",
      "body": "真实训练常常三种并行一起用: DP (数据并行, 每组卡放一份完整模型, 各吃一部分数据)、TP 和 PP。三者的份数乘起来要等于总卡数。只算「参数装不装得下」, 回答不了「训练会不会更快」, 所以 m22 把显存和时间放进同一张预算表一起算。\n它固定一个场景: 1B 参数、24 层、全局 batch 64、8 张卡, 比较 DP×TP×PP 的不同组合。每个组合按四步算:\n- 先查合法性: 给每张卡排好 rank 网格 (这张卡在 DP、TP、PP 三个方向上各排第几), 再查整除约束, 比如 hidden 要能被 TP 整除、层数要能被 PP 整除。DP2×TP3×PP2 一共要 12 张卡, 对不上 8 卡, 直接拒绝, 不给非法配置编出收益。\n- 再算显存: 每卡要放的训练状态 (参数、梯度、fp32 主权重和 Adam 的两份动量) 和激活。\n- 再算通信: 一次通信的时间是「消息条数 × α + 传输字节 ÷ 带宽」。α 是每发一条消息的固定启动延迟, 和消息大小无关。\n- 最后算一步的时间: 计算时间, 加上没能藏到计算后面的通信 (只有 DP 的梯度同步能和计算重叠), 再乘上 PP 气泡带来的倍率。\n三组读数:\n- DP2×TP2×PP2: 每卡状态 3.725 GiB, 一步约 708 ms, 关键卡 (代价模型按它算步时间的那张卡) 一步要发 786 条通信消息。\n- DP1×TP4×PP2: 状态降到 1.863 GiB, 一步却涨到 975 ms, 通信 4640 次。TP 每一层都要同步, TP 越大, 小消息越多, 每条都要付一次 α。省下的显存被多出来的通信时间吃掉了。\n- 同样是 DP1×TP4×PP2, 把 α 从 8 μs 改到 100 μs: 一步 1429 ms。\n计算时间用 Roofline 估: 一次计算能跑多快, 上限由峰值算力和显存带宽里更紧的那个决定。MFU (实际有效计算量占硬件峰值算力的比例) 也是这个代价模型估出来的 (0.694 对 0.504), 不是硬件实测。",
      "source": "llm_train/m22_parallel_budget/demo.py:estimate"
    }
  ],
  "train-collectives-loop": [
    {"title":"本章查通信对不对, 不估快慢", "body":"这一章关心的是通信的语义: 每个 collective (多张卡一起参与的通信操作, 上面四个原语都是) 有哪些卡参与、张量在各卡上怎么摆、按什么顺序调用。例如 reduce-scatter 做完, 每个 rank (一张卡在通信组里的编号) 只持有总和里属于自己的那一片; 接着 all-gather 再把各片拼回完整的张量。\nm09 用 N=4 张卡、S=32 B 的张量逐项核对:\n- reduce-scatter 每卡发 24 B, all-gather 每卡发 24 B, 加起来正好是 all-reduce 的 48 B。\n- ring_all_reduce_sum 真的把张量切成块 (chunk), 沿环一步一步传。断言检查两件事: 实测发出的字节数等于公式, 求和结果等于直接相加的结果。\n显存要多少、一步要多久这类预算估算, 放在模型并行那一章。这里不估时间, 只用真实的同步结果和断点续训后的断言, 检查主循环有没有写对。", "source":null}
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
topic.pages["train-collectives-loop"].related = [{"route": "train-model-parallel", "label": "三种并行组合: 显存与时间一起算", "reason": "本章只讲四个通信原语和训练主循环。DP、TP、PP 怎么组合, 每种组合每卡占多少显存、一步要多久, 在模型并行那一章的预算表里。"}]
topic.pages["train-model-parallel"].related = [{"route": "infer-compute", "label": "从估算到实测: profiler 怎么用", "reason": "本章的预算表是代价模型估出来的, 只能先淘汰明显不行的配置。真正慢在哪, 要用 profiler 记下每个算子的调用轨迹 (trace) 和耗时再看, 那一章讲怎么测。"}]
export default topic

mergePage(topic, 'train-low-precision', 'train-precision-stability', '进阶 · 分块缩放 (block scale) 与 FP4', ["TrainBlockScaleLab"])

topic.pages['train-precision-stability'].title = '低精度训练 · 稳定性、缩放与 FP4'
const precision = topic.pages['train-precision-stability']
precision.sections[0].points[0].body = '低精度格式能表示的范围很窄, 张量里的数要先除以一个缩放系数 scale, 缩进格式的网格里, 再舍入到最近的格点。这里的 scale 和上文 loss scaling 放大 loss 的 scale 不是一回事: 它按每一组元素各算一个。\n- 怎么算: 共用一个 scale 的那组元素叫一个 block。scale 等于 block 里绝对值最大的那个数 (amax) 除以网格的最大值。这样最大的元素正好落在网格顶端, 其余元素按同一个 scale 缩小后再舍入。\n- outlier 的连累: outlier 是个别异常大的值。它把 amax 顶得很高, 同 block 的邻居除以 scale 后都变得很小。outlier 越大, 邻居被挤得越靠近 0; block 越小, 被连累的邻居越少。\nFP8 上这一点上文已经量过: outlier 大到 1e5 倍时, 其余 token 的输出相对误差 per-tensor (整个张量共用一个 scale) 是 0.126, block 是 0.026。\nFP4 更紧。它的网格只有 $\\pm\\{0.5,1,1.5,2,3,4,6\\}$ 和 0, 非零格点里最大的 6 只是最小的 0.5 的 12 倍。不切成小 block, 一个张量里稍微悬殊一点的数就装不下, 所以 block 必须缩到 16~32 个元素。'
topic.pages['train-model-parallel'].points.find(p => p.title.startsWith('PP:')).body = 'PP (流水线并行) 把模型按连续的层切成 PP 段, 每段叫一个 stage, 各放一张卡。卡与卡之间只在 stage 边界传一份激活, 比 TP 在每一层里都要 all-reduce 省得多。\n代价是空等。一个 batch 要从第一个 stage 流到最后一个再倒回来: 刚开始时后面的卡还没拿到数据, 快结束时前面的卡已经没活干。空等时间占总时间的比例叫气泡。把 batch 切成 M 个 micro-batch 依次灌进去, 各 stage 就能同时忙起来, 空等只剩首尾。m04 的模拟里, PP=4、M=8 时气泡还有 27.3%。\nmicro-batch 的执行顺序 (调度) 决定两件事: 空等多久, 以及每个 stage 同时攥着几份在途激活。在途激活是已经做完前向、还没做反向的 micro-batch 留下的激活, 反向要用, 所以一直占着显存。\n- GPipe: 先把 M 个 micro-batch 的前向全做完, 再做反向, 每个 stage 最多攥 M 份。\n- 1F1B: 前向和反向交替做, stage s 最多攥 PP−s 份。它的气泡和 GPipe 一样, 省的只是在途激活。\n气泡公式、逐格时间线和通信代价都放在「1F1B 与交错调度」一章。'

for (const chapter of topic.chapters) chapter.label = chapterLabels[chapter.route] || chapter.label

canonicalizeRelated(topic)
