// 阶段 3 · llm_train 的全部章节页 (完整定义)。
// 数字全部取自对应 demo 的实际输出, 改 Python 后请重新核对。
// points 里 key: true 的那一条 = 这一章只让人记住一句话时, 记这句。
export default {
  stage: 'train',
  chapters: [
    { route: 'train-pipeline-schedules', label: '1F1B 与交错调度', hint: '气泡公式、在途激活、interleaved 1F1B' },
    { route: 'train-lr-schedule', label: 'WSD 学习率调度', hint: 'warmup / cosine / WSD, 为什么能随时加训' },
    { route: 'train-low-precision', label: 'FP8 与 FP4 微缩放', hint: 'block scaling、MXFP4 / NVFP4' },
    { route: 'train-muon', label: 'Muon 优化器', hint: 'Newton–Schulz 正交化、QK-clip' },
    { route: 'train-ulysses', label: 'Ulysses 序列并行', hint: 'all-to-all 把切序列换成切头' },
    { route: 'train-data-packing', label: '数据流水线与 packing', hint: 'MinHash 去重、温度配比、文档 mask' },
    { route: 'train-scaling', label: '规模: scaling law 与 μP', hint: 'IsoFLOP 的 U 形、lr 随宽度迁移' },
    { route: 'train-eval', label: '评测: PPL、污染、pass@k、裁判', hint: '每个评测数字都有口径' },
  ],
  pages: {
    // ---------------------------------------------------------------- 切 batch
    'train-batch-ddp': {
      title: 'batch 与 DDP · 同一个梯度的三种算法',
      subtitle: '读完你能说清 DDP 为什么"就是一个更大的 batch", 以及这句话在什么条件下会悄悄失效。',
      tldr: '同一个全局梯度有三种算法:\n- 单卡: 一次算完。\n- 梯度累积: 用 $K$ 倍串行时间, 换 $1/K$ 的激活。\n- DDP (分布式数据并行, 每卡算一片 batch): 用每步 $2(P-1)/P$ 份梯度的通信, 换 $1/P$ 的时间。\n三者相等的前提: 每一份的样本数相同。',
      question: 'micro-batch 和 DDP 的 rank 都在切 batch, 为什么一个不用通信、另一个每步都要 all-reduce?',
      code: 'llm_train/m01_gradient_accumulation/demo.py · llm_train/m02_data_parallel/demo.py',
      points: [
        {
          title: '三个名词: DDP、rank、all-reduce',
          body: '- DDP (分布式数据并行): 每张卡放一份完整模型, 各算一片 batch。代码里一张卡叫一个 rank。\n- all-reduce: 每卡交出自己的梯度, 每卡拿回所有卡的和。再除以卡数就是平均梯度。\n- micro-batch: 同一张卡把一个 batch 拆成几小份, 一份份串行算, 梯度累加起来再更新一次。',
        },
        {
          title: '切法不同, 梯度相同',
          key: true,
          body: '全局 batch 的梯度就是 N 个样本各自梯度的平均。谁先算、在哪张卡上算、分几批算, 都不改这个平均值。\n- m01: 累积梯度与整 batch 差 2.98e-08。\n- m02: DDP 走完一步后, 参数与单卡差 3.73e-09。\n都是 float32 求和顺序留下的零头, 两边都用 assert 卡着。\n所以 "DDP 就是一个更大的 batch" 是等式。',
        },
        {
          title: '累积换显存, DDP 换时间',
          body: '- 梯度累积: 激活峰值按 micro-batch 算 (m01: 8 个样本 → 2 个)。代价是 K 倍串行, 算力和通信一点没省。\n- DDP: 每卡样本数除以 P。代价是每步一次全模型梯度 all-reduce, 每卡发 $2(P-1)/P$ 份。8 卡 1.75 份, 64 卡 1.97 份, 几乎不随卡数涨。\nDDP 不省显存: 参数、梯度、Adam 状态每卡仍是一整份。这正是 ZeRO 的入口。',
        },
        {
          title: '等价挂在"按样本数加权"上',
          body: 'micro 的 loss 已经在本地取过均值, 累加时要再乘 $n_k/N$。等分时它退化成 $1/K$, 所以漏掉也不报错。\n- m01 里 "直接相加": 梯度恰好放大 $K=4$ 倍, 等于偷偷把学习率乘 4。\n- 换成 [3,3,2] 这种不等长切法: 立刻露馅。\n跨卡同理: all-reduce(mean) 做的是各卡局部均值的简单平均, 只有每卡样本数相同时才等于全局均值。',
        },
      ],
      links: [
        { from: 'llm_basic train.py', to: '梯度累积', body: 'update 从每个 batch 一次变成每 K 个 micro 一次, 其余一行没改。' },
        { from: 'local grads', to: 'all_reduce_mean', body: '本地梯度只是全局 batch 的一片; 漏一次同步, 各副本当场分家。' },
        { from: 'DDP', to: 'ZeRO', body: 'DDP 每卡都存完整参数、梯度、Adam 状态; 存不下时才轮到分片。' },
        { from: '2(N−1)/N', to: 'ring all-reduce', body: '这个系数怎么来的, 见「通信与 full_loop」一章。' },
      ],
      sourceRows: [
        { concept: 'micro 权重', code: 'm01_gradient_accumulation/demo.py:accumulate', takeaway: 'scale=n/len(x) 这一行就是 full-batch mean 的全部; weighted=False 是故意写错的对照组。' },
        { concept: '不等长 micro', code: 'sizes=[3, 3, 2]', takeaway: '最后一个 micro 不满时, 按样本数加权和按 1/K 平均给出不同答案。' },
        { concept: '梯度同步', code: 'core/collectives.py:all_reduce_mean', takeaway: 'all-reduce(sum) 再除以 world: 之后每张卡拿到逐位相同的梯度。' },
        { concept: '副本一致性', code: 'assert max_abs_diff(replicas[0], r) == 0', takeaway: '同步后所有副本必须逐位相同, 否则训练已经分叉, 而且不会报错。' },
        { concept: '通信账', code: 'core/collectives.py:CommCounter', takeaway: '每 rank 每步 $2(N-1)/N$ × 全模型梯度字节, 上限 2 份, 与卡数几乎无关。' },
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
      tldr: '- 单层太宽: 切矩阵, 叫 TP (张量并行)。W1 列切、W2 行切, 前向 1 次 + 反向 1 次 all-reduce。\n- 层数太多: 切层, 叫 PP (流水线并行)。用 micro-batch 把气泡 (卡空等的时间占比) 填小。',
      question: 'TP 的通信为什么在层内、PP 的只在 stage 边界? 前向已经 all-reduce 过了, 反向为什么还要再来一次?',
      code: 'llm_train/m03_tensor_parallel/demo.py · llm_train/m04_pipeline_parallel/demo.py',
      points: [
        {
          title: 'TP: 先列切, 后行切',
          key: true,
          body: '- W1 按列切: 每卡拿完整输入, 算出隐藏维的一段。中间的 ReLU 是逐元素的 (换成 GeLU 同理), 各卡本地算, 不用通信。\n- W2 按行切: 每卡得到一个部分和, 最后一次 all-reduce 加起来。\n顺序反过来 (先行切), 就得在激活函数前先同步一次。\nbias b2 只在 all-reduce 之后加一次, 否则会被加 N 遍。',
        },
        {
          title: '反向也要 all-reduce',
          body: '输入 X 被每张卡都用过, 所以它的梯度是各卡贡献之和。\n每卡算出的 dX 只含它那 $H/N$ 列的部分。不求和就传给上一层, 梯度是错的: m03 里这个差是 1.8e-2。\nall-reduce 次数: MLP 块前向 1 次 + 反向 1 次, attention 块同样两次, 一层 Transformer 共 4 次。',
        },
        {
          title: 'PP: 气泡与在途激活',
          body: '按连续层切成 PP 段, 每段叫一个 stage, 各放一张卡。通信只在 stage 边界传一份激活, 比 TP 省得多。代价是首尾的卡在空等。\n- 气泡 (卡空等的时间占比): $(\\mathrm{PP}-1)/(M+\\mathrm{PP}-1)$, $M$ 是 micro-batch 数。GPipe 与 1F1B 两种调度完全相同。\n- 在途激活 (已前向、还没反向的 micro-batch 留着的激活): 两种调度的区别在这里。stage $s$ 攥着的份数从 $M$ 降到 $\\mathrm{PP}-s$, stage 0 最多。\n细节见「1F1B 与交错调度」。',
        },
      ],
      links: [
        { from: 'llm_models Block', to: 'TP', body: 'TP 落在 attention 和 MLP 的几个大矩阵上, 逐元素算子一律不动。' },
        { from: 'num_layers', to: 'PP', body: '层越深, 按连续层切 stage 越自然, 边界上只传一份激活。' },
        { from: 'micro-batch 数 M', to: '气泡占比', body: 'M 越大气泡越小; M 能开多大取决于在途激活放不放得下。' },
        { from: 'PP', to: '1F1B 与交错调度', body: '三种调度的时间表和显存账在下一章摊开。' },
      ],
      sourceRows: [
        { concept: 'column parallel', code: 'W1_s = np.split(W1, world, axis=1)', takeaway: '[D,H] → N×[D,H/N], 每个 rank 计算 hidden 的一段。' },
        { concept: 'row parallel + g 算子', code: 'tp_out = all_reduce_sum(partial)[0] + b2', takeaway: '部分和求和得到 dense 输出; b2 只加一次。' },
        { concept: 'f 算子的反向', code: 'tp_dx = all_reduce_sum(dx_partial)[0]', takeaway: '前向恒等、反向 all-reduce; 省掉它 dX 就错 1.8e-2。' },
        { concept: '流水线排程', code: 'm04_pipeline_parallel/demo.py:simulate', takeaway: '每卡按固定顺序执行, 每个 op 在"卡空闲且依赖已完成"的最早时刻开始。' },
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
      subtitle: '读完你能自己算: 一个 7B 模型开某个 ZeRO stage 之后, 每卡还剩多少显存留给激活。',
      tldr: '- 混合精度 Adam: 每参数 16 字节 = 2 (fp16 参数) + 2 (fp16 梯度) + 12 (fp32 master + m + v)。\n- ZeRO-1/2/3: 依次把 12 / 14 / 16 除以卡数。\n- 激活: 不在这 16 字节里。用 +33% 的重算, 从 $O(L)$ 压到 $O(\\sqrt{L})$。',
      question: '7B 模型在 64 张卡上开 ZeRO-1, 每卡还要多少 GB? 为什么再加卡也降不下去?',
      code: 'llm_train/m05_zero_fsdp/demo.py · llm_train/m07_activation_checkpointing/demo.py · llm_train/m08_checkpoint_resume/demo.py',
      points: [
        {
          title: '2 + 2 + 12',
          key: true,
          body: '优化器状态占了这 16 字节里的 12。\n$\\Psi$ 是参数个数, $N$ 是卡数。ZeRO 把状态切成 $N$ 片, 每卡只存一片, 分三档:\n- DDP: $16\\Psi$\n- ZeRO-1: 切掉 12, 变成 $4\\Psi+12\\Psi/N$\n- ZeRO-2: 再切梯度, 变成 $2\\Psi+14\\Psi/N$\n- ZeRO-3/FSDP: 连参数一起切, 变成 $16\\Psi/N$\nm05 里 $\\Psi=192$、$N=4$, 每 rank 常驻字节恰好是 3072 / 1344 / 1056 / 768。\nAdam 逐元素更新, 每卡只更新自己那一片, 结果与 DDP 逐位相同 (差 0.0e+00)。\n7B 开 ZeRO-1、64 卡: 7e9 × (4 + 12/64) ≈ 29 GB。那个 $4\\Psi$ 是加多少卡都除不掉的地板。',
        },
        {
          title: '通信账: ZeRO-2 不比 DDP 贵',
          body: '- ZeRO-2: all-reduce 本来就等于 reduce-scatter + all-gather。ZeRO-2 只在两半中间插一步 "各自更新自己的分片", 总字节和 DDP 持平 (m05 实测 1.00×)。\n- ZeRO-3: 每层前向、反向都要先 all-gather 一次参数才能算, 约 1.5×。用完立刻 free, 瞬时只多一层的完整 fp16 参数 (demo 里 +256 B)。',
        },
        {
          title: '激活重算: 用 33% 计算换 √L 显存',
          body: '不存所有层的中间激活, 只存段边界, 反向前重跑段内前向。峰值 $\\approx L/k + k$, $k=\\sqrt{L}$ 时最小。\nm07 里 $L=16$: $k=1$ 要 17 份, $k=4$ 只要 8 份, 梯度逐位相同 (0.0e+00)。\n代价是多跑一次前向: 前反向本来 3 个单位, 变成 4 个, 上限 +33%。demo 里 $L=16$、$k=4$ 实测 +25%。',
        },
        {
          title: '可恢复 = 四样齐全',
          body: '要存四样: 参数、优化器状态、数据游标、RNG。m08 一样样拿掉:\n- 漏 RNG: 差 3.1e-2\n- 漏优化器: 差 5.8e-2\n- 漏数据游标: 差 1.5e-1\n全都不报错, 只是轨迹悄悄偏了。\n那个 checkpoint 共 3047 B, 其中 RNG 状态 2496 B, 玩具模型的参数只有 40 B。',
        },
      ],
      links: [
        { from: 'DDP 每卡全量', to: 'ZeRO 分片', body: '复制换成分片: 显存下降, 通信上升 (stage 3 约 1.5×)。' },
        { from: 'llm_basic 的 cache', to: '激活重算', body: '小模型把前向中间量全存; 大模型只存段边界, 其余反向前重跑。' },
        { from: 'ToyDataStream.cursor', to: 'resume', body: '数据读到哪也是训练状态, 不存就接不回同一条轨迹。' },
        { from: 'reduce-scatter', to: '通信原语', body: 'all-reduce 拆成两半看, 就明白 ZeRO-2 为什么不涨通信。' },
      ],
      sourceRows: [
        { concept: '状态怎么切', code: 'm05_zero_fsdp/demo.py:init_rank_state', takeaway: 'stage 决定哪些数组存整份、哪些只存 $1/N$。这就是 ZeRO 的全部数据结构。' },
        { concept: '梯度同步的分水岭', code: 'reduce_scatter_sum(layer_g)', takeaway: 'stage ≤ 1 用 all-reduce 留整份梯度; stage ≥ 2 只留自己那片。' },
        { concept: 'FSDP gather → compute → free', code: 'full = all_gather([rk["p16"][i] for rk in ranks])', takeaway: '用到哪层 gather 哪层, del full 立刻释放; 瞬时只多一层的完整参数。' },
        { concept: '激活账本', code: 'm07_activation_checkpointing/demo.py:run', takeaway: '重算出来的段内激活也占显存, 必须计进峰值。' },
        { concept: 'resume payload', code: 'm08_checkpoint_resume/demo.py:snapshot', takeaway: '保存 step、model、optimizer、数据游标、RNG; 缺一样轨迹就对不上。' },
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
      subtitle: '读完你能解释现代训练为什么默认 BF16、fp32 master 为什么删不得, 以及一个 Inf 冒出来时框架在做什么。',
      tldr: '低精度只负责算, 更新永远发生在 fp32 master 上。\n- FP16: 要配动态 loss scaling。溢出就跳过这一步, 并把 scale 减半。\n- BF16: 用精度换范围, 省掉 scaling。\n- FP8: 位宽再砍一半, scale 变成必选项。',
      question: '为什么 FP16 训练不是把所有数组 astype(np.float16) 就完事? BF16 和 FP8 又各改了什么?',
      code: 'llm_train/m06_mixed_precision/demo.py · llm_train/m10_training_stability/demo.py · llm_train/m13_fp8_training/demo.py · llm_train/core/numerics.py',
      points: [
        {
          title: '范围 vs 精度: 16 位的两种分法',
          body: '- FP16 (5 位指数 + 10 位尾数): 1e-8 的梯度直接变 0, 70000 的激活直接变 Inf。\n- BF16 (8 + 7): 指数位和 FP32 一样宽。1e-8 存成 1.001e-08、70000 存成 70144, 两头都不炸。\nBF16 的代价是尾数少 3 位: $1+2^{-9}$ 会被舍回 1 (FP16 能存成 1.001953125)。\nBF16 精度比 FP16 低, 赢的只有范围。但范围恰好是训练最怕丢的东西。',
        },
        {
          title: 'loss scaling + fp32 master',
          key: true,
          body: '反向前把 loss 放大, 更新前再除回去, 小梯度就不会在 fp16 里归零。1e-8 直接转 fp16 是 0; 先 ×32768 再转、再除回来, 得到 9.997e-09。\n- 出现 Inf: 跳过这一步, scale 减半。\n- 连续若干个好 step: scale 再翻倍。\nm06 的 24 步里跳了 5 步。\n更新本身永远落在 fp32 master 上。纯 fp16 权重每步加 1e-5 的更新, 100 步后还是 1.0, 一动没动。',
        },
        {
          title: 'FP8 的两套配方',
          body: '格式名直接写位数: E4M3 = 4 位指数 + 3 位尾数, E5M2 = 5 位指数 + 2 位尾数。\n- Transformer Engine: 前向 E4M3、反向 E5M2, 整个张量共用一个 scale (per-tensor)。\n- DeepSeek-V3: 全程 E4M3, 每 128 个元素一个 scale (block)。\nE4M3 自带约 $2^{15}$ 的动态范围, 所以 scale 的粒度平时看不出差别。m13 量了两个指标:\n- 验证 loss (训 200 步, 除以 $\\|y\\|^2$), 每次只动一个旋钮: 反向 E5M2 换成 E4M3 是 4.54e-4 → 4.64e-4, per-tensor 换成 block 是 4.64e-4 → 4.61e-4, 都打平。\n- 其余 token 的输出相对误差: outlier 超过约 1e5 倍才拉开, per-tensor 0.126, block 0.026。',
        },
        {
          title: '去掉 master 比去掉 scaling 更伤',
          body: '同一个消融, 比的都是 200 步后的验证 loss:\n- 去掉 scaling: 差 29×。权重和梯度都掉进 FP8 表示不了的小数区。\n- 去掉 fp32 master: 差 584×。小于 FP8 步长的更新全部丢失。',
        },
      ],
      links: [
        { from: 'Adam state', to: 'fp32 master', body: '参数更新必须保留足够精度, FP16 / FP8 都一样。' },
        { from: 'global grad norm', to: 'DDP/FSDP', body: '裁剪要排在全局同步之后、unscale 之后。' },
        { from: 'LossScaler.scale', to: 'checkpoint', body: 'scale 和 good_steps 也是训练状态, 要写进 checkpoint。' },
        { from: 'FP8 block scaling', to: 'FP4 微缩放', body: '位宽再砍半后 block 必须缩到 16~32, 见「FP8 与 FP4 微缩放」。' },
      ],
      sourceRows: [
        { concept: '动态 loss scale', code: 'm06_mixed_precision/demo.py:LossScaler', takeaway: '溢出 → 减半并跳过; 连续 growth_interval (真实默认 2000) 个好 step → 翻倍。' },
        { concept: '浮点格式三要素', code: 'core/numerics.py:fake_quant_float', takeaway: '尾数位数、最大值、最小 normal 指数。BF16 / FP8 / FP4 共用这一个函数。' },
        { concept: 'NaN guard + clip', code: 'm10_training_stability/demo.py:guarded_step', takeaway: '坏 step 返回 False 且一个参数都不碰; 好 step 先按全局范数裁剪再更新。' },
        { concept: '裁剪不改方向', code: 'g * min(1, max_norm / ‖g‖)', takeaway: 'm10 里 $\\|g\\|$ 111.83 → 5.00, 裁剪前后的方向余弦是 1.000000。' },
        { concept: 'FP8 消融的两个旋钮', code: 'm13_fp8_training/demo.py:make_quantizer', takeaway: 'scaling ∈ {none, tensor, block} × 是否保留 master。A→B 只动 scaling, C→D 只动 master; B→C 同时换了 scale 粒度和反向格式 (E5M2 → E4M3)。' },
        { concept: 'E4M3 max = 448', code: 'FLOAT_FORMATS["e4m3"]', takeaway: '不是 480: 最高的那个尾数编码留给了 NaN。' },
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
      subtitle: '读完你能说清 MoE 的通信量为什么随数据变。也能说清一条放不进单卡的序列, 怎么切给多卡还能算出精确的注意力。',
      tldr: '- EP (专家并行, MoE 的专家分给多张卡): 通信量由 gating 结果决定, 模型结构定不了它。token 出门找专家, 再回家, 来回各一趟 all-to-all。\n- Ring Attention: 通信量固定。但因果 mask 下必须 zigzag 切, 否则计算量省了一半, 墙钟几乎没省。',
      question: 'all-to-all 的通信量为什么由数据决定、不由模型结构决定? Ring Attention 凭什么和完整注意力精确相等?',
      code: 'llm_train/m11_expert_parallel/demo.py · llm_train/m12_sequence_parallel/demo.py',
      points: [
        {
          title: '路由决定通信',
          key: true,
          body: 'token 所在的卡, 常常不是它选中的专家所在的卡。所以要来回两趟 all-to-all (每卡给每张卡各发一份不同的数据):\n- dispatch: token 发往专家所在的卡。\n- combine: 专家算完, 结果原路寄回。\ndemo 的 dispatch 把 token 和专家 id 分两次发, 所以计数是 3 次调用。\n通信量由谁定:\n- DDP: 每步搬多少字节由模型大小定死。\n- EP: all-to-all 发多少、发给谁, 全看 gating (路由器给每个 token 选专家) 这一次把 token 路由到了哪。',
        },
        {
          title: '不均衡会丢 token, 拖慢整步',
          body: 'm11 实测:\n- 负载不均: 64 个 token 分 4 卡, 每卡实际收到 [11, 18, 11, 24] 个 (均匀应为 16)。最热那张卡多干 50% 的活, 其余三张等它。\n- 装不下: 刻意倾斜的 router + 每专家容量 10, 直接有 5 个 token 装不下, 走残差绕过专家。\n所以均衡要写进训练流程。',
        },
        {
          title: '两种压平路由的手段',
          body: '指标是 max/mean: 最热专家收到的 token 数, 除以各专家的平均数。1.00× 是完全均衡。\n- Switch / Mixtral: 加 aux loss $L = E\\cdot\\sum f_e\\cdot P_e$。把 $f$ 当常数对 $P$ 求导, 梯度就在推平路由概率。m11: 60 步后 max/mean 从 1.62× 降到 1.12×, 丢弃 5 → 0。\n- DeepSeek-V3: 给每个专家一个可调 bias, 热门的减、冷门的加, 不往 loss 里掺东西。m11: max/mean 降到 1.00×。\n区别在副作用: aux loss 的梯度会进路由器, 和语言建模抢方向; bias 不进计算图。',
        },
        {
          title: '同一个 online softmax',
          body: '分块算注意力, 再增量合并。每收到一块, 就用新的最大值把旧的 (m, 分母, 加权和) 重新缩放一遍。\n这是恒等变形: m12 实测与完整注意力差 4.4e-16。\n- 单卡内做: FlashAttention。\n- 跨卡传块: Ring Attention, 任何时刻每卡只持有 $1/D$ 的 KV。',
        },
        {
          title: '因果 mask 下要 zigzag 切',
          body: '因果 mask 下, 整块落在未来的 KV 直接跳过, 总计算量省约一半 (q·k 对数 528, 非因果是 1024)。\n怎么切, 决定这一半能不能变成时间。墙钟的算法: 每轮取最慢那张卡的 q·k 对数, 再把各轮加起来。\n- 不利用因果性: 墙钟 256。\n- 连续切分: 卡 r 有 r+1 个块要算。最后一张卡每轮都满载, 其余人等它, 墙钟 228, 只省约 10%。\n- zigzag (序列切 $2D$ 段, 一头一尾配对): 每卡每轮工作量相同, 墙钟降到 132, 比连续切分快 1.73×。',
        },
      ],
      links: [
        { from: 'gating top-k', to: 'all-to-all dispatch', body: 'token 在哪张卡 ≠ 它的专家在哪张卡, 必须重排一次再排回来。' },
        { from: 'capacity factor', to: 'token dropping', body: '64 token、8 专家、cf=1.25 → 每专家容量 10, 溢出的 token 直接走残差。' },
        { from: 'KV 环传递', to: 'online softmax (m, l, acc)', body: '收到新块就增量合并, 任何时刻只持有 $1/D$ 的 KV。' },
        { from: '因果 mask', to: 'zigzag 切分', body: '连续切分时最后一张卡每轮满载、大家等它; 一头一尾配对后每卡每轮等量, 墙钟快 1.73×。' },
      ],
      sourceRows: [
        { concept: 'dispatch 矩阵', code: 'm11_expert_parallel/demo.py:moe_forward_ep', takeaway: '行 = 源卡, 列 = 目标卡, 这张表就是 all-to-all 的发货单。' },
        { concept: 'aux loss 的梯度', code: 'm11_expert_parallel/demo.py:train_router_aux', takeaway: '$L = E\\cdot\\sum f_e\\cdot P_e$, $f$ 视为常数, 对 $P$ 求导就在推平路由。' },
        { concept: 'aux-loss-free', code: 'm11_expert_parallel/demo.py:balance_bias', takeaway: '热门专家的 bias 往下调、冷门往上调, 不往 loss 里掺东西。' },
        { concept: '环上的一步', code: 'm12_sequence_parallel/demo.py:ring_attention', takeaway: '卡 r 在第 s 轮处理来自卡 (r−s)%D 的 KV 块, 算完把块传给右邻居。' },
        { concept: '增量合并', code: 'm_new / scale', takeaway: '用新的最大值修正旧累计量的指数基准, 与一次性 softmax 差 4.4e-16。' },
        { concept: 'zigzag 切分', code: 'm12_sequence_parallel/demo.py:shard_positions', takeaway: '序列切 $2D$ 段, 卡 $r$ 拿第 $r$ 段和第 $(2D-1-r)$ 段, 一头一尾配对。' },
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
      subtitle: '读完你拿到任何一个训练框架, 都能先把它的通信路径还原成四条原语。然后再去看它的封装。',
      tldr: '所有并行策略最后都落到四个通信原语: all-reduce、reduce-scatter、all-gather、all-to-all。\nring 实现的 all-reduce, 每卡每步只发 $2(N-1)/N\\cdot S$ 字节 ($S$ 是张量大小), 与卡数几乎无关。',
      question: '拿到一个陌生的训练框架, 从哪里开始读才最快看懂它的并行方式?',
      code: 'llm_train/core/collectives.py · llm_train/m09_collectives/demo.py · llm_train/full_loop/demo.py',
      points: [
        {
          title: '四个原语各做什么, 谁在用',
          body: '- all-reduce: 每卡交一份, 每卡拿回全部的和。DDP 用它同步梯度。\n- reduce-scatter: 同样求和, 但每卡只拿回和的 $1/N$, 也就是自己那片。\n- all-gather: 每卡交出自己那片, 每卡拿回拼好的完整张量。ZeRO/FSDP 用 reduce-scatter 和 all-gather 这一对。\n- all-to-all: 每卡给每张卡各发一份不同的数据, out[dst][src] = in[src][dst]。MoE 和 Ulysses 用它。',
        },
        {
          title: '通信原语是成本中心',
          key: true,
          body: '同一个数学公式, 不同并行策略的差别在每一步搬什么、搬多少、什么时候搬。以 all-reduce 为例, 张量大小 $S$、卡数 $N$:\n- ring 实现: 张量切 N 块沿环传, 每卡只发 $2(N-1)/N\\cdot S$, 上限 $2S$。\n- 朴素做法 (全发给 0 号卡再广播): 0 号卡收发 $2(N-1)\\cdot S$。$N=4$ 时已是 4 倍, 且随 $N$ 线性涨。',
        },
        {
          title: 'full_loop 是合成章',
          body: '同一个 train_step 里装着: rank 切分、micro 累积、AMP 放大与跳步、全局裁剪、ZeRO 分片 Adam、分片 checkpoint。\n- 40 步跳了 4 步: 3 次真实 fp16 溢出 + 1 次注入的 NaN batch。最终 scale 稳在 $2^{18}$。\n- 中途存盘再续训: master 权重差 0.0e+00。\n同一个函数传 world=1、micro=1、amp=False, 就退化成单卡基线。所以等价性断言是这份代码的自洽检验。',
        },
        {
          title: '读真实框架的入口',
          body: '先回答七个问题:\n- 切分: batch 怎么切、层怎么切、状态怎么切、序列怎么切。\n- 精度与通信: 数用几位存、通信怎么走。\n- 容错: 坏 step 怎么恢复。\n答完这七条, 再去对框架里的类名。',
        },
      ],
      links: [
        { from: 'all-reduce', to: 'DDP', body: '每卡拿到完整平均梯度; 拆开就是 reduce-scatter + all-gather。' },
        { from: 'reduce-scatter + all-gather', to: 'ZeRO / FSDP', body: '梯度和参数在"只留自己那片"与"完整视图"之间来回切换。' },
        { from: 'all-to-all', to: 'MoE / Ulysses', body: '同一个原语: out[dst][src] = in[src][dst], 发货单的转置。' },
        { from: 'full_loop', to: 'llm_finetune', body: '主循环骨架不再变, 后面换的是数据和 loss。' },
      ],
      sourceRows: [
        { concept: 'ring all-reduce', code: 'core/collectives.py:ring_all_reduce_sum', takeaway: '第 t 步 rank r 发 chunk (r−t)%N; $N=8$ 共 14 步, 每 rank 发 $2(N-1)/N\\cdot S$。' },
        { concept: '通信计数器', code: 'core/collectives.py:CommCounter', takeaway: '每个原语都记下每 rank 平均发送字节, 各 demo 打印的通信账都来自它。' },
        { concept: '完整一步', code: 'full_loop/demo.py:train_step', takeaway: '组合 DDP、累积、AMP 跳步、全局裁剪、ZeRO 分片更新。' },
        { concept: '跨分片求范数', code: 'all_reduce_sum([sum(s ** 2) for s in shards])', takeaway: '梯度已经切碎了, 全局范数只能靠每 rank 贡献自己的平方和再 all-reduce。' },
        { concept: '分片 checkpoint', code: 'full_loop/demo.py:save_sharded', takeaway: '每 rank 存自己的优化器分片 + 一个全局 meta; 恢复后轨迹逐位一致。' },
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
      tldr: '- 气泡 (卡空等的时间占比): GPipe 和 1F1B 完全一样, 都是 $(\\mathrm{PP}-1)/(M+\\mathrm{PP}-1)$。\n- 在途激活 (已前向、还没反向的激活): 1F1B 省的是这一笔。\n- 要真的减气泡只有两条路: 把 M 开大, 或者交错。',
      question: '1F1B 明明不减少气泡, 为什么大家都说它"更快"?',
      code: 'llm_train/m04_pipeline_parallel/demo.py',
      points: [
        {
          title: '两种调度只差执行顺序',
          body: 'PP 是 stage 数 (一个 stage 一张卡), M 是 micro-batch 数。\n- GPipe: 全部 micro-batch 前向完, 再全部反向。\n- 1F1B: 先做几个前向热身 (warmup), 之后严格一个前向、一个反向交替。',
        },
        {
          title: '1F1B 省显存, 不省时间',
          key: true,
          body: 'PP=4、M=8 时两种调度的气泡都是 27.3%, 但各 stage 的在途激活峰值不同:\n- GPipe: [8,8,8,8]\n- 1F1B: [4,3,2,1]\n1F1B 对时间的帮助是间接的: 显存省下来 → M 可以开大 → 气泡跟着掉 (PP=8、M=64 时只剩 9.9%)。\n规则只有一行: 卡 s 上 "已前向未反向" 的 micro-batch 不许超过 $\\mathrm{PP}-s$。',
        },
        {
          title: '交错: 气泡再除以 v',
          body: '每张卡不再拿一段连续的层, 而是拿 v 个不相邻的小 stage, micro-batch 绕着卡转 v 圈。\nPP=4、M=8、v=2: 气泡 27.3% → 15.8%。\n直觉: 气泡的绝对长度正比于 "单个 stage 的计算量"。交错把单个 stage 切小了 $v$ 倍, 总有效工作量没变, 所以公式里的 $M$ 变成 $v\\cdot M$。',
        },
        {
          title: '交错的代价',
          body: '三个代价 (PP=4、M=8、v=2):\n- 在途激活: warmup 段更长。各卡峰值 [11,9,7,5] 份, 每份半个 stage。卡 0 最多, 11 × 1/2 = 5.5 个 stage (1F1B 是 4)。\n- 跨卡传输: 每个 micro-batch 从 $2(\\mathrm{PP}-1)$ 次变成 $2(\\mathrm{PP}\\cdot v-1)$ 次。\n- 约束: M 必须是 PP 的整数倍。\n所以它适合 M 已经被显存卡住、加不上去的时候。\ndemo 把跨卡传输的时间记为 0, 真实收益还要扣掉多出来的传输。',
        },
      ],
      links: [
        { from: 'train-model-parallel', to: 'device_order', body: '上一章看时间表, 这一章看时间表是怎么由"每卡的执行顺序"生成的。' },
        { from: 'warmup 个 F', to: '在途激活上限', body: 'warmup+1 就是一张卡上"已 F 未 B"的 micro-batch 数上限。' },
        { from: '在途激活', to: '激活重算', body: '交错多出来的那部分激活, 通常用 activation checkpointing 换回来。' },
      ],
      sourceRows: [
        { concept: '每卡的执行顺序', code: 'm04_pipeline_parallel/demo.py:device_order', takeaway: 'GPipe = 全部 F 再全部 B; 1F1B = warmup 个 F 之后严格一 F 一 B。' },
        { concept: '交错的 warmup', code: '(PP - 1 - s) * 2 + (v - 1) * PP', takeaway: '比普通 1F1B 的 $\\mathrm{PP}-1-s$ 长得多, 在途激活变多就来自这里。' },
        { concept: '虚拟 stage', code: 'k = c * PP + s', takeaway: '共 $\\mathrm{PP}\\cdot v$ 个虚拟 stage, 第 k 个住在卡 k % PP 上。' },
        { concept: '事件模拟', code: 'm04_pipeline_parallel/demo.py:simulate', takeaway: 'op 的开始时刻 $= \\max(\\text{卡空闲}, \\text{依赖完成})$; 气泡 $= 1 - \\text{有效工作量}/\\text{总时长}$。' },
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
      subtitle: '读完你能解释为什么一条 WSD 主干可以随时分叉出成品模型, 而 cosine 训练做不到。',
      tldr: '- cosine: 每一步 lr 都写成 $f(\\text{step}/\\text{total})$, 总步数一改整条曲线都变。\n- WSD (warmup → stable → decay 三段): 稳定段里没有 total。任何一个稳定段 checkpoint 都能接着训, 或分叉出一段短退火。',
      question: '训到 100% 发现 loss 还在降, 想加训: cosine 和 WSD 各要付出什么?',
      code: 'llm_train/m10_training_stability/demo.py',
      points: [
        {
          title: 'warmup 在防什么',
          body: 'Adam 的二阶矩 v 是滑动平均, 最初几百步还没 "热" 起来。归一化后的步长方差很大, 配上大 lr 容易一步走飞。\n线性 warmup 就是用小步子把这段最危险的路走完。',
        },
        {
          title: 'cosine 把总步数写进了每一步',
          key: true,
          body: '$\\text{lr} = f((\\text{step} - \\text{warmup})/(\\text{total} - \\text{warmup}))$。total 出现在公式里, 一变整条曲线都跟着变。\nm10 里 total 从 100 改成 200, 前 90 步的 lr:\n- cosine: 最大差 5.96e-4\n- WSD: 差 0.00e+00\n续训 cosine 只剩两条路: re-warmup (loss 先反弹), 或者按新的 total 从头再来。',
        },
        {
          title: 'WSD 的工程价值',
          body: 'MiniCPM / DeepSeek-V3 / Kimi 用同一类调度:\n- 主干: 一直保持峰值 lr 往前训。\n- 要 "成品": 从任意 checkpoint 分叉出一段短退火 (m10 里退火占最后 10%)。\n数据追加、scaling law 实验都不用重训。退火起点前的那个 checkpoint, 是所有分支的公共祖先。\n代价: 要成品或要评测, 都得从稳定段 checkpoint 另跑一段退火 (m10 里是总步数的 10%)。\n边界: 真实 WSD 的退火形状常用 1−sqrt 或指数, m10 用的是线性。',
        },
      ],
      links: [
        { from: 'train-precision-stability', to: 'warmup', body: 'warmup、clip、NaN guard 同属"别让坏 step 毁掉整条训练"。' },
        { from: 'decay_start', to: 'checkpoint', body: '退火开始前的那个 checkpoint 最值钱: 它是所有分支的公共祖先。' },
        { from: 'WSD 退火', to: 'Muon / AdamW', body: '调度和优化器正交; Kimi K2 用的就是 Muon + WSD。' },
      ],
      sourceRows: [
        { concept: 'warmup + cosine', code: 'm10_training_stability/demo.py:warmup_cosine_lr', takeaway: '$\\text{progress} = (\\text{step} - \\text{warmup})/(\\text{total} - \\text{warmup})$: total 出现在每一步的公式里。' },
        { concept: 'WSD', code: 'm10_training_stability/demo.py:wsd_lr', takeaway: '稳定段直接 return base_lr, 这一行里没有 total_steps。' },
        { concept: '退火起点', code: 'decay_start = total_steps - int(total_steps * decay_frac)', takeaway: '只有这一个量依赖总步数, 而且只影响最后约 10%。' },
        { concept: '两种调度的实测差', code: 'total=100 → 200, 前 90 步 lr 最大变化', takeaway: 'cosine 5.96e-04, WSD 0.00e+00。' },
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
      subtitle: '读完你能解释 4 bit 为什么必须配 block scale, 以及 MXFP4 和 NVFP4 的差距全部来自哪一处。',
      tldr: '位宽砍到 4 bit 后, 单个元素的动态范围只剩 12 倍。\n只能靠每 16~32 个元素共享一个 scale 撑回来。\nMXFP4 和 NVFP4 的差距来自 scale 自己的精度。',
      question: '同样是 4 bit + block scale, 为什么 NVFP4 的误差明显低于 MXFP4? INT4 什么时候反而更好?',
      code: 'llm_train/m13_fp8_training/demo.py · llm_train/m15_fp4_microscaling/demo.py · llm_train/core/numerics.py',
      points: [
        {
          title: 'scale 由 block 里最大的那个元素定',
          key: true,
          body: '一个 block 里的 amax 决定 scale, 其余元素按它缩放后再舍入。\noutlier 越大, 同 block 的邻居被挤得越靠近 0。block 越小, 被连累的邻居越少。\n- FP8: 要等 outlier 超过约 1e5 倍才显形。其余 token 的输出相对误差: per-tensor 0.126, block 0.026。\n- FP4: 网格只有 $\\pm\\{0.5,1,1.5,2,3,4,6\\}$, 动态范围 12 倍, 不切小 block 根本用不了。',
        },
        {
          title: 'scale 自己也有精度',
          body: '格式名直接写位数: E8M0 = 8 位指数 + 0 位尾数, E4M3 = 4 位指数 + 3 位尾数。元素本身是 E2M1。\n- MXFP4: E8M0 scale 只能取 2 的幂。amax/scale 落在 [4,8) 的哪个位置全凭运气, 大于 6 的直接饱和。m15 里 42% 的 block 最大值被截断了。\n- NVFP4: E4M3 scale 带 3 位尾数, 能把 amax 几乎精确贴到 6。\n高斯权重上的相对量化误差 $\\|Q(x)-x\\|/\\|x\\|$:\n- 各用默认 block: MXFP4 (block 32) 0.113, NVFP4 (block 16) 0.095。\n- 同为 block 16: E8M0 scale 0.1138, E4M3 scale 0.0947。差距全部来自 scale 的精度。',
        },
        {
          title: '格式要对着数据分布选',
          body: '比的仍是相对量化误差:\n- 高斯分布的权重: INT4 0.096, MXFP4 0.113, INT4 低 15%。数据集中时, 均匀网格更划算。\n- 重尾数据 (激活、梯度): MXFP4 0.142, INT4 0.147, 对数间距的浮点网格低约 4%。\n两者位数也不同: MXFP4 每元素 4.25 bit, INT4 是 4.50 bit (scale 都摊进去算)。',
        },
      ],
      links: [
        { from: 'train-precision-stability', to: 'FP8 block scaling', body: '同一个 fake_quant_float, 只是换了一组 (尾数位, 最大值, 最小指数)。' },
        { from: 'quant_blockwise', to: 'mxfp4 / nvfp4', body: '两个函数只有 scale 那一行不一样。' },
        { from: 'FP4 权重', to: 'llm_infer 量化', body: 'FP4 目前主要用于推理权重和 QAT; 全程 FP4 预训练还要随机舍入、Hadamard 旋转等技巧。' },
      ],
      sourceRows: [
        { concept: 'block scaling', code: 'core/numerics.py:quant_blockwise', takeaway: '每 block 个元素共享 scale = amax / max_val; block ≥ 张量大小就退化成 per-tensor。' },
        { concept: 'MXFP4 的 scale', code: 'scale = 2.0 ** (np.floor(np.log2(amax)) - 2)', takeaway: '纯 2 的幂 (E8M0); −2 是因为 E2M1 的最大指数是 2 ($6 = 1.5\\cdot 2^2$)。' },
        { concept: 'NVFP4 的 scale', code: 'm15_fp4_microscaling/demo.py:nvfp4', takeaway: 'block scale 本身量化成 E4M3, 再乘一个整张量的 FP32 scale 把它搬进 E4M3 的范围。' },
        { concept: 'FP8 消融', code: 'm13_fp8_training/demo.py:train', takeaway: '验证 loss, 每臂只动一个旋钮: 全程 E4M3 时 per-tensor 4.64e-4, block 4.61e-4, 打平。去掉 master 差 584×。' },
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
      subtitle: '读完你能说清 Muon 赢在哪、输在哪, 以及为什么它只用在 2-D 权重上。',
      tldr: '- Adam: 逐元素看梯度。\n- Muon: 把 2-D 权重的动量当成矩阵正交化后再更新, 让每个奇异方向迈同样大的步子。\n病态方向不与坐标轴对齐时 Muon 赢, 对齐时 Adam 赢。',
      question: 'Adam 已经逐元素自适应了, 为什么还会被"病态方向"拖住? Muon 什么时候反而不如 Adam?',
      code: 'llm_train/m14_muon_optimizer/demo.py',
      points: [
        {
          title: '梯度谱极不均匀',
          body: '把梯度做 SVD: $G = U\\Sigma V^\\top$。奇异值常常差几个数量级, 更新被头几个大方向吃掉了。\nMuon 把 $\\Sigma$ 全换成 1, 更新量变成 $UV^\\top$, 每个奇异方向步长相同。相当于在谱范数下做最陡下降。',
        },
        {
          title: 'Newton–Schulz: 不做 SVD 也能正交化',
          body: '精确 SVD 太慢。Newton–Schulz 迭代 $X \\leftarrow aX + (bA + cA^2)X$ ($A = XX^\\top$) 只用矩阵乘。\nm14 对条件数 1e4 的矩阵迭代 5 步:\n- 最小奇异值: 从 1e-4 抬到 0.041。\n- 最大奇异值: 压在 1.16 附近。\n- 方向: 与精确 $UV^\\top$ 的方向余弦 0.880。',
        },
        {
          title: '与基无关',
          key: true,
          body: 'Adam 的 $1/\\sqrt{v}$ 是逐元素的, 只能修 "对角" 的病态。病态方向一旦是多个坐标的线性组合, 它就无能为力。\nm14 把同一个病态问题旋转一下, 比 150 步后的 loss (两边各扫 5 个 lr 取最好):\n- 非轴对齐: Muon 4.93e-04, Adam 4.52e-03, Muon 好 9.2×。\n- 轴对齐: Adam 5.54e-04, Muon 1.08e-03, Adam 好 1.9×。\nAdam 按坐标轴缩放, Muon 按奇异方向缩放。病态落在谁的轴上, 谁就赢。\n两边 weight decay 都是 0, 只比缩放方式这一处差别。',
        },
        {
          title: '大规模要配 QK-clip',
          body: 'Muon 的更新是满秩的, 大规模训练时 attention logit 容易爆。\nm14 只演示裁剪本身: 手工造一组偏大的 $W_q$、$W_k$, max logit 752.1, softmax 最大概率均值 1.000。已经是 one-hot, 梯度全没了。\n- Kimi K2 的 MuonClip: 每步后若 max logit $S_{\\max}$ 超过 $\\tau$, 就把 $W_q$、$W_k$ 各乘 $\\sqrt{\\tau/S_{\\max}}$。demo 取 $\\tau$=100, 裁剪后正好 100.0。\n- embedding、输出头、norm、bias: 仍然用 AdamW。正交化是矩阵概念, 对向量参数没有意义。',
        },
      ],
      links: [
        { from: 'llm_basic optim.py', to: 'Muon', body: 'Adam 存 m、v 两份状态; Muon 只存动量一份, 反而更省。' },
        { from: 'newton_schulz', to: 'GPU 友好', body: '精确的 $UV^\\top$ 要做 SVD; NS 只用矩阵乘, 5 步的开销相对前反向不到 1%。' },
        { from: '0.2·√max(n,m)', to: 'AdamW 学习率', body: 'Moonlight 的缩放: 让 Muon 更新的 RMS 接近 AdamW, 学习率和 weight decay 可以直接复用。' },
      ],
      sourceRows: [
        { concept: 'Newton–Schulz', code: 'm14_muon_optimizer/demo.py:newton_schulz', takeaway: '先除以 Frobenius 范数保证 $\\sigma \\le 1$, 再对每个奇异值作用 $a\\sigma + b\\sigma^3 + c\\sigma^5$。' },
        { concept: '一步更新', code: 'm14_muon_optimizer/demo.py:Muon', takeaway: 'Nesterov 式动量 → 正交化 → 乘 $0.2\\cdot\\sqrt{\\max(n,m)}$ → 更新。' },
        { concept: 'QK-clip', code: 'm14_muon_optimizer/demo.py:qk_clip', takeaway: '$\\eta = \\min(1, \\tau/S_{\\max})$, $W_q$、$W_k$ 各乘 $\\sqrt{\\eta}$。' },
        { concept: '对照实验', code: 'm14_muon_optimizer/demo.py:run', takeaway: 'rotate=True/False 两种病态, 各自扫 5 个 lr 取最好的再比, 否则比的是调参。' },
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
      tldr: 'Ulysses 在注意力前后各做一次 all-to-all, 换一种切法:\n- 换之前: 每卡一段序列、全部头。\n- 换之后: 每卡完整序列、一部分头。\n注意力核一行都不用改。',
      question: 'Ulysses 的通信量随 P 增大反而下降, 为什么大家没有全部换成它?',
      code: 'llm_train/m16_ulysses_sequence_parallel/demo.py · llm_train/m12_sequence_parallel/demo.py',
      points: [
        {
          title: '换切法, 不换数学',
          key: true,
          body: '- 注意力之外: 每卡持有 [T/P, H, d]。\n- 进注意力前: all-to-all 成 [T, H/P, d]。每卡看到完整序列但只算 $H/P$ 个头, 直接跑普通 FlashAttention, 算完再换回来。\n头与头之间本来互不相干, 所以结果和完整注意力对得上 (m16: P=2/4 差 0.0e+00, P=8 差 4.4e-16)。\n因果 mask 天然均衡, 不需要 zigzag。',
        },
        {
          title: '通信量随 P 下降',
          body: '每卡发 $4(P-1)/P^2$ 份张量: 4 次 all-to-all (Q、K、V、输出), 每次只发不属于自己的 $(P-1)/P$。\nm16 里 T=32、H=8、d=8, P=2/4/8 时每 rank 字节:\n- Ulysses: 16384 / 12288 / 7168\n- Ring: 16384 / 24576 / 28672。P=8 时是 Ulysses 的 4.0 倍。',
        },
        {
          title: '两个硬约束',
          body: '- P 必须整除头数: m16 里 P=16、H=8 直接 AssertionError。GQA/MQA 的 KV 头更少, 限制更紧。\n- 跨机不友好: all-to-all 吃对分带宽 (集群切成两半, 两半之间的总带宽), 不如 Ring 只和邻居点对点传。\n所以实战常见的是机内 8 路 Ulysses × 机间 4 路 Ring。',
        },
      ],
      links: [
        { from: 'train-moe-seq', to: 'Ring Attention', body: 'Ring 让 KV 块流动、Q 不动; Ulysses 让所有人一次性换位。' },
        { from: 'core/collectives.py:all_to_all', to: 'seq_to_head', body: '和 MoE dispatch 是同一个原语: out[dst][src] = in[src][dst]。' },
        { from: '[T/P, H, d]', to: '[T, H/P, d]', body: '盯住这个 shape 变化, 整个算法就看懂了。' },
      ],
      sourceRows: [
        { concept: '序列 → 头', code: 'm16_ulysses_sequence_parallel/demo.py:seq_to_head', takeaway: 'rank s 把自己的第 j 组头发给 rank j; 收到后沿序列维拼起来。' },
        { concept: '头 → 序列', code: 'm16_ulysses_sequence_parallel/demo.py:head_to_seq', takeaway: '完全对称的逆变换。' },
        { concept: '整除约束', code: 'assert H % P == 0', takeaway: '8 个头切不成 16 份, demo 里故意触发这个 AssertionError。' },
        { concept: '通信账', code: '4 * (P - 1) / P * shard_bytes', takeaway: '4 次 all-to-all, 每次只发不属于自己的 $(P-1)/P$。' },
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
      subtitle: '读完你能说清近似去重为什么要 MinHash-LSH, 以及 packing 之后要补哪三样才和逐篇训练等价。',
      tldr: '- 去重: 精确哈希只抓逐字副本, MinHash-LSH 连改过几个字的转载也抓得到。\n- 配比: 数据源温度配比 $p_i \\propto n_i^{1/T}$ 把小语料抬上来, 代价是它被重复很多遍。这里的温度管各来源的占比, 和解码时的采样温度不是一回事。\n- packing: 把多篇文档拼进一行, 省掉 padding。必须加文档 mask, 否则同一行的文档互相看见。',
      question: '转载时改了几个字, 去重还抓得到吗? 把几篇文档拼进同一行训练, 它们会互相看见吗?',
      code: 'llm_train/m17_data_pipeline/demo.py · llm_train/m18_sequence_packing/demo.py',
      points: [
        {
          title: '精确哈希只抓逐字副本',
          body: 'm17 造了 120 篇应删的重复: 30 份逐字副本, 90 份轻改转载。\n轻改指每词约 4% 被改。它们与原件的 Jaccard 均值 0.69。Jaccard 是两篇文档 5-gram 集合的交集大小除以并集大小。\n召回是这 120 篇里被抓到的比例:\n- 精确哈希 (整篇 crc32): 召回 0.258。改一个词哈希就全变, 只抓到逐字副本和 1 篇恰好没被改到的。\n- MinHash-LSH: 召回 0.967, 误杀 0。漏掉的 4 篇真实 Jaccard 是 0.48–0.52, 正好卡在 0.5 的校验线上。',
        },
        {
          title: 'MinHash 估相似度, LSH 只决定谁被比较',
          body: 'MinHash 靠一个性质: 随机置换下, 两个集合最小元素相同的概率恰好等于 Jaccard。\n每篇文档存 128 个这样的最小值当签名。两篇签名里相同位置的比例, 就是 Jaccard 的估计。\n两两全比是 $O(n^2)$。LSH 把 128 位签名切成 $b=32$ 段 × $r=4$ 行, 任一段完全相同才成候选。\n成候选的概率是一条 S 曲线 $1-(1-s^r)^b$, 拐点约 $(1/b)^{1/r} \\approx 0.42$。$s$ = 0.2 / 0.4 / 0.6 / 0.8 时是 0.05 / 0.564 / 0.988 / 1.0。\n候选对还要用签名估的 Jaccard ≥ 0.5 再校验。两个阈值分开调:\n- S 曲线: 管召回和算量。\n- 校验线: 管最终判定。',
        },
        {
          title: '配比: 数据源温度配比',
          body: '按原始比例采样, 2M token 的 math 在 1000M 库存里只占 0.2%, 几乎见不到。数据源温度配比 $p_i \\propto n_i^{1/T}$ 把它抬上来 (预算 500M):\n- T=2: math 上采样 ×13.84, 过 6.9 遍。\n- T=5: ×50.89, 过 25.4 遍, 很可能已经在背诵。\n目标配比定了, 单个 batch 里的实际占比还会偏。\n指标: 一个 batch (64 条) 内, 各来源实际占比与目标占比的最大偏差, 再取所有 batch 里最坏的。\n- iid 抽样: 0.228。\n- 额度调度: 0.021。\n额度调度是确定性的: 每步给各来源加 $p_i$ 的额度, 选额度最大的来源出一条, 扣 1。偏差不会累积。',
        },
        {
          title: '拼行把有效 token 从 0.285 提到 0.990',
          body: 'm18 的 400 篇文档中位数只有 28 token, 序列长 L=128。有效 token 比例是真 token 占全部位置的比例:\n- 每篇一行补 pad: 0.285。\n- 动态 padding (batch 内补到最长): 0.406。\n- FFD packing: 0.990, 400 篇塞进 115 行。\nFFD 的做法: 文档从长到短排, 每篇放进第一个装得下的行, 不切文档。',
        },
        {
          title: 'packing 必须配文档 mask',
          key: true,
          body: 'packing 的代价是同一行的文档互相看见。\n拿逐篇单独前向当基准, 比输出的最大绝对差:\n- 只有因果 mask: RoPE 差 3.0, 绝对位置编码差 4.3。\n- 加上文档 mask: 差 1e-15。\n文档 mask 是 block-diagonal 的: 每个 token 只许看同一篇里排在自己前面的 token。',
        },
        {
          title: '位置和 label 也要管',
          body: '- 位置重置: 绝对位置编码必须每篇从 0 开始, 否则第 2 篇起全部错位 (差 4.6)。RoPE 只看位置差, 有文档 mask 时不重置也等价 (1.3e-15)。\n- 边界 label: 每篇最后一个 token 不许去预测下一篇的第一个。一行 4 篇共 128 token, 有效预测只有 124 个。\n三样都做到, CE 总和 566.353069, 与逐篇完全相同。漏掉边界屏蔽, 变成 577.563075。',
        },
        {
          title: '这个 demo 没覆盖的',
          body: '- 垃圾是合成的: 质量过滤的垃圾按 5 种模式合成, 拦截率 1.000 没有意义。真实流水线在规则之后还要跑质量分类器 (FineWeb / DCLM)。\n- 代码被误杀: 网页规则 100% 误杀代码, 这是真问题。符号比例、停用词、词长都不适合代码, 实际做法是按来源分流, 各走各的过滤。\n- 不加 mask 很常见: 很多预训练 (GPT-2/3 风格) 直接拼接、不加文档 mask。Llama 3 加了, 并说标准预训练里影响有限, 超长序列续训时才重要。',
        },
      ],
      links: [
        { from: 'MinHash 签名', to: 'LSH 分桶', body: '签名只把 "像不像" 压成 128 个整数; 分桶才把 $O(n^2)$ 的全比变成只比同桶。' },
        { from: 'doc_mask', to: 'FlashAttention varlen', body: '真实实现不物化 [L, L] 的 mask, 传 cu_seqlens 只算对角块。本例注意力算量因此只剩 0.466。' },
        { from: 'RoPE', to: '位置重置', body: '$\\langle R(m)q, R(n)k\\rangle$ 只依赖 $n-m$, 块内相对位置没变就够。HF 仍然重置, 因为 bf16 下大角度有误差。' },
        { from: 'packing', to: 'SFT', body: 'SFT 样本短且彼此无关, packing 时文档 mask 和边界 label 屏蔽一般都要加。' },
      ],
      sourceRows: [
        { concept: 'MinHash 签名', code: 'm17_data_pipeline/demo.py:minhash_signatures', takeaway: '第 j 位 = $\\min_x (a_j x + b_j) \\bmod p$, 即随机置换下集合的最小元素。' },
        { concept: 'LSH 分桶 + 校验', code: 'm17_data_pipeline/demo.py:lsh_dedup', takeaway: '同桶的都连到桶首, 签名估 Jaccard ≥ 0.5 才确认; 并查集每簇留下标最小的一篇。' },
        { concept: 'crc32 而非 hash()', code: 'zlib.crc32(" ".join(words[i:i + n]).encode())', takeaway: 'Python 的 hash(str) 每个进程加盐, 两次运行签名不同, 去重结果不可复现。' },
        { concept: '温度配比', code: 'm17_data_pipeline/demo.py:temperature_weights', takeaway: 'T=1 按原始比例, T→∞ 趋于均匀。' },
        { concept: '文档 mask', code: 'm18_sequence_packing/demo.py:doc_mask', takeaway: '因果 ∧ 同一篇文档, 一行代码。' },
        { concept: '位置重置', code: 'm18_sequence_packing/demo.py:reset_positions', takeaway: '[0,1,2, 0,1, 0,1,2,3 …]: 每篇开头归零。' },
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
      subtitle: '读完你能用 $L(N,D)$ 算出固定算力下的最优模型大小。也能说清在 SP (标准参数化) 下, 小模型调好的 lr 为什么搬不到大模型。',
      tldr: '- scaling law: $L(N,D)=E+A/N^\\alpha+B/D^\\beta$。算力 $C \\approx 6ND$ 定死时, 模型大小有一个最优值, 两头都亏。\n- μP: 按宽度改初始化和每层 lr, 小模型上扫出的 lr 直接搬到大模型。\n本章的指数是玩具任务拟合的, 与 Chinchilla 论文的完全不同, 不能拿来推 LLM。',
      question: '钱一定, 模型做大还是数据加多? 小模型上调好的学习率, 为什么搬到大模型就不灵了?',
      code: 'llm_train/m19_scaling_laws/demo.py · llm_train/m20_mup/demo.py',
      points: [
        {
          title: 'loss 被三件事卡住',
          body: '$N$ 是参数量, $D$ 是训练用的 token 数, $L$ 是 loss。\n- 数据本身的噪声: $E$。\n- 模型太小: $A/N^\\alpha$。\n- 数据太少: $B/D^\\beta$。\nm19 真训了 6 个宽度 × 6 个数据量, 共 36 个单隐层 MLP。拟合出 $E / A / \\alpha / B / \\beta$ = 0.0080 / 1.213 / 1.075 / 460.3 / 1.125, 相对残差 0.100。\n- 噪声地板: $E$ = 0.0080, 真实标签噪声方差是 0.01。$E$ 确实是谁也消不掉的那部分。\n- 被模型卡住: N=21 的模型, 数据从 16K 加到 65K, loss 0.0608 → 0.0603, 纹丝不动。',
        },
        {
          title: '固定算力, IsoFLOP 是 U 形',
          key: true,
          body: '算力 $C \\approx 6ND$: 每个参数每个 token 前向约 2 次浮点运算, 反向约 4 次。\nIsoFLOP 曲线就是固定 $C$, 只换 $N$ 和 $D$ 的分法, 看 loss 怎么变。m19 在 $C \\approx 8 \\times 10^6$ 这条线上:\n- 太小: N=21, loss 0.0603。\n- 中间: N=81, loss 0.0285, 最好。拟合预测的最优是 N=89。\n- 太大: N=641, loss 0.0853。\n把 $D = C/6N$ 代进去对 $N$ 求最小, 得 $N_{\\text{opt}} \\propto C^a$, $a = \\beta/(\\alpha+\\beta)$。\n本例 $a$ = 0.511: 算力从 1e7 到 1e9, $N_{\\text{opt}}$ 从 100 到 1054。模型和数据大约各拿一半指数。',
        },
        {
          title: '这些数字不能拿去推 LLM',
          body: '上面的数字不能拿去推 LLM:\n- 指数: 这里 $\\alpha \\approx 1.08$、$\\beta \\approx 1.13$, 论文是 0.34 / 0.28。玩具 teacher 的 $1/k$ 谱决定了 $\\alpha$。\n- $a \\approx 0.5$ 是巧合: $a$ 只看 $\\alpha$ 与 $\\beta$ 的比值, 这里两者碰巧接近。换个 teacher 就变。\n- D/N ≈ 150, 不是 20: 这里一个 "token" 是一个回归样本, 信息量和语言 token 不可比。\n- 不是样本外检验: IsoFLOP 的 6 个点也参与了拟合。每格只跑 1 个种子。',
        },
        {
          title: 'SP 下最优 lr 随宽度漂',
          body: 'SP (标准参数化) 就是默认做法: 默认初始化, 各层同一个 lr。m20 扫了 3 个宽度 × 10 个 lr:\n- SP 最优 $\\log_2 \\text{lr}$: −5 / −7 / −11 (宽 32 / 128 / 512)。宽 16 倍, lr 左移 64 倍。\n- 直接搬: 把宽 32 调出的 lr 用到宽 512, loss 0.0976。该宽度最优 0.0215, 差 4.5 倍。',
        },
        {
          title: '为什么会漂: width 个小更新叠在一起',
          body: '原因在 Adam: 每个元素的更新约等于 lr, 与梯度大小无关。\n隐藏层有 width 个输入, width 个同向小更新叠加, 下一层激活的变化 $\\propto \\text{lr} \\cdot \\text{width}$。\n这个机制只预测 lr $\\propto 1/\\text{width}$, 也就是 16 倍。实测是 64 倍, 比预测的多。\n这组实验分不清多出来的部分: SP 的曲线不光滑 (宽 512 在 −9 到 −7 之间来回跳), lr 网格也是 2 倍一格。',
        },
        {
          title: 'μP: 三处多除一个 m',
          body: 'μP (最大更新参数化) 设 $m = \\text{width}/\\text{base}$, 比 SP 多改三处, 都是除以 $m$:\n- 输出层初始化: 标准差多除 $m$ (方差除 $m^2$)。\n- 隐藏层的 Adam lr: 除以 $m$。\n- 输出层的 Adam lr: 除以 $m$。\n输入层的 fan_in 固定是 $d_{\\text{in}}$, lr 不缩。\nm20 的结果:\n- 最优 lr: 三个宽度都是 $2^{-5}$。\n- 同一 lr 下的 loss: 越宽越好 (0.0186 → 0.0101 → 0.0088)。',
        },
        {
          title: '玩具规模下的边界',
          body: '- 规模: 只到 512 宽、150 步。\n- lr 调度: 换成常数 lr, 最优点会在 −5 / −7 / −8 之间漂, 所以 demo 用了线性衰减到 0。',
        },
      ],
      links: [
        { from: 'Kaplan (2020)', to: 'Chinchilla (2022)', body: 'Kaplan 所有规模共用一个 cosine 周期, 小数据的模型没退火完就被评估, 低估了数据 ($N_{\\text{opt}} \\propto C^{0.73}$)。m19 的 cosine 周期 = 本次的 D。' },
        { from: 'Chinchilla 最优', to: '过训练', body: '它只最小化训练算力。Llama 3 8B 训了 15T token (D/N ≈ 1900), 因为小模型推理便宜。' },
        { from: 'm19 共用 lr=0.03', to: 'μP', body: '真实的 scaling 研究要每个规模单独调 lr, 或者用 μP 让 lr 不随宽度漂。' },
        { from: 'μP', to: 'lr 幂律', body: '更省事的替代: 直接拟合最优 lr 随规模的幂律 (DeepSeek LLM), 不改参数化。' },
      ],
      sourceRows: [
        { concept: '拟合', code: 'm19_scaling_laws/demo.py:fit_scaling_law', takeaway: '变量投影: $\\alpha$、$\\beta$ 走 79×79 网格, 每格用线性最小二乘解 $E, A, B \\ge 0$, 残差按相对误差计。' },
        { concept: '闭式最优', code: 'm19_scaling_laws/demo.py:compute_optimal', takeaway: '$G = (\\alpha A / \\beta B)^{1/(\\alpha+\\beta)}$, $N_{\\text{opt}} = G (C/6)^a$; main 里用数值搜索核对到 1% 以内。' },
        { concept: '单遍 + 退火到 D', code: 'm19_scaling_laws/demo.py:train', takeaway: '每步取新样本, cosine 周期等于本次数据量。这正是 Chinchilla 对 Kaplan 的修正。' },
        { concept: 'SP vs μP', code: 'm20_mup/demo.py:init_and_lrs', takeaway: '两种参数化的全部差别是三处 /m: 输出层初始化的标准差, 隐藏层的 lr, 输出层的 lr。' },
      ],
      snippetTitle: '算力最优与 μP 的全部改动',
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
      subtitle: '读完你能看出评测数字的口径问题: 分词器、测试集污染、pass@k 估计法、裁判位置偏差。',
      tldr: '每个评测数字都有口径:\n- PPL 跟着分词器变, 跨模型要比 bits-per-byte。\n- n-gram 查得到原题, 查不到改写过的题。\n- 朴素 pass@k 系统性偏低, 要用无偏估计。\n- LLM 裁判偏爱先出现的回答, 要交换 A/B 各判一次。',
      question: '分数高是模型真会, 还是背过题、换了分词器、或者裁判偏心?',
      code: 'llm_train/m21_llm_eval/demo.py',
      points: [
        {
          title: 'PPL 是等效分支数',
          body: '$\\mathrm{PPL} = e^{\\mathrm{CE}}$, CE 是逐 token 负对数似然的均值。直观含义: 模型平均在几个词里犹豫。\nm21 用一条 20 词的马尔可夫链生成语料:\n- 均匀分布: PPL 恰好 20.000, 等于词表大小。\n- bigram 计数: 5.501。\n- 真实分布: 5.448。',
        },
        {
          title: 'PPL 不能跨分词器比',
          body: '同一段文本、同一个模型, 分词器把 token 数减半:\n- PPL: 从 5.501 变成 30.265。\n- bits-per-char: 都是 2.4598。\n总信息量没变, 只是摊到了更少的 token 上。\n平均口径也要对齐:\n- 按 token 平均: 5.501。\n- 按文档平均: 5.573, 被短文档拉高。',
        },
        {
          title: '去掉被标记的题还不够',
          key: true,
          body: '8-gram 检测 (GPT-3 用 13-gram): 测试题的任一 8-gram 在训练集出现过就判脏。\nm21 往训练语料里混了 30 道原题、30 道轻改、30 道重度改写:\n- 原题、轻改 (换 2 个词): 命中 1.00。\n- 重度改写 (每 3 词换 1): 命中 0.00, 没有完整的 8-gram 幸存。\n分数: 0.570 → 去掉被标记的题 0.436 → 真实能力 0.300。漏网的改写题仍在虚高。',
        },
        {
          title: '朴素 pass@k 偏低',
          body: 'pass@k 是每题生成 k 个答案, 至少一个通过的概率。\n每题采 n 个样本, 其中 c 个对。直接代 $1-(1-c/n)^k$ 会偏低: 它对 $c/n$ 是凹函数, 由 Jensen 不等式, 期望偏小。\n无偏估计 $1-\\binom{n-c}{k}/\\binom{n}{k}$ 是 "从 n 个里无放回挑 k 个, 至少一个对" 的概率。\nm21 ($n=20$, 300 题 × 400 次重复), $k=10$ 时:\n- 真值: 0.5864。\n- 无偏: 0.5861。\n- 朴素: 0.5470, 偏 −0.0394。\n$k=1$ 时两者相同。',
        },
        {
          title: '裁判要交换顺序',
          body: 'm21 的规则裁判给第一个位置加 0.8 分:\n- 单次判: 70.0% 选第一个 (真实 A 更好的只有 49.8%)。好的在 A 时准确率 0.994, 在 B 时只有 0.593。\n- 交换判 (A、B 各当一次第一个, 两次一致才算): 两种情况的准确率 0.993 / 0.991。\n代价:\n- 调用: 2 倍。\n- 平局: 40.7%, 都是质量接近、偏差压过真实差距的那些对。',
        },
        {
          title: '合成语料下的边界',
          body: '- 语料是合成的: 随机文本里 8-gram 几乎不会撞上, 所以干净题零误标。真实语料里版权声明、代码模板会造成误标, 一般要忽略高频 n-gram。\n- 记忆率是假设: "见过就 90% 答对" 只用来说明污染怎么抬分。\n- 交换只消位置偏差: 真实 LLM 裁判还有长度偏差和自我偏好。',
        },
      ],
      links: [
        { from: '训练 loss', to: 'PPL', body: '训练 loss 就是 CE, $e^{\\text{loss}}$ 就是 PPL, 两者一一对应。' },
        { from: '数据去重', to: '去污染', body: '同一类工具: 去重在训练集内部找近似副本, 去污染在训练集与测试集之间找。' },
        { from: '改写题', to: '其他检测法', body: 'embedding 相似度检索、让模型补全题目看能否逐字复现、只用发布日期之后的新题。' },
        { from: '平局', to: '多裁判', body: '真实做法把平局记 0.5 分, 或再加几个裁判投票。' },
      ],
      sourceRows: [
        { concept: '逐 token NLL', code: 'm21_llm_eval/demo.py:token_nll', takeaway: '均值就是 CE, $e^{\\mathrm{CE}}$ 就是 PPL。' },
        { concept: 'n-gram 污染检测', code: 'm21_llm_eval/demo.py:ngram_flags', takeaway: '训练集所有 8-gram 建一个 set, 测试题任一 8-gram 命中即判脏。' },
        { concept: '无偏 pass@k', code: 'm21_llm_eval/demo.py:pass_at_k', takeaway: 'Codex 论文的数值稳定写法: $1-\\prod_{i=n-c+1}^{n}(1-k/i)$, 不算大组合数。' },
        { concept: '交换判', code: 'm21_llm_eval/demo.py:judge_swap', takeaway: '两次都选 A 才算 A 赢, 都选 B 才算 B 赢, 不一致记平局。' },
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
