// 阶段 3 · llm_train 自测题。每章 3 题, 考取舍和"为什么", 错误选项都是常见误解。
export default {
  train: [
    {
      q: '模型放得进单卡, 只是训练太慢。第一步该上哪种并行?',
      options: ['数据并行 (DDP)', '张量并行 (TP)', '流水线并行 (PP)', 'ZeRO-3'],
      answer: 0,
      why: '模型放得下时, DDP 通信最少 (每步一次梯度 all-reduce), 且几乎线性扩展。TP/PP/ZeRO-3 都是"放不下"时才付的通信代价。',
    },
    {
      q: 'llm_train 的每个 demo 都先算一个 dense / 单机基线再 assert 一致, 这在验证什么?',
      options: ['多卡吞吐高于单机基线', '各 rank 种子一致、数据切分无重叠', '加速比接近线性、没有额外开销', '分布式路径与单机训练数学等价'],
      answer: 3,
      why: '并行只是换了"谁算哪一块、什么时候通信", 不能改变 loss → grad → update 的结果。等价性是所有并行策略的验收标准。',
    },
    {
      q: '下面哪一对"技术 → 解决的瓶颈"是错的?',
      options: ['梯度累积 → 激活显存限制了 batch', 'ZeRO → 优化器状态占显存', '1F1B → 流水线气泡太大', '激活重算 → 反向要存的中间激活太多'],
      answer: 2,
      why: '1F1B 的气泡与 GPipe 完全相同, 它省的是在途激活显存; 减气泡要靠增大 micro-batch 数或交错调度。',
    },
  ],
  'train-batch-ddp': [
    {
      q: '把 batch=32 拆成 8 + 24 两个 micro-batch 做梯度累积, 每个 micro 的 loss 已各自取均值。怎样累加才等于整 batch 的梯度?',
      options: ['直接相加: 每个 micro 已取过均值', '按样本数加权: 8/32 和 24/32', '相加后除以 micro 个数 2', '各乘 micro 大小后相加, 不再除'],
      answer: 1,
      why: 'full-batch 的 mean 是对 32 个样本平均; 每个 micro 自己的均值要再乘 micro_size/full_size 才能拼回去, 等分时才退化成"除以 N"。',
    },
    {
      q: '梯度累积和 DDP 都在"切 batch", 区别在哪?',
      options: ['累积在时间上串行换显存, DDP 在空间上并行换吞吐', '累积要通信但能提速, DDP 不通信但只省显存', 'DDP 按卡平均梯度, 与单卡大 batch 不再等价', '累积在每个 micro 之后都要 all-reduce 一次'],
      answer: 0,
      why: '两者得到的梯度在数学上相同。\n- 梯度累积: 不需要通信, 但不提速。\n- DDP: 提速, 但每步要同步一次全模型梯度。',
    },
    {
      q: 'DDP 从 8 卡扩到 64 卡, 每张卡每步要发送的梯度字节数大约怎么变?',
      options: ['变成 8 倍: 每卡要和其余所有卡交换', '变成 1/8: 每卡只负责 $S/N$ 那块', '按 $\\log_2 N$ 增长: 从 3 份变 6 份', '几乎不变, 约 $2(N-1)/N \\cdot S$'],
      answer: 3,
      why: 'ring all-reduce 把张量切 $N$ 块沿环传, 每卡只发 $2(N-1)$ 块、每块 $S/N$; 这正是 DDP 能扩到上千卡的原因。',
    },
  ],
  'train-model-parallel': [
    {
      q: 'PP=4、M=8 时, 1F1B 相比 GPipe 到底省了什么?',
      options: ['总时间槽从 22 降到 16', '气泡比例从 27.3% 降到 15.8%', 'stage 0 的激活峰值从 8 降到 4', 'stage 间的 P2P 通信量减半'],
      answer: 2,
      why: '两种调度的总时间和气泡比例 $(\\mathrm{PP}-1)/(\\mathrm{PP}-1+M)$ 完全相同。1F1B 只是限制每张卡在途的 micro-batch 不超过 $\\mathrm{PP}-s$, 所以省的是激活显存。',
    },
    {
      q: 'Megatron 张量并行里, MLP 的两个矩阵为什么是"先列切、后行切"?',
      options: ['顺序无所谓, 先行切后列切通信量一样', 'GeLU 可在各卡本地算, 前向只需一次 all-reduce', '这样前向完全不通信, 只在反向 all-reduce', '列切后先 all-gather 隐藏层, GeLU 才能算对'],
      answer: 1,
      why: '- 列切: 每张卡拿到完整输入、算出一段隐藏维, GeLU 逐元素可本地做。\n- 行切: 再把各段乘回来, 只在最后求和处 all-reduce 一次。',
    },
    {
      q: 'TP 的前向已经 all-reduce 过输出了, 反向还需要通信吗?',
      options: ['需要: 各卡的 dX 只是部分和, 要 all-reduce 求和', '不需要: 前向已同步过, 各卡的 dX 本来就完整', '需要, 但只是 all-gather 拼接各卡的 dX 分块', '只有 bias 的梯度要 all-reduce, 权重各卡独立'],
      answer: 0,
      why: '输入 X 被每张卡都用过, 它的梯度是各卡贡献之和。省掉这次 all-reduce, 传给上一层的梯度就是错的 (m03 里差 1.8e-2)。\n所以一层 Transformer 共 4 次 all-reduce。',
    },
  ],
  'train-memory': [
    {
      q: '混合精度 Adam 训练, 每个参数的 16 字节里最大的一块是什么?',
      options: ['fp16 参数 + fp16 梯度 (共 4 字节)', 'fp16 梯度的 fp32 累加缓冲 (4 字节)', '反向要保存的激活 (每参数约 8 字节)', 'fp32 master + m + v (12 字节)'],
      answer: 3,
      why: '2 + 2 + 12: 优化器状态占 3/4, 所以 ZeRO-1 只切它就能把 $16\\Psi$ 降到接近 $4\\Psi$; 激活不在这 16 字节里, 要另算。',
    },
    {
      q: 'ZeRO-2 把梯度也切成了 $1/N$, 通信量比 DDP 多多少?',
      options: ['多一倍: reduce-scatter 和 all-gather 各算一次全量', '多 50%: 同 ZeRO-3, 每层还要 all-gather 参数', '不多: all-reduce 本就是 reduce-scatter + all-gather', '少一半: 只需 reduce-scatter, 省掉了 all-gather'],
      answer: 2,
      why: 'DDP 的 all-reduce 拆开就是 reduce-scatter + all-gather。\n- ZeRO-2: 在 reduce-scatter 之后先各自更新分片, 再 all-gather 参数。总量不变。\n- ZeRO-3: 每层都要 gather 参数, 涨到约 1.5×。',
    },
    {
      q: 'L 层网络做激活重算, 把段长 k 设成 L (只保存输入) 能把激活显存降到最低吗?',
      options: ['能: 只常驻 1 份输入, 峰值与 L 无关', '不能: 反向要重算整段 L 层, 峰值又回到 ~L', '能, 但每段都要重算, 计算量涨到约 L 倍前向', '不能: 重算的激活有舍入差, 梯度不再精确'],
      answer: 1,
      why: '峰值 $\\approx L/k\\,(\\text{常驻边界}) + k\\,(\\text{当前段瞬时激活})$, $k=\\sqrt{L}$ 时最小; 重算的梯度与全存逐位相同, 总计算只多约一次前向 (+33% 以内)。',
    },
  ],
  'train-precision-stability': [
    {
      q: 'BF16 相比 FP16, 赢在哪、输在哪?',
      options: ['范围同 FP32 免 loss scaling, 但尾数少 3 位', '尾数多 3 位更精确, 但范围小, 仍要 loss scaling', '范围和精度都优于 FP16, 只是硬件支持更少', '范围大到免 scaling, 精度也与 FP32 相同'],
      answer: 0,
      why: 'BF16 = 8 位指数 + 7 位尾数: 1e-8 不下溢、70000 不上溢, 但 $1+2^{-9}$ 会被舍回 1; FP16 = 5 + 10 位, 精度高 8 倍但小于 6e-8 就归零。',
    },
    {
      q: '动态 loss scaling 检测到梯度里有 Inf 时, 正确的做法是?',
      options: ['把 Inf 替换成 0 继续更新', '把 Inf 裁剪到 65504 继续更新', '回滚到上一个 checkpoint, 用更小 scale 重跑', '跳过这一步不更新, 并把 scale 减半'],
      answer: 3,
      why: '溢出的梯度没有任何可信信息, 任何"修补"都会污染 fp32 master。跳过一步几乎没有代价, scale 减半后下一步就能恢复正常。',
    },
    {
      q: 'm13 的 FP8 消融里, "去掉 scaling"和"去掉 fp32 master weights"哪个伤害更大?',
      options: ['去掉 scaling (差 29×)', '两者一样', '去掉 master (差 584×)', '都没有影响'],
      answer: 2,
      why: '两者缺一不可, 但 master 更要命:\n- scaling: 决定数值能不能进 FP8 的网格。\n- master: 没有高精度 master 时, 小更新每一步都被舍掉, 训练根本不收敛。',
    },
  ],
  'train-moe-seq': [
    {
      q: 'EP 的 all-to-all 和 DDP 的 all-reduce, 通信量上最大的区别是?',
      options: ['all-to-all 只传 top-k 的 token, 总量总是更小', 'all-to-all 的量取决于路由结果, 倾斜时出现热点卡', '两者都由模型大小固定, 只是传输拓扑不同', 'all-reduce 随 batch 变, all-to-all 由专家参数量固定'],
      answer: 1,
      why: 'all-reduce 每步搬的字节数由模型大小固定。all-to-all 发多少、发给谁取决于 gating: 路由倾斜 = 某些卡收爆、其余闲着, 所以均衡是训练目标的一部分。',
    },
    {
      q: '因果注意力下做 Ring Attention, 序列连续切给 4 张卡。mask 省掉了近一半计算, 墙钟时间省了多少?',
      options: ['只省约 11%: 每轮都要等满载的最后一张卡', '省约一半: 每张卡的计算都减半, 墙钟跟着减半', '省约 48%: 与 zigzag 切分后的效果相同', '一点没省: mask 掉的块仍要走完一轮通信'],
      answer: 0,
      why: '环上每一轮的耗时由最忙的卡决定 (228 vs 256)。zigzag 把序列切 2D 段、一头一尾配对, 每卡每轮工作量相同, 墙钟降到 132, 快 1.73×。',
    },
    {
      q: 'Ring Attention 分块算出的注意力与一次性算的相比?',
      options: ['近似: 每块各自 softmax 再平均, 误差随块数增大', '只在无因果 mask 时精确, 有 mask 时有截断误差', '等价, 但必须先 all-reduce 出全局最大值再算', '精确等价: online softmax 合并时重缩放旧累计量'],
      answer: 3,
      why: '每收到一块, 就用新的最大值重新缩放旧的累计量 (m, 分母, 加权和)。这与 FlashAttention 是同一个恒等式, 不是近似。',
    },
  ],
  'train-collectives-loop': [
    {
      q: 'Ring all-reduce 为什么要把张量切成 N 块?',
      options: ['切块后环上只需走 2 步, 通信延迟更低', '分块后每卡只需 $S/N$ 的缓冲区, 省下显存', '所有链路同时传不同的块, 每卡只发 $2(N-1)/N \\cdot S$', '每块可单独压成低精度再传, 省带宽'],
      answer: 2,
      why: '- 不切块: 每步要传整个张量。\n- 切成 $N$ 块: 所有链路同时满载, 且互不重复。\n代价是步数变成 $2(N-1)$, 所以小张量要先攒成 bucket。',
    },
    {
      q: 'ZeRO-2 / FSDP 的梯度同步用的是哪个原语?',
      options: ['all-reduce', 'reduce-scatter', 'reduce + broadcast', 'all-to-all'],
      answer: 1,
      why: 'reduce-scatter = 求和后每卡只留自己那 $1/N$。这正是"每卡只负责更新自己那片参数"所需要的。它也是 ring all-reduce 的前半段。',
    },
    {
      q: 'full_loop 的 40 步里有 4 步被跳过。这说明了什么?',
      options: ['溢出和坏 batch 是常态, 系统要能跳步且状态不乱', '实现有 bug: 正确的混合精度不该出现溢出', '学习率过大导致发散, 应降低 LR 后重训', '数据里有坏样本, 应停训清洗后从头再来'],
      answer: 0,
      why: '3 次真实 fp16 溢出 + 1 次注入的 NaN batch 都被 guard 拦下, master 未被污染; resume 后轨迹逐位一致。容错是主循环的一部分, 不是补丁。',
    },
  ],
  'train-pipeline-schedules': [
    {
      q: '既然 1F1B 不减少气泡, 它为什么能间接让训练更快?',
      options: ['它把气泡里的空等填上了其他 micro 的反向', '前向反向交替, 让 stage 间 P2P 通信量减半', '在途激活少了, 每个 micro 的反向算得更快', '省下的激活显存可用来加大 M, 气泡随之下降'],
      answer: 3,
      why: 'GPipe 的在途激活 = M, M 一大就爆显存; 1F1B 把它钉在 PP, M 可以放心加大, PP=8、M=64 时气泡只剩 9.9%。',
    },
    {
      q: '交错 1F1B (v=2) 把气泡从 27.3% 降到 15.8%, 代价是什么?',
      options: ['每卡存 v 份参数副本, 参数显存约翻 v 倍', '梯度要额外 all-reduce, 结果与 1F1B 不再逐位一致', 'P2P 次数约 v 倍, 在途激活也更多', 'P2P 次数不变, 但每次传输量变成 v 倍'],
      answer: 2,
      why: '每卡拿 v 个不相邻的小 stage, micro-batch 要绕卡 v 圈。warmup 更长, 更多激活在途。\n所以只在 M 加不上去、PP 又必须跨机时才值得开。',
    },
    {
      q: '为什么气泡公式里是 $v \\cdot M$?',
      options: ['等效于每卡同时处理 v 倍的 micro-batch', '每个小 stage 耗时缩成 $1/v$, 灌满/排空的空等也缩 v 倍', '流水线深度变成 v·PP 级, 卡数等效翻了 v 倍', '小 stage 的反向快了 v 倍, 只压缩了反向的空等'],
      answer: 1,
      why: '气泡的绝对长度 $\\propto (\\mathrm{PP}-1) \\times$ 单个 stage 的耗时; 交错把"单个 stage"切小了 v 倍, 而总有效工作量不变。',
    },
  ],
  'train-lr-schedule': [
    {
      q: '用 warmup + cosine 训完 100% 后想再加训 60%, 最大的麻烦是?',
      options: ['LR 已退到近 0: 只能 re-warmup 或按新总步数重来', '只需把 total 改大, 已训部分的曲线不受影响', 'cosine 与 batch 大小绑定, 续训必须保持原 batch', 'Adam 的 m、v 随 LR 归零失效, 必须重置优化器'],
      answer: 0,
      why: 'cosine 的 $\\mathrm{lr} = f(\\mathrm{step}/\\mathrm{total})$, total 一变整条曲线都变 (m10: 前 90 步最大差 5.96e-4); 已经走过的退火无法"撤销"。',
    },
    {
      q: 'WSD 为什么能随时从主干分叉出一个"成品"模型?',
      options: ['全程 LR 都很小, 任何 checkpoint 都已收敛', '省掉了 warmup, 从第 0 步起就能直接导出', '衰减用 cosine 且全程进行, 任意一步截下都能用', '稳定段 LR 恒定、与总步数无关, 任一点接短退火即可'],
      answer: 3,
      why: '只有最后 ~10% 的退火段依赖总步数。主干继续高 LR 往前训, 分支各自退火。加数据、做 scaling law 实验都不用重训。',
    },
    {
      q: 'warmup 主要在防什么?',
      options: ['开头梯度很小, 大 LR 下 fp16 容易下溢', '防过拟合: 前几百步模型会记住早期 batch', '开头 Adam 的二阶矩 v 还不准, 大 LR 一步就飞', '开头激活值大, 大 LR 会让显存峰值溢出'],
      answer: 2,
      why: 'v 的滑动平均在前几百步还没"热"起来, 归一化后的步长方差很大。线性 warmup 用小步子走过这段最危险的路。',
    },
  ],
  'train-low-precision': [
    {
      q: 'FP8 E4M3 训练里, per-tensor scale 和 block-128 scale 在普通张量上的差距有多大?',
      options: ['block 好约 5 倍 (0.126 vs 0.026), 普通张量也如此', '基本打平 (4.54e-4 vs 4.61e-4), 有大 outlier 才拉开', 'per-tensor 会让小值大面积下溢, 误差高一个量级', 'block 明显更差: 每块 scale 的舍入抵消了收益'],
      answer: 1,
      why: 'E4M3 自带 $2^{15}$ 以上的动态范围, 没有极端 outlier 时一个 scale 就够; 细粒度 scale 是给 outlier 上的保险 (1e5 倍时误差 0.126 vs 0.026)。',
    },
    {
      q: '同为 block 16 的 FP4, E4M3 scale (NVFP4) 为什么比 E8M0 scale (MXFP4) 准?',
      options: ['E8M0 只能取 2 的幂, 贴不准 amax; E4M3 能贴到 6', 'E4M3 的指数范围比 E8M0 大, 能覆盖更极端的 block', 'E8M0 scale 占 8 位, 挤占了元素的尾数位', 'scale 精度相同, 差距来自 NVFP4 多一层 per-tensor scale'],
      answer: 0,
      why: 'E2M1 的最大值是 6。scale 对不准 amax, 要么浪费量程要么截断最大值 (42% 的 block 被饱和); 同为 block16 时误差 0.114 vs 0.095, 差距全部来自 scale 的精度。',
    },
    {
      q: '"FP4 的对数网格一定比 INT4 的均匀网格好", 对吗?',
      options: ['对: 对数网格在 0 附近更密, 任何分布误差都更小', '对: FP4 有指数位, 动态范围大, 从不截断', '不对: INT4 在所有分布上都更好, 包括重尾激活', '不对: 高斯权重上 INT4 反而更好, 重尾分布上浮点占优'],
      answer: 3,
      why: '- 对数网格: 0 附近密、远处疏, 适合偶发大值的重尾分布。\n- 均匀网格: 集中的高斯分布用它更划算。\n格式要对着数据分布选。',
    },
  ],
  'train-muon': [
    {
      q: 'Muon 对动量矩阵做 Newton–Schulz 正交化, 效果是什么?',
      options: ['逐元素取符号, 每个元素的更新都变成 ±1', '把更新的谱范数裁到 1, 奇异值比例不变', '奇异值都推到 ~1, 各奇异方向步长相近', '只保留前几个大奇异方向, 做低秩更新'],
      answer: 2,
      why: '$\\mathrm{NS}(M) \\approx UV^\\top$: 保留奇异向量、抹平奇异值。原本被头几个大奇异方向主导的更新, 变成各方向齐步走 (5 步把 1e-4 抬到 0.041, 放大数百倍)。',
    },
    {
      q: 'Adam 已经逐元素自适应了, 为什么在"旋转过的病态问题"上还是输给 Muon 9.2×?',
      options: ['Adam 的 LR 没调到最优, 调好后就能追平', 'Adam 只能修轴对齐的病态, 正交化与基无关', 'Muon 用了二阶 (Hessian) 信息, Adam 只有一阶', 'Adam 的 $\\epsilon$ 太大, 抹掉了小梯度方向'],
      answer: 1,
      why: '病态方向一旦是多个坐标的线性组合, 逐元素的 $1/\\sqrt{v}$ 就无能为力。反过来, 病态恰好轴对齐时 Adam 赢 1.9×。两者是互补的归纳偏置。',
    },
    {
      q: '为什么 Muon 只用于 2-D 权重矩阵, embedding / 输出头 / norm / bias 仍用 AdamW?',
      options: ['正交化只对线性映射有意义; 向量和 token 表不是', 'Muon 的状态比 AdamW 多, 大词表上显存吃不消', 'NS 迭代太慢, 大词表上每步开销不可接受', '这些参数梯度稀疏, 正交化会把噪声方向放大'],
      answer: 0,
      why: 'Muon 的前提是"这个参数是一个线性映射, 谱范数是它合适的几何"。Muon 状态只有 1 份动量, 比 Adam 的 $m+v$ 省一半, 并不更费显存。',
    },
  ],
  'train-ulysses': [
    {
      q: 'Ulysses 在注意力前做的 all-to-all, 把每卡的张量从什么形状变成什么形状?',
      options: ['[T, H, d] → [T/P, H, d]', '[T/P, H, d] → [T/P, H/P, d]', '[T, H/P, d] → [T, H, d]', '[T/P, H, d] → [T, H/P, d]'],
      answer: 3,
      why: '从"每卡一段序列、全部头"换成"每卡完整序列、一部分头"。头与头互不相干, 所以每卡直接跑普通 FlashAttention, 结果与完整注意力逐位相同。',
    },
    {
      q: 'P 从 2 增到 8, Ulysses 和 Ring 每卡的通信量分别怎么变?',
      options: ['都上升: 卡越多, 每卡要连的对端越多', '都下降: 每卡序列变短, 两者都约按 $1/P$ 缩', 'Ulysses 按约 $1/P$ 下降, Ring 基本不降', 'Ring 下降、Ulysses 不变: 只有 Ring 分块传 KV'],
      answer: 2,
      why: 'demo: Ulysses 16384 → 7168 字节, Ring 16384 → 28672, P=8 时差 4 倍。Ulysses 每次只发自己 $1/P$ 里的 $(P-1)/P$; Ring 要把 KV 块传 $P-1$ 轮。',
    },
    {
      q: '既然 Ulysses 通信更省, 为什么长序列训练还常常要叠一层 Ring?',
      options: ['Ulysses 换头后结果是近似, Ring 才逐位精确', 'P 须整除头数, 且 all-to-all 跨机吃带宽', 'Ulysses 不支持因果 mask, 要靠 Ring 处理', 'Ulysses 显存随 P 上升, 超过 8 卡放不下'],
      answer: 1,
      why: '8 个头最多 8 路 Ulysses; 再往上只能靠 Ring。常见组合是机内 (NVLink) 8 路 Ulysses × 机间 4 路 Ring, 各取所长。',
    },
  ],
  'train-data-packing': [
    {
      q: 'MinHash-LSH 用 $b=32$ 段 × $r=4$ 行。Jaccard 0.6 的一对文档成为候选的概率约是多少?',
      options: ['约 0.13 ($0.6^4$)', '约 0.56', '约 0.988', '1.0: 超过拐点 0.42 就一定成候选'],
      answer: 2,
      why: '$1-(1-0.6^4)^{32} = 1-(1-0.1296)^{32} \\approx 0.988$。单段 4 行全相同的概率只有 0.13, 但 32 段里任一段撞上就行。S 曲线不是阶跃: $s=0.4$ 时仍有 0.564。',
    },
    {
      q: 'shingle 的哈希为什么用 zlib.crc32, 不用 Python 自带的 hash()?',
      options: ['hash() 在长字符串上碰撞多, crc32 更均匀', 'hash(str) 每进程随机加盐, 签名不可复现', 'crc32 有 C 实现, 比 hash() 快一个量级', 'hash() 返回 64 位, 签名矩阵太占内存'],
      answer: 1,
      why: 'Python 默认开 PYTHONHASHSEED 随机化, 同一个字符串在两个进程里 hash 不同。分布式去重的各个 worker 也就对不上签名。',
    },
    {
      q: '长度 [3, 2, 4] 的三篇文档 pack 成一行, 位置 id 应该是什么?',
      options: ['[0,1,2,3,4,5,6,7,8]', '[0,0,0,1,1,2,2,2,2]', '[0,1,2,3,4,0,1,2,3]', '[0,1,2,0,1,0,1,2,3]'],
      answer: 3,
      why: 'reset_positions 让每篇从 0 开始。[0,0,0,1,1,2,2,2,2] 是文档 id, 不是位置。\n不重置时:\n- RoPE: 只要有文档 mask, 仍然等价 (m18 差 1.3e-15)。\n- 绝对位置编码: 差 4.6。',
    },
    {
      q: '一行 pack 了 4 篇文档, 共 128 个 token。有效的 next-token 预测有几个?',
      options: ['124', '127', '128', '32'],
      answer: 0,
      why: '128 − 1 − 3 = 124:\n- 整行最后一个 token: 没有下一个 (−1)。\n- 另外 3 篇的末尾 token: 不许去预测下一篇的开头 (−3)。\nm18 里不屏蔽边界时是 127 个, CE 总和 577.56。逐篇是 566.35。',
    },
  ],
  'train-scaling': [
    {
      q: '若拟合出 $\\alpha = \\beta$, 算力翻 4 倍, 最优模型大小 $N_{\\text{opt}}$ 变几倍?',
      options: ['×1', '×4', '×2', '×16'],
      answer: 2,
      why: '$a = \\beta/(\\alpha+\\beta) = 0.5$, $4^{0.5} = 2$。$D_{\\text{opt}}$ 也 ×2, 乘起来正好 ×4。',
    },
    {
      q: 'm19 拟合出 $a = 0.511$, 和 Chinchilla 的 0.5 很接近。这说明什么?',
      options: ['玩具任务复现了 Chinchilla 的 scaling law', '说明不了什么: $a$ 只取决于 $\\alpha/\\beta$, 碰巧接近', 'scaling 指数跨任务普适, 玩具任务也会得 0.5', '说明最优 D/N 也应该是 20 左右'],
      answer: 1,
      why: '指数本身 (1.08 / 1.13) 和论文 (0.34 / 0.28) 完全不同。换个 teacher 谱, $\\alpha$ 和 $a$ 都会变。\n本例最优 D/N ≈ 150, 因为一个 "token" 是一个回归样本, 和语言 token 的信息量不可比。',
    },
    {
      q: 'Llama 3 8B 训了 15T token, D/N ≈ 1900, 远超 Chinchilla 最优的 ~20。为什么?',
      options: ['D/N 越大训练越省算力, 同 loss 下总 FLOPs 更低', '小模型要多看 100 倍数据, 才能到同算力下最低 loss', 'Chinchilla 公式有误, 后续研究把最优比改到了 ~2000', 'Chinchilla 只算训练成本, 没算推理成本'],
      answer: 3,
      why: '推理成本不在 $L(N,D)$ 里。同样的 loss, 小模型每次推理都便宜, 多花的训练算力很快就摊回来了。',
    },
    {
      q: 'base 宽度 256, 目标宽度 4096。按 Adam 下的 μP, 隐藏层的 lr 应该是多少?',
      options: ['η', 'η/16', 'η/4', 'η/256'],
      answer: 1,
      why: '$m = 4096/256 = 16$, 隐藏层和输出层的 lr 都除 $m$。输入层的 fan_in 是固定的 $d_{\\text{in}}$, 不缩。对照: SP 下 m20 宽 16 倍时最优 lr 左移了 64 倍。',
    },
  ],
  'train-eval': [
    {
      q: '每题采 $n=20$ 个样本, 其中 $c=2$ 个对。pass@10 的无偏估计是多少?',
      options: ['0.651', '0.763', '0.10', '1.0'],
      answer: 1,
      why: '$1-\\binom{18}{10}/\\binom{20}{10} = 1-43758/184756 \\approx 0.763$。0.651 是朴素公式 $1-0.9^{10}$, 它按有放回抽样算, 偏低。',
    },
    {
      q: '两个模型用不同的分词器, 想比谁的语言建模更好, 该比什么?',
      options: ['PPL, 各自分词器下越低越好', '每 token 平均交叉熵 (nats/token)', '同一测试集上各自的 token 级准确率', 'bits-per-byte'],
      answer: 3,
      why: 'PPL 和 loss 都是按 token 平均的, 跟着分词器变。\nm21: 同一模型同一文本, token 数减半, PPL 从 5.501 变成 30.265, bits-per-char 都是 2.4598。',
    },
    {
      q: 'm21 去掉被 8-gram 标记的题后分数是 0.436, 真实能力只有 0.300。差的那部分从哪来?',
      options: ['8-gram 太短, 把大量干净题误标成污染', '去掉题目后样本变少, 0.436 方差太大', '模型在剩下的干净题上运气好, 属采样噪声', '改写过的题没有完整 8-gram 幸存, 漏检了'],
      answer: 3,
      why: '干净题的误标率是 0.00, 不是误标的问题。改写题要靠别的办法:\n- embedding 检索。\n- 补全测试: 让模型补全题目, 看能否逐字复现。\n- 新题: 只用发布日期之后的题。',
    },
    {
      q: 'LLM 裁判有位置偏差。交换 A/B 顺序各判一次, 两次一致才算, 为什么能消掉偏差?',
      options: ['每个回答在 A、B 各出现一次, 位置加分两边抵消', '两次分数取平均, 位置偏差被平均掉一半', '交换后裁判意识到被测试, 就不再偏心', '它同时抵消了位置、长度和自我偏好三种偏差'],
      answer: 0,
      why: 'm21: 准确率 (好的在 A / 在 B) 从 0.994 / 0.593 变成 0.993 / 0.991。代价是 2 倍调用和约 40% 平局。\n长度偏差、自我偏好与位置无关, 交换消不掉。',
    },
  ],
}
