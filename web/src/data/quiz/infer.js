// 阶段 5 · llm_infer 自测题 (每章 3 题; 错误选项都是常见误解)
export default {
  infer: [
    {
      q: '训练时最贵的是反向传播, 推理服务里 decode 阶段最常见的瓶颈是什么?',
      options: ['tokenizer: 每出 1 个 token 都要把整段文本重新分词', '矩阵乘的 FLOPs: 每个 token 约 2×参数量次乘加, 算力先打满', '显存带宽: 每个 token 都要把权重和 KV 读一遍', 'softmax 的指数运算: 长上下文下 exp 次数随 t 平方增长'],
      answer: 2,
      why: 'decode 每步只算 1 个 token, 算术强度极低, 时间花在"搬数据"上。所以量化、GQA/MLA、投机解码都在减少每个 token 要搬的字节, 或摊薄这次搬运。',
    },
    {
      q: '下面哪一项优化会改变模型的输出 (greedy 下)?',
      options: ['PagedAttention', '权重 INT4 量化', 'continuous batching', 'KV cache'],
      answer: 1,
      why: '- 无损: cache、分页、调度、前缀复用、投机解码都只改执行路径, 输出逐 token 相同。\n- 有损: 量化、sink cache、稀疏 attention, 要看误差。',
    },
    {
      q: 'mini-vLLM 的 Engine.step 每一步做的第一件事是什么?',
      options: ['postprocess: 先回收已结束序列的 block', 'model.forward: 先对所有 running 序列跑一次前向', 'sample: 先为上一步的 logits 采出新 token', 'scheduler.schedule(): 定哪些序列各算几个 token'],
      answer: 3,
      why: '推理引擎首先是调度器和资源管理器: 先决定 batch = [(seq, n)], 才轮到前向和采样。',
    },
  ],
  'infer-kv-memory': [
    {
      q: 'block_size=8, 某条序列的页表是 [9, 4, 6, 1, 3]。第 37 个位置 (pos=37) 的 KV 存在哪里?',
      options: ['block 1 的第 5 格', 'block 37 的第 0 格', 'block 6 的第 5 格', 'block 3 的第 5 格'],
      answer: 3,
      why: '逻辑页 = 37 // 8 = 4 → 页表第 4 项是物理 block 3; 页内偏移 = 37 % 8 = 5。和操作系统的虚拟内存翻译完全一样。',
    },
    {
      q: 'PagedAttention 让 attention 算得更快了吗?',
      options: ['是: 分成 block 后, 每块可以独立并行算 attention', '是: 只读有效 block, 省掉了 padding 的 FLOPs', '没有: 它换来的是显存利用率和更大 batch', '没有: 它的目的是突破单卡的上下文长度上限'],
      answer: 2,
      why: '多一层页表间接寻址, 单次 attention 甚至略慢。收益在于不再按 max_len 预留连续显存。浪费只剩每条序列最后一个 block 的尾巴, 同样显存能塞下更多请求。',
    },
    {
      q: '有了 KV cache 之后, 第 t 步 decode 还有什么开销是随 t 增长的?',
      options: ['旧 token 的 K 要按新位置重新做 RoPE 旋转', 'MLP 要对全部 t 个 token 重算, 因为 LayerNorm 统计量变了', '没有了: 只算 1 个新 token, 每步严格 $O(1)$', '新 query 要和全部 t 个缓存 key 做点积'],
      answer: 3,
      why: 'cache 省掉的是旧 token 的 K/V 投影和 MLP。新 query 对 $t$ 个 key 的 attention 仍是 $O(t)$ 的访存。长上下文 decode 因此变慢, 也因此需要 m22 稀疏读取。',
    },
  ],
  'infer-scheduler': [
    {
      q: '队首请求因为 block 不够而无法被接纳, prefill 优先的调度器这一步应该怎么办?',
      options: ['跳过队首, 先接纳后面放得下的短请求', '落到 decode: 继续推进 running 的序列', '抢占全部 running 序列, 把 block 腾给队首', '返回空 batch, 等 running 序列结束释放 block'],
      answer: 1,
      why: '返回空 batch 会活锁: running 不前进 → block 永远不释放 → 队首永远进不来。Python 里就是 `_admit(budget) or _schedule_running(budget)` 这一行。',
    },
    {
      q: 'token 预算 B=256, 当前有 10 条序列在 decode。这一步最多能塞多大的 prefill chunk?',
      options: ['25', '256', '246', '128'],
      answer: 2,
      why: 'decode 优先: 每条 running 先拿 1 个 token, 剩下 $256 - 10 = 246$ 个预算给 prefill chunk。预算封顶了每步耗时, 也就封顶了 TBT。',
    },
    {
      q: '开启 chunked prefill (B=128) 后, demo 里 decode 用户的最大 TBT 从 297 ms 降到 52 ms。代价是什么?',
      options: ['总吞吐减半: 每步 token 数被 B=128 封顶', '长请求的 TTFT 从 276 ms 涨到 445 ms', '输出会变: 分块 prefill 的 logits 与整段不同', '没有代价: 切块只改执行顺序, 谁都不会变慢'],
      answer: 1,
      why: '长 prompt 被切成多步, 每步还要付一次固定开销, 所以新请求的首 token 更晚。chunked prefill 是拿新请求的 TTFT 换其他人的延迟平稳。',
    },
  ],
  'infer-decode-control': [
    {
      q: 'draft 模型很差 (接受率很低), 投机解码的输出质量会怎样?',
      options: ['变差: 被接受的 token 来自 draft 的分布', '不变, 只是变慢 (甚至比不投机还慢)', 'greedy 下不变, 采样时分布偏向 draft', '略变差: 被拒的位置改从 draft 重采样'],
      answer: 1,
      why: '每个被接受的 token 都经过 target 验证; 拒绝时从残差 $\\max(0, p-q)$ 重采样, 合起来的分布恰好是 target 的 $p$。draft 只影响速度。greedy 和采样两种模式下都成立。',
    },
    {
      q: '接受率 $\\alpha=0.8$ 时, 把 draft 长度 $K$ 从 4 加到 100, 每次 target 调用的期望 token 数最多到多少?',
      options: ['5', '约 3.4', '101', '约 80'],
      answer: 0,
      why: '$(1-\\alpha^{K+1})/(1-\\alpha)$ 的上限是 $1/(1-\\alpha) = 5$。第一个猜错后面全废, 所以链式 draft 加长很快就没有收益, 这正是树形投机的动机。',
    },
    {
      q: '采样模式下, 如果把接受规则改成"draft 的 token 全部接受", 会发生什么?',
      options: ['更快且分布不变: target 仍验证了每个槽位', '输出分布变成 draft 的分布, 不再无损', '只有 bonus token 有偏, 其余位置不变', '接受率高时仍无损, 接受率低时才有偏'],
      answer: 1,
      why: 'demo 里这个错误规则的 $\\chi^2$ 是 585 (临界值 37.7), 而正确的 $\\min(1, p/q)$ + 残差重采样 $\\le 13.6$。"无损"完全来自这条接受规则。bonus token 直接采自 target, 有偏的是 draft 填的那几位 (demo 里第 2、3 位)。',
    },
    {
      q: 'beam 宽度 $w=2$ 时, 10 个 prompt 里有 2 个的序列 log 概率反而低于 greedy ($w=1$)。原因是?',
      options: ['长度惩罚把更长的 greedy 序列分数压了下去', '各 beam 拷贝 KV 时引入数值误差, 逐步累积', 'beam 内部有采样, 两次运行的随机噪声不同', 'greedy 前缀某步排到第 3 被剪掉, 之后回不来'],
      answer: 3,
      why: 'beam 不是精确搜索, 只保证每步留下当时最高的 $w$ 条。demo 里:\n- $w=2/4/8$: 分别在 2/1/1 个 prompt 上输给 greedy, 只有平均意义上 $\\ge$ greedy。\n- $w=32$: 这 10 个 prompt 全部 $\\ge$ greedy。',
    },
    {
      q: '模型会输出 EOS, beam search 不加长度惩罚 ($\\alpha=0$)。输出最可能是什么样?',
      options: ['比 greedy 更长: 累计概率高的长序列被保留', '近乎空: 第 1 步就收尾的候选分数最高', '和 greedy 相同: 不加惩罚时 beam 退化成 greedy', '长度随 $w$ 增大: 搜得越宽, 越能找到长句'],
      answer: 1,
      why: '分数是负的 log 概率累加, 多一个 token 只会更低。demo 给 EOS logit +3 后, $\\alpha=0$ 和 $0.5$ 的平均长度都是 1.3 token, $\\alpha=1$ 才到 15.7 (greedy 11.4)。',
    },
    {
      q: 'TinyLM 上 (10 个 prompt × 20 token, 不停在 EOS) beam $w=32$ 的 logP/token 从 greedy 的 −2.620 升到 −2.255, rep-2 从 0.453 升到 0.563。下面哪句话对?',
      options: ['8 条 beam 候选差异很大, 平均差 19.7 个位置', '输出变短了: 序列一短, logP/token 自然升高', 'rep-2 随宽度单调上升, 每宽一档都更重复', '更可能的序列更爱复读: 似然高不等于质量好'],
      answer: 3,
      why: '似然和质量不是一回事。同一 prompt 的 8 条 beam 候选平均只差 3.5/20 个位置, 8 条采样差 19.7。开放式对话要多样性, 所以默认用采样。\nrep-2 中间几档不单调 ($w=4$ 是 0.447)。这组实验固定生成 20 个 token, 长度没变。',
    },
  ],
  'infer-compute': [
    {
      q: 'FlashAttention 的核心收益是什么?',
      options: ['把 attention 改成了线性注意力', 'attention 的 FLOPs 从 $O(N^2)$ 降到 $O(N)$', '不物化 $N \\times N$ 矩阵: 显存 $O(N)$, 并大幅减少 HBM 读写', '用近似 softmax 换速度'],
      answer: 2,
      why: 'FLOPs 没少 (甚至略多), 结果也是精确的。快是因为每个 tile 只进 SRAM 一次, 省掉了对 $N \\times N$ 中间矩阵的反复读写。',
    },
    {
      q: 'online softmax 处理到一个新 block, 发现了更大的 max。之前累计的分母 $l$ 要怎么处理?',
      options: ['乘以 $\\exp(m_{\\text{old}} - m_{\\text{new}})$ 再加上新 block 的和', '乘以 $\\exp(m_{\\text{new}} - m_{\\text{old}})$ 再加上新 block 的和', '不用动: 分母只在最后除一次, max 变了不影响', '清零, 用新的 max 把之前所有 block 重算一遍'],
      answer: 0,
      why: '旧的和是以 $m_{\\text{old}}$ 为基准算的 $\\sum \\exp(s - m_{\\text{old}})$; 换基准到 $m_{\\text{new}}$ 只需整体乘 $\\exp(m_{\\text{old}} - m_{\\text{new}})$。这一行让分块结果与整块 softmax 逐位一致。',
    },
    {
      q: 'CUDA Graph 在什么情况下收益最大?',
      options: ['动态形状请求: 每步形状不同, 图能自动适配', '大 batch prefill: kernel 最多, 省下的 launch 最多', '小 batch decode: kernel 很短, launch 占比高', '长上下文 decode: 捕获后 attention kernel 被融合加速'],
      answer: 2,
      why: 'graph 省的是 CPU 逐个 launch kernel 的开销。\n- kernel 越短: 这部分占比越高。\n- 大 prefill: GPU 计算远大于 launch 开销, 几乎没收益。\n(本仓库的数字是注入开销的模型, 不是实测。)',
    },
    {
      q: 'B=1、H=32 的长对话 decode, FlashAttention 在 108 个 SM 的 A100 上只用了约 30% 的 SM。根本原因是?',
      options: ['decode 只有 1 行 Q, 并行单位只剩 $B \\times H = 32$', 'online softmax 的 max 要全局同步, 各 SM 轮流算', 'kernel launch 太慢, GPU 大半时间在等 CPU 提交', 'KV 长达 32k, 单个 SM 的 SRAM 装不下整段 K/V'],
      answer: 0,
      why: '- prefill: Q 有几千行, 按 Q 块切有的是并行度。\n- decode: 每条序列只有一行 Q。\nFlash-Decoding 改沿 KV 长度切 $S$ 段, 并行单位变成 $B \\cdot H \\cdot S$。',
    },
    {
      q: 'Flash-Decoding 把 KV 切成 $S$ 段, 各段得到 $(O_s, \\mathrm{lse}_s)$。最后怎么合并?',
      options: ['按各段 $\\exp(\\mathrm{lse}_s)$ 加权求和, 最后再除以段数 $S$', '各段 $O_s$ 直接取算术平均: 每段都已经各自归一化过了', '按 $\\exp(\\mathrm{lse}_s - \\mathrm{lse})$ 加权求和, lse 为全局值', '按段长 $T_s / T$ 加权求和: 段越长, 占的权重越大'],
      answer: 2,
      why: '每段的 softmax 分母不同, $\\exp(\\mathrm{lse}_s - \\mathrm{lse})$ 就是第 $s$ 段占全局 softmax 的质量。和普通 attention 数学相等, 13 组配置最坏误差 2.06e-07。',
    },
    {
      q: 'B=2、8 个 KV head、108 个 SM。按本章的代价模型, split-K 的加速比上限约是?',
      options: ['6.75×', '13.5×', '108×', '2×'],
      answer: 0,
      why: '上限 $N_{\\text{SM}} / (B \\cdot H) = 108 / 16 = 6.75$: 最多把闲着的 SM 全用上。并行单位越少, split-K 越有用。B=64、H=32 时 SM 早就满了, 加速 1.00×。',
    },
  ],
  'infer-engine': [
    {
      q: '一条请求的 prompt 完全命中了 prefix cache。还需要做前向吗?',
      options: ['需要: 整段 prompt 重算, 命中的 KV 只用于校验', '需要: 只重算第一个 block, 用来对齐位置编码', '不需要: cache 里连末位的 logits 也存了, 直接采样', '需要: 至少真算最后 1 个 token 才有 logits'],
      answer: 3,
      why: 'cache 里存的是 KV, 不是 logits。match_prefix 故意只匹配 (len−1)//bs 个 block, 保证至少留 1 个 token 真算。',
    },
    {
      q: '引擎用 recompute 式抢占: 被抢占的序列已经生成的 token 怎么办?',
      options: ['全部丢弃, 回来时从原 prompt 重新生成', '保留文本、归还 block, 回来时连已生成部分一起重算 KV', '把 KV 换出到 CPU 内存, 回来时再搬回 GPU', '继续占着 block 挂起, 等别的序列结束再恢复'],
      answer: 1,
      why: 'output_ids 原样保留, num_computed 归 0 回 waiting 队首。重算的是 KV 不是文本, 所以 greedy 输出仍与朴素生成逐 token 相同 (demo: 9 个 block, 4 次抢占)。',
    },
    {
      q: 'demo 第一轮共 305 个待算 token, 前缀命中 72 个。runner.tokens_computed 应该是多少?',
      options: ['233', '72', '305', '377'],
      answer: 0,
      why: '305 = 233 个真前向 + 72 个命中跳过。引擎把这两个数分开统计并要求对得上账: "命中"必须是真的没算, 而不是算了再扔。',
    },
  ],
  'infer-prefix-radix': [
    {
      q: 'block_size=16, 两条请求的前 100 个 token 相同。链式 block hash 能复用多少个 token 的 KV?',
      options: ['96', '112', '0, 因为后面不同', '100'],
      answer: 0,
      why: '只能命中完整 block: $\\lfloor 100/16 \\rfloor \\cdot 16 = 96$。radix tree 以 token 为粒度, 能命中全部 100 个, 代价是实现更复杂 (边分裂)。',
    },
    {
      q: '为什么每个 block 的 hash 要把"父 block 的 hash"也算进去?',
      options: ['让 block 按父 hash 排序, 查表降到 $O(\\log n)$', '防哈希冲突: 只 hash 本块 16 个 token 太容易撞', 'KV 依赖整个前缀: 同段 token 接在不同前缀后 KV 不同', '让 LRU 能沿链从尾部开始驱逐, 不会先删中间块'],
      answer: 2,
      why: 'attention 让位置 i 的 KV 取决于整个前缀。链式 hash 使一个 hash 唯一标识"到此为止的整段前缀", 不会把别的上下文里的同名 block 误当命中。',
    },
    {
      q: 'radix cache 满了, 为什么只能从叶子开始驱逐?',
      options: ['中间节点被后代依赖, 扔掉它整棵子树都作废', '中间节点的 KV 已在后代里各存一份, 删了也能恢复', '驱逐叶子不用改父节点的 slots, 实现最简单', '叶子占的 KV 最多, 驱逐一次腾出的显存最大'],
      answer: 0,
      why: '前缀是链式依赖: 命中必须从根连续走到某处。驱逐中间节点等于把整棵子树作废, 所以按 LRU 挑 ref_count=0 的叶子。',
    },
  ],
  'infer-structured-output': [
    {
      q: '为什么不在每步解码时对 V 个 token 逐字符试走 FSM?',
      options: ['在线试走结果不对: 多字符 token 跨状态, 只能离线判定', '每步 $O(V \\cdot \\mathrm{len})$ 的 CPU 开销, 卡在前向和采样之间', 'FSM 状态会随生成无限增长, 在线没法枚举', 'mask 只能在 GPU 上算, CPU 试走的结果传不过去'],
      answer: 1,
      why: '词表十几万, 每步都试走不可接受。FSM 状态有限, 离线对每个 (状态, token) 试走一次存成表, 在线只查一行。',
    },
    {
      q: 'token \'":\' 在"正在写 key"的状态下合法吗?',
      options: ['只有前半合法: 引号结束 key, 冒号得等下一步', '取决于 logits: 模型给它的概率够高才合法', '不合法: 一个 token 不能跨越两个语法状态', '合法: 逐字符都走得通, 落点是"等待 value"'],
      answer: 3,
      why: '多字符 token 可以一口气跨几个状态: \'"\' 结束 key, \':\' 进入 value。判定标准只有一条: 整段字符全部走得通。',
    },
    {
      q: '只靠 prompt 写"请输出 JSON"为什么不够?',
      options: ['模型 SFT 时没见过 JSON, prompt 再清楚也写不对', 'prompt 只提高概率; 在 logits 上置 −inf 才严格为 0', '温度高才会乱, 设 T=0 再配合 prompt 就一定合法', 'JSON 太长会被截断, 在 prompt 里限定长度即可'],
      answer: 1,
      why: '采样总有概率落到非法 token。mask 让非法 token 概率严格为 0, 再配合 need 表保证在长度上限内能收尾, 输出一定可解析。',
    },
  ],
  'infer-quant-awq': [
    {
      q: 'AWQ 凭什么说"激活大的通道上的权重更重要"?',
      options: ['激活大说明权重离群, 按 $\\|W - \\hat W\\|$ 算误差最大', '这些通道的权重数值更大, 舍入的绝对误差也更大', '误差 $= \\sum x_i \\cdot \\Delta W_i$: 大激活把舍入误差放大', '大激活通道的梯度更大, 训练时权重学得更充分'],
      answer: 2,
      why: '量化要最小化的是 $\\|XW - X\\hat W\\|$ 而不是 $\\|W - \\hat W\\|$。demo: INT4 g32 的 RTN 误差 0.0759, AWQ 降到 0.0273。',
    },
    {
      q: 'AWQ 的缩放指数 $\\alpha$ 为什么不是越大越好 (demo 最优 $\\alpha=0.4$)?',
      options: ['放大的行撑大 group 的 range, 同组其它权重格点变粗', '$\\alpha$ 越大 $s$ 越大, 推理时除以 $s$ 的开销越大', '放大后该行的相对误差反而变大, 格点也跟着放大', '$\\alpha$ 大了激活被除得太小, fp16 下溢成 0'],
      answer: 0,
      why: '放大 $s$ 倍让该行相对误差缩小约 $s$ 倍, 但 group 共用一组 scale: range 变大 → 其它行误差上升。所以要在校准集上网格搜索。',
    },
    {
      q: 'KIVI 为什么对 K 按通道分组、对 V 按 token 分组?',
      options: ['K 按通道分组的组数更少, scale/zero 开销更小', 'K 要做 RoPE, 按通道分组才能让旋转后 scale 不变', 'K 有固定离群通道, 按 token 分组会撑大每行 scale', 'V 按 token 分组可以和 K 共用 scale, 省一半元数据'],
      answer: 2,
      why: 'per-channel 把离群通道关在自己的组里 (demo INT4: 误差 0.026 vs per-token 0.18)。V 没有固定离群通道, 按 token 分组还能在每个新 token 写入时独立量化。',
    },
    {
      q: '若校准激活的 $H = X^\\top X / N$ 恰好是对角矩阵, GPTQ 的结果和 RTN 是什么关系?',
      options: ['完全相同: 补偿项全为 0, 误差不摊给后面的行', '比 RTN 好: 对角元仍按激活幅度给各行加权', '无法计算: 对角的 $H$ 做不了 Cholesky 分解', '比 RTN 差: 没有相关性可用, 补偿项只剩噪声'],
      answer: 0,
      why: '$H^{-1}$ 也对角, 它的 Cholesky 因子非对角元为 0。GPTQ 的收益全部来自激活通道间的相关性。iid 激活、校准只有 256 条时, held-out 误差反而是 RTN 的 1.56×。',
    },
    {
      q: 'SmoothQuant 把 $\\alpha$ 从 0.5 调到 1。W8A8 误差 (不平滑时 0.0278) 会怎样?',
      options: ['回到 0.0278: $\\alpha=1$ 等价于完全不平滑', '继续降到 0.0035 左右: 激活全平了, 只剩舍入误差', '从 0.0070 涨到 0.0297: 离群值全压给了权重', '持平在 0.0070: $\\alpha$ 只在激活和权重间搬 scale'],
      answer: 2,
      why: '$s_j = \\max|X_j|^\\alpha / \\max|W_j|^{1-\\alpha}$。$\\alpha=1$ 时激活平了, 权重却吃下全部离群幅度, 只量化 W 的误差到 0.0288。最优在两者之间 ($\\alpha=0.5$, 降 4.0×)。',
    },
    {
      q: '把 per-tensor scale 换成 per-channel, 哪种格式受益最大?',
      options: ['INT8', 'FP8 E5M2', '都一样', 'FP8 E4M3'],
      answer: 0,
      why: 'INT8 的格点等间距, 离不开细粒度 scale (误差 0.0543 → 0.0070, 7.71×)。FP8 的格距随数值伸缩, per-tensor 已经够用 (E4M3 只改善 1.02×)。',
    },
  ],
  'infer-kv-footprint': [
    {
      q: 'LLaMA-3-8B (32 层, 8 个 KV 头, d_head=128, fp16) 每个 token 的 KV 是多少?',
      options: ['512 KiB', '2 MiB', '16 KiB', '128 KiB'],
      answer: 3,
      why: '$2 \\cdot 8 \\cdot 128 \\cdot 32 \\cdot 2$ 字节 = 131072 B = 128 KiB。同形状的 MHA (32 个 KV 头) 是 512 KiB, 差的正好是 $n_{\\text{head}}/n_{\\text{kv}} = 4$ 倍。',
    },
    {
      q: 'MLA 每 token 的 cache 公式里为什么没有"2·"?',
      options: ['它只缓存 K, V 在 attention 时由 K 线性变换得到', 'K 和 V 共用一个 latent, 用时再各自上投影', '它把 K/V 存成 int8, 字节减半正好抵掉那个 2', '相邻两层共享一份 cache, 层数这一项折半了'],
      answer: 1,
      why: 'cache 里只有 $(d_c + d_{\\text{rope}})$ 维: latent 同时是 K 和 V 的来源。DeepSeek-V3: $(512+64) \\cdot 61 \\cdot 2$ = 68.6 KiB, 比同尺寸 MHA 省 56.9×。',
    },
    {
      q: '40 GiB 的 KV 预算、上下文 8192: MHA 的 7B 只能同时服务 10 条, 换成 GQA-8 呢?',
      options: ['40 条', '80 条', '10 条, 并发由算力决定', '20 条'],
      answer: 0,
      why: '并发上限 = KV 预算 ÷ (每 token 字节 × 上下文长度)。每 token 字节降 4 倍, 并发就是 4 倍。结构选择在推理侧直接变成吞吐。',
    },
  ],
  'infer-attention-sinks': [
    {
      q: '纯滑动窗口把最开头几个 token 逐出 cache 后, 模型为什么会崩?',
      options: ['多余注意力被学着倒在开头 token 上; 它们没了其余权重被迫膨胀', '窗口放不下完整句子, 后面的 token 看不到主语', '删掉第 0 页后页表整体错位, 读到的 KV 全乱了', '开头通常是 system prompt, 语义最重要, 丢了就答非所问'],
      answer: 0,
      why: 'sink token 收走的是"无处安放"的注意力质量, 和语义无关。逐出它们等于突然改掉 softmax 的分母, 这是训练时从未见过的分布。',
    },
    {
      q: 'SinkCache 为什么存未旋转 (pre-RoPE) 的 K?',
      options: ['V 不做 RoPE, K 也存未旋转的才能和 V 共用页表', 'pre-RoPE 的 K 数值范围更小, 存 fp16 更省显存', '位置按槽位重编号、每步现转; 存转好的就改不了位置', 'RoPE 只作用在 query 上, key 本来就不该旋转'],
      answer: 2,
      why: '逐出中间 token 后, 留下的 token 槽位会前移。用槽位 $0, \\dots, L-1$ 当位置, 才能保证永远不超过训练长度。\n代价是每步 $O(L \\cdot D)$ 的重转。',
    },
    {
      q: '用了 attention sinks, 模型就能"记住"无限长的上下文了吗?',
      options: ['不能: 窗口外的内容彻底丢了, sink 只保证不崩', '能: 位置重编号后 RoPE 可以外推到任意长度', '能, 前提是 window 至少覆盖模型的训练长度', '能: sink token 会把窗口外的内容压缩成摘要保留'],
      answer: 0,
      why: 'StreamingLLM 解决的是"无限输入下稳定生成 + 显存有界", 不是长程记忆。要记得就得保留 KV (offload) 或者检索。',
    },
  ],
  'infer-tree-speculation': [
    {
      q: '同样让 target 一次验证 16 个 token, 树 [3,2,1] 为什么比 K=15 的链接受得多 (demo 每轮接受 1.80 vs 1.40)?',
      options: ['树的 draft 吃了 target 的 hidden state, 猜得更准', '链首个猜错后面全废; 树在易错的浅层留了备选', '树按 $\\min(1, p/q)$ 放宽了接受条件, 更易通过', '树里兄弟节点互相可见, 可以彼此纠正猜错的 token'],
      answer: 1,
      why: '接受长度由"第一处不一致"决定。把验证预算花在浅层的多个候选上, 比花在一条很深但早早断掉的链上划算。',
    },
    {
      q: 'tree attention mask 里, 一个节点能看到哪些位置?',
      options: ['只有它的父节点', '整棵树', '上下文 + 所有编号比它小的节点', '上下文 + 它的祖先 + 它自己'],
      answer: 3,
      why: '每个节点看到的恰好是"根到自己"这条链, 等价于把每条路径单独顺序 decode 一次。兄弟互不可见, 且同深度共享同一个 RoPE 位置。',
    },
    {
      q: 'EAGLE 的 draft 和 m07 的独立小模型 draft, 关键区别是什么?',
      options: ['EAGLE 把 draft 蒸馏得比 m07 的小模型更大、更准', 'EAGLE 吃 target 的 hidden state, 共享它的 lm_head', 'EAGLE 的 draft token 不用 target 验证, 直接接受', 'EAGLE 用 target 自己的前几层当 draft, 提前退出'],
      answer: 1,
      why: '验证那次 forward 白送了下一轮需要的特征。验证循环和接受规则完全复用 m07, 只换 drafter。\n误差随 draft 步数累积, 所以越靠后的槽位越难接受。',
    },
  ],
  'infer-kv-offload': [
    {
      q: 'demo 里从磁盘层命中 1 个 block: 加载 2.70 ms, 重算 2.00 ms。该怎么办?',
      options: ['加载: 命中块会被提升回 GPU, 下一轮就快了', '加载: 搬比重算便宜 25 倍, 命中了就该用', '重算: 命中块数不到 $n^* \\approx 1.54$, 加载不划算', '重算: 磁盘层只存 hash 不存 KV 本体, 没法加载'],
      answer: 2,
      why: '$\\text{load} = \\text{latency} + n \\cdot \\text{每块搬运}$, $\\text{recompute} = n \\cdot \\text{每块重算}$。$n^* = \\text{latency} / (\\text{每块重算} - \\text{每块搬运})$。每层都要做这个判断。',
    },
    {
      q: '加了一层 0.5 GB/s 的远端缓存, "命中就加载"的 TTFT 从 34.8 ms 变成 68.9 ms。原因是?',
      options: ['这条链路每块加载 4.19 ms, 比重算 2.00 ms 还慢', '每次查找都要先问远端, 查询延迟叠进了 TTFT', '远端层挤掉了 CPU 层的位置, CPU 命中变少了', '远端容量太小, 频繁驱逐, 整体命中率下降了'],
      answer: 0,
      why: '缓存层不是越多越好: 链路比重算慢时, 命中是负收益。策略改成"加载/重算取小"后回到 34.8 ms。',
    },
    {
      q: '沿 hash 链查找时, 第 3 块在所有层都 miss, 但第 4、5 块还在 CPU 层。m20 的 lookup 里, 第 4、5 块的 KV 还能直接用吗?',
      options: ['取决于 block 大小: 块够大时单块也能独立使用', '能: 第 4、5 块命中, 第 3 块留空不影响 attention', '不能: 在第一个 miss 处停, 第 3 块起全部当 miss 重算', '能: 用第 4 块自己的 token 重新算 hash 再查即可'],
      answer: 2,
      why: 'lookup 在第一个全层 miss 处停, 这是设计选择, 和前缀缓存一致。\n- 严格说第 4、5 块的 KV 仍然有效: 它们的前缀 token 没变。\n- 但要先重算第 3 块才接得上。实现上直接从第一个 miss 起全部重算最简单。',
    },
  ],
  'infer-moe-serving': [
    {
      q: '专家并行下, 评价负载均衡为什么看 max/mean 而不是方差或平均值?',
      options: ['all-to-all 按最忙 rank 分配带宽, 其余链路被限速', 'max 能从 rank_load 直接读出, 方差要多一遍统计', 'max 决定显存上限: 最忙 rank 的激活要放得下', 'combine 要等齐所有 rank, 一步耗时由最忙的卡决定'],
      answer: 3,
      why: 'max/mean = 2.95 意味着平均 rank 利用率只有 $1/2.95 \\approx 34\\%$, 其余时间都在等最慢的那张卡。',
    },
    {
      q: '最热专家的负载是平均 rank 负载的 1.42 倍。只重新摆放专家 (不复制) 能做到均衡吗?',
      options: ['能: 把它单独放一张卡, 其余专家摊给另外 3 张', '不能: 它一个就超过单卡份额, 必须复制并拆分它', '能: 按负载从大到小贪心摆放, 就能压到 1.0×', '不能: top-2 让每个 token 占两个专家, 没法拆'],
      answer: 1,
      why: '单个专家不可分 → 它所在的 rank 至少 1.42× 均值。demo: 贪心无副本 1.48×, EPLB 给专家 0 加 3 个副本、专家 1 加 1 个后 1.02×。',
    },
    {
      q: '给热专家加冗余副本后, 模型输出会变吗?',
      options: ['会: 路由改成优先选有空闲副本的那个专家', '只在 top-1 时不变, top-2 的两个专家加权会变', '会: 副本分走 token 后, gate 要按副本数重新归一化', '不会: 副本权重相同, 只改 token 被送去哪个 slot'],
      answer: 3,
      why: 'EPLB 只改"逻辑专家 → 物理 slot"的映射, 路由和 gate 都不变; demo 断言输出与朴素实现 $\\max|\\Delta| = 0$。',
    },
  ],
  'infer-sparse-decode': [
    {
      q: 'Quest 给 block 打分用的是 $q \\cdot k$ 的"上界"。为什么要上界而不是均值这类估计?',
      options: ['上界不会低估 block 内任何 $q \\cdot k$, 含针的块不会漏', '均值会被 softmax 归一化抵消, 各块分数都一样', '上界只需存 kmax 一份摘要, 比均值省一半显存', '上界比均值更紧, 分数更接近真实注意力质量'],
      answer: 0,
      why: '逐维取 $\\max(q_i \\cdot \\mathrm{kmin}_i,\\ q_i \\cdot \\mathrm{kmax}_i)$ 再求和。代价是偏松 (demo 平均 6.7×), 但"不漏针"比"估得准"重要。',
    },
    {
      q: 'needle 负载上 k 从 4 加到 8 (读 1.6% → 3.1% 的 KV), 误差从 0.83 断崖降到 0.023。说明什么?',
      options: ['Quest 上界在 k≥8 时变紧, 打分才开始准确', '误差与读取比例成反比: 多读一倍, 误差减半', '注意力集中在少数 block, 全进 top-k 后其余 KV 几乎无关', 'k=8 起强制保留的 sink 块才被选中, 分母恢复正常'],
      answer: 2,
      why: '稀疏 decode 的前提是注意力高度集中。同样读 3.1%, 随机选块误差是 1.06。',
    },
    {
      q: '在随机权重的 TinyLM 上, Quest 选块并不比随机选块好。为什么?',
      options: ['随机权重下注意力近乎均匀, 没有少数重要 token', '随机权重的注意力太尖, 集中在单个 token 上', 'Quest 只对训练见过的长度有效, 这里超出了', '实现有 bug: 选块时 forced 块被重复计入了 k'],
      answer: 0,
      why: '方法的收益来自数据分布而不是算法本身。训练过的 LLM 注意力很尖, 才有 Quest / NSA / DSA 的空间。demo 把这个反例也打印了出来。',
    },
  ],
  'infer-test-time-compute': [
    {
      q: '单步正确率 0.75、链长 4。单条正确率约是多少? 完美判分器下 best-of-N 要到 95%, N 至少多少?',
      options: ['0.32 / 4', '0.32 / 8', '0.75 / 2', '0.56 / 5'],
      answer: 1,
      why: '$0.75^4 \\approx 0.316$。完美判分器下 best-of-N = pass@N = $1-(1-q)^N$: $1 - 0.684^8 \\approx 0.952$, 而 $N=7$ 只有 0.930。',
    },
    {
      q: '在"模型最想写的答案就是错的"那类题上, 哪种方法随 N 增大反而变差?',
      options: ['PRM beam search', 'pass@N', 'best-of-N + ORM', '多数投票'],
      answer: 3,
      why: '陷阱题上最终答案分布的众数是那个误解。N 越大, 样本众数越稳定地等于它: 0.254 (N=16) → 0.108 (N=64)。ORM / PRM 是外部判分器, 能从少数派里挑出正确答案。',
    },
    {
      q: '同样 28 个 token, PRM beam (0.865) 为什么比 best-of-7 (0.747) 好?',
      options: ['PRM 判分没有噪声, 而 ORM 带 $\\mathcal{N}(0, 0.5^2)$', '错步在出现的那一步就被剪掉, 不再为它付费', 'beam 采样用了更低的温度, 单条正确率更高', '同样 28 token, PRM beam 采到了更多条完整链'],
      answer: 1,
      why: 'best-of-N 要把每条错链的全部 $K$ 步写完, 到终点才判。PRM beam 省下的 token 花在别的候选上。大预算时 (52 token) 两者打平 (0.830 vs 0.849), 输在判分噪声。',
    },
    {
      q: 'budget forcing 把预算 B 从 4 加到 8, 正确率从 0.240 变成多少? 为什么?',
      options: ['0.055: 预算一长, 自查反而容易把对的改错', '0.305: 和不干预时一样, 模型想停就提前停了', '0.460: 多出的 4 个 token 够再查一轮、纠一次错', '仍是 0.240: 重写要写到结尾, 8 个 token 不够'],
      answer: 3,
      why: '4 个 token 刚好写完第一稿, 之后每查一步、每重写一步都各花 1 个 token。重写要从出错那一步写到结尾, B=8 不够; B=16 才涨到 0.460。',
    },
  ],
  'infer-multi-replica-routing': [
    {
      q: '纯前缀感知路由下, 8 个副本为什么最多只有 4 个在干活?',
      options: ['平手轮转失效, 请求总落到编号最小的那几个副本', '8 个副本共享一棵 radix tree, 只长出 4 棵子树', '只有 4 个 system prompt, 新对话总选已缓存它的副本', 'LRU 把冷门 system prompt 驱逐了, 其余副本全 miss'],
      answer: 2,
      why: '新对话在已缓存其 system prompt 的副本上命中 512 token, 在其它副本命中 0。每个 system prompt 只在第一次出现时选过一次副本, 两个还可能落到同一个副本。本例只有 3/8 在干活。',
    },
    {
      q: '纯前缀感知的命中率最高 (93.9%), TTFT 却最差。原因是?',
      options: ['热点副本排队: 命中省下的 prefill 抵不过积压', 'hash 冲突导致误命中, 拿到错 KV 后又得重算', '命中率太高, 热前缀反复被驱逐又重新载入', '命中的 KV 要跨副本传输, 读取比重算还慢'],
      answer: 0,
      why: '负载 max/mean 4.18, 热点副本利用率超过 1, 队列模型发散 (TTFT 105.954 s)。加 1s 阈值后命中 83.6%、max/mean 1.19、TTFT 0.314 s, 比最少负载低 33%。',
    },
    {
      q: '负载阈值从 1s 收紧到 0.05s, 会发生什么?',
      options: ['命中率基本不变, TTFT 降到最优的 0.284 s', '命中率和 TTFT 同时改善, 负载也更均匀', '命中跌到 72.9%, TTFT 升到 0.308 s, 趋近最少负载', '热点重新堆积, max/mean 回到 4.18 左右'],
      answer: 2,
      why: '阈值太紧, 稍有积压就放弃缓存。本例 TTFT 最优在 0.5s (0.284 s)。阈值是命中率和均衡之间的旋钮, 最优值取决于负载。',
    },
    {
      q: '路由器探测 8 个副本的前缀长度时, 为什么不能直接调 RadixCache.match?',
      options: ['match 会 split 树、刷新 LRU, 扰动没被选中的副本', 'match 会给路径 ref_count 加 1, 节点被锁住没法驱逐', 'match 要加全局锁, 8 个副本只能串行探测', 'match 只返回整块命中, 探测不到 token 级前缀'],
      answer: 0,
      why: '探测不等于使用。只读的 prefix_len 走法相同, 但不 split、不刷新 last_used。只有被选中的副本才调 match。',
    },
  ],
}
