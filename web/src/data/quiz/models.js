// 阶段 2 · llm_models 自测题 (含手写总览页 models / attention / position / blocks / moe / diffusion)
export default {
  models: [
    {
      q: '阶段 2 说 “几十种主流模型 = 少数零件的不同组合”。这些零件槽位指的是?',
      options: ['优化器、学习率、batch size、精度', 'attention、FFN、norm、位置编码', 'tokenizer、数据集、loss、采样器', 'GPU、通信、并行策略、checkpoint'],
      answer: 1,
      why: 'Pre-LN Block 的骨架几乎不变; LLaMA、Mistral、DeepSeek 的差别都落在这四个槽位各选了哪个实现。',
    },
    {
      q: '为什么阶段 2 换成 PyTorch, 而不再像阶段 1 那样手写反向?',
      options: ['numpy 没有批量矩阵乘和 einsum, 写不出多头注意力', '原理阶段 1 已手写过; 本阶段比结构, autograd 让换零件只改前向', 'autograd 的梯度比手写反向更精确, 能消除累积的数值误差', '链式法则只能手推单层网络, 多层 Transformer 推不出来'],
      answer: 1,
      why: '要比较十几种架构, 每种都手推反向成本太高且与主题无关。阶段 1 已经手写过一遍 autograd 做的事。',
    },
    {
      q: '所有 decoder 语言模型都混入同一个 GenerationMixin, 它对宿主模型的要求是?',
      options: ['每层暴露 k_proj / v_proj, 由 generate() 统一算出并拼接各层 K/V', '用 RoPE, 且 forward 接受 position_ids, 以便 decode 时传入当前位置', '有 layers、max_len, forward(idx, cache=None) 接受 cache', '实现 prefill() 和 decode_step() 两个方法分别处理两阶段'],
      answer: 2,
      why: 'generate() 只管 “喂什么 token、怎么采样”; cache 里放 K/V、MLA latent 还是 Mamba 状态, 由各层自己决定, 所以 Mamba 也能复用它。',
    },
  ],
  attention: [
    {
      q: 'MHA → GQA 主要省的是什么?',
      options: ['Q 投影参数: 同组的 Q 头共用一个 Q 投影矩阵', '分数矩阵的 $O(T^2)$ 计算: 共享 K 后只需算 kv 头数那么多份', 'KV cache: 几个 Q 头共用一组 K/V, cache 按比例缩小', 'KV cache: K/V 先压成低秩 latent 再存, 用时升维'],
      answer: 2,
      why: '推理瓶颈是每步都要读全部 KV cache (访存受限)。GQA 几乎不掉点就把它缩小 4–8 倍。分数矩阵的计算量并没有变。',
    },
    {
      q: 'MLA 的 KV cache 里存的是什么?',
      options: ['低秩 latent c_kv 和所有头共享的 post-RoPE k_rope', '每个 KV 组一份完整 K/V, 组数少于 Q 头数', '只存低秩 latent c_kv, RoPE 在升维出 K 后现场施加', '每个头完整的 K 和 V, 只是量化成 FP8 存'],
      answer: 0,
      why: 'K-nope 和 V 每步从 c_kv 现场升维; 教学配置下每 token 每层 MHA 1024 / GQA 256 / MLA 96 个数 (9.4%)。',
    },
    {
      q: 'MLA 为什么要把 Q/K 拆成 nope 和 rope 两段, 只旋转 rope 段?',
      options: ['只旋转一小段, 长上下文时 RoPE 的 sin/cos 计算量少很多', '旋转会改变 c_kv 的秩, 整段旋转后 latent 就不再低秩', '让 Q 和 K 维度可以不同, Q 侧从而能用更低的秩压缩', 'RoPE 随位置变, 旋转升维后的 K 会让 $W_{UK}$ 无法预先吸收'],
      answer: 3,
      why: '带位置的旋转矩阵夹在 $W_{UK}$ 和 $Q$ 之间, 无法预先合并。所以单独留一小段带 RoPE 的共享 key 承担位置信息。',
    },
  ],
  position: [
    {
      q: 'RoPE 为什么说编码的是 “相对位置”?',
      options: ['它按相对距离查表得到一个偏置, 加在注意力分数上', '$q$、$k$ 按各自位置旋转后, 点积只依赖 $m - n$', '它只旋转 $k$ 不旋转 $q$, 旋转角取位置差 $m - n$', '每对维度的旋转角是按相对距离学出来的参数'],
      answer: 1,
      why: '$\\langle R_m q, R_n k \\rangle = \\langle q, R_{n-m} k \\rangle$: 旋转矩阵的转置相乘只剩角度差。RoPE 没有任何可学参数。',
    },
    {
      q: 'RoPE 的不同维度对用不同频率 $\\theta_i = \\mathrm{base}^{-2i/d}$, 高频和低频各管什么?',
      options: ['高频 (波长短) 区分远距离, 低频 (波长长) 分辨相邻先后', '高频管训练长度以内, 低频只在超出训练长度后才起作用', '高频 (波长短) 分辨相邻先后, 低频 (波长长) 区分远距离', '高频只旋转 query, 低频只旋转 key, 合起来得到相对位置'],
      answer: 2,
      why: '像时钟的秒针和时针:\n- 秒针: 分辨得了 1 秒, 但 60 秒后就重复。\n- 时针: 慢, 却能区分几个小时。\n长度外推出问题的正是 “时针” 那一端。',
    },
    {
      q: 'M-RoPE (Qwen2-VL) 处理纯文本 token 时会怎样?',
      options: ['三个轴的位置 id 相同, 严格退化为普通 1-D RoPE', '只有 T 轴那段频率旋转, H/W 两段维度保持不转', '三轴 id 取 (t, 0, 0), 结果接近 1-D RoPE 但不相等', '没有图像时退回 sinusoidal, 加在 embedding 上'],
      answer: 0,
      why: 'head_dim 的频率轴被切成 T/H/W 三段, 每段用各自的位置 id; 文本三轴 id 相同, 结果与 1-D RoPE 逐元素相等。',
    },
  ],
  blocks: [
    {
      q: 'Pre-LN (norm 放在子层之前) 相比原始 Transformer 的 Post-LN, 好处是?',
      options: ['每层省掉一个 norm 的 $\\gamma$、$\\beta$, 参数更少', '主干每层都被归一化, 所以激活幅度不随深度增长', '主干上没有 norm, 梯度直通, 深层无需精细 warmup', '子层输入都已归一化, 所以输出端不再需要 final norm'],
      answer: 2,
      why: 'Post-LN 每层都对 “残差 + 子层” 整体归一化, 梯度每过一层被缩放一次; Pre-LN 的主干是纯加法。',
    },
    {
      q: 'SwiGLU 有三个线性层, 为什么参数量仍与两层 GELU FFN 大致持平?',
      options: ['其中一个线性层不含参数', '中间维度从 $4d$ 缩到约 $8d/3$', '三个线性层共享权重', '它去掉了 bias 所以持平'],
      answer: 1,
      why: '$3 \\times d \\times (8d/3) = 8d^2 = 2 \\times d \\times 4d$。门控 (一路做开关、一路做内容) 在同等参数下效果更好。',
    },
    {
      q: 'RMSNorm 相比 LayerNorm 去掉了什么?',
      options: ['可学增益 $\\gamma$: 只除以均方根, 不再乘缩放', '除以标准差: 只减均值, 不再做缩放', '对 batch 维的统计: 改成在单个样本内归一化', '减均值 (及偏置 $\\beta$), 只按均方根缩放'],
      answer: 3,
      why: '实验表明重新居中贡献很小, 起作用的是重新缩放; 去掉后更省算、数值上也更简单。',
    },
  ],
  moe: [
    {
      q: 'MoE 的 “总参数量” 和 “激活参数量” 分别决定什么?',
      options: ['总参数决定显存占用, 激活参数决定每 token 计算量', '总参数决定每个 token 计算量, 激活参数决定显存占用', '总参数只决定训练显存, 推理时只需加载激活参数', '两者只差路由器参数, 对显存和计算的影响相同'],
      answer: 0,
      why: '每个 token 只过 top-k 个专家, FLOPs 按激活参数算; 但所有专家都得放在显存里: MoE 用显存换算力。',
    },
    {
      q: 'DeepSeek MoE 的共享专家 (shared experts) 为什么存在?',
      options: ['兜底被路由丢弃的 token, 容量溢出时不丢信息', '替代负载均衡 loss, 路由器不再需要辅助约束', '通用知识放进总激活的共享专家, 路由专家才能分化', '共享专家的权重被各路由专家复用, 总参数更少'],
      answer: 2,
      why: '没有共享专家时, 每个路由专家都得重复学一份通用能力, 浪费容量。细粒度专家 + 共享专家是 DeepSeek 路线的两个要点。',
    },
    {
      q: '不做任何负载均衡, MoE 训练会发生什么?',
      options: ['路由 softmax 会自然把 token 摊平, 负载趋于均匀', '被选多的专家学得更好 → 更易被选, 滚雪球成路由坍缩', '各专家收到相近梯度, 学成一样, 退化成稠密 FFN', '只是训练变慢, 容量因子丢弃溢出 token 后效果不变'],
      answer: 1,
      why: '这是一个正反馈回路。坍缩后其余专家等于白占显存。专家并行下, 热点卡还会拖慢整个 step。',
    },
  ],
  'models-mtp': [
    {
      q: '滑动窗口注意力 (SWA) 每层只看最近 W 个 token, 远处的信息怎么到达当前位置?',
      options: ['到不了: 每层感受野都是 $W$, 堆多少层都一样', '靠 RoPE 的低频维度把远处位置的信息带过来', '跨层接力: 感受野约 $W^L$, 每层把窗口乘一次', '跨层接力: $L$ 层的理论感受野约 $L \\cdot W$'],
      answer: 3,
      why: '第 $l$ 层的位置 $i$ 看到的是第 $l-1$ 层的表示, 它们已经各自汇聚了前 $W$ 个位置的信息。',
    },
    {
      q: 'MTP (多 token 预测) 的第 k 级为什么不破坏因果性?',
      options: ['它用双向注意力, 但 loss 只计算在位置 i 之前的 token 上', 'MTP 头只在推理时当草稿, 训练时不参与, 谈不上因果', '位置 i 输入真实 $t_{i+k}$ 的 embedding, 预测 $t_{i+k+1}$', '位置 i 输入真实 $t_{i+k+1}$ 的 embedding, 回头预测 $t_{i+k}$'],
      answer: 2,
      why: '每深一级多 “看” 一个真实 token、多预测一步, 因果链完整。推理时这些额外的头可以直接当投机解码的草稿。',
    },
    {
      q: 'Gated DeltaNet 的状态是固定大小的 $D_h \\times D_h$ 矩阵。相比 KV cache, 它的根本代价是?',
      options: ['有损: 至多 $D_h$ 个正交 key 槽位, 写多了必然互相污染', '状态随上下文长度线性增长, 长序列时比 KV cache 还大', '$\\alpha$ 门只能整体遗忘, 无法覆写单个 key, 旧值会一直残留', '每步都要重读全部历史来更新状态, 解码是 $O(T)$'],
      answer: 0,
      why: 'delta rule 能精确覆写同一个 key, $\\alpha$ 门能整体遗忘, 但容量上限摆在那。所以 Qwen3-Next 每 4 层保留 1 层全注意力, 兜底精确召回。',
    },
  ],
  diffusion: [
    {
      q: 'DDPM 训练时, 网络的输入和预测目标分别是?',
      options: ['输入干净图 $x_0$ 和 $t$, 预测第 $t$ 步要加的噪声', '输入 $x_t$ 和 $t$, 预测加进去的噪声 $\\varepsilon$', '输入纯噪声, 迭代 $t$ 步得到 $x_0$ 后再算 loss', '输入 $x_t$ (不给 $t$), 预测它对应的时间步'],
      answer: 1,
      why: '随机抽 $t$、按闭式公式一步加噪到 $x_t$, 让网络回归 $\\varepsilon$。采样时从纯噪声出发, 一步步减去预测的噪声。',
    },
    {
      q: 'Flow Matching (rectified flow) 的插值路径 $x_t = (1-t) \\cdot x_0 + t \\cdot \\varepsilon$ 有什么好处?',
      options: ['路径是直线, 所以一步就能从噪声精确走到数据, 无需积分', '速度 $v = x_0 - \\varepsilon$ 与 $t$ 无关, 所以网络不需要输入 $t$', '路径是直线、速度 $v = \\varepsilon - x_0$ 恒定, 几步欧拉即可', '插值让 $x_t$ 方差恒为 1, 不再需要噪声日程表'],
      answer: 2,
      why: 'DDPM 的路径是弯的, 要很多小步才能跟住。直线路径让采样步数从上千降到几十甚至个位数, 所以 SD3 / Sora 一类模型都换了过来。',
    },
    {
      q: 'Flow Matching 的 $t \\in [0,1]$, 送进时间步 embedding 之前为什么要 ×1000?',
      options: ['时间 embedding 频率按整数步设计, $t \\in [0,1]$ 时几乎不变', '和 DDPM 的步数对齐, 才能直接加载 DDPM 预训练的时间 embedding', '$t$ 太小时 sin/cos 的输入会在 fp16 下溢成 0, 放大后才稳定', '插值系数 $(1-t)$、$t$ 放大 1000 倍, $x_t$ 的信噪比才够高'],
      answer: 0,
      why: '×1000 之后相似度降到约 0.17, 网络才分得清不同噪声强度。注意只缩放 embedding 的输入, 插值公式里的 $t$ 不动。',
    },
  ],
  'models-generation': [
    {
      q: '为什么 KV cache 只缓存 K 和 V, 不缓存 Q?',
      options: ['旧 token 的 Q 会随新 token 加入而变, 缓存了也会失效', '只有新 token 的 Q 要查历史 K/V, 旧 Q 再也用不到', 'Q 不经过 RoPE、与位置无关, 每步重算很便宜', 'Q 带 $1/\\sqrt{d}$ 缩放, 缓存后会有数值漂移, 只能重算'],
      answer: 1,
      why: '因果 mask 下旧位置的输出不受新 token 影响, 不需要重算, 它们的 Q 自然没用。而新 token 要和每一个旧 K 做点积。',
    },
    {
      q: 'prompt 6 个 token、生成 16 个, 无 cache 与有 cache 的 token 前向次数分别约为?',
      options: ['22 和 22', '216 和 21', '96 和 16', '256 和 16'],
      answer: 1,
      why: '无 cache 每步重算整个前缀: $6+7+\\dots+21 = 216$; 有 cache 只 prefill 6 个, 之后每步 1 个: $6+15 = 21$。差距随长度平方增长。',
    },
    {
      q: '序列长度超过 max_len、窗口必须左移时, generate() 为什么丢掉整个 cache 重新 prefill?',
      options: ['cache 按 max_len 预分配, 占满后只能整体释放再重建', '只需丢最左边一个 token 的 K/V, 但张量不能原地裁剪', '左移后因果 mask 的形状变了, 旧 K/V 与新 mask 对不上', '左移后绝对位置全变了, 旧 K 带的是旧位置的 RoPE 角度'],
      answer: 3,
      why: 'K 是 RoPE 之后才缓存的。滑窗层不受此限: 它按绝对位置继续往后编号, 只是把窗口外再也看不到的 K/V 丢掉。',
    },
  ],
  'models-qknorm-yarn': [
    {
      q: '已经除以 $\\sqrt{d}$ 了, 为什么注意力 logit 还会爆炸?',
      options: ['$\\sqrt{d}$ 只抵消维度, 不抵消训练中 q、k 范数的增长', '点积方差随 $d$ 线性增长, 除 $\\sqrt{d}$ 只抵消了一半', 'RoPE 每层旋转都会放大 q、k 的范数, 越深越大', 'softmax 在 fp16 下 $\\exp$ 溢出, 把 logit 推成 inf'],
      answer: 0,
      why: 'Python demo: 权重 ×10, 最大 logit 从 1.2 涨到 120; 开 QK-Norm 后稳定在 3.4 左右, 上界是 $g^2 \\cdot \\sqrt{d_{\\text{head}}}$。',
    },
    {
      q: 'QK-Norm 为什么必须放在 RoPE 之前?',
      options: ['RoPE 之后向量维度翻倍, norm 的增益 $g$ 形状对不上', 'RoPE 保范数; 先归一化再旋转, 范数和相对位置性质都保得住', 'RoPE 会放大范数, 只有先归一化, 旋转后才仍是单位长度', '顺序其实无所谓, 放前面只为和 QKV 投影融合成一个 kernel'],
      answer: 1,
      why: '若在 RoPE 之后再乘逐维增益 $g$, 会破坏成对维度的旋转结构, 点积不再只依赖 $m - n$。',
    },
    {
      q: '把上下文扩到 4 倍时, YaRN 对不同频率的处理是?',
      options: ['所有频率一律 ÷4 (线性位置插值), 再乘 mscale 补偿注意力熵', '只改 base (NTK-aware): 低频自动变慢, 高频基本不动, 无需 mscale', '转够 32 圈的高频不动, 不足 1 圈的低频 ÷4, 中间过渡; 乘 mscale', '低频原样保留, 转够 32 圈的高频 ÷4, 中间线性过渡; 另乘 mscale'],
      answer: 2,
      why: '一律 ÷4 是 PI, 会抹掉局部分辨率; 只改 base 是 NTK-aware, 中段压不够。d_head=64、L=2048 时 YaRN 有 9 个最高频维度完全不动, $\\mathrm{mscale} = 0.1 \\cdot \\ln 4 + 1 \\approx 1.139$。',
    },
  ],
  'models-mamba': [
    {
      q: 'Mamba 里 $\\Delta_t$ 很大意味着什么?',
      options: ['保留率 $\\to 1$、写入 $\\to 0$: 当前 token 等于没出现过', '保留率和写入都 $\\to 0$: 状态清空, 当前 token 也不写', '保留率 $\\exp(\\Delta \\cdot A) \\to 0$, 当前输入被强写入', '只放大写入、保留率不变, 当前 token 叠加在旧状态上'],
      answer: 2,
      why: '一个标量同时控制遗忘门和写入门; $\\Delta \\to 0$ 则相反: 保留率 $\\to 1$、写入 $\\to 0$, 当前 token 等于没出现过。',
    },
    {
      q: '线性时不变 (LTI) 的 SSM 为什么做不好 “从一堆废话里记住关键词”?',
      options: ['状态维度 N 太小, 装不下整段废话, 把 N 调大就能解决', '递推是非线性的, 长序列上梯度消失, 关键词记不住', '卷积核长度有限, 离结尾太远的 token 根本看不到', '参数与输入无关, 每个 token 的贡献只看离结尾多远'],
      answer: 3,
      why: '实验台里 LTI 模式下 “7” 的占比上限是 1/10; 让 $\\Delta$、$B$、$C$ 依赖输入之后才能按内容选择。',
    },
    {
      q: 'Mamba 用 “选择性” 换来了能力, 放弃了什么?',
      options: ['可写成全局卷积的性质, 训练改用并行 scan', '$O(1)$ 解码状态: 选择性让状态随长度增长', '训练并行性: 只能逐 token 串行递推, 无法并行', '因果性: 选择性参数要看完整段序列才能算'],
      answer: 0,
      why: '参数随时间变化后, 系统不再是卷积。好在递推满足结合律, 可以用并行 scan 在 $O(\\log T)$ 深度内算完。\n解码仍是 $O(1)$ 状态 (demo 里生成加速 16×)。',
    },
  ],
  'models-moe-balance': [
    {
      q: 'Aux-loss-free 均衡里, 路由偏置 $b$ 参与了哪一步?',
      options: ['只参与 top-k 选谁; 门控权重仍用原始分数', '只参与门控权重; top-k 仍按原始分数选', 'top-k 选择和门控权重都用加了 $b$ 的分数', '不进前向计算, 只作为正则项加进 loss 里'],
      answer: 0,
      why: '这样 $b$ 只改变 “谁上场”, 不改变 “上场后信多少”, 均衡和语言建模目标互不干扰。实验台里 “路由分数被改动量” 恒为 0。',
    },
    {
      q: '更新规则 $b \\mathrel{+}= \\gamma \\cdot \\mathrm{sign}(\\mathrm{mean} - \\mathrm{load})$ 只用符号、不用差值大小, 好处和风险分别是?',
      options: ['好处: 过载越多纠正越猛; 风险: 离群 batch 让 $b$ 跳变', '好处: 精确解出均衡的 $b$; 风险: 每步解一次, 计算贵', '好处: 不用调 $\\gamma$; 风险: $b$ 单调增长, 最终溢出', '好处: 步长恒为 $\\gamma$, 好调; 风险: $\\gamma$ 大时负载来回震荡'],
      answer: 3,
      why: '过载多少都只 $-\\gamma$。sigmoid 分数之间的差距也就 0.1 量级, 所以 $\\gamma$ 要小 (DeepSeek-V3 用 0.001)。',
    },
    {
      q: '$\\text{aux loss} = E \\cdot \\sum f_e \\cdot P_e$ 在完全均衡和完全坍缩时分别等于多少 ($E=8$, $K=2$)?',
      options: ['0 和 1', '1 和 8', '2 和 8', '8 和 2'],
      answer: 2,
      why: '均衡时 $f_e = K/E$、$\\sum P_e = 1 \\to K$; 坍缩时趋向 $E$。梯度只从 $P_e$ (路由概率) 这一支流回路由器, $f_e$ 是计数、不可导。',
    },
  ],
  'models-dsa': [
    {
      q: '只用语言模型 loss 训练, LightningIndexer 能学到东西吗?',
      options: ['能: top-k 用直通估计 (STE) 把梯度传回 indexer 的分数', '能: 被选中位置的注意力输出对 indexer 分数可导', '不能: top-k 不可导, indexer 梯度为 None, 需另加 KL loss', '不能: indexer 没有可学参数, 分数直接取 key 的范数'],
      answer: 2,
      why: '$\\mathrm{KL}(\\text{主注意力} \\,\\|\\, \\mathrm{softmax}(\\text{indexer 分数}))$ 把主注意力当 teacher。demo 里 KL 从 0.114 降到 0.005, top-8 召回从 0.45 升到 0.92。',
    },
    {
      q: 'DSA 省下的是什么?',
      options: ['KV cache: 只保留被 top-k 选中过的 K/V, 其余逐出', '主注意力计算 $O(T^2) \\to O(T \\cdot k)$; cache 反而多一份', '计算和 cache 都省: 没被选中的 K/V 可以直接丢掉', '参数量: indexer 取代了主注意力的 Q/K 投影'],
      answer: 1,
      why: '每个历史 token 仍可能被未来某个 query 选中, 所以 K/V 都得留着; 省的是每个 query 只对 k 个位置做昂贵的 MLA。',
    },
    {
      q: '为什么要先 dense warmup (主注意力看全部位置、只训 indexer), 再切换到稀疏?',
      options: ['dense 时梯度能经主注意力流回 indexer, 稀疏后就断了', '稀疏注意力不能反向传播, 要先 dense 把主模型训好冻结', 'dense 阶段让 indexer 与主注意力共享 key 投影, 省显存', '初始 indexer 等于随机挑 key, 直接稀疏会漏掉注意力大头'],
      answer: 3,
      why: '实验台里把 “对齐程度” 拖到 0: top-8 的输出误差接近 100%; 对齐之后同样的 k 误差只有百分之几。',
    },
  ],
  'models-gptoss': [
    {
      q: 'GPT-OSS 把滑窗层和全注意力层交替排布, 相比 “全部滑窗” 和 “全部全注意力” 各赢在哪?',
      options: ['比全滑窗: 一层即可直达全部历史; 比全注意力: 半数层 KV 封顶 W', '比全滑窗: KV cache 更小; 比全注意力: 能看到更远的历史', '比全滑窗: 参数更多、容量更大; 比全注意力: 所有层的 KV 都封顶 W', '比全滑窗: 训练更快; 比全注意力: 滑窗层不需要位置编码'],
      answer: 0,
      why: 'mini 模型 (T=40, W=8) 各层 cache 长度 [8, 40, 8, 40], 比全注意力省 40%; 裁掉的 K/V 本来就被 mask 成 −inf, 输出不变。',
    },
    {
      q: '标准 softmax 注意力的一个 head 在 “没什么值得看” 时会怎样?',
      options: ['logit 都很低时概率接近全 0, head 输出近似零向量', '概率退化成 one-hot, 集中到当前 token 自己身上', '概率和仍被迫为 1, 只能输出无关 value 的加权平均', '训练会把该 head 的 $W_O$ 压成 0, 这一层被旁路'],
      answer: 2,
      why: 'softmax 只认 logit 的相对差。这就是 attention sink 现象, 也是滑窗把开头 token 挤出 cache 后模型崩溃的原因。',
    },
    {
      q: 'GPT-OSS 的可学 sink logit 与 StreamingLLM “保留开头 4 个 token” 的区别是?',
      options: ['两者等价: sink logit 就是把开头 4 个 token 的 logit 学成常数', 'sink 是可学的虚拟 token, 有自己的 K/V, 要常驻 cache', 'sink 只在推理时加, 训练时不存在, 专为滑窗推理补救', '每 head 一个标量, 只进 softmax 分母, 无 value、不占 cache'],
      answer: 3,
      why: '3 个分数为 0 的 key 加一个 $\\ln 3$ 的 sink: 真实 key 合计只分到 0.5, 其余一半被直接丢弃, 每行概率和 $< 1$。',
    },
  ],
  'models-llada': [
    {
      q: 'LLaDA 和 BERT 都是 “遮盖再预测”, 为什么 LLaDA 能用来生成?',
      options: ['也固定遮 15%, 但采样时反复迭代, 每轮再遮 15% 重预测', '遮盖比例 $t$ 在 $(0,1)$ 随机, 覆盖从几乎全遮到只遮一点', '用因果 mask 训练, 遮盖只落在序列末尾, 等价于续写', '遮盖比例随机, 但 loss 只算在没被遮的位置上'],
      answer: 1,
      why: 'BERT 固定遮 15%, 从没见过 “几乎全是 [MASK]” 的输入; 配上 $1/t$ 加权, LLaDA 的 loss 还是负对数似然的上界。',
    },
    {
      q: '采样时 “低置信度重遮” 相比 “随机重遮” 好在哪?',
      options: ['高置信度位置可跳过后续前向, 总前向次数更少', '先定稿低置信度的位置, 高置信度的留到后面反复确认', '先定稿最有把握的位置, 作为上下文再定难的', '已定稿的 token 还能被重遮修改, 错误可以回滚'],
      answer: 2,
      why: 'demo 的填空准确率: 低置信度重遮 1.00、随机重遮 0.91、一步定完 0.89。生成顺序由置信度决定, 而不是从左到右。',
    },
    {
      q: 'LLaDA 采样为什么用不了 KV cache?',
      options: ['[MASK] 的 embedding 每步都会更新, 旧 K/V 随之失效', '步数本来就少于 token 数, 缓存命中率太低不值得', '因果 mask 下旧位置虽不变, 但 [MASK] 的 K/V 太多放不下', '双向注意力, 且每步任意位置都可能变, K/V 全要重算'],
      answer: 3,
      why: 'KV cache 成立的前提是因果 mask 下旧位置的表示不再改变。LLaDA 赚的是串行步数可以少于 token 数, 赔的是每步整段重算。',
    },
  ],
  'models-var': [
    {
      q: 'VAR 的自回归单位是什么?',
      options: ['下一分辨率的整张 token map, 级内并行生成', '下一个 patch: 每个分辨率上按光栅顺序逐个生成', '下一个去噪时间步: 每步整张图同时更新一遍', '下一个尺度的单个 token: 从粗到细、级内逐个生成'],
      answer: 0,
      why: '生成整张图只需 K 次前向 (demo: 1→2→4 共 3 次, 光栅要 16 次), 并且保留了二维邻接结构。',
    },
    {
      q: '多尺度 VQ 的第 k 级量化的是什么?',
      options: ['原图特征直接下采样到 $s_k \\times s_k$', '上一级量化结果上采样到本级后的特征', '目标特征减去前面所有级上采样之和的残差', '目标特征减去上一级 (仅一级) 上采样的残差'],
      answer: 2,
      why: '残差量化让细尺度能修正粗尺度留下的误差; 代价是 token 总数 $\\sum s^2$ 比单尺度多 (8×8 时 85 对 64)。',
    },
    {
      q: 'VAR Transformer 用的 “块状因果 mask” 规则是?',
      options: ['同一尺度内按光栅顺序因果, 可以看所有更粗的尺度', '同一尺度内互相可见, 并且只能看更粗的尺度', '全部尺度互相可见, 靠 loss mask 防止信息泄漏', '只能看同一尺度, 跨尺度靠上采样残差传递'],
      answer: 1,
      why: '训练时一次前向就能算出所有尺度的 loss。demo 里改动第 2 级的一个 token, 只会影响第 3 级的 logits。\n最细一级的 token 从不作为输入。',
    },
  ],
}
