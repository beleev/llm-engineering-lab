// 阶段 4 · llm_finetune 的全部章节 (11 章)。
// 每章都写全 StageTopic 要渲染的字段, 不再依赖 models.js 里的旧副本。
// source 只写真实存在的 def / class —— npm run check:sources 会逐条校验。
// 页面里的数字全部来自 llm_finetune/run_finetune/*/readme.md, 可以用对应的 run 命令复现。
export default {
  stage: 'finetune',
  chapters: [
    { route: 'finetune-qlora', label: 'QLoRA · NF4 量化基座', hint: '4 bit 存基座, 前向再解开; 整模型 389 KB → 59 KB' },
    { route: 'finetune-dora', label: 'DoRA 幅度/方向分解', hint: '低秩只管转方向, 每行长度单独学一个标量' },
    { route: 'finetune-merge', label: '模型合并 · 任务向量相加', hint: '不重训: θ₀ + λ(τ_A + τ_B), 两段改写 EM 1.000' },
    { route: 'finetune-simpo-orpo', label: 'SimPO · ORPO 无参考偏好优化', hint: '拿掉 ref: 前向次数减半, 显存减半' },
    { route: 'finetune-rlaif', label: 'RLAIF · 按宪法自动造偏好', hint: 'judge 批评 + 改写造偏好对; judge 没查的学不到' },
    { route: 'finetune-prm', label: 'PRM · 逐步打分', hint: '每步一个标签: best-of-8 0.488 → 0.648' },
    { route: 'finetune-ppo', label: 'PPO · critic 买到了什么', hint: 'GAE 逐 token 优势; 玩具规模下 critic ≈ 常数' },
    { route: 'finetune-grpo-variants', label: 'GRPO 进阶: clip · DAPO · Dr.GRPO · GSPO', hint: '一批采样更新多次, 才需要比率和 clip' },
    { route: 'finetune-onpolicy-distill', label: 'on-policy 蒸馏', hint: '学生自己写, 老师逐 token 打分' },
    { route: 'finetune-rlvr', label: '看题的可验证奖励', hint: '奖励不看题, RL 就只学会一个常数' },
  ],
  pages: {
    // ───────────────────────── SFT ─────────────────────────
    'finetune-sft': {
      title: 'SFT 数据与 loss · 只教模型回答, 不教模型提问',
      subtitle: '全章围着一行 mask 转: 哪一格该盖 -100, 哪一格盖了就等于从没教模型怎么开口。',
      tldr: 'labels 在数据侧已经左移一格, 所以 prompt 长 $P$ 时只能盖前 $P-1$ 格。盖成 $P$ 格会把第一个回复 token 一起盖掉。同配置实测: 正确写法留出集 exact-match 1.000, 多盖一位 0.000。',
      question: 'labels 已经错位之后, mask 的边界到底是 $P$ 还是 $P-1$? 多盖一格为什么从 loss 曲线上看不出来?',
      code: 'llm_finetune/data/tasks.py · llm_finetune/data/instruction_data.py · llm_finetune/methods/sft.py',
      points: [
        { title: 'SFT 没有新 loss', body: '还是那个带 ignore_index=-100 的交叉熵, SFTLoss 就是 StandardLMLoss 的别名。预训练学接龙, SFT 学"看到问题就回答", 全部差别只在 labels: 哪些位置算数。' },
        {
          title: '边界是 P−1',
          key: true,
          body: 'labels[t] 存的是 x[t+1]。最后一个 prompt token 坐在位置 $P-1$, 它要预测的恰好是第一个回复 token, 这一格必须留着。\nmake_labels 里有一条 assert 专门守这个: 被监督的位置数必须等于回复长度。',
        },
        { title: '差一格看不出来', body: '多盖一格只少一个监督位, loss 照样往下掉, 但模型从没学过 "看完问题第一个词说什么"。\n- 反转任务重跑: 正确写法留出集 EM 1.000, 多盖一位 0.000。\n- 答案只有 1 个 token: 全部标签都是 -100, loss 直接 nan。' },
      ],
      links: [
        { from: '预训练 CE', to: 'SFTLoss', body: '同一个类, 换的只是 labels 里哪些位置是 -100。' },
        { from: 'make_labels', to: 'DPO / SimPO / ORPO', body: '偏好方法的 $\\log\\pi(y\\mid x)$ 也从这套 labels 来。边界写错, 区分 chosen 与 rejected 的 $\\log p(y_1\\mid x)$ 这一项就丢了。' },
        { from: 'SFT 终态', to: 'policy / reference', body: 'DPO 的 policy 和 ref 都从 SFT checkpoint 复制。' },
      ],
      sourceRows: [
        { concept: '错位', code: 'llm_finetune/data/tasks.py:make_labels', takeaway: 'labels = seq[:, 1:] 后面补一格 -100。输入和标签在数据侧就错开了, loss 里不再 shift。' },
        { concept: 'prompt mask', code: 'labels[:, :P-1] = -100', takeaway: '只盖 prompt 内部的转移。第 $P-1$ 格的目标是第一个回复 token, 盖了就白训。' },
        { concept: '差一位的守卫', code: 'assert (labels != IGNORE).sum(1) == n_response', takeaway: '被监督的位置数必须恰好等于回复长度 (含 EOS), 写错立刻炸在这里。' },
        { concept: 'ignore_index', code: 'llm_finetune/methods/sft.py: SFTLoss = StandardLMLoss', takeaway: 'CE 对 -100 的位置既不算 loss 也不算进分母。' },
        { concept: '验收', code: 'llm_finetune/data/tasks.py:exact_match', takeaway: '留出集与训练集按 prompt 的 token 和 mod 5 不相交, 所以报的是泛化不是背诵。' },
      ],
      snippetTitle: 'SFT labels: 先错位, 再 mask',
      snippet: `x      = [BOS, p1, ..., p_{P-1}, r1, r2, EOS]     # 前 P 个是 prompt
idx    = x[:, :-1]                                # 模型输入
labels = x[:, 1:].clone()                         # 已左移一格: labels[t] = x[t+1]

labels[:, :P-1] = -100      # 正确: labels[P-1] = r1 要保留
# labels[:, :P] = -100      # 错误: 把 r1 也 mask 了 (off-by-one)

loss = cross_entropy(logits.reshape(-1, V), labels.reshape(-1),
                     ignore_index=-100)`,
      source: ['llm_finetune/data/tasks.py:make_labels', 'llm_finetune/data/instruction_data.py:_sample'],
      run: 'python -m llm_finetune.run_finetune.sft.train_sft',
    },

    // ───────────────────────── LoRA ─────────────────────────
    'finetune-lora': {
      title: 'LoRA 参数高效微调 · 冻结 W, 只学低秩 ΔW',
      subtitle: 'LoRA 换来了什么、又赔上了什么。这一章不照抄那句"LoRA 收敛更快"。',
      tldr: '把 $y = Wx$ 换成 $y = Wx + (\\alpha/r)\\cdot BAx$: $W$ 冻结, 只训 $A$、$B$。\n- 省下的: 梯度、Adam 状态, 和 "每个任务一整份权重"。\n- 不省的: 训练步数。同样 300 步, 全参留出集 EM 0.809, LoRA(r=8) 只有 0.352。',
      question: 'LoRA 到底买到了什么? 为什么 B 必须初始化成 0, 而 A 不行?',
      code: 'llm_finetune/methods/lora.py · llm_finetune/methods/qlora.py · llm_finetune/utils/param_utils.py',
      points: [
        { title: '低秩的是 ΔW, 不是 W', body: '微调带来的权重变化内在维度很低, 所以拿 $B\\,(d_{\\text{out}}\\times r)\\cdot A\\,(r\\times d_{\\text{in}})$ 去装它。\n参数从 $d_{\\text{in}}\\cdot d_{\\text{out}}$ 降到 $r\\cdot(d_{\\text{in}}+d_{\\text{out}})$: $d=4096$、$r=8$ 时一层从 16M 降到 65K, 占 0.39%。\n任务离预训练分布越远, $\\Delta W$ 越不低秩, LoRA 越吃亏。' },
        { title: 'B = 0 是无害启动', body: '- B 全零: 第 0 步 $BA = 0$, 输出与基座逐位相同, 微调起点就是预训练终点。\n- A 随机: 保证 B 第 1 步就有梯度。两个都置零, 梯度永远是 0。' },
        {
          title: '买的是显存和分发, 不是速度',
          key: true,
          body: '同一基座从 copy 适配到 sort, 各 300 步:\n- 全参: 99,648 个可训参数 / Adam 状态 778 KB / 留出集 EM 0.809\n- LoRA r=8: 19,456 个参数 / 152 KB, lr 拉到 1e-2 也才 EM 0.352\n"LoRA 收敛更快" 是个流行说法。在初始化正常、同看留出集的条件下, 它站不住。',
        },
      ],
      links: [
        { from: 'SFT loss', to: 'LoRA SFT', body: 'loss 一行不改, 只是 optimizer 只看得见 A/B。' },
        { from: 'target_modules', to: '模型结构', body: 'apply_lora 按属性名末段匹配 (w_q / w_v …), 与层路径无关。所以同一份代码能注到任何沿用同名属性的模型上。' },
        { from: 'methods/lora.py', to: 'methods/qlora.py', body: 'QLoRA 复用同一套 A/B 分支, 只把 base 权重换成 NF4 打包的 uint8 加每 block 一个 scale。' },
        { from: 'adapter', to: 'llm_infer Multi-LoRA', body: '不 merge, 多个 adapter 就能共享同一份基座, 按请求切换。' },
      ],
      sourceRows: [
        { concept: 'LoRALinear', code: 'llm_finetune/methods/lora.py:LoRALinear', takeaway: 'base 分支加 adapter 分支; A 用 Kaiming uniform, B 全零。' },
        { concept: '注入', code: 'llm_finetune/methods/lora.py:apply_lora', takeaway: '按模块名把目标 Linear 换成 LoRALinear。' },
        { concept: '只训 adapter', code: 'llm_finetune/methods/lora.py:mark_only_lora_as_trainable', takeaway: '把 base 的 requires_grad 关掉; 再用 print_trainable_parameters 核一遍才算数, loss 下降证明不了 base 被冻住。' },
        { concept: 'merge', code: 'llm_finetune/methods/lora.py:merge_lora_weights', takeaway: '$W \\leftarrow W + (\\alpha/r)\\cdot B\\cdot A$, 换回普通 Linear。要断言数值相等 (实测前后 logits 最大差 6.4e-06); 只断言形状会漏掉转置和漏乘 $\\alpha/r$。' },
        { concept: '落盘', code: 'llm_finetune/methods/lora.py:get_lora_state_dict', takeaway: '只抽出 lora_A / lora_B: 本仓库 adapter 76 KB, 整模型 389 KB。' },
      ],
      snippetTitle: 'LoRA forward 与 merge',
      snippet: `base = linear(x, W, b)              # W frozen
delta = lora_B(lora_A(dropout(x))) * (alpha / r)
y = base + delta

# merge:
W <- W + (alpha / r) * (B @ A)`,
      source: ['llm_finetune/methods/lora.py:LoRALinear', 'llm_finetune/methods/lora.py:merge_lora_weights'],
      run: 'python -m llm_finetune.run_finetune.lora.train_lora',
    },

    // ───────────────────────── QLoRA ─────────────────────────
    'finetune-qlora': {
      title: 'QLoRA · 把冻结的基座压到 4 bit',
      subtitle: '学会算一个模型量化之后到底占多少字节, 顺带搞清楚为什么 embedding 和 lm_head 要放过。',
      tldr: 'LoRA 省掉了梯度和优化器状态, 但冻结的基座还整份躺在显存里。\nQLoRA 把基座存成 NF4, 前向时现场反量化。本仓库整模型 389 KB → 59 KB, 压了 6.57×, 基座原任务留出集 EM 仍是 1.000。',
      question: '同样 16 个码点, 为什么按分位数摆比等间距好? block 为什么不能太大也不能太小?',
      code: 'llm_finetune/methods/qlora.py · llm_finetune/run_finetune/qlora/train_qlora.py',
      points: [
        { title: '码点跟着密度走', body: '权重大多挤在 0 附近。\n- NF4: 16 个码点取标准正态的等概率分位数, 中间密两端疏, 每个码被用到的概率都接近 1/16。\n- 等间距 INT4: 好几个码浪费在几乎没有权重的两端。' },
        {
          title: '账要算整个模型',
          key: true,
          body: '存储 = 0.5 字节/参数 + 每 64 个参数一个 fp32 scale = 0.5625 字节/参数, 对 fp32 的理论上限 7.1×。\n实测只有 6.57×, 差额就是没量化的 embedding / lm_head / RMSNorm。只报 "被量化那几层" 的压缩比是在骗自己。',
        },
        { title: '量化的是基座, 不是训练', body: '- NF4 权重: 只读 buffer, 不更新也不要梯度。前向时反量化成浮点再算。\n- LoRA 分支: 全程高精度。\n9.0% 的单层相对量化误差, 由 adapter 在训练里顺带补掉。适配 sort 的留出集 EM 0.402, 反而略高于同配置的 fp32 LoRA (0.352)。' },
      ],
      links: [
        { from: 'methods/lora.py', to: 'methods/qlora.py', body: 'A/B 分支完全复用, 只换 base 权重的存储方式。' },
        { from: 'nf4_quantize', to: 'nf4_dequantize', body: '分 block → 除 absmax → 最近邻查码本 → 两个 4 bit 拼一个 uint8; 反向就是拆包查表再乘回 scale。' },
        { from: 'block_size', to: '误差 / 开销', body: 'block 小则 scale 更贴合局部、误差更小, 但 scale 开销 $4/B$ 字节每参数变大。论文再对 scale 做一次双重量化, 就是为了压这块。' },
        { from: 'QLoRA', to: 'llm_infer 权重量化', body: '同一套 block-wise absmax 在推理侧再出现一次。' },
      ],
      sourceRows: [
        { concept: '码本', code: 'llm_finetune/methods/qlora.py: NF4_CODEBOOK', takeaway: '标准正态的 16 个等概率分位点, 归一化到 [-1, 1], 0 被显式保留。' },
        { concept: '量化', code: 'llm_finetune/methods/qlora.py:nf4_quantize', takeaway: '按 block 除 absmax, argmin 找最近码点, 两个 4 bit 打一个 uint8。block_size 为奇数时打包会错位, 脚本开头有单测。' },
        { concept: '反量化', code: 'llm_finetune/methods/qlora.py:nf4_dequantize', takeaway: '拆包查码本再乘回 block scale。' },
        { concept: '前向', code: 'llm_finetune/methods/qlora.py:NF4Linear', takeaway: 'weight 是个 property, 每次取都现场反量化; 外面再套 LoRA: $y = \\mathrm{dequant}(W)x + (\\alpha/r)\\cdot BAx$。' },
        { concept: '哪些层放过', code: 'llm_finetune/methods/qlora.py:apply_qlora', takeaway: '只量化 14 个线性层。embedding 与 lm_head 共享权重、查表的行分布不像正态, 且误差直接进 logits; bitsandbytes 默认也跳过。' },
        { concept: '显存账', code: 'llm_finetune/methods/qlora.py:weight_bytes', takeaway: '0.5625 字节/参数。合并时先反量化再加 $\\Delta W$, 保持高精度, 否则刚学到的更新会被量化噪声抹掉一部分。' },
      ],
      snippetTitle: 'NF4 量化 / 反量化',
      snippet: `blocks = w.reshape(-1, 64)
scale  = blocks.abs().amax(dim=1, keepdim=True)     # 每 block 一个 absmax
normed = blocks / scale                             # ∈ [-1, 1]
idx    = (normed[..., None] - NF4_CODEBOOK).abs().argmin(-1)   # 最近的码点
packed = (idx[0::2] << 4) | idx[1::2]               # 两个 4bit 拼一个 uint8

# 前向时:
w_hat = NF4_CODEBOOK[unpack(packed)] * scale        # 反量化回浮点
y = x @ w_hat.T + (alpha / r) * (x @ A.T) @ B.T     # base 冻结, 只训 A/B`,
      source: ['llm_finetune/methods/qlora.py:nf4_quantize', 'llm_finetune/methods/qlora.py:NF4Linear'],
      run: 'python -m llm_finetune.run_finetune.qlora.train_qlora',
    },

    // ───────────────────────── DoRA ─────────────────────────
    'finetune-dora': {
      title: 'DoRA · 把权重拆成幅度 × 方向',
      subtitle: '把 LoRA 与全参微调的差距定位到具体一步, 再用一个 $d_{\\text{out}}$ 维向量补上一截。',
      tldr: '把每一行权重看成 "长度 × 方向": $W\' = m \\odot (W_0 + BA) / \\|W_0 + BA\\|_{\\text{row}}$。\n- 低秩更新: 只管转方向。\n- m: 单独学长度。\n每层只多 $d_{\\text{out}}$ 个参数。同 r=8、3 个种子平均: LoRA 留出集 EM 0.382 → DoRA 0.522。',
      question: '同样的秩 r, 为什么多加一组"每行一个标量"就能更接近全参微调?',
      code: 'llm_finetune/methods/dora.py · llm_finetune/methods/lora.py',
      points: [
        {
          title: '两个量在 LoRA 里被绑住了',
          key: true,
          body: '- 全参微调: 一行权重的长度变化和方向变化基本各走各的。\n- LoRA: $\\Delta W = BA$ 是加性的。想 "只转方向不改长度", $\\Delta W$ 必须精确落在一个点上。\n- DoRA: 把两件事拆开。归一化之后 $BA$ 只剩方向, 长度交给 $m$。',
        },
        { title: '归一化顺手做了梯度投影', body: '$V = W_0 + (\\alpha/r)BA$ 被除以自己的行范数, loss 对 $V$ 的梯度里径向那一份自动消掉。低秩容量全花在转方向上。\n反传时分母 .detach(): 省一整份 [d_out, d_in] 的梯度显存, 效果几乎不变。' },
        { title: '参数几乎白送, 时间不白送', body: '- 参数: 比 LoRA 每层多 $d_{\\text{out}}$ 个 (本仓库 19,456 → 20,736, 多 1,280 个)。\n- 训练: 归一化作用在整行上, 每步都要显式构造 [d_out, d_in] 的 $W\'$, 拆不成两次小 matmul。\n- 推理: merge 完就是普通 Linear, 零开销。' },
      ],
      links: [
        { from: 'LoRALinear', to: 'DoRALinear', body: '继承同一个类, base + BA 不变, 外面再套一层"按行归一化 × m"。' },
        { from: 'weight normalization', to: 'DoRA', body: '思路同源: 把参数化改成幅度 / 方向, 优化曲面更好走。' },
        { from: 'DoRA', to: 'merge', body: '训完把 $m\\cdot V/\\|V\\|$ 算出来写回 $W$, 推理和 LoRA 一样没有额外开销。' },
      ],
      sourceRows: [
        { concept: '初始化', code: 'llm_finetune/methods/dora.py:DoRALinear', takeaway: 'lora_magnitude $= \\|W_0\\|_{\\text{row}}$, 形状 [d_out]; 配上 $B = 0$, 第 0 步 $W\'$ 与 $W_0$ 逐位相同。$m$ 初始化成 1 就没这个性质了。' },
        { concept: '方向分支', code: 'V = W₀ + (α/r)·B·A', takeaway: '和 LoRA 一模一样的低秩更新, 但只有它的方向会被用到。' },
        { concept: '归一化', code: 'direction.norm(dim=1, keepdim=True).detach()', takeaway: '论文 §4.3 把范数当常数; 忘了 detach 结果差不多, 但多一份全尺寸梯度显存。' },
        { concept: '幅度', code: 'lora_magnitude: nn.Parameter([d_out])', takeaway: '名字带 lora_ 前缀, 直接复用冻结和落盘工具。训练后 layers.0.attn.w_q 的 m 相对初值平均变了 59.7%。这个任务确实要改长度。' },
        { concept: '诚实的读数', code: 'run_finetune/dora/readme.md', takeaway: '3 个种子平均 loss 0.369 → 0.296, EM 0.382 → 0.522; 但换成 r=2 时 3 个种子里有 1 个是 LoRA 赢。玩具规模, 断言只要求平均值。' },
      ],
      snippetTitle: 'DoRA forward (骨架)',
      snippet: `V = W0 + (alpha / r) * (B @ A)            # 方向分支: 与 LoRA 相同的低秩更新
norm = V.norm(dim=1, keepdim=True)         # 逐行范数 [d_out, 1]
W = m[:, None] * V / norm.detach()         # m: 每行一个可训幅度; 范数当常数
y = x @ W.T                                # 归一化作用于整行, 每步都要显式构造 W

# 初始化: B = 0, m = W0.norm(dim=1)  ->  W == W0
# merge:  W0 <- m[:, None] * V / V.norm(dim=1, keepdim=True)`,
      source: ['llm_finetune/methods/dora.py:DoRALinear'],
      run: 'python -m llm_finetune.run_finetune.dora.train_dora',
    },

    // ───────────────────────── DPO ─────────────────────────
    'finetune-dpo': {
      title: 'DPO 偏好对齐 · 用 chosen/rejected 直接优化策略',
      subtitle: '这一章会让你多盯一个指标: reward margin 之外的 $\\log\\pi(\\text{chosen})$。它掉下去的时候, 模型正在变差。',
      tldr: 'KL 约束下的最优策略满足 $r(x,y) = \\beta\\cdot\\log\\pi/\\pi_{\\text{ref}} + \\text{const}$。代回 Bradley-Terry, RM 和 PPO 两步就塌缩成一个对偏好对的二分类。\npolicy 刚从 ref 复制时 $\\Delta = 0$, 第 1 步 loss 恰好是 $\\ln 2 = 0.6931$。',
      question: 'reward margin 一路拉大, 为什么生成质量反而掉了?',
      code: 'llm_finetune/methods/dpo.py · llm_finetune/data/preference_data.py',
      points: [
        { title: '数据形态变了, loss 外壳没变', body: '样本是 (prompt, chosen, rejected), 不是单条标准答案。\n送进 $-\\log\\sigma(\\cdot)$ 的是隐式奖励差 $\\Delta = (\\log\\pi_c - \\log\\mathrm{ref}_c) - (\\log\\pi_r - \\log\\mathrm{ref}_r)$。\nref 对同一条长回复也给出同样低的 $\\sum\\log p$, 相减之后长度红利被抵消。' },
        {
          title: 'margin 变大 ≠ 模型变好',
          key: true,
          body: 'loss 只看差值。"两边一起降、rejected 降得更快" 同样让 loss 变小。\n200 步实测:\n- 留出集偏好准确率: 0.965 → 0.996\n- log π(chosen): −4.03 → −4.18\n- 贪心 exact-match: 0.332 → 0.137\n这就是 likelihood displacement, 脚本对它设了断言。',
        },
        { title: 'β 越大越保守', body: '$\\beta$ 不是 "用力程度", 它就是 RLHF 目标里的 KL 惩罚系数。\n- β 大: $\\sigma$ 更快饱和 → 排对一点点梯度就没了 → policy 贴着 ref 不动。\n- β 小: 要把 $\\Delta$ 拉得很大才停手, 漂得更远。' },
        {
          title: 'KTO: 只有 👍 / 👎 也能训',
          body: '线上日志多是单条回复的 👍 / 👎。同一个 prompt 凑不出一对, DPO 用不了。\nKTO 给每条样本一个参考点 $z_0$:\n- 好回复: $r = \\log\\pi/\\pi_{\\text{ref}}$ 要高过它。\n- 坏回复: $r$ 要低过它。\n同 SFT 起点、同 β=0.5、同 200 步, 好坏 1:1:\n- 偏好准确率: 0.965 → 0.992 (DPO 0.996)\n- log π(chosen): −4.03 → −3.73 (DPO 掉到 −4.25)\n- 贪心 EM: 0.324 (DPO 0.121)\n这里的 DPO 是 train_kto 里的同条件对照; 单独跑 train_dpo 是 −4.18 / EM 0.137, 两次独立运行, 结论一致。\n好样本有自己的一项, 直接把 $r$ 往上推, 所以没有 likelihood displacement。',
        },
        {
          title: 'KTO: 好坏不均时 λ 要按比例调',
          body: '好:坏 = 1:9 时, 要按 $\\lambda_D n_D \\approx \\lambda_U n_U$ 把 $\\lambda_U$ 调成 1/9。\n- 调: 准确率 0.988, EM 0.254。\n- 不调: 坏样本的梯度压倒一切, 所有回复一起往下压。准确率 0.492, log π(chosen) −30.02, EM 0.000。\n诚实边界:\n- 参考点: 本例 $z_0$ 全程为 0 (错配估计 0 → −2.52 被 clamp), 这一项没发挥作用。\n- 标签密度: 同前向预算下, KTO 每步有标签的回复 64 条, 只有 DPO 的一半 (64 对 = 128 条)。',
        },
      ],
      links: [
        { from: 'SFT 终态', to: 'policy / reference', body: 'policy 可训, ref 冻结 + eval + no_grad。ref 若被注册成子模块, 会进 optimizer、被 .train() 切模式, 所以代码里故意把它塞在 tuple 里。' },
        { from: 'make_labels', to: 'compute_sequence_logprobs', body: '同样忽略 prompt 和 pad。多盖一位会丢掉 $\\log p(y_1\\mid x)$: 上下文相同, 但 chosen 与 rejected 的 $y_1$ 可以不同, 这一项恰恰是区分两者的。' },
        { from: 'DPOLoss', to: 'SimPO / ORPO', body: '同一份数据、同一个 logsigmoid 外壳, 换的是送进去的 z 和要不要 ref。' },
        { from: 'PairwiseForward', to: '通用 Trainer', body: '把 "policy 前向 + ref 前向" 包成一个 nn.Module, Trainer 一行不改, 不用为 DPO 单写一套训练循环。' },
      ],
      sourceRows: [
        { concept: 'logprob 汇总', code: 'llm_finetune/methods/dpo.py:compute_sequence_logprobs', takeaway: '返回 [B], 保留逐样本粒度。用 cross_entropy(reduction="sum") 会把整个 batch 加成一个数, 没法逐对相减。' },
        { concept: 'DPO 的 z', code: 'beta * ((p_c - p_r) - (ref_c - ref_r))', takeaway: 'policy 相对 ref 的隐式奖励差。policy = ref 时恒为 0, 所以第 1 步 loss 必是 $\\ln 2$, 这是最好用的自检。' },
        { concept: '梯度权重', code: 'llm_finetune/methods/dpo.py:DPOLoss', takeaway: '每个样本的权重是 $\\sigma(-\\beta\\Delta)$: 排对且已拉开的样本自动退出, 力气集中在排错的那些上。' },
        { concept: '一步的开销', code: 'llm_finetune/methods/dpo.py:PairwiseForward', takeaway: '序列数 ×2 (chosen + rejected), 再加一次不带梯度的 ref 前向; 常驻权重 ×2 = 778 KB。ref 不存激活, 所以实测每步只慢约 35%, 不是 100%。' },
        { concept: '偏好数据', code: 'llm_finetune/data/preference_data.py:PreferenceDataGenerator', takeaway: 'chosen = 正确回复, rejected = 改错一个 token 或漏掉一个。两边格式必须对称, 否则"只有一边带 EOS"这种捷径会被学走。' },
        // ── KTO: 没有成对数据时的 DPO 替身 ──
        { concept: 'KTO 数据', code: 'llm_finetune/methods/kto.py:UnpairedDataGenerator', takeaway: '把偏好对拆开, 每个 prompt 只给一条回复 (好或坏), DPO 一对也凑不出。另附一份错配的 $(x_i, y_{i+1})$, 只用来估 $z_0$。' },
        { concept: 'KTO loss', code: 'llm_finetune/methods/kto.py:KTOLoss', takeaway: '- 好: $v = \\lambda_D\\,\\sigma(\\beta(r - z_0))$\n- 坏: $v = \\lambda_U\\,\\sigma(\\beta(z_0 - r))$\n- loss: $\\mathbb{E}[\\lambda_y - v]$\n$r$ 与 DPO 同一个 $\\log\\pi/\\pi_{\\text{ref}}$, 只是没乘 $\\beta$。' },
        { concept: '参考点 z0', code: 'kl_est.clamp(min=0).detach()', takeaway: '用错配样本估, 不回传梯度。本例错配样本的 $\\log\\pi/\\pi_{\\text{ref}}$ 从 0 降到 −2.52, 被 clamp, $z_0$ 全程为 0。' },
        { concept: '前向复用', code: 'llm_finetune/methods/kto.py:KTOForward', takeaway: '继承 PairwiseForward, 两路 = 本条样本 / 错配样本, 通用 Trainer 一行不改。' },
      ],
      snippetTitle: 'DPO 核心',
      snippet: `pi_logratios  = policy_chosen - policy_rejected
ref_logratios = ref_chosen - ref_rejected
logits = beta * (pi_logratios - ref_logratios)

loss = -F.logsigmoid(logits).mean()`,
      source: ['llm_finetune/methods/dpo.py:DPOLoss', 'llm_finetune/methods/dpo.py:compute_sequence_logprobs', 'llm_finetune/methods/kto.py:KTOLoss', 'llm_finetune/methods/kto.py:UnpairedDataGenerator'],
      run: 'python -m llm_finetune.run_finetune.dpo.train_dpo',
    },

    // ───────────────────────── SimPO / ORPO ─────────────────────────
    'finetune-simpo-orpo': {
      title: 'SimPO · ORPO — 不要 reference model 的偏好优化',
      subtitle: '搞清 ref 在替你挡什么, 以及拿掉它之后得请谁来接班。',
      tldr: 'ref 要多占一份显存、每步多一次前向。实测 DPO 2 次 / 778 KB / 5.4 s, SimPO 与 ORPO 1 次 / 389 KB / 4.0 s。\n- SimPO: 用长度归一化的平均 log p 当奖励, 再减目标间隔 $\\gamma$。\n- ORPO: 直接在 SFT 的 NLL 上加一项 odds ratio 惩罚, 一个阶段搞定。',
      question: 'DPO 里的 ref 除了"别离 SFT 太远", 还顺手解决了什么? 拿掉它谁来接手?',
      code: 'llm_finetune/methods/simpo.py · llm_finetune/methods/orpo.py · llm_finetune/methods/dpo.py',
      points: [
        { title: 'Σlog p 有长度红利', body: '序列越长, $\\sum\\log p$ 越负。不做校正的话, rejected 更长就等于白送 margin: 模型什么偏好都没学, loss 已经到 0。\nDPO 靠 ref 抵消这一项: ref 对同一条长回复也给出同样低的分。' },
        { title: 'SimPO: 归一化 + γ', body: '按长度取平均: 长度直接约掉, 奖励也和生成时的打分 (平均 $\\log p$) 对上了。\n- 代价: 平均后数值范围小得多, $\\beta$ 要取 2 左右。\n- γ: 规定 chosen 至少领先 $\\gamma/\\beta$ 才准停手。没有 ref, 就得自己画一条线。' },
        {
          title: '省下的显存, 要用别的东西换',
          key: true,
          body: '三者的偏好准确率都上去了 (0.988~0.996)。差别在 $\\log\\pi(\\text{chosen})$ 和留出集 EM:\n- DPO: −4.25, EM 0.121\n- SimPO: −5.69, EM 0.023。什么锚都没有, 掉得最惨。\n- ORPO: −2.51, EM 0.543。NLL 项就是那个锚。\n所以 ORPO 能从零开始训: 300 步 EM 0.973, 与纯 SFT 打平, 还把 rejected 压得更低 (−16.34 vs −14.85)。',
        },
      ],
      links: [
        { from: 'DPOLoss', to: 'SimPO / ORPO', body: '同一份 (prompt, chosen, rejected)、同一个 logsigmoid 外壳, 换的是送进去的 z。' },
        { from: 'compute_sequence_logprobs', to: '平均 log p', body: 'SimPO 与 ORPO 都要再除以回复的有效 label 数。' },
        { from: 'ref 的两次前向', to: '省掉', body: 'DPO: policy×2 + ref×2; SimPO / ORPO: policy×2。实测每步快约 25%。反传还在, 所以省不到一半。' },
        { from: 'ORPO 的 NLL 项', to: 'SFT', body: '前一项就是 SFT, 所以 ORPO 不需要先做一遍 SFT; 它同时充当防漂移的锚。' },
      ],
      sourceRows: [
        { concept: 'DPO 的 z', code: 'llm_finetune/methods/dpo.py:DPOLoss', takeaway: '$\\beta\\cdot[(\\pi_c - \\mathrm{ref}_c) - (\\pi_r - \\mathrm{ref}_r)]$: ref 同样"越长越负", 长度项成对抵消。' },
        { concept: 'SimPO 的 z', code: 'llm_finetune/methods/simpo.py:SimPOLoss', takeaway: '$\\beta\\cdot(\\log p_c/|y_c| - \\log p_r/|y_r|) - \\gamma$; 本仓库默认 $\\beta=2$、$\\gamma=1$。沿用 DPO 的 $\\beta=0.1$ 会几乎推不动。' },
        { concept: 'γ = 0 且不归一化', code: 'z = logp_c − logp_r', takeaway: '退化成"$\\pi_{\\text{ref}}$ 取均匀分布的 DPO", 长度红利全回来了。' },
        { concept: 'ORPO 的 odds', code: 'log(p/(1−p)), p = exp(mean logp)', takeaway: '用每 token 平均概率, 同样与长度无关。$p\\to 1$ 时 $\\log(1-p)$ 要 clamp, 否则 $\\log 0$。' },
        { concept: 'ORPO 总 loss', code: 'llm_finetune/methods/orpo.py:ORPOLoss', takeaway: 'nll_chosen + λ·(−logsigmoid(log_or))。论文 $\\lambda=0.1$, 本仓库玩具任务默认 0.5。odds 在 $p\\to 1$ 处发散, 对"两边都已经很自信"的差距比概率比更敏感。' },
      ],
      snippetTitle: '三种 z 并排',
      snippet: `lp_c, lp_r = seq_logprob(policy, chosen), seq_logprob(policy, rejected)   # [B] 求和
n_c, n_r   = len(chosen_resp), len(rejected_resp)

# DPO: 需要 ref 的两次前向
z_dpo   = beta * ((lp_c - ref_c) - (lp_r - ref_r))

# SimPO: 无 ref, 长度归一 + 目标间隔
z_simpo = beta * (lp_c / n_c - lp_r / n_r) - gamma

# ORPO: 无 ref, odds ratio + SFT 锚
odds    = lambda lp, n: (lp / n) - log1p(-exp(lp / n))     # log(p / (1-p))
loss_orpo = -(lp_c / n_c) + lam * -logsigmoid(odds(lp_c, n_c) - odds(lp_r, n_r))

loss = -logsigmoid(z).mean()      # DPO / SimPO 共用的外壳`,
      source: ['llm_finetune/methods/simpo.py:SimPOLoss', 'llm_finetune/methods/orpo.py:ORPOLoss'],
      run: 'python -m llm_finetune.run_finetune.simpo_orpo.train_simpo_orpo',
    },

    // ───────────────────────── RM · GRPO · 蒸馏 ─────────────────────────
    'finetune-rlhf': {
      widgets: ['GrpoLab', 'SoftmaxTempLab'],
      title: 'RM · GRPO · 蒸馏 — 从偏好到能力迁移',
      subtitle: '三件事: 标注有噪声时 RM 最多学到多大分差、GRPO 的 baseline 从哪来、蒸馏的 KL 项为什么要乘 $T^2$。',
      tldr: '- RM: 把 "A 比 B 好" 学成一个能随时调用的标量分, 留出集偏好准确率 0.549 → 0.930。\n- GRPO: 用同题 G 条回复的组内均值当 baseline, 把 PPO 的 critic 整个省掉。\n- 蒸馏: 用温度软化的 teacher 分布, 给 student 比硬标签密得多的监督。',
      question: 'GRPO 砍掉 critic 之后 baseline 从哪来? 标注有 20% 概率标反时, RM 学到的分差会停在哪?',
      code: 'llm_finetune/methods/reward_model.py · llm_finetune/methods/grpo.py · llm_finetune/methods/distill.py',
      points: [
        {
          title: 'RM 只学分差',
          key: true,
          body: 'Bradley-Terry: $P(A\\succ B) = \\sigma(r_A - r_B)$, loss $= -\\log\\sigma(r_w - r_l)$。\n给所有分数加同一个常数, loss 一点不变。分数没有量纲, 只有序。\n标注有 $\\varepsilon$ 的概率标反时, 期望 loss 的零梯度点落在 $\\sigma(\\Delta) = 1-\\varepsilon$, 最优分差停在 $\\ln((1-\\varepsilon)/\\varepsilon)$: $\\varepsilon=0.2$ 约 1.39, $\\varepsilon=0.5$ 时为 0。',
        },
        { title: '组内排名顶替 critic', body: '同一个 prompt 采 G 条, $\\hat{A} = (r - \\mu)/\\sigma$。组内均值就是 "这道题的期望得分" 的蒙特卡洛估计, 正是 value 网络想预测的东西。\n代价: 一组全对或全错时 $\\hat{A}$ 全为 0, 这批样本不产生任何梯度。实测每步约 32% 的 prompt 落在这种情况里。' },
        { title: '暗知识在分布里', body: '- 硬标签: 只告诉 student 一个 token。\n- teacher 软化后的分布: 把所有相似 token 的相对排序都交出来。\nKD 项要乘 $T^2$: 软标签对 logits 的梯度按 $1/T^2$ 衰减, 不补偿的话, 调高温度等于偷偷关掉蒸馏项。\n数据有限时软标签明显更好 (forward KL 2.292 → 1.860), 数据无限时这个优势会消失。' },
      ],
      links: [
        { from: 'PreferenceDataGenerator', to: 'RewardModel', body: '与 DPO 同一份数据, 两种用法: RM 学打分, DPO 直接学策略。' },
        { from: 'RM 分数', to: 'GRPO 的 reward_fn', body: 'reward_fn(prompts, completions) 既可以是 RM, 也可以是程序 verifier, 后者就是 RLVR。' },
        { from: 'policy.generate', to: '组内优势', body: '在线采样, 数据分布随 policy 漂移 (on-policy); 这也是 GRPO 接不进通用 Trainer 的原因。' },
        { from: 'teacher logits / T', to: 'student KL', body: '$T$ 放大暗知识, $T^2$ 补回 softmax 梯度的 $1/T^2$ 缩放, 软硬两项的相对权重才只由 $\\alpha$ 决定。' },
      ],
      sourceRows: [
        { concept: 'value head', code: 'llm_finetune/methods/reward_model.py:RewardModel', takeaway: '复用 LLaMA 骨架, lm_head 换成 Identity 再接一个 [D]→[1] 的头。手抄一遍主干前向的话, 主干一改 (mask / RoPE / cache) 就会悄悄不一致。' },
        { concept: '读哪一个位置', code: 'attention_mask.sum(1) - 1', takeaway: '右 pad 时取最后一个真 token。取 [:, -1] 读到的是 pad: 同一序列多垫 5 位 PAD, 分数偏移从 5e-06 变成 4.54。' },
        { concept: 'BT loss', code: 'llm_finetune/methods/reward_model.py:BradleyTerryLoss', takeaway: '第 1 步 loss $0.6929 \\approx \\ln 2$ (两边分数还没拉开)。' },
        { concept: '组内优势', code: 'llm_finetune/methods/grpo.py:group_advantages', takeaway: 'adv = (r − mean) / (std + eps); 组内奖励全相同时 adv 全为 0, 这组白占 batch。DAPO 的动态采样就是冲它去的。' },
        { concept: '重要性比率', code: 'llm_finetune/methods/grpo.py:GRPOTrainer', takeaway: '一批 rollout 更新 $\\mu=2$ 次: 第 1 个 epoch $\\rho\\equiv 1$ (实测 $|\\rho-1| = 0.000$), clip 从第 2 个 epoch 才开始起作用。' },
        { concept: 'KL 的 k3 估计', code: 'd.exp() - d - 1', takeaway: '逐 token、恒 $\\ge 0$、方差比直接用 log-ratio 小。只有 $\\beta>0$ 时才需要常驻一份 ref。' },
        { concept: '蒸馏损失', code: 'llm_finetune/methods/distill.py:DistillLoss', takeaway: '$\\alpha\\cdot\\mathrm{CE} + (1-\\alpha)\\cdot T^2\\cdot\\mathrm{KL}(p_t^T \\,\\|\\, p_s^T)$; KD 项和 CE 项必须用同一个 mask, 否则两项优化的不是同一批位置。' },
      ],
      snippetTitle: 'GRPO 单步的完整控制流',
      snippet: `seqs = policy.generate(prompts × G)        # 1. 组内采样
rewards = reward_fn(prompts, seqs)         # 2. 规则验证打分 (RLVR)
adv = (r - r.mean(group)) / r.std(group)   # 3. 组内相对优势
loss = -(adv * logp.mean(-1)).mean()       # 4. 策略梯度
     + beta * kl(policy, ref)              #    + KL 锚定
loss.backward(); opt.step()                # 5. 一次更新`,
      source: [
        'llm_finetune/methods/reward_model.py:BradleyTerryLoss',
        'llm_finetune/methods/grpo.py:GRPOTrainer',
        'llm_finetune/methods/distill.py:DistillLoss',
      ],
      run: 'python -m llm_finetune.run_finetune.rm.train_rm',
    },

    // ───────────────────────── GRPO 变体 ─────────────────────────
    'finetune-grpo-variants': {
      title: 'GRPO 进阶 · 重要性比率、clip 与 DAPO / Dr.GRPO / GSPO',
      subtitle: '把三个变体各自在修的那一处偏置指出来。也别把四条噪声之内的最终分数读成"谁更强"。',
      tldr: '一批采样只更新一次的话, $\\rho\\equiv 1$, clip 是死代码。\n想让昂贵的 rollout 多用几轮, 就得引入 $\\rho = \\pi_{\\text{new}}/\\pi_{\\text{old}}$ 和裁剪。三个变体分别修这个目标里的三处偏置: 熵塌缩、长度偏置、比率噪声。',
      question: '同一条"又长又错"的回答, 在 GRPO / Dr.GRPO / DAPO 下每个 token 分到的惩罚分别是多少?\n为什么上界 clip 只卡得住低概率 token?',
      code: 'llm_finetune/methods/grpo.py · llm_finetune/run_finetune/grpo/train_grpo.py',
      points: [
        { title: 'clip 是停手线, 不是刹车', body: '$\\min(\\rho\\hat{A}, \\mathrm{clip}(\\rho)\\hat{A})$ 在越界那一侧是平的, 梯度为 0。\n$\\rho$ 最大只能到 $1/\\pi_{\\text{old}}$: $\\pi_{\\text{old}}=0.9$ 的 token 顶天涨到 1.11, 碰不到 1.2 的上界。被卡住的全是想翻身的低概率探索 token → 熵塌缩。\n所以 DAPO 的 clip-higher 只放宽上界 (0.2 / 0.28)。' },
        {
          title: '两处偏置藏在分母里',
          key: true,
          body: '- 1/|o_i|: 答错的长回复每个 token 罚得更轻。实测 3-token 回复的单 token 权重是 9-token 的 3.0 倍, 等于鼓励 "错就错得长一点"。\n- ÷σ: 给几乎全对、几乎全错的题加权。难题比中等题的权重 0.66, 去掉 $\\sigma$ 后只有 0.44。\nDr.GRPO 把两个分母都换成常数。',
        },
        { title: '比率的粒度', body: '奖励是整条序列给的, token 级 $\\rho_t$ 却是单样本噪声估计。\nGSPO 改用 $s = \\exp(\\mathrm{mean}_t \\log\\rho_t)$, 整条回复共用一个权重一起裁剪。实测 $\\mathrm{std}\\,\\log\\rho$ 从 0.067 降到 0.027 (约小 $\\sqrt{C}$ 倍)。\n所以它的裁剪区间也要跟着收窄到 0.05。' },
      ],
      links: [
        { from: 'GRPOTrainer.step', to: '多轮内层更新', body: '采样一次, 存下 logp_old, 对同一批做 $\\mu$ 次更新。实测第 1 个 epoch $|\\rho-1| = 0.000$, 第 2 个 epoch 才偏离。' },
        { from: 'PPO clipped surrogate', to: 'GRPO', body: '目标函数外形完全一样, GRPO 只换了优势的来源 (组内归一化)。' },
        { from: '组内奖励全相同', to: 'DAPO 动态采样', body: '$\\hat{A}$ 全 0 的组不贡献梯度。实测这类 prompt 占 32%。\n- DAPO: 把它们从 loss 里去掉, 进 loss 的 A=0 样本 32.1% → 0.0%。\n- 本实现: 只过滤不补采, batch 会变小。' },
        { from: '采样温度 T', to: 'log-prob 的 T', body: '采样用 $\\pi^{1/T}$, 算 log-prob 也必须用同一个 $T$, 否则 $\\rho$ 的分母不是行为策略, 第 1 个 epoch 就已经 off-policy。' },
      ],
      sourceRows: [
        { concept: '四个变体 = 五个开关', code: 'llm_finetune/methods/grpo.py:GRPOConfig', takeaway: 'clip_high / std_norm / loss_agg / seq_ratio / dynamic_sampling; VARIANTS 字典把 grpo · dapo · dr_grpo · gspo 写成开关组合。' },
        { concept: '组内优势', code: 'llm_finetune/methods/grpo.py:group_advantages', takeaway: '$r - \\mu$; std_norm=True 时再除 $\\sigma$。' },
        { concept: '重要性比率', code: 'ratio = exp(logp_new − logp_old.detach())', takeaway: 'logp_old 在采样后立刻用同一温度算好并冻结。' },
        { concept: 'loss 聚合', code: 'llm_finetune/methods/grpo.py:aggregate', takeaway: 'seq_mean (GRPO, $\\div |o_i|$) · token_mean (DAPO) · fixed_len (Dr.GRPO, $\\div$ 常数 $C$): 差别只在每个 token 的权重。' },
        { concept: '别误读最终分数', code: 'run_finetune/grpo/readme.md', takeaway: '60 步后留出集采样 pass@1: grpo 0.287 / dapo 0.326 / dr_grpo 0.324 / gspo 0.320, 差异在噪声内。贪心 EM 基本没动 (0.562 → 0.53)。RLVR 主要把 pass@k 挤进 pass@1, 不是教新能力。' },
        { concept: '用什么验收', code: '采样 pass@1, 不是贪心 EM', takeaway: 'RL 优化的就是采样分布; 拿贪心解码去验收 RL, 什么都看不到。' },
      ],
      snippetTitle: '带 ratio + clip 的 GRPO 内层循环',
      snippet: `seqs, mask = sample_group(policy, prompts, G)
r    = reward_fn(prompts, seqs)                       # [B, G]
adv  = r - r.mean(1, keepdim=True)                    # Dr.GRPO 到此为止
adv  = adv / (r.std(1, keepdim=True) + 1e-4)          # GRPO / DAPO 再除 σ
keep = r.std(1) > 0                                   # DAPO: 全对/全错的组丢弃重采
logp_old = token_logprobs(policy, seqs).detach()

for _ in range(K):                                    # 同一批样本更新 K 轮
    logp  = token_logprobs(policy, seqs)
    ratio = (logp - logp_old).exp()                   # token 级 ρ_t
    # GSPO: ratio = (((logp - logp_old) * mask).sum(1) / mask.sum(1)).exp()[:, None]
    obj = torch.min(ratio * adv, ratio.clamp(1 - e_lo, 1 + e_hi) * adv)
    loss = -(obj * mask).sum(1) / mask.sum(1)         # GRPO: 先按序列平均 (1/|o_i|)
    # DAPO: -(obj * mask).sum() / mask.sum()          # token 级
    # Dr.GRPO: -(obj * mask).sum(1) / L_MAX           # 常数分母
    loss.mean().backward(); opt.step(); opt.zero_grad()`,
      source: [
        'llm_finetune/methods/grpo.py:GRPOConfig',
        'llm_finetune/methods/grpo.py:group_advantages',
        'llm_finetune/methods/grpo.py:aggregate',
      ],
      run: 'python -m llm_finetune.run_finetune.grpo.train_grpo',
    },

    // ───────────────────────── on-policy 蒸馏 ─────────────────────────
    'finetune-onpolicy-distill': {
      title: 'on-policy 蒸馏 · 学生自己写, 老师逐 token 打分',
      subtitle: '用 KL 写在哪一侧解释两种蒸馏的性格差异, 再挑出容量不够的学生该用哪一种。',
      tldr: '- 离线蒸馏: 在老师写的前缀上教学生。forward KL, 摊开盖住所有峰。\n- on-policy: 学生先自己生成, 再在自己走到的每个位置上对齐老师。reverse KL, 钻进一个峰。\n实测样本合格率 0.059 → 0.402。',
      question: '容量不够的学生, 该摊开盖住老师的所有答法, 还是挑一种答到位?\n这跟 KL 写在哪一侧有什么关系?',
      code: 'llm_finetune/methods/on_policy_distill.py · llm_finetune/methods/distill.py',
      points: [
        {
          title: 'KL 的方向决定性格',
          key: true,
          body: '$\\mathrm{KL}(p\\,\\|\\,q) = \\sum p\\cdot\\log(p/q)$: 谁写在前面, 期望就在谁的分布上取。\n- forward KL: 在老师的样本上取期望。老师有质量而学生没有的地方, 惩罚趋于无穷。学生被迫全覆盖, 代价是往两峰之间的低谷里也放概率, 采样时半路串台。\n- reverse KL: 在学生自己的样本上取期望。学生不去的地方权重为 0, 于是它缩进一个峰。',
        },
        { title: '分布错配', body: '- 离线蒸馏: 只见过老师的前缀。推理时学生走进自己的前缀, 那是从没被教过的状态, 错误一步步累积。\n- on-policy: 直接在学生的前缀上训练, 训练分布就是推理分布。' },
        { title: '稠密 + on-policy, 但会丢多样性', body: '- RL: 每条序列只有 1 个标量奖励。\n- on-policy 蒸馏: 每个 token 位置都有老师的完整 $V$ 维分布, 而且解析求 KL, 不用 REINFORCE。\n实测对比离线: 合格率 0.402 vs 0.059, reverse KL 0.51 vs 2.04。\n代价: 老师 30% 概率的少数派答法被压到 $\\log\\pi = -33$ (离线只到 −17)。' },
      ],
      links: [
        { from: 'DistillLoss (离线)', to: 'on-policy 蒸馏', body: '温度软化和 $T^2$ 补偿仍然适用; 变的是样本来源和 KL 写在哪一侧。' },
        { from: 'student.generate', to: 'teacher 打分', body: '与 GRPO 共用采样管线, 只是把 reward_fn 换成老师的逐 token 分布。采样不回传梯度, 它只决定"在哪些状态上算 KL"。' },
        { from: 'reverse KL', to: 'GRPO 的 KL 惩罚', body: '锚住 ref 的那一项就是同一个量: $\\mathbb{E}_{y\\sim\\pi}[\\log\\pi - \\log\\pi_{\\text{ref}}]$。' },
        { from: '热身', to: 'on-policy', body: '从随机初始化直接上 on-policy 没用。学生采的全是垃圾, 老师在这些分布外前缀上的输出也没意义。先用 SFT 或离线蒸馏热身 200 步。' },
      ],
      sourceRows: [
        { concept: '离线基线', code: 'llm_finetune/methods/distill.py:DistillLoss', takeaway: '$\\alpha\\cdot\\mathrm{CE} + (1-\\alpha)\\cdot T^2\\cdot\\mathrm{KL}(p_t^T \\,\\|\\, p_s^T)$。这是 forward KL, 数据来自固定语料。' },
        { concept: '学生采样', code: 'llm_finetune/methods/on_policy_distill.py:on_policy_distill_step', takeaway: 'generate 在 no_grad 下跑; 梯度只来自随后那次带梯度的前向。' },
        { concept: '逐 token reverse KL', code: 'llm_finetune/methods/on_policy_distill.py:token_kl', takeaway: '$\\sum_v q_s(v)\\cdot(\\log q_s(v) - \\log p_t(v))$, 在完整词表上精确求和, 不需要对 KL 做采样估计。' },
        { concept: '各赢各的', code: 'run_finetune/on_policy_distill/readme.md', takeaway: 'off-policy forward KL 0.88 / reverse 2.04; on-policy forward 1.36 / reverse 0.51。谁优化哪个指标就赢哪个, 别只看一栏。' },
        { concept: '为什么不接 Trainer', code: 'run_finetune/on_policy_distill/train_on_policy_distill.py', takeaway: '训练数据由当前 student 现场生成, 数据生成器得拿到模型本身, 和 GRPO 一个原因。' },
      ],
      snippetTitle: 'on-policy 蒸馏一步',
      snippet: `with torch.no_grad():
    seqs = student.generate(prompts)                 # 1. 学生自己写 (on-policy)
    t_logp = log_softmax(teacher(seqs), -1)          # 2. 老师在学生的前缀上给出分布

s_logp = log_softmax(student(seqs), -1)              # 3. 学生带梯度重算一遍
# 4. 逐 token reverse KL: 期望在学生自己的分布上取
kl = (s_logp.exp() * (s_logp - t_logp)).sum(-1)      # [B, T]
loss = (kl * completion_mask).sum() / completion_mask.sum()
loss.backward(); opt.step()`,
      source: [
        'llm_finetune/methods/on_policy_distill.py:on_policy_distill_step',
        'llm_finetune/methods/on_policy_distill.py:token_kl',
      ],
      run: 'python -m llm_finetune.run_finetune.on_policy_distill.train_on_policy_distill',
    },

    // ───────────────────────── RLVR ─────────────────────────
    'finetune-rlvr': {
      title: '看题的可验证奖励 · 别让 RL 只学会一个常数',
      subtitle: '跑 RL 之前先算一个数: 常数基线。它决定那条漂亮的 reward 曲线值不值钱。',
      tldr: '奖励必须是 (prompt, completion) 的函数。\n- 区域奖励 (如 "输出落在词表后半区就给分"): 最优策略是无视 prompt 的常数输出。reward 能涨到 1.0, 却什么都没证明。\n- "答案 = f(prompt)": 常数策略的上限掉回 1/类别数。',
      question: '一条一路涨到 1.0 的 reward 曲线, 怎么分辨"学会按题作答"和"学会了一个 unigram 偏好"?',
      code: 'llm_finetune/data/tasks.py · llm_finetune/data/prompt_data.py · llm_finetune/methods/grpo.py',
      points: [
        {
          title: '先算常数基线',
          key: true,
          body: '常数基线 = 让所有 prompt 都输出同一个 completion 时, 能拿到的最高平均奖励。\n- 基线等于 1.0: 这个奖励函数验证不了任何条件行为。\n- 看题任务: 基线通常只有 1/类别数, 模型超过它才说明在读 prompt。\n训练前先算一遍, 当作 "没学会" 的参考线。',
        },
        { title: '塌缩是最短路径', body: '区域奖励下, RL 会飞快把概率压到某个高分 token 上: 熵归零、输出与输入无关, 曲线却很好看。\n奖励不看题时, 这是拿高分最省力的走法, 不是模型偷懒。' },
        { title: '真实 RLVR 天然看题', body: '数学答案比对、单元测试、格式校验都依赖题目本身。自己造玩具任务时要主动保证这一点。\n本仓库的 SeqTask 有 $13^6 \\approx 480$ 万种 prompt, 训练集与留出集按 token 和 mod 5 不相交, 背不下来。' },
      ],
      links: [
        { from: 'SeqTask.verify', to: 'reward_fn(prompts, completions)', body: '奖励函数必须同时看到 prompt 和 completion。只看 completion 的奖励是个 bandit: 常数策略就能拿满分, 检验不出模型有没有在读题。' },
        { from: 'PromptDataGenerator', to: 'SeqTask', body: 'prompt 里编码题目 (copy / reverse / sort 一段序列), 答案由程序算出。' },
        { from: '组内零方差', to: 'DAPO 动态采样', body: '看题任务早期常见"整组全错": 这些组优势全为 0, 不产生梯度, 见 GRPO 进阶一章。' },
        { from: 'verify', to: '训练与验收', body: '同一个判分器既当奖励也当留出集指标。前提是它真的依赖 prompt。' },
      ],
      sourceRows: [
        { concept: '区域奖励 (反例)', code: 'reward = (completion >= V//2).mean()', takeaway: '只看 completion 里落在 $[V/2, V)$ 的比例, 与 prompt 无关。本仓库早期版本用过, 已换掉。' },
        { concept: '看题奖励', code: 'llm_finetune/data/tasks.py:SeqTask', takeaway: 'verify(prompts, completions): 前 R 个 token 与 target(prompts) 完全一致 (含 EOS) 才得 1 分。' },
        { concept: '判分器', code: 'llm_finetune/data/tasks.py:verify', takeaway: '0/1 判分, 不给部分分。任务太难时"整组全错"的比例会很高, 这时要么降难度、要么增大 G、要么给部分分。' },
        { concept: '留出集怎么切', code: 'llm_finetune/data/tasks.py:sample_prompts', takeaway: '留出集 = token 和 $\\equiv 0 \\pmod{5}$ 的 prompt, 与训练集严格不相交。' },
        { concept: '评估', code: '按 prompt 分桶的准确率', takeaway: '只看平均 reward 会被常数策略骗过去。' },
      ],
      snippetTitle: '两种 reward_fn 的签名',
      snippet: `def region_reward(completion):                  # 不看题: 常数策略即可满分
    return (completion >= V // 2).float().mean(-1)

def sum_reward(prompt, completion):             # 看题: 答案随 prompt 变
    target = (prompt[:, -2] + prompt[:, -1]) % 10
    return (completion[:, 0] == target).float()

# 训练前的体检: 常数基线
best_const = max(sum_reward(prompts, full_like(c)).mean() for c in range(10))
# region_reward 的 best_const = 1.0  ->  这个任务验证不了条件生成`,
      source: ['llm_finetune/data/tasks.py:SeqTask', 'llm_finetune/data/tasks.py:verify'],
      run: 'python -m llm_finetune.run_finetune.grpo.train_grpo',
    },

    // ───────────────────────── 模型合并 ─────────────────────────
    'finetune-merge': {
      title: '模型合并 · 把 "微调改了什么" 当向量加回基座',
      subtitle: '不重训、不要数据, 几秒钟把两个微调模型拼成一个。读完你能判断两个任务能不能合, 以及 λ 为什么必须调。',
      tldr: '把 "微调改了什么" 当向量 $\\tau = \\theta_{\\text{ft}} - \\theta_0$, 直接加回基座。前提是两个任务改的地方不重叠。\n- 单任务模型: 两个都做不了 "两段一起改写", EM ≤ 0.027。\n- $\\theta_0 + 1\\cdot(\\tau_A + \\tau_B)$: 留出集 EM 1.000。',
      question: '不重训, 能把两个微调模型合成一个吗? TIES、DARE 这些 "更聪明" 的合并, 真的比直接相加强吗?',
      code: 'llm_finetune/methods/merge.py · llm_finetune/run_finetune/merge/train_merge.py',
      points: [
        { title: '为什么要合并', body: '两个团队从同一个基座各微调了一个模型, 一个会 A, 一个会 B。想要两样都会:\n- 重训: 两份数据合起来再训, 要数据、要算力。\n- 部署两份: 两份权重都上线。\n- 合并: 只要权重, 不训练, 几秒钟。\n本仓库的 A 把低段 token 3–8 各 +1, B 把高段 10–15 各 −1。单任务模型做两段一起改写, EM 只有 0.004 / 0.027。' },
        { title: '合并 = 对任务向量做加法', body: '任务向量 $\\tau_t = \\theta_t - \\theta_0$。Task Arithmetic: $\\theta = \\theta_0 + \\lambda\\sum_t \\tau_t$。\n- λ=1: 两段全对 EM 1.000, 干扰 0.000。\n- λ=0.5: 就是简单平均 $(\\theta_A + \\theta_B)/2$。每个任务向量只加了一半, 两段都只改到一半, EM 0.039。\nλ 在验证集上从 {0.5, 0.7, 1, 1.5} 里挑, 表里报的是留出集。' },
        {
          title: '调好 λ 的相加, 没被 TIES / DARE 赢过',
          key: true,
          body: '每种方法各自在验证集上挑过 λ, 留出集两段全对 EM:\n- Task Arithmetic: 1.000\n- DARE p=0.5: 0.875\n- TIES d=0.5: 0.734\n- DARE p=0.9: 0.066。论文里 "丢 90% 几乎无损" 在这里不成立。\n- SLERP t=0.5: 0.090, ≈ 简单平均。\n换种子 1、2: Task Arithmetic 0.973 / 0.617, 仍是最高。\nTIES 要修的是大模型里大量小幅、冗余坐标的冲突。这里 τ 只有约 10 万参数, 修掉一半就伤到了有用的坐标。',
        },
        { title: '能合并, 是因为任务这么设计', body: '相加能成立, 前提是两个任务在函数上几乎不重叠。A 的训练数据里高段照抄, 所以 $\\tau_A$ 只被要求改 "低段怎么输出"。\n调参时试过两种不满足前提的设置:\n- 同一输入要两种输出 (reverse vs sort): 所有方法合并后都不明显好于基座。\n- 按输入分段的 ±1 平移: 两个 τ 学成全局的 +1 / −1, 相加互相抵消。3 个种子 EM 0.01–0.68。\n边界: 只合并了 2 个任务。任务更多时冲突更多, TIES 的价值才可能显出来, 这里没测。' },
      ],
      links: [
        { from: 'LoRA merge', to: '任务向量', body: 'LoRA 的任务向量就是合并后的 $\\Delta W = (\\alpha/r)BA$。merge.py 只处理 state_dict, 不碰模型结构, 两种都能合。' },
        { from: 'cos(τ_A, τ_B)', to: 'sign_conflict', body: '$\\cos = +0.046$, 几乎正交。但两者都非零的坐标里, 符号冲突有 47.4% (随机符号是 50%)。\ncos 看的是整体, 冲突看的是逐坐标。' },
        { from: 'SLERP', to: '简单平均', body: '插的是完整权重 $\\theta_A$、$\\theta_B$。两者共享 $\\theta_0$, 夹角 $\\Omega = 0.333$ rad。t=0.5 时只比线性平均放大 $1/\\cos(\\Omega/2) = 1.014$ 倍。' },
        { from: '同一个 θ₀', to: 'task arithmetic', body: '$\\tau$ 的定义就要求同一个基座。从不同基座或不同随机初始化出发的两个模型, 相减没有意义。' },
      ],
      sourceRows: [
        { concept: '任务向量', code: 'llm_finetune/methods/merge.py:task_vectors', takeaway: '$\\tau_t = \\theta_t - \\theta_0$, 只取浮点张量; 整型 buffer 没有 "方向" 可言。' },
        { concept: '相加', code: 'llm_finetune/methods/merge.py:task_arithmetic', takeaway: '$\\theta_0 + \\lambda\\sum_t\\tau_t$。除 SLERP 外, 所有方法都只是先对 τ 做逐坐标处理, 再走这一步。' },
        { concept: 'TIES', code: 'llm_finetune/methods/merge.py:ties_merge', takeaway: '- 修剪: 每个 τ 只留 $|\\cdot|$ 最大的 density 比例。\n- 选符号: $\\gamma = \\mathrm{sign}(\\sum_t\\hat\\tau_t)$, 按幅度投票。\n- 不相交平均: 每个坐标只对非零且与 γ 同号的任务取平均。\n脚本开头用 [3, −1, 0.1] 和 [−2, −2, 0.1] 手算验过: 结果 [3, −1.5, 0]。' },
        { concept: 'DARE', code: 'llm_finetune/methods/merge.py:dare', takeaway: '每个坐标以概率 p 置零, 其余 $\\times 1/(1-p)$, 期望不变。不除的话, 等于把 λ 缩成 $(1-p)$ 倍。' },
        { concept: 'SLERP', code: 'llm_finetune/methods/merge.py:slerp', takeaway: '逐张量球面插值; $\\sin\\Omega \\to 0$ 时退化为线性插值。' },
        { concept: '干扰度量', code: 'llm_finetune/methods/merge.py:sign_conflict', takeaway: '两者都非零的坐标里, 符号相反的比例。' },
        { concept: '两个任务', code: 'llm_finetune/run_finetune/merge/train_merge.py:EditTask', takeaway: 'copy, 但把指定 token 平移。A 的数据里高段照抄, 反之亦然, 所以两个 τ 各管一段。' },
      ],
      snippetTitle: '四种合并, 都是对 τ 的逐坐标操作',
      snippet: `tau = [ft - base for ft in (theta_A, theta_B)]            # 任务向量
merged = base + lam * (tau[0] + tau[1])                     # Task Arithmetic; lam=0.5 即平均

# TIES: 修剪 → 选符号 → 不相交平均
t = [keep_topk(abs, density)(x) for x in tau]
sign = sign(t[0] + t[1])                                    # 按幅度投票
agree = [(sign(x) == sign) & (x != 0) for x in t]
merged = base + lam * sum(x * a for x, a in zip(t, agree)) / sum(agree).clamp(min=1)

# DARE: 随机丢 p, 其余放大 1/(1-p), 再做 Task Arithmetic
tau = [x * (rand_like(x) >= p) / (1 - p) for x in tau]`,
      source: [
        'llm_finetune/methods/merge.py:task_vectors',
        'llm_finetune/methods/merge.py:task_arithmetic',
        'llm_finetune/methods/merge.py:ties_merge',
        'llm_finetune/methods/merge.py:dare',
        'llm_finetune/methods/merge.py:slerp',
        'llm_finetune/run_finetune/merge/train_merge.py:EditTask',
      ],
      run: 'python -m llm_finetune.run_finetune.merge.train_merge',
    },

    // ───────────────────────── RLAIF ─────────────────────────
    'finetune-rlaif': {
      title: 'RLAIF · 按宪法让 AI 批评、改写, 自动造偏好对',
      subtitle: '不找人标注也能造偏好对。读完你能说清它省掉了什么, 又把什么原封不动交给了 judge。',
      tldr: '让 AI 按宪法批评、改写, 自动造偏好对。标签质量的上限就是 judge 的质量。\n- 流程: 4096 条采样 → judge 批评 2539 条 → 2539 个偏好对 → 现成 DPO 150 步。\n- judge 查的: 违规率 0.648 → 0.031。\n- 宪法写了、规则没查的变体 14: 只从 0.418 降到 0.320。',
      question: '不找人标注, 能对齐吗? 条文写了但规则没查, 会怎样?',
      code: 'llm_finetune/methods/rlaif.py · llm_finetune/run_finetune/rlaif/train_rlaif.py',
      points: [
        { title: '人是瓶颈', body: 'RLHF 的偏好标签来自人: 贵、慢, 标注员之间还不一致。有害内容也得有人一条条看。\nConstitutional AI 的做法:\n- 写条文: 先把 "什么是好回答" 写成条文。\n- AI 反馈: AI 反馈者按条文批评、改写。\n- 成对: 改写前后两个版本天然就是一对, 改写后 ≻ 改写前。\n- 训练: 之后照旧用 DPO。' },
        { title: '一轮怎么走', body: '- 采样: 起点是 "只求有用" 的模型, 什么都照抄, 包括脏话 15 和电话号码 13。温度 1 采 4096 条。\n- 批评: judge 逐条原则检查, 2539 条被指出问题。\n- 改写: 按批评把 15、13 打码成 3 (***)。\n- 成对: 被批评的才成对。没问题的回复没有 "更好的版本", 直接丢掉。' },
        {
          title: 'judge 没查的, DPO 学不到',
          key: true,
          body: '宪法条文写着 "变体 14 也算脏话", judge 的规则只查了 15。于是 14 在 chosen 和 rejected 里同时出现, 训练对它没有直接压力。\n- 实测: 0.418 → 0.320。\n- 降的那点: 来自 DPO 的副作用 (首个回复 token 被带偏, 偶尔把 14 也打了码), 不是学会了原则。\n条文是给人看的, 生效的是规则。换成 LLM judge, 漏检不会消失, 只是更难被发现。',
        },
        { title: 'judge 的偏差也会被学走', body: '调参时加过第三条 "不啰嗦: 不要连说两遍"。改写就是删掉重复, 所以 chosen 总比 rejected 短。\nDPO 学到的是 "早点说 EOS", 模型只说 1 个 token 就结束:\n- 违规: 0.605 → 0.000\n- 理想回复: 0.395 → 0.000\n- 那组设置: lr=3e-4、200 步、1024 条采样\n最终版只保留不改长度的原则。' },
        { title: '有用性要一起看', body: '只看违规率会被 "什么都不说" 骗过去。\n- 非敏感位置照抄率: 1.000 → 0.856, 错误集中在第一个回复 token (DPO 的老毛病)。\n- 整条 = 理想回复 (照抄 + 打码): 0.352 → 0.488。\n- lr 从 1e-4 提到 3e-4: 理想回复掉到 0.051。' },
      ],
      links: [
        { from: 'CONSTITUTION', to: 'critique / revise', body: '每条原则 = (名字, 条文, 检查函数, 修改函数)。条文给人看, 训练信号只来自后两个函数。' },
        { from: 'build_preference_pairs', to: 'DPOLoss', body: '输出格式与 PreferenceDataGenerator 相同, 直接接 PairwiseForward 和现成的 DPOLoss, 一行不改。' },
        { from: 'DPO 的 likelihood displacement', to: 'RLAIF 的有用性损失', body: '照抄率掉在第一个回复 token 上, 和 DPO 一章 log π(chosen) 下降是同一件事, 不是 RLAIF 特有的。' },
        { from: '一轮离线 DPO', to: '真实 RLAIF', body: '真实做法和这里的差别 (这里都没做):\n- 轮数: 常迭代多轮, 或用 AI 偏好训 RM 再做 PPO。\n- 生成: 批评和改写也由 LLM 生成, 改写本身会出错。' },
      ],
      sourceRows: [
        { concept: '宪法', code: 'llm_finetune/methods/rlaif.py: CONSTITUTION', takeaway: '「无害」不说 15 (条文还写了 14), 「隐私」不复述 13, 都要求打码成 3。规则只查 15 和 13。' },
        { concept: '批评', code: 'llm_finetune/methods/rlaif.py:critique', takeaway: '逐条原则检查, 返回 "违反「无害」: 位置 [3, 5]" 这样的批评。空列表 = 没问题。' },
        { concept: '改写', code: 'llm_finetune/methods/rlaif.py:revise', takeaway: '依次套用每条原则的 fix, 长度补 PAD 回 R。示例: [4, 8, 12, 15, 13, 15] → [4, 8, 12, 3, 3, 3]。' },
        { concept: '造偏好对', code: 'llm_finetune/methods/rlaif.py:build_preference_pairs', takeaway: '只对被批评的回复造 (chosen = revise(y), rejected = y)。EOS 之后的 token 先清成 PAD, 不算回复。' },
        { concept: '离线池', code: 'llm_finetune/methods/rlaif.py:AIPreferenceData', takeaway: '偏好对只造一次, DPO 每步从池子里随机取 64 对。' },
        { concept: '验收', code: 'llm_finetune/run_finetune/rlaif/train_rlaif.py:evaluate', takeaway: '留出 prompt 贪心回复: judge 的违规率 + judge 看不见的变体率 + 照抄率 + 整条 = 理想回复。' },
      ],
      snippetTitle: '一轮 RLAIF',
      snippet: `y = policy.generate(x, temperature=1.0)            # 1. 只求有用的模型照抄一切
pairs = []
for xi, yi in zip(x, y):
    if critique(yi):                                  # 2. judge 按宪法逐条检查
        pairs.append((xi, revise(yi), yi))            # 3. 改写后 ≻ 改写前
    # 没问题的回复没有更好的版本, 丢掉

fit(PairwiseForward.with_frozen_copy(policy),         # 4. 现成的 DPO
    AIPreferenceData(pairs), DPOLoss(beta=0.5), steps=150, lr=1e-4)`,
      source: [
        'llm_finetune/methods/rlaif.py:critique',
        'llm_finetune/methods/rlaif.py:revise',
        'llm_finetune/methods/rlaif.py:build_preference_pairs',
        'llm_finetune/methods/rlaif.py:AIPreferenceData',
        'llm_finetune/run_finetune/rlaif/train_rlaif.py:evaluate',
      ],
      run: 'python -m llm_finetune.run_finetune.rlaif.train_rlaif',
    },

    // ───────────────────────── PRM ─────────────────────────
    'finetune-prm': {
      title: 'PRM · 逐步打分, 而不只看最终答案',
      subtitle: '同样的解、同样的训练步数, 只把标签从 "答案对不对" 换成 "每一步对不对"。看它能多挑对几道题, 能不能指出错在哪一步。',
      tldr: '同样的解, 逐步标签比最终标签好学得多, 还能指出第一个错步。4 步模 10 算术链, 训练 600 步:\n- BCE: ORM 停在 0.682 (ln 2 = 0.693, 几乎没学), PRM 降到 0.359。\n- best-of-8 挑解: 随机 0.473 / ORM 0.488 / PRM 0.648。',
      question: '只看最终答案的奖励模型差在哪? 过程错、答案蒙对的解, 谁会被骗?',
      code: 'llm_finetune/methods/prm.py · llm_finetune/run_finetune/prm/train_prm.py',
      points: [
        { title: 'ORM 得自己猜错在哪', body: '一条 4 步的解, ORM 只拿到 1 个 "最终答案对不对"。答案错了, 它得自己从这一个 0/1 里推出是哪一步坏的。信号稀疏, 学得慢。\n过程错、答案蒙对的解 (候选里占 4.5%), 在 ORM 的训练数据里就是 "好"。' },
        { title: 'PRM 每步一个标签', body: '- 标签数: N 条解, ORM 拿到 N 个标签, PRM 拿到 $N\\cdot K$ 个。\n- 局部判断: 每个标签只问 "这一行算式对不对"。\n第 $i$ 步 "对" = $v_i = \\mathrm{op}_i(v_{i-1}, k_i)$, 其中 $v_{i-1}$ 是解里写的值。\n前面错了、这一步照着错的值算对了, 这一步仍然算对。像批改真人的草稿。' },
        {
          title: '同样预算, 差距来自标签粒度',
          key: true,
          body: '同一个主干 + 标量头、同一初始化、同一串训练数据, 留出集 512 题 × 8 个候选:\n- best-of-8 答案对: 随机 0.473 / ORM 0.488 / PRM 0.648\n- best-of-8 每步都对: 0.426 / 0.422 / 0.627\n- 定位第一个错步: PRM 0.575, 最好的常数猜法 "总说第 1 步错" 0.341。ORM 只有一个分, 无从定位。',
        },
        { title: '解分取 min', body: 'PRM 给整条解打分, 取各步概率的最小值。一步错, 整条就错; 取平均会被其余对的步稀释。' },
        { title: '诚实边界', body: '- 候选: 来自带噪的程序求解器 (每步 20% 写错), 不是 LM 采样, 错误分布比真实简单。\n- ORM 学得慢, 不是学不会: 脚本外另跑同配置, 约 2000 步才开始降, 3000 步 BCE ≈ 0.55。这个数不在脚本里, readme 已注明。\n- 标签成本: 逐步标签由程序判, 不花钱。真实 PRM 的主要成本正是标签, 这里体现不了。\n- PRM 也没训满: 8 个候选里至少一个答案对的比例是 0.992, PRM 只挑中 0.648。' },
      ],
      links: [
        { from: 'RewardModel', to: 'PRM / ORM', body: '同一个 "主干 + 标量头", token_scores 逐位置取分。差别只在 BCE 落在哪几个位置:\n- ORM: 落在 EOS。\n- PRM: 落在 4 个 $v_i$。' },
        { from: 'PRM 解分', to: 'infer · 推理时计算', body: '推理阶段的 best-of-N 与 PRM beam 用的是程序化 PRM。这一章讲的是怎么把它训出来。' },
        { from: 'PRM 的逐步分', to: 'PPO 的逐 token 优势', body: '逐步分可以当稠密奖励, 正好给 critic 和 GAE 做逐 token 的信用分配, 见下一章。' },
        { from: 'step_positions', to: 'v_i 所在位置', body: '第 $i$ 步的分取自 $v_i$ 所在位置 (读完 $v_i$ 之后)。取在前一个位置, 模型还没看到 $v_i$, 判断不了它对不对。' },
      ],
      sourceRows: [
        { concept: '玩具数据', code: 'llm_finetune/methods/prm.py:ArithChain', takeaway: '题 a0 op1 k1 … op4 k4 SEP, 解每步写成 "op k 结果"。留出集 = $(a_0 + \\sum k) \\equiv 0 \\pmod 5$ 的题。' },
        { concept: '逐位置取分', code: 'llm_finetune/methods/prm.py:token_scores', takeaway: 'RewardModel.forward 只取末位置; 这里要每个位置一个 logit。' },
        { concept: 'PRM loss', code: 'llm_finetune/methods/prm.py:prm_loss', takeaway: '在 4 个 $v_i$ 位置上做 BCE, 标签是 step_ok。' },
        { concept: 'ORM loss', code: 'llm_finetune/methods/prm.py:orm_loss', takeaway: '只在 EOS 位置做一次 BCE, 标签是 outcome_ok。' },
        { concept: '整条解的分', code: 'llm_finetune/methods/prm.py:solution_scores', takeaway: '- PRM: 各步概率取 min。\n- ORM: 末位置概率。\nbest-of-8 取分最高的那条。' },
      ],
      snippetTitle: 'ORM 与 PRM 只差 BCE 落在哪',
      snippet: `h = rm.backbone(seqs)                          # [N, T, D]
s = rm.value_head(h).squeeze(-1)               # [N, T] 每个位置一个 logit

orm = bce(s[:, eos_pos], outcome_ok)           # 每条解 1 个标签
prm = bce(s[:, step_pos], step_ok)             # 每条解 K 个标签

score_prm = sigmoid(s[:, step_pos]).min(1)     # 一步错整条错: 取短板
best = score.view(n, 8).argmax(1)              # best-of-8`,
      source: [
        'llm_finetune/methods/prm.py:ArithChain',
        'llm_finetune/methods/prm.py:prm_loss',
        'llm_finetune/methods/prm.py:orm_loss',
        'llm_finetune/methods/prm.py:solution_scores',
      ],
      run: 'python -m llm_finetune.run_finetune.prm.train_prm',
    },

    // ───────────────────────── PPO ─────────────────────────
    'finetune-ppo': {
      title: 'PPO · GRPO 省掉的 critic 到底买到了什么',
      subtitle: '把 critic 装回去, 同样的采样预算比一比。读完你能手算 GAE, 也能说清在什么规模下 critic 值不值它的显存。',
      tldr: 'critic 是学出来的 baseline。学不好时, 它就是个昂贵的常数。\n同一个 sort 任务、每步 256 条回复、60 步, 留出集 pass@1:\n- PPO: 0.381\n- GRPO: 0.287\n- critic 冻住 (lr=0): 0.389\n高出的那截来自 "一题一采", 不是 critic。',
      question: 'GRPO 省掉的 critic, 到底买到了什么?',
      code: 'llm_finetune/methods/ppo.py · llm_finetune/run_finetune/ppo/train_ppo.py',
      points: [
        { title: '为什么要 baseline', body: '策略梯度拿回报当权重, 方差很大。减一个 baseline $b$ 不改期望梯度, 只降方差。\n- GRPO: $b$ = 同题 G 条的均值。免费, 但必须一题多采, 整条回复共用一个 $A$。\n- PPO: $b$ = critic 学出来的 $V(s_t)$。一题采 1 条就够, 每个 token 各有一个 $A_t$。代价是多养一个和 policy 同尺寸的网络。' },
        { title: 'GAE 在 TD(0) 和 MC 之间插值', body: '$\\delta_t = r_t + \\gamma V(s_{t+1}) - V(s_t)$, $A_t = \\sum_k (\\gamma\\lambda)^k \\delta_{t+k}$。\n- λ=0: $A_t = \\delta_t$, 只信下一步的 $V$。方差小, $V$ 不准时有偏。\n- λ=1: 回报目标 $= \\sum_k \\gamma^k r_{t+k}$, 只用真实回报。无偏, 方差大。\n脚本单测两端: λ=0 与 TD(0) 误差 0.0, λ=1 与 MC 回报误差 4.8e-7。' },
        { title: '奖励落在哪', body: 'verifier 分 (排序全对得 1) 只落在末 token。每个 token 再扣 $\\beta(\\log\\pi_{\\text{old}} - \\log\\pi_{\\text{ref}})$。\n- PPO: 沿用 InstructGPT 的做法, KL 写进奖励, 会被 critic 学、被 GAE 累加。\n- GRPO: 把 KL 直接加在 loss 上。' },
        {
          title: '玩具规模上 critic 几乎没用',
          key: true,
          body: '- 解释方差: $V(s_0)$ 只解释 2.5% 的回报方差, 免费的组均值解释 36.5%。"方差更小" 在这里不成立。\n- 冻结对照: critic 的 lr 设成 0, pass@1 0.389, 和 PPO 的 0.381 在噪声内。\n- 高出 GRPO 0.09 的来源: PPO 是 256 道不同的题各采 1 条。GRPO 的 32 道题里约 32% 的组全对或全错, $A\\equiv 0$, 白占预算。\n- 逐 token 信用分配: 在 60 步、64 维的 critic 上也没显出来。\n边界: 大模型上 critic 从 RM 或 SFT 初始化并先预热, 结论可能不同。',
        },
        { title: '代价是实打实的', body: '- 要训的参数: 99,648 → 199,360 (×2)\n- 常驻参数: 99,648 → 299,008 (×3: policy + ref + critic)\n- 60 步耗时: 本机两次实测 GRPO 5.7–6.1 s, PPO 11.8–12.0 s, 约 ×2\n这就是 DeepSeek 用 GRPO 去掉 critic 的动机。' },
      ],
      links: [
        { from: 'RewardModel', to: 'critic', body: 'critic 就是 RM 的主干 + 标量头, 从 SFT 权重起步。token_values 取每个位置的分。位置 $P-1+t$ 读完了 prompt 与 $o_{\\lt t}$, 正是 $V(s_t)$。' },
        { from: '组均值', to: 'critic', body: '两者都是 baseline。\n- 组均值: G=1 时就是样本自己, $A\\equiv 0$, 所以 GRPO 必须一题多采。\n- critic: 与组无关, 一题一采也有信号。' },
        { from: 'PPO clipped surrogate', to: 'GRPO 进阶', body: '目标外形一样: ratio + clip, 同一批 rollout 更新 μ=2 次。只是 $A$ 的来源不同。' },
        { from: 'PRM 逐步分', to: '逐 token 优势', body: '有了逐步奖励, GAE 才有东西可分。只有末 token 一个分时, critic 得自己学出 "哪个前缀已经没救了"。' },
      ],
      sourceRows: [
        { concept: '逐位置估值', code: 'llm_finetune/methods/ppo.py:token_values', takeaway: '复用 RewardModel, 取位置 $P-1+t$ 的分当 $V(s_t)$, 与 completion_logprobs 同一个错位。' },
        { concept: 'GAE', code: 'llm_finetune/methods/ppo.py:gae', takeaway: '从后往前一次扫描。mask 外视为终止, $r = V = 0$。否则 GAE 会从 EOS 之后的垃圾 token 上自举。' },
        { concept: '逐 token 奖励', code: 'rewards[arange(N), last] += score', takeaway: 'verifier 分只落在末 token; 其余 token 只有 $-\\beta\\cdot$KL。' },
        { concept: '一步', code: 'llm_finetune/methods/ppo.py:PPOTrainer', takeaway: 'rollout → 冻结 old_logp / ref_logp / old_values → GAE → 批内白化 → μ=2 轮 clip + value loss。policy 与 critic 各一个优化器。' },
        { concept: '两端单测', code: 'llm_finetune/run_finetune/ppo/train_ppo.py:check_gae', takeaway: '4 条长度不等的随机序列, γ=0.9: λ=0 与 TD(0) 误差 0.0e+00, λ=1 与 MC 回报误差 4.8e-07。' },
        { concept: '解释方差', code: 'llm_finetune/run_finetune/ppo/train_ppo.py:baseline_explained_var', takeaway: '$1 - \\mathrm{Var}(R-b)/\\mathrm{Var}(R)$, 留出集 64 题 × 8 条。组均值包含样本自己, 这个数略偏乐观。' },
      ],
      snippetTitle: 'PPO 一步 (骨架)',
      snippet: `seqs = policy.generate(prompts)                      # 每题只采 1 条
with no_grad():
    old_logp, ref_logp = logp(policy, seqs), logp(ref, seqs)
    V = critic_values(critic, seqs)                  # [N, C]  V(s_t)
r = -beta * (old_logp - ref_logp)                    # KL 写进奖励
r[arange(N), last] += verifier(prompts, seqs)        # 分数只落在末 token
A, G = gae(r, V, mask, gamma=1.0, lam=0.95)          # 逐 token 优势 + 回报目标
A = (A - A.mean()) / A.std()                         # 批内白化

for _ in range(2):                                   # 同一批更新 μ=2 次
    ratio = (logp(policy, seqs) - old_logp).exp()
    pi_loss = -min(ratio * A, ratio.clamp(0.8, 1.2) * A).mean()
    v_loss = 0.5 * ((critic_values(critic, seqs) - G) ** 2).mean()
    (pi_loss + v_loss).backward(); opt.step(); critic_opt.step()`,
      source: [
        'llm_finetune/methods/ppo.py:gae',
        'llm_finetune/methods/ppo.py:PPOTrainer',
        'llm_finetune/methods/ppo.py:token_values',
        'llm_finetune/run_finetune/ppo/train_ppo.py:check_gae',
      ],
      run: 'python -m llm_finetune.run_finetune.ppo.train_ppo',
    },

    // ───────────────────────── 压轴: 训练脚本与选型 ─────────────────────────
    'finetune-runs': {
      title: '训练脚本与落盘 · 11 种方法的代价结构',
      subtitle: '在"有什么数据 / 有多少显存 / 能不能在线采样"之后, 直接点名方法。还要说出它每步几次前向、常驻几份权重、最后落盘什么。',
      tldr: '方法之间的区别不在 loss 写得好不好看, 在代价结构:\n- 要不要常驻 reference model\n- 要不要在线采样\n- 落盘是全量权重还是 adapter\n实测 DPO 每步 2 次前向 / 常驻 778 KB / 5.4 s; SimPO 与 ORPO 1 次 / 389 KB / 4.0 s。',
      question: '手上只有 (问, 答) 且显存紧张, 该用哪个? 换成成对偏好呢? 换成"答案能被程序判对错"呢?',
      code: 'llm_finetune/run_finetune/{sft,lora,dora,qlora,rm,dpo,simpo_orpo,grpo,distill,on_policy_distill}/train_*.py',
      points: [
        { title: '两个维度可以自由组合', body: '- SFT / DPO / GRPO: 决定 "优化什么目标"。\n- LoRA / QLoRA / DoRA: 决定 "更新哪些参数"。\n两者互相正交, LoRA-SFT、LoRA-DPO 都很常见。\n代码上: methods/ 放可复用算法, run_finetune/ 放一次实验的编排。' },
        {
          title: '代价写在三处',
          key: true,
          body: '- 每步 LM 前向次数: DPO 2 次, SimPO / ORPO 1 次, GRPO 系还要先采 G 条。\n- 常驻权重份数: DPO 要 policy + ref = 778 KB, 无 ref 的只要 389 KB。\n- 落盘产物: 全量权重 / adapter / value head。\n数据形态先砍掉一半候选, 这三项再砍一半。',
        },
        { title: '别只存 merge 后的权重', body: 'LoRA adapter 只有 $r\\cdot(d_{\\text{in}}+d_{\\text{out}})$ 个参数: 本仓库 76 KB, 整模型 389 KB。\n- 存 adapter: 能独立分发、按请求热切换。\n- 只留 merge 后的完整权重: 退化成每个任务一份大模型。' },
      ],
      links: [
        { from: 'methods/*.py', to: 'run_finetune/*/train_*.py', body: '算法类被训练脚本实例化并喂 batch; 脚本还负责留出集评估和最后那条断言。' },
        { from: 'PairwiseForward / TeacherStudent', to: '通用 Trainer', body: '一步要跑多次前向的方法 (DPO / SimPO / ORPO / RM / 蒸馏) 都包成一个 nn.Module, Trainer 一行不用改。' },
        { from: 'GRPO / on-policy 蒸馏', to: '不用 Trainer', body: '数据由当前策略现场采样, GRPO 还要在同一批 rollout 上更新 $\\mu$ 次。"取 batch → 更新一次"的约定不成立, 硬塞只会更难读。' },
        { from: 'print_trainable_parameters', to: '验收 PEFT', body: 'loss 下降和显存占用都证明不了 base 被冻住; 统计 requires_grad 的参数量才是直接证据。' },
        { from: 'adapter state_dict', to: 'llm_infer Multi-LoRA', body: '多租户服务靠 adapter 能独立切换。' },
      ],
      sourceRows: [
        { concept: '共用基座', code: 'llm_finetune/run_finetune/common.py:pretrained_base', takeaway: '所有脚本从同一个 copy 任务的基座出发, 指标之间才可比。' },
        { concept: 'SFT / LoRA / DoRA / QLoRA', code: 'run_finetune/{sft,lora,dora,qlora}/train_*.py', takeaway: '数据是 (x, y), 每步 1 次前向, 不要 ref 也不要采样; 区别只在"哪些参数可训、基座怎么存"。' },
        { concept: 'RM', code: 'run_finetune/rm/train_rm.py', takeaway: '同一份偏好数据, 学的是标量分。落盘多一个 value head; 换掉 lm_head 时会原地改掉传入的 backbone。' },
        { concept: 'DPO', code: 'run_finetune/dpo/train_dpo.py', takeaway: 'policy + ref 双份权重, 每步 2 次前向 (2B 条序列)。' },
        { concept: 'SimPO / ORPO', code: 'run_finetune/simpo_orpo/train_simpo_orpo.py', takeaway: '同样的数据, 去掉 ref: 常驻权重减半、前向减半, 实测每步快约 25%。' },
        { concept: 'GRPO 系', code: 'run_finetune/grpo/train_grpo.py', takeaway: '只要 prompt, 但必须有 verifier 或 RM, 且必须能在线采样 (G 条/prompt)。$\\beta>0$ 时才额外常驻一份 ref。' },
        { concept: '蒸馏', code: 'run_finetune/{distill,on_policy_distill}/train_*.py', takeaway: '都要一个更强的 teacher 常驻; 离线版从语料取数, on-policy 版由 student 现场采样。' },
        { concept: '参数统计', code: 'llm_finetune/utils/param_utils.py:print_trainable_parameters', takeaway: '打印可训 / 总参数与占比, 用来验收"真的只训了 adapter"。' },
      ],
      snippetTitle: '读一个 run_finetune 脚本, 盯这 5 行',
      snippet: `1. dataset / collate_fn 产出哪些字段?
2. labels 中哪些位置是 -100?
3. 哪些参数 requires_grad=True?
4. loss 需要 policy/ref/chosen/rejected 哪些输入?
5. checkpoint 保存 full model 还是 adapter?`,
      source: [
        'llm_finetune/run_finetune/common.py:pretrained_base',
        'llm_finetune/utils/param_utils.py:print_trainable_parameters',
      ],
      run: 'python -m llm_finetune.run_all',
    },
  },
}
