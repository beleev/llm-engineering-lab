import { canonicalizeRelated } from '../../utils/consolidate.js'
// 阶段 1 · llm_basic: 四个章节的完整页面定义。章节目录登记在 data/models.js 的 basicChapters, 所以这里 chapters 为空。
const topic = {
  stage: 'basic',
  chapters: [],
  pages: {
    'basic-data': {
      widgets: ['BpeLab'],
      title: '数据与 tokenizer · 把文本变成可训练张量',
      subtitle: '神经网络只接收数字张量。本章拆解原始文本如何映射为固定词表下的整数序列, 并通过错位切片构造自回归训练批次。',
      tldr: '- 训练前运行一次 prepare.py: 建立包含 65 个字符的词表 (存入 meta.npz), 将 input.txt 按 9:1 顺序切分并编码为 train.bin 与 val.bin。\n- 训练阶段直接通过 memmap 按需读取二进制 token 序列, 生成阶段复用同一份词表把预测出的整数还原回字符。',
      question: '字符级 tokenizer 一个字母占一个 token, 序列明显偏长; BPE 能把同一句话压缩一半以上, 为什么这里训练主线仍然采用字符级?',
      evolution: {
        title: '词表大小与序列长度的折中',
        subtitle: '根问题: 网络只能处理整数 id, 但自然语言词汇开放: 粒度太粗会遇到未登录词, 粒度太细又会拉长序列。',
        steps: [
          { name: '按词切', pain: '(原点) 网络只对数字做运算, 文本必须先切块并映射为整数 id',
            fix: '按空格切词, 一词一 id; 但语料外的新词与拼写变体都没有对应编号' },
          { name: '字符级', pain: '词表随语料无限膨胀, 未登录词无法编码',
            fix: '按单字符编号: Tiny Shakespeare 仅需 65 个 id, 永不缺词; 代价是一个单词要拆成 5~10 个 token' },
          { name: 'BPE', year: 2016, pain: '序列变长后, 注意力开销按长度平方 $O(T^2)$ 增长, 模型还得从头学拼写',
            fix: '迭代合并最高频的相邻符号对: 示例句子可从 60 个字符压到 24 个 token' },
          { name: 'Byte-level BPE', year: 2019, pain: '从字符表起步合并时, 遇到训练语料外的生僻字符仍会丢字',
            fix: '改从 256 个基础字节起步合并: 任意 UTF-8 文本均可无损编解码, bpe.py 演示的就是这一路线' },
        ],
      },
      code: 'llm_basic/{prepare.py,tokenizer.py,bpe.py,input.txt,train.bin,val.bin,meta.npz}',
      points: [
        {
          title: '词表在预处理阶段固定, 训练与采样严格共用',
          body: 'tokenizer 负责在文本字符与整数 id 之间做双向映射, 这张映射表在预处理阶段生成后即固定不变。本模块的训练主线采用字符级分词, 构建词表只需一行 sorted(set(text)): 先对语料出现过的字符去重, 再按字典序排列, 共得到 65 个字符。\n- 显式排序: Python 的 set 遍历顺序不固定, 排序后能保证不同机器生成的 id 映射完全一致。\n- 独立落盘: 词表保存在 meta.npz 中, 而 input.txt 的 1,115,394 个字符被一次性编码为 1,003,854 个训练 token (train.bin) 和 111,540 个验证 token (val.bin); sample.py 解码时加载同一份 meta.npz, 避免模型权重与词表编号错位。\n- 紧凑存储: 词表大小 65 小于 256, 单个 token 用单字节无符号整数 (uint8) 即可存下, prepare.py 中通过 assert vocab_size < 256 做边界校验。',
        },
        {
          key: true,
          title: '训练主线用字符级减负, BPE 拆作独立实验',
          body: '目录中包含两套分词代码, 分工明确:\n- tokenizer.py (65 字符词表): 接入 prepare.py、train.py、sample.py 与自带的 ckpt.npz, 让训练主线聚焦在注意力与手写反向传播上。\n- bpe.py (Byte-level BPE): 直接读取 input.txt 独立演示现代子词合并算法, 不增加主训练脚本的复杂度。\nBPE (Byte Pair Encoding) 从 256 个基础字节出发, 每轮统计相邻 token 对的出现频次, 将最高频的一对合并为新 token。运行 python bpe.py 执行默认的 300 次合并后, 同一句 “First Citizen: Before we proceed any further, hear me speak.” 的编码长度显著缩短:\n- 字符级编码: 60 个 token。\n- BPE 编码: 24 个 token, 平均每个 token 承载 2.50 个字符。\n由于自注意力计算量随序列长度平方增长, 序列缩短一半以上能大幅降低计算与显存开销; 下方实验台完整展示了合并规则的生成与重放过程。',
        },
        {
          title: '监督标签 y 就是输入序列 x 右移一位',
          body: '自回归语言模型的训练目标是根据已有前文预测下一个 token, 监督信号天然包含在连续文本内部, 无需人工标注。\n- get_batch 的切片方式: 从 train.bin 随机抽取 32 个起点 (batch_size=32), 从起点截取 64 个 token (seq_len=64) 作为输入 x, 再从起点加 1 的位置截取 64 个 token 作为标签 y, 使 y[i] 恰好等于 x[i+1]。\n- 单条序列并行提供 T 道预测题: 一次前向传播中, 位置 0 根据 x[0] 预测 y[0], 位置 1 根据 x[0..1] 预测 y[1], 直至位置 63, 一条长度为 64 的样本同时计算 64 个位置的交叉熵损失。\n- 依赖因果掩码防止信息泄漏: 因为答案 y[i] 已经出现在同一条输入的 x[i+1] 位置上, 注意力层必须施加因果 mask 屏蔽所有后续位置, 保证位置 i 只能读取 x[0..i]。',
        },
      ],
      links: [
        { from: 'input.txt', to: 'prepare.py', body: 'prepare.py 将 1,115,394 个字符的莎士比亚语料映射为 65 种整数 id, 按先后顺序切出前 90% 训练、后 10% 验证。不做随机抽样, 避免验证集句子与训练集上下文相邻重叠导致验证指标虚高。' },
        { from: 'train.bin', to: 'train.py:get_batch', body: 'train.bin 是一维连续排列的 uint8 数组, 文件字节数即 token 总数。get_batch 使用 np.memmap 按需映射磁盘切片, 无需把整个语料一次性常驻内存。' },
        { from: 'meta.npz', to: 'sample.py', body: 'meta.npz 保存字符列表以构建反查表 itos (id 到字符)。模型权重记住的是编号间的转移概率, 推理时必须配套创建该 ckpt 时的同一份词表。' },
      ],
      sourceRows: [
        { concept: 'vocab 构造', code: 'prepare.py', takeaway: 'chars = sorted(set(text)) 提取 65 个去重排序后的字符, stoi 构建字符到整数的正向查表, 全程不含可训练参数。' },
        { concept: '训练/验证切分', code: 'prepare.py', takeaway: '按文本先后位置做 90/10 连续切分后再落盘, 确保验证集是一段模型未见过的完整篇章。' },
        { concept: 'uint8 存储', code: 'prepare.py', takeaway: 'assert vocab_size < 256 保证单字节足以容纳全部 id; 现代大模型词表通常在数万级, 需改用 uint16 或 uint32 存储。' },
        { concept: 'batch 对齐', code: 'train.py:get_batch', takeaway: 'x = data[i:i+T], y = data[i+1:i+T+1]。随机起点上界限制为 len−T−1, 为标签末尾预留一格以防越界。' },
        { concept: 'BPE 训练 (独立 demo)', code: 'bpe.py:train_bpe', takeaway: '循环执行三步: 统计相邻对频次、将最高频对替换为新 id、重复 num_merges 轮, 本质是确定性的贪心压缩算法。' },
        { concept: 'BPE 编码', code: 'bpe.py:encode', takeaway: '编码新文本时严格按训练期记录的合并优先级逐轮重放, 保证同一单词始终切分出相同的 token 序列。' },
      ],
      snippetTitle: 'next-token 数据形态',
      snippet: `text:  "hello"
ids:   [h, e, l, l, o]

x:     [h, e, l, l]
y:     [e, l, l, o]

# 位置 i 看到 x[:i+1], 要猜出 y[i]。这就是 next-token prediction`,
      source: ['llm_basic/bpe.py:train_bpe'],
      run: 'cd llm_basic && python prepare.py',
    },

    'basic-forward': {
      title: 'forward 与形状流 · 从 ids 到 logits',
      subtitle: '沿着从整数 id 到词表打分 logits 的前向计算链, 理清每一层的张量形状变化, 以及无自动求导框架时如何显式保存反向所需的中间状态。',
      tldr: '- 前向传播将形状为 [B,T] 的整数序列映射为 [B,T,V] 的词表未归一化打分 logits: 依次经过词与位置嵌入相加、n_layer 层 Pre-LN Transformer 块, 以及末尾归一化与线性投影。\n- 由于不使用 PyTorch autograd, 每个 *_forward 函数在返回输出的同时都会打包一份 cache 元组, 把反向求导必需的中间激活留到 backward 阶段。',
      question: '前向传播算出 logits 就能计算损失了, 为什么每个算子还要额外返回一份 cache?',
      evolution: {
        title: '从定长局部窗口到全序列并行建模',
        subtitle: '根问题: 前文长度不断变化, 模型却要在每个位置输出固定维度 V 的下一词预测分布。',
        steps: [
          { name: 'Bigram 计数表', pain: '(原点) 预测下一字符需要参考前文, 但前文组合数随长度指数爆炸',
            fix: '只看前 1 个字符查 [V,V] 转移表, 更早的上下文信息全部丢失' },
          { name: '窗口 MLP', year: 2003, pain: '把统计窗口扩到 n 个字符时, 计数表膨胀到 $V^n$ 且极其稀疏',
            fix: '用 embedding 将字符映射为 D 维稠密向量, 拼接定长窗口过 MLP 共享统计规律; 但视野仍被窗口长度卡死' },
          { name: 'RNN', pain: '固定窗口外的远距离依赖无法捕获, 加大窗口又令参数量线性膨胀',
            fix: '用定长隐状态按时间步递推压缩历史信息; 代价是 T 个位置只能串行计算, 远距信息易衰减' },
          { name: 'Causal Transformer', year: 2017, pain: 'RNN 逐步串行难以并行加速, 长距离信号需要多步中转',
            fix: '用带因果掩码的自注意力让每个位置直接汇聚所有前文, T 个位置一次并行算完, 词序由 pos_emb 注入' },
        ],
      },
      code: 'llm_basic/model.py:{embedding_forward,rmsnorm_forward,attention_forward,block_forward,transformer_forward}',
      points: [
        {
          key: true,
          title: '显式 cache 承担了计算图中间激活的保存工作',
          body: '训练不仅需要前向算出损失, 还需要反向传播求各层参数的梯度。根据链式法则, 算子的反向公式往往依赖前向阶段的输入或中间结果——例如线性层求权重梯度 $dW = x^\\top \\cdot d\\text{out}$ 需要输入 $x$, RMSNorm 求导需要输入 $x$、缩放参数 $g$ 与均方根 rms, 注意力求导需要归一化后的权重矩阵 attn。\n在深度学习框架中, autograd 会自动在后台计算图里缓存这些张量。手写 numpy 实现时没有自动求导图, 因此每层 *_forward 都显式返回 (out, cache), 由对应的 *_backward 解包使用。\n- 按需缓存: 只保存反向公式真正用到的量。例如 softmax 的导数可直接由其输出 attn 算出, 因此无需再缓存求指数前的 scores。\n- 显存折中: 前向攒下的各层 cache 必须一直驻留内存直到反向逐层消费完毕, 这也是大模型训练激活显存占大头的直接原因。',
        },
        {
          title: '主干张量形状始终围绕 B、T、D、V 四个维度流转',
          body: '整份前向代码的张量形状只涉及四个维度符号: 批大小 B、序列长度 T、隐藏层维度 D (默认 64) 与词表大小 V (65)。\n进入 embedding 之后, 主干残差流在任意层输入和输出处都保持 [B,T,D] 不变, 即每条序列的每个位置始终表示为一个 D 维向量。全模型只有两处临时切换到其他形状:\n- 自注意力相关性矩阵 [B,T,T]: 同一条序列内 T 个位置两两计算缩放点积得分。\n- 输出层打分 [B,T,V]: 最终隐藏状态乘上 [D,V] 的语言模型头, 为每个位置给出词表 V 个候选字符的分数。\n阅读源码时先核对矩阵乘法两侧消掉与保留的维度, 计算脉络就会非常清晰。',
        },
        {
          title: '输入输出同形让堆叠层数退化为简单循环',
          body: '由于每个 Pre-LN Transformer 块接收 [B,T,D] 并输出同样形状的 [B,T,D], 增加网络深度不需要改动任何算子接口, 在 transformer_forward 中用一层 for i in range(num_layers(W)) 依次串联即可。\n这里的层数直接通过扫描参数字典中以 _norm1_g 结尾的键数量动态推断, 无需在 ckpt.npz 里额外维护配置元数据。默认单层配置共 45,568 个参数, 传入 --n-layer 2 即可切换为双层结构。',
        },
      ],
      links: [
        { from: 'learned pos_emb', to: 'RoPE', body: '自注意力本身对排列顺序不敏感, 这里通过直接相加一张 [T_max, D] 的可学习位置表注入词序, 因此推理长度被 T_max=64 硬性限制。阶段 2 改用 RoPE 旋转位置编码, 不占用可学习参数且消除了硬性长度上限 (跨出训练长度后的外推衰减详见 YaRN 章节)。' },
        { from: '单头 attention', to: 'MHA / GQA / MLA', body: '本实现采用单头注意力 (head_dim = D) 以简化维度变换; 阶段 2 的多头注意力 (MHA) 只需将 D 拆分为多组子空间并行计算, GQA 与 MLA 则进一步压缩推理期的 K/V 缓存开销。' },
        { from: 'cache 全存', to: 'activation checkpoint', body: '将全部前向中间激活留在内存中等反向消费会随层数线性推高显存峰值; 阶段 3 的激活重算 (activation checkpointing) 选择只存块边界输入, 反向时局部重跑前向以时间换显存。' },
      ],
      sourceRows: [
        { concept: 'embedding', code: 'model.py:embedding_forward', takeaway: '通过整数索引 W[ids] 取出对应行向量; cache 仅记录 ids 与权重形状, 反向时按索引把梯度累加回原行即可。' },
        { concept: 'RMSNorm', code: 'model.py:rmsnorm_forward', takeaway: '每个位置的 D 维向量除以自身均方根 rms 后乘增益 g; 因 rms 由整行元素共同决定, 反向存在跨元素耦合项, 需同时缓存 (x, g, rms)。' },
        { concept: '缩放点积', code: 'model.py:attention_forward', takeaway: '乘以缩放因子 $1/\\sqrt{D}$ 控制内积方差, 防止 D 增大时分数绝对值过高导致 softmax 饱和、梯度趋近于零。' },
        { concept: 'causal mask', code: 'model.py:causal_mask', takeaway: '构造 [T,T] 上三角掩码并将未来位置 ($j > i$) 置为 -inf, 经 softmax 后权重严格归零, 阻止当前位置读取后续输入。' },
        { concept: '残差', code: 'model.py:block_forward', takeaway: '加法节点 h = x + a 的导数为恒等传递, 反向直接将上游梯度原样分发给两条支路, 无需缓存任何中间张量。' },
        { concept: '层循环', code: 'model.py:num_layers', takeaway: '统计参数名中 _norm1_g 的个数自动识别层数, 让检查点文件保持纯张量字典结构。' },
        { concept: '语言模型头', code: 'model.py:transformer_forward', takeaway: '[B,T,D] @ [D,V] -> [B,T,V] 完成隐藏状态到词表分数的线性投影; 在数万词表的真实模型中, 该层的参数与激活占用尤为可观。' },
      ],
      snippetTitle: '主干形状',
      snippet: `ids[B,T]
  → tok_emb + pos_emb        [B,T,D]
  for i in range(n_layer):               # 默认 1, --n-layer 可调
      → RMSNorm → Attention  [B,T,D]  + residual
      → RMSNorm → MLP        [B,T,D]  + residual
  → final RMSNorm            [B,T,D]
  → lm_head                  [B,T,V]

# 默认 V=65, D=64, H=128, T=64, n_layer=1 → 45,568 个参数`,
      source: ['llm_basic/model.py:transformer_forward'],
      run: 'cd llm_basic && python train.py --max-iters 300 --eval-interval 100 --out /tmp/c.npz',
    },

    'basic-backward': {
      title: '手写 backward · cache、链式法则与 gradcheck',
      subtitle: '从交叉熵损失出发沿计算图反向回传梯度, 拆解多路分支累加与归一化耦合项的求导细节, 并用中心差分验证每一行手写导数。',
      tldr: '- 反向传播从损失对 logits 的解析梯度 dlogits = (p - onehot) / N 起步, 逆序调用各算子的 *_backward, 结合前向保存的 cache 逐层算出输入梯度与参数梯度。\n- 手写导数即使漏掉耦合项或写错下标累加, 只要张量形状匹配就不会抛异常, 因此必须通过 gradcheck.py 与中心有限差分逐算子、端到端比对验证。',
      question: '如果把词嵌入反向里的 np.add.at 误写成 +=, 程序不报错且 loss 依然下降, 该怎样及时揪出这类隐蔽错误?',
      evolution: {
        title: '高效求导与数值校验互补',
        subtitle: '根问题: 每步更新都需要损失对全部 45,568 个参数的偏导数, 既要算得快又要保证公式推导零差错。',
        steps: [
          { name: '数值差分', pain: '(原点) 想知道每个参数微调一点会让 loss 变化多少',
            fix: '逐个参数加减 $\\varepsilon$ 各跑一次前向算斜率; 公式简单不易错, 但单步更新就要跑 2 × 45,568 次前向' },
          { name: '反向传播', year: 1986, pain: '按参数逐一扰动前向计算量过大, 参数稍多便无法训练',
            fix: '利用链式法则从标量 loss 向输入端逐层回传: 1 次前向加 1 次反向即可拿到全部参数梯度' },
          { name: 'forward / backward 成对', pain: '把整座网络的复合导数展开成单一公式过于庞杂, 极易推错',
            fix: '按基础算子拆成独立的 forward/backward 函数对, 通过局部 cache 隔离内部状态' },
          { name: 'gradcheck', pain: '手写反向漏项或重复索引覆盖不会引发运行时报错, 训练曲线仍可能缓慢下降',
            fix: '在小规模随机输入上请回中心差分做基准裁判, 逐算子、端到端核对解析梯度精度' },
        ],
      },
      code: 'llm_basic/model.py:*_backward · llm_basic/gradcheck.py',
      points: [
        {
          key: true,
          title: '形状匹配不代表梯度正确, 必须用中心差分做基准校验',
          body: '手写反向传播最棘手的地方在于静默出错: 矩阵转置写反、求和维度偏置或归一化漏掉耦合项时, 只要数组形状恰好对齐, 代码就不会报错, 优化器拿到的近似方向甚至仍能让 loss 缓慢下降。\n为此需要引入不依赖链式法则推导的中心有限差分作为黄金参照: 对待测标量元素 $w$ 分别施加 $+\\varepsilon$ 与 $-\\varepsilon$ 扰动各跑一次前向, 用 $\\tfrac{f(w+\\varepsilon) - f(w-\\varepsilon)}{2\\varepsilon}$ 估计真实导数, 其截断误差为 $O(\\varepsilon^2)$。\ngradcheck.py 将手写解析梯度与数值差分逐元素对照。例如故意删去 rmsnorm_backward 中的均值耦合项后, 单算子校验会立即检出 3.11e-01 的相对误差, 而正确实现下误差稳定在 1e-10 量级。',
        },
        {
          title: '前向出现分叉或重复索引时, 反向梯度必须累加',
          body: '根据多元复合函数的链式法则, 同一个变量在前向计算中被多条路径读取时, 其反向梯度等于各条支路回传梯度之和。代码中共有三处典型汇合点:\n- Q/K/V 投影分叉: 同一个归一化输出 x 同时乘三组权重生成 Q、K、V, 反向回传至 x 时需将 dx_q + dx_k + dx_v 相加。\n- 残差连接 out = x + f(x): 上游梯度 dout 一路沿恒等捷径直达, 另一路穿过子层 f_backward, 在分叉口汇合相加。\n- 词嵌入重复查表: 同一批次内同一个 token id 往往出现多次, 对应权重表同一行的梯度必须逐次累加。NumPy 中直接写 dW[ids] += dout 遇到重复下标只会保留最后一次赋值, 必须调用 np.add.at(dW, ids, dout) 原地无缓冲累加; check_ops 专门构造了含重复编号的 [[0,1,1],[5,1,0]] 来拦截这一常见疏漏。',
        },
        {
          title: '单算子全量核验与端到端抽样核验配合定位问题',
          body: 'gradcheck.py 分两层组织测试, 兼顾定位精度与运行速度:\n- 逐算子全元素检查 (check_op): 独立构造算子输入与同形状随机权重张量 $R$, 定义代理标量目标 $L = \\sum(\\text{out} \\odot R)$, 此时上游梯度 $dL/d\\,\\text{out}$ 恰好就是 $R$。无需拼装整网即可遍历算子每个输入元素做中心差分, 7 个算子的相对误差均在 1e-11 至 2e-10 之间。\n- 端到端抽样检查: 串联完整模型与交叉熵损失, 在 n_layer = 1 和 2 下为每个参数张量随机抽取 8 个坐标核对差分。判定条件采用混合容差 $|g_a - g_n| \\le \\text{atol} + \\text{rtol}\\cdot\\max(|g_a|,|g_n|)$, 避免在未命中 token 行等真实梯度接近 0 的坐标上因分母过小而误报相对误差超标。',
        },
      ],
      links: [
        { from: 'softmax backward', to: '所有注意力变体', body: 'softmax 的反向核心式为 $ds = a \\odot (da - \\sum a\\cdot da)$, 其中 $a$ 为归一化概率。从标准多头注意力到分块重算的 FlashAttention, 反向推导都围绕这一结构展开。' },
        { from: 'gradcheck', to: 'llm_train 的等价性断言', body: '这里用慢但可靠的数值差分校验手写解析梯度; 阶段 3 实现分布式并行与算子优化时延续同样思路, 以单卡朴素前向/反向为基准断言数值一致性。' },
        { from: 'RMSNorm backward', to: '深层训练稳定性', body: '前向除以均方根使输出对输入整体缩放保持不变, 对应到反向求导中, 沿输入向量 $x$ 自身方向的梯度分量会被耦合项精确抵消。' },
      ],
      sourceRows: [
        { concept: 'CE 的梯度', code: 'model.py:cross_entropy_forward_backward', takeaway: 'dlogits = (p - onehot) / N, 其中 p 为预测分布、onehot 为真实标签独热向量。由于每行概率和与独热和均为 1, 每行梯度元素之和恒等于 0。' },
        { concept: 'embedding backward', code: 'np.add.at(dW, ids, dout)', takeaway: '将各位置上游梯度按 token id 原地累加回嵌入表对应行, 避免花式索引 += 丢弃重复下标梯度。' },
        { concept: 'softmax backward', code: 'model.py:attention_backward', takeaway: '$ds = a \\odot (da - \\sum a\\cdot da)$。被因果掩码屏蔽的位置满足 $a = 0$, 乘积因子自动将该处梯度清零。' },
        { concept: 'RMSNorm 耦合项', code: 'model.py:rmsnorm_backward', takeaway: '$dx = c/\\text{rms} - x\\cdot s/(D\\cdot\\text{rms}^3)$, 其中 $c = d\\text{out}\\odot g$, $s = \\sum (c\\odot x)$。第二项刻画了均方根分母对整行输入的耦合导数。' },
        { concept: '数值梯度', code: 'gradcheck.py:numeric_grad', takeaway: '通过 $(f(w+\\varepsilon) - f(w-\\varepsilon))/2\\varepsilon$ 计算中心差分, 原地微调数组元素并在求值后恢复原值。' },
        { concept: '逐算子检查', code: 'gradcheck.py:check_op', takeaway: '用随机张量 $R$ 充当上游梯度直接测试单个算子, 一旦报错可立即锁定问题函数。' },
        { concept: '多层反向', code: 'model.py:transformer_backward', takeaway: '按 reversed(range(n_layer)) 逆序回传各层梯度, 最终在嵌入端将同一份 dh 分别交给词嵌入与位置嵌入。' },
      ],
      snippetTitle: '反向入口 + 两级 gradcheck',
      snippet: `loss, dlogits = cross_entropy_forward_backward(logits, y)
grads = transformer_backward(dlogits, cache)   # 内部: for i in reversed(range(n_layer))

# 1) 逐算子: 用随机 R 当 dout, 全元素差分
out, cache = op_forward(x);  analytic = op_backward(R, cache)
numeric = numeric_grad(lambda: (op_forward(x)[0] * R).sum(), x)   # 中心差分, eps=1e-5
assert norm(analytic - numeric) / max(norm(analytic), norm(numeric)) < 1e-6

# 2) 端到端: n_layer = 1, 2 各一遍, 每个参数抽 8 个位置
assert abs(g_a - g_n) <= atol + rtol * max(abs(g_a), abs(g_n))    # atol=1e-7, rtol=1e-4`,
      source: ['llm_basic/gradcheck.py:numeric_grad', 'llm_basic/gradcheck.py:check_op'],
      run: 'cd llm_basic && python gradcheck.py',
    },

    'basic-optim-sample': {
      title: 'Adam 与采样 · 训练后如何生成文本',
      subtitle: '拿到梯度后如何稳定更新各层尺度悬殊的参数, 训练完成后又如何从下一个词的概率分布中自回归采样出连贯文本。',
      tldr: '- Adam 为每个参数维护梯度的一阶矩 (平滑方向) 与二阶矩 (自适应缩放步长), 使不同梯度量级的参数每步更新幅度都归一化到学习率 lr 附近。\n- 文本生成采用串行自回归循环: 每步取序列末尾位置的 logits, 经温度缩放与 top-k 截断后按概率抽样出一个新字符, 拼接到末尾再进入下一轮前向计算。',
      question: '训练时 64 个位置可以在一次前向里并行算完, 为什么生成文本时只能逐个 token 串行推进?',
      evolution: {
        title: '自适应参数更新与可控随机解码',
        subtitle: '根问题: 各层梯度量级相差悬殊导致统一学习率难以调优; 训练产出的又是概率分布而非确定字符。',
        steps: [
          { name: 'SGD', pain: '(原点) 梯度给出了损失下降方向, 但各层梯度幅值相差数个数量级',
            fix: '全局共用单一学习率 $W \\leftarrow W - \\text{lr}\\cdot g$: 迁就最陡方向会让平缓方向几乎停滞' },
          { name: 'Adam', year: 2014, pain: '单一学习率受限于最大曲率方向, 深层 Transformer 收敛极慢',
            fix: '逐坐标除以梯度平方滑动平均的平方根 $\\sqrt{\\hat v}$, 把单步位移拉齐到 lr 量级; 2000 步内 loss 从 4.17 降至约 2.0' },
          { name: '贪心解码', pain: '模型前向只输出词表分布 [V], 需要解码策略选出具体字符',
            fix: '每步固定选概率最高的字符 (argmax); 自带检查点实测数十步后极易陷入 “the the the” 死循环' },
          { name: 'temperature + top-k', pain: '纯贪心缺乏多样性且一旦进循环便无法跳出, 全词表随机抽样又容易抽中低概率乱码',
            fix: '先用 temperature 调节分布锐度, 再用 top-k 截断长尾噪声后按概率抽样' },
        ],
      },
      code: 'llm_basic/{optim.py,sample.py,train.py}',
      points: [
        {
          key: true,
          title: '除以二阶矩平方根, 为每个坐标赋予自适应步长',
          body: '在朴素 SGD ($W \\leftarrow W - \\text{lr}\\cdot g$) 中, 所有参数共用同一个标量学习率, 最大稳定步长受限于损失曲面最陡方向的曲率 $2/\\lambda_{\\max}$; 而 Transformer 中词嵌入、注意力与输出层的梯度幅值往往相差几个数量级, 单一 lr 会导致梯度小的参数几乎不动。\nAdam 为每个参数坐标独立维护两个指数滑动平均: 一阶矩 $m$ 平滑梯度方向, 二阶矩 $v$ 估计梯度平方的近期尺度。更新量写为 $\\hat m / \\sqrt{\\hat v}$, 相当于把每个坐标的梯度除以它自身的典型幅值, 将无量纲比值规整到 $\\pm 1$ 附近, 乘上 lr 后每个参数单步实际移动的距离都大致落在 lr 量级。\n由于 $m$ 和 $v$ 从零初始化起步, 前几步估计值会偏小, 因此需除以 $(1 - \\beta^t)$ 做偏置修正。修正后第 1 步更新恰好退化为 $\\text{lr}\\cdot\\mathrm{sign}(g)$, 每个坐标精确移动 lr (optim.py 自检中对此做了断言)。',
        },
        {
          title: '训练靠已知前文并行计算, 生成因上下文自依赖只能串行',
          body: '训练与推理在计算形态上的差异来自输入是否事先完整已知:\n- 训练阶段采用 teacher forcing: 整条真实文本早已在手, 配合因果 mask 隔离未来信息后, 一次前向就能把 64 个位置的预测分布与交叉熵损失同时算完。\n- 生成阶段没有现成下文: 第 $t+1$ 步的输入序列必须包含第 $t$ 步刚抽样出的字符, 数据依赖决定了外层循环只能逐个 token 串行推进。\n串行本身无法消除, 但中间存在大量可省的重复计算: sample.py 每追加一个新字符就把整段历史重跑一遍前向, 单步注意力 $O(T^2)$ 累积 $T$ 步达到 $O(T^3)$。实际上前文各层算出的 K 和 V 在后续步骤中完全不变, 阶段 5 的 KV cache 正是把它们缓存下来以避免重算。',
        },
        {
          title: '正则化与学习率调度需匹配训练规模, 短程训练全开反而欠拟合',
          body: 'optim.py 实现了大模型训练常用的三项扩展开关, 默认均处于关闭状态:\n- 解耦权重衰减 (weight_decay, 即 AdamW): 直接把 $\\lambda W$ 加在参数更新项上, 不混入二阶矩 $v$, 使二维权重按统一比例向零收缩。\n- 全局梯度范数裁剪 (grad_clip): 将全部参数梯度视作单一长向量计算总范数, 超出阈值时整体等比缩放, 只限幅不改向。\n- 预热与余弦退火 (--cosine): 先线性升温保护初期未对齐的参数, 再沿余弦曲线平滑衰减。\n在 2000 步的极小规模对比中, 开启 2 层 + AdamW(0.1) + clip(1.0) + cosine (78,656 参数) 的验证集损失为 2.042, 反而不如默认单层全关配置 (45,568 参数) 的 1.986。原因在于总步数太短, 余弦调度过早将学习率压低至 3e-5, 新增的一层参数尚未充分收敛——这也说明工程技巧的收益高度依赖模型尺度与训练步数预算。',
        },
      ],
      links: [
        { from: 'Adam', to: 'llm_train 的稳定性工具', body: '阶段 3 的学习率预热、WSD/余弦退火、梯度裁剪与混合精度动态缩放, 均围绕 AdamW 更新主干构建大规模训练的数值防线。' },
        { from: 'sample.py', to: 'llm_infer 的采样', body: '温度缩放与 top-k 截断本质上都是在 softmax 前后变换一维 logits 分布; 阶段 5 将在同一接口上继续扩展 top-p、min-p 与重复惩罚。' },
        { from: '每步重算整段前向', to: 'KV cache', body: '相邻两步解码的输入仅末尾多出一个新 token, 历史位置的 K/V 张量恒定不变; 缓存历史 K/V 后单步解码只需计算最新 1 个 token。' },
      ],
      sourceRows: [
        { concept: '一阶矩', code: 'optim.py:adam_step', takeaway: '$m = \\beta_1 m + (1-\\beta_1)g$ 对近期梯度做指数平滑 ($\\beta_1=0.9$ 约对应最近 10 步有效窗口), 抑制单批次采样噪声。' },
        { concept: '二阶矩', code: 'optim.py:adam_step', takeaway: '$v = \\beta_2 v + (1-\\beta_2)g^2$ 追踪各坐标梯度平方尺度, 除以 $\\sqrt{\\hat v}$ 后消除层间梯度量级差异。' },
        { concept: '偏置修正', code: 'optim.py:adam_step', takeaway: '分别除以 $1-\\beta_1^t$ 与 $1-\\beta_2^t$, 补偿零初始化在训练初期带来的向零偏置。' },
        { concept: 'AdamW 与 L2 的区别', code: 'optim.py:adam_step', takeaway: '解耦衰减直接将 $\\lambda W$ 叠加在最终步长上, 避免并入梯度 $g$ 后被自适应分母 $\\sqrt{v}$ 抵消; 一维增益参数不做衰减。' },
        { concept: '全局范数裁剪', code: 'optim.py:clip_grad_norm', takeaway: '汇总全部参数梯度算全局二范数 $\\|g\\|$, 按 $\\min(1, \\text{max\\_norm}/\\|g\\|)$ 统一缩放, 保持梯度方向不变。' },
        { concept: 'warmup + cosine', code: 'optim.py:cosine_lr', takeaway: '前 warmup 步从接近 0 线性爬升至峰值学习率, 随后沿半周期余弦曲线平滑降至 min_lr。' },
        { concept: '自回归循环', code: 'sample.py:generate', takeaway: '每轮取出末尾位置 logits[0, -1], 依次做温度缩放、top-k 掩码与多项式分布采样, 将新 id 追加回上下文序列。' },
        { concept: '滑动窗口', code: 'sample.py:generate', takeaway: '受限于可学习 pos_emb 仅有 T_max = 64 行, 序列超长时每步只截取末尾 64 个 token 送入前向计算。' },
      ],
      snippetTitle: 'Adam 一步 + 采样一轮',
      snippet: `# optim.py: adam_step
m = beta1 * m + (1 - beta1) * g
v = beta2 * v + (1 - beta2) * g * g
update = (m / (1 - beta1**t)) / (sqrt(v / (1 - beta2**t)) + eps)
if weight_decay and w.ndim >= 2:       # AdamW: 加在 update 上, 不进 m/v
    update = update + weight_decay * w
W = w - lr * update

# sample.py: 每生成一个 token, 整段重来一遍
for _ in range(max_new_tokens):
    ctx    = ids[-T_max:]                          # pos_emb 只有 T_max 行
    logits = transformer_forward(W, ctx)[0, -1]    # 只要最后一个位置
    probs  = softmax(top_k_filter(logits / temperature, k))
    ids.append(sample(probs))`,
      source: ['llm_basic/optim.py:adam_step', 'llm_basic/sample.py:generate'],
      run: 'cd llm_basic && python sample.py "ROMEO:" --max-new 120 --temperature 0.8',
    },
  },
}

// 机制实验的原理、运行入口与源码引用。
const supplements = {
  "basic-forward": [
    {
      "title": "感受野: 单个输出位置能够读取的历史范围",
      "body": "感受野衡量某个位置的输出在结构上最多能接收哪些输入 token 的信息, 它描述的是计算图中的连通上界, 而非训练后实际利用的注意力强度。\n本章模型使用因果全注意力, 位置 t 在第 1 层就能直接读取 0 到 t 的全部前文。若改用窗口宽度为 W (含当前位置自身) 的滑动窗口注意力, 单层只能向前多看 W−1 个历史位置; 但第 2 层读取的第 1 层输出本身已经汇总过前一层窗口的信息, 因此每多堆叠一层, 有效历史跨度就会再向前延伸 W−1 个位置, L 层叠加后的感受野上界为 1 + L×(W−1)。\n- 数值示例: 当 W=8、L=4 时, 感受野宽度为 1 + 4×7 = 29, 而非 4×8 = 32 (后者把每层窗口里的当前位置自身重复计入了 4 次)。\n- 序列起点截断: 位置 t 及其前文总共只有 t+1 个 token, 实际可触达的输入数还需与 t+1 取较小值。",
      "source": null
    }
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
export default topic

canonicalizeRelated(topic)

// 不用 tokenizer 的路线。
topic.pages['basic-data'].related = [...(topic.pages['basic-data'].related || []), { route: 'models-byte-latent', label: '不用 tokenizer: 按熵切 patch', reason: 'BPE 的词表在训练前定死, 同一句话在不同语言下 token 数可能相差一倍; BLT 直接读字节, 让切分在运行时按下一字节熵决定。' }]
