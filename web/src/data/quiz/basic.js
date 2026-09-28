// 阶段 1 · llm_basic 自测题: 每章 3 题以上, 考"为什么这么写", 错误选项都是常见误解
export default {
  basic: [
    {
      q: '每个算子都写成 xxx_forward / xxx_backward 一对, forward 多返回的那个 cache 是干什么的?',
      options: ['存下各层输出, 下次 forward 遇到相同输入直接复用', '存 Adam 的一阶、二阶矩, 供更新参数时读取', '装反向要用的中间量, 相当于手写的 autograd tape', '存每层的 K/V, 生成新 token 时不用重算前文'],
      answer: 2,
      why: '反向要用到前向算过的输入、softmax 概率、归一化统计。PyTorch 靠 autograd 偷偷记下来, 这里必须自己装进 cache, 再按逆序取用。',
    },
    {
      q: '这个阶段为什么全程用 float64, 而不是训练常用的 float32 / bf16?',
      options: ['gradcheck 的差分要够精度: float32 下误差谷底过不了 1e-6 的线', 'float32 的累加误差会让 loss 训到中途变 NaN, 模型收敛不了', 'float64 下 exp 不会上溢, softmax 可以省掉减最大值那一步', 'numpy 只对 float64 调 BLAS, float32 矩阵乘会慢好几倍'],
      answer: 0,
      why: '中心差分的总误差 $\\approx \\varepsilon^2 + \\delta/\\varepsilon$。float32 的 $\\delta \\approx 6 \\times 10^{-8}$, 最低误差只有 1e-6 ~ 1e-5, 分不清 “实现正确” 和 “有个小 bug”。',
    },
    {
      q: '训练一步的正确顺序是?',
      options: ['backward 得梯度 → forward 得 loss → Adam 更新参数', 'forward 得 loss → Adam 更新参数 → backward 校正梯度', 'forward 得梯度 → backward 得 loss → Adam 更新参数', 'forward 得 loss → backward 得梯度 → Adam 更新参数'],
      answer: 3,
      why: '梯度是 loss 对参数的导数, 必须先有 loss; 优化器只消费梯度, 不关心它是怎么算出来的。',
    },
  ],
  'basic-data': [
    {
      q: '训练流水线实际用的是哪种 tokenizer? bpe.py 是什么角色?',
      options: ['训练用 bpe.py 合并 300 次的词表; tokenizer.py 只给 sample.py 解码', '训练用字符级 tokenizer.py (65 个字符); bpe.py 是独立演示', '先用 tokenizer.py 切字符, 再由 bpe.py 合并成子词喂给训练', '训练用 tokenizer.py, 但 meta.npz 里的词表是 bpe.py 生成的'],
      answer: 1,
      why: 'prepare / train / sample 和自带的 ckpt.npz 都基于 65 个字符的词表, 为的是让注意力和梯度当主角。bpe.py 单独演示 “反复合并最高频相邻对”。',
    },
    {
      q: '字符级 tokenizer 相比 BPE, 最主要的代价是什么?',
      options: ['训练集没出现过的词只能编成 <unk>, 新词无法表示', '同一段文本的 token 序列长得多, 而注意力是 $O(T^2)$', '词表只有 65, lm_head 太窄, 模型容量被它卡住', '字符之间的空格在编码时丢掉, id 无法无损还原原文'],
      answer: 1,
      why: 'bpe.py 的实测: 同一句话字符级要 60 个 token, 300 次合并后的 BPE 只要 24 个。序列短了, 模型也不用自己学拼写。',
    },
    {
      q: 'get_batch 里 y 为什么是 x 右移一位?',
      options: ['x、y 错开一位相当于数据增强, 能减轻过拟合', 'pos_emb 从第 1 行开始编号, 标签要后移一格才对齐', '留出末尾一个 token 做验证, 避免 train/val 泄漏', '目标是 next-token: 位置 $i$ 的标签是第 $i+1$ 个 token'],
      answer: 3,
      why: '因果 mask 保证位置 $i$ 只看得到 $\\le i$ 的 token, 所以一条长度 $T$ 的序列一次前向就提供了 $T$ 个训练样本。',
    },
  ],
  'basic-forward': [
    {
      q: '把 --n-layer 从 1 改成 2, transformer_forward 要改哪里?',
      options: ['不用改: 循环按 block_{i}_ 前缀取参数, 层数从参数名数出来', '要再写一份 block_forward, 因为两层的 cache 结构不同', '要把 lm_head 输入改成 2D, 两层的输出会拼接起来', '要把 n_layer 写进 ckpt 配置, 否则不知道循环几次'],
      answer: 0,
      why: '每个 block 进去出来都是 [B,T,D], 残差流形状不变, 所以堆层只是一个循环。层数甚至是从参数名 block_{i}_norm1_g 数出来的。',
    },
    {
      q: 'attention_forward 的 cache 里存的是 softmax 之后的 attn, 而不是 softmax 之前的 scores。为什么?',
      options: ['attn 是 [B,T,D], 比 [B,T,T] 的 scores 省一个数量级内存', 'attn 已乘过 $1/\\sqrt{D}$, 存它反向时就不用再乘缩放', 'softmax 反向 $ds = a \\odot (da - \\sum a \\cdot da)$ 只用输出 $a$', 'scores 被 np.where 原地改写成 attn, 原值已经不在了'],
      answer: 2,
      why: 'forward 存什么, 完全由 backward 要什么决定。这顺带解释了为什么被 mask 的位置不用再 mask 一次: 因子 $a = 0$ 让梯度自动归零。',
    },
    {
      q: 'logits 的形状是 [B,T,V]。训练时 T 个位置的 loss 为什么能一次算完?',
      options: ['RMSNorm 逐位置归一化, 位置之间没有信息交换', 'teacher forcing 让每个位置拿上一步的预测当输入', 'causal mask 让位置 i 看不到之后的 token, 每处都是合法的题', 'loss 只算最后一个位置, 前面的 logits 只是副产品'],
      answer: 2,
      why: '没有 causal mask, 位置 i 会直接 “看到答案”; 有了它, 一次前向等于并行做了 T 道 next-token 题。',
    },
    {
      q: '默认配置 V=65, D=64, H=128, T=64, n_layer=1 一共 45,568 个参数。把 T 从 64 拖到 256, 参数和激活分别怎么变?',
      options: ['参数不变 (注意力权重都是 [D,D]), 激活按 T 线性涨 4 倍', '只有 pos_emb 变长, 参数 +27%; 激活涨几倍, attn 是 [B,T,T]', '参数按 $T^2$ 涨 (mask 是 [T,T]), 激活按 T 线性涨 4 倍', 'Q/K/V 权重是 [T,D], 参数和激活都按 T 线性涨 4 倍'],
      answer: 1,
      why: '参数表里根本没有 $T^2$, 只有 pos_emb 的行数是 $T_{\\max}$。长上下文的账记在激活和注意力的 $T^2$ 上, 上面的形状流水线里可以直接拖出来看。',
    },
  ],
  'basic-backward': [
    {
      q: 'softmax + cross-entropy 合并后, 对 logits 的梯度是?',
      options: ['$(p - \\mathrm{onehot}(y))/N$', '$-\\mathrm{onehot}(y)/(N\\,p)$', '$(\\mathrm{diag}(p) - pp^\\top)/N$', '$(\\mathrm{onehot}(y) - p)/N$'],
      answer: 0,
      why: 'CE 的本地导数 $-1/p[y]$ 和 softmax 的雅可比 $\\mathrm{diag}(p)-pp^\\top$ 一相乘, $1/p[y]$ 正好被约掉。结果既简单又数值稳定, 所以代码把两者合成一个函数。',
    },
    {
      q: 'gradcheck 的中心差分里, $\\varepsilon$ 为什么不是越小越好?',
      options: ['太小时截断误差 $O(\\varepsilon^2)$ 反而变大, 差分偏离导数', '太小时 $\\varepsilon$ 下溢成 0, 除以 $2\\varepsilon$ 直接报除零', '太小时两次 f 的差被舍入误差淹没, 误差按 $\\delta/\\varepsilon$ 涨', '太小时差分要迭代更多次才收敛, 计算量按 $1/\\varepsilon$ 涨'],
      answer: 3,
      why: '总误差 $\\approx \\varepsilon^2\\,(\\text{截断}) + \\delta/\\varepsilon\\,(\\text{舍入})$, 是一条 U 形曲线; float64 下谷底在 1e-5 附近, 所以 gradcheck.py 取 eps=1e-5。',
    },
    {
      q: 'embedding_backward 为什么必须用 np.add.at(dW, ids, dout), 不能写 dW[ids] += dout?',
      options: ['dW[ids] += dout 改的是 dW 的副本, 原 dW 完全不变', '同一 id 在 batch 里出现多次时, += 只加一次, 梯度会丢', 'ids 是 [B,T] 二维数组, 花式索引会直接报 IndexError', 'add.at 按 id 出现次数取平均, 防止高频 token 梯度过大'],
      answer: 1,
      why: '这类 bug 不报错, loss 照样下降, 只是高频 token 的梯度被严重低估。逐算子 gradcheck 故意用含重复 id 的输入 [[0,1,1],[5,1,0]] 来抓它。',
    },
  ],
  'basic-optim-sample': [
    {
      q: '在病态损失上 (某个方向比另一个陡 $\\kappa$ 倍), 纯 SGD 的学习率被什么卡住?',
      options: ['被最平的方向: lr 必须大于 $1/\\kappa$, 否则平缓方向停滞', '被最陡的方向: lr 必须小于 $2/\\kappa$, 平缓方向几乎不动', '被 batch size: 梯度噪声要求 lr 小于 $1/B$, 与曲率无关', '被参数个数: lr 要按 $1/\\sqrt{n}$ 缩小, 否则总步长爆炸'],
      answer: 1,
      why: '陡方向每步乘 $(1 - \\mathrm{lr} \\cdot \\kappa)$, 绝对值超过 1 就发散; 满足它之后, 平方向每步只缩 $(1 - \\mathrm{lr})$, 慢得要命。',
    },
    {
      q: 'Adam 用 $\\hat m/\\sqrt{\\hat v}$ 更新, 最关键的性质是?',
      options: ['二阶矩相当于曲率信息, 保证在非凸 loss 上也收敛到全局最优', '偏差修正后步长由 $\\hat m$ 自己决定, 不再需要设学习率', '每个参数的步长约等于 lr, 与该方向梯度的绝对大小基本无关', '除以 $\\sqrt{\\hat v}$ 放大梯度大的方向, 帮它更快冲出平坦区'],
      answer: 3,
      why: '除以 $\\sqrt{\\hat v}$ 等于给每个坐标单独归一化, 相当于每个参数有自己的学习率。偏差修正则补回 $m$、$v$ 从 0 起步被拖小的那部分。',
    },
    {
      q: 'README 实测: 2 层 + AdamW + clip + cosine 跑 2000 步, val_loss 2.045, 比默认 1 层的 1.981 还差。最该得出的结论是?',
      options: ['AdamW 把 1 维的 RMSNorm gain 也衰减到 0, 两层的归一化失效了', '开关不是开了就好: 步数太短, cosine 过早把 lr 压到 3e-5, 新层没学起来', '78,656 个参数已经超过数据能支撑的量, 2 层模型过拟合了', '层数多了反向链更长, 手写 backward 的舍入误差累积拖垮了训练'],
      answer: 1,
      why: '这些开关要学的是怎么实现、各自解决什么问题。规模、步数、调度必须配套, 否则加容量反而拖慢收敛。',
    },
    {
      q: '为什么训练时所有位置并行算, 生成时却只能一个 token 一个 token 来?',
      options: ['训练有答案 (teacher forcing); 生成时第 $t+1$ 个依赖刚抽出的第 $t$ 个', '训练 batch 大能填满并行度; 生成时 batch=1, 没有东西可并行', 'temperature 和 top-k 要完整分布, 只能逐个位置算 softmax', 'pos_emb 只有 64 行, 生成时只能滑动窗口一格一格往前挪'],
      answer: 0,
      why: '这条依赖链是算法固有的, 省不掉; 能省的只是重复计算: 这就是阶段 5 的 KV cache。temperature / top-k 只是对 logits 的后处理。',
    },
  ],
}
