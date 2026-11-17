import { canonicalizeRelated } from '../../utils/consolidate.js'
// 阶段 1 · llm_basic: 四个章节的完整页面定义。章节目录登记在 data/models.js 的 basicChapters, 所以这里 chapters 为空。
const topic = {
  stage: 'basic',
  chapters: [],
  pages: {
    'basic-data': {
      widgets: ['BpeLab'],
      title: '数据与 tokenizer · 把文本变成可训练张量',
      subtitle: '读完你能说清: 1,115,394 个字符怎么变成 train.bin 里的 1,003,854 个 token, 每个 batch 是从哪儿切出来的, 以及训练目标 y 为什么就是 x 往右挪一格。',
      tldr: '- 训练前先跑一次 prepare.py: 把 input.txt 里的 1,115,394 个字符, 按 “一个字符一个整数编号” 的办法编成 1,003,854 个训练 token 和 111,540 个验证 token, 写进 train.bin / val.bin; 65 个字符的编号表存进 meta.npz。\n- 之后训练只用 memmap 读这两个二进制文件, 词表再也不会变: 训练和采样用的是同一张表。',
      question: '字符级 tokenizer 一个字母一个 token, 序列很长; BPE 能把同一句话压短一半多, 为什么这里训练偏偏用字符级?',
      evolution: {
        title: '词表大小和序列长度之间来回拉扯',
        subtitle: '根问题: 网络只认整数 id, 而文本是开放的: 词表切太粗会缺词, 切太细序列太长。',
        steps: [
          { name: '按词切', pain: '(原点) 网络只会对数字做运算, 文本得先切块, 每块编一个整数 id',
            fix: '按空格切, 一个词一个 id; 语料里没出现过的词、拼错的词都没有 id' },
          { name: '字符级', pain: '词表跟着语料一直涨, 新词编不出来',
            fix: '一个字符一个 id: Tiny Shakespeare 只要 65 个, 不缺词; 一个单词要 5~10 个 token' },
          { name: 'BPE', year: 2016, pain: '序列拉长, 注意力的计算量按序列长度的平方 $O(T^2)$ 涨; 模型还得自己学拼写',
            fix: '反复把最高频的相邻对合并成新 token: 同一句话从 60 个字符变成 24 个 token' },
          { name: 'Byte-level BPE', year: 2019, pain: '合并从字符表起步, 训练时没见过的字符还是编不出来',
            fix: '从 256 个字节起步再合并: 任何 UTF-8 文本都能编, 解码无损。bpe.py 写的就是这种' },
        ],
      },
      code: 'llm_basic/{prepare.py,tokenizer.py,bpe.py,input.txt,train.bin,val.bin,meta.npz}',
      points: [
        {
          title: '词表只建一次, 之后谁都不许改',
          body: 'tokenizer 把文本变成整数编号, 它的编号表 (词表) 只建一次, 训练和采样读的是同一张。这里的 tokenizer 是字符级的: 一个字符对应一个整数 id。所谓 “训练 tokenizer” 就一行 sorted(set(text)): 把语料里出现过的字符去重 (set), 再按字典序排好 (sorted), 一共 65 个, 排第几就是几号。\n- 为什么排序: set 的顺序不保证, 排序后换台机器也得到同一张表, 同一个字符永远是同一个 id。\n- 为什么存盘: 这张表存进 meta.npz。prepare.py 用它把整份文本一次性编成 id, 存进 train.bin / val.bin, train.py 训练时直接读这些 id; sample.py 生成时用它把 id 变回字符。两边读同一个文件, 今天训好的 ckpt.npz 明天解码出来才不会是乱码。\n- 存多大: 65 小于 256, 一个 id 用一个字节 (uint8) 就够, prepare.py 里有 assert vocab_size < 256 把关。',
        },
        {
          key: true,
          title: '训练用字符级, BPE 只是独立演示',
          body: '本目录有两个 tokenizer, 只有一个接进了训练。\n- 训练、采样、自带的 ckpt.npz: 全部基于 tokenizer.py 的 65 个字符。\n- bpe.py: 只读 input.txt, 自己跑自己的, 不接训练流水线。\nBPE (Byte Pair Encoding) 是另一种切法:\n1. 先把文本拆成 256 种字节。\n2. 反复找 “当前出现次数最多的相邻两个 token”, 把它们合并成一个新 token。\n合并几百次之后, 常见的词和词组就各自变成了单个 token。跑 python bpe.py (默认 300 次合并), 看同一句话 “First Citizen: Before we proceed any further, hear me speak.”:\n- 字符级: 60 个 token, 一个字符一个。\n- BPE: 24 个 token, 压缩率 2.50 字符/token。\n序列短了一半多, 注意力的计算量跟着降 (它按序列长度的平方涨)。这里训练仍用字符级, 是为了把篇幅留给注意力和梯度, 分词细节不展开; BPE 本身照样重要, 所以单独留了一个演示, 上面的实验台把合并过程一步步摊开了。',
        },
        {
          title: 'y 就是 x 右移一位',
          body: '语言模型的训练目标只有一个: 看着前面的文本, 猜下一个字符是什么。所以 “答案” 不用另外标注, 它就在文本里。\n- get_batch 怎么切:\n  1. 从 train.bin 里随机挑 32 个起点 (batch_size=32)。\n  2. 从每个起点往后切 64 个 token (seq_len=64) 当输入 x。\n  3. 再从起点加 1 的位置切 64 个当目标 y。于是 y 的第 i 个 token 正好是 x 的第 i+1 个。\n- 一条序列 64 道题: 位置 0 看着 x[0] 猜 y[0], 位置 1 看着 x[0..1] 猜 y[1], 一直到位置 63。一次前向把 64 个位置一起算, 一条长度 64 的序列就提供了 64 道 next-token 题。\n- 前提: 答案 y[i] 就是 x[i+1], 它明明在输入里。所以注意力里必须有因果 mask, 让位置 i 只能看到 x[0..i], 看不到 x[i+1] 以后; 否则模型直接抄答案, 什么也学不到。',
        },
      ],
      links: [
        { from: 'input.txt', to: 'prepare.py', body: '左边是原始的莎士比亚文本, 1,115,394 个字符; 右边的脚本把它变成 65 种 id, 再按前后顺序切成前 90% 训练、后 10% 验证。不随机抽: 随机抽会让验证集的句子在训练集里找到前后文, 验证分数就虚高了。' },
        { from: 'train.bin', to: 'train.py:get_batch', body: 'train.bin 就是一长串 uint8 整数: 词表 65 小于 256, 一个 token 占一个字节, 所以文件有多少字节就有多少 token。get_batch 用 np.memmap 把文件映射进来, 不整个读进内存, 切到哪段才读哪段。' },
        { from: 'meta.npz', to: 'sample.py', body: 'meta.npz 存着 65 个字符的列表; sample.py 用它建反查表 itos (id 到字符), 把模型生成的整数还原成文字。词表要跟 ckpt 一起走: 模型学到的是 “第 17 号后面常跟第 42 号”, 换一张表这些编号就全错了。' },
      ],
      sourceRows: [
        { concept: 'vocab 构造', code: 'prepare.py', takeaway: 'chars = sorted(set(text)) 得到 65 个去重并排序的字符; stoi 是它的反查表, 字符到 id。整个 “tokenizer 训练” 就这两行, 没有任何可学的参数。' },
        { concept: '训练/验证切分', code: 'prepare.py', takeaway: '先按位置切 90/10 (前 90% 训练, 后 10% 验证) 再落盘。验证集是一段模型没见过的连续文本, 测出来的 loss 才可信。' },
        { concept: 'uint8 存储', code: 'prepare.py', takeaway: 'assert vocab_size < 256: 一个 id 一字节装得下。越过这条线就得换 uint16 (两字节), 文件直接翻倍。' },
        { concept: 'batch 对齐', code: 'train.py:get_batch', takeaway: 'x = data[i:i+T], y = data[i+1:i+T+1]: y 比 x 整体往后挪一格。起点 i 最多取到 len−T−1, 留一位给 y 的末尾, 否则切 y 时会越过文件结尾。' },
        { concept: 'BPE 训练 (独立 demo)', code: 'bpe.py:train_bpe', takeaway: '三步循环:\n1. 数一遍相邻 token 对各出现几次。\n2. 把最高频的那一对合并成一个新 id。\n3. 重复 num_merges 次。\n纯贪心的计数和替换, 一个梯度都没有。' },
        { concept: 'BPE 编码', code: 'bpe.py:encode', takeaway: '对新文本按训练时的合并先后顺序重放规则: 每轮在当前序列里找出 “训练时最早被合并” 的那一对先合。顺序错了, 同一个词会切成不同 token, 模型就认不出它。' },
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
      subtitle: '读完你能报出每一步张量的形状, 说出每个 forward 往 cache 里塞了什么、为什么非塞不可。',
      tldr: '- forward 是 “把输入 id 变成每个位置的打分” 的那一趟计算。输入 ids 形状 [B,T] (B 条序列, 每条 T 个 token), 输出 logits 形状 [B,T,V]: 每个位置给词表里 V 个字符各打一个分, 分越高越可能是下一个字符。\n- 路径:\n  1. embedding 查表把 id 变成向量。\n  2. 过 n_layer 个 Pre-LN block (每个 block 先归一化再进注意力和 MLP)。\n  3. final norm 加 lm_head 打分。\n- 每一步都顺手把反向传播要用的中间量装进 cache 带回来, 因为这里没有 autograd 替你记。',
      question: 'forward 算出 logits 不就完事了吗, 为什么还要拖着一长串 cache 一起返回?',
      evolution: {
        title: '让每个位置都看得见全部前文',
        subtitle: '根问题: 前文有长有短, 模型却要把它变成一个固定长度 V 的分布。',
        steps: [
          { name: 'Bigram 计数表', pain: '(原点) 猜下一个字要看前文, 前文的组合数随长度指数增长',
            fix: '只看前 1 个字, 查一张 [V,V] 计数表; 再往前的上下文全丢了' },
          { name: '窗口 MLP', year: 2003, pain: '看前 n 个字, 计数表就要 $V^n$ 格, 大多数格从没出现过',
            fix: 'embedding 把每个 id 变成 D 维向量, 窗口内的向量拼起来过一个小神经网络 (MLP), 相近的字共享统计; 窗口长度写死' },
          { name: 'RNN', pain: '窗口外的字看不见, 窗口每加长一格参数就跟着涨',
            fix: '一个定长向量 (隐状态) 逐字往后传, 能装下全部前文; 代价: T 个位置只能排队一个个算' },
          { name: 'Causal Transformer', year: 2017, pain: 'RNN 要排队算 T 步, 远处的字要被转手 T 次才到',
            fix: '每个位置直接和全部前文算相关度 (causal attention), T 个位置一次算完; 词序靠位置向量 pos_emb 补回' },
        ],
      },
      code: 'llm_basic/model.py:{embedding_forward,rmsnorm_forward,attention_forward,block_forward,transformer_forward}',
      points: [
        {
          key: true,
          title: 'cache 替代了 autograd',
          body: 'cache 是每个 forward 替反向存下的中间值, 做的是 torch 里 autograd 做的事。forward 只负责算出 logits, 但训练还要反向传播: 从 loss 倒推每个参数该往哪动。反向要用到前向算过的中间值, 例如 RMSNorm 除过的那个 rms、softmax 的输出 attn、每个 linear 层的输入 x。\n在 torch 里, autograd 会在背后把这些中间值记在一张 “磁带” (tape) 上, 你看不见。这里没有 autograd, 所以每个 xxx_forward 自己把反向要用的东西打包成一个 tuple 返回, 这个 tuple 就是 cache; 对应的 xxx_backward 再从里面取。\n- 该存的没存: 对应的 backward 写不出来。比如 linear_backward 要算 dW = xᵀ @ dout, 没有存 x 就没法算。\n- 用不上的不存: softmax 的反向只需要它的输出 attn, 不需要输入 scores, 所以 attention 的 cache 存 attn, 不存 scores。\n代价是显存: 整段前向的中间量都得留到反向结束。',
        },
        {
          title: '形状只有四个字母',
          body: '这份代码里的形状只用四个字母: B、T、D、V。\n- B: batch 里有几条序列。\n- T: 每条序列多长。\n- D: 模型内部向量的维度 (默认 64)。\n- V: 词表大小 (65)。\n主干上所有张量都是 [B,T,D]: 每条序列的每个位置, 都是一个 D 维向量。只有两处例外:\n- 注意力里的 [B,T,T]: 每个位置对每个位置打一个相关度分, 所以是 T 乘 T 的矩阵。\n- 最后的 logits [B,T,V]: 每个位置对词表里 V 个字符各打一分。\n读代码先对上矩阵乘法两边的形状: [B,T,D] 乘 [D,V] 得到 [B,T,V], 对上了再看数值细节。',
        },
        {
          title: '加层只是一个 for 循环',
          body: '每个 block 吃进 [B,T,D], 吐出来还是 [B,T,D]。输入输出同形, 所以堆几层都不需要新推导: transformer_forward 里就是 for i in range(num_layers(W)), 把上一层的输出当下一层的输入。\n层数没有写成配置项, 而是从参数名数出来的: 每层都有一个叫 block_{i}_norm1_g 的参数 (第 i 层第一个 RMSNorm 的增益), num_layers 数一数有几个以 _norm1_g 结尾的键, 就知道有几层。好处是 ckpt.npz 不用存 n_layer 字段: 自带的 ckpt 就没有这个字段, 照常加载。\n默认 n_layer=1, 参数量 45,568; 加 --n-layer 2 就是两层。',
        },
      ],
      links: [
        { from: 'learned pos_emb', to: 'RoPE', body: '注意力本身不知道词序, 位置得另外告诉它。这里的做法是一张 [T_max, D] 的可学习表, 第 i 行直接加到第 i 个位置的向量上, 所以序列长度 T 被 T_max=64 卡死, 第 65 个位置没有对应的行。阶段 2 改成 RoPE: 不加表, 而是按位置旋转 Q 和 K, 不占参数, 长度也没有硬上限 (超出训练长度仍会退化, 见 YaRN 一章)。' },
        { from: '单头 attention', to: 'MHA / GQA / MLA', body: '这里只有一个注意力头, 每个头的维度 head_dim 就等于 D。多头 (MHA) 只是把 D 切成几段, 每段各算一遍注意力再拼回来, 形状账一模一样; GQA 和 MLA 再在 K、V 上省空间。' },
        { from: 'cache 全存', to: 'activation checkpoint', body: 'cache 里的中间量就是训练时的显存大头: 前向每一层的输入输出都留着等反向。大模型显存不够时会故意丢掉一部分 cache, 反向走到那一层时重新算一遍前向, 拿时间换空间, 这叫 activation checkpoint。' },
      ],
      sourceRows: [
        { concept: 'embedding', code: 'model.py:embedding_forward', takeaway: 'W[ids] 就是查表: 每个整数 id 取出 W 的那一行当向量。cache 只存 ids 和 W 的形状, 不存 W 本身: 反向只需要知道每个位置取了哪一行, 把梯度加回那一行就行。' },
        { concept: 'RMSNorm', code: 'model.py:rmsnorm_forward', takeaway: 'RMSNorm 把每个位置的 D 维向量除以它自己的均方根 rms, 再乘可学的增益 g。cache 是 (x, g, rms) 三件套: 反向里 rms 依赖整行 x, 带来一个耦合项, 所以三样全要。' },
        { concept: '缩放点积', code: 'model.py:attention_forward', takeaway: 'scale = $1/\\sqrt{D}$。Q 和 K 的点积是 D 个乘积之和, 不缩放的话方差正比于 D; 分数一大, softmax 就几乎只选一个 (饱和), 梯度传不动。' },
        { concept: 'causal mask', code: 'model.py:causal_mask', takeaway: '一张 [T,T] 的布尔表, 上三角 ($j > i$, 意思是 key 在 query 之后) 填 −inf, softmax 后这些位置的权重为 0。有了它, 一次前向等于并行做了 T 道 “看前文猜下一个” 的题, 谁都抄不到答案。' },
        { concept: '残差', code: 'model.py:block_forward', takeaway: 'h = x + a 这种加法没有 cache: 加法的反向是把上游梯度 dout 原样发给两条路 (x 和 a), 不需要任何中间量。' },
        { concept: '层循环', code: 'model.py:num_layers', takeaway: 'num_layers 数参数名里以 _norm1_g 结尾的有几个, 就是几层。ckpt 里不用单独存层数。' },
        { concept: '语言模型头', code: 'model.py:transformer_forward', takeaway: '[B,T,D] @ [D,V] → [B,T,V]: 每个位置的 D 维向量乘一张 [D,V] 的表, 得到对 V 个字符的打分。V 一大 (真实模型词表几万到十几万), 这一步的参数和中间结果都会压过整个主干。' },
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
      subtitle: '读完你能指着任意一个 *_backward, 说出它从 cache 里取了什么、为什么必须取。写错了怎么被抓出来, 你也能说清。',
      tldr: '- 反向传播在算什么: loss 是一个数, 衡量模型猜得有多差; 反向传播算的是 “每个参数往哪边动一点, loss 会变多少”, 这个量叫梯度。有了梯度, 优化器才知道怎么改参数。\n- 怎么算: 把 forward 倒着走一遍。从 loss 对 logits 的梯度 dlogits = (p − onehot)/N 出发 (p 是模型给的概率, onehot 是正确答案), 每个 backward 从 cache 取中间量, 算出对自己输入的梯度 dx, 再往前一层递。\n- 写错不会报错: loss 照样下降, 只是模型悄悄变笨。所以 gradcheck 必须跑, 用数值差分当裁判。',
      question: '把 np.add.at 写成 +=, loss 照样往下掉, 那我怎么知道反向写错了?',
      evolution: {
        title: '先算出梯度, 再证明它没算错',
        subtitle: '根问题: 每一步都要 loss 对 45,568 个参数各自的导数, 而这里没有 autograd。',
        steps: [
          { name: '数值差分', pain: '(原点) 想知道每个参数动一点点, loss 会变多少',
            fix: '逐个参数加减 $\\varepsilon$ 各跑一次前向; 一步训练就要 2 × 45,568 次前向' },
          { name: '反向传播', year: 1986, pain: '差分要为每个参数单独跑前向, 参数越多越算不动',
            fix: '复合函数的导数是各段导数相乘 (链式法则), 从 loss 往回一段段乘: 1 次前向加 1 次反向拿到全部梯度; 前向中间量得存下来' },
          { name: 'forward / backward 成对', pain: '整个 GPT 的导数写成一个式子太长, 手推不出来',
            fix: '每个算子写一对函数, cache 存中间量; 整模型的反向就是把前向倒着走一遍' },
          { name: 'gradcheck', pain: '手写反向漏一项、转置写反都不报错, loss 照样下降',
            fix: '请回第一步的差分当裁判: 它慢但不会错, 逐算子逐元素对答案' },
        ],
      },
      code: 'llm_basic/model.py:*_backward · llm_basic/gradcheck.py',
      points: [
        {
          key: true,
          title: '反向写错不报错, 模型只会悄悄变笨',
          body: '反向传播写错了, 程序也不会崩。常见的错有矩阵转置写反、sum 加错了维度、softmax 的导数漏一项: 形状照样对得上, 算出来的梯度也是个像模像样的数, 只是方向略偏。loss 照样下降, 只是降得慢一点, 从训练曲线上很难发现。\n所以要一个 “不可能写错” 的参照: 数值梯度。做法:\n1. 把某个参数 w 加一点点 $\\varepsilon$ 跑一次前向看 loss。\n2. 再减一点点跑一次。\n3. 两次 loss 之差除以 $2\\varepsilon$, 就是 loss 对 w 的导数, 公式是 $\\tfrac{f(w+\\varepsilon) - f(w-\\varepsilon)}{2\\varepsilon}$ (中心差分)。\n它慢 (每个参数要跑两次前向), 但不涉及任何手推公式, 不会错。\ngradcheck.py 就拿它和手写 backward 算出来的梯度逐个元素比。README 做了一个实验: 故意删掉 rmsnorm_backward 的耦合项 (rms 依赖整行 x 的那一项), 逐算子检查立刻报 dx 相对误差 3.11e-01; 正常时这个数在 1e-10 量级。',
        },
        {
          title: '一个张量被用了几次, 梯度就是几路之和',
          body: '链式法则有一条规矩: 一个量在前向里被用了几次, 反向时它的梯度就是几路加起来。这份代码里有三处:\n- Q/K/V: 注意力里同一个 x 同时喂给三条支路 (算 Q、算 K、算 V), 反向要 dx_q + dx_k + dx_v。\n- 残差 out = x + f(x): x 走了两条路, 一条直接加到输出 (捷径), 一条穿过子层 f。反向是捷径那份 dout, 加上穿过子层回来的那份。\n- embedding: 同一个 token id 在 batch 里出现多次, 它查的是表里同一行, 梯度必须累加。代码用 np.add.at(dW, ids, dout)。写成 dW[ids] += dout 的话, numpy 对重复下标只加最后一次, 其余几次悄悄丢了, 梯度偏小但不报错。\ncheck_ops 故意用含重复 id 的输入 [[0,1,1],[5,1,0]] 来考最后这一点: id 1 出现了三次, += 写法当场露馅。',
        },
        {
          title: 'gradcheck 分两级, 各管一头',
          body: 'gradcheck 分两级: 先逐个算子单独验, 再整个模型端到端验。\n- 逐算子: 单独拿一个算子 (linear、rmsnorm、attention 等) 来验, 不搭整个模型。它没有 loss, 所以造一个: 拿一个随机张量 R, 令 $L = \\sum(\\text{out} \\odot R)$ (输出和 R 逐元素相乘再求和), 这样 $dL/d\\,\\text{out}$ 正好是 $R$, 把 R 当 dout 喂给 backward 就行。然后对每个输入的每个元素做差分, 和 backward 的结果比。7 个算子的相对误差落在 1e-11 到 2e-10, 哪个算子错一目了然。\n- 端到端: 整个模型加 CE (交叉熵 loss, 衡量预测分布和正确答案差多远), n_layer = 1 和 2 各跑一遍, 每个参数随机抽 8 个位置做差分 (全抽太慢, 每个位置要两次完整前向)。\n端到端的判定用 $\\text{atol} + \\text{rtol}\\cdot\\max(|g_a|,|g_n|)$, 意思是允许的误差等于一个绝对下限加一个和梯度大小成比例的量。不直接比相对误差, 是因为抽样常落在真实梯度约等于 0 的位置 (比如 batch 里没出现过的 token 那一行 tok_emb), 两个 1e-10 量级的数相对误差可以很大, 但并不是 bug。',
        },
      ],
      links: [
        { from: 'softmax backward', to: '所有注意力变体', body: '$ds = a \\odot (da - \\sum a\\cdot da)$ 是 softmax 的反向公式: a 是 softmax 的输出, da 是上游传来的梯度, ds 是对输入分数的梯度。从 MHA 到 FlashAttention, 每换一次注意力的实现, 这一行都要重新写对一次。' },
        { from: 'gradcheck', to: 'llm_train 的等价性断言', body: '这里拿数值梯度当参照来验手写反向。阶段 3 做并行训练时用同一个套路: 先算一个单卡的 dense baseline, 再断言并行版本和它一致。两处都是先有一个确定对的答案, 再拿新实现去对。' },
        { from: 'RMSNorm backward', to: '深层训练稳定性', body: '前向里 RMSNorm 把每行向量的均方根 (RMS) 拉回固定值, 层数一多数值才不会越叠越大。\n反向的耦合项是同一件事的另一面: 把 x 整体放大, 输出不变 (因为除以了自己的 RMS), 所以沿 x 本身方向的梯度被减掉, 这就是公式里减去的第二项。' },
      ],
      sourceRows: [
        { concept: 'CE 的梯度', code: 'model.py:cross_entropy_forward_backward', takeaway: 'dlogits = (p − onehot)/N:\n- p: softmax 给每个字符的概率。\n- onehot: 在正确答案那一位是 1 其余是 0。\n- N: 位置总数。\n每一行的 p 加起来是 1, onehot 也是 1, 所以每行梯度加起来恰好为 0。gradcheck.py 断言的是所有行的总和为 0。' },
        { concept: 'embedding backward', code: 'np.add.at(dW, ids, dout)', takeaway: '把每个位置的梯度加回它查过的那一行。花式索引的 += 遇到重复 id 只加一次, 必须用 add.at。' },
        { concept: 'softmax backward', code: 'model.py:attention_backward', takeaway: '$ds = a \\odot (da - \\sum a\\cdot da)$。被 mask 的位置 $a = 0$, 公式带一个因子 a, 这些位置的梯度自动归零, 不用再 mask 一次。' },
        { concept: 'RMSNorm 耦合项', code: 'model.py:rmsnorm_backward', takeaway: '$dx = c/\\text{rms} - x\\cdot s/(D\\cdot\\text{rms}^3)$。代码里 c = dout·g, s 是 c·x 整行求和。\n- 第一项是 “除以 rms” 的直接导数。\n- 第二项来自 rms 本身依赖整行 x, 一行里每个元素的梯度因此互相牵连。' },
        { concept: '数值梯度', code: 'gradcheck.py:numeric_grad', takeaway: '$(f(w+\\varepsilon) - f(w-\\varepsilon))/2\\varepsilon$: 把 w 加一点、减一点各跑一次前向, 差值除以 $2\\varepsilon$。截断误差是 $O(\\varepsilon^2)$, 意思是 ε 缩小 10 倍误差缩小 100 倍。代码就地改 w、算完还原。' },
        { concept: '逐算子检查', code: 'gradcheck.py:check_op', takeaway: '随机张量 R 当 dout: 任何算子都能单独验, 不用搭整个模型, 错在哪个算子直接定位。' },
        { concept: '多层反向', code: 'model.py:transformer_backward', takeaway: 'reversed(range(n_layer)): 从最后一层往第一层逐层回传, 和前向正好相反。前向是 h = tok + pos, 所以 tok_emb 和 pos_emb 最后收到同一个 dh。' },
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
      subtitle: '读完你能说清 Adam 每步到底走多远, 以及生成为什么只能一个 token 一个 token 往外挤。',
      tldr: '- Adam 是优化器, 负责拿着梯度去改参数。它给每个参数单独定步长: 一阶矩 (梯度的滑动平均) 定方向, 二阶矩 (梯度平方的滑动平均) 把步长按坐标归一化。结果是每步每个参数的位移大约就是 lr (学习率), 和梯度本身多大无关。\n- 采样是训练完拿模型写字: 不再算 loss, 而是:\n  1. 只取序列最后一个位置的 logits。\n  2. 按概率抽一个字符接到后面。\n  3. 再把整段送进去重来一遍。',
      question: '训练时 64 个位置一次算完, 生成时为什么只能一个一个来?',
      evolution: {
        title: '先把参数挪到位, 再把分布变成字',
        subtitle: '根问题: 梯度只给方向不给步长; 训好的模型也只给分布, 不给字。',
        steps: [
          { name: 'SGD', pain: '(原点) 梯度指向 loss 上升最快的方向, 参数该往反方向挪多远?',
            fix: '所有参数共用一个 lr: $W \\leftarrow W - \\text{lr}\\cdot g$; 可各层梯度尺度差几个数量级' },
          { name: 'Adam', year: 2014, pain: '一个 lr 被最陡的方向卡死, 平缓的方向几乎不动',
            fix: '每个坐标除以自己梯度平方的滑动平均开根号 $\\sqrt{\\hat v}$, 每步约走 lr; 2000 步 loss 从 4.17 降到 2.0 左右' },
          { name: '贪心解码', pain: '训完只拿到下一个字的分布 [V], 一个字还没写出来',
            fix: '每步取概率最大的字 (argmax) 接上再前向; 自带 ckpt 实测几十个字后只剩 “the the the”' },
          { name: 'temperature + top-k', pain: 'argmax 每步都挑最稳的字, 一旦进入循环就出不来',
            fix: '按概率抽样: temperature 调分布尖平, top-k 砍掉长尾里离谱的字' },
        ],
      },
      code: 'llm_basic/{optim.py,sample.py,train.py}',
      points: [
        {
          key: true,
          title: '除以 √v̂: 每个参数有自己的学习率',
          body: 'Adam 把每个参数的梯度除以这个参数自己梯度的典型大小 $\\sqrt{\\hat v}$, 相当于每个参数有了自己的学习率。对比 SGD (最朴素的优化器): 所有参数共用一个学习率 lr, 每步 $W \\leftarrow W - \\text{lr}\\cdot g$, 意思是每个参数沿梯度反方向走 “lr 乘梯度” 那么远。问题是 Transformer 各层的梯度尺度差几个数量级: 梯度大的参数一步跳得很远, 梯度小的几乎不动。\n- SGD 的上限: lr 超过 $2/\\lambda_{\\max}$ 就发散, $\\lambda_{\\max}$ 是 loss 曲面最陡那个方向的曲率。所以 lr 只能按最陡的方向定, 平缓方向就被拖得几乎不动。\n- Adam 的做法: 给每个参数分别记两个滑动平均, m 是梯度本身的平均 (管方向), v 是梯度平方的平均 (管大小)。更新量是 $\\hat m/\\sqrt{\\hat v}$: 梯度除以自己的典型大小, 结果每个元素都在 ±1 上下。再乘 lr, 每步每个参数的位移约等于 lr, 跟这个方向的坡度基本无关。\n- 偏置修正: m、v 从 0 起步, 前几步的平均值被低估 (第 t 步只攒到真实值的 $1-\\beta^t$ 倍), 所以除以 $1 - \\beta^t$ 补回来。第 1 步修正后 $\\hat m/\\sqrt{\\hat v}$ 正好等于 $\\mathrm{sign}(g)$ (只剩正负号), 每个元素恰好移动 lr; optim.py 的自检断言了这一点。',
        },
        {
          title: '训练并行, 生成串行, 这条依赖链省不掉',
          body: '训练时所有位置一次前向算完, 生成时只能一个 token 一个 token 往外出。\n- 训练: 完整答案 y 事先已知 (这叫 teacher forcing: 每个位置的输入用真实的前文, 而不是模型自己猜的)。因果 mask 让 T 个位置互不偷看, 所以一次前向就能算完所有 64 个位置的 CE (交叉熵 loss)。\n- 生成: 第 t+1 个 token 的输入里得包含第 t 个, 而第 t 个要等模型抽出来才有。所以只能一个一个来, 没法并行。\n生成能省的只有重复计算: sample.py 每生成 1 个 token, 就把整段 (含前面已经算过的 token) 重新前向一遍; 每次前向注意力是 $O(T^2)$, 生成 T 个 token 就要重算 T 次, 按它的注释整体是 $O(T^3)$。前面的 token 在每层算出的 K、V 每次都一样, 阶段 5 的 KV cache 把它们存下来, 省掉的就是这部分重复计算。',
        },
        {
          title: '三个开关默认全关, 开了也不一定更好',
          body: 'optim.py 里有三个可选项, 默认都不启用, 跑的就是原始 Adam 加恒定 lr:\n- AdamW 解耦衰减 (weight_decay): 每步把参数自身往 0 拉一点, 防止参数越长越大。\n- 全局范数裁剪 (grad_clip): 所有梯度拼成一个长向量, 长度超过阈值就整体等比缩小, 防单步跳太远。\n- warmup+cosine (--cosine): 学习率先线性升, 再按余弦曲线降到 lr/10。\nREADME 的实测, 都是 2000 步:\n- 2 层 + AdamW(0.1) + clip(1.0) + cosine, 78,656 参数: val_loss 2.042。\n- 默认 1 层、三个开关全关, 45,568 参数: val_loss 1.986。\n开了反而更差。原因是步数这么少, cosine 过早把 lr 压到 3e-5, 多出来那层还没学起来。这一节演示的是这些开关的实现, 步数少时开了不一定更好。',
        },
      ],
      links: [
        { from: 'Adam', to: 'llm_train 的稳定性工具', body: '阶段 3 的 warmup、cosine、grad clip、混合精度的 loss scaling, 都是在 Adam 这个更新式外面再加一层保护, 让大模型训练不炸。' },
        { from: 'sample.py', to: 'llm_infer 的采样', body: 'temperature 把 logits 除以一个数 (小于 1 分布更尖, 大于 1 更平), top-k 只保留概率最高的 k 个候选。两者都只是对 logits 的后处理, 不碰参数。推理章节在同一个位置接着加 top-p、min-p、重复惩罚。' },
        { from: '每步重算整段前向', to: 'KV cache', body: '第 t 步和第 t+1 步的输入只差最后一个 token, 前 t 个 token 在每层算出的 K/V 完全一样。把它们缓存下来, 下一步就只算新 token 的那一份。' },
      ],
      sourceRows: [
        { concept: '一阶矩', code: 'optim.py:adam_step', takeaway: '$m = \\beta_1 m + (1-\\beta_1)g$: 梯度的指数滑动平均, 管更新方向。$\\beta_1=0.9$ 相当于对最近约 $1/(1-\\beta_1) = 10$ 步的梯度做平均, 抹掉单步的抖动。' },
        { concept: '二阶矩', code: 'optim.py:adam_step', takeaway: '$v = \\beta_2 v + (1-\\beta_2)g^2$: 梯度平方的滑动平均, 管更新大小。除以 $\\sqrt{\\hat v}$ 后每个坐标的步长都被归一化到 lr 量级。' },
        { concept: '偏置修正', code: 'optim.py:adam_step', takeaway: '$m/(1-\\beta_1^t)$、$v/(1-\\beta_2^t)$: 补回从 0 起步造成的低估。optim.py 的自检断言第 1 步更新恰好是 $\\text{lr}\\cdot\\mathrm{sign}(g)$, 每个元素正好走 lr。' },
        { concept: 'AdamW 与 L2 的区别', code: 'optim.py:adam_step', takeaway: '$\\lambda W$ 加在 update 上而不是加进 $g$, 因此不被 $1/\\sqrt{v}$ 缩放, 所有参数按同一比例衰减; 加进 g 的 L2 会被梯度大的参数自己抵消掉。1 维的 gain/bias 不衰减。' },
        { concept: '全局范数裁剪', code: 'optim.py:clip_grad_norm', takeaway: '所有参数的梯度拼成一个长向量算 $\\|g\\|$, 整体同乘 $\\min(1, \\text{max\\_norm}/\\|g\\|)$: 只缩不放, 方向不变。逐参数裁会改变梯度方向。' },
        { concept: 'warmup + cosine', code: 'optim.py:cosine_lr', takeaway: '前 warmup 步 lr 从接近 0 线性升到 max_lr (一开始参数还乱, 先走小步), 之后沿半个余弦曲线降到 min_lr (后期走小步收敛)。' },
        { concept: '自回归循环', code: 'sample.py:generate', takeaway: '每轮五步:\n1. 取 logits[0, -1] (只要最后一个位置)。\n2. 除以 temperature。\n3. top-k 过滤。\n4. 按概率抽样。\n5. append 到 ids。\n然后整段再 forward 一次。' },
        { concept: '滑动窗口', code: 'sample.py:generate', takeaway: 'pos_emb 只有 T_max = 64 行, 第 65 个位置没有对应的向量。所以上下文超长时只保留最后 64 个 token 送进模型, 更早的直接看不到。' },
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
      "title": "感受野: 这个位置能从哪些输入取得信息",
      "body": "感受野指的是: 某个位置的输出, 最多能受到哪些输入 token 的影响。它说的是信息有没有路可走, 不是模型「看得清不清楚」。\n本页的模型用因果全注意力: 位置 t 在每一层都能直接看到位置 0 到 t 的全部 token, 所以一层就连到了前面所有输入。\n滑动窗口注意力 (简称滑窗) 让每个位置只看最近的 W 个 token, 这 W 个里包括它自己, 更早的被 mask 挡掉。所以一层滑窗只能往前多看 W−1 个位置。但第 2 层的输入是第 1 层的输出, 而第 1 层的每个位置已经汇总过自己窗口里的内容, 所以每多叠一层, 能间接够到的范围就再往前伸 W−1 个。L 层叠下来, 上界是 1+L×(W−1): 那个 1 是位置自己, 每层再贡献 W−1 个更早的位置。\n- 例子: W=8、L=4 时是 1+4×7=29, 不是 4×8=32。32 把每层窗口里的「自己」重复算了 4 次。\n- 截断: 位置 t 前面 (含自己) 一共只有 t+1 个 token, 所以实际能够到的数还要和 t+1 取较小的那个。\n这只是结构上限: 信息有路传过来, 不代表训练后的模型记住了远处的细节。",
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
