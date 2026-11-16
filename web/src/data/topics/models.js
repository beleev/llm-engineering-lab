import { canonicalizeRelated } from '../../utils/consolidate.js'
import { chapterLabels } from '../chapterLabels.js'
import { mergePage } from '../../utils/consolidate.js'
// 阶段 2 · llm_models 的扩展章节: 2024–2025 的新零件。每章一个 (或两个) 实验台, 挂载关系见 data/labmap/models.js。
// models-mtp 的章节登记在 data/models.js 的 modelChapters 里, 所以不出现在这里的 chapters。
const A = 'llm_models/layers/core/attention.py'
const RUN = 'python -m llm_models.run_models'

const topic = {
  stage: 'models',
  chapters: [
    { route: 'models-generation', label: 'KV cache 生成', hint: '旧 token 的 K/V 不会变: prefill 一次, 之后每步只算 1 个' },
    { route: 'models-qknorm-yarn', label: 'QK-Norm 与 YaRN', hint: '一个管 logit 别炸, 一个管 RoPE 能不能拉长' },
    { route: 'models-mamba', label: 'Mamba 选择性 SSM', hint: '输入自己决定 Δ: 记谁、忘谁; 状态大小恒定' },
    { route: 'models-moe-balance', label: 'Aux-loss-free 均衡', hint: '偏置只管选谁上场, 不碰门控权重' },
    { route: 'models-dsa', label: 'DSA 稀疏注意力', hint: '便宜的 indexer 先挑 k 个 key, 主注意力教它' },
    { route: 'models-gptoss', label: 'GPT-OSS 结构', hint: '滑窗与全注意力交替, 每个 head 多一个 sink' },
    { route: 'models-llada', label: 'LLaDA 扩散语言模型', hint: '随机比例遮盖训练, 采样时把没把握的遮回去' },
    { route: 'models-var', label: 'VAR 逐尺度生成', hint: '先 1×1 再 2×2、4×4, 由粗到细, 级内并行' },
    { route: 'models-multimodal', label: '图像与音频怎样接入语言模型', hint: '图切 patch 当前缀, 声谱走 cross-attention' },
  ],
  pages: {
    'models-mtp': {
      widgets: ['MtpLab'],
      title: 'SWA · MTP · 混合线性 — 三种给 Transformer 减负的改法',
      subtitle: '读完这章, 你看一个新模型时能马上分辨它动了哪一处:\n- 把注意力的 mask 裁窄了, 每个 token 只看最近一段 (Mistral 的 SWA)?\n- 让每个位置多预测几步 (MTP)?\n- 还是把装历史的容器从 KV cache 换成了定长状态 (Qwen3-Next)?',
      tldr: '- SWA (滑动窗口注意力): 不新增任何类, 只把下三角 mask 换成带状的, 每个 token 只看最近 W 个。KV cache 从随长度涨的 $O(T)$ 封顶到 $O(W)$。\n- MTP (多 token 预测): 在主干后面接级联模块, 共享 lm_head, 让位置 i 除了预测下一个, 还预测下下个。每个位置的监督从 1 份变成 K+1 份; 推理时还白送一份投机解码的草稿。\n- 混合线性 (Qwen3-Next): 大部分层用固定大小的状态矩阵代替 KV cache, 每 4 层留 1 层全注意力兜底精确召回。',
      question: '每层只看最近 W 个 token, 远处的信息怎么过来? 一次预测好几个 token, 为什么不算偷看未来?',
      evolution: {
        title: '读的账和写的账分开省',
        subtitle: '根问题: 读得越长, 每层要存的 K/V 越多; 写的时候一次只出 1 个 token',
        steps: [
          { name: '全因果注意力', pain: '(原点) 当前 token 可能用到任意远的历史, 所以历史全得留着', fix: 'LLaMA 每层看全部历史: 召回最准, 每层 KV cache 随上下文长度 T 一直涨' },
          { name: 'SWA', year: 2023, pain: '上下文一长, 每一层的 cache 都跟着 T 涨', fix: 'Mistral 每层只看最近 W 个, cache 封顶 W; 更远的信息靠逐层接力传过来' },
          { name: '混合线性', year: 2025, pain: '窗口外的 token 被直接丢掉, 只剩层间接力这条间接通路', fix: 'Qwen3-Next 用定长状态矩阵把全部历史压进去, 每 4 层留 1 层全注意力兜底' },
          { name: 'MTP', year: 2024, pain: '读的账省下了, 写仍一次只出 1 个 token; 训练时每个位置也只学下一个', fix: '位置 i 级联地多预测 i+2、i+3…: 训练信号更密, 对 t+2 的预测还能当投机解码草稿' },
        ],
      },
      code: 'llm_models/models/language_models/{mistral.py,mtp.py,qwen3_next.py} · llm_models/layers/sparse/linear_attention.py',
      points: [
        { title: '深度换宽度', body: 'T=64、W=8 时, 全因果 mask 里每个 token 看它之前的全部, 可见格子 2080 个; 带状 mask 每个 token 只看最近 8 个, 可见格子 484 个。\n单层只看 W 个位置。但第 2 层的输入是第 1 层的输出, 而第 1 层里每个位置已经汇总过它自己的 8 个邻居。所以第 2 层看 8 个邻居, 等于间接看到了 15 个位置, 信息跨层接力。\nL 层的理论感受野上界是 $1+L(W-1)$, 位置 t 再取不超过 t+1 (前面总共就那么多 token)。Mistral 32 层、W=4096 时是 131041。这是结构上存在的路径, 不保证训练后真的会用。\ndemo 特意用 1 层模型验证: 改位置 0 的 token, 只有位置 0 到 7 的输出变了, 位置 8 以后纹丝不动。' },
        { title: '换一张 mask 就是换一个架构', body: '注意力里的 QKV 投影、softmax、乘 V 这些计算, 几个模型都一样; 谁能看见谁, 全由 mask 决定:\n- 全因果 (下三角): LLaMA。\n- 带状: Mistral。\n- 带状加 sink (开头几列永远可见): GPT-OSS。\n- top-k (每行只留打分最高的 k 个): DSA。\n所以 Mistral 和 LLaMA 的参数量完全相同, demo 里都是 3,076,352: 结构差异只在一张不占参数的布尔表上。推理时 Mistral 每层只留最近 W 个 K/V (rolling buffer), 读了 21 个 token, 每层 cache 仍只有 8 个。', key: true },
        { title: '级联保因果', body: 'MTP 让位置 i 多预测几步, 但不能偷看未来。DeepSeek-V3 的做法是串行级联:\n第 k 级在位置 i 的输入, 是上一级的 hidden 拼上真实 token $t_{i+k}$ 的 embedding, 过一个 Transformer Block, 输出去预测 $t_{i+k+1}$。例: 主干在位置 i 看着 $t_{\\le i}$ 预测 $t_{i+1}$; 第 1 级再把真实的 $t_{i+1}$ 喂进来, 预测 $t_{i+2}$。\n每深一级, 多看一个真 token (训练时用真实答案, 叫 teacher forcing)、多预测一步, 因果链没断: 预测 $t_{i+2}$ 时确实看到了 $t_{i+1}$, 但没看到 $t_{i+2}$ 自己。\n总损失 $= \\mathrm{CE}(\\text{main}) + \\lambda\\cdot\\mathrm{mean}_k\\,\\mathrm{CE}(\\text{mtp}_k)$: 主 head 的交叉熵, 加各级 MTP 交叉熵的平均乘 λ。DeepSeek-V3 的 $\\lambda$ 取 0.3, 后期降到 0.1 (本库固定 0.3)。demo 里主干只有 2 层, 加一级 MTP 参数从 1,666,304 涨到 2,503,168; 真实的 61 层模型同样加一级只多约 2%。' },
        { title: '状态替代缓存', body: 'Gated DeltaNet 是一种线性注意力: 不存历史 K/V, 只维护一个 Dh×Dh 的状态矩阵 S (Dh 是每头维度)。每来一个 token, 先用 k 从 S 里读出当前存的值, 擦掉, 再写入新的 v (这叫 delta rule); 另有一个 α 门负责整体遗忘, 越旧的信息衰减越多。读取时用 q 乘 S。\n它的 “cache” 就是这个 S, 与读了多少 token 无关。demo 里 T 从 8 涨到 32, 同类层加总的缓存元素数 (batch 2):\n- 注意力层 (共 2 层): 4096 变 16384, 涨 4 倍。\n- delta 层 (共 6 层): 恒为 49152。\n代价是压缩有损: 定长矩阵装不下任意长的历史, 精确召回很久以前的某个 token 不如注意力。所以 Qwen3-Next 每 4 层留 1 层全注意力, 兜底精确召回。' },
        { title: '序列短的时候, 状态反而更大', body: '上面的 49152 比 16384 大, 看起来状态更费。拆到单层、单条序列再比:\n- 注意力层: 每个 token 存 128 个数 (2·Hkv·Dh = 2×2×32: K 和 V 各一份, 2 个 KV head, 每头 32 维), 随 T 线性涨。\n- delta 层: 固定 4096 个数 (H·Dh·Dh = 4×32×32: 4 个头, 每头一个 32×32 的矩阵), 不随 T 变。\n4096 ÷ 128 = 32, 所以 T 超过 32 之后, 状态才比 KV cache 小。线性注意力的优势只在长上下文才出现。' },
      ],
      links: [
        { from: 'build_sliding_window_mask', to: 'Mistral.forward', body: 'Mistral 与 LLaMA 的全部结构差异就是这一张带状 mask, 连参数量都一样; forward 里多做的只是每步把 cache 裁到最近 W 个。' },
        { from: 'MTPModule', to: 'lm_head (共享)', body: '每级 MTP 只新增一个拼接投影 (2D 维到 D 维) 加 1 个 Block。embedding 和输出头都借主干的, 所以丢掉 MTP 模块后, 主 logits 与同权重的 LLaMA 逐元素相等, demo 有断言。' },
        { from: 'mtp_logits', to: 'speculative decoding', body: 'MTP 第 1 级对 t+2 的预测可以直接当草稿, 让主模型一次验证多个 token, 这就是投机解码, 对应 llm_infer/m07。本库没实现这条路径, 带 cache 生成时只跑主干。' },
      ],
      sourceRows: [
        { concept: '带状掩码', code: 'masks.py:build_sliding_window_mask', takeaway: '可见条件是 ($j \\le i$) 且 ($j > i-W$): key 在 query 之前, 且距离不超过 W。sink_tokens > 0 时额外保留开头 S 列。' },
        { concept: 'KV 上限', code: 'mistral.py:kv_cache_entries', takeaway: 'rolling buffer 只留最近 W 个 K/V, 每层条目数是 $\\min(T, W)$。读了 21 个 token, 每层 cache 仍然只有 8 个。' },
        { concept: '级联输入', code: 'mtp.py:MTPModule.forward', takeaway: 'h 和 embedding 各自过 RMSNorm 再拼接。h 经过多层残差累加, 数值尺度比 embedding 大得多, 不先归一化就得让投影矩阵花容量去对齐尺度。拼完投影回 d 维, 过 Block。' },
        { concept: '标签对齐', code: 'loss.py:MTPLoss.compute', takeaway: '第 k 级在位置 i 的目标是 labels[i+k], 把 labels 整体左移 k 位; 末尾 k 个位置没有未来 token, 填 −100, cross_entropy 会跳过它们。' },
        { concept: 'delta rule', code: 'layers/sparse/linear_attention.py:GatedDeltaNet', takeaway: '$S \\leftarrow \\alpha(S - \\beta k(k^\\top S)) + \\beta k v^\\top$: 先擦掉 k 方向的旧值, 再写入 β 倍的新值, 整体乘遗忘门 α。q/k 先做 L2 归一化, $I - \\beta k k^\\top$ 才是收缩映射, 状态不会数值爆炸。' },
        { concept: '混合排布', code: 'qwen3_next.py:layer_types', takeaway: '[Δ,Δ,Δ,A,…] 周期排布, 每 4 层的最后一层是全注意力。因果 mask 和 rope 对 Δ 层是 no-op (递推天然因果, 位置靠遗忘门隐式编码), 主干循环不区分层类型。' },
      ],
      snippetTitle: 'SWA 与 MTP 的核心差异行',
      snippet: `# Mistral = LLaMA + 一张带状 mask
visible = (j <= i) & (j > i - window_size)      # 谁能看见谁

# MTP 级联: 位置 i 的第 k 级多看一个真实 token
shifted[:, :-k] = idx[:, k:]                     # teacher forcing
h = mtp_block(cat([norm(h), norm(emb(shifted))]))
loss = ce_main + lam * mean(ce_mtp_k)            # 联合训练`,
      source: [
        'llm_models/models/language_models/mtp.py:MTPModule',
        'llm_models/layers/sparse/linear_attention.py:GatedDeltaNet',
      ],
      run: 'python -m llm_models.run_models.language_models.mistral.infer_mistral',
    },

    'models-generation': {
      title: 'KV cache 生成 · 推理为什么不用每步重算整段',
      subtitle: '看完这章, 你能对着 generate() 的三个分支说出每一支在干什么, 也能解释 MLA 为什么只缓存 latent 而不缓存 K/V。',
      tldr: '- 模型生成是一个 token 一个 token 往外吐的。每吐一个, 序列只长 1, 而旧 token 在每一层算出来的 K 和 V 一个字节都不变。把它们存下来 (这就是 KV cache), 生成 N 个 token 时累计送进模型的 token 数从 $O(N^2)$ 降到 $O(N)$。\n- MHA/GQA: 存 K、V 本体。\n- MLA: 只存一个低秩的 latent c_kv 和所有 head 共享的 k_rope, 每步现场把它们升维成 K 和 V, 用一点算力换 cache 变小。',
      question: '为什么只缓存 K/V, 不缓存 Q? 上下文窗口一满, 整个 cache 为什么就作废了?',
      evolution: {
        title: '把重复计算一刀刀剔掉',
        subtitle: '根问题: 自回归是串行的, 每出 1 个 token 都要跑一次前向, 前缀越长这一次越贵',
        steps: [
          { name: '每步重算', pain: '(原点) 第 t 个 token 要等前 t−1 个都生成完, 只能一步步来', fix: '每步把整段前缀重新前向一遍: 生成 N 个 token 共前向 $O(N^2)$ 个 token' },
          { name: 'KV cache', pain: '因果 mask 下旧 token 的 K/V 每步算出来都一样, 全是重复劳动', fix: '把每层的 K/V 存起来, 新 token 只算自己那一份再去查: $O(N^2) \\to O(N)$' },
          { name: 'prefill + decode', pain: 'cache 起初是空的; prompt 也逐个喂的话, 又退回串行', fix: 'prefill 整段并行填满 cache, decode 每步只喂 1 个; 窗口一满就整段重新 prefill' },
        ],
      },
      code: 'llm_models/utils/generation.py · llm_models/layers/core/attention.py',
      points: [
        { title: '只有 K/V 值得存', body: '注意力层给每个 token 算三个向量: Q (query, 它要去问什么)、K (key, 它能被谁问到)、V (value, 被问到时交出什么)。生成时第 t 步只需要最后一个 token 的输出: 拿最新 token 的 Q, 去查所有历史 token 的 K 和 V。\n- 旧 token 的 Q: 只在它自己那一步用来向前文提问, 之后再也用不上。\n- 旧 token 的 K/V: 因果 mask 让每个 token 只看它前面的, 后面来什么都不会改它。所以旧 token 在每一层的 K、V 永远不会变。\n不变的东西算一次存起来就够。于是缓存 K/V, 每步只前向 1 个新 token。本库每层一个 dict, GQA 往里放 cache["k"]、cache["v"]。LLaMA 的 infer 脚本里生成 200 个 token, 有 cache 和无 cache 的输出完全一致, 加速约 5.7x。', key: true },
        { title: 'prefill、decode 是两种负载', body: 'cache 一开始是空的, 得先把 prompt 塞进去。所以生成分两个阶段:\n- prefill: 一次吃下整段 prompt, T 个 token 并行算, 顺手把每层的 K/V 都写进 cache。算力密集: 矩阵乘法够大, GPU 能跑满。\n- decode: 之后每步只处理 1 个新 token, 但要把所有层的 cache 读一遍。访存密集: 计算量很小, 时间都花在搬数据上, 而且只能串行, 一步等一步。\ngenerate() 看 cache.pos (已缓存的 token 数) 是不是 0 来分这两条路: 是 0 就 prefill, 否则 decode 只喂最后一个 token idx[:, -1:]。阶段 5 的调度器就是冲着这两种负载设计的: prefill 吃算力, decode 吃带宽, 混着排才能把卡用满。' },
        { title: 'MLA 存 latent, 不存 K/V', body: 'MLA (Multi-Head Latent Attention, DeepSeek 用的注意力) 换了一种存法。它不存每个 head 的 K 和 V, 而是先把输入压成一个低维向量 c_kv (叫 latent, 维度 r), 存的是它; 用的时候再乘两个固定矩阵 k_up、v_up 升维回 K 和 V。cache 里只有两样: c_kv [S, r], 和做过 RoPE (旋转位置编码) 之后、所有 head 共享的 k_rope [S, rope]。\n多算一点升维, 换 cache 变小:\n- DeepSeek-V3: r=512 加 rope=64, 每 token 每层存 576 个数。\n- 同规模的 MHA (128 头 × 128 维, K 和 V 各一份): 要 32768 个。MLA 省约 98%。\n本库 demo 的小配置 (d_model=512, 8 头, r=64, rope=32): MHA 每 token 每层 1024 个数, GQA 256 个, MLA 96 个。' },
        { title: 'c_kv 不能带 RoPE', body: '先说为什么 latent 值得存。升维矩阵 k_up 是固定的, 推理时可以把它提前乘进 Q 侧的投影里, 这样 Q 直接和 c_kv 做点积, K 根本不用真的升维, 这一步叫吸收 (absorb)。\nRoPE 是随位置变化的旋转: 位置 m 的 K 要转 m 倍的角度。如果 c_kv 带了 RoPE, 旋转就夹在 “升维矩阵” 和 “Q” 之间, 升维矩阵乘不进 Q 侧, 吸收做不了, latent 就白存了。\n所以每个 head 的维度拆成两段:\n- nope 段: 走 latent, 不旋转, 可以吸收。\n- rope 段: 单独一个小投影, 旋转后作为 k_rope 存起来, 所有 head 共享一份。\n本章的教学代码每步照常把 c_kv 升维, 没做吸收。llm_infer 的 m18 有 absorb=True 分支。' },
        { title: '窗口一满, cache 整段作废', body: '模型有个最大长度 max_len (RoPE 表和 mask 都只建到这么长)。序列长到 max_len 之后, 每多生成一个 token, 就得把最早的一个挤出窗口。挤掉之后旧 cache 为什么不能接着用:\n- 内容变了: 第 2 层起, 每个 token 的输入是上一层看着全部前文算出来的。被挤掉的那个 token 参与过它们的计算, 留下来的 K/V 都 “含” 着它, 和按新窗口重算的结果不同。\n- 位置变了: 代码里重新 prefill 时位置从 0 重新编号, 旧 K/V 里烙着的 RoPE 角度对不上。\n所以 generate() 发现 cache.pos + 1 > max_len 时, 直接新建一个 KVCache, 把最后 max_len 个 token 整段重新 prefill。窗口满了之后每一步都走这条分支, 等于退回每步重算。' },
      ],
      links: [
        { from: 'llm_basic/sample.py', to: 'GenerationMixin.generate', body: '阶段 1 的采样每生成一个字就把整段重新 forward 一遍; 这里的 generate() 加上 cache 分支, 做同样的事但不重算, benchmark 断言两种方式的输出逐 token 完全一致。' },
        { from: 'KVCache', to: 'llm_infer KV 与缓存内存', body: '这里的 cache 很朴素: 每层一个 dict, K/V 用 torch.cat 一直往后拼, 只服务一条请求。阶段 5 把它换成分页的 block 表, 好让同一张卡同时服务多个请求。' },
        { from: 'MultiHeadLatentAttention', to: 'DSA', body: 'DSA 用的就是这里的 MLA 和它的 latent cache, 只是在 cache 里再多存一份 indexer 的 key (idx_k), 用来给旧位置打分选 top-k。' },
      ],
      sourceRows: [
        { concept: '三个分支', code: 'generation.py:GenerationMixin.generate', takeaway: '无 cache: 每步喂整个 (裁到 max_len 的) 前缀; pos==0 或窗口已满: (重新) prefill 整段; 其余情况 decode, 只喂最后一个 token idx[:, -1:]。' },
        { concept: '窗口满了就作废', code: 'cache.pos + 1 > max_len', takeaway: '条件成立就新建 KVCache 重新 prefill。原因: 窗口左移挤掉了最早的 token, 第 2 层起的旧 K/V 都依赖它, 位置也从 0 重新编号。窗口满后每一步都走这条分支。' },
        { concept: '等价性断言', code: 'generation.py:benchmark_kv_cache', takeaway: '贪心 (temperature=0) 生成两遍, 一遍不用 cache 一遍用, 再 assert torch.equal。cache 只许变快, 不许改结果; 不一致说明 position_ids 或 mask 切片没和 cache.pos 对齐。' },
        { concept: 'MLA 的 cache', code: 'cache["c_kv"], cache["k_rope"]', takeaway: '只往 cache 里追加本步的 latent c_kv 和 k_rope; 然后 k_up / v_up 每步对全部 S 个位置现场升维出 K 和 V, 升维的结果不存。' },
      ],
      snippetTitle: 'generate() 的骨架',
      snippet: `cache = KVCache(num_layers) if use_cache else None
for _ in range(max_new_tokens):
    if cache is None:
        inp = idx[:, -max_len:]          # 每步重算整个前缀: O(T) 个 token
    elif cache.pos == 0 or cache.pos + 1 > max_len:
        cache = KVCache(num_layers)      # prefill; 或窗口已满, 旧位置全部失效
        inp = idx[:, -max_len:]
    else:
        inp = idx[:, -1:]                # decode: 只喂 1 个新 token
    logits = model(inp, cache=cache)[:, -1]
    idx = cat([idx, sample(logits)], dim=1)

# attention 内部 (每层):
k = cat([cache["k"], k_new]); v = cat([cache["v"], v_new])   # MHA / GQA
c_kv = cat([cache["c_kv"], c_new]); k, v = k_up(c_kv), v_up(c_kv)  # MLA`,
      source: ['llm_models/utils/generation.py:GenerationMixin.generate'],
      run: `${RUN}.language_models.llama.infer_llama`,
    },

    'models-qknorm-yarn': {
      title: 'QK-Norm 与 YaRN · 两个给注意力兜底的补丁',
      subtitle: '- 管数值: 注意力 logit (softmax 之前的分数) 不许随 q、k 的范数一起长大。\n- 管位置: RoPE 超出训练长度时, 不许看到没见过的角度。\n读完你能说出两者各改了哪一行, 以及为什么顺序不能调。',
      tldr: '- QK-Norm: q、k 在 RoPE 之前各过一次 head_dim 上的 RMSNorm (把向量长度拉回固定值), 于是 logit 只剩夹角和一个可学增益, 上界锁死在 $g^2\\sqrt{d}$。\n- YaRN: RoPE 的每对维度像一根转速不同的表针。按 “训练长度内转了几圈” 给每个频率分段缩放: 高频不动, 低频 $\\div s$ (s 是上下文要扩几倍), 中间线性过渡, 再乘 mscale 补回注意力温度。',
      question: '都已经除以 $\\sqrt{d}$ 了, logit 为什么还会炸? 长度外推时, 为什么先出问题的是低频维度而不是高频?',
      evolution: {
        title: '先管住范数, 再管住角度',
        subtitle: '根问题: 注意力 logit 由 q、k 的范数和夹角决定, 两者一旦超出训练时见过的范围, softmax 就失控',
        steps: [
          { name: '缩放点积', year: 2017, pain: '(原点) 点积的方差随维度 d 线性涨, softmax 一开训就饱和', fix: '除以 $\\sqrt{d}$ 抵消维度; 但训练中 q、k 的范数自己会涨, 这一项没人管' },
          { name: 'QK-Norm', pain: 'q、k 范数变大, logit 按平方涨, softmax 塌成 one-hot 收不到梯度', fix: 'RoPE 之前对 q、k 各做一次 RMSNorm: logit 只剩夹角 $\\cos\\theta$ 和一个可学增益' },
          { name: 'PI', year: 2023, pain: 'logit 只看角度了; 位置超过训练长度, 低频维度转到没见过的角度', fix: '所有 RoPE 频率一律 $\\div s$, 角度全拉回训练范围; 高频也被压扁, 相邻 token 分不开' },
          { name: 'YaRN', year: 2023, pain: '高频维度训练时早已转遍所有角度, 陪着一起压扁是白亏', fix: '按训练期转过的圈数分段: 高频不动, 低频 $\\div s$, 中间过渡, 再乘 mscale 补回温度' },
        ],
      },
      code: `${A} · llm_models/layers/core/position_encoding.py`,
      points: [
        { title: 'logit 随范数二次增长', body: '注意力 logit 是 $q\\cdot k/\\sqrt{d}$: q 和 k 的点积除以 $\\sqrt{d}$。点积等于两个向量的长度 (范数) 相乘再乘夹角的余弦。$\\sqrt{d}$ 只抵消维度数带来的方差: d 个随机分量加起来, 方差正比于 d。它不抵消范数: 训练中 q、k 的范数一起涨 2 倍, logit 就涨 4 倍。\ndemo 把 W_q、W_k 的权重乘 10: 最大 logit 从 1.20 冲到 120.31, 正好 100 倍。\n后果:\n- softmax 塌成 one-hot (概率几乎全压在一个 key 上)。softmax 的导数 $\\mathrm{diag}(p) - pp^\\top$ 在这时趋近 0, 这个 head 收不到梯度, 等于死了。\n- 低精度 (bf16 / fp16) 下 logit 这么大还会直接溢出。', key: true },
        { title: 'QK-Norm 只留方向和一个增益', body: '做法是对 q、k 各做一次 RMSNorm, 作用在每个 head 的维度上 (Qwen3 / OLMo-2 / Gemma-3 都这么干)。归一化后 q、k 的范数被拉回 $\\sqrt{d}$ 乘可学增益 g, 于是 $\\text{logit} = g^2\\cdot\\sqrt{d}\\cdot\\cos\\theta$: 只剩夹角和增益, 和 W_q、W_k 多大无关。\n同样把权重乘 10, 最大 logit 稳在 3.44 不动。模型仍能调可学的 g 决定注意力有多尖 (g 大, logit 范围大, softmax 更尖), 但只剩这一个旋钮。\n必须放在 RoPE 之前。RoPE 是纯旋转, 不改变范数, 所以先归一再旋转, 范数仍是定的; 反过来先旋转再乘逐维增益 g, 成对的两个维度被乘上不同的数, 旋转结构就被破坏了。' },
        { title: '低频维度没见过那么大的角度', body: 'RoPE 把 q、k 的每两个维度配成一对, 位置 m 处把这一对转 $m\\cdot\\theta_i$ 的角度, 不同对的转速 $\\theta_i$ 不同。把每对维度想成一根表针:\n- 高频针 (θ 大): 训练长度 L 内转了几百圈, 什么角度都见过, 位置再长也只是再转几圈。\n- 低频针 (θ 小): 波长大于 L, 训练时一圈都没转完, 模型只见过一小段弧。位置一超过 L, 它就指向训练分布之外的方向, 这是外推崩坏的根源。\n衡量标准是越界倍数: 位置 $sL$ 处 (扩到 s 倍长度) 的角度, 除以训练时见过的最大角度。1.00× 就是没越界; 什么都不做时, 低频维度是 4.00× (demo 用 s=4)。' },
        { title: '三种外推方案, 差在动哪些频率', body: '三种办法都是 “让针转慢一点”, 区别是让哪些针慢。demo 配置 d_head=64 (32 对维度)、L=2048、s=4:\n- PI (位置内插): 所有针一律慢 s 倍, 相当于把位置坐标除以 4。低频全部拉回 1.00×, 但高频也被压扁: 相邻 token 的角度差从 1.00 降到 0.25 rad, 近处的 token 分不开。\n- NTK-aware: 只改 RoPE 的 base, 让最高频不动、最低频恰好慢 s 倍, 中间按幂次过渡。高频几乎不动, 但中段压不够: 11 个低频维度里只有最低的那个回到 1.00×, 其余最多还越界 1.56×。\n- YaRN: 按训练期转过的圈数 r 分段。9 个最高频维度 (r>32) 完全不动, 11 个低频维度 (r<1) 全部 $\\div s$ 拉回 1.00×, 中间线性过渡, 两头都保住。再乘一个 mscale 补回注意力温度。\n边界: demo 只验证了角度层面的性质, 没训模型、没测 ppl。' },
      ],
      links: [
        { from: 'position · RoPE', to: 'scaled_inv_freq', body: '位置编码一章讲 RoPE 为什么能编码相对位置; 这里只改 $\\theta_i$ 这一张频率表 (代码里的 inv_freq), 旋转公式一行不动。' },
        { from: 'RMSNorm', to: 'q_norm / k_norm', body: '和 Block 里的 RMSNorm 是同一个类, 只是作用在 head_dim 上而不是 d_model 上, 每个 head 共享一组增益 g。' },
        { from: 'QK-Norm', to: 'llm_train 精度与稳定性', body: '阶段 3 从训练侧管稳定性 (梯度裁剪、warmup、loss scaling), 是在出事后兜底; QK-Norm 是从结构侧直接去掉一个不稳定源, 让 logit 根本炸不起来。' },
      ],
      sourceRows: [
        { concept: 'QK-Norm 开关', code: 'attention.py:GroupedQueryAttention', takeaway: 'self.q_norm = RMSNorm(head_dim) if qk_norm else None, k_norm 同理。forward 里在 _call_rope 之前调用: 先归一化, 再旋转。' },
        { concept: 'NTK-aware', code: 'base * factor ** (d / (d - 2))', takeaway: '指数 $d/(d-2)$ 正好让最后一个频率 (最低频) $\\div s$, 第一个频率 (最高频) 原样不变。d=64、s=4 时新 base ≈ 41,829。' },
        { concept: 'YaRN 分段', code: 'gamma = ((rotations - beta_slow) / (beta_fast - beta_slow)).clamp(0, 1)', takeaway: '$r = L\\cdot\\theta/2\\pi$ 是训练期转过的圈数; gamma 在 r 从 beta_slow=1 到 beta_fast=32 之间线性从 0 升到 1。$\\gamma=1$ 原样外推, $\\gamma=0$ 按 PI 内插 ($\\div s$)。' },
        { concept: '温度补偿', code: '0.1 * math.log(factor) + 1.0', takeaway: '这就是 mscale。上下文变长后 softmax 的分母项变多、分布变平, 用 mscale 把 logit 放大一点补回来。s=4 时 mscale = 1.1386; 本库把它乘进 cos/sin 表, q、k 各乘一次, logits 乘 mscale² = 1.2965。' },
      ],
      snippetTitle: 'QK-Norm 与 YaRN 的实现',
      snippet: `# QK-Norm: RoPE 之前, 对每个 head 的 q / k 归一化
q = q_norm(q)            # RMSNorm over head_dim, 带可学增益 g
k = k_norm(k)
q, k = rope(q), rope(k)  # 旋转不改范数
scores = q @ k.T / sqrt(d)          # |score| <= g² · sqrt(d)

# YaRN: 只改 RoPE 的频率表
theta = base ** (-arange(0, d, 2) / d)
r = L_train * theta / (2 * pi)                       # 训练长度内转了几圈
gamma = clip((r - beta_slow) / (beta_fast - beta_slow), 0, 1)
theta_new = (1 - gamma) * theta / s + gamma * theta  # 低频内插, 高频外推
mscale = 0.1 * ln(s) + 1`,
      source: ['llm_models/layers/core/position_encoding.py:scaled_inv_freq'],
      run: `${RUN}.foundation.rope_scaling.infer_rope_scaling`,
    },

    'models-mamba': {
      title: 'Mamba · 让状态空间模型学会挑着记',
      subtitle: '- 注意力: 所有历史的 K/V 都留着, 用的时候再挑。\n- SSM (状态空间模型): 只有一个固定大小的状态, 每来一个 token 就得当场决定留什么、丢什么。\n读完你能说清 Mamba 的答案 (让步长 $\\Delta$ 由输入自己决定), 以及它为此放弃了什么。',
      tldr: '- Mamba 不用注意力, 用一个递推: $h_t = \\exp(\\Delta_t\\cdot A)\\cdot h_{t-1} + \\Delta_t\\cdot B_t\\cdot x_t$, $y_t = C_t\\cdot h_t$。这个式子在说: 新状态等于旧状态乘一个衰减系数, 再加上当前输入写入的一份; 输出从状态里读出来。\n- $\\Delta_t$、$B_t$、$C_t$ 全由当前 token $x_t$ 线性投影得到, 这就是 “选择性”: Δ 大, 清掉旧状态、写入当前 token; Δ 接近 0, 当前 token 被跳过。\n- 解码每步只要 $O(1)$ 的状态, 和读过多少 token 无关。训练在生产实现里靠并行 scan, 教学版是逐 token 的顺序循环。',
      question: 'S4 这种参数不随位置变的 SSM 可以写成卷积、训练飞快, Mamba 为什么宁可把这个性质扔了?',
      evolution: {
        title: '在定长状态里学会挑着记',
        subtitle: '根问题: 要把任意长的历史带到当前 token, 内存和算力却不能跟着长度一直涨',
        steps: [
          { name: 'RNN', pain: '(原点) 历史可以任意长, 能留给它的内存却是固定的', fix: '压进一个定长隐状态: 解码每步 $O(1)$, 但训练只能逐 token 串行, 远处梯度会消失' },
          { name: '注意力', year: 2017, pain: '逐 token 串行训不快, 远处的信息要一步步传过来', fix: '历史全留着, 整段并行训练; 代价是 KV cache 随长度线性涨, 算力 $O(T^2)$' },
          { name: 'S4 (LTI SSM)', year: 2021, pain: 'cache 和算力都随长度涨, 超长序列撑不住', fix: '回到定长状态; 参数不随时间变, 能写成卷积整段并行训练; 但只能按距离加权' },
          { name: 'Mamba', year: 2023, pain: 'token 的分量只看离结尾多远, 关键词和废话分不开', fix: 'Δ、B、C 由当前输入投影得到, 记谁忘谁由 token 自己定; 写不成卷积, 训练改用 scan' },
        ],
      },
      code: 'llm_models/layers/sparse/ssm.py · llm_models/models/language_models/mamba.py',
      points: [
        { title: '一个 Δ 同时当写入门和遗忘门', body: 'SSM 原本是连续时间的方程 $h^\\prime(t) = A h(t) + B x(t)$: 状态 h 的变化率由自身 (乘 A) 和输入 (乘 B) 决定。token 是离散的, 所以要按一个步长 $\\Delta$ 走一步, 得到离散的递推:\n- 旧状态的保留率: $\\bar A = \\exp(\\Delta\\cdot A)$。代码里 A 恒负 (A = −exp(A_log)), Δ 恒正 (过 softplus), 所以它落在 $(0,1)$ 之间, 状态只会衰减不会爆炸。\n- 新输入的写入强度: $\\bar B \\approx \\Delta\\cdot B$。\n同一个 $\\Delta$ 出现在两处, 取两头时:\n- Δ 大: 保留率趋近 0, 写入变强。“忘掉过去, 记住这个”。\n- Δ 趋近 0: 保留率趋近 1, 写入趋近 0。“这个 token 当没看见”。\n一个标量同时控制 “忘” 和 “记” 两扇门, 而且它由当前 token 算出来: 每个 token 自己决定自己值不值得被记住。', key: true },
        { title: '参数不随位置变, 就只能按距离加权', body: 'LTI (线性时不变) 的意思是 Δ、B、C 固定, 不随位置或内容变。这时 token j 对最终状态的贡献只取决于它离结尾多远 (衰减了多少步), 和它是什么无关。\n它没法表达 “这个词重要、那个是废话”。语言建模要的恰恰是这个; induction (看到 “A B ... A” 要接 B) 和 selective copying (从一串 token 里只复制特定几个) 这类任务更是直接考它。实验台里 LTI 模式下, 关键 token “7” 的占比上限只有 1/10: 它只能和其它 9 个 token 平分。\n让 $\\Delta$ 随输入变就能挑着记。代价是系统不再时不变: S4 之所以训练快, 是因为固定参数的递推可以展开成一个固定的卷积核, 整段并行算; 参数随 token 变, 卷积核就不存在了, 训练只能换成 scan (沿序列逐步递推, 生产实现用并行 scan 算法加速)。' },
        { title: '状态大小与序列长度无关', body: '每层的 “cache” 只有两样: SSM 的状态 h, 形状 [D_inner, N] (d_inner 个通道, 每个通道 N 维状态); 加一小段卷积缓冲 (最近 d_conv−1 个输入)。infer 脚本读完 64 个 token 后每层仍是 conv (2,256,3) 加 h (2,256,16), 生成 100 万 token 也不增长。\ndemo 生成 300 个 token: 有 cache 递推约 0.4 s, 每步重算整段约 7 s, 快 16×, 两种方式 logits 最大差 4.77e-07。\n代价是压缩有损: 全部历史被挤进一个定长向量, 要精确召回很久以前的某个 token, 它不如注意力 (注意力把每个 token 的 K/V 都原样留着)。Jamba、Qwen3-Next 这类混合架构就是为补这一刀出现的: 大部分层用定长状态, 少数层保留全注意力。' },
      ],
      links: [
        { from: 'Gated DeltaNet', to: 'SelectiveSSM', body: '两者都是 “固定大小状态 + 由输入决定的门”。DeltaNet 的状态是一个矩阵, 用 delta rule (先擦掉 k 方向的旧值再写新值) 覆写; Mamba 的状态是对角 SSM, 用 $\\Delta$ 控制衰减和写入。' },
        { from: 'KV cache', to: 'cache["h"]', body: '同一个 GenerationMixin.generate() 驱动两种模型: 注意力层往 cache 里追加 K/V, 越来越长; Mamba 层就地更新 cache["h"], 大小不变。' },
        { from: 'MambaBlock', to: 'Block 组装器', body: 'Mamba block = Pre-RMSNorm + MambaLayer + 残差, 没有单独的 FFN。MambaLayer 里有一条门控分支 SiLU(gate) 逐元素乘到主路上, FFN 那份非线性已经由它承担了。' },
      ],
      sourceRows: [
        { concept: '选择性参数', code: 'self.x_proj(x).split([dt_rank, N, N])', takeaway: 'x_proj 把当前输入 $x_t$ 投影后切成三段: $\\Delta$ 的低秩表示、$B$、$C$。三者全部来自当前 token, “选择性” 三个字的全部实现就是这一行。' },
        { concept: 'Δ 恒正', code: 'F.softplus(self.dt_proj(dt_low))', takeaway: 'softplus 保证 $\\Delta>0$; 再配上 $A = -\\exp(\\text{A\\_log}) < 0$, 保留率 $\\exp(\\Delta\\cdot A)$ 一定落在 $(0,1)$。把 A 的负号去掉, README 实测第 57 步就溢出成 inf。' },
        { concept: '递推本体', code: 'h = A_bar[:, t] * h + Bx[:, t]', takeaway: '旧状态乘保留率, 加上当前写入。教学版是 Python 顺序 scan; 生产实现换成 work-efficient 并行 scan 加 kernel fusion。' },
        { concept: 'O(1) 解码', code: 'cache["h"] = h', takeaway: '给了 cache 就从上一步的 h 续着跑, 结束时写回。每步的工作量和已经读过多少 token 无关。' },
      ],
      snippetTitle: '选择性扫描',
      snippet: `dt, B_t, C_t = x_proj(x).split([r, N, N])     # 全部依赖输入
delta = softplus(dt_proj(dt))                   # [B, T, D, 1]  > 0
A = -exp(A_log)                                 # [D, N]        < 0

A_bar = exp(delta * A)                          # 保留率 ∈ (0, 1)
Bx    = delta * B_t * x                         # 写入量

h = zeros(B, D, N)
for t in range(T):                              # 训练时换成并行 scan
    h = A_bar[:, t] * h + Bx[:, t]
    y[t] = (h * C_t[:, t]).sum(-1)
return y + D_skip * x`,
      source: ['llm_models/layers/sparse/ssm.py:SelectiveSSM'],
      run: `${RUN}.language_models.mamba.infer_mamba`,
    },

    'models-moe-balance': {
      title: 'Aux-loss-free 负载均衡 · 会自己动的路由偏置',
      subtitle: 'MoE (混合专家) 模型里每个 token 只交给少数几个专家算, 专家之间的负载必须均衡, 否则热门专家排队、冷门专家闲着。传统做法加一项 aux loss 来压热门专家, 但它往语言建模目标里掺了一股不相干的梯度。\n读完你能说清 DeepSeek-V3 怎么把 “均衡” 从 loss 里搬出来, 变成一个不收梯度的控制器。',
      tldr: '每个专家配一个偏置 $b_e$:\n- 选 top-k (决定哪几个专家上场) 时: 用 $s+b$, s 是路由器给的分数。\n- 算门控权重 (上场的专家各信多少) 时: 只用 $s$。\n每个训练 step 之后, $b_e \\mathrel{+}= \\gamma\\cdot\\mathrm{sign}(\\text{平均负载} - \\text{专家 } e \\text{ 的负载})$: 接到的 token 比平均多就减一点 γ, 比平均少就加一点 γ。',
      question: '同样是把热门专家压一压, 为什么改偏置比加 aux loss 对模型质量更友好?',
      evolution: {
        title: '均衡从 loss 里搬出来',
        subtitle: '根问题: top-k 路由有正反馈, 放着不管就会坍缩到少数专家, 专家并行时一张卡拖慢整步',
        steps: [
          { name: '放任路由', pain: '(原点) 被选得多的专家学得更好, 于是更常被选', fix: '什么都不做: 少数专家包揽 token, 其余白占显存; 专家并行时热门那张卡收到的 token 爆容量' },
          { name: '容量上限', year: 2020, pain: '热门专家收到的 token 超出它那张卡能装下的量', fix: '每个专家设容量上限, 超出的 token 跳过 FFN 只走残差; 只处理了溢出, 路由照样偏' },
          { name: 'Aux loss', year: 2021, pain: '丢 token 治标, router 还是往热门专家挤', fix: '加一项 $E\\cdot\\sum f_e P_e$ (被选频率乘平均路由概率), 梯度直接压热门专家的路由 logit; 这股梯度和 LM loss 抢方向' },
          { name: '路由偏置', year: 2024, pain: 'aux 权重 α 大了干扰语言建模, 小了又均衡不住', fix: '每个专家一个不收梯度的 bias, 只进 top-k 排序, 每步按负载的符号 ±γ: CV 从 0.273 降到 0.124' },
        ],
      },
      code: 'llm_models/models/moe/deepseekV3.py · llm_models/training/loss.py',
      points: [
        { title: '选人和加权解耦', body: 'DeepSeek 的 MoE 层里, 路由器给每个 token 对每个专家打一个分 $\\mathrm{sigmoid}(\\text{logit})$。接下来有两件事:\n- 选人: 按分数取前 k 个专家 (top-k)。这一步用的是 $\\mathrm{sigmoid}(\\text{logit}) + b$, 每个专家加上自己的偏置 b。\n- 加权: 被选中的专家各自算一遍, 输出按权重加起来。权重取自不带 b 的原始分数, 再归一化到和为 1。\n所以 b 只改变 “谁上场”: 偏置被压低的专家排名靠后, 边缘 token 就被换到相邻分数的专家那里。b 不改变 “上场后信多少”: 同一个专家只要上场, 它的权重和没有偏置时一样。这就是 aux-loss-free 的全部: 均衡机制碰的是排序, 不碰任何进入梯度的量。', key: true },
        { title: '负载不均衡减半, LM loss 没动', body: 'demo 故意把路由器初始化得偏心 (专家 0、1 的权重乘 4), 模拟路由刚开始坍塌; aux loss 权重设为 0。量的是负载变异系数 CV: 各专家接到的 token 数的标准差除以均值, 0 是完全均衡, 越大越偏。\n- 不更新 bias: 最后 20 步平均 CV 0.273, 第 100 步 LM loss 2.979, 第 0 层负载 [37, 35, 33, 22, 27, 27, 40, 35]。\n- 更新 bias ($\\gamma$=1e-3): 最后 20 步平均 CV 0.124, 第 100 步 LM loss 2.982, 第 0 层负载 [31, 32, 36, 37, 27, 31, 29, 33]。\n不均衡减了一半, LM loss 几乎没变 (2.979 对 2.982)。原因在上一条: bias 既不进梯度也不进加权, 只把边缘 token 挪到相邻分数的专家。数据是固定的一个随机 batch, loss 本身只说明模型在背它。' },
        { title: 'b 是一个只看符号的控制器', body: 'b 不在计算图里 (更新函数带 @torch.no_grad), 它是 buffer 不是 Parameter: 不走梯度、不进 optimizer, 但进 state_dict, 否则 checkpoint 恢复后路由就对不上。\n更新规则只看符号: 先用 bincount 数出本 batch 每个专家接到多少 token (负载), 过载就 $-\\gamma$, 欠载就 $+\\gamma$, 和超载多少无关。步长恒为 $\\gamma$、与 batch 大小无关, 所以好调。\n- γ 太大: 一步就跨过专家分数之间的典型差距, 排序来回翻转, 负载震荡。demo 实测 $\\gamma$=1e-2 的均衡效果反而不如 1e-3。\n- γ 太小: 路由器自己在训练中漂移, 偏置追不上。' },
        { title: 'aux loss 的代价', body: '传统做法是加一项 $L_{\\text{aux}} = \\alpha\\cdot E\\cdot\\sum f_e\\cdot P_e$: E 是专家数, $f_e$ 是专家 e 被选中的频率, $P_e$ 是路由器给它的平均概率。热门专家的 f 和 P 都大, 乘积大, 梯度就直接压低它的路由 logit。\n问题是这股梯度和语言建模的梯度抢方向: 语言建模可能就是想把某类 token 都送给同一个专家, aux loss 偏要拆开。$\\alpha$ 大了干扰语言建模, 小了又均衡不住, 很难调。\n教学代码保留了 MoELMLoss 作为对照 (demo 里权重设 0)。真实的 V3 还留了一个权重极小的序列级 aux loss, 防极端情况。' },
      ],
      links: [
        { from: 'moe · Mixtral vs DeepSeek', to: 'routing_bias', body: 'MoE 路由一章讲 top-k 选专家、共享专家、softmax 门控与 sigmoid 门控的区别; 这里只补 “怎么保持均衡” 这一块, 落在 routing_bias 这一个 buffer 上。' },
        { from: 'update_routing_bias', to: 'train loop', body: '它不是模型 forward 的一部分, 必须在 optimizer.step() 之后由训练循环显式调用一次。忘了调, b 永远是 0, 整套机制退化成无均衡, 负载表一步都不会动。' },
        { from: '负载均衡', to: 'llm_train EP 与序列并行', body: '专家并行 (EP) 把不同专家放在不同的卡上。不均衡时, 热门专家那张卡在 all-to-all 通信里收到的 token 撑爆容量, 其它卡等它, 整个 step 被拖慢。' },
      ],
      sourceRows: [
        { concept: '偏置只进排序', code: 'select_scores = sigmoid_scores + self.routing_bias', takeaway: '带偏置的分数只用来排序: topk 只作用在 select_scores 上, 得到 selected_experts。' },
        { concept: '权重不带偏置', code: 'topk_sigmoid = sigmoid_scores.gather(-1, selected_experts)', takeaway: '从原始的 sigmoid_scores 里 gather 出被选中专家的分数, 再除以它们的和归一到 1。routing_bias 不出现在这里。' },
        { concept: '更新规则', code: 'deepseekV3.py:update_routing_bias', takeaway: 'bincount 数出每个专家的负载; $\\text{bias} \\mathrel{+}= \\gamma\\cdot\\mathrm{sign}(\\text{mean} - \\text{load})$。它是 buffer, 不走梯度也不进 optimizer, 但要进 state_dict。' },
        { concept: '对照: aux loss', code: 'loss.py:MoELMLoss', takeaway: '$f_e$ (被选频率) 乘 $P_e$ (平均路由概率) 求和再乘 E。$f_e$ 是计数、不可导, 只有 $P_e$ 这一支有梯度流回路由器。' },
      ],
      snippetTitle: '前向选人 + 步后调偏置',
      snippet: `s = sigmoid(router(x))                       # [N, E] 原始亲和度
_, picked = topk(s + bias, k)                 # 偏置只影响 “选谁”
w = s.gather(-1, picked)
w = w / w.sum(-1, keepdim=True)               # 门控权重: 不含 bias
out = shared(x) + sum(w_i * expert_i(x))

# optimizer.step() 之后:
with no_grad():
    load = bincount(picked.flatten(), minlength=E)
    bias += gamma * sign(load.mean() - load)  # 过载 −γ, 欠载 +γ`,
      source: ['llm_models/models/moe/deepseekV3.py:update_routing_bias'],
      run: `${RUN}.moe.deepseek.train_deepseek`,
    },

    'models-dsa': {
      title: 'DSA · 用一个便宜的 indexer 挑出值得算注意力的那 k 个',
      subtitle: 'DeepSeek-V3.2 的稀疏注意力 (DSA) 不规定每个 token 该看哪些位置, 而是让模型自己学: 一个便宜的打分器 (indexer) 给所有 key 打分, 每个 query 只在得分最高的 k 个 key 上做 MLA 注意力。\n读完你能说清 indexer 是怎么被训出来的, 以及为什么不能一上来就稀疏。',
      tldr: '- 稀疏注意力的意思是每个 query 不看全部 T 个 key, 只看一小部分。DSA 用一个很小的 LightningIndexer 给每对 (query t, key s) 打分 $I[t,s] = \\sum_h \\mathrm{ReLU}(q_h\\cdot k_h/\\sqrt{d})$ (几个小 head 的点积过 ReLU 再相加), 每行只留分最高的 k 个, 主注意力只看这 k 个。\n- 选 top-k 是 “取最大的 k 个” 这种离散操作, 不可导, 语言模型的梯度传不到 indexer。所以 indexer 单独训: 让 $\\mathrm{softmax}(I)$ 去模仿主注意力的分布, 两者的差距用 KL 散度衡量 (KL 是衡量两个概率分布差多远的量, 0 表示完全一样)。\n- 训练分三步: 先稠密预训练, 再冻住主模型只预热 indexer, 最后切成稀疏。',
      question: 'indexer 自己也要给所有 (t, s) 打分, 还是 $O(T^2)$, 那到底省在哪?',
      evolution: {
        title: '稀疏的形状从人定到学出来',
        subtitle: '根问题: 128K 上下文里每个 query 都和全部 key 算注意力, 算力 $O(T^2)$, 真正有用的 key 只占一小部分',
        steps: [
          { name: '稠密注意力', pain: '(原点) 每个 query 都要和全部 T 个 key 算分数、做 softmax、乘 V', fix: 'MLA (把 KV cache 压成低维 latent 的注意力) 也照样全算: cache 压小了, 算力仍是 $O(T^2)$' },
          { name: '固定稀疏 mask', year: 2019, pain: '真正有用的 key 只占一小部分, 却每个都要算', fix: '人定 mask 形状 (滑窗、跨步), 每个 query 只算少数 key; 关键 token 落在形状外就永远看不到' },
          { name: '按内容选 top-k', pain: '该看哪些 key 取决于内容, 固定形状猜不中', fix: '便宜的 indexer 按内容打分, 只把 top-k 交给 MLA; top-k 不可导, indexer 收不到梯度' },
          { name: 'DSA', year: 2025, pain: 'LM 梯度到不了 indexer, 随机初始化时等于随机丢 key', fix: '主注意力当 teacher, 用 KL 训 indexer; 预热后再稀疏: top-8 召回从 0.45 升到 0.92' },
        ],
      },
      code: A,
      points: [
        { title: '省的是贵的那部分', body: 'indexer 确实也要给所有 (t, s) 对打分, 复杂度仍是 $O(T^2)$; DSA 省的不是这一步, 是后面那一步。\n- indexer 有多便宜: head 少 (demo 里 2 个)、每头维度小、没有 value、不做 softmax, 只做点积和 ReLU, 还可以用低精度算。\n- 主注意力有多贵: MLA 要先把缓存的 latent 升维成 K 和 V, 再对全部 key 做 softmax, 再乘 V。DSA 让这一步只在 k 个 key 上做, 从 $O(T^2)$ 降到 $O(T\\cdot k)$。\n长上下文下 k 远小于 T: 比如 T 取 128K、k 取 2048, 主注意力的计算量只剩 k/T 约 1.6%。总成本于是由 indexer 主导, 它虽然也是 $T^2$, 但前面的常数很小。\n边界: 本库的教学代码仍构造稠密的 [B, T, S] mask, 给没选中的位置填 −inf, 算力没有真的省下来; 生产版靠自定义 kernel 真正跳过未选中的 K/V。', key: true },
        { title: 'indexer 是被蒸馏出来的', body: 'top-k 是离散选择: 分数变一点点, 选出来的集合要么不变, 要么突然换一个, 没有平滑的导数。所以语言模型 loss (LM loss) 的梯度传到 top-k 这一步就断了, indexer 收不到任何梯度。demo 直接打印出来: 不加对齐 loss 时, LM loss 对 indexer 参数的梯度全是 None。\n所以单独加一项对齐 loss (index_loss), 像老师教学生:\n- teacher: 主注意力的概率分布, 各个 head 取平均, 然后 detach (切断梯度, 老师不跟着学生变)。\n- student: 把 indexer 的分数做 softmax 得到一个分布, 用 KL 散度拉它去贴近老师。\nindexer 的输入 q 也 detach 了: 对齐 loss 只更新 indexer 自己的 w_q 和 w_k, 不会反过来改动主干的表示。两条 loss 互不干扰。' },
        { title: '分三步训, 不能一上来就稀疏', body: '刚初始化的 indexer 等于随机挑 key。这时候就切稀疏, 主注意力每步只能看到随机的 k 个位置, 模型会被毁掉。而且初始化时主注意力也近似均匀, 没有可学的结构, indexer 学也白学。所以 train 脚本分三步:\n- 稠密预训练 (60 步): 主注意力看全部位置, 只训主模型, 让 teacher 先长出结构。\n- indexer 预热 (80 步): 冻结主模型, 注意力仍看全部位置, 只用 KL 训 indexer。\n- 切稀疏 (40 步): 注意力只看 top-k, LM loss、aux loss (MoE 的负载均衡项)、index loss 一起训。\n前两步都开着 set_dense_warmup, 意思是 MLA 走稠密、indexer 只旁听学习。预热这 80 步里:\n- KL: 从 0.1144 降到 0.0046。\n- top-8 召回率 (主注意力最看重的 8 个 key 里, indexer 选中了几个): 从 0.450 升到 0.922。0.450 恰好等于随机乱选的期望, 说明没训过的 indexer 就是在瞎选。\n实验台里 “对齐程度” 滑杆演示的就是这个过程。数据是固定的一个随机 batch, 这些数字说明 “机制能跑通”, 不说明泛化。' },
        { title: '先 mask, 再 top-k', body: '顺序是: 先把因果 mask 之外的分数 (未来位置、pad 位置) 填 −inf, 再在剩下的里取 top-k。\n反过来会出两个问题。第一, top-k 会选中未来位置, 它们随后被因果 mask 挡掉, 这一行真正有效的 key 就不足 k 个。第二, indexer 对齐时学到的是 “哪些未来 token 重要” 这种泄漏的信号, 推理时根本看不到未来。\n还有一个细节: 序列开头几行本来就不到 k 个可见位置 (第 0 行只能看 1 个), top-k 会把填了 −inf 的位置也选进来。所以最后再和因果 mask 取一次交集, 把它们剔掉。demo 里 T=24、k=8, 每个 query 看到的 key 数是 1, 2, 3 一直到 8, 之后封顶在 8。' },
      ],
      links: [
        { from: 'attention · MLA', to: 'MultiHeadLatentSparseAttention', body: '左边是 MLA, 右边是 DSA 的类。DSA 没有改注意力本身: 它就是 LightningIndexer 打分, 加一张由 top-k 生成的 mask, 加一个原封不动的 MLA。' },
        { from: 'models-mtp · 掩码实验台', to: 'top-k 稀疏', body: '那一章的实验台有一个 top-k 模式, 给你看 mask 长什么样; 这一章讲这张 mask 是怎么被学出来的, 以及 k 取多少才够。' },
        { from: 'cache["idx_k"]', to: 'KV cache 生成', body: '解码时每个新 token 都要让 indexer 给全部历史位置打分, 所以历史 token 的 indexer key 也得缓存, 存在 cache 的 idx_k 键里, demo 里每 token 每层多存 64 个数。这是 DSA 多付的代价: 它省算力, 不省 cache。' },
      ],
      sourceRows: [
        { concept: 'indexer 打分', code: 'attention.py:LightningIndexer', takeaway: '每个小 head 算 q 和 k 的点积, 过 ReLU, 再对 head 求和。它只需要分数能排序, 不需要归一化成概率, 所以没有 softmax。' },
        { concept: '输入 detach', code: 'self.indexer(q.detach(), mask=mask, cache=cache)', takeaway: '传给 indexer 的 q 先 detach: 对齐 loss 的梯度只到 indexer 自己的 w_q / w_k 为止, 不会流回主干。' },
        { concept: 'top-k → mask', code: '_sparse_mask_from_topk', takeaway: '先把因果 mask 之外的分数填 −inf, 再取 top-k, 每行把选中的 k 个下标 scatter 成 True。顺序不能反: 先 top-k 再 mask 会选中未来位置, 它们随后被 mask 掉, 有效 key 就不足 k 个。开头几行可见位置不到 k 个, 会选到 −inf, 所以最后再与因果 mask 取一次交集。' },
        { concept: 'KL 对齐', code: 'kl = (p * (p.clamp_min(1e-9).log() - log_q)).sum(dim=-1)', takeaway: 'p = attn.mean(dim=1) 是 detach 的 teacher (各 head 平均后的主注意力分布); log_q 是 indexer 分数的 log_softmax。两边都限定在同一个 used_mask 集合上, 稀疏阶段就只比选中的 k 个位置。' },
        { concept: 'loss 汇总', code: 'loss.py:MoELMLoss', takeaway: 'total = LM + aux_loss_weight·aux + index_loss_weight·index_loss。index_loss 由每层的注意力模块算好, 搭 routing_info 的便车交给 MoELMLoss。' },
      ],
      snippetTitle: 'DSA 的三步',
      snippet: `# 1) 便宜的 indexer 给每对 (t, s) 打分
I = relu(q_idx @ k_idx.T / sqrt(d_idx)).sum(heads)     # [T, S]
I = I.masked_fill(~causal, -inf)

# 2) 每行只留 top-k, 昂贵的 MLA 只在这些位置上算
keep = zeros_like(I).scatter(-1, I.topk(k).indices, True) & causal
out, attn = mla(x, mask=causal if dense_warmup else keep)

# 3) top-k 不可导 → indexer 用单独的 KL 对齐 loss 训练
p = attn.mean(heads).detach()                          # teacher
index_loss = (p * (log(p) - log_softmax(I))).sum(-1).mean()`,
      source: ['llm_models/layers/core/attention.py:LightningIndexer'],
      run: `${RUN}.moe.deepseek_v3_2.train_deepseek_v3_2`,
    },

    'models-gptoss': {
      title: 'GPT-OSS 结构 · 滑窗/全注意力交替 + 可学 sink + MoE',
      subtitle: 'OpenAI 2025 年的开放权重模型没发明新零件, 只是把三个已知零件拼得很讲究: 滑动窗口注意力、attention sink、MoE。\n读完你能说清交替排布省在哪, 以及为什么滑窗把开头 token 挤出去之后它不会崩。',
      tldr: '- 偶数层用滑动窗口注意力 (SWA): 每个 token 只看最近 W 个, 所以这些层的 KV cache 封顶在 W, 不随上下文长度涨。\n- 奇数层用全注意力: 看全部历史, 兜住远处的信息。\n- sink logit: 每个注意力 head 多一个可学的分数, 拼进 softmax 的分母一起算, 算完就丢掉它那一列。效果是 head 可以 “谁都不看”, 滑窗也不再依赖开头那几个 token。',
      question: '滑窗会把开头 token 挤出 cache, StreamingLLM 发现这会让模型崩溃, GPT-OSS 为什么不怕?',
      evolution: {
        title: '省 cache, 又不丢远处和开头',
        subtitle: '根问题: 长上下文下每层都存全部 K/V, cache 随长度涨; 裁掉一部分, 远处的信息和开头的 token 又不能丢',
        steps: [
          { name: '全注意力', pain: '(原点) 每层都要看全部历史, 每层 cache 都随上下文长度 T 涨', fix: 'LLaMA 每层全因果: 任何位置一步可达, cache 也最大' },
          { name: '全滑窗', year: 2023, pain: '长上下文下每一层的 cache 都是 T', fix: 'Mistral 每层只看最近 W 个, cache 封顶 W; 远处信息逐层接力, 每接力一次压缩一次' },
          { name: '交替排布', pain: '远处的信息接力几次后被压糊, 精确召回变难', fix: '滑窗层和全注意力层交替 (Gemma 2、GPT-OSS): 一半层 cache 封顶 W, 另一半保留一步直达的通路' },
          { name: '保留首 token', year: 2023, pain: '模型把多余注意力倒在开头几个 token 上, 滑窗一挤掉它们就崩', fix: 'StreamingLLM 永远保留开头 4 个 token 的 K/V; 窗口里得一直给它们留位置' },
          { name: '可学 sink', year: 2025, pain: '垃圾桶绑在具体 token 上, 这些 token 必须一直占着 cache', fix: 'GPT-OSS 给每个 head 一个可学 logit, 只进 softmax 分母: 行和可以 < 1, 不占 KV' },
        ],
      },
      code: `llm_models/models/moe/gpt_oss.py · ${A} · llm_models/utils/masks.py`,
      points: [
        { title: '交替排布: 用一半的 KV 买全部感受野', body: '先说滑窗 (SWA, sliding window attention): 每个 token 只看最近 W 个 token, 更早的看不到。单层这样, 但第 2 层看到的 “邻居” 已经在第 1 层各自汇总过它们的邻居, 信息可以一层一层往后接力。L 层纯滑窗模型的感受野 (一个 token 最远能间接看到多远) 只有 $L\\cdot(W-1)+1$, 而且远处信息每接力一次就被压缩一遍, 越传越糊。\nGPT-OSS 隔一层插一层全注意力: 任何位置一步就能直接够到全部历史, 不用接力。\n省在哪: 滑窗层的 KV 是滚动缓冲, 上限 W, 和上下文长度无关; 全注意力层的 KV 随长度涨。长上下文下 cache 几乎只由全注意力层决定。gpt-oss-20b (24 层, W=128) 在 T=131072 时, 比 24 层全是全注意力约省 50%。\ndemo 的 4 层 mini 模型 (W=8) 读完 40 个 token:\n- 各层 cache 长度: [8, 40, 8, 40], 加起来 96, 比全是全注意力的 160 省 40%。\n- 贪心生成 100 个 token: 有 cache 比无 cache 快 3.1×, 输出逐 token 一致。', key: true },
        { title: 'softmax 不会弃权', body: '注意力用 softmax 把一行分数变成概率: 它只看分数之间的相对差, 一行概率加起来恒为 1。这带来一个问题: 一个 head 在某个位置没什么可看时 (比如这一步它不需要任何上文), 它也必须把 1 分完, 输出一堆无关 value 的加权平均, 等于被迫说话。\n没有 sink 的模型在训练中自己找了个垃圾桶: 把多余的概率倒在开头几个 token 上, 因为它们每一步都在, 位置稳定。这就是 attention sink 现象 (StreamingLLM 论文的观察: 首 token 拿走大量注意力)。\n也正因为如此, 滑窗一旦把首 token 挤出窗口, 垃圾桶没了, 多余的概率只能倒给别的 token, 输出分布就崩掉。' },
        { title: 'sink 做成参数, 不做成 token', body: '两种补救:\n- StreamingLLM: 永远保留开头 4 个 token 的 K/V, 不管窗口怎么滑。本库 masks.py 的 build_sliding_window_mask 有 sink_tokens 参数, 开头 S 列永远可见。代价是这几个 token 必须一直占着 cache, 窗口里得给它们留位置。\n- GPT-OSS: 给每个 head 一个可学的标量 logit (self.sink)。算注意力时把它当作多出来的一列拼进分数, 一起做 softmax, 然后把这一列切掉。分给它的概率不乘任何 value, 直接丢弃, 于是剩下各列的概率和小于 1。head 想 “谁都不看” 时就把概率倒给 sink。它不占 KV, 也不依赖任何特定 token 留在窗口里。' },
        { title: 'demo 里 sink 几乎没动', body: 'train 脚本在固定的一个随机 batch 上训 60 步, sink logit 从 0 只动到约 ±0.01 (第 0 层是 [0.001, 0.0031, 0.0059, 0.0014] 这个量级), 梯度全部非零。\n为什么几乎没动: 随机 token 没有 “这一步该不该看上文” 的结构, 模型没理由把概率往 sink 倒。所以 demo 证明的只是梯度通路是通的, 不是 sink 在起作用。真实模型里这些值会明显非零, 而且各 head 不同。' },
      ],
      links: [
        { from: 'models-mtp · SWA', to: '交替排布', body: 'Mistral 全部层都是滑窗 (那一章讲的); GPT-OSS 和 Gemma 系列改成滑窗层与全注意力层交替, 比例不同: Gemma 2 是 1:1, Gemma 3 是 5:1。' },
        { from: 'build_sliding_window_mask', to: 'use_sink', body: '两个 sink 是两个层面的东西。sink_tokens 是 mask 层面的: 保留开头几个真实 token 永远可见。use_sink 是 softmax 层面的: 每个 head 一个可学 logit, 没有 K 也没有 V。' },
        { from: 'moe · MixtralMoE', to: 'GPT-OSS FFN', body: 'GPT-OSS 的 FFN 槽位是标准的 MoE (每个 token 从 E 个专家里选 top-k 个来算), 路由方式与 Mixtral 同族, 本库直接复用 MixtralMoE。' },
      ],
      sourceRows: [
        { concept: '带状 mask', code: 'masks.py:build_sliding_window_mask', takeaway: '可见条件是 ($j \\le i$) 且 ($j > i-W$): key 在 query 之前, 且距离不超过 W。sink_tokens > 0 时额外保留开头 S 列永远可见。' },
        { concept: 'sink 参数', code: 'self.sink = nn.Parameter(torch.zeros(num_heads))', takeaway: '每个 head 一个标量, 初始 0。初始 0 相当于 “多了一个分数为 0 的空 key”: 未训练时它和其它分数为 0 的 key 分到一样的概率。' },
        { concept: '参与 softmax 后丢弃', code: 'F.softmax(torch.cat([scores, sink], dim=-1), dim=-1)[..., :-1]', takeaway: '把 sink 拼到分数最后一列一起 softmax, 再把最后一列切掉, 每行概率和就小于 1。demo 初始化时第 0 行只有 1 个真实 key 加 1 个 sink, 和是 0.502 约等于 1/2; 第 39 行 40 个真实 key, 和是 0.976 约等于 40/41; 差额进了 sink。' },
        { concept: '滚动裁剪 cache', code: 'layer_cache["k"][:, :, -self.window_size:]', takeaway: '只有滑窗层每步把 K/V 裁到最近 W 个, 全注意力层不裁。裁掉的 K/V 在带状 mask 下本来就是 −inf, 所以输出逐 token 不变; K 存的是 RoPE 之后的, 位置也不会错。' },
        { concept: 'MoE FFN', code: 'gpt_oss.py:GPTOSSBlock', takeaway: '直接复用 MixtralBlock。GPT-OSS 官方是 “先 top-k 再对 k 个 logit 做 softmax”, Mixtral 是 “先 softmax 再取 top-k 再重归一”, 两者逐项相等, demo 有断言。' },
      ],
      snippetTitle: '一层 GPT-OSS 风格的注意力',
      snippet: `# 层排布: 偶数层滑窗, 奇数层全注意力
mask = sliding_window_mask(T, W) if layer % 2 == 0 else causal_mask(T)

scores = q @ k.T / sqrt(d)                       # [H, T, S]
scores = scores.masked_fill(~mask, -inf)

sink = self.sink.view(H, 1, 1).expand(H, T, 1)   # 每个 head 一个可学 logit
probs = softmax(cat([scores, sink], -1), -1)     # sink 只进分母
probs = probs[..., :-1]                          # 丢掉 sink 列: 行和 < 1
out = probs @ v

# KV cache: 滑窗层只留最近 W 条, 全注意力层留全部
kv_entries = W if layer % 2 == 0 else T`,
      source: ['llm_models/models/moe/gpt_oss.py:is_swa', 'llm_models/utils/masks.py:build_sliding_window_mask'],
      run: `${RUN}.moe.gpt_oss.infer_gpt_oss`,
    },

    'models-llada': {
      title: 'LLaDA · 不从左到右写的语言模型',
      subtitle: '扩散模型原本是给图像加噪再学去噪。LLaDA 把它搬到离散 token 上: 加噪就是随机把 token 换成 [MASK], 去噪就是用双向 Transformer 一次预测所有 [MASK]。读完你能说清它和 BERT 差在哪, 以及为什么它用不了 KV cache。',
      tldr: '- 训练: 每条序列抽一个遮盖率 $t$ (0 到 1 之间), 每个 token 以概率 $t$ 被换成 [MASK]。只在被遮位置算交叉熵 CE, 再乘 $1/t$。乘完之后这个 loss 是 $-\\log p(x)$ 的上界 (ELBO), 意思是压低它就是在抬高模型给真实文本的概率, 它因此是一个正经的生成模型。\n- 采样: 从全是 [MASK] 的序列出发, 每步并行预测全部位置, 留下最有把握的几个, 其余重新遮回去。剩余 [MASK] 的个数按线性日程递减到 0。',
      question: 'BERT 也是遮盖再预测, 为什么 BERT 不能拿来生成而 LLaDA 能?',
      evolution: {
        title: '从左到右之外的另一条生成路',
        subtitle: '根问题: 自回归一次只出 1 个 token, 顺序写死从左到右, 只会续写',
        steps: [
          { name: '自回归 LM', pain: '(原点) 生成 N 个 token 要串行前向 N 次, 每个 token 只能看左边', fix: '因果 mask + 预测下一个: 训练简单; 但只会续写, 学了 “A 后面是 B” 答不出 “B 前面是 A”' },
          { name: 'BERT (MLM)', year: 2018, pain: '自回归只看左边, 生成顺序写死', fix: '双向注意力, 遮 15% 再还原; 从没见过几乎全遮的输入, 没法从零生成' },
          { name: '掩码扩散', year: 2025, pain: '固定遮 15% 只学会补少量空, 从零生成那一段没练过', fix: 'LLaDA 让遮盖率 t 铺满 (0,1), loss 乘 $1/t$ 成为似然上界; 采样从全 [MASK] 逐步去噪' },
          { name: '低置信度重遮', pain: '同一步定下的 token 互相看不见, 一步定得越多越容易打架', fix: '每步只留最有把握的, 其余遮回去下一步再定: 15 步填空准确率 1.000, 1 步并行 0.887' },
        ],
      },
      code: 'llm_models/models/language_models/llada.py',
      points: [
        { title: '随机遮盖比例: 一整族去噪任务', body: '- BERT: 每次固定遮 15% 的 token 让模型还原。它只学会了 “补少量空”, 从没见过几乎全是 [MASK] 的输入, 所以不会从零生成。\n- LLaDA: 遮盖率 $t$ 在 $(0,1)$ 上均匀取值, 每条序列一个。$t\\approx 1$ 时几乎全遮, 模型要从零猜整句; $t\\approx 0$ 时只遮一两个词, 只是补空。一个模型同时学会了从 “几乎全空” 到 “只缺一个” 的所有难度。\n采样正是从 $t=1$ (全 [MASK]) 走到 $t=0$ (全填满), 每一步遇到的遮盖比例训练时都见过。demo 的任务是 16 环置换链: 固定一个置换 π, 序列满足 x[i+1] = π(x[i]), 只有起点随机; 每步抽新数据、新遮盖, 所以 loss 下降是真的学会了 π。', key: true },
        { title: '乘 1/t 之后 loss 才是似然上界', body: '$t$ 小的样本被遮的 token 很少, 只在被遮位置算 CE 的话, 这些样本几乎不贡献 loss, 而它们恰恰对应采样的最后几步。乘 $1/t$ 把每条样本的权重补齐: 遮得少, 每个被遮位置就值更多。\n乘完之后有一个数学结论: 这个 loss 是 $-\\log p(x)$ 的上界 (ELBO, 变分下界的负数形式)。$-\\log p(x)$ 是模型给真实文本的负对数概率, 越小说明模型越认为这段文本像真的。压低上界就是在抬高似然, 模型因此是一个生成模型, 不只是一个填空器。\n量纲也对上了: $\\mathbb{E}[\\text{被遮数}/t] = L$ (序列长度), 所以均匀瞎猜时 loss 的期望正好是 $\\ln V$, 和自回归模型的 CE 可以直接比。demo 里初始 loss 3.014 对 $\\ln V$ = 2.833; 训练后 held-out loss 0.189, 理论下界 0.173。\n代码里 batch 内的 $t$ 用分层采样 (B 个点等间隔铺满区间), 压住 $1/t$ 带来的方差。' },
        { title: '生成顺序由置信度决定', body: '每一步模型给所有 [MASK] 位置都预测一个 token, 并给出一个置信度 (所选 token 的概率)。低置信度重遮 (low-confidence remasking) 的规则: 先定 “显然” 的 token, 把置信度垫底的那些重新遮回 [MASK], 下一步它们就能看着已经定下的邻居再决定。已定稿的 token 置信度记为 $+\\infty$, 永不重遮。\ndemo 的填空准确率 (给首 token 续写 15 个位置):\n- 15 步低置信度重遮: 1.000。\n- 15 步随机重遮 (随机挑谁遮回去): 0.910。\n- 1 步并行 (一次全部定下): 0.887。\n步数越少, 同一步里定下的 token 越多。它们是并行预测的, 互相看不见对方选了什么, 容易打架 (两个相邻位置各自合理、拼起来不对), 所以 1 步并行最差。' },
        { title: '没有 KV cache, 步数是旋钮', body: '自回归模型能用 KV cache, 是因为因果 mask 保证旧 token 的 K/V 不随新 token 改变。LLaDA 是双向注意力: 每个位置都看左右两边, 任何一个 [MASK] 被填上, 所有位置的 K/V 都会变。所以没有 KV cache, 每步整段重算。\n- 步数多: 每步定的 token 少, 质量高, 但前向次数多。\n- 步数少: 快, 但并行定下的 token 互相看不见, 质量掉。\n好处是位置任意: 续写 (给开头)、倒推 (给结尾)、两头填 (给中间) 都是同一个 sample() 函数, demo 三个方向的 15 步准确率都是 1.000。' },
      ],
      links: [
        { from: 'BERT · MaskedLMLoss', to: 'LLaDALoss', body: '两个 loss 都只在被遮位置算 CE。区别有三处: 遮盖率随机而不是固定 15%, 乘 $1/t$, 再除以总 token 数而不是被遮 token 数。少一处就不是似然上界。' },
        { from: 'diffusion · 连续扩散', to: 'forward_process', body: '图像扩散加的是高斯噪声, 这里的 “噪声” 是把 token 换成 [MASK]。两者都是训练时随机抽一个噪声强度 $t$, 采样时从 $t=1$ 一步步走回 0。' },
        { from: 'KV cache 生成', to: 'LLaDA.sample', body: '自回归靠因果 mask 才有 KV cache 可用; LLaDA 每步任何位置都可能变, 没法复用, sample() 每步调一次完整的 forward。' },
      ],
      sourceRows: [
        { concept: '加噪', code: 'llada.py:forward_process', takeaway: '每条序列抽一个 $t$, 每个 token 独立以概率 $t$ 换成 [MASK]。batch 内的 $t$ 用分层采样: 一个随机起点, B 个值等间隔铺满 (eps, 1)。' },
        { concept: 'ELBO loss', code: '(ce * masked / t[:, None]).sum() / labels.numel()', takeaway: 'ce 乘 masked 清掉没被遮的位置, 除以每条序列自己的 t, 最后除以总 token 数 labels.numel() 而不是被遮 token 数。只有这样 $1/t$ 才有 “补齐期望” 的含义。' },
        { concept: '线性日程', code: 'n_masked = torch.round(n_gen * (1 - s / steps)).long()', takeaway: '第 s 步结束后应该剩几个 [MASK]: 总数乘 (1 − s/steps) 再取整。生成 15 个、分 4 步: 11, 8, 4, 0。' },
        { concept: '重遮', code: 'x.masked_fill(rank < n_masked[:, None], self.mask_id)', takeaway: 'rank 是置信度升序名次, 0 是最没把握的; 名次小于 n_masked 的位置被遮回 [MASK]。已定稿位置 conf=+inf, 排在最后, 永远不会被遮。' },
      ],
      snippetTitle: '训练一步 + 采样循环',
      snippet: `# 训练: 随机比例遮盖
t = uniform(eps, 1, size=[B])                    # 每条序列一个遮盖率
masked = rand(B, T) < t[:, None]
logits = model(x.masked_fill(masked, MASK))      # 双向注意力, 无因果 mask
loss = (CE(logits, x) * masked / t[:, None]).sum() / (B * T)

# 采样: 从全 [MASK] 迭代去噪
x = full([B, T], MASK)
for s in range(1, steps + 1):
    probs = softmax(model(x)); x0 = probs.argmax(-1)
    conf = where(x == MASK, probs.max(-1), +inf)   # 已定稿的永不重遮
    x = where(x == MASK, x0, x)                    # 先全部填上
    n = round(T * (1 - s / steps))                 # 本步后应剩的 [MASK] 数
    x[conf 最低的 n 个] = MASK                     # 没把握的遮回去`,
      source: ['llm_models/models/language_models/llada.py:forward_process'],
      run: `${RUN}.language_models.llada.infer_llada`,
    },

    'models-var': {
      title: 'VAR · 自回归的单位从 “下一个 token” 换成 “下一个分辨率”',
      subtitle: '把图像按光栅顺序 (一行一行) 展平再做 next-token 生成: 既慢 (步数等于 token 数), 又把二维结构拆了。\nVAR 让模型先画 1×1 的总体印象, 再画 2×2、4×4, 一次出一整级。读完你能说清级内并行为什么不会把画面搞乱。',
      tldr: '- 多尺度 VQ: 先用一个 tokenizer 把图像编码成一座 token 金字塔 (1×1、2×2、4×4 三张 token 表)。每一级量化的是 “前面所有级还没解释掉的残差”, 粗级管大块颜色, 细级补细节。\n- Transformer: 用块状因果 mask 逐级预测。同一级的 token 互相可见, 另外只能看更粗的级, 看不到更细的。\nscales=(1,2,4) 时生成整张图只要 3 次前向 (一次出一级); 光栅序 4×4 的 16 个 token 要 16 次。',
      question: '同一尺度内的 token 是一次并行采样出来的。它们看不到彼此的采样结果, 画面为什么不会乱?',
      evolution: {
        title: '自回归的一步从一个 token 变成一整级',
        subtitle: '根问题: 自回归要一维序列, 图像却是二维的; 逐 token 生成, 步数随边长的平方涨',
        steps: [
          { name: '光栅序 next-token', pain: '(原点) 自回归要一维序列, 图像却是二维网格', fix: '按行展平, 逐 token 生成: 步数等于 token 数, 上下相邻的 token 在序列里隔了一整行' },
          { name: '同尺度并行', year: 2022, pain: '边长翻倍, 串行步数翻 4 倍', fix: 'MaskGIT 一次前向并行定一批 token; 同批 token 彼此看不见, 不确定性大时细节容易打架' },
          { name: 'next-scale', year: 2024, pain: '同一分辨率上每个 token 都还很不确定, 独立采样就乱', fix: 'VAR 按分辨率由粗到细: 粗级先定构图, 细级只补细节, 级内并行也不乱' },
          { name: '多尺度残差 VQ', pain: '逐级生成要每一级各有一套 token, 普通 VQ 只给一张单尺度 token map', fix: '每一级只量化前面还没解释掉的残差, 各级共用码本; token 总数变多 (8×8 时 85 对 64)' },
        ],
      },
      code: 'llm_models/models/generative/var.py · llm_models/layers/diffusion/vq.py',
      points: [
        { title: '由粗到细符合图像的统计结构', body: '并行采样的问题在于: 同一步里定下的 token 互相看不见对方抽到了什么, 各自合理、拼起来可能不对。VAR 的办法是按分辨率分级, 让每一级的不确定性都很小。\n- 低分辨率 (1×1、2×2): 决定构图和明暗, 这是最不确定的部分, 但 token 很少, 一级就几个。\n- 高分辨率 (4×4 及更细): 粗的已经定死, 每个位置只是在已定的大块颜色上补细节, 可选范围已经很窄。\n粗的先定死之后, 细的那一级每个 token 的不确定性已经很小, 并行独立采样带来的不一致也就很小。demo 的 train 里 CE 停在 0.27 而不是 0: 8 张图的 1×1 token 只有 3 种取值, 同一个粗 token 后面跟着不同的 2×2, 无条件模型只能学到分布, 这个 batch 的理论下界是 0.2476。', key: true },
        { title: '残差量化: 后面的级修正前面的级', body: 'tokenizer 先把图像编码成一张特征图 f (demo 里 4×4 个 16 维向量), 再分级量化。量化的意思是把每个连续向量换成码本里最近的那个码字, 记下编号。\n第 k 级的输入是 $f - \\sum_{j<k} \\mathrm{upsample}(z_j)$: 把前面各级的码字放大到 f 的尺寸加起来, 从 f 里减掉, 剩下的残差再缩到本级分辨率去量化。粗尺度量化得再糙, 误差都留在残差里交给下一级修。\n每一级量化的都是同一个特征空间里的残差, 所以各级共用同一个码本 (demo 64 个码字)。解码时把各级码字上采样后求和。\n代价是 token 总数 $\\sum s^2$ 比单尺度多: 8×8 时 1+4+16+64 = 85 个, 对单尺度的 64 个。' },
        { title: '块状因果 mask', body: '序列是三级 token 拼接: [1×1 的 1 个 | 2×2 的 4 个 | 4×4 的 16 个], 共 21 个位置。mask 规则: 同一块内全可见, 另外只能看到更早 (更粗) 的块, 看不到更细的块。代码按每个位置所属的级号比较: 级号小于等于自己的都可见。\n- 训练: 真实的 token 金字塔一次送进去 (teacher forcing), 一次前向算完所有 21 个位置的 loss。\n- 采样: 3 次前向, 每次产出一整级, 级内 s² 个 token 并行抽样。\n普通自回归的 mask 是逐 token 的下三角, 这里是逐块的下三角: 一步的单位从 1 个 token 变成一整级。' },
        { title: '本级的 token 只喂给下一级', body: '第 k 级每个位置的输入, 不是上一个 token, 而是前 k−1 级的累计重建 (各级码字上采样求和), 再下采样到本级分辨率。第 1 级没有更粗的级, 用一个可学习的 start 向量。\n第 k 级自己的 token 不进本级的输入, 只影响下一级的输入。所以序列里没有普通语言模型那种 “输入是前一个 token、标签是当前 token” 的 shift-by-one 错位: 错位发生在级与级之间。\ndemo 验证, 改动第 2 级的一个 token:\n- 第 1、2 级的 logits: 纹丝不动 (变化 0.00e+00)。\n- 第 3 级的 logits: 变了 4.95e-01。\n最细一级 (第 3 级) 的 token 只当标签, 从不当输入。' },
      ],
      links: [
        { from: 'diffusion · DiT', to: 'VARModel', body: '扩散是在噪声强度上由粗到细 (先去大噪声再去小噪声), VAR 是在分辨率上由粗到细; 两者都绕开了逐像素串行。' },
        { from: 'MultiScaleVQ', to: 'ImageTokenizer', body: 'tokenizer 必须先单独训 (重建 loss + vq_loss, demo 150 步 recon MSE 从 0.3700 降到 0.0114), 码本才有意义。码本随机时 token 和图像内容无关, 后面的 Transformer 学到的只是噪声。' },
        { from: 'LLaDA', to: '级内并行采样', body: '两者都是一次前向定多个 token。LLaDA 靠置信度挑先定谁, 没把握的遮回去; VAR 靠尺度结构分批, 粗的先定, 细的后定。' },
      ],
      sourceRows: [
        { concept: 'token 金字塔', code: 'self.num_tokens = sum(s * s for s in scales)', takeaway: '各级边长平方之和。scales=(1,2,4) 时 L = 1+4+16 = 21, 序列长度就是各级面积之和。' },
        { concept: '块状因果 mask', code: 'var.py:block_causal_mask', takeaway: '给每个位置标上它所属的级号, mask[i, j] = 级号(j) ≤ 级号(i): 同级和更粗的级可见, 更细的不可见。' },
        { concept: '逐级采样', code: 'var.py:sample_tokens', takeaway: '每级三步: 把已采样的各级码字累加成重建 f_hat, 下采样成本级的输入特征, 一次前向只取最新一级的 logits, 整级并行采样。' },
        { concept: '下一级的输入', code: 'feats.append(q._down(f_hat, s))', takeaway: '喂给 Transformer 的是 “到目前为止的重建” 在目标分辨率下的样子, 不是上一级的 token 本身。所以序列里没有 shift-by-one。' },
      ],
      snippetTitle: 'next-scale 采样',
      snippet: `f_hat = 0                                   # 累计重建 [D, H, H]
feats, indices = [], []
for k, s in enumerate(scales):              # 1, 2, 4, 8 ...
    if k > 0:
        f_hat = f_hat + upsample(codebook[indices[-1]], H)
        feats.append(downsample(f_hat, s))  # 下一级的输入
    logits = transformer(feats, mask=block_causal)[-s*s:]
    idx = multinomial(softmax(logits / T))  # 整级 s×s 个 token 并行采样
    indices.append(idx.view(s, s))
image = decoder(f_hat + upsample(codebook[indices[-1]], H))

# tokenizer 侧 (多尺度残差量化):
#   r = f - f_hat;  z_k = quantize(downsample(r, s_k));  f_hat += upsample(z_k)`,
      source: ['llm_models/models/generative/var.py:sample_tokens'],
      run: `${RUN}.generative.var.infer_var`,
    },

    'models-multimodal': {
      title: '图像与音频怎样接入语言模型 · 前缀、交叉注意力与 Talker',
      subtitle: 'CLIP、Whisper、Qwen2-VL、Qwen2.5-Omni 四套模型都在回答同一个问题: 不是文本的输入 (图、声音) 怎么变成 Transformer 认的 token, 又从哪一扇门进语言模型。\n读完你能对着任意一个多模态模型说出它的 token 化方式、接入点和训练目标, 并算出一张图要占多少上下文。',
      tldr: '- token 化: 图像切成不重叠的小方块 (patch), 每块变一个向量; 224×224 的图切 16×16 的块得 196 个。声音先变成 mel 声谱 (时间 × 频率的二维图), 沿时间用卷积切成帧; Whisper 把 30 秒的 3000 帧压成 1500 帧。\n- 接入: CLIP 只把图和文各压成一个向量比相似度, 不生成; Qwen2-VL 把视觉 token 当前缀拼在文本前面送进 decoder; Whisper 的 decoder 每层用 cross-attention 去读音频编码器的输出; Omni 的 Talker 用 cross-attention 读 Thinker 的隐状态。\n- 训练目标: CLIP 是图文双向的对比交叉熵; Qwen2-VL 和 Whisper 是文本位置的 next-token 交叉熵; Omni 是文本 CE 加 0.5 倍语音 CE。',
      question: '一张图要占语言模型多少个 token? 把它拼在文本前面, 和让 decoder 用 cross-attention 去看, 差在哪?',
      evolution: {
        title: '把眼睛和耳朵接到同一个 Transformer 上',
        subtitle: '根问题: Transformer 只吃 token 序列, 图像和声音都不是 token; 接进去之后, 前缀越长注意力越贵',
        steps: [
          { name: '各训各的专用模型', pain: '(原点) 视觉模型只认固定 N 类标签, 语音识别是声学 + 语言 + 对齐三级流水线', fix: '每种模态一套结构、一套标签; 换个类别就要重新标注, 彼此的表示对不上' },
          { name: 'CLIP 对比对齐', year: 2021, pain: '标签要人标, 类别定死; 网上 4 亿图文对里的自然语言没法当监督', fix: '图像塔 ViT (看图的 Transformer) 和文本塔各出一个向量, batch 内配对的拉近、其余 B−1 个推开; 只能打分, 不能生成' },
          { name: 'Whisper 交叉注意力', year: 2022, pain: 'CLIP 只会打分; 要从声音生成文字, 得有一个 decoder 去读模态编码器', fix: 'mel 经卷积切成帧, 编码器双向读完; decoder 每层用 cross-attention 取它当 K/V' },
          { name: 'Qwen2-VL 前缀', year: 2024, pain: 'cross-attention 要在 LLM 每层加一块, 预训练好的结构得改; 多图交错不好排', fix: 'patch 经 ViT、projector 变成 token 拼在文本前, LLM 不改; 文本看得到图, 反之不行' },
          { name: 'Thinker / Talker', year: 2025, pain: '前缀路线只出文本; 要说话得再接一个 TTS, 语气丢了, 还得等整句写完', fix: 'Thinker 出文本, Talker 用 cross-attn 读它的隐状态, 吐语音 codec token' },
        ],
      },
      code: 'llm_models/models/multimodal/{clip,whisper,qwen2_vl,qwen2_5_omni}.py · llm_models/layers/multimodal/multimodal.py',
      points: [
        { title: '进语言模型只有两扇门', body: '非文本输入先被各自的编码器变成一串向量, 再从两扇门之一进语言模型。\n- 拼前缀 (Qwen2-VL、Omni 的 Thinker): 视觉 token 拼在文本 token 前面, 同一个因果 decoder 从头读到尾, 文本位置因为排在后面, 天然看得到前面的图。好处是 LLM 一行不改, 多张图可以和文本交错排, KV cache 照常用。代价是每张图占掉 N_v 个上下文位置。\n- cross-attention (Whisper、Omni 的 Talker): 编码器的输出不进 decoder 的序列, 只当 K/V; decoder 每层在自注意力之后多一块 cross-attention, 用自己的 Q 去读它们。好处是不占 decoder 的序列长度; 代价是每层都要加一块, 预训练好的 LLM 结构得改。\nCLIP 两扇门都不走: 它把图和文各压成一个向量比相似度, 没有 decoder, 只能打分, 不能生成。', key: true },
        { title: '图切 patch, 声谱切帧', body: '- 图像: PatchEmbed2D 用一个 kernel = stride = patch 的卷积把图切成不重叠的方块, 每块线性变换成一个向量, 和 “切块、展平、过 Linear” 完全等价。224×224 配 16×16 的 patch 得 14×14 = 196 个 token (不计 CLS, CLIP 视觉塔额外拼的一个汇总 token); Qwen2-VL 原版 patch 是 14, 同一张图 16×16 = 256 个。\n- 音频: 波形先经采样率对齐、分帧 / STFT、mel 滤波和 log, 变成 mel 声谱 [80, T_mel], 每 10 ms 一帧、80 个频率通道。Whisper 再用两层 Conv1d 沿时间卷积 (第二层 stride=2, 帧数减半) 切成帧 token: 30 秒 3000 帧变 1500 帧 (每秒 50 个)。demo 里 mel (2, 80, 100) 进去, 编码器出 (2, 50, 128)。\n- Omni 的音频走另一条路: 把 mel 声谱当 1 通道的图像, 和图像共用 PatchEmbed2D 切 patch。\n边界: 本库没有音频前处理, Whisper 的 mel 是 randn 直接构造的。' },
        { title: '一张图占多少上下文', body: '视觉 token 进了 decoder 之后, 和文本 token 待遇一样: 一样要过注意力, 一样要存 KV。\n- 边长翻倍, patch 数变 4 倍: 224 变 448 时, 196 变 784 个。\n- 拼成前缀后序列长度从 T 变成 T+N: 注意力分数矩阵按 $(T+N)^2$ 涨, KV cache 按 T+N 涨。\n压缩能省: 本库的 Qwen2-VL 用 Perceiver Resampler, 它有一组可学的 latent 向量当 query, 用 cross-attention 从 N 个 patch 里 “挑” 信息, 输出个数只由 latent 的个数决定 (demo 把 16 个 patch 压成 8 个)。代价是每个 latent 不再对应图上一个具体位置, 小字和细节会丢; 所以原版 Qwen2-VL 不用 Resampler, 用相邻 2×2 patch 合并再过 MLP, 保留网格结构。' },
        { title: '前缀是单向的: 文本看图, 图看不到文本', body: 'Qwen2-VL 整条序列都走因果 mask, 视觉 token 在前。infer 脚本的三条断言:\n- 换图, 文本位置 logits 变化 0.0034: 文本看得到视觉前缀。\n- 换文本, 视觉位置输出变化 0.0e+00: 前缀看不到后面的文本。\n- 改 pad 处的 token, 其余位置变化 0.0e+00: 文本的 padding mask 自动在前面补了 N_v 个 1, 视觉段永远有效。\nloss 只算文本位置, 视觉位置 label = −100 (cross_entropy 跳过): 图是条件, 不是预测目标。\n位置编码用 M-RoPE: 把每个 head 的维度切成 (时间, 行, 列) 三段, 各自按那一轴的坐标旋转。patch 的位置是 (0, 行, 列), 文本三轴取同一个值。所以纯文本时它与普通 1-D RoPE 逐位相等 (demo 数值差 0.0e+00); 对调视觉 patch 的行列坐标, 输出变 5.3e-05, 未训练的 4×4 小网格上很小但非零。' },
        { title: '三种训练目标, 都靠一条下界验收', body: 'demo 都在固定的随机数据上训, 怎么证明模型真的读到了另一个模态, 而不是光靠文本背答案? 办法是算一个 “不看另一个模态最多能做到多好” 的下界, loss 低于它才算数。\n- CLIP: B 张图、B 句话拼成 B×B 相似度矩阵, 对角线是配对。$\\mathrm{logits} = \\exp(t)\\cdot I\\,T^\\top$, I 和 T 是归一化后的图、文向量; loss 是图找文、文找图两个方向 CE 的平均。$\\exp(t)$ 是可学的温度倒数 $1/\\tau$, 初值 1/0.07 ≈ 14.3, 上限 100, 否则模型靠无限放大 logits 压 loss。demo: 初始 loss 2.180 ≈ ln 8 (8 个候选里瞎猜), 30 步后 0.0004, 对角线 top-1 从 0.12 到 1.00。\n- Qwen2-VL / Whisper: 文本位置的 next-token CE。train 脚本故意让 2 条样本文本相同、图 (或音频) 不同、标签不同: 只看文本最多在两个答案里二选一, loss 下界 ln 2 = 0.693。Qwen2-VL 100 步后 0.198, Whisper 150 步后 0.041, 都低于下界, 说明文本位置真的读到了另一个模态。\n- Omni: total = 文本 CE + 0.5 × 语音 CE。text_loss 从 6.269 降到 0.006, audio_loss 从 5.294 降到 0.025; 两条样本的 codec 输入相同、标签不同, 不读 Thinker 时 audio_loss 的下界也是 ln 2, 实测远低于它。' },
        { title: 'Talker 只读隐状态, 梯度会回流', body: 'Omni 有两个模型: Thinker 是吃全部模态、输出文本的 LLM; Talker 是一个小得多的自回归模型, 输出语音 codec token (离散的语音编码, 由外部 codec 还原成波形)。Talker 不读 Thinker 生成的文本, 读的是 Thinker 最后一层 norm 之后的隐状态 thinker_hidden: 文本只是隐状态的有损投影, 语气这类信息留在隐状态里。\ninfer 脚本 (text_logits [2, 4+4+4+10, 500]: 图、视频、音频各 4 个 token 加 10 个文本 token; audio_logits [2, 12, 200]):\n- 换图, 语音 logits 变化 4.0e-05: 图经 Thinker 到 Talker 这条通路是通的; 未训练时权重 std 0.02, 所以数值很小。\n- 换 codec 输入, 文本 logits 变化 0.0e+00: 前向不走回头路, Thinker 不依赖 Talker。\n- 改未来 codec, 过去位置变化 0.0e+00: Talker 自己也是因果的。\n两套词表、两个 head: Thinker 的 lm_head 和 embedding 共享权重, Talker 的不共享。训练时梯度会回流: thinker_hidden 传给 Talker 前没有 detach, audio_loss 的梯度一路流回 Thinker; 要隔离就在传之前 .detach()。' },
        { title: '实验边界: 通路可学, 不等于会看会听', body: '- 数据是固定的随机 batch: 图是 randn, 文本是随机 token; Whisper 的 mel 也是 randn 直接构造的, 不是从波形算的。loss 降到 0 只说明模型背下了这个 batch, 不说明它会看图或听声。\n- 规模: CLIP 两塔各 2 层、d=64, batch 8 (原版 32768, 负样本多任务才难); Qwen2-VL 的 LLM 和 ViT 各 2 层、d=128, 图 56×56; Omni Tiny 总参数 504,352。\n- 没有的东西: 音频前处理 (重采样、分帧、mel 滤波)、把 codec token 还原成波形的 codec 和声码器、Qwen2-VL 与 Omni 的 generate() 和 KV cache、原版的动态分辨率和 TMRoPE (按真实时间戳对齐音视频位置)。\n- 本库约定: Qwen2-VL 和 Omni 的 Thinker 绑 lm_head 权重且 embedding 乘 √D; Whisper 绑权重但不乘 √D; CLIP 不乘; Omni 的 Talker 乘 √D 但不绑。初始化用 N(0, 0.02²) 才有初始 loss ≈ ln V; 换成默认 N(0,1) 初始化加绑权重, 初始 loss 变成 Qwen2-VL 130.29、Whisper 82.30、Omni 的 text_loss 124.15。' },
      ],
      links: [
        { from: 'Qwen2VLDecoder', to: 'LLaMA', body: 'Qwen2-VL 的 decoder 就是 LLaMA 式的 GQA + SwiGLU + RMSNorm + RoPE, 多出来的只是一个 inputs_embeds 入口 (让外面拼好的向量序列直接进来) 和 M-RoPE 开关。Omni 的 Thinker 直接复用这个类。' },
        { from: 'PreLNCrossBlock', to: 'Transformer (2017) 的 decoder', body: 'Whisper 的 decoder 和 Omni 的 Talker 都是 2017 原版 decoder 的样子: 因果 self-attention, 再 cross-attention 读编码器, 再 FFN。Whisper 和原版 Transformer 同构, 只把 encoder 的输入从 token 换成 log-mel 声谱。' },
        { from: '视觉前缀', to: 'KV cache 生成', body: '前缀路线天然支持 KV cache: 图像 token 只在 prefill 时算一次, 之后每步 decode 只算新的文本 token。本库的 Qwen2VLDecoder 没接 GenerationMixin, 只能整段前向。' },
        { from: 'PatchEmbed2D', to: 'diffusion · DiT', body: '多模态理解和图像生成在 “先切 patch 再过 Transformer block” 这一步是一回事; 区别是 DiT 切的是 VAE 压缩后的 latent, 这里切的是像素和声谱。' },
      ],
      sourceRows: [
        { concept: '切 patch', code: 'multimodal.py:PatchEmbed2D', takeaway: 'kernel = stride = patch 的 Conv2d: 无重叠卷积, 和 “切块、展平、过 Linear” 等价。token 数 N = (H/p)·(W/p); 图像 (C=3) 和 mel 声谱 (C=1) 共用同一个类。' },
        { concept: '对比 logits', code: 'clip.py:CLIPModel.forward', takeaway: 'forward 只返回归一化后的图、文特征和 logit_scale = exp(t).clamp(max=100); [B, B] 相似度矩阵在 loss 里算, 对角线是配对, 其余 B−1 个是免费的负样本。' },
        { concept: '拼前缀', code: 'embeds = torch.cat([vision_embeds, embeds], dim=1)', takeaway: '视觉在前, 文本在后。文本的 padding mask 前面补 N_v 个 1。视觉 token 不乘 √D, 尺度由 projector 末尾的 LayerNorm 定。' },
        { concept: 'M-RoPE 位置', code: 'qwen2_vl.py:build_mrope_position_ids', takeaway: '返回 [3, B, H·W + T] 的三轴位置: patch 是 (0, 行, 列), 文本三轴同值、从 max(H, W) 起步, 接在视觉段之后。' },
        { concept: '声谱切帧', code: 'whisper.py:WhisperAudioEncoder.forward', takeaway: '两层 Conv1d, 第二层 stride=2 把帧数减半, 再加 sin 位置编码; 编码器没有 mask, 整段音频互相可见。' },
        { concept: '唯一通道', code: 'layer(x, context=encoder_hidden, self_mask=self_mask, context_mask=None)', takeaway: 'encoder_hidden 是音频进 decoder 的唯一通道, 当 cross-attention 的 K/V。cross-attention 不设 mask: 整段音频在解码前已完整可得, 每个文本位置都能看全部帧。' },
        { concept: 'Thinker → Talker', code: 'qwen2_5_omni.py:Qwen2_5_OmniModel.forward', takeaway: 'Thinker 以 return_hidden=True 返回 (logits, hidden); hidden 不 detach 直接当 Talker 的 context, 维度不同时先过 thinker_to_talker 线性层。' },
      ],
      snippetTitle: '两扇门: 拼前缀, 或 cross-attention',
      snippet: `# 1) token 化: 图切 patch, 声谱切帧
v = projector(resampler(vit(patch_embed(image))))     # [B, N_v, D]      Qwen2-VL
a = encoder(gelu(conv2(gelu(conv1(mel)))))            # [B, T_mel/2, D]  Whisper

# 2a) 拼前缀: 视觉在前, 同一个因果 decoder 从头读到尾
x = cat([v, emb(text) * sqrt(D)], dim=1)              # [B, N_v + T, D]
logits = decoder(inputs_embeds=x)                     # loss 只算文本位置 (视觉 label = -100)

# 2b) cross-attention: 编码器输出只当 K/V, 不占 decoder 序列
for layer in decoder.layers:
    h = h + self_attn(h, mask=causal)
    h = h + cross_attn(q=h, kv=a)                     # 每个文本位置都能看整段音频
    h = h + ffn(h)

# 3) Omni: Thinker 出文本, Talker 用 cross-attention 读 Thinker 的隐状态
text_logits, hidden = thinker(inputs_embeds=cat([v, a, emb(text)]), return_hidden=True)
audio_logits = talker(codec_ids, context=hidden)      # 没有 detach: 语音 loss 的梯度会流回 Thinker
loss = ce(text_logits) + 0.5 * ce(audio_logits)`,
      source: [
        'llm_models/models/multimodal/qwen2_vl.py:Qwen2VLModel',
        'llm_models/models/multimodal/whisper.py:WhisperAudioEncoder',
        'llm_models/models/multimodal/qwen2_5_omni.py:Qwen2_5_OmniModel',
      ],
      run: `${RUN}.multimodal.qwen2_vl.infer_qwen2_vl`,
    },
  },
}

// 机制实验的原理、运行入口与源码引用。
const supplements = {
  "models-mtp": [
    {
      "title": "可达范围是上限, 不是实际能力",
      "body": "第 1 条算的感受野上界是一个结构上限: 信息最多能从多远的位置流到这里。模型实际用不用得上, 要另外测。\n这个上界只在三个条件同时成立时才精确:\n- 纯因果滑窗: 每一层都是滑窗, 中间没有插全注意力层。\n- 窗口 W 含当前位置: 窗口里的 W 个位置有一个是自己, 所以每层只往前多够 W−1 个。\n- 没有 dilation, 也没有全局通路: dilation 是窗口里隔几个位置取一个, 用同样的格数伸得更远; 全局通路指全注意力层或永远可见的 sink 列, 能让信息一步跨过很远。有了这两样, 上界就不是下面这个式子了。\n条件满足时, 位置 t 能够到的输入数是 min(t+1, 1+L×(W−1))。取 min 是因为位置 t 前面 (含自己) 一共只有 t+1 个 token。\n- W=8、L=4: 1+4×7=29, 不是 4×8=32。L×W 把每层的「自己」都重复算了一次, 只在窗口很大时才是好的近似。\n- Mistral 32 层、W=4096: 1+32×4095=131041。\n这个数既不决定模型的上下文上限 (最多接受多少 token), 也不决定长距离检索能不能成功。这两样都要另外测。",
      "source": "llm_models/models/language_models/mistral.py:receptive_field"
    }
  ],
  "models-gptoss": [
    {
      "title": "一步够得到, 不等于用得上",
      "body": "第 1 条说过, GPT-OSS 隔一层插一层全注意力。所以从结构上看, 任何位置在全注意力层里一步就能够到全部历史, 不用逐层接力。\n但「够得到」只说明信息有路可走。模型实际会不会去用远处的某个 token, 取决于三件事: 训练时见过什么样的数据 (训练分布); 注意力权重最后把概率分给了谁; 中间各层的表示有没有把那条信息留下来。这三样都不是结构图能告诉你的。\n下方的滑窗实验台算的是可达范围: 某个输出位置能连到哪些输入。它不是长上下文任务的准确率, 那个要拿真实任务另外测。",
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
topic.pages["models-mtp"].related = [{"route": "attention", "label": "复习注意力掩码", "reason": "掩码决定谁能看见谁; 这里只讲多 token 预测, 滑杆实验不再重复。"}]
export default topic

const qkYarn = topic.pages['models-qknorm-yarn']
const mixed = topic.pages['models-mtp']
const supplement = (page, id, title, points, labNames) => ({ ...page, id, title, points, labNames, evolution: null, links: [], related: [], widgets: [] })
topic.pages.attention = { sections: [
  { ...supplement(qkYarn, 'qk-norm', 'QK-Norm · 控制注意力数值', qkYarn.points.slice(0, 2), ['ModelQkNormLab']), tldr: '- 问题: 注意力分数 (logit, 进 softmax 之前的分数) 是 q 和 k 的点积。q、k 的长度一起涨 2 倍, 分数就涨 4 倍。demo 把 W_q、W_k 乘 10, 最大 logit 从 1.20 冲到 120.31, softmax 塌成只盯一个位置。\n- 做法 (QK-Norm): 在 RoPE 旋转之前, 对每个头的 q、k 各做一次 RMSNorm, 把长度拉回固定值。分数只由夹角和一个可学增益决定, 同样乘 10 也稳在 3.44。' },
  { ...supplement(mixed, 'swa', '滑窗与跨层感受野', [mixed.points[0], mixed.points[1], mixed.points[5]], []), tldr: '- 上面的掩码实验台只画了一层: 滑窗里每个 token 只看最近 W 个。这一段看多层叠起来以后, 信息怎样一层一层往前接力: L 层最多够到 $1+L(W-1)$ 个位置, W=8、L=4 时是 29 个。\n- 够得到只是结构上限。训练后的模型能不能真从那么远取回信息, 要另外测。' },
] }
topic.pages.position = { sections: [
  { ...supplement(qkYarn, 'yarn', 'YaRN · 位置频率怎样外推', qkYarn.points.slice(2), ['ModelRopeScaleLab']), tldr: '- 问题: RoPE 把每两个维度当成一根表针, 按位置转角度, 不同的针转速不同。转得慢的低频针在训练长度内一圈都没转完; 位置一超过训练长度, 它就转到模型没见过的角度, 长度外推就从这里崩。\n- YaRN: 按每根针在训练长度内转了几圈来分段缩放。高频针不动, 低频针转速 $\\div s$ (s 是长度要扩的倍数), 中间平滑过渡。实验台把 PI、NTK-aware、YaRN 放在一起比两件事: 相邻 token 还分不分得开 (局部区分), 远处位置有没有转出训练范围 (长距离外推)。' },
] }
topic.pages['models-mamba'].sections = [
  { ...supplement(mixed, 'linear-hybrid', '进阶 · DeltaNet 与混合线性架构', mixed.points.slice(3, 5), ['ModelLinearAttnLab']), tldr: '- Gated DeltaNet 和 Mamba 一样, 用一个固定大小的状态代替 KV cache, 但两者的机制不同: DeltaNet 的状态是一个矩阵, 每来一个 token 先用 k 擦掉旧值, 再写入新的 v (delta rule); Mamba 的选择性 SSM 靠 Δ 控制衰减和写入。\n- 两者都把全部历史压进定长状态, 压缩有损, 精确召回很久以前的某个 token 不如注意力。所以 Qwen3-Next 每 4 层留 1 层全注意力兜底。' },
]
mixed.title = 'MTP 多 token 预测 · 一次前向, K+1 份监督'
mixed.subtitle = '读完你能说清: MTP (multi-token prediction, 让每个位置多预测后面几步) 的第 k 级明明看到了后面的真 token $t_{i+k}$, 为什么不算偷看答案; 以及训练时拼进去的真 token 和生成时它给出的草稿, 为什么不是一回事。'
mixed.tldr = '- 做法: 普通语言模型在位置 i 只预测下一个 token $t_{i+1}$。MTP 在主干后面串几级小模块: 第 k 级在位置 i 把真实 token $t_{i+k}$ 的 embedding 拼进来, 过一个 Transformer Block, 预测 $t_{i+k+1}$。每深一级多看一个真 token, 也多预测一步, 所以永远看不到自己要预测的那个 token, 因果链不断。\n- 训练: 总损失是 $\\mathrm{CE}(\\text{main}) + \\lambda\\cdot\\mathrm{mean}_k\\,\\mathrm{CE}(\\text{mtp}_k)$。这个式子在说: 主干的交叉熵, 加上各级 MTP 交叉熵的平均再乘 λ。DeepSeek-V3 的 $\\lambda$ 取 0.3, 后期降到 0.1。\n- 成本和边界: 每级只新增一个拼接投影和 1 个 Block, embedding 和输出头 lm_head 都借主干的。丢掉 MTP 模块, 主 logits 和同权重的 LLaMA 逐元素相等。生成时它对 t+2 的预测只是投机草稿, 还要目标模型验证才能收下。'
mixed.points = [mixed.points[2]]
mixed.evolution = null
mixed.related = [{ route: 'attention', label: '滑窗与跨层感受野', reason: '滑窗 (每层只看最近 W 个) 和跨层感受野挪到了注意力一章的补充段, 和掩码实验台放在一起讲。' }, { route: 'models-mamba', label: '状态空间与混合线性架构', reason: '用定长状态代替 KV cache 的两条路 (Mamba 和 DeltaNet) 放在同一章对照: 各省了什么, 各丢了什么。' }, { route: 'infer-tree-speculation', label: '草稿怎样被验证', reason: 'MTP 训练出的 t+2 预测只是草稿。生成时草稿怎样交给目标模型一次验证、哪些被收下, 在那一章。' }]
delete topic.pages['models-qknorm-yarn']
topic.chapters = topic.chapters.filter(c => c.route !== 'models-qknorm-yarn')
topic.pages['models-dsa'].links.find(l => l.from.includes('掩码实验台')).from = 'attention · 掩码实验台'

for (const chapter of topic.chapters) chapter.label = chapterLabels[chapter.route] || chapter.label

// 进阶段落按主题选择源码与骨架。
const qk = topic.pages.attention.sections[0]
const swa = topic.pages.attention.sections[1]
const yarn = topic.pages.position.sections[0]
const delta = topic.pages['models-mamba'].sections[0]
qk.sourceRows = qkYarn.sourceRows.filter(r => /norm|logit|增益/i.test(r.concept + r.code))
yarn.sourceRows = qkYarn.sourceRows.filter(r => !qk.sourceRows.includes(r))
qk.source = qkYarn.source.filter(s => /[Nn]orm/.test(s))
yarn.source = qkYarn.source.filter(s => !qk.source.includes(s))
qk.snippet = 'q, k = q_norm(q), k_norm(k)\nq, k = rope(q, k)\nscores = q @ k.transpose(-1, -2) / sqrt(head_dim)'
yarn.snippet = 'turns = train_length * inv_freq / (2 * pi)\n# 高频保持, 低频按扩展倍率缩放, 中频插值\ninv_freq = yarn_scale(inv_freq, turns, factor)'
swa.sourceRows = mixed.sourceRows.filter(r => ['带状掩码', 'KV 上限'].includes(r.concept))
delta.sourceRows = mixed.sourceRows.filter(r => ['delta rule', '混合排布'].includes(r.concept))
swa.source = mixed.source.filter(s => s.includes('mistral'))
delta.source = mixed.source.filter(s => s.includes('linear_attention'))
swa.snippet = 'visible = (j <= i) & (j > i - window_size)\nreachable = min(position + 1, 1 + layers * (window_size - 1))'
delta.snippet = 'state = alpha * (state - beta * outer(k, k @ state)) + beta * outer(k, v)\ny = q @ state  # 压缩有损; 混合架构另保留全注意力层'
mixed.sourceRows = mixed.sourceRows.filter(r => ['级联输入', '标签对齐'].includes(r.concept))
mixed.source = mixed.source.filter(s => s.includes('/mtp.py'))
mixed.links = mixed.links.filter(l => !l.from.includes('sliding'))
mixed.snippet = mixed.snippet.slice(mixed.snippet.indexOf('# MTP'))
mixed.snippetTitle = 'MTP 的监督与级联'
mixed.code = 'llm_models/models/language_models/mtp.py'
mixed.evolution = { title: '从一份监督到 K+1 份', subtitle: '根问题: 每个位置只有下一个 token 这 1 份监督, 生成时也一步只出 1 个 token', steps: [
  { name: '下一 token 预测', pain: '(原点) 位置 i 只学紧接的下一个 token $t_{i+1}$: 每个位置只有 1 份监督', fix: '主干算出 hidden, 输出头 lm_head 预测紧接的下一个 token; 训练和生成都是一步一个' },
  { name: '并行多头 MTP', year: 2024, pain: '监督稀: hidden 只为下一个 token 服务, 不用为更远处打算; 生成也一步只出 1 个', fix: 'Gloeckle 等人并排接 K 个独立输出头, 第 k 个直接预测 t+k; 它看不到中间的 token, 头与头之间没有因果链' },
  { name: '级联 MTP', year: 2024, pain: '各头独立预测 t+k, 中间隔着的 token 只能靠猜, 因果链断了', fix: 'DeepSeek-V3 改成串联: 第 k 级拼上真 token $t_{i+k}$ 的 embedding, 过一个 Block 再预测; 输出头借主干的' },
  { name: '当投机草稿', pain: '训练时有真 token 可拼, 生成时没有, 仍一步只出 1 个', fix: '主干先出 t+1, 喂给第 1 级 MTP 得到 t+2 的草稿; 草稿交目标模型验证, 不直接全收 (llm_infer m07)' },
] }
for (const route of ['attention', 'position']) Object.assign(topic.pages[route], { points: [], links: [], sourceRows: [], widgets: [] })

canonicalizeRelated(topic)

qk.source = ['llm_models/layers/core/attention.py:GroupedQueryAttention']
qk.question = '都已经除以 $\\sqrt{d}$ 了, 为什么还要单独管 q、k 的长度?'
yarn.question = '把上下文拉长 4 倍时, 怎么分别检查「相邻 token 还分得开」和「低频维度没转出训练范围」?'
swa.question = '每层只看最近 W 个位置, 叠很多层以后能连到很远, 那模型是不是就真能记住很远的内容?'
delta.question = '把 KV cache 换成一个固定大小的状态矩阵, 省下了什么, 又丢掉了什么?'
mixed.question = '第 k 级已经把位置 i 后面的真 token $t_{i+k}$ 拼了进来, 为什么不算提前看到了它要预测的 $t_{i+k+1}$?'
qk.related = [{ route: 'position', anchor: 'yarn', label: '继续看 YaRN: 把 RoPE 拉到更长' }]
yarn.related = [{ route: 'attention', anchor: 'qk-norm', label: '对照 QK-Norm: 管住注意力分数的大小' }]
