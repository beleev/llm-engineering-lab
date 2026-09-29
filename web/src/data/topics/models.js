// 阶段 2 · llm_models 的扩展章节: 2024–2025 的新零件。每章一个 (或两个) 实验台, 挂载关系见 data/labmap/models.js。
// models-mtp 的章节登记在 data/models.js 的 modelChapters 里, 所以不出现在这里的 chapters。
const A = 'llm_models/layers/core/attention.py'
const RUN = 'python -m llm_models.run_models'

export default {
  stage: 'models',
  chapters: [
    { route: 'models-generation', label: 'KV cache 生成', hint: '旧 token 的 K/V 不会变: prefill 一次, 之后每步只算 1 个' },
    { route: 'models-qknorm-yarn', label: 'QK-Norm 与 YaRN', hint: '一个管 logit 别炸, 一个管 RoPE 能不能拉长' },
    { route: 'models-mamba', label: 'Mamba 选择性 SSM', hint: '输入自己决定 Δ: 记谁、忘谁; 状态大小恒定' },
    { route: 'models-moe-balance', label: 'Aux-loss-free 均衡', hint: '偏置只管选谁上场, 不碰门控权重' },
    { route: 'models-dsa', label: 'DSA 稀疏注意力', hint: '便宜的 indexer 粗选 top-k, 再用 KL 把它教会' },
    { route: 'models-gptoss', label: 'GPT-OSS 结构', hint: '滑窗/全注意力隔层交替 + 可学 sink logit + MoE' },
    { route: 'models-llada', label: 'LLaDA 扩散语言模型', hint: '随机比例遮盖训练, 采样时把没把握的遮回去' },
    { route: 'models-var', label: 'VAR 逐尺度生成', hint: '1×1 → 2×2 → …, 由粗到细, 每级内部并行' },
  ],
  pages: {
    'models-mtp': {
      widgets: ['AttnMaskLab', 'MtpLab'],
      title: 'SWA · MTP · 混合线性 — 三种给 Transformer 减负的改法',
      subtitle: '读完这章, 你看一个新模型时能马上分辨它动了哪一处:\n- 把 mask 裁窄了 (Mistral)?\n- 让每个位置多预测几步 (MTP)?\n- 还是把装历史的容器换掉了 (Qwen3-Next)?',
      tldr: '- SWA: 不新增任何类, 只换一张 mask。KV cache 从 $O(T)$ 封顶到 $O(W)$。\n- MTP: 用共享 lm_head 的级联模块, 每个位置的监督从 1 份变成 K+1 份。推理时还白送一份投机解码草稿。\n- 混合线性 (Qwen3-Next): 用固定大小的状态矩阵代替 KV cache。每 4 层留 1 层全注意力兜底。',
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
        { title: '深度换宽度', body: 'T=64、W=8 时, 带状 mask 把可见格子从 2080 降到 484。\n单层只看 W 个位置。但第 2 层的邻居已经各自汇总过它们的邻居, 信息跨层接力。\nL 层的理论感受野 $\\approx L\\cdot W$ (Mistral 32 层 × 4096 ≈ 131K)。\ndemo 特意用 1 层模型验证: 改位置 0, 只影响位置 0..7 的输出。' },
        { title: 'mask 即架构', body: '- 全因果: LLaMA\n- 带状: Mistral\n- 带状加 sink: GPT-OSS\n- top-k: DSA\nQKV 投影一行不改, 谁能看见谁全由 mask 决定。所以 Mistral 和 LLaMA 参数量完全相同 (demo: 都是 3,076,352)。', key: true },
        { title: '级联保因果', body: 'MTP 第 k 级在位置 i: 拼上真实 token $t_{i+k}$ 的 embedding → 过一个 Block → 预测 $t_{i+k+1}$。\n每深一级, 多看一个真 token、多预测一步, 因果链没断。\n总损失 $= \\mathrm{CE}(\\text{main}) + \\lambda\\cdot\\mathrm{mean}_k\\,\\mathrm{CE}(\\text{mtp}_k)$。DeepSeek-V3 的 $\\lambda$ 取 0.3, 后期降到 0.1。' },
        { title: '状态替代缓存', body: 'Gated DeltaNet 是一种线性注意力: 不存历史 K/V, 只维护一个 Dh×Dh 的状态矩阵 (Dh 是每头维度)。\n- delta rule: 先擦掉 k 方向的旧值, 再写入新值。\n- α 门: 负责整体遗忘。\n它的 “cache” 与读了多少 token 无关。demo 里 T 从 8 涨到 32, 同类层加总的缓存元素数 (batch 2):\n- 注意力层 (共 2 层): 4096 → 16384。\n- delta 层 (共 6 层): 恒为 49152。\n代价是压缩有损。所以 Qwen3-Next 每 4 层留 1 层全注意力, 兜底精确召回。' },
        { title: '序列短的时候, 状态反而更大', body: '上面的 49152 比 16384 大。拆到单层、单条序列再比:\n- 注意力层: 每个 token 存 128 个数 (2·Hkv·Dh = 2×2×32), 随 T 涨。\n- delta 层: 固定 4096 个数 (H·Dh·Dh = 4×32×32)。\nT 超过 32 (4096 ÷ 128), 状态才比 KV cache 小。' },
      ],
      links: [
        { from: 'build_sliding_window_mask', to: 'Mistral.forward', body: 'Mistral 与 LLaMA 的全部结构差异就是这张带状 mask, 连参数量都一样。' },
        { from: 'MTPModule', to: 'lm_head (共享)', body: '每级只新增一个拼接投影加 1 个 Block。embedding 和输出头都借主干的, 所以丢掉 MTP 模块后, 主 logits 与同权重 LLaMA 逐元素相等。' },
        { from: 'mtp_logits', to: 'speculative decoding', body: 'MTP 对 t+2 的预测可以直接当草稿送去验证, 对应 llm_infer/m07。' },
      ],
      sourceRows: [
        { concept: '带状掩码', code: 'masks.py:build_sliding_window_mask', takeaway: '可见条件是 ($j \\le i$) 且 ($j > i-W$); sink_tokens > 0 时额外保留开头 S 列。' },
        { concept: 'KV 上限', code: 'mistral.py:kv_cache_entries', takeaway: 'rolling buffer 只留最近 W 个: $\\min(T, W)$。读了 21 个 token, 每层 cache 仍然只有 8 个。' },
        { concept: '级联输入', code: 'mtp.py:MTPModule.forward', takeaway: 'h 和 embedding 各自 RMSNorm 再拼接。h 经过多层残差累加, 尺度比 embedding 大得多, 不先归一化就得让投影矩阵花容量去对齐尺度。拼完投影回 d 维, 过 Block。' },
        { concept: '标签对齐', code: 'loss.py:MTPLoss.compute', takeaway: '第 k 级的目标是 labels 左移 k 位, 末尾 k 个位置没有未来 token, 填 −100 屏蔽。' },
        { concept: 'delta rule', code: 'layers/sparse/linear_attention.py:GatedDeltaNet', takeaway: '$S \\leftarrow \\alpha(S - \\beta k(k^\\top S)) + \\beta k v^\\top$。q/k 先做 L2 归一化, $I - \\beta k k^\\top$ 才是收缩映射, 状态不会数值爆炸。' },
        { concept: '混合排布', code: 'qwen3_next.py:layer_types', takeaway: '[Δ,Δ,Δ,A,…] 周期排布。mask 和 rope 对 Δ 层是 no-op, 主干循环不区分层类型。' },
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
      subtitle: '看完这章, 你能对着 generate() 的三个分支说出每一支在干什么。你也能解释 MLA 为什么只缓存 latent 而不缓存 K/V。',
      tldr: '每生成一个 token, 序列只长 1, 旧 token 的 K/V 一个字节都不变。\n存下来之后, 生成 N 个 token 的 token 前向次数从 $O(N^2)$ 降到 $O(N)$。\n- MHA/GQA: 存 K、V 本体。\n- MLA: 只存低秩 latent c_kv 和共享的 k_rope, 每步现场升维。',
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
        { title: '只有 K/V 值得存', body: '第 t 步要算的只有最后一个 token 的输出: 拿最新 token 的 Q, 查所有历史 token 的 K/V。\n- 旧 token 的 Q: 再也用不上。\n- 旧 token 的 K/V: 因果 mask 下永远不会变。\n所以缓存 K/V, 每步只前向 1 个 token。', key: true },
        { title: 'prefill 和 decode 是两种负载', body: '- prefill: 一次吃下整段 prompt。算力密集, 整段并行。\n- decode: 每步只处理 1 个 token。访存密集, 只能串行。\ngenerate() 看 cache.pos 是不是 0 来分这两条路。阶段 5 的调度器就是冲着这两种负载设计的。' },
        { title: 'MLA 存 latent, 不存 K/V', body: 'MLA 的 cache 里只有两样: c_kv [S, r], 和 RoPE 之后的共享 k_rope [S, rope]。\nK-nope 和 V 每步从 latent 现场升维算出来。多算一点, 换 cache 变小:\n- DeepSeek-V3: 每 token 每层存 576 个数 (r=512 加 rope=64)。\n- 同规模 MHA (128 头 × 128 维): 要 32768 个。MLA 省约 98%。' },
        { title: 'c_kv 不能带 RoPE', body: '升维矩阵是固定的。推理时可以把它提前乘进 Q 侧的投影, K 就不用真的升维, 这一步叫吸收。\nRoPE 随位置变。c_kv 带了 RoPE, 旋转就夹在中间, 升维矩阵乘不进去, latent 白存。\n所以每头拆成两段:\n- nope 段: 走 latent, 不旋转。\n- rope 段: 单独投影, 再旋转。\n本章的教学代码每步照常升维, 没做吸收。llm_infer 的 m18 有 absorb=True 分支。' },
        { title: '窗口一满, cache 整段作废', body: '窗口左移会挤掉最早的 token。\n- 内容变了: 第 2 层起, 留下来的 K/V 都是看着那个 token 算出来的, 和按新窗口重算的结果不同。\n- 位置变了: 代码里位置从 0 重新编号, 旧 K/V 的 RoPE 角度对不上。\n所以只能整段重建。窗口满了之后每一步都走这条分支, 等于退回每步重算。' },
      ],
      links: [
        { from: 'llm_basic/sample.py', to: 'GenerationMixin.generate', body: '阶段 1 的采样每步重算整段 forward; 这里加上 cache 分支, 输出逐 token 完全一致。' },
        { from: 'KVCache', to: 'llm_infer KV 与缓存内存', body: '这里的 cache 是每层一个 dict; 阶段 5 把它换成分页的 block 表, 好同时服务多个请求。' },
        { from: 'MultiHeadLatentAttention', to: 'DSA', body: 'DSA 在 MLA 的 cache 上再多存一份 indexer 的 key, 用来给旧位置打分。' },
      ],
      sourceRows: [
        { concept: '三个分支', code: 'generation.py:GenerationMixin.generate', takeaway: '无 cache: 喂整个前缀; pos==0 或窗口已满: (重新) prefill; 其余情况 decode, 只喂 idx[:, -1:]。' },
        { concept: '窗口满了就作废', code: 'cache.pos + 1 > max_len', takeaway: '窗口左移挤掉了最早的 token。第 2 层起的旧 K/V 都依赖它, 位置也从 0 重新编号, 所以重新 prefill。窗口满后每一步都走这条分支。' },
        { concept: '等价性断言', code: 'generation.py:benchmark_kv_cache', takeaway: '贪心生成两遍再 assert torch.equal: cache 只许变快, 不许改结果。' },
        { concept: 'MLA 的 cache', code: 'cache["c_kv"], cache["k_rope"]', takeaway: '只往里追加 latent; k_up / v_up 每步对全部 S 个位置现场升维。' },
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
      subtitle: '- 管数值: logit 不许随 q/k 范数一起长大。\n- 管位置: RoPE 超出训练长度时, 不许看到没见过的角度。\n读完你能说出两者各改了哪一行, 以及为什么顺序不能调。',
      tldr: '- QK-Norm: q、k 在 RoPE 之前各过一次 head_dim 上的 RMSNorm, logit 上界锁死在 $g^2\\sqrt{d}$。\n- YaRN: 按 “训练期转了几圈” 给每个 RoPE 频率分段缩放。高频不动, 低频 $\\div s$, 中间线性过渡, 再乘 mscale 补回注意力温度。',
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
        { title: 'logit 随范数二次增长', body: '$q\\cdot k/\\sqrt{d}$ 里的 $\\sqrt{d}$ 只抵消维度带来的方差, 不抵消范数。训练中 q、k 范数一起涨 2 倍, logit 就涨 4 倍。\ndemo 把权重 ×10: 最大 logit 从 1.20 冲到 120.31。\n- softmax 塌成 one-hot, 雅可比 $\\mathrm{diag}(p) - pp^\\top \\to 0$, 这个 head 收不到梯度。\n- 低精度下还会直接溢出。', key: true },
        { title: 'QK-Norm 只留方向和一个增益', body: '对 q、k 各做一次 RMSNorm (Qwen3 / OLMo-2 / Gemma-3 都这么干): $\\text{logit} = g^2\\cdot\\sqrt{d}\\cdot\\cos\\theta$。\n同样把权重 ×10, 最大 logit 稳在 3.44 不动。模型仍能调可学的 g 决定注意力有多尖, 但只剩这一个旋钮。\n必须放在 RoPE 之前。RoPE 是纯旋转, 不改变范数; 先旋转再乘逐维增益, 会破坏成对维度的旋转结构。' },
        { title: '低频维度没见过那么大的角度', body: '把 RoPE 的每对维度想成一根表针:\n- 高频针: 训练长度 L 内转了几百圈, 什么角度都见过。\n- 低频针: 波长大于 L, 一圈都没转完。位置一超过 L, 就指向训练分布之外。\n衡量标准是越界倍数: 位置 $sL$ 处的角度, 除以训练时见过的最大角度。1.00× 就是没越界, 不处理时是 4.00×。' },
        { title: '三种外推方案, 差在动哪些频率', body: 'd_head=64、L=2048、s=4 时:\n- PI: 所有针一律慢 s 倍。高频被压扁, 相邻 token 的角度差从 1.00 降到 0.25 rad, 分不开。\n- NTK-aware: 只改 base。高频几乎不动, 但中段压不够: 11 个低频维度里只有最低的那个回到 1.00×, 其余最多还越界 1.56×。\n- YaRN: 按圈数 r 分段, 两头都保住。9 个最高频维度 (r>32) 完全不动, 11 个低频维度 (r<1) 全部拉回 1.00×。' },
      ],
      links: [
        { from: 'position · RoPE', to: 'scaled_inv_freq', body: '位置编码一章讲 RoPE 为什么编码相对位置; 这里只改 $\\theta_i$ 这一张频率表, 旋转公式一行不动。' },
        { from: 'RMSNorm', to: 'q_norm / k_norm', body: '和 Block 里的 RMSNorm 是同一个类, 只是作用在 head_dim 上, 每个 head 共享一组 g。' },
        { from: 'QK-Norm', to: 'llm_train 精度与稳定性', body: '阶段 3 从训练侧管稳定性 (clip、warmup、loss scaling); QK-Norm 是从结构侧直接去掉一个不稳定源。' },
      ],
      sourceRows: [
        { concept: 'QK-Norm 开关', code: 'attention.py:GroupedQueryAttention', takeaway: 'self.q_norm = RMSNorm(head_dim) if qk_norm else None。forward 里在 _call_rope 之前调用。' },
        { concept: 'NTK-aware', code: 'base * factor ** (d / (d - 2))', takeaway: '指数 $d/(d-2)$ 正好让最后一个频率 $\\div s$, 第一个频率原样不变。d=64、s=4 时新 base ≈ 41,829。' },
        { concept: 'YaRN 分段', code: 'gamma = ((rotations - beta_slow) / (beta_fast - beta_slow)).clamp(0, 1)', takeaway: '$r = L\\cdot\\theta/2\\pi$ 是训练期转过的圈数; $\\gamma=1$ 原样外推, $\\gamma=0$ 按 PI 内插。' },
        { concept: '温度补偿', code: '0.1 * math.log(factor) + 1.0', takeaway: '上下文变长后 softmax 的分母项变多、分布变平。用 mscale 把 logit 放大一点补回来: s=4 时 mscale = 1.1386, logits ×1.2965。' },
      ],
      snippetTitle: '两处改动各只有几行',
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
      subtitle: '- 注意力: 所有历史都留着, 用的时候再挑。\n- SSM: 只有一个固定大小的状态, 写入那一刻就得决定留什么。\n读完你能说清 Mamba 的答案 (让步长 $\\Delta$ 由输入自己决定), 以及它为此放弃了什么。',
      tldr: '$h_t = \\exp(\\Delta_t\\cdot A)\\cdot h_{t-1} + \\Delta_t\\cdot B_t\\cdot x_t$, $y_t = C_t\\cdot h_t$。\n$\\Delta_t$、$B_t$、$C_t$ 全由 $x_t$ 线性投影得到, 这就是 “选择性”:\n- Δ 大: 清掉旧状态, 写入当前 token。\n- Δ≈0: 当前 token 被跳过。\n解码只需 $O(1)$ 状态。训练在生产实现里靠并行 scan, 教学版是顺序循环。',
      question: '线性时不变的 SSM (S4) 可以写成卷积、训练飞快, Mamba 为什么宁可把这个性质扔了?',
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
        { title: '一个 Δ 同时当写入门和遗忘门', body: 'SSM 原本是连续时间的方程 $h^\\prime(t) = A h(t) + B x(t)$。按步长 $\\Delta$ 走一步, 得到离散的递推:\n- 旧状态的保留率: $\\bar A = \\exp(\\Delta\\cdot A) \\in (0,1)$。\n- 新输入的写入强度: $\\bar B \\approx \\Delta\\cdot B$。\n$\\Delta$ 取两头时:\n- Δ 大: 保留率 → 0、写入变强。“忘掉过去, 记住这个”。\n- Δ → 0: 保留率 → 1、写入 → 0。“这个 token 当没看见”。\n一个标量控制两扇门。', key: true },
        { title: 'LTI 只能按距离加权', body: '$\\Delta$ 固定时, token j 对最终状态的贡献只取决于它离结尾多远。\n它没法表达 “这个词重要、那个是废话”。语言建模 (以及 induction / selective copying 这类任务) 要的恰恰是这个。\n实验台里 LTI 模式下, 关键 token “7” 的占比上限只有 1/10。\n让 $\\Delta$ 随输入变就能挑着记, 代价是系统不再时不变, 写不成一个固定的卷积核。S4 那种卷积训练用不上, 只能换成 scan。' },
        { title: '状态大小与序列长度无关', body: '每层的 “cache” 就是 h [D_inner, N] 加一小段卷积缓冲, 生成 100 万 token 也不增长。\ndemo 生成 300 个 token: 递推约 0.4 s, 每步重算约 7 s, 快 16×。\n代价是压缩有损: 要精确召回很久以前的某个 token, 它不如注意力。Jamba、Qwen3-Next 这类混合架构就是为补这一刀出现的。' },
      ],
      links: [
        { from: 'Gated DeltaNet', to: 'SelectiveSSM', body: '两者都是 “固定大小状态 + 输入相关的门”。DeltaNet 的状态是矩阵、用 delta rule 覆写; Mamba 的状态是对角 SSM、用 $\\Delta$ 控制衰减。' },
        { from: 'KV cache', to: 'cache["h"]', body: '同一个 GenerationMixin.generate(): 注意力层往 cache 里追加 K/V, Mamba 层就地更新 h。' },
        { from: 'MambaBlock', to: 'Block 组装器', body: 'Mamba block = Pre-RMSNorm + MambaLayer + 残差, 没有单独的 FFN。门控分支 SiLU(gate) 已经把这部分非线性承担了。' },
      ],
      sourceRows: [
        { concept: '选择性参数', code: 'self.x_proj(x).split([dt_rank, N, N])', takeaway: '$\\Delta$ (低秩)、$B$、$C$ 全部来自当前输入 $x_t$。“选择性” 三个字的全部实现就是这一行。' },
        { concept: 'Δ 恒正', code: 'F.softplus(self.dt_proj(dt_low))', takeaway: 'softplus 保证 $\\Delta>0$; 再配上 $A = -\\exp(\\text{A\\_log}) < 0$, $\\exp(\\Delta\\cdot A)$ 一定落在 $(0,1)$。' },
        { concept: '递推本体', code: 'h = A_bar[:, t] * h + Bx[:, t]', takeaway: '教学版是顺序 scan; 生产实现换成 work-efficient 并行 scan 加 kernel fusion。' },
        { concept: 'O(1) 解码', code: 'cache["h"] = h', takeaway: '给了 cache 就从上一步的 h 续着跑, 和已经读过多少 token 无关。' },
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
      subtitle: 'MoE 必须均衡负载, 但传统 aux loss 往语言建模目标里掺了一股不相干的梯度。\n读完你能说清 DeepSeek-V3 怎么把 “均衡” 从 loss 里搬出来, 变成一个不收梯度的控制器。',
      tldr: '每个专家配一个偏置 $b_e$:\n- 选 top-k 时: 用 $s+b$。\n- 算门控权重时: 只用 $s$。\n每个训练 step 之后, $b_e \\mathrel{+}= \\gamma\\cdot\\mathrm{sign}(\\text{平均负载} - \\text{专家 } e \\text{ 的负载})$。',
      question: '同样是把热门专家压一压, 为什么改偏置比加 aux loss 对模型质量更友好?',
      evolution: {
        title: '均衡从 loss 里搬出来',
        subtitle: '根问题: top-k 路由有正反馈, 放着不管就会坍缩到少数专家, 专家并行时一张卡拖慢整步',
        steps: [
          { name: '放任路由', pain: '(原点) 被选得多的专家学得更好, 于是更常被选', fix: '什么都不做: 少数专家包揽 token, 其余白占显存; 专家并行时热门那张卡收到的 token 爆容量' },
          { name: '容量上限', year: 2020, pain: '热门专家收到的 token 超出它那张卡能装下的量', fix: '每个专家设容量上限, 超出的 token 跳过 FFN 只走残差; 只处理了溢出, 路由照样偏' },
          { name: 'Aux loss', year: 2021, pain: '丢 token 治标, router 还是往热门专家挤', fix: '加 $E\\cdot\\sum f_e P_e$, 梯度直接压热门专家的路由 logit; 这股梯度和 LM loss 抢方向' },
          { name: '路由偏置', year: 2024, pain: 'aux 权重 α 大了干扰语言建模, 小了又均衡不住', fix: '每个专家一个不收梯度的 bias, 只进 top-k 排序, 每步按负载的符号 ±γ: CV 0.273 → 0.124' },
        ],
      },
      code: 'llm_models/models/moe/deepseekV3.py · llm_models/training/loss.py',
      points: [
        { title: '选人和加权解耦', body: 'top-k 的排序用 $\\mathrm{sigmoid}(\\text{logit}) + b$, 但乘到专家输出上的权重取自不带 b 的 sigmoid 分数再归一化。b 只改变 “谁上场”, 不改变 “上场后信多少”。', key: true },
        { title: '负载不均衡减半, LM loss 没动', body: 'demo 量的是负载变异系数 CV: 各专家负载的标准差除以均值, 0 是完全均衡。\n- 不更新 bias: 最后 20 步平均 CV 0.273, 第 100 步 LM loss 2.979。\n- 更新 bias ($\\gamma$=1e-3): 最后 20 步平均 CV 0.124, 第 100 步 LM loss 2.982。' },
        { title: 'b 是一个只看符号的控制器', body: 'b 不在计算图里 (@torch.no_grad), 更新规则只看符号: 过载就 $-\\gamma$, 欠载就 $+\\gamma$。\n步长恒为 $\\gamma$、与 batch 大小无关, 所以好调。\n- γ 太大: 大过专家分数之间的典型差距就来回震荡。demo 实测 $\\gamma$=1e-2 的均衡效果反而不如 1e-3。\n- γ 太小: 追不上路由器的漂移。' },
        { title: 'aux loss 的代价', body: '$L_{\\text{aux}} = \\alpha\\cdot E\\cdot\\sum f_e\\cdot P_e$ 的梯度直接压低热门专家的路由 logit。$\\alpha$ 大了干扰语言建模, 小了又均衡不住。\n教学代码保留了 MoELMLoss 作为对照。真实的 V3 还留了一个权重极小的序列级 aux loss, 防极端情况。' },
      ],
      links: [
        { from: 'moe · Mixtral vs DeepSeek', to: 'routing_bias', body: 'MoE 路由一章讲 top-k、共享专家、softmax 与 sigmoid 门控的区别; 这里只补 “怎么保持均衡” 这一块。' },
        { from: 'update_routing_bias', to: 'train loop', body: '它必须在 optimizer.step() 之后显式调用。忘了调, b 永远是 0, 整套机制退化成无均衡。' },
        { from: '负载均衡', to: 'llm_train EP 与序列并行', body: '专家并行下不均衡 = 某张卡 all-to-all 收到的 token 撑爆容量, 直接拖慢整个 step。' },
      ],
      sourceRows: [
        { concept: '偏置只进排序', code: 'select_scores = sigmoid_scores + self.routing_bias', takeaway: 'topk 只作用在 select_scores 上。' },
        { concept: '权重不带偏置', code: 'topk_sigmoid = sigmoid_scores.gather(-1, selected_experts)', takeaway: '从原始分数里取被选中专家的值, 再归一到和为 1。' },
        { concept: '更新规则', code: 'deepseekV3.py:update_routing_bias', takeaway: 'bincount 数出每个专家的负载; $\\text{bias} \\mathrel{+}= \\gamma\\cdot\\mathrm{sign}(\\text{mean} - \\text{load})$。它是 buffer, 不走梯度也不进 optimizer, 但要进 state_dict。' },
        { concept: '对照: aux loss', code: 'loss.py:MoELMLoss', takeaway: '$f_e$ (被选频率) × $P_e$ (平均路由概率) 求和; $f_e$ 是计数、不可导, 只有 $P_e$ 这一支有梯度。' },
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
      subtitle: 'DeepSeek-V3.2 的稀疏注意力不规定 mask 形状, 让模型自己学。每个 query 只在 indexer 选出的 top-k 个 key 上做 MLA。\n读完你能说清 indexer 怎么被训出来, 以及为什么不能一上来就稀疏。',
      tldr: 'LightningIndexer 用几个小 head 算 $I[t,s] = \\sum_h \\mathrm{ReLU}(q_h\\cdot k_h/\\sqrt{d})$, 每行只留 top-k, 主注意力只看这 k 个。\ntop-k 不可导, 所以:\n- indexer 靠 $\\mathrm{KL}(\\text{主注意力分布} \\,\\|\\, \\mathrm{softmax}(I))$ 单独训练。\n- 训练分三步: 稠密预训练 → indexer 预热 → 切稀疏。',
      question: 'indexer 自己也要给所有 (t, s) 打分, 还是 $O(T^2)$, 那到底省在哪?',
      evolution: {
        title: '稀疏的形状从人定到学出来',
        subtitle: '根问题: 128K 上下文里每个 query 都和全部 key 算注意力, 算力 $O(T^2)$, 真正有用的 key 只占一小部分',
        steps: [
          { name: '稠密注意力', pain: '(原点) 每个 query 都要和全部 T 个 key 算分数、做 softmax、乘 V', fix: 'MLA 也照样全算: cache 压小了, 算力仍是 $O(T^2)$' },
          { name: '固定稀疏 mask', year: 2019, pain: '真正有用的 key 只占一小部分, 却每个都要算', fix: '人定 mask 形状 (滑窗、跨步), 每个 query 只算少数 key; 关键 token 落在形状外就永远看不到' },
          { name: '按内容选 top-k', pain: '该看哪些 key 取决于内容, 固定形状猜不中', fix: '便宜的 indexer 按内容打分, 只把 top-k 交给 MLA; top-k 不可导, indexer 收不到梯度' },
          { name: 'DSA', year: 2025, pain: 'LM 梯度到不了 indexer, 随机初始化时等于随机丢 key', fix: '主注意力当 teacher, 用 KL 训 indexer; 预热后再稀疏: top-8 召回 0.45 → 0.92' },
        ],
      },
      code: A,
      points: [
        { title: '省的是贵的那部分', body: '- indexer: head 少、维度小、没有 value、不做 softmax, 可以用低精度算。\n- 主注意力 (MLA 升维 + softmax + 乘 V): 从 $O(T^2)$ 降到 $O(T\\cdot k)$。\n长上下文下 k (比如 2048) 远小于 T (比如 128K), 主注意力计算量只剩约 1.6%。总成本由 indexer 这个小常数的 $T^2$ 主导。', key: true },
        { title: 'indexer 是被蒸馏出来的', body: 'top-k 是离散选择, LM loss 的梯度传不到 indexer。demo 直接打印出来: 它对 indexer 的梯度全是 None。\n所以单独加一项对齐 loss:\n- teacher: 各头平均后的主注意力分布。\n- student: $\\mathrm{softmax}(I)$ 去拟合它。\n输入也 detach, 对齐 loss 不动主干表示。' },
        { title: '分三步训, 不能一上来就稀疏', body: '刚初始化的 indexer 等于随机挑 key。直接稀疏就是随机丢 key, 模型会被毁掉。\n所以分三步:\n- 稠密预训练 (60 步): 主注意力看全部位置, 只训主模型。teacher 得先长出结构。\n- indexer 预热 (80 步): 冻结主模型, 注意力仍看全部位置, 只用 KL 训 indexer。\n- 切稀疏 (40 步): 注意力只看 top-k, LM、aux、index loss 一起训。\n前两步都开着 set_dense_warmup。预热这 80 步:\n- KL: 从 0.1144 降到 0.0046。\n- top-8 召回率: 从 0.450 (恰好等于随机乱选的期望) 升到 0.922。\n实验台里 “对齐程度” 滑杆演示的就是这个过程。' },
        { title: '先 mask, 再 top-k', body: '顺序是: 先把因果 mask 之外的分数填 −inf, 再取 top-k。\n反过来会选中未来位置。它们随后被 mask 掉, 有效 key 就不足 k 个, indexer 学到的也是泄漏的信号。\n开头几行可见位置本来不到 k 个, 会选到 −inf。所以最后再和因果 mask 取一次交集, 把它们剔掉。' },
      ],
      links: [
        { from: 'attention · MLA', to: 'MultiHeadLatentSparseAttention', body: 'DSA = LightningIndexer + 一张 top-k mask + 原封不动的 MLA。' },
        { from: 'models-mtp · 掩码实验台', to: 'top-k 稀疏', body: '那里的 top-k 模式给你看 mask 长什么样; 这里给你看 mask 是怎么被学出来的, 以及 k 取多少才够。' },
        { from: 'cache["idx_k"]', to: 'KV cache 生成', body: '解码时历史 token 的 indexer key 也得缓存, 否则没法给旧位置打分。这是 DSA 多付的代价, 它省算力不省 cache。' },
      ],
      sourceRows: [
        { concept: 'indexer 打分', code: 'attention.py:LightningIndexer', takeaway: 'ReLU 之后对 head 求和; 只需要排序, 不需要归一化成概率。' },
        { concept: '输入 detach', code: 'self.indexer(q.detach(), mask=mask, cache=cache)', takeaway: '对齐 loss 只训练 indexer 自己的 w_q / w_k。' },
        { concept: 'top-k → mask', code: '_sparse_mask_from_topk', takeaway: '先把因果 mask 之外的分数填 −inf, 再取 top-k, 每行 scatter 出 k 个 True。顺序不能反: 先 top-k 再 mask 会选中未来位置, 它们随后被 mask 掉, 有效 key 就不足 k 个。开头几行可见位置不到 k 个, 会选到 −inf, 所以最后再与因果 mask 取一次交集。' },
        { concept: 'KL 对齐', code: 'kl = (p * (p.clamp_min(1e-9).log() - log_q)).sum(dim=-1)', takeaway: 'p = attn.mean(dim=1) 是 detach 的 teacher; 两边都限定在同一个 used_mask 集合上。' },
        { concept: 'loss 汇总', code: 'loss.py:MoELMLoss', takeaway: 'total = LM + aux_loss_weight·aux + index_loss_weight·index_loss。' },
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
      subtitle: 'OpenAI 2025 年的开放权重模型没发明新零件, 只是把三个已知零件拼得很讲究。\n读完你能说清交替排布省在哪, 以及为什么滑窗把开头 token 挤出去之后它不会崩。',
      tldr: '- 偶数层: 滑窗, KV 封顶在窗口 W。\n- 奇数层: 全注意力, 兜住长程。\n- sink logit: 拼进 softmax 分母, 算完就丢。head 可以 “谁都不看”, 滑窗也不再依赖开头那几个 token。',
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
        { title: '交替排布: 用一半的 KV 买全部感受野', body: '纯滑窗的 L 层模型感受野只有 $L\\cdot(W-1)+1$, 远处信息每接力一次就被压缩一遍。\n隔一层插一层全注意力, 任何位置一步就能够到全部历史。\n滑窗层的 KV 是滚动缓冲 (上限 W, 与上下文长度无关), 长上下文下 cache 几乎只由全注意力层决定。gpt-oss-20b (24 层, W=128) 在 T=131072 时约省 50%。\ndemo 的 4 层 mini 模型 (W=8) 读完 40 个 token:\n- 各层 cache 长度: [8, 40, 8, 40], 比全是全注意力的 160 省 40%。\n- 贪心生成 100 个 token: 有 cache 比无 cache 快 3.1×, 输出逐 token 一致。', key: true },
        { title: 'softmax 不会弃权', body: 'softmax 只看 logit 的相对差, 一行概率和恒为 1。一个 head 在某位置没什么可看时, 也得输出一堆无关 value 的加权平均。\n没有 sink 的模型就自己找了个垃圾桶: 把多余概率倒在开头几个 token 上。\n这就是 attention sink 现象, 也是首 token 滑出窗口后模型崩掉的原因。' },
        { title: '把 sink 做成参数, 而不是 token', body: '两种补救:\n- StreamingLLM: 永远保留开头 4 个 token (masks.py 里的 sink_tokens 参数)。\n- GPT-OSS: 给每个 head 一个可学 logit 参与 softmax。概率分给它就等于丢掉, 行和 < 1, 不占 KV, 也不依赖任何特定 token 留在窗口里。' },
        { title: 'demo 里 sink logit 几乎没动', body: '在随机 token 上训练, sink logit 只动了约 ±0.01。\n随机数据没有 “该不该看” 的结构, 模型没理由用 sink。\n所以 demo 证明的是梯度通路是通的。真实模型里这些值会明显非零, 而且各 head 不同。' },
      ],
      links: [
        { from: 'models-mtp · SWA', to: '交替排布', body: 'Mistral 全部层都是滑窗; GPT-OSS 与 Gemma 系列改成滑窗与全注意力交替 (Gemma 2 是 1:1, Gemma 3 是 5:1)。' },
        { from: 'build_sliding_window_mask', to: 'use_sink', body: 'sink_tokens 是 mask 层面的 sink (保留开头 token)。use_sink 是 softmax 层面的 sink (一个可学 logit)。' },
        { from: 'moe · MixtralMoE', to: 'GPT-OSS FFN', body: 'FFN 槽位就是标准的 top-k softmax 路由 MoE, 与 Mixtral 同族。' },
      ],
      sourceRows: [
        { concept: '带状 mask', code: 'masks.py:build_sliding_window_mask', takeaway: '($j \\le i$) 且 ($j > i-W$); sink_tokens > 0 时额外保留开头 S 列。' },
        { concept: 'sink 参数', code: 'self.sink = nn.Parameter(torch.zeros(num_heads))', takeaway: '每个 head 一个标量, 初始 0 相当于 “多了一个分数为 0 的空 key”。' },
        { concept: '参与 softmax 后丢弃', code: 'F.softmax(torch.cat([scores, sink], dim=-1), dim=-1)[..., :-1]', takeaway: '最后一列切掉, 每行概率和就小于 1。demo 第 0 行 0.502、第 39 行 0.976, 差额进了 sink。' },
        { concept: '滚动裁剪 cache', code: 'layer_cache["k"][:, :, -self.window_size:]', takeaway: '只有滑窗层裁。裁掉的 K/V 在带状 mask 下本来就是 −inf, 所以输出逐 token 不变。' },
        { concept: 'MoE FFN', code: 'gpt_oss.py:GPTOSSBlock', takeaway: '直接复用 MixtralBlock: “先 top-k 再 softmax” 与 “softmax → top-k → 重归一” 逐项相等。' },
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
      subtitle: '把扩散搬到离散 token 上: 加噪就是随机把 token 换成 [MASK], 去噪就是双向 Transformer 一次预测所有 [MASK]。读完你能说清它和 BERT 差在哪, 以及为什么它用不了 KV cache。',
      tldr: '- 训练: 每条序列抽一个遮盖率 $t$, 每个 token 以概率 $t$ 被遮。只在被遮位置算 CE, 再乘 $1/t$。乘完之后 loss 是 $-\\log p(x)$ 的上界 (ELBO), 压低它就是在抬高似然。\n- 采样: 从全 [MASK] 出发, 每步预测全部 → 留下最有把握的 → 其余重新遮住。剩余 [MASK] 数按线性日程递减。',
      question: 'BERT 也是遮盖再预测, 为什么 BERT 不能拿来生成而 LLaDA 能?',
      evolution: {
        title: '从左到右之外的另一条生成路',
        subtitle: '根问题: 自回归一次只出 1 个 token, 顺序写死从左到右, 只会续写',
        steps: [
          { name: '自回归 LM', pain: '(原点) 生成 N 个 token 要串行前向 N 次, 每个 token 只能看左边', fix: '因果 mask + 预测下一个: 训练简单; 但只会续写, 学了 A→B 不会 B→A' },
          { name: 'BERT (MLM)', year: 2018, pain: '自回归只看左边, 生成顺序写死', fix: '双向注意力, 遮 15% 再还原; 从没见过几乎全遮的输入, 没法从零生成' },
          { name: '掩码扩散', year: 2025, pain: '固定遮 15% 只学会补少量空, 从零生成那一段没练过', fix: 'LLaDA 让遮盖率 t 铺满 (0,1), loss 乘 $1/t$ 成为似然上界; 采样从全 [MASK] 逐步去噪' },
          { name: '低置信度重遮', pain: '同一步定下的 token 互相看不见, 一步定得越多越容易打架', fix: '每步只留最有把握的, 其余遮回去下一步再定: 15 步填空准确率 1.000, 1 步并行 0.887' },
        ],
      },
      code: 'llm_models/models/language_models/llada.py',
      points: [
        { title: '随机遮盖比例 = 一整族去噪任务', body: '- BERT: 固定遮 15%, 只学会了 “补少量空”, 从没见过几乎全是 [MASK] 的输入。\n- LLaDA: $t$ 均匀铺满 $(0,1)$。$t\\approx 1$ 时几乎从零生成, $t\\approx 0$ 时只补一两个词。\n采样正是从 $t=1$ 走到 $t=0$, 每一步遇到的遮盖比例训练时都见过。', key: true },
        { title: '1/t 权重让 loss 成为 $-\\log p(x)$ 的上界', body: '$t$ 小的样本被遮的 token 少, 不加权几乎不贡献 loss。\n乘 $1/t$ 之后, loss 是 $-\\log p(x)$ 的上界 (ELBO)。压低它就是在抬高似然, 模型因此是一个生成模型。\n量纲也对上了: $\\mathbb{E}[\\text{被遮数}/t] = L$, 均匀瞎猜时 loss 期望正好是 $\\ln V$, 和自回归 CE 可以直接比。\n代码里用分层采样铺 $t$, 压住 $1/t$ 带来的方差。' },
        { title: '生成顺序由置信度决定', body: '低置信度重遮: 先定 “显然” 的 token, 它们成为上下文后再定难的。已定稿的 token 置信度记 $+\\infty$, 永不重遮。\ndemo 的填空准确率:\n- 15 步低置信度重遮: 1.000\n- 15 步随机重遮: 0.910\n- 1 步并行: 0.887\n步数越少, 同一步里定下的 token 越多。它们互相看不见对方, 所以 1 步并行最差。' },
        { title: '没有 KV cache, 步数是旋钮', body: '双向注意力下每步任何位置都可能变, 没有 KV cache, 每步整段重算。\n- 步数多: 质量高。\n- 步数少: 速度快。\n续写、倒推、两头填是同一个函数。' },
      ],
      links: [
        { from: 'BERT · MaskedLMLoss', to: 'LLaDALoss', body: '同样只在被遮位置算 CE; 区别是遮盖率随机、乘 $1/t$、再除以总 token 数。' },
        { from: 'diffusion · 连续扩散', to: 'forward_process', body: '连续扩散加的是高斯噪声, 这里的 “噪声” 是 [MASK]。两者都是训练时随机抽噪声强度 $t$, 采样时从 $t=1$ 走回 0。' },
        { from: 'KV cache 生成', to: 'LLaDA.sample', body: '自回归靠因果 mask 才有 KV cache; LLaDA 每步任何位置都可能变, 没法复用。' },
      ],
      sourceRows: [
        { concept: '加噪', code: 'llada.py:forward_process', takeaway: '每条序列一个 $t$, 每个 token 独立以概率 $t$ 换成 [MASK]; batch 内 $t$ 用分层采样。' },
        { concept: 'ELBO loss', code: '(ce * masked / t[:, None]).sum() / labels.numel()', takeaway: '除以总 token 数而不是被遮 token 数, $1/t$ 才有正确的含义。' },
        { concept: '线性日程', code: 'n_masked = torch.round(n_gen * (1 - s / steps)).long()', takeaway: '第 s 步结束后应该剩几个 [MASK]。生成 15 个、分 4 步: 11, 8, 4, 0。' },
        { concept: '重遮', code: 'x.masked_fill(rank < n_masked[:, None], self.mask_id)', takeaway: 'rank = 置信度升序名次; 已定稿位置 conf=+inf, 排在最后, 不会被遮。' },
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
      subtitle: '图像按光栅顺序展平再做 next-token: 既慢 (步数 = token 数), 又把二维结构拆了。\nVAR 让模型先画 1×1 的总体印象, 再画 2×2、4×4。读完你能说清级内并行为什么不会把画面搞乱。',
      tldr: '- 多尺度 VQ: 把图像编码成一座 token 金字塔。每一级量化的是 “前面所有级还没解释掉的残差”。\n- Transformer: 用块状因果 mask 逐级预测。同级互相可见, 另外只能看更粗的级。\nscales=(1,2,4) 时生成整张图只要 3 次前向; 光栅序 4×4 要 16 次。',
      question: '同一尺度内的 token 是一次并行采样出来的。它们看不到彼此的采样结果, 画面为什么不会乱?',
      evolution: {
        title: '自回归的一步从一个 token 变成一整级',
        subtitle: '根问题: 自回归要一维序列, 图像却是二维的; 逐 token 生成, 步数随边长的平方涨',
        steps: [
          { name: '光栅序 next-token', pain: '(原点) 自回归要一维序列, 图像却是二维网格', fix: '按行展平, 逐 token 生成: 步数 = token 数, 上下相邻的 token 在序列里隔了一整行' },
          { name: '同尺度并行', year: 2022, pain: '边长翻倍, 串行步数翻 4 倍', fix: 'MaskGIT 一次前向并行定一批 token; 同批 token 彼此看不见, 不确定性大时细节容易打架' },
          { name: 'next-scale', year: 2024, pain: '同一分辨率上每个 token 都还很不确定, 独立采样就乱', fix: 'VAR 按分辨率由粗到细: 粗级先定构图, 细级只补细节, 级内并行也不乱' },
          { name: '多尺度残差 VQ', pain: '逐级生成要每一级各有一套 token, 普通 VQ 只给一张单尺度 token map', fix: '每一级只量化前面还没解释掉的残差, 各级共用码本; token 总数变多 (8×8 时 85 对 64)' },
        ],
      },
      code: 'llm_models/models/generative/var.py · llm_models/layers/diffusion/vq.py',
      points: [
        { title: '由粗到细符合图像的统计结构', body: '- 低分辨率: 决定构图和明暗。\n- 高分辨率: 只补细节。\n粗的先定死之后, 细的那一级每个 token 的不确定性已经很小。\n并行独立采样带来的不一致也就很小。', key: true },
        { title: '残差量化: 后面的级修正前面的级', body: '第 k 级的输入是 $f - \\sum_{j<k} \\mathrm{upsample}(z_j)$。粗尺度量化得再糙, 误差都留在残差里交给下一级。\n每一级量化的都是同一个特征空间里的残差, 所以各级共用同一个码本。\n代价是 token 总数 $\\sum s^2$ 比单尺度多: 8×8 时 85 对 64。' },
        { title: '块状因果 mask', body: '序列 = [1×1 | 2×2 | 4×4 | …] 拼接。mask 规则: 同一块内全可见, 另外只能看到更早 (更粗) 的块。\n- 训练: 一次前向算完所有级的 loss。\n- 采样: K 次前向, 每次产出一整级。' },
        { title: '本级的 token 只喂给下一级', body: '第 k 级的输入是前 k−1 级的累计重建, 下采样到本级分辨率。\n第 k 级自己的 token 不进本级的输入, 只影响下一级的输入。所以序列里没有 shift-by-one。\ndemo 验证, 改动第 2 级的一个 token:\n- 第 1、2 级的 logits: 纹丝不动 (变化 0.00e+00)。\n- 第 3 级的 logits: 变了 4.95e-01。' },
      ],
      links: [
        { from: 'diffusion · DiT', to: 'VARModel', body: '扩散是在噪声强度上由粗到细, VAR 是在分辨率上由粗到细; 两者都绕开了逐像素串行。' },
        { from: 'MultiScaleVQ', to: 'ImageTokenizer', body: 'tokenizer 必须先单独训 (重建 + vq_loss)。码本随机时 token 和图像内容无关, AR 学到的只是噪声。' },
        { from: 'LLaDA', to: '级内并行采样', body: '两者都是一次前向定多个 token; LLaDA 靠置信度挑, VAR 靠尺度结构分批。' },
      ],
      sourceRows: [
        { concept: 'token 金字塔', code: 'self.num_tokens = sum(s * s for s in scales)', takeaway: 'scales=(1,2,4) 时 L = 21; 序列长度就是各级面积之和。' },
        { concept: '块状因果 mask', code: 'var.py:block_causal_mask', takeaway: '按 token 所属的级编号比较: 级号 ≤ 自己的都可见。' },
        { concept: '逐级采样', code: 'var.py:sample_tokens', takeaway: '每级三步: 累计重建 f_hat → 下采样成下一级的输入特征 → 一次前向 → 整级并行采样。' },
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
  },
}
