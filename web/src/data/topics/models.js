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
      subtitle: '围绕滑动窗口掩码、多 token 预测与定长线性状态三条路线, 拆解现代架构怎样给长上下文注意力减负。',
      tldr: "- SWA (滑动窗口注意力): 无需新增模型类, 仅将下三角因果 mask 替换为带状掩码, 令每个 token 只关注最近 W 个位置, 将每层 KV cache 从随序列长度增长的 $O(T)$ 封顶为 $O(W)$。\n- MTP (多 token 预测): 在主干后串联轻量级联模块并共享 lm_head, 让位置 i 除预测下一 token 外继续预测后续多步, 将每个位置的监督信号从 1 份扩充到 K+1 份, 同时在推理期提供投机解码草稿。\n- 混合线性注意力 (Qwen3-Next): 多数层以固定大小的状态矩阵替代 KV cache, 每 4 层保留 1 层全注意力负责精确长程召回。",
      question: "单层注意力只关注最近 W 个 token 时, 远距离前文信息如何传递到当前位置? 单个位置同时预测后续多个 token, 为什么不会泄漏未来信息?",
      evolution: {
        title: '读的账和写的账分开省',
        subtitle: "根问题: 上下文越长, 每层累积的 K/V 缓存越大; 自回归解码每步却只能生成 1 个 token。",
        steps: [
          { name: '全因果注意力', pain: "(原点) 当前位置可能依赖任意远的前文, 所有历史位置的 K/V 都必须保留", fix: "LLaMA 每层读取全部前文: 长程召回最准, 但每层 KV cache 随上下文长度 T 线性增长" },
          { name: 'SWA', year: 2023, pain: "上下文拉长后, 每一层的 KV cache 都随序列长度 T 同步膨胀", fix: "Mistral 每层只读取最近 W 个位置, 缓存上限固定为 W; 更远的前文信息依靠跨层堆叠接力传递" },
          { name: '混合线性', year: 2025, pain: "滑窗外的早期 K/V 被直接丢弃, 远距离信息只能依赖多层间接传递", fix: "Qwen3-Next 用定长状态矩阵有损压缩全部前文, 每 4 层保留 1 层全注意力补足精确召回" },
          { name: 'MTP', year: 2024, pain: "前文缓存开销降下来后, 解码仍每步只生成 1 个 token, 训练时每个位置也只学习紧邻下一词", fix: "在位置 i 串行级联预测 i+2、i+3 等后续位置: 加密训练监督信号, 对 t+2 的预测还可作为投机解码草稿" },
        ],
      },
      code: 'llm_models/models/language_models/{mistral.py,mtp.py,qwen3_next.py} · llm_models/layers/sparse/linear_attention.py',
      points: [
        { title: '深度换宽度', body: "带状掩码限制单层只读取最近 W 个位置, 超出窗口的前文信息依靠多层堆叠逐层接力传递。在序列长度 T=64、窗口大小 W=8 时, 两种掩码的注意力可见元素数对比明显:\n- 全因果 mask: 每个 token 可读取自身及之前的全部位置, 共 2080 个可见元素。\n- 带状 mask: 每个 token 只读取最近 8 个位置, 可见元素缩减为 484 个。\n虽然单层视野仅为 W, 但第 2 层的输入来自第 1 层的输出, 而第 1 层每个位置已聚合了各自最近 8 个位置的特征; 第 2 层再汇聚最近 8 个位置时, 实际覆盖范围便扩展到 15 个输入位置。\n堆叠 L 层后的理论感受野上界为 $1+L(W-1)$, 在位置 t 处还受限于已有前文总数 t+1。以 Mistral 的 32 层、W=4096 配置计算, 理论感受野上界达 131041 个位置; 不过这只是网络拓扑上存在的信息通路, 并不保证训练后的模型实际会利用如此远的上下文。\ndemo 使用 1 层模型单独验证局部掩码边界: 修改位置 0 的输入 token 后, 仅位置 0 到 7 的输出发生变化, 位置 8 及之后的输出保持严格不变。" },
        { title: '换一张 mask 就是换一个架构', body: "多种主流架构共用完全相同的 QKV 投影、softmax 与乘 V 运算, 架构间的核心差异仅在于注意力 mask 定义的可见范围:\n- 全因果掩码 (下三角): LLaMA 采用该模式, 允许访问全部历史位置。\n- 带状掩码: Mistral 仅开放主对角线附近宽度为 W 的带状区域。\n- 带状加 sink 掩码: GPT-OSS 在滑窗基础上令序列开头的若干列始终可见。\n- top-k 稀疏掩码: DSA 在每行仅保留检索器打分最高的 k 个历史位置。\n因此同配置下 Mistral 与 LLaMA 的可训练参数量完全相同, 在 demo 中均为 3,076,352, 两者的结构区别仅体现在一张不占参数的布尔掩码表上。推理阶段 Mistral 用滚动缓冲区 (rolling buffer) 每层只保留最近 W 组 K/V, 在窗口设为 8 的实验中输入 21 个 token 后, 每层缓存条目依然封顶在 8 个。", key: true },
        { title: '级联保因果', body: "MTP 让单个位置连续预测未来多个 token, 同时通过串行级联严格维持自回归因果性。在 DeepSeek-V3 的设计中, 第 k 级模块在位置 i 按三步推进:\n1. 将上一级输出的 hidden 状态与真实 token $t_{i+k}$ 的 embedding 拼接融合。\n2. 送入单个独立的 Transformer Block 提取特征。\n3. 复用主干输出头预测下一目标 $t_{i+k+1}$。\n例如主干在位置 i 基于前文 $t_{\\le i}$ 预测 $t_{i+1}$, 第 1 级 MTP 模块随即以 teacher forcing 方式补入真实标签 $t_{i+1}$, 在此基础上继续预测 $t_{i+2}$。每加深一级才多引入一个真实历史 token 并向后多预测一步, 因此预测 $t_{i+2}$ 时虽然已知 $t_{i+1}$, 却始终不会提前看到待预测的 $t_{i+2}$ 本身, 完整保留了因果依赖链。\n联合训练总损失为 $\\mathcal{L} = \\mathrm{CE}(\\text{main}) + \\lambda\\cdot\\mathrm{mean}_k\\,\\mathrm{CE}(\\text{mtp}_k)$, 即主干交叉熵加上各级 MTP 平均交叉熵的 $\\lambda$ 倍加权。DeepSeek-V3 将 $\\lambda$ 初值设为 0.3 并在训练后期退火至 0.1 (本代码库固定取 0.3)。demo 中由于主干仅设 2 层, 增加 1 级 MTP 模块会使参数量从 1,666,304 增至 2,503,168; 而在 61 层主干的真实规模下, 同样增加 1 级仅带来约 2% 的额外参数。" },
        { title: '状态替代缓存', body: "Gated DeltaNet 属于线性注意力机制, 用固定大小的 $D_h \\times D_h$ 状态矩阵 S ($D_h$ 为单头维度) 替代逐 token 累积的 K/V 缓存。每读入一个新 token 时, 模型按 delta rule 执行三步状态更新:\n1. 用当前键向量 k 从状态矩阵 S 中检索该方向已存储的旧值。\n2. 从 S 中减去这部分旧值以腾出容量。\n3. 写入当前对应的新值向量 v, 同时配合门控系数 $\\alpha$ 对历史状态施加整体衰减, 输出阶段直接计算 q 与 S 的乘积。\n因此该层的推理缓存就是矩阵 S 本身, 元素总数与已处理的序列长度无关。在 demo 中固定 batch 为 2, 将序列长度 T 从 8 增至 32 时, 两类层的累计缓存元素数变化如下:\n- 全注意力层 (共 2 层): 从 4096 增至 16384, 随序列长度增长 4 倍。\n- DeltaNet 层 (共 6 层): 始终保持 49152 不变。\n这种定长表示的代价是有损压缩: 有限维度的矩阵无法无损容纳任意长度的前文, 对早期特定 token 的精确召回能力弱于标准注意力, 因此 Qwen3-Next 每 4 层保留 1 层全注意力兜底长距离精确检索。" },
        { title: '序列短的时候, 状态反而更大', body: "在短序列场景下, 线性注意力的定长状态矩阵反而比标准 KV cache 占用更多空间。上一条实验在 T=32 时测得 6 层 DeltaNet 占 49152 个元素、高于 2 层注意力的 16384 个; 若折算到单层、单条序列做公平对比:\n- 标准注意力层: 每个 token 需存储 $2 \\cdot H_{kv} \\cdot D_h = 2 \\times 2 \\times 32 = 128$ 个浮点数 (K 与 V 各一份, 共 2 个 KV 头、每头 32 维), 总开销随序列长度 T 线性增长。\n- DeltaNet 层: 固定存储 $H \\cdot D_h \\cdot D_h = 4 \\times 32 \\times 32 = 4096$ 个浮点数 (共 4 个头、每头维护一个 $32 \\times 32$ 矩阵), 不随 T 变化。\n由于 $4096 / 128 = 32$, 只有当序列长度 T 超过 32 时, 单层定长状态的内存占用才会低于 KV cache, 线性注意力的显存优势仅在长上下文区间体现。" },
      ],
      links: [
        { from: 'build_sliding_window_mask', to: 'Mistral.forward', body: "Mistral 与同配置 LLaMA 的结构差异全部集中在这张带状掩码上, 二者参数量完全一致; Mistral.forward 仅在推理步末尾额外将每层缓存裁剪为最近 W 个位置。" },
        { from: 'MTPModule', to: 'lm_head (共享)', body: "每级 MTPModule 仅新增一个将 2D 维压回 D 维的拼接投影层与 1 个 Transformer Block, 词嵌入层和输出头 lm_head 均直接复用主干权重。推理时若直接卸载 MTP 模块, 主干输出的 logits 与同权重 LLaMA 逐元素严格相等, demo 中对此设有显式断言校验。" },
        { from: 'mtp_logits', to: 'speculative decoding', body: "第 1 级 MTP 模块对位置 t+2 的预测可直接充当候选草稿, 交由主干模型在单次前向中并行验证多个 token, 即投机解码 (详见 llm_infer/m07)。本阶段代码未接入投机验证闭环, 开启 KV cache 生成时仅运行主干分支。" },
      ],
      sourceRows: [
        { concept: '带状掩码', code: 'masks.py:build_sliding_window_mask', takeaway: "位置可见条件为 $j \\le i$ 且 $j > i - W$, 即历史 key 位于当前 query 之前且相对距离不超过窗口 W; 当 sink_tokens 大于 0 时, 掩码额外放行序列开头的 S 列。" },
        { concept: 'KV 上限', code: 'mistral.py:kv_cache_entries', takeaway: "滚动缓冲区 rolling buffer 仅保留最近 W 个位置的 K/V, 单层缓存条目数为 $\\min(T, W)$; 在窗口为 8 的配置下读入 21 个 token 后, 每层缓存仍稳定维持在 8 条。" },
        { concept: '级联输入', code: 'mtp.py:MTPModule.forward', takeaway: "前向计算依次执行四步:\n1. 对主干隐状态 h 与下一位置 embedding 分别做独立 RMSNorm: h 历经多层残差累加后幅值远大于初始嵌入, 预先归一化可避免后续线性层消耗额外容量去平衡两路尺度。\n2. 沿特征维拼接两路向量。\n3. 经线性投影压回隐藏层维度 d。\n4. 送入本级独立的 Transformer Block。" },
        { concept: '标签对齐', code: 'loss.py:MTPLoss.compute', takeaway: "第 k 级在位置 i 的监督目标为 labels[i+k], 实现上将标签序列整体左移 k 位; 序列末尾缺失对应未来目标的 k 个位置统一填充为 -100, 由 cross_entropy 在归约时自动忽略。" },
        { concept: 'delta rule', code: 'layers/sparse/linear_attention.py:GatedDeltaNet', takeaway: "状态递推遵循 $S \\leftarrow \\alpha(S - \\beta k(k^\\top S)) + \\beta k v^\\top$: 先从旧状态中擦除 k 方向上的原有投影并乘以遗忘门系数 $\\alpha$ 衰减, 再写入未经衰减的新项 $\\beta k v^\\top$。输入 q 与 k 需预先完成 L2 归一化, 确保算子 $I - \\beta k k^\\top$ 构成收缩映射, 防止多步递推后状态数值发散。" },
        { concept: '混合排布', code: 'qwen3_next.py:layer_types', takeaway: "各层按 [Δ, Δ, Δ, A, ...] 周期交替排布, 每 4 层中的最后 1 层使用全注意力。由于状态递推天然满足因果性且通过遗忘门隐式编码相对顺序, 传入的因果 mask 与 RoPE 在 Δ 层内直接忽略为空操作, 从而让主干前向循环无需按层类型编写分支。" },
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
      subtitle: '自回归生成每步只新增 1 个 token, 历史位置的 K 和 V 在因果掩码下始终不变。围绕这一性质拆解 prefill 与 decode 两段流程, 并对比 MHA/GQA 与 MLA 的缓存形态。',
      tldr: '- 避免重复前向: 首次 prefill 并行算完整段输入并把各层的历史 K/V 存入缓存, 后续 decode 每步只送入最新 1 个 token, 累计送入模型的 token 数从 $O(N^2)$ 降为 $O(N)$。\n- 两种缓存形态: MHA 与 GQA 直接缓存 K、V 张量本体; MLA 只缓存低维压缩向量 c_kv 与共享位置向量 k_rope, 用少量升维计算换取更小的显存占用。',
      question: "为什么自回归解码只需缓存历史位置的 K 和 V, 而无需缓存 Q? 当生成长度触及上下文窗口上限时, 为什么整段旧缓存会同时失效?",
      evolution: {
        title: '把重复计算一刀刀剔掉',
        subtitle: "根问题: 自回归解码按步串行推进, 每生成 1 个新 token 都需执行一次模型前向, 前缀越长单步开销越高。",
        steps: [
          { name: '每步重算', pain: "(原点) 第 t 个 token 的采样依赖前 t-1 个已生成结果, 解码过程无法跨步并行", fix: "每步将完整前缀重新送入模型前向计算: 连续生成 N 个 token 累计处理 $O(N^2)$ 个 token" },
          { name: 'KV cache', pain: "因果掩码下历史位置的 K 和 V 在后续步中恒定不变, 反复重算纯属冗余开销", fix: "驻留各层历史 K/V, 每步仅计算最新 token 的投影并检索缓存, 累计前向 token 数由 $O(N^2)$ 降至 $O(N)$" },
          { name: 'prefill + decode', pain: "初始缓存为空, 若连输入 prompt 也逐个 token 串行送入, 首字延迟将严重退化", fix: "prefill 阶段整段并行计算并填满初始缓存, decode 阶段每步仅送入 1 个新 token; 超出窗口上限时重建缓存并重新 prefill" },
        ],
      },
      code: 'llm_models/utils/generation.py · llm_models/layers/core/attention.py',
      points: [
        { title: '只有 K/V 值得存', body: "自回归解码只需缓存历史 token 的 K 和 V, 无需保存历史位置的 Q。在自注意力层中, 每个位置都会投影出三组分工不同的向量:\n- Q (query): 当前位置发起检索的查询向量。\n- K (key): 供后续位置匹配打分的索引向量。\n- V (value): 匹配命中后参与加权汇总的内容向量。\n解码推进到第 t 步时, 模型只需计算最末尾位置的下一词分布, 即仅用最新 token 的 Q 去检索全部历史与当前位置的 K 和 V:\n- 历史位置的 Q: 仅在生成该位置自身的下一词时使用一次, 后续步骤不再读取。\n- 历史位置的 K 和 V: 因果 mask 决定了每个位置只能访问前文, 新增的后文不会反向改变早期位置的隐藏状态, 因此历史 token 在每一层的 K、V 始终保持不变。\n既然历史 K/V 恒定不变, 只需计算一次并驻留缓存, 后续每步便只需前向传播最新的 1 个 token。本代码库以每层一个 dict 维护缓存, GQA 层直接写入 cache[\"k\"] 与 cache[\"v\"]。在 LLaMA 的 infer 脚本中连续生成 200 个 token, 开启与关闭缓存的输出序列逐位一致, 实测推理速度提升约 5.7 倍。", key: true },
        { title: 'prefill、decode 是两种负载', body: "带缓存的自回归生成天然拆分为 prefill 与 decode 两个硬件特征截然不同的阶段。由于请求刚到达时缓存为空, 模型必须先一次性处理完整段输入 prompt, 随后再逐个生成后续 token:\n- prefill 阶段: 单次前向并行处理整段长度为 T 的 prompt, 同时将各层算出的 K/V 写入缓存。此时矩阵乘法的批与序列维度充足, 属于典型的算力受限负载, 能够充分跑满 GPU 计算单元。\n- decode 阶段: 后续每步仅输入最新的 1 个 token, 却必须从显存完整读取所有层的模型权重与历史 KV cache。此时单步算力需求极小而数据搬运量巨大, 属于显存带宽受限负载, 且各步严格串行依赖。\ngenerate() 通过检查已缓存长度 cache.pos 是否等于 0 来分流: 等于 0 时走整段 prefill, 大于 0 时走单步 decode 并仅切取末尾输入 idx[:, -1:]。阶段 5 的推理调度器正是围绕这两种负载特性构建的, 将算力密集的 prefill 与访存密集的 decode 混合组批以提高硬件整体利用率。" },
        { title: 'MLA 存 latent, 不存 K/V', body: "MLA (Multi-Head Latent Attention, 多头潜在注意力) 不按注意力头分别缓存完整的 K 和 V, 而是改存低维压缩潜在向量。模型先将输入投影为维度为 r 的低维向量 c_kv 并写入缓存, 计算注意力时再通过两个固定矩阵 k_up、v_up 还原出各头的 K 与 V。每层缓存仅包含两部分张量:\n- 低维内容潜在向量 c_kv, 形状为 [S, r]。\n- 施加 RoPE 旋转位置编码后由所有头共享的位置键向量 k_rope, 形状为 [S, rope]。\n通过引入少量线性升维计算, 缓存体积得以大幅缩减:\n- DeepSeek-V3 规模: 取 r=512、rope=64, 每个 token 在每层仅存 576 个数值。\n- 同规模标准 MHA: 128 个头乘每头 128 维且 K、V 各存一份, 共需 32768 个数值, MLA 节省约 98% 的缓存开销。\n在本代码库的小型 demo 配置 (d_model=512, 8 个头, r=64, rope=32) 下, 每个 token 每层的缓存元素数从 MHA 的 1024 个、GQA 的 256 个进一步降至 MLA 的 96 个。" },
        { title: 'c_kv 不能带 RoPE', body: "压缩向量 c_kv 之所以不能施加 RoPE, 是为了保留推理阶段的矩阵吸收 (absorb) 优化条件。由于升维投影矩阵 k_up 在推理期固定不变, 利用矩阵乘法结合律可将其预先合并到 Q 侧投影中, 让查询向量直接在低维空间与 c_kv 计算点积, 免去显式展开 K 张量的开销。\n然而 RoPE 是在位置 m 处施加与坐标相关的旋转变换。一旦对 c_kv 或升维后的 K 施加位置旋转, 该随位置变化的旋转矩阵就会横亘在 Q 与固定矩阵 k_up 之间, 阻断矩阵结合吸收, 迫使解码每步仍需将全部历史位置的潜在向量显式还原为高维 K。\n为此 MLA 采用解耦位置编码设计, 将单头特征拆分为两段:\n- nope 段: 经低维潜在向量 c_kv 压缩存储, 不施加旋转变换, 支持矩阵吸收。\n- rope 段: 经独立小投影生成并施加 RoPE 旋转后存为 k_rope, 由所有注意力头共享同一份以控制缓存增量。\n实验边界: 本章教学实现每步仍显式调用升维投影还原 K 和 V, 未做矩阵吸收折叠; 完整的矩阵吸收路径实现在 llm_infer 模块 m18 的 absorb=True 分支中。" },
        { title: '窗口一满, cache 整段作废', body: "当生成序列超过预设最大长度 max_len (RoPE 预计算表与掩码矩阵的长度上限) 时, 原有 KV cache 必须整段丢弃重建。序列填满 max_len 后每多生成 1 个新 token, 都需要将最早的 1 个 token 移出上下文窗口, 而剩余位置的旧缓存无法直接截取复用的原因有两点:\n- 深层特征已耦合被淘汰的前文: 从第 2 层开始, 每个位置的输入均由上一层汇聚全部前文计算所得; 已滑出窗口的首个 token 早已参与了后续各层 K/V 的生成, 旧缓存数值与按新窗口重算的结果不再一致。\n- 旋转位置坐标发生偏移: 窗口截断后序列位置从 0 重新编号, 而旧 K 张量中已经乘入了平移前的 RoPE 旋转角度, 直接复用会导致相对位置错乱。\n因此当 generate() 检测到 cache.pos + 1 > max_len 时, 会直接实例化新的 KVCache, 截取末尾 max_len 个 token 重新执行整段 prefill; 一旦触及窗口上限, 后续每步生成都会触发该重算分支, 退化为无缓存时的每步全量前向。" },
      ],
      links: [
        { from: 'llm_basic/sample.py', to: 'GenerationMixin.generate', body: "阶段 1 的采样循环每生成 1 个字符都要对整段前缀重新执行完整前向; 本章的 generate() 新增 KV cache 分支消除了历史位置的重复计算, 并在 benchmark 中通过断言验证有缓存与无缓存的输出序列逐 token 严格一致。" },
        { from: 'KVCache', to: 'llm_infer KV 与缓存内存', body: "本章采用面向教学的朴素缓存结构: 每层维护一个 Python dict, 解码时用 torch.cat 沿序列维不断拼接新 K/V, 仅面向单请求离线推理; 阶段 5 将其升级为按块分页管理的 block 表, 支持单张 GPU 高效并发服务多条变长请求。" },
        { from: 'MultiHeadLatentAttention', to: 'DSA', body: "后续 DSA 章节直接复用此处的 MLA 主干及其潜在向量缓存, 仅在缓存字典中额外追加一份轻量检索器的键向量 idx_k, 用于对历史位置快速打分并筛选 top-k 子集。" },
      ],
      sourceRows: [
        { concept: '三个分支', code: 'generation.py:GenerationMixin.generate', takeaway: "输入切分包含三个互斥分支:\n- 未启用缓存: 每步送入截断至 max_len 的完整前缀。\n- 首次步进 (pos 等于 0) 或缓存窗口已满: 新建缓存并对整段有效前缀执行 prefill。\n- 常规增量步进: 进入 decode 模式, 仅送入最末尾的单个 token idx[:, -1:]。" },
        { concept: '窗口满了就作废', code: 'cache.pos + 1 > max_len', takeaway: "一旦满足溢出条件即重新创建 KVCache 并重跑整段 prefill: 滑窗右移淘汰最早的 token 后, 第 2 层及更深层的历史 K/V 因包含该 token 的残差贡献而失效, 同时序列位置需从 0 重新编码 RoPE。触及上限后的每一步均会进入该重算路径。" },
        { concept: '等价性断言', code: 'generation.py:benchmark_kv_cache', takeaway: "在贪心解码 (temperature=0) 下分别运行无缓存与有缓存两轮生成, 并用 assert torch.equal 校验输出张量: 缓存机制只能加速推理而不能改变数学结果, 若断言失败则表明 position_ids 偏移或因果 mask 切片未与 cache.pos 正确对齐。" },
        { concept: 'MLA 的 cache', code: 'cache["c_kv"], cache["k_rope"]', takeaway: "每步仅向缓存追加当前位置的低维潜在向量 c_kv 与共享旋转键 k_rope, 随后由 k_up 与 v_up 对全部 S 个缓存位置实时升维还原出 K 和 V, 展开后的高维张量用完即弃、不驻留缓存。" },
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
      subtitle: '分别从注意力分数上限与位置旋转频率入手: QK-Norm 锁住点积数值范围, YaRN 按频率分段缩放实现长上下文外推。',
      tldr: "- QK-Norm: 在施加 RoPE 之前对每个头的 q、k 分别沿 head_dim 执行 RMSNorm 固定向量模长, 使注意力 logit 仅由方向夹角与可学习增益决定, 数值上界严格受控于 $g^2\\sqrt{d}$。\n- YaRN: 根据 RoPE 各频率分量在训练长度内转过的完整圈数做分段频率缩放, 高频分量保持原样、低频分量除以扩展倍率 $s$、中频区间线性插值过渡, 最后乘上温度补偿系数 mscale 修正长序列下的注意力分布锐度。",
      question: "缩放点积注意力已经除以了 $\\sqrt{d}$, 为什么训练中注意力 logit 仍会异常放大? 上下文长度外推时, 为什么率先失效的是低频旋转维度而非高频维度?",
      evolution: {
        title: '先管住范数, 再管住角度',
        subtitle: "根问题: 注意力分数同时取决于 q、k 的模长乘积与旋转夹角, 任一者越出训练分布都会导致 softmax 失稳。",
        steps: [
          { name: '缩放点积', year: 2017, pain: "(原点) 向量内积的方差随头维度 d 线性增大, 未经缩放时训练初期 softmax 即陷入饱和", fix: "点积除以 $\\sqrt{d}$ 抵消维度带来的方差增长; 但训练过程中 q、k 权重与模长自身的膨胀仍缺乏约束" },
          { name: 'QK-Norm', pain: "q、k 模长同步增大时注意力 logit 按平方级放大, 导致 softmax 退化为单点分布并阻断梯度", fix: "在 RoPE 旋转前对每个头的 q、k 分别做 RMSNorm, 使注意力分数仅取决于方向余弦 $\\cos\\theta$ 与可学习增益" },
          { name: 'PI', year: 2023, pain: "模长受控后分数聚焦于夹角, 但推理位置超出训练长度时低频维度会转入未见过的角度区间", fix: "位置内插将全部 RoPE 频率统一除以倍率 $s$, 把角度拉回训练区间, 但高频分辨率同步受损、相邻 token 难以区分" },
          { name: 'YaRN', year: 2023, pain: "高频维度在训练期内已完整覆盖全部相位角, 与低频一并压缩会无谓牺牲局部位置分辨率", fix: "按训练长度内的旋转圈数分段处理: 高频保持不变、低频除以 $s$、中频线性过渡, 并乘 mscale 补偿注意力温度" },
        ],
      },
      code: `${A} · llm_models/layers/core/position_encoding.py`,
      points: [
        { title: 'logit 随范数二次增长', body: "注意力打分随 q、k 向量模长的乘积呈二次增长, 固定缩放因子 $1/\\sqrt{d}$ 无法抑制训练期模长漂移。在标准缩放点积 $q\\cdot k/\\sqrt{d}$ 中, 分子内积等于两向量模长乘积与夹角余弦之积; 分母 $\\sqrt{d}$ 仅用于在分量独立同分布假设下抵消 d 个维度累加带来的方差放大, 对参数更新引起的模长增长毫无约束——一旦训练中 q 与 k 的模长同步扩大 2 倍, 注意力 logit 就会放大 4 倍。\ndemo 将投影矩阵 W_q 与 W_k 的权重同时放大 10 倍后, 实测最大注意力 logit 从 1.20 跃升至 120.31, 精确放大 100 倍。这类数值膨胀会直接引发两类训练故障:\n- 梯度消失停滞: softmax 输出退化为接近 one-hot 的尖锐分布, 其雅可比矩阵 $\\mathrm{diag}(p) - pp^\\top$ 趋近于 0, 导致对应注意力头无法回传有效梯度而停止更新。\n- 低精度数值溢出: 在 bf16 或 fp16 混合精度训练下, 过大的未归一化 logit 在求指数时极易触发上溢。", key: true },
        { title: 'QK-Norm 只留方向和一个增益', body: "QK-Norm 通过逐头归一化消除投影权重幅值的影响, 使注意力 logit 仅由向量夹角与可学习增益决定。Qwen3、OLMo-2 与 Gemma-3 等现代架构均对每个头的 q、k 在单头维度上分别施加一次 RMSNorm: 归一化将向量模长重置为 $\\sqrt{d}$ 乘以可学习增益参数 g, 当增益为标量时注意力打分简化为 $\\text{logit} = g^2\\cdot\\sqrt{d}\\cdot\\cos\\theta$, 与上游 W_q、W_k 的权重尺度完全解耦。\n在相同实验条件下将 W_q、W_k 放大 10 倍后, 启用 QK-Norm 的最大 logit 稳定保持在 3.44 不变; 模型依然可以通过调整可学习增益 g 控制注意力分布的尖锐程度, 但彻底切断了权重膨胀引发的平方级失控。\n计算顺序上 QK-Norm 必须置于 RoPE 旋转之前, 原因在于正交旋转与逐维缩放不满足交换律:\n- 先归一化再旋转: RoPE 是保长正交旋转变换, 不会改变归一化后已受控的向量模长。\n- 先旋转再乘逐维增益 g: 若在旋转后按维度乘上互不相等的增益系数, 同一旋转平面内的两个配对分量会被非均匀拉伸, 从而破坏 RoPE 依赖的相对位置正交旋转结构。" },
        { title: '低频维度没见过那么大的角度', body: "上下文长度外推时率先失效的是低频旋转维度, 因为它们在超长位置处会转入训练阶段从未覆盖的相位角区间。RoPE 将单头向量的每两个相邻分量配成一个二维平面, 在位置 m 处旋转角度 $m\\cdot\\theta_i$, 不同维度对的基频 $\\theta_i$ 从高频到低频呈指数递减, 犹如一组转速各异的表针:\n- 高频分量 ($\\theta_i$ 大): 在训练窗口长度 L 内已旋转数百整圈, 完整遍历了 $[0, 2\\pi)$ 内的所有方向角, 推理序列继续变长也只是在同一圆周上重复旋转。\n- 低频分量 ($\\theta_i$ 小): 波长超过训练长度 L, 整个训练期内连一整圈都未转完, 模型权重仅适应了其中一小段圆弧; 位置一旦越过 L, 向量就会指向训练分布之外的陌生方向。\n实验中用角度越界倍数衡量外推风险, 定义为扩展长度 $sL$ 处的累计转角与训练期最大可见转角之比: 1.00× 代表未超出训练角度范围; 在 demo 设定的扩展倍率 s=4 下, 若不做任何频率修正, 低频维度的越界倍数直接达到 4.00×。" },
        { title: '三种外推方案, 差在动哪些频率', body: "三种主流长度外推方案的核心思路都是调慢旋转角速度, 差异在于对不同频段施加的缩放比例。在 demo 配置 d_head=64 (共 32 对旋转维度)、训练长度 L=2048、扩展倍率 s=4 下对比三者表现:\n- PI (位置内插): 将全部维度对的频率统一除以 s (等价于将位置坐标除以 4)。所有低频维度的越界倍数均被拉回 1.00×, 但高频分量也被一并压缩, 最高频处相邻 token 的相位差从 1.00 rad 降至 0.25 rad, 削弱了局部相邻位置的区分度。\n- NTK-aware 缩放: 通过放大 RoPE 的底数 base, 令最高频保持不变、最低频恰好放慢 s 倍, 中间维度按幂律平滑过渡。高频局部精度得以保留, 但中低频段压缩不足: 波长超标的 11 个低频维度中仅有最低频一维回到 1.00×, 其余维度最高仍存在 1.56× 的角度越界。\n- YaRN 分段缩放: 直接按训练长度内转过的圈数 r 划分三段——最高的 9 个高频维度 (r>32) 原样保留, 11 个低频维度 (r<1) 全部除以 s 拉回 1.00×, 中间频段线性插值过渡, 同时兼顾局部精度与长程不越界, 最后再乘 mscale 补偿长窗口下的注意力温度衰减。\n实验边界: 本章 demo 仅在几何角度层面验证各方案的相位与分辨率性质, 未执行长文本微调训练, 也未实测语言模型困惑度 (ppl)。" },
      ],
      links: [
        { from: 'position · RoPE', to: 'scaled_inv_freq', body: "位置编码章节已推导 RoPE 通过正交旋转编码相对距离的原理; 本节所有外推方案均只替换预计算的基准频率表 $\\theta_i$ (即代码中的 inv_freq), 下游复数旋转算子本身无需改动任何一行。" },
        { from: 'RMSNorm', to: 'q_norm / k_norm', body: "q_norm 与 k_norm 直接复用 Transformer Block 中的同一个 RMSNorm 类, 区别仅在于归一化维度从隐藏层宽度 d_model 改为单头维度 head_dim, 且所有注意力头共享同一组长度为 head_dim 的增益参数 g。" },
        { from: 'QK-Norm', to: 'llm_train 精度与稳定性', body: "阶段 3 介绍的梯度裁剪、学习率 warmup 与 loss scaling 属于训练优化侧的事后保护手段; QK-Norm 则从网络结构侧直接封住注意力内积随模长平方放大的数值缺口, 从源头消除 logit 失控隐患。" },
      ],
      sourceRows: [
        { concept: 'QK-Norm 开关', code: 'attention.py:GroupedQueryAttention', takeaway: "初始化时按配置开关构造 self.q_norm = RMSNorm(head_dim) if qk_norm else None (k_norm 同理), 并在 forward 中严格置于 _call_rope 之前执行, 保证先完成模长归一化再施加旋转编码。" },
        { concept: 'NTK-aware', code: 'base * factor ** (d / (d - 2))', takeaway: "底数放大指数取 $d/(d-2)$, 恰好令下标最大的最低频分量除以扩展倍率 $s$, 同时令下标为 0 的最高频分量保持不变; 在 d=64、s=4 时修正后的新 base 约为 41,829。" },
        { concept: 'YaRN 分段', code: 'gamma = ((rotations - beta_slow) / (beta_fast - beta_slow)).clamp(0, 1)', takeaway: "用 $r = L\\cdot\\theta / (2\\pi)$ 计算各维度在训练长度 L 内转过的圈数, 混合系数 gamma 在 beta_slow=1 到 beta_fast=32 区间内从 0 线性递增至 1: $\\gamma=1$ 的高频段保持原频率外推, $\\gamma=0$ 的低频段按位置内插除以 $s$。" },
        { concept: '温度补偿', code: '0.1 * math.log(factor) + 1.0', takeaway: "该经验公式计算注意力温度补偿系数 mscale: 上下文窗口拉长后 softmax 分母累加项增多、输出分布趋于平缓, 需适度放大注意力分数予以抵消。在 s=4 时 mscale 等于 1.1386; 本实现将其预乘入 cos/sin 缓存表, q 与 k 旋转时各乘一次, 等效于将注意力 logits 整体乘上 $\\text{mscale}^2 = 1.2965$。" },
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
      subtitle: '注意力机制把全部历史都留在缓存里供随时检索, 而状态空间模型 (SSM) 只能把历史压进固定大小的状态。Mamba 的突破在于让步长与投影参数随当前输入动态变化, 在定长状态里实现按内容取舍。',
      tldr: '- 选择性状态递推: 状态按 $h_t = \\exp(\\Delta_t A)h_{t-1} + \\Delta_t B_t x_t$ 更新, 再经 $y_t = C_t h_t$ 读出输出。其中步长 $\\Delta_t$ 与 $B_t, C_t$ 均由当前输入 $x_t$ 动态投影生成 —— $\\Delta_t$ 变大时冲刷旧状态并写入当前词, 接近 0 时则跳过无关词。\n- 常数级推理状态: 解码每步只需维护固定大小的状态张量 ($O(1)$ 内存), 序列再长也不膨胀; 代价是放弃了固定卷积核训练, 需改用并行扫描 (parallel scan)。',
      question: "S4 等线性时不变状态空间模型因参数与位置无关而能转化为全局卷积高效并行训练, Mamba 为什么要主动放弃这一性质?",
      evolution: {
        title: '在定长状态里学会挑着记',
        subtitle: "根问题: 模型需要将任意长度的历史上下文传递到当前位置, 同时又要避免状态显存与计算量随序列长度无限膨胀。",
        steps: [
          { name: 'RNN', pain: "(原点) 前文序列可以无限延伸, 但推理阶段能分配给历史记忆的显存空间有限", fix: "将历史信息递推压缩进定长隐状态, 单步解码开销固定为 $O(1)$, 但训练只能逐 token 串行且长程梯度易衰减" },
          { name: '注意力', year: 2017, pain: "按时间步串行递推无法发挥 GPU 并行算力, 远距离前文信号经多步中转后严重损耗", fix: "完整保留所有历史位置的表示以支持全序列并行训练; 代价是 KV cache 随长度线性增长、算力达 $O(T^2)$" },
          { name: 'S4 (LTI SSM)', year: 2021, pain: "缓存体积与注意力计算开销随序列增长过快, 处理超长上下文时显存与延迟难以承受", fix: "回归定长线性状态并固定演化参数与时间无关, 可展开为全局卷积并行训练, 但只能按相对距离固定加权" },
          { name: 'Mamba', year: 2023, pain: "历史 token 的留存权重仅取决于距当前位置的远近, 无法按语义内容区分关键信息与噪声", fix: "令步长 $\\Delta$ 与投影矩阵 B、C 随当前输入动态生成, 实现按内容选择性记忆; 放弃卷积形式并改用并行 scan 训练" },
        ],
      },
      code: 'llm_models/layers/sparse/ssm.py · llm_models/models/language_models/mamba.py',
      points: [
        { title: '一个 Δ 同时当写入门和遗忘门', body: "Mamba 利用动态离散化步长 $\\Delta$ 同时充当历史状态的遗忘门与当前输入的写入门。状态空间模型 (SSM) 源自连续时间线性微分方程 $h^\\prime(t) = A h(t) + B x(t)$, 即隐状态 $h$ 的变化率由自身状态乘 $A$ 与外部输入乘 $B$ 共同决定; 处理离散 token 序列时需按采样步长 $\\Delta$ 离散化为单步递推系数:\n- 历史状态保留率: $\\bar A = \\exp(\\Delta\\cdot A)$。实现中将状态矩阵参数化为恒负值 $A = -\\exp(\\text{A\\_log})$, 步长 $\\Delta$ 经 softplus 保证恒正, 因此保留率严格落在开区间 $(0,1)$ 内, 多步递推只会平稳衰减而不会数值发散。\n- 当前输入写入量: 一阶近似下离散写入系数为 $\\bar B \\approx \\Delta\\cdot B$。\n由于同一个步长 $\\Delta$ 同时作用于保留率与写入系数, 在两个极端取值下呈现出类似门控 RNN 的互补行为:\n- $\\Delta$ 显著增大时: 保留率 $\\bar A$ 趋近于 0, 写入系数 $\\bar B$ 同步增强, 等效于冲刷旧记忆并重点写入当前 token。\n- $\\Delta$ 趋近于 0 时: 保留率 $\\bar A$ 趋近于 1, 写入系数 $\\bar B$ 趋近于 0, 等效于原样维持历史状态并跳过当前无关输入。\n由于 $\\Delta$ 直接由当前位置的输入投影算出, 每个 token 都能根据自身内容动态决定是否写入状态记忆。", key: true },
        { title: '参数不随位置变, 就只能按距离加权', body: "演化参数固定的状态空间模型只能按相对距离对历史输入施加静态衰减权重。在 LTI (Linear Time-Invariant, 线性时不变) 设定下, $\\Delta$、$B$、$C$ 均为与时间步无关的常系数, 历史位置 j 对末端隐状态的贡献权重仅由相隔步数 $T-j$ 决定, 与该位置承载的具体语义完全无关。\n这种静态衰减机制无法根据内容做选择性留存, 难以胜任自然语言建模中的核心模式匹配任务, 例如归纳头任务 induction (见到前文「A B ... A」时推断下一词为 B) 与选择性复制任务 selective copying (从夹杂噪声的序列中仅提取指定标记)。在下方实验台的 LTI 模式下, 目标关键 token「7」在最终状态中的贡献占比上限仅为 1/10, 只能与其余 9 个干扰位置均摊记忆容量。\n将 $\\Delta$、$B$、$C$ 改为输入相关的动态函数即可获得按内容筛选的能力, 代价是系统失去了线性时不变性: S4 之所以能够高速训练, 正是因为常系数线性递推可等价展开为固定全局卷积核并行计算; 一旦递推系数随每个位置的输入变化, 全局卷积核便不复存在, 训练前向必须改用沿序列推进的扫描算子 scan (生产级实现采用硬件感知的并行关联 scan 加速)。" },
        { title: '状态大小与序列长度无关', body: "Mamba 在自回归推理时每层只需维护固定大小的状态张量, 显存占用与已处理的序列长度完全无关。单层解码缓存仅由两项定长缓冲组成:\n- 选择性 SSM 隐状态 h: 单条序列形状为 [D_inner, N], 即 d_inner 个特征通道各维护一个 N 维状态向量。\n- 局部一维卷积滑动缓冲: 仅保留最近 d_conv-1 个时间步的通道输入。\n运行 infer 脚本处理完 64 个输入 token 后, 批大小为 2 时的单层缓存固定为卷积缓冲 conv 形状 (2, 256, 3) 与隐状态 h 形状 (2, 256, 16), 即便连续生成 100 万个 token 也不会继续膨胀。\ndemo 连续生成 300 个 token 的基准对比中, 启用状态缓存单步递推耗时约 0.4 s, 每步重算完整前缀则耗时约 7 s, 推理提速约 16 倍, 且两条路径输出的 logits 最大绝对误差仅为 4.77e-07。\n常数级缓存的固有代价是有损记忆压缩: 任意长度的历史上下文都必须压入固定维度的状态张量, 面对需要逐字精确还原早期细节的任务时仍弱于保留完整历史 K/V 的标准注意力。Jamba 与 Qwen3-Next 等混合架构正是针对这一短板折中设计的——在绝大多数层采用定长状态层降本, 仅穿插少数全注意力层保障精确长程检索。" },
      ],
      links: [
        { from: 'Gated DeltaNet', to: 'SelectiveSSM', body: "两类架构均遵循「固定维度状态加输入动态门控」的设计范式, 核心区别在于状态结构与更新法则:\n- Gated DeltaNet: 维护二维键值关联矩阵状态, 依靠 delta rule 先擦除键向量 k 方向上的旧投影再写入新值。\n- Mamba: 维护逐通道独立的对角化 SSM 状态向量, 依靠动态步长 $\\Delta$ 协同调节指数衰减率与输入写入强度。" },
        { from: 'KV cache', to: 'cache["h"]', body: "两类模型共用同一个 GenerationMixin.generate() 解码主循环, 仅在层内缓存读写策略上各自特化:\n- 标准注意力层: 每步向缓存字典追加新的 K/V 切片, 张量长度随生成步数线性递增。\n- Mamba 层: 每步就地覆写定长隐状态 cache[\"h\"] 与短卷积缓冲, 缓存体积全程恒定。" },
        { from: 'MambaBlock', to: 'Block 组装器', body: "单个 MambaBlock 直接由 Pre-RMSNorm、扩维后的 MambaLayer 与残差相加构成, 后方不再串联独立的 SwiGLU FFN 子层: MambaLayer 内部已将输入投影拆出一条并行的 SiLU(gate) 门控支路并与 SSM 主路逐元素相乘, 原先由 FFN 承担的通道扩维与非线性门控已合并在同一层内完成。" },
      ],
      sourceRows: [
        { concept: '选择性参数', code: 'self.x_proj(x).split([dt_rank, N, N])', takeaway: "线性层 x_proj 将当前时间步输入 $x_t$ 一次性映射并切分为三段: 步长 $\\Delta$ 的低秩中间表示 dt_low、输入矩阵 $B_t$ 与输出矩阵 $C_t$。三组演化参数全部由当前 token 动态导出, Mamba「选择性」机制的核心代码即落于此行。" },
        { concept: 'Δ 恒正', code: 'F.softplus(self.dt_proj(dt_low))', takeaway: "经 softplus 激活保证步长 $\\Delta > 0$, 配合恒负对角阵参数化 $A = -\\exp(\\text{A\\_log}) < 0$, 使单步保留率 $\\exp(\\Delta\\cdot A)$ 严格受限于 $(0,1)$ 区间。若去掉 $A$ 前的负号令指数为正, README 实测前向递推至第 57 步即发生数值溢出变为 inf。" },
        { concept: '递推本体', code: 'h = A_bar[:, t] * h + Bx[:, t]', takeaway: "每步将上一时刻隐状态乘以保留率后再叠加当前步写入项; 本教学实现采用直观的 Python 循环顺序 scan, 生产环境则替换为 work-efficient 并行关联 scan 并融合片上内存算子 (kernel fusion) 以避免中间状态反复读写显存。" },
        { concept: 'O(1) 解码', code: 'cache["h"] = h', takeaway: "传入缓存字典时直接读取上一步保存的隐状态 h 继续单步递推, 计算完毕后就地写回最新状态, 单步解码的计算量与显存访问量均与历史序列长度无关。" },
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
      subtitle: '混合专家模型 (MoE) 容易把流量集中在少数热门专家上, 传统辅助损失 (aux loss) 虽然能强行打散负载, 却会干扰语言建模梯度。DeepSeek-V3 用一组不参与反向传播的动态路由偏置解开了这个矛盾。',
      tldr: '- 选人与加权分离: 给每个专家维护一个不收梯度的偏置 $b_e$; 挑选 top-k 上场专家时按 $s + b_e$ 排序, 但汇总专家输出时仍使用不带偏置的原始亲和度 $s$。\n- 步后符号反馈: 每个训练步结束后统计各专家实际接到的 token 数, 超出均值就将 $b_e$ 减去 $\\gamma$, 低于均值就加上 $\\gamma$, 在不污染主损失梯度的前提下动态拉平负载。',
      question: "同样是抑制热门专家过载, 为什么调节路由偏置比引入辅助损失 (aux loss) 更不容易损害语言建模质量?",
      evolution: {
        title: '均衡从 loss 里搬出来',
        subtitle: "根问题: top-k 路由存在马太效应正反馈, 缺乏约束时会坍缩到少数专家, 在专家并行下由过载单卡拖慢整步训练。",
        steps: [
          { name: '放任路由', pain: "(原点) 被高频选中的专家获得更多梯度更新、能力更强, 进而被路由器更频繁地选中", fix: "不加负载均衡约束: 少数专家包揽大部分 token, 其余专家空占显存; 专家并行时热门专家所在设备收到的 token 超出容量" },
          { name: '容量上限', year: 2020, pain: "热门专家分配到的 token 数量超出单卡显存与算力容量上限", fix: "为每个专家设定容量上限, 超额 token 直接跳过 FFN 仅走残差支路; 虽截断了显存溢出, 但路由分布依然偏斜" },
          { name: 'Aux loss', year: 2021, pain: "丢弃超额 token 仅能治标, 路由器仍持续将流量集中投向热门专家", fix: "引入辅助损失 $E\\cdot\\sum f_e P_e$ (选中频率乘平均路由概率), 反向梯度直接压低热门专家的路由 logit, 但会与语言建模主损失产生梯度冲突" },
          { name: '路由偏置', year: 2024, pain: "辅助损失权重 $\\alpha$ 过大会干扰语言建模收敛, 过小则无法抑制路由坍缩", fix: "为每个专家维护不参与梯度的路由偏置, 仅介入 top-k 排序并在步后按负载偏差符号增减 $\\gamma$: 负载 CV 从 0.273 降至 0.124" },
        ],
      },
      code: 'llm_models/models/moe/deepseekV3.py · llm_models/training/loss.py',
      points: [
        { title: '选人和加权解耦', body: "路由偏置 $b$ 只参与 top-k 专家筛选, 不参与专家输出的门控加权。在 DeepSeek 的 MoE 层中, 路由器先为每个 token 对每个专家计算亲和度得分 $s = \\mathrm{sigmoid}(\\text{logit})$, 随后将专家筛选与输出加权拆成独立的两步:\n- 专家筛选: 按 $s + b$ 取前 $k$ 个专家 (top-k), 每个专家在原始亲和度上叠加各自的偏置 $b$。\n- 门控加权: 被选中的 $k$ 个专家分别完成前向计算后加权求和, 门控权重直接取自不带 $b$ 的原始得分 $s$, 并归一化使和为 1。\n因此偏置 $b$ 只改变「选谁上场」: 过载专家的偏置下调后排名靠后, 原本处在边界上的 token 会转交得分相近的其他专家。它不改变「上场后的权重」: 只要某个专家仍被选中, 其门控权重与无偏置时完全一致。整个 aux-loss-free 均衡机制只调整离散排序, 不触碰任何参与反向传播的连续变量。", key: true },
        { title: '负载不均衡减半, LM loss 没动', body: "启用动态偏置更新后, 专家负载不均衡度下降超过一半, 而语言模型损失几乎不变。实验脚本将专家 0 与专家 1 的初始路由权重乘以 4 以模拟路由坍缩初期, 并将 aux loss 权重设为 0; 均衡程度用负载变异系数 CV (各专家接收 token 数的标准差除以均值, 0 代表完全均衡, 数值越大越偏斜) 度量:\n- 不更新偏置: 最后 20 步平均 CV 为 0.273, 第 100 步 LM loss 为 2.979, 第 0 层各专家负载为 [37, 35, 33, 22, 27, 27, 40, 35]。\n- 更新偏置: 当 $\\gamma$ 取 1e-3 时, 最后 20 步平均 CV 降至 0.124, 第 100 步 LM loss 为 2.982, 第 0 层各专家负载变为 [31, 32, 36, 37, 27, 31, 29, 33]。\n两组第 100 步的 LM loss 分别为 2.979 与 2.982, 差距极小, 因为偏置既不产生梯度也不改动门控权重, 仅将边界 token 迁移至得分相邻的专家。\n边界: 实验在单个固定随机 batch 上迭代训练, 损失下降本身仅反映对该批次样本的拟合。" },
        { title: 'b 是一个只看符号的控制器', body: "路由偏置 $b$ 注册为模型的持久化 buffer 而非 Parameter, 更新过程在 @torch.no_grad 下执行, 脱离自动求导计算图。它不累积梯度、也不交给 optimizer 更新, 但会写入 state_dict, 确保从 checkpoint 恢复训练或推理时能复现一致的路由行为。\n偏置采用只看偏差符号的定步长控制器: 每步先用 bincount 统计当前 batch 内各专家接收的 token 负载, 高于均值则减去 $\\gamma$, 低于均值则加上 $\\gamma$, 单步修正幅度恒为 $\\gamma$, 与超载绝对量及 batch 大小均无关, 超参易于标定:\n- 步长过大: 单步修正量直接跨过相邻专家得分的典型间距, 导致 top-k 排序频繁翻转、负载来回震荡; demo 实测 $\\gamma$ 取 1e-2 时的均衡效果反而劣于 1e-3。\n- 步长过小: 偏置调整速度落后于路由器自身参数在训练中的漂移, 无法及时拉平负载。" },
        { title: 'aux loss 的代价', body: "传统负载均衡通过额外引入辅助损失项 (aux loss) 约束路由器, 但其反向梯度会与语言建模主目标相互冲突。Switch Transformer 风格的辅助损失定义为 $L_{\\text{aux}} = \\alpha\\cdot E\\cdot\\sum f_e\\cdot P_e$:\n- 专家总数: $E$ 为参与路由的候选专家个数。\n- 选中频率: $f_e$ 为当前批次内专家 $e$ 被选中的比例。\n- 平均概率: $P_e$ 为路由器分配给专家 $e$ 的归一化平均亲和度。\n热门专家的 $f_e$ 与 $P_e$ 同时偏高, 该项梯度会直接压低其路由 logit。当语言建模本身需要将同一类语义的 token 集中送往特定专精专家时, 辅助损失仍强行将流量拆散; 权重系数 $\\alpha$ 偏大时会牺牲语言建模质量, 偏小时又难以阻止坍缩。\n本库教学代码在 MoELMLoss 中保留了该辅助损失以供对照 (demo 默认将其权重设为 0); DeepSeek-V3 原论文除动态偏置外, 仍保留了一项权重极小的序列级 aux loss 以防范单条序列内的极端偏斜。" },
      ],
      links: [
        { from: 'moe · Mixtral vs DeepSeek', to: 'routing_bias', body: "MoE 路由章节已拆解 top-k 专家筛选、共享专家以及 softmax 与 sigmoid 门控的差异; 本章专门补齐负载均衡机制, 代码改动集中在 routing_bias 这一持久化 buffer 上。" },
        { from: 'update_routing_bias', to: 'train loop', body: "偏置更新不属于模型 forward 的一部分, 必须在每次 optimizer.step() 之后由外部训练循环显式调用一次 update_routing_bias。若遗漏该调用, routing_bias 将始终停留在 0, 整套机制退化为无均衡路由, 负载分布不会得到任何修正。" },
        { from: '负载均衡', to: 'llm_train EP 与序列并行', body: "专家并行 (EP) 将不同专家分散部署在不同设备上。一旦路由失衡, 热门专家所在设备在 all-to-all 通信后接收的 token 超出容量, 其余设备必须同步等待该慢卡算完, 从而拖慢整个训练步。" },
      ],
      sourceRows: [
        { concept: '偏置只进排序', code: 'select_scores = sigmoid_scores + self.routing_bias', takeaway: "叠加偏置后的 select_scores 仅用于排序筛选: topk 只作用在 select_scores 上, 产出被选中专家的索引 selected_experts。" },
        { concept: '权重不带偏置', code: 'topk_sigmoid = sigmoid_scores.gather(-1, selected_experts)', takeaway: "门控权重直接从不带偏置的原始 sigmoid_scores 中按索引 gather 取出, 再除以选中专家的得分总和归一化为 1, 全程不引入 routing_bias。" },
        { concept: '更新规则', code: 'deepseekV3.py:update_routing_bias', takeaway: "先用 bincount 统计各专家当前批次负载, 再按 $\\text{bias} \\mathrel{+}= \\gamma\\cdot\\mathrm{sign}(\\text{mean} - \\text{load})$ 原地更新。该张量注册为持久化 buffer, 不参与梯度回传与 optimizer 更新, 但随 state_dict 保存。" },
        { concept: '对照: aux loss', code: 'loss.py:MoELMLoss', takeaway: "对各专家的选中频率 $f_e$ 与归一化平均路由概率 $P_e$ 求内积后再乘以专家总数 $E$。其中 $f_e$ 来自离散计数、不可求导, 反向梯度仅沿 $P_e$ 支路流回路由器。" },
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
      subtitle: 'MLA 把 KV 缓存压小了, 但超长上下文下主注意力的计算量仍按 $O(T^2)$ 增长。DeepSeek-V3.2 的动态稀疏注意力 (DSA) 先用极轻量的打分器筛出 top-k 个候选位置, 只在这些位置上运行昂贵的主注意力。',
      tldr: '- 先粗筛再精算: 轻量级 LightningIndexer 用少量小维度头快速给历史位置打分, 每个 query 只保留得分最高的 $k$ 个位置送入 MLA 计算。\n- 蒸馏式三段训练: top-k 离散截断会阻断主损失梯度, 因此 indexer 通过拟合主注意力分布的 KL 散度单独学习, 并按「稠密预训练 → 冻结主干预热 indexer → 切换稀疏联合训练」分阶段推进。',
      question: "既然 indexer 仍需遍历全部 (t, s) 位置对打分、渐近复杂度依然是 $O(T^2)$, 整套动态稀疏机制究竟省掉了哪部分开销?",
      evolution: {
        title: '稀疏的形状从人定到学出来',
        subtitle: "根问题: 128K 长上下文下每个 query 都要与全部历史 key 计算注意力, 计算开销按 $O(T^2)$ 增长, 但真正贡献权重的有效 key 仅占极少数。",
        steps: [
          { name: '稠密注意力', pain: "(原点) 每个 query 位置都必须与全部 T 个历史 key 计算内积、执行 softmax 并加权汇总 V", fix: "即便换用 MLA (将 KV cache 压成低维潜向量的注意力), 缓存体积虽大幅缩小, 注意力计算量依然维持 $O(T^2)$" },
          { name: '固定稀疏 mask', year: 2019, pain: "实际占主导权重的历史 key 仅占极少比例, 稠密注意力却对所有位置耗费同等算力", fix: "人工预设固定稀疏掩码 (如局部滑窗或固定跨步), 让每个 query 仅访问少数位置; 但落在预设图案外的关键 token 将被永久漏掉" },
          { name: '按内容选 top-k', pain: "需要关注哪些历史位置取决于当前上下文语义, 固定几何形状无法动态命中", fix: "用轻量 indexer 按内容相关性打分, 仅筛出 top-k 位置交由 MLA 精算; 但离散 top-k 截断了梯度, 主损失无法训练 indexer" },
          { name: 'DSA', year: 2025, pain: "语言模型损失的梯度无法传回 indexer, 随机初始化的筛选器等同于随机丢弃历史 key", fix: "以主注意力分布为 teacher, 通过独立 KL 散度蒸馏训练 indexer; 稠密预热后再切换稀疏模式, top-8 召回率从 0.45 升至 0.92" },
        ],
      },
      code: A,
      points: [
        { title: '省的是贵的那部分', body: "DSA 节省的是昂贵的主注意力计算开销, 而非 indexer 粗筛打分阶段的开销。indexer 确实仍需遍历全部 (t, s) 位置对打分, 渐近复杂度保持 $O(T^2)$, 但两者的单对位置计算常数相差悬殊:\n- 轻量级 indexer: 注意力头数少 (demo 配置仅 2 个头)、每头维度极小、不设 V 投影、也不做 softmax 归一化, 仅执行低维点积与 ReLU 激活, 生产环境中还可进一步采用低精度运算。\n- 昂贵的 MLA 主注意力: 需要先将缓存的低维潜向量升维还原出多头 K 与 V, 再计算全序列注意力权重并加权乘 V; DSA 将主注意力限制在筛出的 $k$ 个候选位置上, 把这一步从 $O(T^2)$ 降至 $O(T\\cdot k)$。\n在超长上下文中 $k$ 远小于 $T$: 例如当 $T$ 取 128K、$k$ 取 2048 时, 主注意力计算量仅剩原来的 $k/T$ (约 1.6%), 整体开销转由常数因子极小的 indexer 主导。\n边界: 本库教学实现仍通过构造稠密 [B, T, S] 布尔掩码并将未选中位置填为 -inf 来模拟稀疏注意力, 并未在硬件层面省掉矩阵乘算力; 工业部署依赖自定义稀疏 kernel 真正跳过未选中的 K/V。", key: true },
        { title: 'indexer 是被蒸馏出来的', body: "由于离散 top-k 操作阻断了反向梯度, 语言模型主损失无法训练 indexer, 必须通过独立的分布蒸馏损失单独训练。top-k 筛选属于阶跃式离散集合操作, 候选分数的微小扰动要么不改变选中集合, 要么引发索引跳变, 不存在平滑导数; demo 脚本实测验证, 若不引入对齐损失, LM loss 对 indexer 参数求导得到的梯度全部为 None。\n为此 DSA 单独构造了一项对齐损失 index_loss, 让 indexer 拟合主注意力的关注分布:\n- teacher 分布: 取主注意力各 head 权重的均值并调用 detach 截断梯度, 防止主干注意力反向迁就筛选器。\n- student 分布: 在同一可见位置集合上对 indexer 打分做 softmax 归一化, 通过最小化两者的 KL 散度促使 student 逼近 teacher。\n同时, 送入 indexer 的隐藏状态输入 q 也会预先 detach, 使 index_loss 的梯度仅更新 indexer 自身的 w_q 与 w_k, 不会反向改动主干表示, 从而让两条损失支路互不干扰。" },
        { title: '分三步训, 不能一上来就稀疏', body: "DSA 不能从随机初始化直接开启稀疏训练, 必须按「稠密预训练、筛选器预热、稀疏联合训练」分三步推进。随机初始化的 indexer 等同于随机抽取候选位置, 若在起步阶段直接切换为稀疏模式, 主注意力每步只能读取随机的 $k$ 个位置, 主干表示会迅速退化; 同时初始阶段的主注意力分布本身近似均匀, 尚无可蒸馏的有效结构。因此 train 脚本拆分为三个阶段:\n1. 稠密预训练 (60 步): 主注意力读取全部因果可见位置, 仅训练主干模型, 让 teacher 先学出清晰的注意力分布结构。\n2. indexer 预热 (80 步): 冻结主干模型参数, 主注意力仍保持稠密计算, 单独用 KL 对齐损失训练 indexer。\n3. 切换稀疏联合训练 (40 步): 主注意力切换为仅读取 top-k 候选位置, 同时联合优化 LM loss、aux loss (MoE 负载均衡项) 与 index_loss。\n前两个阶段均开启 set_dense_warmup, 使 MLA 保持稠密前向、indexer 同步拟合主注意力分布。在 80 步预热期间:\n- KL 散度: 从 0.1144 降至 0.0046。\n- top-8 召回率: 主注意力权重最高的 8 个 key 被 indexer 命中的比例从 0.450 提升至 0.922, 其中初始值 0.450 恰好等于无偏随机抽样的理论期望。\n实验台中的「对齐程度」滑杆即对应这一收敛过程。\n边界: 实验基于单个固定随机 batch 运行, 上述读数用于验证蒸馏与稀疏切换机制有效, 不代表泛化性能。" },
        { title: '先 mask, 再 top-k', body: "构造稀疏掩码时必须先施加因果与 padding 掩码将非法位置的打分填为 -inf, 随后再执行 top-k 筛选。若颠倒顺序先取 top-k 再做掩码过滤, 会同时引发两处缺陷:\n- 有效候选名额缩水: top-k 可能选中高分的未来位置, 这些位置随后被因果 mask 屏蔽, 导致当前行实际保留的历史有效 key 不足 $k$ 个。\n- 未来信息泄漏: 训练阶段对齐分布时会诱导 indexer 学习「哪些未来 token 重要」这类泄漏信号, 而推理时根本看不到未来位置。\n此外, 序列起始处的前几行可见历史总数本身不足 $k$ 个 (例如第 0 行仅有 1 个可见位置), 此时定长 top-k 会将填了 -inf 的位置一并选入凑数, 因此末尾还需与因果 mask 再取一次交集将其剔除。demo 配置 $T=24$、$k=8$ 时, 各 query 实际看到的 key 数依次为 1, 2, 3 直至 8, 此后恒定封顶在 8。" },
      ],
      links: [
        { from: 'attention · MLA', to: 'MultiHeadLatentSparseAttention', body: 'DSA 并没有改动 MLA 注意力本身, 只是在外层套上 LightningIndexer 打分与 top-k 稀疏掩码, 再交给原有的 MLA 执行计算。' },
        { from: 'models-mtp · 掩码实验台', to: 'top-k 稀疏', body: "掩码实验台章节已通过 top-k 模式直观展示了内容自适应稀疏掩码的形态; 本章重点拆解这张动态掩码如何通过蒸馏训练习得, 以及候选预算 $k$ 的容量与召回权衡。" },
        { from: 'cache["idx_k"]', to: 'KV cache 生成', body: "自回归解码阶段, 每生成一个新 token 都需要由 indexer 对全部历史位置重新打分, 因此历史 token 的 indexer key 也必须保存在缓存字典的 idx_k 键下, demo 配置中每 token 每层需额外存储 64 个浮点数。这是 DSA 引入的额外空间代价: 它节省了主注意力的计算量, 但不节省 KV cache。" },
      ],
      sourceRows: [
        { concept: 'indexer 打分', code: 'attention.py:LightningIndexer', takeaway: "每个轻量级 head 分别计算 q 与 k 的缩放点积, 经 ReLU 激活后沿 head 维求和。筛选阶段只需依据相对大小排序而不要求归一化为概率分布, 因此省去了前向 softmax。" },
        { concept: '输入 detach', code: 'self.indexer(q.detach(), mask=mask, cache=cache)', takeaway: "传入 indexer 前先对隐藏状态 q 调用 detach, 确保对齐损失的反向梯度停在 indexer 自身的 w_q 与 w_k 处, 不回流干扰主干网络。" },
        { concept: 'top-k → mask', code: '_sparse_mask_from_topk', takeaway: "按四步生成稀疏布尔掩码:\n1. 先将基础因果 mask 之外的非法位置打分填为 -inf。\n2. 沿最后一维稳定排序取出 top-k 下标。\n3. 将每行选中的 k 个下标通过 scatter_ 置为 True。\n4. 最后与基础因果 mask 再取一次交集, 剔除序列开头因可见位置不足 k 个而被凑数选入的 -inf 位置。\n掩码与筛选顺序不可颠倒: 若先取 top-k 再施加 mask, 未来位置会挤占候选名额并在后续屏蔽后导致有效历史 key 不足 k 个。" },
        { concept: 'KL 对齐', code: 'kl = (p * (p.clamp_min(1e-9).log() - log_q)).sum(dim=-1)', takeaway: "teacher 分布 p 取自 attn.mean(dim=1) (已 detach 的主注意力多头平均权重), student 对数概率 log_q 由 indexer 分数经 log_softmax 得到。两者严格限定在同一 used_mask 集合上归一化, 进入稀疏阶段后仅在选中的 k 个候选位置上计算 KL 散度。" },
        { concept: 'loss 汇总', code: 'loss.py:MoELMLoss', takeaway: "总损失由 LM 损失、aux_loss_weight 乘 aux 损失与 index_loss_weight 乘 index_loss 三项相加汇总。每层的 index_loss 由各自的稀疏注意力模块算好后写入 routing_info 字典, 随路由统计一并交由 MoELMLoss 聚合并求均值。" },
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
      subtitle: 'GPT-OSS 没有引入全新的底层算子, 而是把滑动窗口注意力、全注意力、可学习的 attention sink 与 MoE 路由组合在一起, 在砍掉近一半长上下文缓存的同时保住远距召回与滑窗稳定性。',
      tldr: '- 隔层交替省缓存: 偶数层使用滑动窗口注意力, KV 缓存封顶在窗口长度 $W$; 奇数层保留全因果注意力, 确保任意远的历史信息都能一步直达。\n- 参数化 sink 吸收冗余概率: 每个注意力头额外配一个只参与 softmax 分母的可学习标量 sink, 让注意力头在不需关注上文时有处安放权重, 即使首 token 滑出窗口也不会分布崩溃。',
      question: "滑动窗口在序列变长后会将句首 token 移出缓存, StreamingLLM 指出这会导致注意力分布崩溃, GPT-OSS 为何无需保留句首 token 也能稳定运行?",
      evolution: {
        title: '省 cache, 又不丢远处和开头',
        subtitle: "根问题: 长上下文下若每层都保存完整历史 K/V, 缓存会随序列长度线性膨胀; 若直接裁剪窗口外缓存, 又会损失远距离精确召回并破坏句首注意力锚点。",
        steps: [
          { name: '全注意力', pain: "(原点) 每一层都保留对全部历史位置的直接访问, 各层 KV cache 均随上下文长度 T 线性增长", fix: "LLaMA 全层采用标准全因果注意力: 任意远距离位置均可单层一步直达, 但缓存开销最高" },
          { name: '全滑窗', year: 2023, pain: "长上下文场景下每层缓存长度都达到 T, 显存占用难以承受", fix: "Mistral 全层改为仅读取最近 W 个位置, 单层缓存封顶在 W; 但远距离信息只能跨层逐跳中转, 每中转一层都会发生有损压缩" },
          { name: '交替排布', pain: "远距离特征经多层滑窗中转后逐渐模糊, 难以精确召回早期具体细节", fix: "将滑窗层与全注意力层交替排布 (如 Gemma 2 与 GPT-OSS): 一半层的缓存封顶为 W, 另一半层保留全局一步直达通路" },
          { name: '保留首 token', year: 2023, pain: "模型习惯将富余注意力权重集中分配给句首几个 token, 滑窗一旦将句首挤出缓存就会导致输出崩溃", fix: "StreamingLLM 强制永久保留句首 4 个 token 的 K/V 不被淘汰; 代价是滚动缓存中必须始终为特定真实 token 预留槽位" },
          { name: '可学 sink', year: 2025, pain: "注意力吸收槽与具体句首 token 绑定, 导致这些位置必须常驻 KV cache", fix: "GPT-OSS 为每个注意力头增设一个只参与 softmax 分母的可学习标量 logit, 允许有效注意力行和小于 1 且不占任何 KV 缓存" },
        ],
      },
      code: `llm_models/models/moe/gpt_oss.py · ${A} · llm_models/utils/masks.py`,
      points: [
        { title: '交替排布: 用一半的 KV 买全部感受野', body: "将滑动窗口层与全注意力层隔层交替排布, 既能削减约一半的长上下文缓存开销, 又能保住远距离信息的一步直达通路。滑动窗口注意力 (SWA, sliding window attention) 限制每个位置只读取最近 $W$ 个 token; 虽然第 2 层能通过读取第 1 层已聚合过的邻域特征实现跨层接力, 但 $L$ 层纯滑窗模型的理论感受野上限仅为 $L\\cdot(W-1)+1$, 且远距离信息每经一层中转都会被压缩损耗。\nGPT-OSS 采用隔层交替插入全注意力层的设计, 让任意位置在全注意力层无需中转即可一步访问全部历史。两类层的缓存开销规律截然不同:\n- 滑动窗口层: KV cache 采用定长滚动缓冲, 容量上限为 $W$, 与总上下文长度无关。\n- 全注意力层: 保留完整历史 K/V, 缓存随序列长度 $T$ 线性增长。\n长上下文场景下总缓存几乎只由全注意力层决定: gpt-oss-20b (24 层, $W=128$) 在 $T=131072$ 时, 相比 24 层全注意力结构节省约 50% 的缓存。本库 demo 的 4 层 mini 模型 ($W=8$) 读完 40 个 token 时:\n- 各层缓存长度: 依次为 [8, 40, 8, 40], 合计 96 个条目, 相比全注意力结构的 160 个节省 40%。\n- 贪心生成 100 个 token: 开启 cache 比无 cache 重算快 3.1 倍, 且输出序列逐 token 完全一致。", key: true },
        { title: 'softmax 不会弃权', body: "标准 softmax 强制将每一行注意力得分归一化为总和恒等于 1 的概率分布, 使注意力头在无需读取上文时无法选择「弃权」。当某个注意力头在当前位置并不需要提取任何历史特征时, 标准注意力仍迫使它将总和为 1 的概率质量分完, 输出无关 V 向量的加权平均并混入残差流。\n未引入专门吸收机制的模型会在预训练期间自发寻找固定泄压口: 将富余的注意力概率集中分配给句首的几个 token, 因为句首位置在所有因果前缀中始终可见且坐标恒定, 这一现象即 StreamingLLM 论文观察到的 attention sink (句首 token 吸收大量注意力权重)。\n一旦滑动窗口向后推进并将句首 token 挤出缓存窗口, 原本承接富余概率的锚点突然消失, 这部分权重被迫重新摊派给窗口内的普通上下文 token, 进而引发输出分布崩溃。" },
        { title: 'sink 做成参数, 不做成 token', body: "将 attention sink 做成可学习标量参数而非绑定真实 token, 句首位置就无需长期占用 KV cache。针对滑窗淘汰句首引发的失稳, 共有两种补救路线:\n- 保留句首 token: StreamingLLM 无论窗口如何滑动, 始终保留句首前 4 个真实 token 的 K/V; 本库 masks.py 中的 build_sliding_window_mask 提供 sink_tokens 参数, 令掩码前 $S$ 列永久可见。代价是这几个真实 token 必须常驻缓存, 并在滑动窗口内持续占用槽位。\n- 参数化虚拟列: GPT-OSS 为每个注意力头单独配置一个可学习标量参数 self.sink。计算注意力时将其作为额外一列拼接进得分矩阵共同执行 softmax, 随后直接截除该列; 分配给 sink 的概率不乘任何 V 向量并直接丢弃, 剩余真实历史列的概率之和因此小于 1。当注意力头倾向于「不读取任何上文」时即可将概率质量转入 sink, 既不占用 KV cache, 也不依赖特定真实 token 留在窗口内。" },
        { title: 'demo 里 sink 几乎没动', body: "在固定随机数据上训练 60 步后, 各头的 sink 参数仅从初值 0 微调至约 $\\pm 0.01$ 量级, 但反向梯度全部非零。train 脚本在单个固定随机 batch 上迭代 60 步后, 第 0 层 4 个注意力头的 sink 参数值处于 [0.001, 0.0031, 0.0059, 0.0014] 量级。\n参数幅度变化极小的原因在于随机采样的 token 序列缺乏「当前步是否需要参考上文」的真实语义结构, 模型没有动机主动将概率质量转入 sink。\n边界: 本 demo 仅用于验证参数化 sink 的梯度回传链路畅通, 并未在真实语料上展示 sink 的分流效应; 真实预训练模型中各头的 sink 参数会显著偏离 0 且彼此分化。" },
      ],
      links: [
        { from: 'models-mtp · SWA', to: '交替排布', body: "Mistral 采用全层统一滑动窗口设计 (见滑动窗口章节); GPT-OSS 与 Gemma 系列则改为滑动窗口层与全注意力层交替排布, 只是层间配比不同: GPT-OSS 与 Gemma 2 为 1:1, Gemma 3 为 5:1。" },
        { from: 'build_sliding_window_mask', to: 'use_sink', body: "代码库中的两种 sink 分别作用于不同计算层面:\n- sink_tokens: 掩码层面的配置, 在滑动窗口布尔掩码中保留句首若干个真实 token 永久可见。\n- use_sink: softmax 归一化层面的配置, 为每个注意力头增设一个无对应 K 与 V 的可学习标量 logit。" },
        { from: 'moe · MixtralMoE', to: 'GPT-OSS FFN', body: "GPT-OSS 的 FFN 槽位采用标准稀疏 MoE 结构 (每个 token 从 $E$ 个专家中选出 top-k 个参与计算), 门控路由逻辑与 Mixtral 同族, 因此本库直接复用 MixtralMoE。" },
      ],
      sourceRows: [
        { concept: '带状 mask', code: 'masks.py:build_sliding_window_mask', takeaway: "带状掩码的可见条件为 ($j \\le i$) 且 ($j > i-W$), 即历史 key 位于当前 query 之前且相对距离不超过窗口 $W$; 当 sink_tokens 大于 0 时, 额外保留最左侧前 $S$ 列永久可见。" },
        { concept: 'sink 参数', code: 'self.sink = nn.Parameter(torch.zeros(num_heads))', takeaway: "每个注意力头配置一个初值为 0 的可学习标量。初值设为 0 等效于在候选池末尾拼入一个「得分为 0 的空 key」, 未训练状态下它与其他得分为 0 的真实 key 平分相同的注意力概率。" },
        { concept: '参与 softmax 后丢弃', code: 'F.softmax(torch.cat([scores, sink], dim=-1), dim=-1)[..., :-1]', takeaway: "将 sink 拼接到注意力得分矩阵最后一列共同执行 softmax, 随后截掉最后一列, 剩余真实位置每行的概率和便小于 1。demo 在初始状态下实测:\n- 第 0 行: 仅含 1 个真实可见 key 与 1 个 sink, 真实列概率和为 0.502 (约为 1/2)。\n- 第 39 行: 含 40 个真实可见 key 与 1 个 sink, 真实列概率和为 0.976 (约为 40/41)。\n与 1 之间的差额即被 sink 列吸收。" },
        { concept: '滚动裁剪 cache', code: 'layer_cache["k"][:, :, -self.window_size:]', takeaway: "推理时仅滑动窗口层在每步结束后将 K/V 缓存裁剪至最近 $W$ 个位置, 全注意力层不做裁剪。被裁掉的早期 K/V 在带状 mask 下原本就对应 -inf 屏蔽位置, 因此生成输出逐 token 完全不变; 同时缓存中保存的 K 已预先完成绝对位置 RoPE 旋转, 裁剪不会造成位置错位。" },
        { concept: 'MoE FFN', code: 'gpt_oss.py:GPTOSSBlock', takeaway: "直接继承复用 MixtralBlock。GPT-OSS 官方实现为「先取 top-k 再对这 $k$ 个 logit 做 softmax」, Mixtral 实现为「先做全量 softmax、取 top-k 后再重归一化」; 因全局分母在重归一化时恰好约去, 两种写法逐项严格相等, demo 脚本对此设有断言。" },
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
      subtitle: '把扩散去噪的思想搬到离散文本上: 训练时按随机比例把词遮成 [MASK] 再并行还原, 生成时从全遮盖序列出发、按模型把握大小分批落字, 打破从左到右的固定生成顺序。',
      tldr: '- 全范围随机遮盖训练: 每条样本在 $(0, 1)$ 间抽取遮盖率 $t$ 将词替换为 [MASK], 仅对被遮位置计算交叉熵并乘上权重 $1/t$, 使损失构成真实对数似然的变分上界。\n- 低置信度重遮采样: 从全 [MASK] 序列开始迭代, 每步双向并行预测全部位置, 只锁定把握最高的一批词, 把低置信度位置重新遮回留给下一步结合新上下文再定。',
      question: "BERT 同样通过遮盖部分 token 再并行还原来训练, 为什么 BERT 无法直接用于从零生成文本, 而 LLaDA 可以?",
      evolution: {
        title: '从左到右之外的另一条生成路',
        subtitle: "根问题: 自回归语言模型每步串行生成单个 token, 且生成顺序固定为从左到右, 难以灵活完成双向上下文约束下的逆推与任意位置填空。",
        steps: [
          { name: '自回归 LM', pain: "(原点) 生成 N 个 token 必须串行执行 N 次前向计算, 且每个位置只能读取左侧上文", fix: "施加因果掩码预测下一词: 训练简洁高效, 但仅支持正向续写, 学过「A 后面是 B」却往往答不出「B 前面是 A」" },
          { name: 'BERT (MLM)', year: 2018, pain: "单向因果掩码屏蔽了右侧下文, 生成方向被固定为从左到右", fix: "改用双向注意力随机遮盖 15% 的 token 并还原; 但训练期从未见过接近全遮盖的输入, 无法从零生成完整序列" },
          { name: '掩码扩散', year: 2025, pain: "固定 15% 的低遮盖率仅训练了局部少量补空, 从全遮盖起步的生成轨迹完全空白", fix: "LLaDA 让遮盖率 $t$ 均匀覆盖 $(0, 1)$ 全区间, 损失乘以 $1/t$ 构成负对数似然上界, 采样时从全 [MASK] 序列逐步迭代去噪" },
          { name: '低置信度重遮', pain: "同一去噪步内并行落定的多个 token 彼此不可见具体采样结果, 单步敲定过多极易引发上下文搭配冲突", fix: "每步仅锁定置信度最高的一批位置, 其余低把握位置重新遮回 [MASK] 留待下一步再定: 15 步填空准确率达 1.000, 而 1 步并行为 0.887" },
        ],
      },
      code: 'llm_models/models/language_models/llada.py',
      points: [
        { title: '随机遮盖比例: 一整族去噪任务', body: "LLaDA 为每条训练序列独立抽取随机遮盖率 $t$, 而非像 BERT 那样固定遮盖 15% 的位置, 从而让单一模型覆盖一整族不同难度的去噪任务:\n- BERT (固定遮盖率): 每次固定遮盖 15% 的 token 供双向还原, 模型只学会了已知充足上下文时的「补少量空」, 从未见过几乎全为 [MASK] 的输入, 因此无法从零生成文本。\n- LLaDA (随机遮盖率): 遮盖率 $t$ 在 $(0, 1)$ 区间上均匀取值, 每条序列独立抽取。当 $t$ 接近 1 时几乎全句遮盖, 模型需从零预测整句结构; 当 $t$ 接近 0 时仅遮盖一两个位置, 退化为局部精细补空。单一模型由此同时掌握了从「几乎全空」到「只缺一词」的全部去噪难度。\n推理采样从 $t=1$ (全序列均为 [MASK]) 逐步推进至 $t=0$ (全部填满), 沿途每一步遇到的遮盖比例在训练阶段均已充分见过。demo 采用词表大小为 16 的循环置换链任务验证: 固定一个置换 $\\pi$, 序列严格满足 $x[i+1] = \\pi(x[i])$ 且仅起点随机; 训练每步动态抽取新序列与新遮盖掩码, 因此损失下降确系模型学会了底层置换规律 $\\pi$。", key: true },
        { title: '乘 1/t 之后 loss 才是似然上界', body: "将被遮位置的交叉熵损失乘上权重因子 $1/t$ 并除以序列总长度后, 训练目标才在数学上构成真实文本负对数似然 $-\\log p(x)$ 的上界 (ELBO, 变分下界的负数形式)。若仅对被遮位置直接累加 CE 而不做逆概率加权, 小 $t$ 样本因被遮 token 极少而几乎不贡献梯度, 导致对应采样最后几步的精细去噪欠训; 乘以 $1/t$ 恰好将每条样本的期望权重补齐, 使遮盖越少的样本中单个被遮位置的权重越高。压低这一负对数似然上界即等价于抬高真实文本的生成似然, 模型因此具备完整的生成模型定义, 而非单纯的局部填空器。\n这一加权同时对齐了损失量纲: 由于长度为 $L$ 的序列满足 $\\mathbb{E}[\\text{被遮数}/t] = L$, 未训练模型均匀随机猜测时的损失期望恰好等于 $\\ln V$, 可与自回归模型的 CE 直接对比。demo 实测初始 loss 为 3.014 (对应理论基准 $\\ln V$ 为 2.833); 训练收敛后独立验证集 held-out loss 降至 0.189, 逼近理论下界 0.173。\n为抑制小 $t$ 处 $1/t$ 权重带来的高方差, 代码对单个 batch 内的 $t$ 采用分层采样, 将 $B$ 个采样点等间隔铺满区间。" },
        { title: '生成顺序由置信度决定', body: "LLaDA 的落字顺序并非按位置固定, 而是由每步预测的置信度通过低置信度重遮 (low-confidence remasking) 机制动态决定。每个去噪步中, 模型并行预测所有 [MASK] 位置的候选 token 并提取所选 token 的概率作为置信度, 按三步执行:\n1. 暂填高把握位置: 先在所有 [MASK] 处填入当前预测结果, 先前已定稿位置的置信度记为 $+\\infty$ 以保证永不回退。\n2. 重遮低把握位置: 按去噪日程将置信度排名垫底的一批位置重新遮回 [MASK]。\n3. 结合新邻居重判: 被重遮的位置在下一步即可参考刚刚定稿的高把握邻居上下文再作决定。\ndemo 在给定首 token 续写其余 15 个位置的填空评测中对比了三种采样方式的准确率:\n- 15 步低置信度重遮: 准确率达到 1.000。\n- 15 步随机重遮: 每步随机挑选重遮位置时, 准确率为 0.910。\n- 1 步全并行生成: 单次前向一次敲定全部位置时, 准确率降至 0.887。\n采样步数越少, 同一步内并行锁定的 token 越多; 由于同批预测的位置彼此看不见对方的具体采样结果, 极易出现相邻位置各自合理但组合矛盾的冲突, 因此 1 步并行准确率最低。" },
        { title: '没有 KV cache, 步数是旋钮', body: "由于采用双向注意力, LLaDA 无法使用 KV cache, 每一去噪步都必须对整段序列重跑完整前向计算。自回归模型之所以能复用 KV cache, 是因为单向因果 mask 保证了旧 token 的 K/V 不随后续新增 token 改变; 而 LLaDA 的每个位置同时读取左右两侧上下文, 任意一个 [MASK] 被填上新词都会改变全序列所有位置的 K 与 V。因此去噪步数直接成为推理速度与生成质量之间的调节旋钮:\n- 增加步数: 每步锁定的 token 数更少, 生成一致性与质量更高, 但完整前向次数相应增加。\n- 减少步数: 推理耗时更短, 但单步并行锁定的 token 增多且彼此不可见, 生成质量随之下降。\n双向建模带来的优势在于生成位置完全自由: 正向续写 (给定开头)、逆向倒推 (给定结尾) 与中间填空 (给定中部或两端) 均直接调用同一个 sample() 函数, demo 实测三个方向在 15 步采样下的准确率均为 1.000。" },
      ],
      links: [
        { from: 'BERT · MaskedLMLoss', to: 'LLaDALoss', body: "MaskedLMLoss 与 LLaDALoss 均只在被遮位置计算 CE, 但两者存在三处关键区别:\n- 动态遮盖比例: 遮盖率 $t$ 在 $(0, 1)$ 随机抽取, 而非固定遮盖 15%。\n- 逆遮盖率加权: 每条序列的损失乘以 $1/t$。\n- 全局长度归一: 分母除以全序列总 token 数, 而非被遮 token 数。\n缺少其中任意一处, 损失都不再构成真实负对数似然的变分上界。" },
        { from: 'diffusion · 连续扩散', to: 'forward_process', body: "连续图像扩散向像素或潜变量叠加高斯噪声, 离散掩码扩散则以将 token 替换为吸收态 [MASK] 作为前向加噪过程; 两者的训练与采样框架一致, 均在训练时随机抽取噪声强度 $t$, 并在采样时从 $t=1$ 逐步退火走回 0。" },
        { from: 'KV cache 生成', to: 'LLaDA.sample', body: "自回归模型依赖单向因果 mask 固定历史状态才能启用 KV cache; LLaDA 在双向注意力下每步任意位置的变动都会改变全局键值表示, 无法复用历史缓存, 因此 sample() 每步均需调用一次完整的 forward。" },
      ],
      sourceRows: [
        { concept: '加噪', code: 'llada.py:forward_process', takeaway: "每条序列抽取一个遮盖率 $t$, 序列内各 token 独立以概率 $t$ 替换为 [MASK]。同一 batch 内的 $t$ 采用分层采样: 取一个共享随机起点, 将 $B$ 个采样值等间隔铺满 (eps, 1) 区间。" },
        { concept: 'ELBO loss', code: '(ce * masked / t[:, None]).sum() / labels.numel()', takeaway: "按三步计算变分上界损失:\n1. 用逐位置交叉熵 ce 乘以布尔掩码 masked 清零未遮盖位置。\n2. 除以每条序列各自的遮盖率 t。\n3. 求和后除以总 token 数 labels.numel() 而非被遮 token 数。\n唯有同时满足这三步, 因子 $1/t$ 才具备补齐整句期望损失的数学含义。" },
        { concept: '线性日程', code: 'n_masked = torch.round(n_gen * (1 - s / steps)).long()', takeaway: "线性去噪日程按待生成总数乘以 (1 - s/steps) 后四舍五入取整, 决定第 s 步结束时应剩余的 [MASK] 个数; 例如待生成 15 个位置、分 4 步完成时, 各步结束后剩余的 [MASK] 数依次为 11, 8, 4, 0。" },
        { concept: '重遮', code: 'x.masked_fill(rank < n_masked[:, None], self.mask_id)', takeaway: "rank 为各位置置信度的升序排名 (0 代表把握最低), 排名严格小于 n_masked 的位置会被重新置回 [MASK]; 已定稿位置的置信度预先设为 +inf, 始终排在队尾而永不被重遮。" },
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
      subtitle: '把二维图像按行展平逐个预测 token 既慢又割裂空间邻域。视觉自回归 (VAR) 改为按尺度由粗到细生成: 先定 1×1 的整体色调, 再逐级并行补齐 2×2、4×4 的局部细节。',
      tldr: '- 多尺度残差量化: 图像先编码成由粗到细的 token 金字塔, 每一级只量化前序粗尺度尚未还原的残差特征, 各级共享同一套码本。\n- 逐尺度块状自回归: Transformer 采用块状因果掩码, 同尺度内的位置彼此可见且只依赖更粗尺度; scales=(1, 2, 4) 时只需 3 次前向就能生成整张图, 无需像光栅序那样串行走 16 步。',
      question: "同一尺度内的全部 token 在单次前向中彼此独立地并行采样, 既然看不到同级其他位置的采样结果, 生成画面为什么不会失去空间一致性?",
      evolution: {
        title: '自回归的一步从一个 token 变成一整级',
        subtitle: "根问题: 自回归建模依赖一维因果序列, 但图像本质是二维网格; 若按行展平逐个生成 token, 串行步数会随图像边长按平方增长。",
        steps: [
          { name: '光栅序 next-token', pain: "(原点) 自回归模型要求输入为一维有序序列, 而图像天然呈二维空间网格结构", fix: "将二维网格按光栅序逐行展平并逐个生成 token: 串行步数等于网格总位置数, 且空间上下相邻的位置在序列中被隔开整整一行" },
          { name: '同尺度并行', year: 2022, pain: "分辨率边长每翻一倍, 光栅序串行解码步数就膨胀为原来的 4 倍", fix: "MaskGIT 在单次前向中并行采样一批 token 以压缩步数; 但同批位置彼此不可见采样结果, 分布不确定性较高时局部细节容易相互冲突" },
          { name: 'next-scale', year: 2024, pain: "直接在最终分辨率网格上做并行独立采样时, 各位置条件分布熵值过高, 极易产生空间伪影", fix: "VAR 改为按尺度由粗到细逐级自回归: 低分辨率尺度先锁定全局构图与色调, 高分辨率尺度仅在此约束下补齐局部细节, 级内并行采样仍能保持空间连贯" },
          { name: '多尺度残差 VQ', pain: "逐尺度自回归要求每个分辨率层级都有对应的离散表示, 而常规 VQ 自编码器只输出单尺度离散网格", fix: "多尺度残差 VQ 在每一级仅量化前序粗尺度尚未还原的特征残差, 各级共享同一套码本; 代价是序列总 token 数高于单尺度网格 (8×8 分辨率下为 85 对 64)" },
        ],
      },
      code: 'llm_models/models/generative/var.py · llm_models/layers/diffusion/vq.py',
      points: [
        { title: '由粗到细符合图像的统计结构', body: "VAR 通过由粗到细的尺度分解大幅压低了单步条件分布的不确定性, 使级内并行独立采样不再破坏画面连贯性。在同一步内并行采样多个位置时, 各位置无法获知彼此的具体抽样结果, 若条件分布很宽, 各自合理的局部采样拼合后便容易失调:\n- 低分辨率尺度 (1×1、2×2): 负责敲定全局构图与明暗分布, 虽然先验不确定性最高, 但单级仅含极少数 token。\n- 高分辨率尺度 (4×4 及更细): 粗粒度结构已经作为已知条件固定下来, 每个位置只需在既定色块上补充局部纹理, 候选分布范围已高度收窄。\n凭借粗尺度提供的强全局约束, 细尺度各位置的条件熵显著下降, 独立采样引入的空间冲突也随之消退。本模块训练 demo 中交叉熵损失 CE 收敛在 0.27 附近而非归零 (0): 批次内 8 张图像的 1×1 粗尺度 token 仅出现 3 种取值, 相同的粗 token 后续对应着不同的 2×2 组合, 无条件模型只能拟合一对多的条件概率分布, 该 batch 对应的理论损失下界即为 0.2476。", key: true },
        { title: '残差量化: 后面的级修正前面的级', body: "多尺度残差量化让后续细尺度专门拟合前序粗尺度留下的重建误差, 且全金字塔共享同一套离散码本。tokenizer 先将图像编码为连续特征图 $f$ (demo 中为 4×4 个 16 维向量), 再逐级执行向量量化——即将每个连续向量替换为码本中距离最近的码字并记录其离散编号。\n第 $k$ 级待量化的输入残差为 $f - \\sum_{j<k} \\mathrm{upsample}(z_j)$, 计算分三步完成:\n1. 将前序各级的量化码字 $z_j$ 上采样至特征图 $f$ 的空间尺寸并逐级累加。\n2. 从原始特征图 $f$ 中减去上述累计重建结果, 提取尚未解释的特征残差。\n3. 将残差下采样至第 $k$ 级的目标分辨率, 在码本中查找最近邻码字完成本级量化。\n粗尺度量化留下的高频损失与离散化误差都会保留在残差中, 交由后续更细的尺度逐步修正。由于每一级都在同一连续特征空间内逼近残差, 各级可直接共享同一个码本 (demo 配置为 64 个码字), 解码时将各级码字上采样求和即可还原重建特征。\n代价是序列总 token 数 $\\sum s^2$ 高于单尺度表示: 最高分辨率为 8×8 时共需 $1+4+16+64 = 85$ 个 token, 多于单尺度网格的 64 个。" },
        { title: '块状因果 mask', body: "VAR 用按尺度划分的块状因果掩码替代逐位置下三角掩码, 将自回归单步推进的粒度从单个 token 扩展为一整级分辨率。在三级尺度配置下, 序列由 [1×1 的 1 个 | 2×2 的 4 个 | 4×4 的 16 个] 拼接而成, 共计 21 个位置。掩码按每个位置所属的尺度级号判定可见性, 凡是级号小于或等于当前位置级号的位置均可见: 同一尺度块内部双向全可见, 同时可读取更早 (更粗) 的尺度块, 但严格屏蔽更细的后续尺度块。\n- 训练阶段: 采用 teacher forcing 将完整的真实 token 金字塔一次性送入模型, 单次前向即可并行算出全部 21 个位置的损失。\n- 采样阶段: 共执行 3 次前向传播, 每次产出一整级的预测分布并对级内 $s^2$ 个 token 并行抽样。\n常规自回归使用的是逐 token 的下三角掩码, 而此处变为逐尺度块的下三角掩码, 单步生成的单位也随之从 1 个 token 变为一整级分辨率。" },
        { title: '本级的 token 只喂给下一级', body: "第 $k$ 级各位置的模型输入并非前一个位置的离散 token, 而是前 $k-1$ 级累计重建特征下采样到本级分辨率后的连续向量。第 1 级之前不存在更粗的尺度, 因此使用一个可学习的 start 向量作为起始输入; 从第 2 级起, 均由前序各级码字上采样累加后再下采样至本级尺寸。\n由于第 $k$ 级自身的离散 token 不进入本级输入、只参与构造下一级的输入, 序列内部不存在常规语言模型「输入前一个 token、标签取当前 token」的逐位置 shift-by-one 错位, 因果错位整体发生在相邻两个尺度级之间, 而最细一级 (第 3 级) 的 token 仅作为监督标签、从不充当模型输入。\ndemo 通过单点扰动断言验证了这一因果边界, 当单独改动第 2 级中的 1 个 token 时:\n- 第 1、2 级的 logits: 完全不变 (最大差值为 0.00e+00)。\n- 第 3 级的 logits: 最大变化达 4.95e-01。" },
      ],
      links: [
        { from: 'diffusion · DiT', to: 'VARModel', body: "扩散模型是在噪声强度维度上由粗到细迭代 (先去除大幅度噪声还原轮廓, 再去除细微噪声补充纹理), VAR 则是在空间分辨率维度上由粗到细递进; 两条路线都避开了逐像素串行生成。" },
        { from: 'MultiScaleVQ', to: 'ImageTokenizer', body: "视觉 tokenizer 必须先独立完成第一阶段训练 (联合优化重建损失与 vq_loss, demo 训练 150 步后重建 MSE 从 0.3700 降至 0.0114), 使码本建立起有效的特征对应关系; 若码本仍处于随机未训练状态, 离散 token 与图像内容毫无关联, 后续 Transformer 拟合的只是无意义噪声。" },
        { from: 'LLaDA', to: '级内并行采样', body: "两者都在单次前向中并行确定多个位置的 token, 但分批依据不同:\n- LLaDA: 依据模型预测置信度动态筛选本步保留的位置, 将低置信度位置重新掩码留待后续迭代修正。\n- VAR: 依据二维图像的空间尺度金字塔静态分批, 先确定粗尺度全局结构, 再逐级补充细尺度局部细节。" },
      ],
      sourceRows: [
        { concept: 'token 金字塔', code: 'self.num_tokens = sum(s * s for s in scales)', takeaway: "序列总长度为各尺度网格边长的平方和: 当 scales=(1, 2, 4) 时总长度 $L = 1+4+16 = 21$, 即各级空间网格面积之和。" },
        { concept: '块状因果 mask', code: 'var.py:block_causal_mask', takeaway: "为序列每个位置记录所属尺度级号 $s$, 当且仅当 $s_j \\le s_i$ 时令 mask[i, j] 可见: 同尺度内部及更粗的前序尺度全部可见, 更细的后续尺度严格屏蔽。" },
        { concept: '逐级采样', code: 'var.py:sample_tokens', takeaway: "逐尺度采样每级执行三步:\n1. 将已采样各级的码字上采样累加为当前累计重建特征 f_hat。\n2. 将 f_hat 下采样至本级目标分辨率作为输入特征。\n3. 执行一次前向传播并截取最新一级的输出 logits, 对整级位置并行采样。" },
        { concept: '下一级的输入', code: 'feats.append(q._down(f_hat, s))', takeaway: "送入第 $k$ 级 Transformer 的是「截至上一级的累计重建特征」在当前目标分辨率下的下采样表示, 而非上一级的离散 token 本身, 因此序列内部不存在逐位置的 shift-by-one 错位。" },
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
      subtitle: '对比 CLIP、Whisper、Qwen2-VL 与 Qwen2.5-Omni 四种代表架构, 梳理非文本信号怎样切成连续向量序列, 又通过哪条路径接入语言模型完成理解与生成。',
      tldr: '- 模态切片化: 图像经无重叠卷积切成二维 patch 向量, 音频先转为 mel 声谱再沿时间轴卷积压成帧序列。\n- 三种融合与输出架构:\n  - 双塔对比 (CLIP): 图文各自编码成单个全局向量算相似度, 只做检索打分、不负责生成。\n  - 序列前缀 (Qwen2-VL): 视觉向量直接拼在文本前面送入同一个因果解码器, 不改 LLM 主干但占用上下文窗口。\n  - 交叉注意力与双核协作 (Whisper / Omni): 用 cross-attention 读取音频编码器特征, 或由 Talker 读取 Thinker 的隐状态流式生成语音编码。',
      question: "一张图像进入语言模型要占用多少个 token 位置? 将视觉特征直接拼在文本前缀中, 与让 decoder 通过 cross-attention 按需读取, 在架构改造和计算开销上差在哪里?",
      evolution: {
        title: '把眼睛和耳朵接到同一个 Transformer 上',
        subtitle: "根问题: Transformer 主干只接收向量序列, 而图像像素与音频波形并非离散 token; 切片接入之后, 多模态前缀越长, 注意力计算与缓存开销就越高。",
        steps: [
          { name: '各训各的专用模型', pain: "(原点) 传统视觉模型局限于封闭的 N 类分类标签, 语音识别则拆分为声学、发音对齐与语言模型三级流水线", fix: "各模态分别维护独立的专用网络与封闭标签集; 每扩展新类别都需重新采集标注, 不同模态的表征空间也无法互通" },
          { name: 'CLIP 对比对齐', year: 2021, pain: "人工标注成本高且类别封闭固定, 互联网上数以亿计 (如 4 亿图文对) 的自然语言描述无法直接充当监督信号", fix: "图像塔 ViT 与文本塔各自编码出单个归一化向量, 在 batch 内拉近配对图文并推开其余 $B-1$ 个负样本; 但无解码器, 只能做检索打分而不能生成文本" },
          { name: 'Whisper 交叉注意力', year: 2022, pain: "CLIP 双塔只会计算全局相似度打分; 若要从连续音频生成转写文字, 必须引入自回归 decoder 读取模态编码器特征", fix: "log-mel 声谱经卷积切分为时间帧并由编码器双向编码, 自回归 decoder 在每层通过 cross-attention 将编码器输出作为 K 和 V 读取" },
          { name: 'Qwen2-VL 前缀', year: 2024, pain: "cross-attention 需在预训练 LLM 每层额外插入新模块并改动主干结构, 且难以自然处理多图与文本任意交错的输入", fix: "图像 patch 经 ViT 与 projector 映射为连续 token 直接拼在文本前端, 保持 LLM 结构不变; 因果掩码下后续文本可看图像, 图像看不到后续文本" },
          { name: 'Thinker / Talker', year: 2025, pain: "纯前缀路线只输出离散文本; 若外接独立 TTS 合成语音, 既丢失了隐状态里的语气信息, 又必须等整句文本生成完毕", fix: "采用双核协作: Thinker 融合多模态输入并输出文本, Talker 通过 cross-attention 读取 Thinker 的隐状态并自回归输出语音 codec token" },
        ],
      },
      code: 'llm_models/models/multimodal/{clip,whisper,qwen2_vl,qwen2_5_omni}.py · llm_models/layers/multimodal/multimodal.py',
      points: [
        { title: '进语言模型只有两扇门', body: "非文本信号经各自模态编码器转换为连续向量序列后, 接入自回归语言模型主要有序列前缀拼接与跨模态交叉注意力两条路径:\n- 序列前缀拼接: Qwen2-VL 与 Omni 的 Thinker 采用这一路径——将视觉 token 直接排在文本 token 之前, 由同一个因果 decoder 从头至尾统一处理, 排在后方的文本位置天然能够读取前方的图像特征。优势是预训练 LLM 主干无需改动一行代码, 多张图像可与文本任意交错排列, 标准 KV cache 也能原样复用; 代价是每张图像都会占用 $N_v$ 个上下文窗口位置。\n- 交叉注意力接入: Whisper 与 Omni 的 Talker 采用这一路径——模态编码器的输出不进入 decoder 主序列, 仅充当键和值; decoder 在每层自注意力之后额外增加一层 cross-attention, 用自身查询向量去读取编码器输出。优势是不挤占 decoder 序列长度; 代价是每层均需插入新模块, 必须改动预训练 LLM 的层结构。\nCLIP 则不走上述任一生成路径: 它将图像与文本各自压缩为单个全局向量比较余弦相似度, 不含 decoder, 因此只能做跨模态检索打分而无法自回归生成。", key: true },
        { title: '图切 patch, 声谱切帧', body: "连续的二维图像与一维音频在进入编码器前, 分别通过空间方块卷积与时频声谱一维卷积切分为向量序列:\n- 图像分块投影: PatchEmbed2D 采用卷积核与步幅均等于 patch 尺寸的无重叠二维卷积将图像切为局部方块, 并将每块线性映射为向量, 在数学上与「切块、展平再过 Linear」完全等价。输入分辨率为 224×224 时, 配 16×16 的 patch 可得 14×14 共 196 个视觉 token (不计 CLIP 视觉塔额外拼接的全局汇总 CLS token); 原版 Qwen2-VL 的 patch 边长为 14, 同一张图像会切出 16×16 共 256 个 token。\n- 音频声谱切帧: 原始波形先经采样率对齐、分帧与短时傅里叶变换 (STFT)、mel 滤波及对数压缩, 转为形状 [80, T_mel] 的 mel 声谱 (每 10 ms 一帧、共 80 个频率通道)。Whisper 再用两层 Conv1d 沿时间轴卷积 (第二层设 stride=2 将帧数减半) 生成帧级 token: 30 秒音频对应的 3000 帧声谱被压为 1500 帧 (即每秒 50 个); 本模块 demo 输入形状为 (2, 80, 100) 的 mel 张量, 编码器输出形状为 (2, 50, 128)。\n- 二维声谱复用: Omni 的音频分支将 mel 声谱视作通道数为 1 的二维图像, 与视觉分支直接共用 PatchEmbed2D 切分二维 patch。\n边界: 本仓库未内置波形级音频前处理算子, Whisper 实验中的 mel 张量由 randn 直接构造。" },
        { title: '一张图占多少上下文', body: "在前缀拼接架构下, 视觉 token 进入 decoder 后与文本 token 承担完全相同的自注意力计算与 KV cache 存储开销:\n- 分辨率平方膨胀: 图像边长翻倍时 patch 数量增至 4 倍——分辨率从 224 升至 448 时, 16×16 切分下的视觉 token 数从 196 增至 784。\n- 上下文开销叠加: 将 $N$ 个视觉 token 拼作前缀后总序列长度由 $T$ 增至 $T+N$, 单层注意力分数矩阵按 $(T+N)^2$ 增长, KV cache 容量按 $T+N$ 线性增长。\n特征压缩可显著缩减序列长度: 本仓库的 Qwen2-VL 教学实现采用 Perceiver Resampler, 以固定数量的可学习 latent 向量作为 query, 通过 cross-attention 从 $N$ 个视觉 patch 中聚合信息, 输出 token 数仅由 latent 个数决定 (demo 将 16 个 patch 压为 8 个)。代价是重采样抹去了每个输出向量与原图局部网格的一一对应关系, 细小文字与局部细节容易受损; 因此原版 Qwen2-VL 并未采用 Resampler, 而是将空间相邻的 2×2 个 patch 拼接合并后再过 MLP, 在压缩长度的同时保留二维网格拓扑。" },
        { title: '前缀是单向的: 文本看图, 图看不到文本', body: "Qwen2-VL 对整条拼接序列统一施加因果掩码, 排在前方的视觉前缀对后续文本单向可见, 而视觉位置自身不受后续文本影响。infer 脚本通过三条扰动断言验证了这一单向可见性与掩码边界:\n- 替换输入图像: 文本位置的输出 logits 变化 0.0034, 证实后续文本能够读取视觉前缀。\n- 替换后续文本: 视觉位置的输出变化严格为 0.0e+00, 证实因果掩码阻断了前缀向后读取文本。\n- 修改填充位置: 改动文本段 pad 位置的 token 时其余有效位置变化严格为 0.0e+00; 代码在文本 padding mask 前端自动补齐了 $N_v$ 个 1, 保证视觉前缀段始终有效。\n训练损失仅在文本位置上计算, 视觉位置的监督标签设为 -100 以令 cross_entropy 自动跳过——图像在此充当条件上下文而非预测目标。\n位置编码采用 M-RoPE: 将每个注意力头的维度切分为 (时间, 行, 列) 三段并分别按对应坐标轴旋转。静态图像 patch 的三轴坐标为 (0, 行, 列), 后续文本三轴取相同标量值; 因此在纯文本输入下它与标准 1-D RoPE 逐位相等 (demo 实测数值差为 0.0e+00), 而对调视觉 patch 的行列坐标会使输出产生 5.3e-05 的变化, 在未训练的 4×4 小网格上幅值虽小但严格非零。" },
        { title: '三种训练目标, 都靠一条下界验收', body: "在固定随机样本的过拟合验收中, 只有训练损失跌破「忽略跨模态输入时的理论信息熵下界」, 才能证明模型确实读取了另一模态。由于各训练 demo 均在固定随机批次上迭代, 模型完全可能仅靠记忆文本背出标签; 因此需先算出不读取跨模态信号所能达到的最佳理论下界, 实测损失显著低于该下界才算通路打通:\n- CLIP 对比损失: 将批内 $B$ 张图像与 $B$ 条文本拼成 $B\\times B$ 相似度矩阵, 主对角线为配对正样本。打分矩阵定义为 $\\mathrm{logits} = \\exp(t)\\cdot I\\,T^\\top$, 其中 $I$ 与 $T$ 为归一化后的图文特征, 总损失取图像检索文本与文本检索图像双向 CE 的均值。可学习因子 $\\exp(t)$ 对应温度倒数 $1/\\tau$, 初值设为 $1/0.07 \\approx 14.3$, 上限截断在 100 以防止模型靠无限放大 logits 压低损失。demo 实测 ($B=8$): 初始损失为 2.180 (接近 8 个候选中均匀盲猜的 $\\ln 8$), 训练 30 步后降至 0.0004, 对角线检索 top-1 准确率从 0.12 升至 1.00。\n- 条件自回归损失: Qwen2-VL 与 Whisper 优化文本位置的 next-token CE。train 脚本特意构造了 2 条文本完全相同、所配图像 (或音频) 不同且目标标签不同的样本: 若模型不看跨模态输入, 面对相同文本最多在两个答案间等概率二选一, 损失理论下界为 $\\ln 2 \\approx 0.693$。实测 Qwen2-VL 训练 100 步后降至 0.198, Whisper 训练 150 步后降至 0.041, 均跌破下界, 证明文本位置成功读取了跨模态特征。\n- Omni 双核联合损失: 总损失为文本 CE 加上 0.5 倍语音 CE。训练后 text_loss 从 6.269 降至 0.006, audio_loss 从 5.294 降至 0.025; 脚本同样令两条样本的 codec 输入完全相同而标签不同, 不读取 Thinker 隐状态时 audio_loss 的理论下界同样为 $\\ln 2$, 实测损失远低于该阈值。" },
        { title: 'Talker 只读隐状态, 梯度会回流', body: "Omni 的语音生成核 Talker 直接通过交叉注意力读取主模型 Thinker 的顶层连续隐状态, 而非读取其生成的离散文本。Omni 内部由两个自回归模型协作:\n- Thinker 主干: 接收全部多模态前缀并生成文本响应的主语言模型。\n- Talker 语音核: 参数规模小得多的轻量自回归解码器, 负责逐帧输出语音 codec token (由外部音频编解码器还原为波形的离散语音编码)。\n由于离散文本只是连续隐状态向词表投影后的有损结果, 情绪与语气等副语言信息仍保留在 Thinker 最后一层归一化后的隐状态 thinker_hidden 中, 由 Talker 直接读取可避免级联信息损失。\ninfer 脚本在包含图像、视频、音频各 4 个 token 加 10 个文本 token 的输入配置下 (text_logits 形状 [2, 4+4+4+10, 500], audio_logits 形状 [2, 12, 200]), 验证了三条数据流断言:\n- 替换输入图像: 语音 logits 变化 4.0e-05, 证实视觉信息经 Thinker 传至 Talker 的前向通路畅通 (未训练模型权重初始化标准差为 0.02, 因此扰动幅值较小)。\n- 替换 codec 输入: 文本 logits 变化严格为 0.0e+00, 证实前向计算单向传递、Thinker 不依赖 Talker。\n- 修改未来 codec: Talker 历史位置输出变化严格为 0.0e+00, 证实 Talker 自身同样满足严格因果性。\n两个模型各自维护独立词表与输出头: Thinker 的 lm_head 与词嵌入共享权重, Talker 则不共享。由于 thinker_hidden 传入 Talker 前未调用 detach, 联合训练时 audio_loss 的梯度会一路反向回流至 Thinker; 若需隔离梯度干扰, 只需在传入前调用 .detach()。" },
        { title: '实验边界: 通路可学, 不等于会看会听', body: "本章各模块的实验仅用于验证跨模态张量接入、掩码边界与梯度回传链路正确可学, 并不代表模型具备真实的视觉或听觉理解能力:\n- 合成随机数据: 训练数据固定为单个随机 batch——图像与 Whisper 的 mel 声谱均由 randn 直接构造而非从真实波形提取, 文本也为随机整数 token; 损失降至 0 仅说明模型记住了该固定批次, 不代表具备看图或听声的泛化能力。\n- 玩具级模型规模: CLIP 双塔各 2 层、隐藏维 $d=64$、batch 大小仅 8 (原版批大小达 32768, 对比学习在海量负样本下才具区分难度); Qwen2-VL 的 LLM 与 ViT 各 2 层、$d=128$、图像尺寸 56×56; Omni Tiny 总参数量仅 504,352。\n- 省略的工程组件: 未包含波形级音频前处理 (重采样、分帧、mel 滤波)、将 codec token 还原为波形的编解码器与声码器、Qwen2-VL 与 Omni 的自回归 generate() 及 KV cache, 以及原版的原生动态分辨率和 TMRoPE (按真实物理时间戳对齐音视频位置)。\n- 权重绑定与缩放约定:\n  - Qwen2-VL 与 Thinker: 绑定 lm_head 权重且输入 embedding 乘以 $\\sqrt{D}$。\n  - Whisper 解码器: 绑定输出头权重但不乘 $\\sqrt{D}$。\n  - CLIP 文本塔: 不乘 $\\sqrt{D}$。\n  - Omni 的 Talker: 输入 embedding 乘以 $\\sqrt{D}$ 但不绑定输出头权重。\n各模型须采用 $\\mathcal{N}(0, 0.02^2)$ 初始化才能保证初始损失接近均匀分布下的 $\\ln V$; 若改用默认标准正态分布 $\\mathcal{N}(0, 1)$ 初始化并配合权重绑定, 初始损失会激增至 Qwen2-VL 130.29、Whisper 82.30、Omni text_loss 124.15。" },
      ],
      links: [
        { from: 'Qwen2VLDecoder', to: 'LLaMA', body: "Qwen2-VL 的 decoder 主干即标准 LLaMA 式的 GQA + SwiGLU + RMSNorm + RoPE 结构, 仅额外增加了直接接收外部拼接连续向量序列的 inputs_embeds 入口与三维 M-RoPE 位置编码开关; Omni 的 Thinker 直接复用该类。" },
        { from: 'PreLNCrossBlock', to: 'Transformer (2017) 的 decoder', body: "Whisper 的 decoder 与 Omni 的 Talker 均沿用 2017 年原版 Transformer 解码器的三子层结构: 先过因果 self-attention, 再经 cross-attention 读取编码器特征, 最后过 FFN。Whisper 与原版编码器-解码器 Transformer 完全同构, 仅将 encoder 前端输入由离散 token 替换为 log-mel 声谱。" },
        { from: '视觉前缀', to: 'KV cache 生成', body: "序列前缀路线天然兼容标准 KV cache: 图像 token 只需在首轮 prefill 阶段编码并缓存一次, 后续每步 decode 仅对新增文本 token 做增量计算。边界在于本仓库的 Qwen2VLDecoder 未接入 GenerationMixin, 仅支持整段序列前向。" },
        { from: 'PatchEmbed2D', to: 'diffusion · DiT', body: "多模态理解模型与扩散生成模型在「先将二维网格切分为 patch 序列、再送入 Transformer block」这一步上完全同构; 区别仅在于 DiT 切分的是经 VAE 压缩后的潜空间 latent, 而此处直接切分原始图像像素与二维声谱。" },
      ],
      sourceRows: [
        { concept: '切 patch', code: 'multimodal.py:PatchEmbed2D', takeaway: "采用卷积核与步幅均等于 patch 边长 $p$ 的无重叠 Conv2d, 在数学上与「切块、展平再过 Linear」严格等价; 输出 token 数为 $N = (H/p)\\cdot(W/p)$, 三通道图像 ($C=3$) 与单通道 mel 声谱 ($C=1$) 直接共用同一个类。" },
        { concept: '对比 logits', code: 'clip.py:CLIPModel.forward', takeaway: "forward 仅返回 L2 归一化后的图文特征向量与截断上限为 100 的温度缩放因子 exp(t).clamp(max=100); 形状为 [B, B] 的相似度矩阵在损失函数内计算, 主对角线为配对正样本, 批内其余 $B-1$ 个样本自动充当负样本。" },
        { concept: '拼前缀', code: 'embeds = torch.cat([vision_embeds, embeds], dim=1)', takeaway: "按视觉在前、文本在后的顺序沿序列维拼接, 并在文本 padding mask 前端自动补齐 $N_v$ 个 1; 视觉 token 不乘 $\\sqrt{D}$ 缩放因子, 其数值尺度由 projector 末尾的 LayerNorm 约束。" },
        { concept: 'M-RoPE 位置', code: 'qwen2_vl.py:build_mrope_position_ids', takeaway: "返回形状为 [3, B, H·W + T] 的时间、行、列三轴位置坐标: 图像 patch 坐标取 (0, 行, 列), 后续文本段三轴赋予相同递增坐标且从 max(H, W) 起步, 紧接在视觉段之后。" },
        { concept: '声谱切帧', code: 'whisper.py:WhisperAudioEncoder.forward', takeaway: "经两层 Conv1d 提取局部时频特征 (第二层设 stride=2 将时间帧数减半) 后叠加正弦位置编码; 音频编码器不施加掩码, 整段声谱各时间帧之间双向全可见。" },
        { concept: '唯一通道', code: 'layer(x, context=encoder_hidden, self_mask=self_mask, context_mask=None)', takeaway: "encoder_hidden 是音频特征进入 decoder 的唯一通道, 在每层 cross-attention 中充当 K 和 V; 由于整段音频在转写解码前已完整就绪, cross-attention 不设掩码, 每个文本位置均可检索全部音频帧。" },
        { concept: 'Thinker → Talker', code: 'qwen2_5_omni.py:Qwen2_5_OmniModel.forward', takeaway: "Thinker 以 return_hidden=True 同时返回 (logits, hidden); 顶层隐状态 hidden 不做 detach 直接作为 Talker 的交叉注意力 context, 若两者隐藏维不同则先经 thinker_to_talker 线性层投影对齐。" },
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
      "body": "前文推导的跨层感受野上界只是网络拓扑层面的理论上限, 代表前文信号在结构上最远能从多少步之外传导至当前位置, 并不代表训练后的模型真正具备同等跨度的信息利用能力。\n这一精确上界仅在以下三个前提同时成立时适用:\n- 纯因果滑窗堆叠: 所有层均采用滑动窗口注意力, 中间未穿插全注意力层。\n- 窗口宽度 W 包含当前位置自身: 窗口覆盖的 W 个位置中有 1 个是当前 token 本身, 因此每叠加 1 层只能向前多延伸 $W-1$ 个位置。\n- 不含空洞采样 (dilation) 与全局通路: 空洞滑窗通过间隔采样扩大单层跨度, 而全注意力层或常驻可见的 sink 列则允许信号单步跨越任意距离; 一旦引入这两类结构, 可达范围便不再服从该线性公式。\n满足上述前提时, 位置 t 可触及的输入位置数为 $\\min(t+1, 1 + L \\times (W - 1))$, 其中对 $t+1$ 取最小值是因为位置 t 及其前文总共只有 $t+1$ 个输入 token:\n- 小窗口示例 (W=8, L=4): 实际可达位置数为 $1 + 4 \\times 7 = 29$, 而非 $4 \\times 8 = 32$; 直接按 $L \\times W$ 估算会把每层的当前位置自身重复计入, 仅在 W 远大于 1 时才接近真实值。\n- Mistral 配置 (32 层, W=4096): 理论可达位置数为 $1 + 32 \\times 4095 = 131041$。\n这一拓扑上界既不决定模型允许接收的最大序列长度, 也不能保证长距离大海捞针检索必定成功, 两项指标均需通过独立评测验证。",
      "source": "llm_models/models/language_models/mistral.py:receptive_field"
    }
  ],
  "models-gptoss": [
    {
      "title": "一步够得到, 不等于用得上",
      "body": "结构上存在单步直达的连接通路, 并不代表模型在实际推理中一定能有效利用远距离历史信息。如第 1 条所述, GPT-OSS 每隔一层插入一层全注意力, 因此从拓扑结构看, 任意输出位置在全注意力层均无需逐层接力即可一步访问全部历史。\n但「结构可达」仅保证信息传递路径未被掩码切断, 模型最终能否召回并利用远处的特定 token, 还取决于三项无法从静态结构图直接判定的因素:\n- 训练数据分布: 训练阶段是否充分见过对应跨度的长程依赖样本。\n- 注意力权重分配: softmax 归一化后是否真正将有效概率分给了目标历史位置。\n- 中间层表示留存: 中间各层的隐藏状态演化是否将该条早期信息保留了下来。\n边界: 下方滑动窗口实验台计算的仅是拓扑可达范围 (即某一输出位置在连接关系上能触达哪些输入位置), 并非长上下文任务的实际准确率, 后者必须在真实评测任务上单独检验。",
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
topic.pages["models-mtp"].related = [{"route": "attention", "label": "复习注意力掩码", "reason": "注意力掩码决定各位置之间的可见关系; 本节聚焦多 token 预测机制, 掩码滑杆交互实验详见注意力章节。"}]
export default topic

const qkYarn = topic.pages['models-qknorm-yarn']
const mixed = topic.pages['models-mtp']
const supplement = (page, id, title, points, labNames) => ({ ...page, id, title, points, labNames, evolution: null, links: [], related: [], widgets: [] })
topic.pages.attention = { sections: [
  { ...supplement(qkYarn, 'qk-norm', 'QK-Norm · 控制注意力数值', qkYarn.points.slice(0, 2), ['ModelQkNormLab']), tldr: "- 数值失控根源: 注意力打分 logits 由查询向量 $q$ 与键向量 $k$ 的点积决定; 当 $q$、$k$ 的范数同步增大 2 倍时, 点积打分会按平方放大至 4 倍。demo 将投影权重 W_q、W_k 乘以 10 后, 最大打分从 1.20 跃升至 120.31, 导致 softmax 饱和退化为只集中于单个位置的尖峰分布。\n- QK-Norm 机制: 在执行 RoPE 旋转之前, 对每个注意力头的 $q$、$k$ 分别做一次 RMSNorm 将向量范数拉回固定尺度, 使注意力打分仅由向量夹角与可学习增益决定; 在同样将权重乘以 10 的条件下, 最大打分稳定保持在 3.44。" },
  { ...supplement(mixed, 'swa', '滑窗与跨层感受野', [mixed.points[0], mixed.points[1], mixed.points[5]], []), tldr: "- 跨层感受野扩张: 上方掩码实验台展示的是单层局部视野——滑动窗口内每个 token 仅直接读取最近 $W$ 个位置; 堆叠多层后历史信息可沿层间逐层向前接力, $L$ 层网络最多可覆盖 $1+L(W-1)$ 个历史位置 ($W=8$、$L=4$ 时理论上限为 29 个)。\n- 结构上界与能力分离: 该公式给出的只是信息存在传播通路的拓扑上限, 训练后的模型能否真正从对应距离有效召回历史信息, 仍需通过长距离检索任务独立验证。" },
] }
topic.pages.position = { sections: [
  { ...supplement(qkYarn, 'yarn', 'YaRN · 位置频率怎样外推', qkYarn.points.slice(2), ['ModelRopeScaleLab']), tldr: "- 外推失效根源: RoPE 将每对通道维度视作以特定角频率随位置旋转的指针, 不同维度对的转速呈几何级数递减; 转速最慢的低频维度对在预训练长度内连一整圈都未转完, 位置一旦超出训练长度便会转入模型从未见过的角度区间, 导致长上下文外推崩溃。\n- YaRN 分频缩放: 依据各维度对在预训练长度内累计转过的圈数分段处理:\n  - 高频维度: 保持原始转速不变, 完整保留局部相对位置分辨率。\n  - 低频维度: 将转速除以长度扩展倍率 $s$, 把远距旋转角拉回训练见过的弧段。\n  - 中频维度: 在高低频之间做线性平滑过渡。\n- 实验台对比指标: 将线性位置内插 (PI)、NTK-aware 与 YaRN 并列对照两项核心性质:\n  - 局部邻域区分度: 相邻 token 之间的旋转角差是否依然可分。\n  - 长距离外推稳定性: 扩展窗口后远端位置是否转出了预训练见过的角度范围。" },
] }
topic.pages['models-mamba'].sections = [
  { ...supplement(mixed, 'linear-hybrid', '进阶 · DeltaNet 与混合线性架构', mixed.points.slice(3, 5), ['ModelLinearAttnLab']), tldr: "- Gated DeltaNet 与 Mamba 均以固定大小的状态张量替代随长度增长的 KV cache, 但状态更新机制各有侧重:\n  - Gated DeltaNet 维护键值关联矩阵状态, 每步按 delta rule 先擦除键向量 k 方向上的旧投影, 再写入当前值向量 v。\n  - Mamba 采用选择性对角状态空间模型 (SSM), 依靠输入导出的动态步长 $\\Delta$ 同步调节历史衰减与新信息写入。\n- 两类定长状态均对长历史做有损压缩, 对远距离特定 token 的精确召回弱于标准注意力, 因此 Qwen3-Next 采用每 4 层保留 1 层全注意力的混合排布兜底长程检索。" },
]
mixed.title = 'MTP 多 token 预测 · 一次前向, K+1 份监督'
mixed.subtitle = '普通语言模型在每个位置只学紧邻的下一个词。多 token 预测 (MTP) 在主干后串联轻量模块同时预测未来多步, 既加密了训练监督信号, 又能顺手为推理提供投机解码草稿。'
mixed.tldr = '- 串行级联不断因果链: 第 $k$ 级模块在位置 $i$ 接收上一级隐状态与真实词 $t_{i+k}$ 的嵌入, 经单个 Block 预测 $t_{i+k+1}$ —— 每往后推一步才多看一个真实词, 始终看不到待预测目标本身。\n- 共享词头与联合损失: 各级模块复用主干的输入嵌入与 lm_head, 总损失为主干交叉熵加上权重为 $\\lambda$ 的各级 MTP 平均交叉熵 (DeepSeek-V3 取 0.3 后退火至 0.1)。推理时可将 MTP 分支裁掉还原纯主干, 或拿它生成的 $t+2$ 候选交给主模型做投机验证。'
mixed.points = [mixed.points[2]]
mixed.evolution = null
mixed.related = [{ route: 'attention', label: '滑窗与跨层感受野', reason: "滑动窗口注意力 (每层仅读取最近 $W$ 个位置) 与跨层感受野上界已并入注意力一章的补充小节, 与掩码实验台集中对照讲解。" }, { route: 'models-mamba', label: '状态空间与混合线性架构', reason: "用固定大小状态替代线性增长 KV cache 的两条路线 (Mamba 与 DeltaNet) 在同一章并列对照, 拆解各自节省的缓存开销与有损压缩取舍。" }, { route: 'infer-tree-speculation', label: '草稿怎样被验证', reason: "MTP 分支产出的 $t+2$ 预测仅是候选草稿; 解码阶段如何将多步草稿交由目标主模型在单次前向中并行验证与筛选, 详见该投机解码章节。" }]
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
mixed.evolution = { title: '从一份监督到 K+1 份', subtitle: "根问题: 标准自回归在每个位置仅提供下一个 token 这 1 份训练监督信号, 解码阶段每步前向也只能串行产出 1 个 token。", steps: [
  { name: '下一 token 预测', pain: "(原点) 标准语言模型在位置 $i$ 仅监督紧邻的下一个 token $t_{i+1}$, 每个位置只有 1 份训练信号", fix: "主干网络算出隐状态 hidden 后经输出头 lm_head 预测紧邻下一词, 训练监督与自回归生成均以单步 1 个 token 为单位" },
  { name: '并行多头 MTP', year: 2024, pain: "单步监督信号稀疏, hidden 只需拟合紧邻下一词而无需兼顾后续多步规划, 解码时每步前向也只产出 1 个 token", fix: "Gloeckle 等人在主干后并联 $K$ 个独立输出头, 由第 $k$ 个头直接预测 $t+k$; 但各头看不到中间步骤的 token, 预测头之间缺乏序贯因果链" },
  { name: '级联 MTP', year: 2024, pain: "各输出头彼此独立地跨越中间步直接预测 $t+k$, 在未知中间 token 取值时强行预测导致序贯因果链断裂", fix: "DeepSeek-V3 改为串行级联: 第 $k$ 级拼入真实 token $t_{i+k}$ 的 embedding 并经单个 Block 后再预测下一步, 输出头直接复用主干 lm_head" },
  { name: '当投机草稿', pain: "训练期可拼入真实未来 token 做 teacher forcing, 但推理解码期并无未来真值可用, 若只跑主干仍每步只生成 1 个 token", fix: "推理时主干先预测出 $t+1$ 并送入第 1 级 MTP 产出 $t+2$ 候选草稿; 草稿须交目标主模型并行验证筛选而非无条件全收 (详见 llm_infer m07)" },
] }
for (const route of ['attention', 'position']) Object.assign(topic.pages[route], { points: [], links: [], sourceRows: [], widgets: [] })

canonicalizeRelated(topic)

qk.source = ['llm_models/layers/core/attention.py:GroupedQueryAttention']
qk.question = "缩放点积注意力已经除以了 $\\sqrt{d}$ 来平衡维度方差, 为什么在训练中还要额外通过 QK-Norm 约束查询向量 $q$ 与键向量 $k$ 的长度?"
yarn.question = "将上下文窗口拉长至 4 倍时, 应当如何分别检验「相邻 token 仍能保持局部区分度」与「低频维度未转出预训练角度范围」?"
swa.question = "单层滑窗仅读取最近 $W$ 个位置, 堆叠多层后结构感受野已能延伸至很远的历史位置, 这是否意味着模型真能可靠记住远距离内容?"
delta.question = "将随序列长度增长的 KV cache 替换为固定大小的状态矩阵, 在显存与计算上省下了什么开销, 在历史信息保留上又丢掉了什么?"
mixed.question = "级联 MTP 的第 $k$ 级模块在位置 $i$ 已经拼入了后续真实 token $t_{i+k}$ 的表示, 为什么这不算提前看到了当前要预测的目标 $t_{i+k+1}$?"
qk.related = [{ route: 'position', anchor: 'yarn', label: '继续看 YaRN: 把 RoPE 拉到更长' }]
yarn.related = [{ route: 'attention', anchor: 'qk-norm', label: '对照 QK-Norm: 管住注意力分数的大小' }]
