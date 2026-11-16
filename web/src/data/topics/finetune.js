import { canonicalizeRelated } from '../../utils/consolidate.js'
import { chapterLabels } from '../chapterLabels.js'
import { mergePage } from '../../utils/consolidate.js'
// 阶段 4 · llm_finetune 的全部章节 (15 章)。
// 每章都写全 StageTopic 要渲染的字段。
// source 只写真实存在的 def / class —— npm run check:sources 会逐条校验。
// 页面里的数字全部来自 llm_finetune/run_finetune/*/README.md, 可以用对应的 run 命令复现。
const topic = {
  stage: 'finetune',
  chapters: [
    { route: 'finetune-qlora', label: 'QLoRA · NF4 量化基座', hint: '基座压成 4 bit 存, 算的时候再解开; 389 KiB 变 59 KiB' },
    { route: 'finetune-dora', label: 'DoRA 幅度/方向分解', hint: '低秩只管转方向, 每行长度单独学一个标量' },
    { route: 'finetune-merge', label: '模型合并 · 任务向量相加', hint: '不重训, 把两份微调改动加回基座: EM 1.000' },
    { route: 'finetune-simpo-orpo', label: 'SimPO · ORPO 无参考偏好优化', hint: '拿掉 ref: 前向次数减半, 显存减半' },
    { route: 'finetune-rlaif', label: 'RLAIF · 按宪法自动造偏好', hint: 'judge 批评 + 改写造偏好对; judge 没查的学不到' },
    { route: 'finetune-prm', label: 'PRM · 逐步打分', hint: '给每一步打分, 挑解的准确率从 0.488 升到 0.648' },
    { route: 'finetune-ppo', label: 'PPO · critic 买到了什么', hint: 'critic 给每个 token 估分; 玩具规模下它约等于常数' },
    { route: 'finetune-grpo-variants', label: 'GRPO 进阶: clip · DAPO · Dr.GRPO · GSPO', hint: '一批采样更新多次, 才需要比率和 clip' },
    { route: 'finetune-onpolicy-distill', label: 'on-policy 蒸馏', hint: '学生自己写答案, 老师逐 token 打分' },
    { route: 'finetune-rlvr', label: '看题的可验证奖励', hint: '奖励不看题, RL 就只学会一个常数' },
  ],
  pages: {
    // ───────────────────────── SFT ─────────────────────────
    'finetune-sft': {
      title: 'SFT 数据与 loss · 只教模型回答, 不教模型提问',
      subtitle: '读完你能说清 -100 该盖到哪一格, 以及为什么多盖一格, 模型就永远学不会开口说第一个词。',
      tldr: 'SFT (supervised fine-tuning, 监督微调) 就是拿 (问题, 回答) 对继续做预训练那种「预测下一个 token」的训练, 只是 loss 只算在回答上。做法是把问题位置的 labels 写成 -100, 交叉熵会跳过这些位置。\n坑在边界: labels 在数据侧已经整体左移了一格, 所以 prompt 长 $P$ 时只能盖前 $P-1$ 格。盖满 $P$ 格会把第一个回复 token 一起盖掉, 模型从此学不到「看完问题第一个词说什么」。\n同配置实测: 正确写法在留出集 (训练时没见过的题) 上整条答对的比例 (exact-match, 下面简称 EM) 是 1.000, 多盖一格是 0.000。',
      question: 'labels 已经错位之后, mask 的边界到底是 $P$ 还是 $P-1$? 多盖一格为什么从 loss 曲线上看不出来?',
      evolution: {
        title: '从接龙到只答不问',
        subtitle: '根问题: 预训练只学 "下一个 token 是什么", 分不清哪段是用户写的、哪段该模型自己说',
        steps: [
          { name: '整条序列接着训', pain: '(原点) 预训练模型只会接龙, 给它一个问题, 它可能接着编下一个问题', fix: '把 (问, 答) 拼成一条, 照常做「预测下一个 token」的交叉熵训练: 格式学得会, 但问题那段也在算 loss' },
          { name: 'prompt 置 -100', pain: '问题上的 loss 在教模型生成用户的提问, 回复分到的梯度被稀释', fix: 'prompt 位置的 labels 写 -100, 交叉熵既不算它的 loss, 也不计入分母' },
          { name: '回复带上 EOS', pain: '只监督回复内容, 模型不知道什么时候该停', fix: 'EOS 算回复的一部分一起监督: 回复长 R (含 EOS), 就有 R 个位置算 loss' },
          { name: '边界 P-1 + 断言', pain: 'labels 已左移一格, 盖满 P 格会连第一个回复 token 一起盖掉, loss 曲线看不出', fix: 'make_labels 只盖前 $P-1$ 格, 再断言被监督的位置数 = 回复长度' },
        ],
      },
      code: 'llm_finetune/data/tasks.py · llm_finetune/data/instruction_data.py · llm_finetune/methods/sft.py',
      points: [
        { title: 'SFT 没有新 loss', body: 'SFT 没有自己的 loss 函数。代码里 SFTLoss 就是 StandardLMLoss 的别名, 也就是预训练用的那个带 ignore_index=-100 的交叉熵: 每个位置预测下一个 token, 预测得越准 loss 越小; 标签是 -100 的位置直接跳过, 既不算 loss 也不算进求平均的分母。\n预训练和 SFT 的差别只在数据:\n- 预训练: 整段文本每个位置都算 loss, 模型学的是接龙。\n- SFT: 只有回答部分算 loss, 模型学的是「看到问题就回答」。问题是人写的, 不该成为模型的生成目标。\n全部差别只在 labels 里哪些位置是 -100, 这件事由 data/tasks.py 的 make_labels 决定。' },
        {
          title: '边界是 P−1',
          key: true,
          body: '结论: prompt 长 $P$ 时, mask 只能盖前 $P-1$ 格, 第 $P-1$ 格必须留着。\n先说 labels 是怎么排的。模型在位置 $t$ 的输出是在预测位置 $t+1$ 的 token, 所以数据侧直接把标签整体左移一格: labels[t] 存的是 x[t+1]。举个例子, 序列是 [x1, x2, SEP, y1, y2, EOS], prompt 是前 3 个 (含分隔符 SEP), 所以 $P=3$。位置 0 的标签是 x2, 位置 1 的标签是 SEP, 位置 2 (SEP 自己所在的位置) 的标签是 y1。\n最后一个 prompt token 坐在位置 $P-1$, 它要预测的恰好是第一个回复 token y1。这一格就是「看完问题, 第一个词说什么」, 必须算 loss。所以 prompt mask 写成 labels[:, :P-1] = -100, 只盖 prompt 内部的转移 (x1 预测 x2, x2 预测 SEP)。\nmake_labels 里有一条 assert 专门守这个: 被监督的位置数必须等于回复长度 (含 EOS)。写成 [:, :P] 会少一个位置, 立刻炸在这里。',
        },
        { title: '差一格看不出来', body: '多盖一格, 训练时几乎看不出异常, 但模型永远学不会第一个回复 token。\n为什么看不出: 回复有 R 个 token (含 EOS), 多盖一格只是少了一个监督位, 剩下 R-1 个位置的 loss 照样往下掉, 曲线和正确写法差不多。但模型从没在「看完问题」这个位置上收到过梯度, 它不知道第一个词该说什么。第一个词错了, 后面接得再对, 整条也判错。\n- 反转任务 (把 6 个 token 倒过来输出) 上同配置重跑: 正确写法留出集 EM 1.000, 多盖一位 0.000。\n- 极端情况: 答案只有 1 个 token 时, 唯一的监督位也被盖掉, 全部标签都是 -100, 求平均的分母是 0, loss 直接 nan。' },
      ],
      links: [
        { from: '预训练 CE', to: 'SFTLoss', body: '左边是预训练用的交叉熵, 右边是 SFT 的 loss。两者是同一个类, SFT 只换了数据: labels 里 prompt 位置写成 -100。' },
        { from: 'make_labels', to: 'DPO / SimPO / ORPO', body: 'DPO、SimPO、ORPO 是后面几章的偏好优化方法, 它们都要算整条回复的概率 $\\log\\pi(y\\mid x)$, 用的就是这套 labels: 只把标签不是 -100 的位置的 log 概率加起来。边界多盖一格, 第一个回复 token 那一项 $\\log p(y_1\\mid x)$ 就丢了。chosen (好回复) 和 rejected (差回复) 的 prompt 相同, 但第一个回复 token 可以不同, 这一项恰恰是区分两者的。' },
        { from: 'SFT 终态', to: 'policy / reference', body: 'DPO 训练时要两份模型: policy (在训的那份) 和 ref (冻结不动的参照)。两份都从 SFT 结束时的 checkpoint 复制出来。' },
      ],
      sourceRows: [
        { concept: '错位', code: 'llm_finetune/data/tasks.py:make_labels', takeaway: 'labels = seq[:, 1:] 后面补一格 -100: 标签整体左移一格, 最后一个位置没有下一个 token 可预测, 填 -100。输入和标签在数据侧就错开了, loss 函数里不再做 shift。' },
        { concept: 'prompt mask', code: 'labels[:, :P-1] = -100', takeaway: '只把 prompt 内部的转移盖掉。第 $P-1$ 格 (最后一个 prompt token 所在位置) 的目标是第一个回复 token, 盖了模型就学不到它。' },
        { concept: '差一位的守卫', code: 'assert (labels != IGNORE).sum(1) == n_response', takeaway: '被监督的位置数 (标签不是 -100 的个数) 必须恰好等于回复长度 (含 EOS)。边界写错, 这一行立刻报错, 不会等到训完才发现。' },
        { concept: 'ignore_index', code: 'llm_finetune/methods/sft.py: SFTLoss = StandardLMLoss', takeaway: 'SFTLoss 和预训练的 StandardLMLoss 是同一个类。交叉熵对标签为 -100 的位置既不算 loss, 也不把它算进求平均的分母。' },
        { concept: '验收', code: 'llm_finetune/data/tasks.py:exact_match', takeaway: 'exact_match 在留出集上生成整条回复, 和标准答案逐 token 比, 全对才算 1。留出集和训练集按 prompt 的 token 和 mod 5 切开, 两边不相交, 所以报的是泛化能力, 不是背题。' },
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
      subtitle: '读完你能说清 LoRA 换来了什么、又赔上了什么, 不再照抄那句「LoRA 收敛更快」。',
      tldr: 'LoRA (Low-Rank Adaptation) 微调时不改原来的权重矩阵 $W$, 而是在旁边挂一条小旁路: $y = Wx + (\\alpha/r)\\cdot BAx$。$W$ 冻结, 只训两个小矩阵 $A$ 和 $B$。\n- 省下的: 基座每个参数的梯度和 Adam 优化器状态, 以及「每个任务存一整份权重」。\n- 本章实验没省下的: 训练步数。同样 300 步, 全参微调在留出集 (没见过的题) 上整条答对的比例 (EM) 是 0.809, LoRA (r=8) 只有 0.352。',
      question: 'LoRA 到底买到了什么? 为什么 B 必须初始化成 0, 而 A 不行?',
      evolution: {
        title: '从改整张矩阵到挂一条旁路',
        subtitle: '根问题: 全参微调要给每个参数配梯度和两份 Adam 状态, 每个任务还要存一整份权重',
        steps: [
          { name: '全参微调', pain: '(原点) 适配新任务, 最直接的是让所有参数都动', fix: '每个参数要配 1 份梯度加 Adam 优化器的 2 份状态 (本仓库 778 KiB), N 个任务还要落盘 N 份整模型' },
          { name: 'Adapter', year: 2019, pain: '显存和存储都随参数量涨, 想省就只能训一小撮参数', fix: '每层串进一个小瓶颈 MLP, 只训它; 但它串在前向里, 推理时每层多算一段, 拿不掉' },
          { name: 'Prefix Tuning', year: 2021, pain: 'adapter 拿不掉, 小 batch 推理的延迟跟着涨', fix: '不改层结构, 只学一段拼在输入前的前缀向量: 难优化, 还占掉一截上下文' },
          { name: 'LoRA', year: 2021, pain: '前缀挤占上下文又难训, adapter 又加推理延迟', fix: '和 W 并联一条低秩旁路 $BA$, 只训 A/B; 训完加回 W, 推理零开销' },
        ],
      },
      code: 'llm_finetune/methods/lora.py · llm_finetune/methods/qlora.py · llm_finetune/utils/param_utils.py',
      points: [
        { title: '低秩的是 ΔW, 不是 W', body: 'LoRA 的假设是: 微调对权重的改动量 $\\Delta W$ 是低秩的, 而不是权重 $W$ 本身低秩。\n「低秩」的意思是, 这个矩阵虽然有 $d_{\\text{out}} \\times d_{\\text{in}}$ 那么大, 但它装的信息很少, 可以写成两个瘦矩阵的乘积。LoRA 就用 $B\\,(d_{\\text{out}}\\times r)\\cdot A\\,(r\\times d_{\\text{in}})$ 去装它, $r$ 是中间那个很小的维度, 叫秩。\n参数量从 $d_{\\text{in}}\\cdot d_{\\text{out}}$ 降到 $r\\cdot(d_{\\text{in}}+d_{\\text{out}})$。这个式子在说: 原来要存一整张矩阵, 现在只存两条窄带。$d=4096$、$r=8$ 时, 一层从 16M 个参数降到 65K 个, 只占 0.39%。\n边界: 任务离预训练的分布越远, 需要的改动 $\\Delta W$ 越不低秩, LoRA 越吃亏。本仓库的小模型 d_model=64, r=8 已经占了 19.5% 的参数, 测不出低秩够不够用。' },
        { title: 'B = 0 是无害启动', body: '结论: B 全零、A 随机, 这样训练起点和原模型完全一样, 而且第 1 步就有梯度。\n- B 全零: 旁路的输出是 $BAx$, B 是 0 就意味着 $BA = 0$, 第 0 步模型输出和基座逐位相同。微调的起点就是预训练的终点, 不会一上来就把基座破坏掉。脚本里有断言: 注入 LoRA 后、训练前, 输出必须和基座逐位相等。\n- A 随机 (Kaiming uniform 初始化, 和 nn.Linear 默认的一样): B 的梯度正比于 $Ax$, A 不是 0, B 第 1 步就有梯度可以动。\n- 两个都置零: B 的梯度要乘 A, A 的梯度要乘 B, 两个都是 0, 梯度永远是 0, 永远训不动。\n- 两个都随机: 第 0 步 $BA$ 就不是 0, 基座的输出被随机改动, 起点就坏了。' },
        {
          title: '买的是显存和分发, 不是速度',
          key: true,
          body: '结论: LoRA 省的是显存和存储, 不是训练步数。在本章这个实验里, 同样训 300 步, 它的留出集效果比全参微调差得多。\n实验: 同一个基座 (已经会 copy 任务), 适配到 sort 任务 (把 6 个 token 排序后输出), 各训 300 步:\n- 全参: 99,648 个可训参数 / Adam 状态 778 KiB / 留出集 EM 0.809\n- LoRA r=8: 19,456 个可训参数 / Adam 状态 152 KiB, lr 从 3e-3 拉到 1e-2 也才 EM 0.352 (同 lr 3e-3 时只有 0.281)\nAdam 状态是优化器给每个可训参数额外存的两个数 (一阶、二阶动量), 按每个参数 8 字节算: 可训参数少了, 这部分显存就少了。\n「LoRA 收敛更快」是个流行说法。它通常来自病态初始化 (初始 loss 大到 250 而不是 ln V)、只背几条样本、以及单独给 LoRA 调大 lr。在初始化正常、同看留出集的条件下, 本章这次 copy → sort、300 步的实验没有观察到「LoRA 收敛更快」; 这个结论也不能外推到所有任务和训练配置。\n边界: 表里的 778 KiB / 152 KiB 只是「可训参数 × 8 字节」。前向的激活和反传的中间量仍要过整个基座, 这部分 LoRA 省不掉, 脚本没有量。',
        },
      ],
      links: [
        { from: 'SFT loss', to: 'LoRA SFT', body: '左边是 SFT 的交叉熵 loss, 右边是用 LoRA 做 SFT。loss 一行不改, 变的只是 optimizer 能更新哪些参数: 只有 A、B 的 requires_grad 是 True。' },
        { from: 'target_modules', to: '模型结构', body: 'target_modules 是「要往哪些层挂旁路」的名单。apply_lora 按属性名的最后一段匹配 (w_q / w_v 这些名字), 不管这一层在模型的哪个路径下。所以同一份代码能注到任何沿用这套属性名的模型上; 上游改了名字, 一层也命中不了, 会抛 ValueError。' },
        { from: 'methods/lora.py', to: 'methods/qlora.py', body: 'QLoRA 是下一章: 它复用同一套 A/B 旁路, 只把冻结的基座权重换成 4 bit 存储 (NF4 格式打包成 uint8, 每 64 个权重配一个缩放系数 scale)。' },
        { from: 'adapter', to: 'llm_infer Multi-LoRA', body: 'adapter 就是训完的 A、B 这几个小矩阵。不把它合并进 W, 多个 adapter 就能共享同一份基座, 推理时按请求切换。' },
      ],
      sourceRows: [
        { concept: 'LoRALinear', code: 'llm_finetune/methods/lora.py:LoRALinear', takeaway: '一个线性层被包成两条路: base 分支是冻结的原层, adapter 分支是 A、B。A 用 Kaiming uniform 随机初始化, B 全零。' },
        { concept: '注入', code: 'llm_finetune/methods/lora.py:apply_lora', takeaway: 'apply_lora 按模块名找到目标 Linear 层, 原地换成 LoRALinear。它只注入, 不负责冻结其余参数。' },
        { concept: '只训 adapter', code: 'llm_finetune/methods/lora.py:mark_only_lora_as_trainable', takeaway: '把名字不含 lora_ 的参数的 requires_grad (PyTorch 里「这个参数要不要算梯度」的开关) 全部关掉, embedding、norm、lm_head 也一起冻结。之后要用 print_trainable_parameters 数一遍可训参数才算数: loss 在降, 证明不了 base 被冻住。' },
        { concept: 'merge', code: 'llm_finetune/methods/lora.py:merge_lora_weights', takeaway: '合并就是 $W \\leftarrow W + (\\alpha/r)\\cdot B\\cdot A$, 换回普通 Linear。这个式子在说: 旁路的贡献被一次性加进原矩阵, 推理时就没有旁路了。要断言合并前后输出的数值相等 (实测 logits, 也就是模型最后输出的分数, 最大差 6.4e-06); 只断言形状相等, 会漏掉转置写反和漏乘 $\\alpha/r$ 这两种 bug。' },
        { concept: '落盘', code: 'llm_finetune/methods/lora.py:get_lora_state_dict', takeaway: '只抽出名字含 lora_A / lora_B 的张量存盘: 本仓库 adapter 76 KiB, 整模型 389 KiB。部署时先加载基座, 再 apply_lora, 再把 adapter 装回去。' },
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
      subtitle: '读完你能算出一个模型量化之后到底占多少字节, 还能说清为什么 embedding 和 lm_head 要放过。',
      tldr: 'LoRA 省掉了梯度和优化器状态, 但冻结的基座还整份躺在显存里。\nQLoRA 把基座权重存成 4 bit 的 NF4 格式 (NormalFloat 4-bit: 16 个码点按正态分布的分位数摆放), 前向时现场反量化成浮点再算。本仓库整模型从 389 KiB 压到 59 KiB, 压了 6.57 倍, 基座在原任务上的留出集 EM 仍是 1.000。',
      question: '同样 16 个码点, 为什么按分位数摆比等间距好? block 为什么不能太大也不能太小?',
      evolution: {
        title: '显存大头一块块往下挪',
        subtitle: '根问题: 微调显存 = 权重 + 梯度 + 优化器状态 + 激活, 砍掉一块, 下一块就成了大头',
        steps: [
          { name: 'LoRA + fp32 基座', pain: '(原点) LoRA 砍掉了基座的梯度和 Adam 状态', fix: '冻结的基座仍按 fp32 整份常驻: 只读不写, 却成了显存大头 (本仓库 389 KiB)' },
          { name: '整层 INT4', pain: '基座只读, 每个数用 32 bit 存是浪费', fix: '整层除以最大绝对值 absmax, 缩到 [-1, 1] 后映射到 16 个等间距码点: 一个离群值就把量程撑大' },
          { name: '分 block 缩放', pain: '量程被离群值撑大, 大部分权重挤进中间几个码', fix: '每 64 个权重一个 absmax: 离群值只影响本 block, 多花 0.0625 字节/参数' },
          { name: 'NF4 + LoRA', year: 2023, pain: '权重近似正态, 等间距码点在两端几乎用不上', fix: '码点取正态分位数; 前向现场反量化, 梯度只进高精度 A/B: 389 → 59 KiB' },
        ],
      },
      code: 'llm_finetune/methods/qlora.py · llm_finetune/run_finetune/qlora/train_qlora.py',
      points: [
        { title: '码点跟着密度走', body: '结论: 4 bit 只有 16 个码点, NF4 把它们放在权重最密集的地方, 所以比等间距的 INT4 浪费少。\n量化就是把每个浮点权重换成 16 个固定数值 (码点) 里最近的那一个, 只存它的编号 (4 bit)。码点摆在哪决定误差多大。\n神经网络的权重大多挤在 0 附近, 分布近似正态: 越靠近 0 越密, 两端很稀。\n- NF4: 16 个码点取标准正态分布的等概率分位数, 中间密两端疏。每个码点被用到的概率都接近 1/16, 没有哪个码点闲着。\n- 等间距 INT4: 码点均匀摆在 [-1, 1] 上, 两端那几个码点几乎没有权重落进去, 等于浪费了。\n代码里 NF4_CODEBOOK 就是 bitsandbytes 公布的这 16 个数, 归一化到 [-1, 1], 0 被显式保留。' },
        {
          title: '账要算整个模型',
          key: true,
          body: '结论: 压缩比要按整个模型算, 只报被量化那几层的数字是在骗自己。\n先算理论上限。每个参数存 4 bit 的编号, 是 0.5 字节。另外每 64 个参数共用一个 fp32 的缩放系数 scale (4 字节), 摊到每个参数是 4/64 = 0.0625 字节。加起来 0.5625 字节/参数。fp32 是 4 字节/参数, 所以上限是 4/0.5625 = 7.1 倍。\n实测只有 6.57 倍。差额来自没被量化的层: embedding、lm_head、RMSNorm 还是 fp32。本模型词表只有 16, embedding 很小, 差得不多; 真实 LLM 里 embedding 加 lm_head 占几个百分点, 整模型压缩比会更低一些。\n代码里 weight_bytes 数的就是整模型的真实字节数: 普通参数按各自精度算, NF4 层按 uint8 包加 scale 算。',
        },
        { title: '量化的是基座, 不是训练', body: '结论: 被压成 4 bit 的只有冻结的基座, LoRA 旁路和梯度全程是高精度。\n- NF4 权重: 存成只读 buffer (uint8 包加 scale), 不是 Parameter, 不进 optimizer, 没有梯度。前向时 NF4Linear 的 weight 属性现场反量化成 fp32 再做矩阵乘。\n- LoRA 分支: A、B 和它们的梯度都是 fp32。基座是常数, A、B 的梯度只依赖输入激活和上游梯度, 量化误差只是让「被适配的那个函数」略有不同, 不影响梯度本身的正确性。\n实测: 单层 (layers.0.ffn.w_up) 的相对量化误差 9.0%。这点误差由 adapter 在训练里顺带补掉: 适配 sort 任务的留出集 EM 0.402, 比同 r、同 lr、同步数的 fp32 LoRA (0.352) 还略高。\n边界: 0.402 高于 0.352 不能读成「量化后更好」。两个脚本种子不同, dora 脚本里 LoRA 三个种子的 EM 是 0.352 / 0.406 / 0.387, 0.402 落在这个范围里。' },
      ],
      links: [
        { from: 'methods/lora.py', to: 'methods/qlora.py', body: '左边是 LoRA 的 A/B 旁路, 右边是 QLoRA。A/B 分支完全复用, 只换 base 权重的存储方式: nn.Linear 换成 NF4Linear。' },
        { from: 'nf4_quantize', to: 'nf4_dequantize', body: '量化分四步: 把权重切成每 64 个一块; 每块除以自己的最大绝对值 absmax 缩到 [-1, 1]; 每个数在 16 个码点里找最近的, 记下编号; 两个 4 bit 编号拼进一个 uint8。反量化是倒过来: 拆开字节拿到编号, 查码本拿到数值, 再乘回这一块的 scale。' },
        { from: 'block_size', to: '误差 / 开销', body: 'block 是共用一个 scale 的那一组权重。block 小, scale 更贴合局部, 误差更小; 但 scale 的开销 $4/B$ 字节每参数也变大 ($B$ 是 block_size, 一块里的参数个数)。这个式子在说: 一块越小, 摊到每个参数头上的 scale 字节越多。论文再对 scale 本身做一次量化 (双重量化), 就是为了压这块开销, 本仓库没实现。' },
        { from: 'QLoRA', to: 'llm_infer 权重量化', body: '按块取 absmax 再量化, 这套做法在推理章的权重量化里再出现一次。' },
      ],
      sourceRows: [
        { concept: '码本', code: 'llm_finetune/methods/qlora.py: NF4_CODEBOOK', takeaway: 'NF4_CODEBOOK 是 16 个固定数值: 标准正态分布的 16 个等概率分位点, 归一化到 [-1, 1], 0 被显式保留。' },
        { concept: '量化', code: 'llm_finetune/methods/qlora.py:nf4_quantize', takeaway: 'nf4_quantize 按 64 个一块除以 absmax, 用 argmin 找最近码点, 两个 4 bit 编号打进一个 uint8。block_size 为奇数时编号个数可能是奇数, 打包会错位, 脚本开头有单测。' },
        { concept: '反量化', code: 'llm_finetune/methods/qlora.py:nf4_dequantize', takeaway: 'nf4_dequantize 把每个字节拆成两个编号, 查码本, 再乘回各块的 scale。' },
        { concept: '前向', code: 'llm_finetune/methods/qlora.py:NF4Linear', takeaway: 'NF4Linear 的 weight 是个 property, 每次取都现场反量化, 不缓存, 常驻内存的只有 4 bit 的 buffer。外面再套 LoRA: $y = \\mathrm{dequant}(W)x + (\\alpha/r)\\cdot BAx$。这个式子在说: 基座先解压再乘输入, 旁路照 LoRA 那样算, 两者相加。' },
        { concept: '哪些层放过', code: 'llm_finetune/methods/qlora.py:apply_qlora', takeaway: 'apply_qlora 只量化 14 个线性层 (2 个 block, 每个 4 个注意力投影加 3 个 SwiGLU 矩阵)。embedding 和 lm_head 放过: 两者共享权重; embedding 是查表, 每行的数值分布不像正态; lm_head 的误差直接进 logits (模型输出的分数)。bitsandbytes 默认也跳过它们。' },
        { concept: '显存账', code: 'llm_finetune/methods/qlora.py:weight_bytes', takeaway: 'weight_bytes 按 0.5625 字节/参数数 NF4 层。合并时先反量化再加 $\\Delta W$, 结果保持高精度; 合并后再量化回 4 bit, 刚学到的更新会被量化噪声抹掉一部分。' },
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
      subtitle: '读完你能说清 LoRA 和全参微调的差距卡在哪一步, 以及一个 $d_{\\text{out}}$ 维向量怎么补上一截。',
      tldr: 'DoRA 把权重矩阵的每一行看成「长度 × 方向」, 让 LoRA 的低秩更新只负责转方向, 长度另外用一个向量 $m$ 来学: $W^\\prime = m \\odot (W_0 + BA) / \\|W_0 + BA\\|_{\\text{row}}$。\n- 低秩更新 $BA$: 只管转方向, 因为除以行范数之后长度被归一掉了。\n- $m$: 每行一个数, 单独学长度。\n每层只多 $d_{\\text{out}}$ 个参数。同 r=8、3 个种子平均: LoRA 留出集 EM 0.382, DoRA 0.522。',
      question: '同样的秩 r, 为什么多加一组"每行一个标量"就能更接近全参微调?',
      evolution: {
        title: '把一个增量拆成长度和方向',
        subtitle: '根问题: 秩 r 不变, 怎样让低秩更新改权重的方式更像全参微调',
        steps: [
          { name: '全参微调', pain: '(原点) 每行权重有长度也有方向, 微调时两样都可能要改', fix: '两样能独立调 (DoRA 论文测得二者变化负相关), 但每个参数都要梯度和 Adam 状态' },
          { name: 'LoRA', year: 2021, pain: '全参太贵, 只能用一个低秩增量去改', fix: '$W_0 + BA$ 省下大半参数; 但增量一加, 长度和方向同向一起变 (论文测得正相关)' },
          { name: 'DoRA', year: 2024, pain: '想只转方向不改长度, $BA$ 得恰好保持行范数, 低秩很难凑到', fix: '照 weight norm 写成 $m\\cdot V/\\|V\\|$: 方向交给 BA, 长度交给每行一个 m' },
          { name: '范数 detach', pain: '范数对整行求导, 反传要多存一份 [d_out, d_in] 的梯度', fix: '按论文 §4.3 把分母当常数: 省下这份显存, 效果几乎不变' },
        ],
      },
      code: 'llm_finetune/methods/dora.py · llm_finetune/methods/lora.py',
      points: [
        {
          title: '两个量在 LoRA 里被绑住了',
          key: true,
          body: '结论: LoRA 一加旁路, 每行权重的长度和方向只能一起变; DoRA 把两件事拆开。\n先说「长度」和「方向」指什么。权重矩阵 $W$ 的每一行是一个向量, 它有长度 (行范数, 各元素平方和开根号) 和方向 (除以长度之后剩下的单位向量)。\n- 全参微调: DoRA 论文逐层测了微调前后每行长度变了多少、方向转了多少, 发现两者负相关 (相关系数 −0.62): 长度改得多的层, 方向改得少, 反过来也一样。LoRA 测出来是正相关 (+0.83), 两个量同步变。\n- 为什么 LoRA 绑住: $\\Delta W = BA$ 是直接加到 $W_0$ 上的, 一加上去, 这一行的长度和方向一起变。想「只转方向不改长度」, 就得让 $\\|W_0 + BA\\| = \\|W_0\\|$ 恰好成立。这个式子在说新行的长度要和旧行一模一样, 低秩的 $BA$ 很难凑到。\n- DoRA: 先把 $W_0 + BA$ 除以自己的行范数, 长度归一, 剩下的只有方向; 长度交给另一个向量 $m$ 单独学。',
        },
        { title: '长度不进前向, 低秩只管方向', body: '结论: 归一化之后, $BA$ 把 $V$ 拉得多长都不影响前向, 起作用的只有它转出来的方向。\n记 $V = W_0 + (\\alpha/r)BA$, 前向用的权重是 $W^\\prime = m \\odot V/\\|V\\|_{\\text{row}}$。这个式子在说: 每行先除以自己的长度变成单位向量, 再乘上这一行学出来的长度 $m$。所以 $V$ 的长度怎么变, $W^\\prime$ 都不变。\n反传时「除以范数」这一步有两种处理:\n- 梯度流过范数 (论文原式): 对范数也求导, $V$ 的梯度里沿 $V$ 方向的那一份会被投影掉, 低秩容量全花在转方向上。代价是反传要多存一份 [d_out, d_in] 大小的梯度。\n- 分母 .detach() (本仓库, 按论文 §4.3): 把范数当常数, 梯度只经过分子和 $m$。省下那份全尺寸梯度的显存, 效果几乎不变。代价是上面说的投影不再发生。' },
        { title: '参数几乎白送, 时间不白送', body: '结论: DoRA 多出的参数可以忽略, 但训练时每步多一次全尺寸矩阵的构造; 推理时和 LoRA 一样零开销。\n- 参数: 每层比 LoRA 多 $d_{\\text{out}}$ 个 ($m$ 每行一个数)。本仓库从 19,456 涨到 20,736, 多 1,280 个。\n- 训练: LoRA 的前向可以拆成两次小矩阵乘 (先乘 A 降到 r 维, 再乘 B 升回去), 不用显式构造 $\\Delta W$。DoRA 的归一化作用在整行 $W_0 + BA$ 上, 每步都要先显式构造出 [d_out, d_in] 的 $W^\\prime$ 再做矩阵乘, 拆不开。d_model=64 时这个矩阵很小; d=4096 时每层每步都要多构造一个全尺寸矩阵。\n- 推理: 训完把 $W^\\prime$ 算出来写回 W, 就是普通 Linear, 零开销。\n效果 (同基座 copy 适配到 sort, r=8, lr=1e-2, 300 步, 3 个种子平均): 末步 loss LoRA 0.369, DoRA 0.296; 留出集 EM LoRA 0.382, DoRA 0.522。' },
      ],
      links: [
        { from: 'LoRALinear', to: 'DoRALinear', body: 'DoRALinear 继承 LoRALinear, A、B、缩放系数、delta() 全部沿用, base + BA 不变。只重写两处: merged_weight() 多了「按行归一化再乘 m」, forward() 每步用它现算整个 $W^\\prime$。' },
        { from: 'weight normalization', to: 'DoRA', body: 'weight normalization 是一种老办法: 把一个权重向量拆成「长度 × 单位方向」两个参数分开学, 优化曲面更好走。DoRA 的 $m \\cdot V/\\|V\\|$ 就是这个写法, 只是方向那一半由 LoRA 旁路来改。' },
        { from: 'DoRA', to: 'merge', body: '训完把每行的 $m\\cdot V/\\|V\\|$ 算出来写回 $W$, 模型里就没有 DoRA 的痕迹了。推理和 LoRA 一样没有额外开销。' },
      ],
      sourceRows: [
        { concept: '初始化', code: 'llm_finetune/methods/dora.py:DoRALinear', takeaway: 'lora_magnitude 就是长度向量 $m$, 初始化成 $\\|W_0\\|_{\\text{row}}$ (基座每一行的范数), 形状 [d_out]。配上 $B = 0$, 第 0 步 $V = W_0$, 归一化后再乘回原长度, $W^\\prime$ 和 $W_0$ 逐位相同。$m$ 初始化成 1 就没这个性质了, 第 0 步输出就变了。' },
        { concept: '方向分支', code: 'V = W₀ + (α/r)·B·A', takeaway: 'V 和 LoRA 一模一样: 基座加低秩更新。但后面只用到它的方向, 它的长度会被归一化掉。' },
        { concept: '归一化', code: 'direction.norm(dim=1, keepdim=True).detach()', takeaway: '对 V 按行取范数, 再 .detach() 把它当常数 (论文 §4.3)。忘了 detach 结果差不多, 但反传要多一份全尺寸梯度的显存。' },
        { concept: '幅度', code: 'lora_magnitude: nn.Parameter([d_out])', takeaway: 'lora_magnitude 的名字带 lora_ 前缀, 所以 mark_only_lora_as_trainable 和 get_lora_state_dict 能直接认出它, 冻结和落盘工具不用改。训练后 layers.0.attn.w_q 这一层的 m 相对初值平均变了 59.7%: 这个任务确实要改长度。' },
        { concept: '换个 r 就不稳', code: 'run_finetune/dora/README.md', takeaway: '3 个种子平均: 末步 loss 从 0.369 降到 0.296, EM 从 0.382 升到 0.522。但 r 换成 2 时, 3 个种子里有 1 个是 LoRA 赢 (这个对照不在脚本里)。玩具规模, 脚本的断言只要求 3 个种子的平均 loss DoRA 低于 LoRA。' },
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
      subtitle: '读完你会多盯一个指标: 除了 chosen 和 rejected 的分差 (reward margin), 还要看 chosen 自己的概率 $\\log\\pi(\\text{chosen})$。它掉下去的时候, 模型正在变差。',
      tldr: 'RLHF (用人的偏好做强化学习) 原本要两步: 先用偏好数据训一个给回复打分的奖励模型 (RM), 再用 PPO 这种强化学习算法在线采样、让模型去拿高分。两步都贵。\nDPO (Direct Preference Optimization) 把两步并成一个二分类: 数据是 (prompt, chosen, rejected) 三元组, chosen 是更好的回复, rejected 是更差的。训练目标是让 policy (在训的模型) 相对 ref (冻结的 SFT 起点副本) 更偏向 chosen、更远离 rejected。\n自检: policy 刚从 ref 复制出来时两边概率比都是 1, 差值 $\\Delta = 0$, 第 1 步 loss 恰好是 $\\ln 2 = 0.6931$。',
      question: 'chosen 和 rejected 的分差 (reward margin) 一路拉大, 为什么模型生成的答案反而更差了?',
      evolution: {
        title: '用上 rejected, 又别把模型压坏',
        subtitle: '根问题: 偏好数据里有 "哪条更差" 的信息, 但把概率往下压没有天然的下限',
        steps: [
          { name: '只拿 chosen 做 SFT', pain: '(原点) 手里是 (x, chosen, rejected), 最省事是只用 chosen', fix: 'chosen 学得会, 但 rejected 里 "错在哪" 的信号整个扔掉了' },
          { name: '同时压 rejected', pain: '只用了一半数据, 模型不知道哪种写法是错的', fix: '再加一项把 rejected 的 log p 往下压: 没有下限, 两条共享的 token 也被一起压低' },
          { name: 'DPO', year: 2023, pain: '往下压没有停手线, 也没有 "别离起点太远" 的约束', fix: '两条回复各算「policy 概率 / ref 概率」的对数, 相减后过 $-\\log\\sigma$: 差距拉开后梯度就消失, β 管离 ref 多远' },
          { name: 'KTO', year: 2024, pain: '只看差值, 两边一起降也算赢; 还必须同一题凑成对', fix: '单条样本各对参考点 $z_0$ 比高低, 好样本自己往上推: log π(chosen) −4.03 → −3.73' },
        ],
      },
      code: 'llm_finetune/methods/dpo.py · llm_finetune/data/preference_data.py',
      points: [
        { title: '两步是怎么并成一步的', body: '结论: RLHF 的「先训 RM 再做 RL」可以在数学上合并成一个直接在偏好对上做的二分类, 这就是 DPO。\n先说 RLHF 的目标: 让回复的奖励 $r(x,y)$ 尽量高, 同时别离起点 ref 太远。「别离太远」用 KL 散度 (衡量两个概率分布差多少的量) 做惩罚, 系数叫 $\\beta$。\n- 最优策略长什么样: 这个目标的解满足 $r(x,y) = \\beta\\cdot\\log\\pi/\\pi_{\\text{ref}} + \\text{const}$。这个式子在说: 一条回复的奖励, 等于 policy 相对 ref 把它的概率抬高了多少 (取对数) 再乘 $\\beta$, 外加一个和 $y$ 无关的常数。所以奖励可以直接从 policy 和 ref 的概率比里读出来, 不用另训 RM。\n- Bradley-Terry: 这是训 RM 时用的偏好模型, 把「A 比 B 好」的概率写成 $\\sigma(r_A - r_B)$, $\\sigma$ 是 sigmoid 函数, 分差越大概率越接近 1 (见 RM 一章)。\n把上面的 $r$ 代进 Bradley-Terry, 两条回复的 const 相减消掉, 剩下的就是一个只含 policy 和 ref 概率的式子。RM 和 PPO 两步就塌缩成一个对偏好对的二分类。' },
        { title: '数据形态变了, loss 外壳没变', body: '结论: DPO 的数据是三元组, loss 还是「$-\\log\\sigma$ 套一个分差」的二分类外壳。\n样本是 (prompt, chosen, rejected), 不再是单条标准答案。本仓库里 chosen 是正确回复, rejected 是把正确回复改错一个 token 或漏掉一个 token。\n送进 $-\\log\\sigma(\\cdot)$ 的是隐式奖励差 $\\Delta = (\\log\\pi_c - \\log\\mathrm{ref}_c) - (\\log\\pi_r - \\log\\mathrm{ref}_r)$。这个式子在说: 先算 policy 比 ref 把 chosen 抬高了多少, 再算把 rejected 抬高了多少, 两者相减。这里每个 $\\log\\pi$ 都是整条回复的 log 概率, 也就是回复里每个 token 的 log 概率加起来 ($\\sum\\log p$)。\n一个坑叫长度红利: 每个 token 的 log 概率都是负数, 序列越长 $\\sum\\log p$ 越负。所以如果 rejected 比 chosen 长, 它的 $\\log\\pi_r$ 天然更低, 什么都不学 $\\Delta$ 就是正的, 白得一截 margin。\nref 怎么解决: ref 对同一条长回复也给出同样低的 $\\sum\\log p$, 相减之后这一截被抵消。' },
        {
          title: 'margin 变大不等于模型变好',
          key: true,
          body: '结论: loss 只看 chosen 和 rejected 的差值。「两边一起降、rejected 降得更快」同样让 loss 变小, 这时模型实际在变差。\n看 200 步实测 (起点是 SFT 100 步的「半会」模型, β=0.5, lr=3e-4, 全部指标在留出偏好对上):\n- 留出集偏好准确率 (policy 给 chosen 的概率高于 rejected 的比例): 从 0.965 升到 0.996\n- log π(chosen) (chosen 整条回复的 log 概率): 从 −4.03 降到 −4.18\n- 贪心 exact-match (每步取最大概率的 token 生成, 整条全对的比例): 从 0.332 降到 0.137\n偏好准确率在涨, 说明分差拉开了; 但 chosen 自己的概率在掉, 生成质量跟着掉。这个现象叫 likelihood displacement (概率被挤走), 脚本对它设了断言。\n所以训 DPO 时要同时盯 logp_chosen, 不能只看 reward margin。',
        },
        { title: 'β 越大越保守', body: '结论: $\\beta$ 大, policy 贴着 ref 不敢动; $\\beta$ 小, policy 漂得远。\n$\\beta$ 不是「用力程度」, 它就是 RLHF 目标里 KL 惩罚的系数, 决定离 ref 多远要挨罚。它在 loss 里的位置是 $-\\log\\sigma(\\beta\\Delta)$。\n- β 大: 同样的 $\\Delta$ 乘上大 $\\beta$, $\\sigma$ 很快饱和到 1。饱和之后梯度几乎为 0, 所以 chosen 刚排到 rejected 前面一点点, 这条样本就没梯度了, policy 贴着 ref 不动。\n- β 小: 要把 $\\Delta$ 拉得很大 $\\sigma$ 才饱和, policy 会一直推, 漂得更远。\n本仓库的 DPO 用 β=0.5。' },
        {
          title: 'KTO: 只有 👍 / 👎 也能训',
          body: '结论: KTO (Kahneman-Tversky Optimization) 不需要成对数据, 单条回复加一个好/坏标签就能训, 而且没有 DPO 的 likelihood displacement。\n问题: 线上日志里多是单条回复的 👍 / 👎, 同一个 prompt 凑不出一对, DPO 用不了。\nKTO 的做法: 给每条样本一个共同的参考点 $z_0$, 用它代替 DPO 里「对面那条」。\n- 好回复: 它的隐式奖励 $r = \\log\\pi/\\pi_{\\text{ref}}$ (policy 比 ref 把它抬高了多少) 要高过 $z_0$。\n- 坏回复: $r$ 要低过 $z_0$。\n实测, 同 SFT 起点、同 β=0.5、同 200 步, 好坏 1:1:\n- 偏好准确率: 从 0.965 升到 0.992 (DPO 0.996)\n- log π(chosen): 从 −4.03 升到 −3.73 (DPO 掉到 −4.25)\n- 贪心 EM: 0.324 (DPO 0.121)\n这里的 DPO 是 train_kto 里同条件重跑的对照; 单独跑 train_dpo 是 −4.18 / EM 0.137。两次独立运行差别在随机数, 结论一致。\n为什么 KTO 的 chosen 没掉: 好样本有自己的一项, 直接要求它的 $r$ 高过参考点, 梯度把它往上推。DPO 只要求差值变大, 没有这一项。',
        },
        {
          title: 'KTO: 好坏不均时 λ 要按比例调',
          body: '结论: 好坏样本数量不均时, 要把少数类的权重按比例调高, 否则多数类的梯度压倒一切。\n设置: 好:坏 = 1:9, 坏样本多 9 倍。$\\lambda_D$、$\\lambda_U$ 是好、坏样本各自的 loss 权重, $n_D$、$n_U$ 是各自的样本数。论文建议让两类的总权重相等: $\\lambda_D n_D \\approx \\lambda_U n_U$。这个式子在说: 一类样本少, 就给它每条更大的权重, 两类加起来一样重。$\\lambda_D = 1$ 时 $\\lambda_U$ 取 1/9。\n- 调了: 偏好准确率 0.988, EM 0.254。\n- 不调: 坏样本的梯度压倒一切, 模型把所有回复一起往下压。偏好准确率 0.492 (和瞎猜差不多), log π(chosen) −30.02, EM 0.000, 生成全崩。\n这份 1:9 的数据每个 prompt 只出现一条回复, DPO 一对也凑不出来; KTO 能吃这种数据。',
        },
        {
          title: 'KTO: 本例里参考点没起作用',
          body: '结论: 本脚本里 $z_0$ 全程是 0, 参考点那一项没有发挥作用, 实际跑的是「好样本 σ(βr)、坏样本 σ(−βr)」。\n- 参考点怎么估: $z_0$ 取错配样本 $(x_i, y_{i+1})$ 上 $\\log\\pi/\\pi_{\\text{ref}}$ 的均值。错配就是把别的 prompt 的回复配给这个 prompt, 得到一条「答非所问」的样本; 代码里把同一个 batch 的回复整体挪一行来做错配。估出来的值不回传梯度, 还被 clamp 到不小于 0。\n- 本例的读数: 这个估计从 0 一路降到 −2.52, 因为 policy 把答非所问的回复压得比 ref 还低; 负数被 clamp 成 0, 所以 $z_0$ 全程为 0。\n- 标签密度: 同样的前向预算下, KTO 每步有标签的回复只有 64 条, 另外 64 条前向花在错配样本上估 $z_0$; DPO 每步 64 对等于 128 条有标签的回复。这是上面 DPO 偏好准确率略高的一个原因。',
        },
      ],
      links: [
        { from: 'SFT 终态', to: 'policy / reference', body: 'policy 和 ref 都从 SFT 终态复制。policy 可训; ref 冻结, 置成 eval 模式, 前向在 no_grad (不记录梯度) 下跑。ref 如果被注册成子模块, 它会被加进 optimizer、被 Trainer 的 .train() 切回训练模式, 所以代码里故意把它塞在一个 tuple 里, PyTorch 就不会把它当子模块。' },
        { from: 'make_labels', to: 'compute_sequence_logprobs', body: 'compute_sequence_logprobs 用 SFT 那套 labels 决定哪些位置的 log 概率要加起来: prompt 和 pad 位置是 -100, 跳过。多盖一位会丢掉 $\\log p(y_1\\mid x)$ 这一项。有人会说「这一步的上下文对 chosen 和 rejected 相同, 丢了也没关系」: 上下文相同, 但目标 token $y_1$ 可以不同, 这一项恰恰是区分两者的。' },
        { from: 'DPOLoss', to: 'SimPO / ORPO', body: 'SimPO 和 ORPO 是下一章不要 ref 的两种偏好优化。三者用同一份数据、同一个 $-\\log\\sigma(z)$ 外壳, 换的只是送进去的 z 怎么算, 以及要不要 ref。' },
        { from: 'PairwiseForward', to: '通用 Trainer', body: 'Trainer 每步只会调一次 model(**batch)。DPO 一步要跑两次前向 (policy 一次, ref 一次)。PairwiseForward 把这两次前向包成一个 nn.Module 的 forward, Trainer 一行不改, 不用为 DPO 单写一套训练循环。' },
      ],
      sourceRows: [
        { concept: 'logprob 汇总', code: 'llm_finetune/methods/dpo.py:compute_sequence_logprobs', takeaway: '把每条回复里被监督位置的 log 概率加起来, 返回 [B] 形状的张量, 每条样本一个数。用 cross_entropy(reduction="sum") 会把整个 batch 加成一个数, 没法逐对相减。' },
        { concept: 'DPO 的 z', code: 'beta * ((p_c - p_r) - (ref_c - ref_r))', takeaway: 'z 是 policy 相对 ref 的隐式奖励差: (policy 抬高 chosen 多少) 减 (policy 抬高 rejected 多少), 再乘 β。policy 和 ref 完全相同时每一项都是 0, z 恒为 0, 所以第 1 步 loss 必是 $-\\log\\sigma(0) = \\ln 2$。这是最好用的自检。' },
        { concept: '梯度权重', code: 'llm_finetune/methods/dpo.py:DPOLoss', takeaway: '每个样本的梯度权重是 $\\sigma(-\\beta\\Delta)$: 已经排对且拉开的样本 $\\Delta$ 大, 权重接近 0, 自动退出; 力气集中在排错的那些上。' },
        { concept: '一步的开销', code: 'llm_finetune/methods/dpo.py:PairwiseForward', takeaway: '把 chosen 和 rejected 拼成 2B 条序列, policy 前向一次, 再加一次不带梯度的 ref 前向。常驻权重两份, 778 KiB。ref 不存激活, 所以实测每步只慢约 35%, 不是 100%。' },
        { concept: '偏好数据', code: 'llm_finetune/data/preference_data.py:PreferenceDataGenerator', takeaway: 'chosen = 正确回复, rejected = 改错一个 token 或漏掉一个。两边格式必须对称, 比如都带 EOS; 否则「只有一边带 EOS」这种捷径会被学走, 模型学的就不是偏好了。' },
        // ── KTO: 没有成对数据时的 DPO 替身 ──
        { concept: 'KTO 数据', code: 'llm_finetune/methods/kto.py:UnpairedDataGenerator', takeaway: '把偏好对拆开, 每个 prompt 只给一条回复 (好或坏), 所以 DPO 一对也凑不出。另附一份错配的 $(x_i, y_{i+1})$ (回复挪一行配给别的 prompt), 只用来估 $z_0$。' },
        { concept: 'KTO loss', code: 'llm_finetune/methods/kto.py:KTOLoss', takeaway: '- 好: $v = \\lambda_D\\,\\sigma(\\beta(r - z_0))$\n- 坏: $v = \\lambda_U\\,\\sigma(\\beta(z_0 - r))$\n- loss: $\\mathbb{E}[\\lambda_y - v]$\n这几个式子在说: 好样本的 $r$ 高过参考点越多 $v$ 越大, 坏样本的 $r$ 低过参考点越多 $v$ 越大, loss 要把 $v$ 推向它的上限 $\\lambda_y$。$r$ 与 DPO 是同一个 $\\log\\pi/\\pi_{\\text{ref}}$, 只是没乘 $\\beta$。' },
        { concept: '参考点 z0', code: 'kl_est.clamp(min=0).detach()', takeaway: 'kl_est 是错配样本上 $\\log\\pi/\\pi_{\\text{ref}}$ 的均值, clamp(min=0) 保证不为负, detach 不回传梯度。本例这个均值从 0 降到 −2.52, 被 clamp, $z_0$ 全程为 0。' },
        { concept: '前向复用', code: 'llm_finetune/methods/kto.py:KTOForward', takeaway: '继承 PairwiseForward, 两路前向分别是本条样本和错配样本, 再加一次冻结 ref 的前向。通用 Trainer 一行不改。' },
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
      subtitle: '读完你能说清 ref 在替你挡什么, 以及拿掉它之后得请谁来接班。',
      tldr: 'DPO 要常驻一份 ref (冻结的参照模型), 多占一份显存、每步多一次前向。SimPO 和 ORPO 把 ref 拿掉。\n实测 DPO 每步 2 次 LM 前向 (policy、ref 各 1 次) / 常驻权重 778 KiB / 200 步 5.4 s; SimPO 与 ORPO 1 次 / 389 KiB / 4.0 s。\n- SimPO: 奖励改用按长度平均的 log 概率, 再要求 chosen 至少领先一个目标间隔 $\\gamma$。\n- ORPO: 在 SFT 的 loss (chosen 上的 NLL, 也就是交叉熵) 上加一项 odds ratio 惩罚, 一个阶段同时学任务和偏好。',
      question: 'DPO 里的 ref 除了「别离 SFT 太远」, 还顺手解决了什么? 拿掉它谁来接手?',
      evolution: {
        title: 'ref 拿掉之后, 它的两份活谁来接',
        subtitle: '根问题: ref 多占一份显存、每步多一次前向, 但它同时在抵消长度红利和防止漂移',
        steps: [
          { name: 'DPO 的 ref', pain: '(原点) 偏好优化要一个锚, 否则 policy 想漂多远漂多远', fix: '冻结一份 SFT 副本当 ref: 常驻权重 ×2 (778 KiB), 每步多一次前向' },
          { name: '直接去掉 ref', pain: '一份权重、一次前向, 只为当锚', fix: '只用 policy 的 $\\sum\\log p$ 做差: 省下来了, 但 rejected 更长就白得 margin' },
          { name: 'SimPO', year: 2024, pain: '$\\sum\\log p$ 越长越负, 长度成了免费捷径', fix: '改用平均 log p 再减间隔 γ: 长度约掉, 但没有锚, log π(chosen) 掉到 −5.69' },
          { name: 'ORPO', year: 2024, pain: '没有锚, chosen 自己被一起压低, EM 只剩 0.023', fix: '拿 SFT 的 NLL 当锚, 再加 odds ratio 项: 一个阶段训完, EM 0.543' },
        ],
      },
      code: 'llm_finetune/methods/simpo.py · llm_finetune/methods/orpo.py · llm_finetune/methods/dpo.py',
      points: [
        { title: 'Σlog p 有长度红利', body: '结论: 用整条回复的 log 概率之和做奖励, 短回复天然占便宜; DPO 靠 ref 把这个偏差抵消掉。\n每个 token 的 log 概率都是负数, 加的项越多总和越负。所以序列越长, $\\sum\\log p$ 越负。\n不做校正的话, 只要 rejected 比 chosen 长, 它的分就天然更低, 差值天然是正的。模型什么偏好都没学, loss 已经接近 0, 这就是长度红利。\nDPO 怎么抵消: 它的奖励是 policy 减 ref。ref 对同一条长回复也给出同样低的分, 两者一减, 长度带来的那一截就消掉了。\n本仓库的偏好对长度最多差 1 个 token, 长度红利在这份数据上几乎没机会出现; 真实数据里回复长短差很多, 这个坑更大。' },
        { title: 'SimPO: 归一化 + γ', body: '结论: SimPO 用「平均每 token 的 log 概率」做奖励, 长度被除掉; 再用一个目标间隔 $\\gamma$ 代替 ref 画停手线。\n- 归一化: 奖励取 $(\\beta/|y|)\\cdot\\log\\pi(y|x)$, $|y|$ 是回复的 token 数。这个式子在说: 整条回复的 log 概率除以长度, 变成平均每个 token 的 log 概率。长回复和短回复在同一尺度上比, 长度红利没了。这个量也正好是生成时给候选打分常用的指标, 训练目标和推理指标对上了。\n- 代价: 平均之后数值范围小得多, 所以 $\\beta$ 要取 2 左右 (DPO 常用 0.1); 沿用 0.1 几乎推不动。\n- γ: loss 是 $-\\log\\sigma(r_c - r_r - \\gamma)$, 要求 chosen 的奖励至少比 rejected 高出 $\\gamma$ 才算完成。换成平均 log 概率的差, 就是至少领先 $\\gamma/\\beta$。没有 ref 告诉它「离起点多远该停」, 只能自己画这条线。本仓库 β=2, γ=1。' },
        {
          title: '省下的显存, 要用别的东西换',
          key: true,
          body: '结论: 三个方法的偏好准确率都上去了, 差别在 chosen 自己的概率和生成质量。拿掉 ref 之后, 谁来防止 chosen 被一起压低, 决定了结果。\n同一个 SFT 起点 (100 步, 留出集 EM 0.332)、同样的偏好数据、各 200 步, 留出集:\n- DPO (β=0.5): 偏好准确率 0.996, log π(chosen) −4.25, 贪心 EM 0.121。\n- SimPO (β=2, γ=1): 0.992, −5.69, EM 0.023。什么锚都没有, chosen 的概率被压得最狠, 掉得最惨。\n- ORPO (λ=0.5): 0.988, −2.51, EM 0.543。chosen 的概率反而升了。\nORPO 的锚是它 loss 里的 NLL 项: 就是 chosen 上的 SFT loss, 直接给 chosen 一股向上的梯度, DPO 和 SimPO 的 loss 里都没有这一项。\n所以 ORPO 不需要先做一遍 SFT: 从随机初始化训 300 步, EM 0.973, 与纯 SFT 打平, 还把 rejected 压得更低 (log π(rejected) −16.34, 纯 SFT 是 −14.85)。\n边界: 每个方法只跑了一组超参, 都没有调。SimPO 的 0.023 不能读成「SimPO 普遍比 DPO 差」。NLL 项也不是硬保证, λ 取大时 odds ratio 项仍可能把 chosen 拉低。',
        },
      ],
      links: [
        { from: 'DPOLoss', to: 'SimPO / ORPO', body: '三个 loss 共用同一份 (prompt, chosen, rejected) 数据、同一个 $-\\log\\sigma(z)$ 外壳。换的只是送进去的 z 怎么算: DPO 是带 ref 的求和 log 概率之差, SimPO 是平均 log 概率之差再减 γ, ORPO 是 log odds 之差。' },
        { from: 'compute_sequence_logprobs', to: '平均 log p', body: '同一个函数, 传 average=True 时把回复的 log 概率之和再除以回复的有效 label 数 (不含 prompt 和 pad)。SimPO 与 ORPO 都走这条路。' },
        { from: 'ref 的那次前向', to: '省掉', body: '先说「一次前向」怎么数: 一个模型把 chosen 和 rejected 拼成 2B 条序列跑一遍, 算 1 次 (B 是每步的偏好对数)。\n- DPO: policy 前向一次, ref 前向一次, 共 2 次。\n- SimPO / ORPO: 只有 policy 一次。\n实测每步快约 25%, 不是 50%: ref 那次前向不带梯度、不存激活, 本来就便宜; 带梯度的前向加反传还在, 省不到一半。' },
        { from: 'ORPO 的 NLL 项', to: 'SFT', body: 'ORPO 的 loss 是 nll_chosen 加 λ 乘 odds ratio 项。前一项就是 SFT 的 loss, 所以 ORPO 不需要先做一遍 SFT; 它同时充当防漂移的锚。' },
      ],
      sourceRows: [
        { concept: 'DPO 的 z', code: 'llm_finetune/methods/dpo.py:DPOLoss', takeaway: '$\\beta\\cdot[(\\pi_c - \\mathrm{ref}_c) - (\\pi_r - \\mathrm{ref}_r)]$, 括号里都是整条回复的 log 概率。这个式子在说: policy 和 ref 各自的 log 概率相减, chosen 一份, rejected 一份, 再作差。ref 同样「越长越负」, 长度项在相减时成对抵消。' },
        { concept: 'SimPO 的 z', code: 'llm_finetune/methods/simpo.py:SimPOLoss', takeaway: '$\\beta\\cdot(\\log p_c/|y_c| - \\log p_r/|y_r|) - \\gamma$。这个式子在说: 两条回复各自的平均每 token log 概率之差, 乘 β, 再减掉目标间隔。本仓库默认 $\\beta=2$、$\\gamma=1$。沿用 DPO 的 $\\beta=0.1$ 会几乎推不动, 因为平均值的数值范围小得多。' },
        { concept: 'γ = 0 且不归一化', code: 'z = logp_c − logp_r', takeaway: '把 γ 去掉、长度也不除, z 就只剩两条回复 log 概率之和的差。这等于 $\\pi_{\\text{ref}}$ 取均匀分布的 DPO: 没有 ref 做减法, 长度红利全回来了。' },
        { concept: 'ORPO 的 odds', code: 'log(p/(1−p)), p = exp(mean logp)', takeaway: 'odds 是「发生的概率 / 不发生的概率」, 这里 p 取 exp(平均每 token log 概率), 同样与长度无关。$p\\to 1$ 时 $\\log(1-p)$ 趋于负无穷, 代码里对 p 做 clamp, 否则算出 $\\log 0$。' },
        { concept: 'ORPO 总 loss', code: 'llm_finetune/methods/orpo.py:ORPOLoss', takeaway: 'nll_chosen + λ·(−logsigmoid(log_or)): 第一项是 chosen 上的 SFT loss, 第二项要求 chosen 的 log odds 高过 rejected。论文 $\\lambda=0.1$, 本仓库玩具任务默认 0.5。odds 在 $p\\to 1$ 处发散, 所以对「两边都已经很自信」时的差距比概率比更敏感, 概率比在那里趋于平坦。' },
      ],
      snippetTitle: '三种 z 并排',
      snippet: `lp_c, lp_r = seq_logprob(policy, chosen), seq_logprob(policy, rejected)   # [B] 求和
n_c, n_r   = len(chosen_resp), len(rejected_resp)

# DPO: 多一次 ref 前向 (chosen + rejected 拼成 2B 条, 一次跑完)
z_dpo  = beta * ((lp_c - ref_c) - (lp_r - ref_r))

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
      widgets: ['GrpoLab'],
      title: 'RM · GRPO · 蒸馏 — 从偏好到能力迁移',
      subtitle: '读完你能回答三件事: 标注有噪声时 RM 最多学到多大分差、GRPO 的 baseline 从哪来、蒸馏的 KL 项为什么要乘 $T^2$。',
      tldr: '- RM (reward model, 奖励模型): 把人标的「A 比 B 好」学成一个能随时调用的标量分。留出集偏好准确率从 0.549 升到 0.930。\n- GRPO: 在线强化学习里, PPO 要另训一个 critic 网络来估「这道题平均能得几分」, 当 baseline (参照分) 用。GRPO 用同一道题 G 条回复的组内均值顶替它, 把 critic 整个省掉。\n- 蒸馏: 让小模型 (student) 学大模型 (teacher) 用温度软化过的输出分布, 比只学一个正确 token 的硬标签信息密得多。',
      question: 'GRPO 砍掉 critic 之后 baseline 从哪来? 标注有 20% 概率标反时, RM 学到的分差会停在哪?',
      evolution: {
        title: '奖励从哪来, baseline 从哪来',
        subtitle: '根问题: 在线 RL 每采一条回复都要一个标量分, 还得知道这道题 "平均能得几分"',
        steps: [
          { name: '人打绝对分', pain: '(原点) RL 每采一条都要一个分, 最直接是请人打', fix: '人给每条回复打分: 各人尺度不同, 分数彼此对不上, 还跟不上采样速度' },
          { name: '排序 + RM', pain: '绝对分不稳, RL 每采一条又不能停下来等人', fix: '人只给回复排序 (InstructGPT 一次 4–9 条), 用 Bradley-Terry 模型 (把「A 比 B 好」的概率写成分差的 sigmoid) 学成随时可调的分' },
          { name: 'critic 当 baseline', pain: '拿 RM 分直接当权重, 题目难易不同, 梯度方差大', fix: 'PPO 另训一个 critic 估 "这题平均几分" 并减掉: 又多一个同尺寸网络' },
          { name: 'GRPO', year: 2024, pain: 'critic 与 policy 同尺寸, 要训要存, 还不一定学得准', fix: '同题采 G 条, 组内均值当 baseline, 省掉 critic; 全对或全错的组没有梯度' },
        ],
      },
      code: 'llm_finetune/methods/reward_model.py · llm_finetune/methods/grpo.py · llm_finetune/methods/distill.py',
      points: [
        {
          title: 'RM 只学分差',
          key: true,
          body: '结论: 奖励模型学到的只是两条回复之间的分差, 分数本身没有绝对意义; 标注有噪声时, 分差会停在一个有限的值上。\nRM 的结构: LM 主干读完 (prompt, 回复), 取最后一个真 token 的隐状态, 接一个输出 1 个数的线性头 (value head), 这个数就是分 $r$。\n训练用 Bradley-Terry 模型: $P(A\\succ B) = \\sigma(r_A - r_B)$, loss $= -\\log\\sigma(r_w - r_l)$, 其中 $w$ / $l$ 是 chosen / rejected。这两个式子在说: 「A 比 B 好」的概率只由分差决定, 分差越大概率越接近 1; 训练就是让 chosen 的分高过 rejected。\n给所有分数加同一个常数, 分差不变, loss 一点不变。所以分数没有量纲, 只有顺序。\n标注有噪声时分差停在哪: 假设每条标签有 $\\varepsilon$ 的概率标反。分差 $\\Delta$ 拉得越大, 标对的那部分 loss 越小, 但标反的那部分 loss 越大。期望 loss 的零梯度点落在 $\\sigma(\\Delta) = 1-\\varepsilon$, 解出来最优分差是 $\\ln((1-\\varepsilon)/\\varepsilon)$: $\\varepsilon=0.2$ 时约 1.39, $\\varepsilon=0.5$ (标签完全随机) 时为 0, 什么分差都学不到。\n本仓库实测: 第 1 步 loss 0.6929 (两边分数还没拉开, 约等于 ln 2), 300 步后留出集偏好准确率从 0.549 到 0.930。',
        },
        { title: '组内排名顶替 critic', body: '结论: GRPO 用同一道题的 G 条回复互相当参照, 省掉了 PPO 的 critic 网络。\n先说为什么要 baseline。策略梯度把每条回复的奖励当权重: 分高的回复, 把它的概率抬上去。但题目有难有易, 难题上得 0.5 分可能已经很好, 易题上得 0.5 分可能很差。所以要减掉一个「这道题平均能得几分」的参照分, 剩下的才是这条回复相对的好坏, 叫优势 $\\hat{A}$。PPO 用一个单独训练的 critic 网络 (也叫 value 网络) 来预测这个参照分。\nGRPO 的做法: 同一个 prompt 采 G 条回复, $\\hat{A} = (r - \\mu)/\\sigma$。这个式子在说: 每条回复的奖励减去组内均值 $\\mu$, 再除以组内标准差 $\\sigma$。组内均值就是「这道题的期望得分」的蒙特卡洛估计 (多采几次取平均), 正是 critic 想预测的东西, 但不用训。\n代价: 一组全对或全错时, 每条回复的奖励都等于均值, $\\hat{A}$ 全为 0, 这批样本不产生任何梯度。实测每步约 32% 的 prompt 落在这种情况里。' },
        { title: '暗知识在分布里', body: '结论: 蒸馏 (knowledge distillation, 用大模型 teacher 教小模型 student) 让 student 学 teacher 的整个输出分布, 而不只是一个正确 token; 蒸馏项要乘 $T^2$, 温度和权重才互不干扰。\n- 硬标签: 每个位置只告诉 student 一个正确 token, 别的 token 「有多合理」全丢了。\n- 软标签: teacher 在每个位置输出的是整个词表上的概率分布。用温度 $T$ 软化, 也就是把 logits (模型输出的原始分数) 除以 $T$ 再做 softmax, 分布变平, 第二、第三名的 token 也能看出来。这些「相似 token 的相对排序」就是暗知识。\n为什么乘 $T^2$: softmax(z/T) 对 logits 的梯度自带 $1/T^2$ 的衰减。不补回来的话, 调高温度 $T$ 就等于偷偷把蒸馏项的权重调小, 相当于关掉蒸馏。乘 $T^2$ 之后, 软标签项和硬标签项的相对权重才只由 $\\alpha$ 决定。\n实测 (teacher 学的任务有两种正确答案, 70% 排序 / 30% 照抄): 只有 128 条固定数据时, 软标签明显更好, 留出集 forward KL (在 teacher 的样本上量 student 离 teacher 多远, 越小越近) 从 2.293 降到 1.860。数据无限时硬标签自己就能把分布采出来, 两者都卡在 student 的容量上, 优势消失 (0.906 对 0.960)。' },
      ],
      links: [
        { from: 'PreferenceDataGenerator', to: 'RewardModel', body: '同一份偏好数据 (prompt, chosen, rejected), 两种用法: RM 学打分, DPO 直接学策略。' },
        { from: 'RM 分数', to: 'GRPO 的 reward_fn', body: 'GRPO 只要求一个函数 reward_fn(prompts, completions), 给每条回复一个分。这个函数既可以是 RM, 也可以是程序判分的 verifier (比如比对答案), 后者就是 RLVR (见「看题的可验证奖励」一章)。' },
        { from: 'policy.generate', to: '组内优势', body: 'GRPO 的训练数据是 policy 自己现场采的, policy 一更新, 下一步采出来的数据分布就变了, 这叫 on-policy。通用 Trainer 的约定是「取一个 batch, 更新一次」, 数据生成器拿不到模型, 所以 GRPO 接不进去。' },
        { from: 'teacher logits / T', to: 'student KL', body: '$T$ 把 teacher 的分布摊平, 放大暗知识; $T^2$ 补回 softmax 梯度的 $1/T^2$ 缩放。这样软标签项和硬标签项的相对权重才只由 $\\alpha$ 决定, 不随 $T$ 变。' },
      ],
      sourceRows: [
        { concept: 'value head', code: 'llm_finetune/methods/reward_model.py:RewardModel', takeaway: '复用 LLaMA 主干, 调 forward(..., return_hidden=True) 拿到最后一层 norm (ln_f) 之后的隐状态, 再接一个 [D] 到 [1] 的线性头。如果手抄一遍主干的前向来拿隐状态, 主干一改 (mask / RoPE / cache) 就会悄悄不一致。' },
        { concept: '读哪一个位置', code: 'attention_mask.sum(1) - 1', takeaway: '右 pad (短序列末尾补 PAD) 时, 分要取最后一个真 token 的位置, 下标是 attention_mask.sum(1) − 1。直接取 [:, -1] 读到的是 pad 位置: 同一序列多垫 5 位 PAD, 取真 token 的分数偏移只有 5e-06, 取 [:, -1] 偏移 4.54。' },
        { concept: 'BT loss', code: 'llm_finetune/methods/reward_model.py:BradleyTerryLoss', takeaway: '就是 $-\\log\\sigma(r_w - r_l)$。第 1 步 loss $0.6929 \\approx \\ln 2$: value head 小初始化, 两边分数几乎相等, $\\sigma(0) = 0.5$。' },
        { concept: '组内优势', code: 'llm_finetune/methods/grpo.py:group_advantages', takeaway: 'adv = (r − mean) / (std + eps), 在组内算。组内奖励全相同时分子分母都是 0, adv 全为 0, 这组白占 batch。DAPO 的动态采样就是冲它去的 (见 GRPO 进阶)。' },
        { concept: '重要性比率', code: 'llm_finetune/methods/grpo.py:GRPOTrainer', takeaway: 'GRPOTrainer 把一批 rollout (采样出的回复) 用 $\\mu=2$ 次。第 1 次更新前参数没动, 新旧策略的概率比 $\\rho\\equiv 1$ (实测 $|\\rho-1| = 0.000$); 第 2 次起参数变了, $\\rho\\ne 1$, clip 才开始起作用。' },
        { concept: 'KL 的 k3 估计', code: 'd.exp() - d - 1', takeaway: 'd 是 ref 和 policy 的 log 概率之差, d.exp() − d − 1 是 KL 散度的单样本估计 (Schulman 叫它 k3): 逐 token 算, 恒 $\\ge 0$, 方差比直接用 log-ratio 小。只有 $\\beta>0$ 时才算这一项, 也才需要常驻一份 ref。' },
        { concept: '蒸馏损失', code: 'llm_finetune/methods/distill.py:DistillLoss', takeaway: '$\\alpha\\cdot\\mathrm{CE} + (1-\\alpha)\\cdot T^2\\cdot\\mathrm{KL}(p_t^T \\,\\|\\, p_s^T)$。这个式子在说: 硬标签的交叉熵和「teacher 软分布到 student 软分布」的 KL 按 $\\alpha$ 加权, KL 项乘 $T^2$。KD 项和 CE 项必须用同一个 mask (label 为 -100 的位置都不算), 否则 prompt 和 pad 位置也在被蒸馏, 两项优化的不是同一批位置。' },
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
      subtitle: '读完你能指出三个变体各自在修的那一处偏置, 也不会把四条噪声之内的最终分数读成「谁更强」。',
      tldr: 'GRPO (同一道题采 G 条回复、用组内均值当参照的在线 RL) 里, 一批采样只更新一次的话, 新旧策略完全相同, 概率比 $\\rho\\equiv 1$, clip 是死代码。\n想让昂贵的 rollout (采样出的回复) 多用几轮, 第 2 轮起策略已经变了, 就得引入重要性比率 $\\rho = \\pi_{\\text{new}}/\\pi_{\\text{old}}$ 纠偏, 再用裁剪 (clip) 防止一步走太远。三个变体分别修这个目标里的三处偏置: DAPO 修熵塌缩, Dr.GRPO 修长度和难度偏置, GSPO 修比率噪声。',
      question: '同一条"又长又错"的回答, 在 GRPO / Dr.GRPO / DAPO 下每个 token 分到的惩罚分别是多少?\n为什么上界 clip 只卡得住低概率 token?',
      evolution: {
        title: '一批 rollout 多用几遍, 偏置一处处修',
        subtitle: '根问题: 采样是 RL 最贵的一步, 想让同一批 rollout 多更新几次, 就得处理新旧策略不一致',
        steps: [
          { name: '一批更新一次', pain: '(原点) 策略梯度只对采样时的那个策略成立', fix: '采一批只更新一次就扔: $\\rho\\equiv 1$, 干净, 但每步都要等一轮采样' },
          { name: 'PPO 的 ratio + clip', year: 2017, pain: '同一批多更新几轮, 策略已经变了, 梯度有偏', fix: '乘 $\\rho=\\pi_{\\text{new}}/\\pi_{\\text{old}}$ 纠偏, 越出 $[1-\\epsilon, 1+\\epsilon]$ 就停手' },
          { name: 'DAPO', year: 2025, pain: '上界只卡得住低概率 token, 想翻身的探索 token 被压住, 熵塌缩', fix: '只放宽上界 0.2 → 0.28, 丢掉全对全错的组, loss 改按 token 平均' },
          { name: 'Dr.GRPO', year: 2025, pain: 'loss 按 token 平均了, 优势里的 ÷σ 还在: σ 小的题权重被放大', fix: '去掉 ÷σ, 长度分母换常数: 难题相对中等题的权重 0.66 → 0.44' },
          { name: 'GSPO', year: 2025, pain: '分母修完了, token 级 $\\rho_t$ 仍是单样本噪声, 奖励却给整条', fix: '整条回复共用几何平均的 ρ 一起裁剪: std log ρ 0.067 → 0.027, 区间收到 0.05' },
        ],
      },
      code: 'llm_finetune/methods/grpo.py · llm_finetune/run_finetune/grpo/train_grpo.py',
      points: [
        { title: 'clip 是停手线, 不是刹车', body: '结论: clip 不是把更新变慢, 而是越界后直接没有梯度; 而且上界只卡得住低概率 token, 所以 DAPO 单独放宽上界。\n先说目标: $\\min(\\rho\\hat{A}, \\mathrm{clip}(\\rho)\\hat{A})$, 其中 $\\rho$ 是新策略和采样时的旧策略对同一个 token 的概率比, $\\hat{A}$ 是优势 (这条回复比组内平均好多少)。这个式子在说: 优势为正时, $\\rho$ 涨过上界 $1+\\epsilon$ 就按上界算, 再涨也没收益; 优势为负时, $\\rho$ 跌过下界同理。越界那一侧是平的, 梯度为 0, 所以是停手线。\n为什么上界只卡低概率 token: $\\rho$ 最大只能到 $1/\\pi_{\\text{old}}$, 因为新概率最多是 1。$\\pi_{\\text{old}}=0.9$ 的 token 顶天涨到 1.11, 碰不到 1.2 的上界, 它想涨多少涨多少。被卡住的全是 $\\pi_{\\text{old}}$ 很小的 token, 恰恰是想翻身的探索 token。它们被压住, 分布越来越尖, 不再探索, 这叫熵塌缩。\n所以 DAPO 的 clip-higher 只放宽上界: 下界 $\\epsilon_{\\text{low}}$ 仍是 0.2, 上界 $\\epsilon_{\\text{high}}$ 放到 0.28。' },
        {
          title: '两处偏置藏在分母里',
          key: true,
          body: '结论: GRPO 的 loss 里有两个分母, 一个让长回复受罚更轻, 一个让极端难易的题权重偏高; Dr.GRPO 把两个都换成常数。\n- 1/|o_i|: GRPO 先在每条回复内按长度 $|o_i|$ 平均, 再跨回复平均。这样一条回复的总权重固定, 回复越长, 每个 token 分到的权重越小。答错的长回复每个 token 罚得更轻。实测 3-token 回复的单 token 权重是 9-token 的 3.0 倍, 等于鼓励「错就错得长一点」。\n- ÷σ: 优势是组内奖励减均值再除以组内标准差 $\\sigma$。$\\sigma$ 越小, 除出来的优势越大。几乎全对或几乎全错的题, 组内奖励差异小, $\\sigma$ 小, 优势被放大, 于是这些没什么可学的极端题权重偏高。\n÷σ 的实测: 难题 (8 条对 1 条) 相对中等题 (8 条对 4 条) 的权重, GRPO 0.66, 去掉 $\\sigma$ 的 Dr.GRPO 0.44。\nDr.GRPO 的改法: 不除 σ; 长度分母换成固定常数 C (回复的最大 token 数), 每个 token 的权重和它所在回复的长度无关。',
        },
        { title: '比率的粒度', body: '结论: 奖励是整条回复给的, 而 $\\rho$ 却是逐 token 算的, 噪声大; GSPO 改成整条回复共用一个比率, 噪声小了, 裁剪区间也要跟着收窄。\n每个 token 的 $\\rho_t$ 是单样本估计, 一条回复里有的 token 涨有的跌, 本身很抖。但奖励和优势都是整条回复一个数, 逐 token 裁剪和它对不上。\nGSPO 的做法: 取序列级比率 $s = \\exp(\\mathrm{mean}_t \\log\\rho_t)$。这个式子在说: 把回复里每个 token 的 log 比率取平均再指数回去, 也就是所有 token 比率的几何平均。整条回复共用这一个权重一起裁剪。\n实测 $\\mathrm{std}\\,\\log\\rho$ 从 0.067 降到 0.027, 约小 $\\sqrt{C}$ 倍 ($C$ 是回复的 token 数, 平均能把波动压下这么多)。\n所以它的裁剪区间也要跟着收窄: 本仓库取 0.05 (GRPO 是 0.2), 论文用的是 3e-4 量级。' },
      ],
      links: [
        { from: 'GRPOTrainer.step', to: '多轮内层更新', body: '一步的流程: 采样一次, 用采样时的参数算出 logp_old 并冻结, 然后对同一批做 $\\mu$ 次更新。实测第 1 个 epoch $|\\rho-1| = 0.000$ (参数还没动), 第 2 个 epoch 才偏离。' },
        { from: 'PPO clipped surrogate', to: 'GRPO', body: '目标函数外形完全一样: 概率比乘优势再 clip。GRPO 只换了优势的来源, 从 critic 换成组内归一化。' },
        { from: '组内奖励全相同', to: 'DAPO 动态采样', body: '一组回复全对或全错时 $\\hat{A}$ 全 0, 不贡献梯度, 但仍占着 loss 的分母。实测这类 prompt 占 32%。\n- DAPO: 把它们从 loss 里去掉, 进 loss 的 A=0 样本从 32.1% 降到 0.0%。\n- 本实现: 只过滤不补采, batch 会变小。DAPO 原版会继续采到凑满。' },
        { from: '采样温度 T', to: 'log-prob 的 T', body: '采样时用的分布是 $\\pi^{1/T}$ (logits 除以 T 再 softmax)。算 $\\rho$ 的分母 logp_old 时也必须用同一个 $T$, 否则分母不是真正用来采样的那个分布 (行为策略), 第 1 个 epoch 就已经 off-policy。脚本用 T=0.5 采 4000 个首 token 验过: 经验分布和 softmax(z/T) 的距离 0.000, 和 softmax(z) 的 0.039。' },
      ],
      sourceRows: [
        { concept: '四个变体 = 五个开关', code: 'llm_finetune/methods/grpo.py:GRPOConfig', takeaway: 'GRPOConfig 里五个开关: clip_high (上界) / std_norm (是否除 σ) / loss_agg (按什么平均) / seq_ratio (是否序列级比率) / dynamic_sampling (是否丢掉全同组)。VARIANTS 字典把 grpo · dapo · dr_grpo · gspo 写成开关组合。' },
        { concept: '组内优势', code: 'llm_finetune/methods/grpo.py:group_advantages', takeaway: '算 $r - \\mu$ (奖励减组内均值); std_norm=True 时再除 $\\sigma$。' },
        { concept: '重要性比率', code: 'ratio = exp(logp_new − logp_old.detach())', takeaway: '在 log 域相减再 exp, 不直接除概率。logp_old 在采样后立刻用同一温度算好并冻结。' },
        { concept: 'loss 聚合', code: 'llm_finetune/methods/grpo.py:aggregate', takeaway: 'aggregate 的三种写法: seq_mean (GRPO, 每条回复内 $\\div |o_i|$ 再跨回复平均) · token_mean (DAPO, 全体 token 一起平均) · fixed_len (Dr.GRPO, $\\div$ 常数 $C$)。差别只在每个 token 的权重。' },
        { concept: '别误读最终分数', code: 'run_finetune/grpo/README.md', takeaway: '60 步后留出集采样 pass@1: grpo 0.287 / dapo 0.326 / dr_grpo 0.324 / gspo 0.320, 差异在噪声内, 只有一个种子。贪心 EM 基本没动 (从 0.562 到 0.53)。RLVR 主要把 pass@k (多采几次里有一次对) 挤进 pass@1 (采一次就对), 不是教新能力。' },
        { concept: '用什么验收', code: '采样 pass@1, 不是贪心 EM', takeaway: 'RL 优化的是采样分布, 所以要用采样 pass@1 验收。拿贪心解码 (每步取最大概率) 的 EM 去验收 RL, 什么都看不到。' },
      ],
      snippetTitle: '带 ratio + clip 的 GRPO 内层循环',
      snippet: `seqs, mask = sample_group(policy, prompts, G)
r    = reward_fn(prompts, seqs)                       # [B, G]
adv  = r - r.mean(1, keepdim=True)                    # Dr.GRPO 到此为止
adv  = adv / (r.std(1, keepdim=True) + 1e-4)          # GRPO / DAPO 再除 σ
keep = r.std(1) > 0                                   # DAPO: 全对/全错的组丢弃 (本实现只过滤不补采, batch 会变小)
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
      subtitle: '读完你能用「KL 写在哪一侧」解释两种蒸馏的性格差异, 也能替容量不够的学生挑出该用哪一种。',
      tldr: '蒸馏是让小模型 student (学生) 去模仿大模型 teacher (老师) 的输出分布。两种蒸馏的区别在于学生在谁写的前缀上学, 以及 KL (衡量两个分布差多远的量) 写在哪一侧:\n- 离线蒸馏: 在老师或数据集写好的前缀上教学生。用 forward KL (正向 KL, 在老师的分布上取平均), 它逼学生把老师的每种答法都留一点概率, 摊开盖住所有峰 (峰是老师分布里概率集中的一种答法)。\n- on-policy 蒸馏: 学生先自己生成回复, 老师在学生走到的每个位置上给出分布, 学生去对齐。用 reverse KL (反向 KL, 在学生自己的分布上取平均), 学生会钻进一个峰, 把一种答法做好。\n实测: 学生采样回复的合格率从 0.059 (离线) 升到 0.402 (on-policy); 代价是老师那种 30% 概率的少数派答法被彻底放弃。',
      question: '容量不够的学生, 该摊开盖住老师的所有答法, 还是挑一种答到位?\n这跟 KL 写在哪一侧有什么关系?',
      evolution: {
        title: '从抄老师的卷子到老师批自己的卷子',
        subtitle: '根问题: 小模型装不下老师的全部答法, 又只在老师的前缀上学过, 自己一走偏就没人教过',
        steps: [
          { name: '硬标签', pain: '(原点) 小模型学大模型, 最简单是抄它写出的答案', fix: '在老师写出的答案上做 CE (交叉熵, 普通的监督训练): 每个位置只学到一个 token, 「别的答案多合理」全丢了' },
          { name: '软标签 KD', year: 2015, pain: '硬标签只给一个 token, 相似答案的相对排序没交出去', fix: '用温度 T 调平的老师分布当目标做 forward KL, 再乘 $T^2$ 补回被温度缩小的梯度: 数据少时 KL 从 2.293 降到 1.860' },
          { name: '学生自己采样', year: 2023, pain: '只在老师的前缀上学过, 推理时走进自己的前缀, 错误一步步累积', fix: '学生先写, 老师在它走到的每个位置给分布; 采样不回传梯度 (GKD 的做法)' },
          { name: '逐 token reverse KL', pain: '容量不够时, forward KL 逼学生摊开盖住所有答法, 采样半路串台', fix: '改用 reverse KL (在学生自己的样本上取期望), 钻进一个峰: 合格率 0.402 (离线 0.059), 少数派答法被丢' },
        ],
      },
      code: 'llm_finetune/methods/on_policy_distill.py · llm_finetune/methods/distill.py',
      points: [
        {
          title: 'KL 的方向决定性格',
          key: true,
          body: '结论: KL 写成 KL(老师‖学生) 还是 KL(学生‖老师), 决定了容量不够的学生是「每样都学一点」还是「挑一样学好」。\n先看公式: $\\mathrm{KL}(p\\,\\|\\,q) = \\sum p\\cdot\\log(p/q)$。这个式子在说: 对每个可能的 token, 看 p 和 q 给它的概率之比取 log, 再按 p 给它的概率加权求和。所以写在前面的 p 决定「在哪些地方算差距」: p 认为不可能的地方权重是 0, q 在那里怎么放概率都不算。\n本页的例子: 老师学的任务有两种正确答法, 70% 概率把 prompt 排序 (sort), 30% 概率原样照抄 (copy)。老师的分布有两个峰。学生比老师小 (1 层对 2 层), 装不下两个峰。\n- forward KL, 老师在前: 在老师的样本上取期望。老师给了概率而学生几乎没给的地方, $\\log(p/q)$ 趋于无穷, 惩罚极大。学生被迫两个峰都盖住, 代价是往两峰之间的低谷里也放了概率。采样时一条回复会半路串台, 写着写着从一种答法跳到另一种, 两头都不对。\n- reverse KL, 学生在前: 在学生自己的样本上取期望。学生不去的地方权重为 0, 漏掉老师的一个峰不受罚; 学生在老师认为不可能的地方放概率, 却会被重罚。于是它缩进一个峰, 把这一种答法做好。',
        },
        { title: '离线蒸馏没走过学生的路', body: '结论: 离线蒸馏只在别人写的前缀上教过学生, 学生自己写时会走进从没被教过的状态。on-policy 蒸馏直接在学生自己的前缀上教。\n前缀是回复里已经写出来的那部分 token, 模型看着它决定下一个 token 写什么。\n- 离线蒸馏: 训练时的前缀都来自老师或数据集。推理时是学生自己往下写, 一旦某一步写偏, 后面的前缀就是训练里从没出现过的状态。学生在这种状态下没被纠正过, 错误一步步累积。这个问题叫 exposure bias。\n- on-policy: 训练用的前缀就是学生自己采样出来的, 训练时见到的分布就是推理时会走到的分布。名字里的 on-policy 说的就是这个: 训练数据来自正在训练的这个模型本身。' },
        { title: '每个 token 都有信号, 但丢多样性', body: '结论: on-policy 蒸馏像 RL 一样在学生自己的样本上训练, 但每个 token 都拿到老师的完整分布当信号, 比 RL 密得多。代价是丢掉老师的少数派答法。\n信号量对比:\n- RL (比如 GRPO): 一整条回复只换来 1 个标量奖励, 要靠 REINFORCE 这类策略梯度估计 (拿整条回复的奖励去加权每个 token 的 log 概率梯度) 去推每个 token 该怎么改。\n- on-policy 蒸馏: 每个 token 位置都有老师的完整 $V$ 维分布 (V 是词表大小, 每个候选 token 一个概率, 本例 V=16)。KL 直接在整个词表上求和算出来 (解析求 KL), 不用采样估计, 也不用 REINFORCE。\n实测: 同一个老师、同一个热身后的学生, 两种方法各再训 300 步, 在留出 prompt 上量:\n- 样本合格率 (学生采样的回复正好是 sort 或 copy 两种正确答案之一的比例): on-policy 0.402, 离线 0.059。\n- reverse KL: 0.50 vs 2.07。\n- 多数派 sort 答案的 $\\log\\pi$ (学生给整条答案的 log 概率): on-policy −1.34, 离线 −3.30。\n代价: 老师给 30% 概率的少数派答法 (copy), 在 on-policy 学生里 $\\log\\pi$ 被压到 −33, 离线只到 −17。−33 意味着学生几乎不可能再写出它。forward KL 两头都留着概率, 结果两头都说不利索。' },
      ],
      links: [
        { from: 'DistillLoss (离线)', to: 'on-policy 蒸馏', body: '左边是离线蒸馏的 loss, 右边是 on-policy 蒸馏。用温度 T 软化分布、乘 $T^2$ 补偿梯度, 这两招在 on-policy 里仍然适用。变的只有两样: 样本从哪来 (固定语料, 还是学生现场采), KL 写在哪一侧 (老师在前, 还是学生在前)。' },
        { from: 'student.generate', to: 'teacher 打分', body: '左边是学生采样, 右边是老师打分。采样和 GRPO 共用同一套生成管线, 区别是 GRPO 把回复交给 reward_fn (打分函数) 打一个总分, 这里换成老师在每个位置给出的完整分布。采样在 no_grad 下跑 (不记录计算图), 不回传梯度, 它只决定「在哪些状态上算 KL」。' },
        { from: 'reverse KL', to: 'GRPO 的 KL 惩罚', body: 'GRPO 用一项 KL 惩罚把策略锚在 ref (冻结的起点模型) 附近, 那一项就是同一个量: $\\mathbb{E}_{y\\sim\\pi}[\\log\\pi - \\log\\pi_{\\text{ref}}]$。这个式子在说: 在当前模型自己采的样本上, 看它比 ref 多给了多少 log 概率。和这里一样, 期望在正在训练的模型自己的分布上取, 只是另一边从老师换成了 ref。' },
        { from: '热身', to: 'on-policy', body: '从随机初始化直接上 on-policy 没用: 学生采的全是垃圾, 老师在这些从没见过的 (分布外的) 前缀上给的分布也没意义。所以先用 SFT (在示范回答上做监督微调) 或离线蒸馏热身 200 步; 脚本里是用 forward KL 离线蒸馏 200 步, 两条对照都从这个热身后的学生出发。' },
      ],
      sourceRows: [
        { concept: '离线基线', code: 'llm_finetune/methods/distill.py:DistillLoss', takeaway: '$\\alpha\\cdot\\mathrm{CE} + (1-\\alpha)\\cdot T^2\\cdot\\mathrm{KL}(p_t^T \\,\\|\\, p_s^T)$。这个式子在说: 一部分学硬标签 (CE, 权重 α), 一部分学老师在温度 T 下的整个分布 (KL, 权重 1−α, 再乘 $T^2$)。老师 $p_t$ 在前, 所以这是 forward KL, 数据来自固定语料。本页的对照臂取 T=1、α=0, 只剩纯 KL, 才能和 reverse KL 对称地比。' },
        { concept: '学生采样', code: 'llm_finetune/methods/on_policy_distill.py:on_policy_distill_step', takeaway: '先用 generate 在 no_grad 下采样 (温度 1), 拿到普通张量。再让学生和老师在这同一批回复上各前向一次, 学生带梯度, 老师不带。梯度只来自这次带梯度的前向。采到的前缀被当成常数, 这是 GKD 论文的做法, 梯度有偏但方差小; MiniLLM 论文会保留「采到哪些前缀也随参数变」那一项。' },
        { concept: '逐 token reverse KL', code: 'llm_finetune/methods/on_policy_distill.py:token_kl', takeaway: '$\\sum_v q_s(v)\\cdot(\\log q_s(v) - \\log p_t(v))$。这个式子在说: 对词表里每个 token v, 用学生给它的概率 $q_s(v)$ 加权, 累加学生和老师的 log 概率差。学生在前, 所以是 reverse KL。在完整词表上精确求和, 不需要对 KL 做采样估计。loss 再对全批的有效回复 token 一起平均 (EOS 之后的不算)。' },
        { concept: '各赢各的', code: 'run_finetune/on_policy_distill/README.md', takeaway: 'off-policy 臂 forward KL 0.88 / reverse 2.07; on-policy 臂 forward 1.36 / reverse 0.50。forward KL 在老师的样本上量, reverse KL 在学生自己的样本上量。谁优化哪个指标就在哪个指标上赢, 别只看一栏。' },
        { concept: '为什么不接 Trainer', code: 'run_finetune/on_policy_distill/train_on_policy_distill.py', takeaway: '通用 Trainer 的约定是「取一个现成的 batch, 更新一次」。这里训练数据由当前 student 现场生成, 数据生成器得拿到模型本身, 和 GRPO 一个原因, 所以脚本里是手写循环。' },
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
      tldr: 'RLVR (RL with verifiable rewards, 用程序判分代替奖励模型的强化学习) 的奖励必须是 (prompt, completion) 两者的函数。\n- 区域奖励 (如「输出落在词表后半区就给分」) 只看 completion 不看 prompt: 最优策略是无视 prompt 的常数输出。reward 能涨到 1.0, 却什么都没证明。\n- 「答案 = f(prompt)」的奖励: 常数策略的上限掉回 1/类别数, 超过它才说明模型在读题。',
      question: '一条一路涨到 1.0 的 reward 曲线, 怎么分辨「学会按题作答」和「只是学会了偏爱某几个 token (unigram 偏好)」?',
      evolution: {
        title: '奖励先过常数基线这一关',
        subtitle: '根问题: RL 会沿奖励最省力的方向走; 奖励不依赖题目时, 最省力的就是不看题',
        steps: [
          { name: 'RM 打分', pain: '(原点) RL 要奖励, 最通用的是训一个 RM 给回复打分', fix: 'RM 什么都能打分, 但它是学出来的, 策略会钻它的盲区拿高分' },
          { name: 'RLVR', year: 2024, pain: 'RM 的高分不等于答对, 策略在优化 RM 的漏洞', fix: '把 RM 换成程序判分 (Tülu 3 起的名): 数学比答案、代码跑单测' },
          { name: '只看输出的规则', pain: '程序判分不会被讨好, 但没人保证它看了题', fix: '区域奖励 "落在后半区就给分": 常数输出就能拿满分, 曲线涨到 1.0 却什么都没学' },
          { name: '看题的 verify', pain: '奖励与 prompt 无关时, 常数策略就是最优解', fix: '奖励写成 f(prompt, completion), 训前先算常数基线: 看题任务只有 1/类别数' },
        ],
      },
      code: 'llm_finetune/data/tasks.py · llm_finetune/data/prompt_data.py · llm_finetune/methods/grpo.py',
      points: [
        {
          title: '先算常数基线',
          key: true,
          body: '结论: 跑 RL 之前, 先算「不看题能拿到的最高分」, 它是「没学会」的参考线。\n常数基线的定义: 让所有 prompt 都输出同一个 completion, 能拿到的最高平均奖励。\n- 基线等于 1.0: 不看题就能拿满分, 这个奖励函数验证不了任何「按题作答」的行为。模型就算拿到 1.0, 也可能只是找到了那个常数输出。\n- 看题的任务: 答案由 prompt 决定, 一个固定输出只能碰对一类题, 基线通常只有 1/类别数。模型超过它才说明在读 prompt。\n训练前先算一遍, 训练中拿 reward 曲线和它比。',
        },
        { title: '塌缩是最短路径', body: '结论: 奖励不看题时, 模型最省力的拿分方式就是不看题, RL 会飞快走到那里。\n区域奖励下, RL 会很快把概率压到某个高分 token 上: 输出的熵 (随机性) 归零, 输出和输入无关, 每个 prompt 给的都是同一串。reward 曲线却很好看。\n这不是模型偷懒, 它在照着奖励优化: 奖励说「落在后半区就给分」, 常数输出就是这个奖励下的最优解。' },
        { title: '真实 RLVR 天然看题', body: '结论: 真实任务的判分天然依赖题目, 自己造玩具任务时要主动保证这一点。\n数学题比对答案、代码跑单元测试、格式校验, 这些判分都要先看题目是什么。奖励函数的签名必须是 reward_fn(prompts, completions), 两个都要。\n本仓库的 SeqTask: prompt 是 6 个随机 token (每个有 13 种取值), 共 $13^6 \\approx 480$ 万种 prompt; 答案是 prompt 的 copy / reverse / sort。训练集与留出集按 token 和 mod 5 不相交, 模型背不下来, 只能学规则。' },
      ],
      links: [
        { from: 'SeqTask.verify', to: 'reward_fn(prompts, completions)', body: 'verify 同时看 prompt 和 completion: 先由 prompt 算出标准答案, 再和 completion 比。只看 completion 的奖励是个 bandit 问题 (不看输入的老虎机, 每次拉同一个杆就行): 常数策略就能拿满分, 检验不出模型有没有在读题。' },
        { from: 'PromptDataGenerator', to: 'SeqTask', body: 'PromptDataGenerator 每步只产出 prompt, 没有 labels。prompt 里编码了题目 (copy / reverse / sort 一段序列), 答案由 SeqTask 程序算出, 回复由 policy 现场采样。' },
        { from: '组内零方差', to: 'DAPO 动态采样', body: '看题任务早期常见「整组全错」: 这道题的 G 条回复奖励全是 0, 优势全为 0, 不产生梯度。怎么处理见 GRPO 进阶一章的 DAPO 动态采样。' },
        { from: 'verify', to: '训练与验收', body: '同一个判分器既当训练奖励, 也当留出集指标。前提是它真的依赖 prompt; 否则两边都会被常数策略骗过。' },
      ],
      sourceRows: [
        { concept: '区域奖励 (反例)', code: 'reward = (completion >= V//2).mean()', takeaway: '只看 completion 里落在词表后半区 $[V/2, V)$ 的 token 比例, 和 prompt 无关。常数策略 (全部输出后半区的 token) 就能拿满分。' },
        { concept: '看题奖励', code: 'llm_finetune/data/tasks.py:SeqTask', takeaway: 'verify(prompts, completions): 回复的前 R 个 token 与 target(prompts) 算出的标准答案完全一致 (含 EOS) 才得 1 分。' },
        { concept: '判分器', code: 'llm_finetune/data/tasks.py:verify', takeaway: '0/1 判分, 不给部分分。任务太难时「整组全错」的比例会很高, 这时要么降难度, 要么增大 G (一题多采几条), 要么改成给部分分。' },
        { concept: '留出集怎么切', code: 'llm_finetune/data/tasks.py:sample_prompts', takeaway: '留出集 = token 和 $\\equiv 0 \\pmod{5}$ 的 prompt, 训练集是其余, 两者严格不相交。' },
        { concept: '评估', code: '按 prompt 分桶的准确率', takeaway: '按 prompt 分桶看准确率, 每类题各自对了几成。只看平均 reward 会被常数策略骗过去。' },
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
      subtitle: '不重训、不要数据, 几秒钟把两个微调模型拼成一个。读完你能判断两个任务能不能合, 也能说清 λ (把改动加回基座时乘的倍数) 为什么必须调。',
      tldr: '- 合并在做什么: 同一个基座各自微调出两个模型, 一个会 A, 一个会 B。把「微调改了什么」看成一个向量, 叫任务向量 (task vector): $\\tau = \\theta_{\\text{ft}} - \\theta_0$。这个式子在说: 微调后的权重减去基座权重, 剩下的就是这次微调带来的改动。模型合并就是把两个改动直接加回基座: $\\theta_0 + 1\\cdot(\\tau_A + \\tau_B)$, 不训练, 也不要数据。\n- 实测: 两个单任务模型都做不了「两段一起改写」, 两段全对的 EM (exact match, 整条回复和标准答案完全一致的比例) ≤ 0.027。两个改动相加后, 在留出集 (训练时没见过的 prompt) 上 EM 1.000。\n- 前提: 两个任务在函数上不重叠。A 只改低段 token 的输出, B 只改高段, 所以两个改动加在一起不打架。',
      question: '不重训, 能把两个微调模型合成一个吗? TIES、DARE 这些「更聪明」的合并, 真的比直接相加强吗?',
      evolution: {
        title: '不训练, 在权重空间里做加法',
        subtitle: '根问题: 同一基座微调出的几个模型各会一样, 想要全会, 又没有数据和算力重训',
        steps: [
          { name: '合数据重训', pain: '(原点) 两个模型各会一样, 想要一个全会的', fix: '两份数据合起来再训: 最稳, 但要数据、要算力, 别人的数据常常拿不到' },
          { name: '权重平均', pain: '手里常常只有权重, 没有数据', fix: '两份权重直接取平均: 零训练, 但每个任务的改动只剩一半, 两段全对 EM 0.039' },
          { name: 'Task Arithmetic', year: 2022, pain: '平均就是 λ=1/T (T 是任务数), 任务越多, 每个改动被稀释得越狠', fix: '取 $\\tau=\\theta_{\\text{ft}}-\\theta_0$ (微调前后的权重差), 乘上按验证集挑的 λ 加回基座: λ=1 时 EM 1.000' },
          { name: 'TIES / DARE', year: 2023, pain: '大模型的 τ 里有大量幅度很小、可有可无的坐标 (单个参数), 两个 τ 在上面符号相反, 相加会抵消', fix: 'TIES 先修剪小幅坐标, DARE 先随机丢弃一部分, 再相加; 本例 τ 只有约 10 万参数, 反而不如直接相加' },
        ],
      },
      code: 'llm_finetune/methods/merge.py · llm_finetune/run_finetune/merge/train_merge.py',
      points: [
        { title: '为什么要合并', body: '结论: 想要一个两样都会的模型, 合并是最省的一条路: 只要权重, 不训练。\n场景是这样的: 两个团队从同一个基座 (大家共同的出发点模型) 各微调了一个模型, 一个会 A, 一个会 B。想要两样都会, 有三条路:\n- 重训: 两份数据合起来再训一次。要数据, 也要算力。\n- 部署两份: 两份权重都上线, 各管一件事。\n- 合并: 只拿两份权重做加减, 不训练, 几秒钟。\n本仓库的两个任务都是「copy, 但改写一段 token」。基座本来会把 prompt 原样抄一遍 (copy 任务)。A 要把低段 token 3–8 各 +1 (比如 token 5 写成 6), B 要把高段 10–15 各 −1, token 9 两边都不改。评测要求两段同时改对。只微调 A 或只微调 B 的模型做这件事, 两段全对 EM 只有 0.004 / 0.027, 因为它们各自只会改自己那一段。' },
        { title: '合并就是对任务向量做加法', body: '结论: 最基本的合并叫 Task Arithmetic (任务算术): 把各任务的改动加起来, 乘一个倍数 λ, 加回基座。λ 选对了两段全对; λ 取 0.5 就退化成简单平均, 两段都只改到一半。\n先定义任务向量: $\\tau_t = \\theta_t - \\theta_0$。这个式子在说: 第 t 个任务微调完的权重 $\\theta_t$ 减去基座权重 $\\theta_0$, 逐个参数相减, 得到这次微调把每个参数挪了多少。\n再看合并: $\\theta = \\theta_0 + \\lambda\\sum_t \\tau_t$。这个式子在说: 把所有任务的改动加起来, 乘 λ, 再加回基座。\n- λ=1: 每个改动原样加回去。两段全对 EM 1.000, 干扰 0.000。干扰是合并后的模型在 A 段、B 段上, 相对只微调该任务的模型平均掉了多少准确率。\n- λ=0.5: 代进去是 $\\theta_0 + 0.5(\\tau_A + \\tau_B) = (\\theta_A + \\theta_B)/2$, 正好是两个模型直接取平均。每个任务向量只加了一半, 两段都只改到一半, EM 0.039。\nλ 怎么定: 在验证集 (训练分布的 prompt) 上从 {0.5, 0.7, 1, 1.5} 里挑两段全对 EM 最高的那个, 表里报的是留出集上的数。只看 λ=0.5 会误以为「合并没用」。' },
        {
          title: '调好 λ 的相加, 没被 TIES / DARE 赢过',
          key: true,
          body: '结论: 三种「更聪明」的合并, 每种都单独调过 λ, 在这里都没赢过直接相加。\n先说三种方法各做了什么。TIES 和 DARE 都是先把 τ 改一改再相加; SLERP 不碰 τ, 直接在两个完整模型之间插值。\n- TIES: 想解决两个 τ 在同一个坐标 (单个参数) 上打架。第一步, 每个 τ 只留幅度最大的 d 比例的坐标, 其余置 0。第二步, 每个坐标按幅度投票定一个符号, 幅度大的一方票更重。第三步, 每个坐标只对和这个符号同号的任务取平均, 少数派直接不参与。\n- DARE: 每个坐标以概率 p 随机置零, 剩下的放大 $1/(1-p)$ 倍, 这样平均下来每个坐标的期望值不变。它押的是任务向量里冗余很多, 丢掉大半也不伤。\n- SLERP: 把两个模型的权重看成两个向量, 沿球面而不是沿直线插值。t 是插值位置, t=0.5 就是正中间。\n每种方法各自在验证集上挑过 λ, 留出集两段全对 EM:\n- Task Arithmetic: 1.000\n- DARE p=0.5: 0.875\n- TIES d=0.5: 0.734\n- DARE p=0.9: 0.066。论文里「丢 90% 几乎无损」在这里不成立。\n- SLERP t=0.5: 0.090, 约等于简单平均。\n换种子 1、2 再跑: Task Arithmetic 0.973 / 0.617, 仍是最高 (同种子下 DARE p=0.5 是 0.953 / 0.473, TIES 是 0.695 / 0.211)。\n为什么修剪反而伤: TIES 要修的是大模型里大量小幅、冗余坐标之间的冲突。这里 τ 只有约 10 万参数, 冗余不多, 修掉一半就伤到了有用的坐标。DARE p=0.9 也一样, 论文里能丢 90% 靠的是大模型任务向量的高度冗余, 这里没有。',
        },
        { title: '能合并, 是因为任务这么设计', body: '结论: 直接相加能成功, 靠的是两个任务在函数上几乎不重叠, 而不是参数不重叠。\n- 函数上不重叠: A 的训练数据里, 高段 token 要照抄。所以 $\\tau_A$ 只被要求改「低段怎么输出」, 高段保持基座原样。B 反过来。两个改动各管一段。\n- 参数上是重叠的: 两个 τ 改的是同一批参数。在两者都非零的坐标里, 有 47.4% 符号相反, 一个往上推, 一个往下拉, 而随机符号也才 50%。参数层面冲突接近一半, 合并照样成功。\n调参时试过两种不满足前提的设置 (不在脚本里):\n- 同一输入要两种输出 (reverse 倒序 vs sort 排序, 加任务标记 token 区分): 所有方法合并后都不明显好于基座。同一个输入不可能同时有两个正确答案, 哪种合并都救不了。\n- 按输入分段的 ±1 平移: A 只见低段 prompt, B 只见高段 prompt。两个 τ 各自学成了全局的「+1」和「−1」, 相加互相抵消。LoRA r=4 时, 3 个种子上 Task Arithmetic 的单任务 EM 从 0.01 到 0.68 不等。\n边界: 只合并了 2 个任务。任务更多时 τ 之间的冲突更多, TIES 的价值才可能显出来, 这里没测。' },
      ],
      links: [
        { from: 'LoRA merge', to: '任务向量', body: '左边是 LoRA 的 adapter 合并: LoRA 训练时只训旁挂的两个小矩阵 B 和 A, 训完把 $\\Delta W = (\\alpha/r)BA$ 加回原权重 ($\\alpha/r$ 是缩放系数), 变回普通的线性层。这个 $\\Delta W$ 就是那次微调的任务向量。merge.py 只处理 state_dict (参数名到张量的字典), 不碰模型结构, 所以全参微调和 LoRA 两种结果都能合。' },
        { from: 'cos(τ_A, τ_B)', to: 'sign_conflict', body: '左边是两个任务向量摊平后夹角的余弦, $\\cos = +0.046$, 接近 0, 说明两个 τ 整体上几乎正交 (方向互不相关)。右边是逐坐标的符号冲突: 两者都非零的坐标里, 符号相反的有 47.4%, 随机符号是 50%。\ncos 看的是整体, 冲突看的是一个个坐标, 所以不能从 cos 约等于 0 推出没有冲突。' },
        { from: 'SLERP', to: '简单平均', body: 'SLERP 插的是完整权重 $\\theta_A$、$\\theta_B$, 不是任务向量。两者共享同一个 $\\theta_0$, 方向很接近, 夹角 $\\Omega = 0.333$ rad。t=0.5 时 SLERP 正好等于线性平均再放大 $1/\\cos(\\Omega/2)$ 倍。这个式子在说: 夹角越小, 球面插值和直线平均差得越少。这里只放大了 1.014 倍, 所以结果约等于简单平均。' },
        { from: '同一个 θ₀', to: 'task arithmetic', body: '$\\tau$ 的定义是「微调后减基座」, 这就要求所有模型从同一个基座出发。两个模型如果来自不同基座, 或从不同随机初始化开始训练, 拿其中一个基座去减, 差值里混进了两个起点本身的差别, 不再只是微调的改动, 相减没有意义。' },
      ],
      sourceRows: [
        { concept: '任务向量', code: 'llm_finetune/methods/merge.py:task_vectors', takeaway: '$\\tau_t = \\theta_t - \\theta_0$: 对 state_dict 里的每个张量逐个相减。只取浮点张量; 整型 buffer (模型里存的整数张量, 不是可训练的权重) 没有「方向」可言, 合并时原样取基座的。' },
        { concept: '相加', code: 'llm_finetune/methods/merge.py:task_arithmetic', takeaway: '$\\theta_0 + \\lambda\\sum_t\\tau_t$: 所有任务向量逐坐标相加, 乘 λ, 加回基座。除 SLERP 外, 所有方法都只是先对 τ 做逐坐标处理 (TIES 修剪加投票, DARE 随机丢弃), 再走这一步。' },
        { concept: 'TIES', code: 'llm_finetune/methods/merge.py:ties_merge', takeaway: '- 修剪: 每个 τ 只留 $|\\cdot|$ 最大的 density 比例, 其余置 0。本库逐张量取 top-k, 没有把整个模型摊平后统一排序。\n- 选符号: $\\gamma = \\mathrm{sign}(\\sum_t\\hat\\tau_t)$。这个式子在说: 把修剪后各任务在同一坐标上的值加起来, 取和的正负号。按幅度投票, 不是按人头。\n- 不相交平均: 每个坐标只对非零且与 γ 同号的任务取平均, 冲突的一方不参与, 而不是被平均稀释。\n脚本开头用 [3, −1, 0.1] 和 [−2, −2, 0.1] 手算验过 (density 2/3, 每个只留 2 个): 修剪后是 [3, −1, 0] 和 [−2, −2, 0]; 相加得 [1, −3, 0], 符号是 (+, −, 0); 第 1 个坐标只有 3 同号, 第 2 个坐标两个都为负, 取平均 −1.5, 第 3 个坐标都被修掉。结果 [3, −1.5, 0]。' },
        { concept: 'DARE', code: 'llm_finetune/methods/merge.py:dare', takeaway: '每个坐标以概率 p 置零, 其余乘 $1/(1-p)$。这样期望不变: 一个坐标有 $1-p$ 的概率留下并被放大 $1/(1-p)$ 倍, 平均下来还是原值。不除的话, 等于把 λ 缩成 $(1-p)$ 倍。随机掩码用固定种子生成, 同一个 seed 得到同一张掩码。' },
        { concept: 'SLERP', code: 'llm_finetune/methods/merge.py:slerp', takeaway: '对两个完整模型逐张量做球面插值: $\\frac{\\sin((1-t)\\Omega)}{\\sin\\Omega}\\theta_A + \\frac{\\sin(t\\Omega)}{\\sin\\Omega}\\theta_B$, Ω 是两个张量的夹角。这个式子在说: 按夹角给两边分配权重, 让结果沿球面走。$\\sin\\Omega \\to 0$ 时 (两个张量几乎同向) 除法数值不稳, 退化为线性插值。' },
        { concept: '干扰度量', code: 'llm_finetune/methods/merge.py:sign_conflict', takeaway: '把两个任务向量各自摊平成一个长向量, 逐坐标对齐。在两者都非零的坐标里, 数符号相反的比例。本例 47.4%。' },
        { concept: '两个任务', code: 'llm_finetune/run_finetune/merge/train_merge.py:EditTask', takeaway: 'copy 任务的变体: 先把 prompt 原样抄出来, 再用一张查找表把指定 token 平移 (A 低段 3–8 各 +1, B 高段 10–15 各 −1)。A 的数据里高段照抄, 反之亦然, 所以两个 τ 各管一段。两个任务各全参微调 300 步, lr=1e-3。' },
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
      subtitle: '读完你能说清 RLAIF 省掉了什么, 又把什么原封不动交给了 judge。',
      tldr: 'RLAIF (用 AI 反馈代替人工偏好标注) 不找人标数据: 一个 AI 裁判 (judge) 按一份成文规则 (宪法) 批评模型的回复并改写, 「改写后比改写前好」就是一个偏好对, 之后照旧用 DPO 训练。\n标签质量的上限就是 judge 的上限: 它查得到的违规从 0.648 降到 0.031, 它漏查的违规只从 0.418 降到 0.320。',
      question: '不找人标注, 能对齐吗? 条文写了但规则没查, 会怎样?',
      evolution: {
        title: '标注员从人换成一份宪法',
        subtitle: '根问题: 偏好标签要人一条条标, 贵、慢、标注员之间不一致, 有害内容还得有人看',
        steps: [
          { name: '人工排序', pain: '(原点) 对齐要大量偏好标签, 只能找人标', fix: '人给几条回复排序 (InstructGPT 一次 4–9 条): 质量有保证, 成本随数据量涨' },
          { name: 'Constitutional AI', year: 2022, pain: '人标又贵又慢, 有害样本还得有人一条条看', fix: '好坏写成条文; AI 按条文批评改写后 SFT, 再比较回复出偏好标签, 训 RM 做 RL' },
          { name: '改写前后直接成对', pain: '两段各训一次, 还要另让 AI 比较、训 RM、做 RL', fix: '"改写后 ≻ 改写前" 直接当偏好对交给现成 DPO: 标签上限就是 judge 查了什么' },
        ],
      },
      code: 'llm_finetune/methods/rlaif.py · llm_finetune/run_finetune/rlaif/train_rlaif.py',
      points: [
        { title: '人是瓶颈', body: '结论: RLHF 卡在人工标注上, Constitutional AI 把标注员换成一份条文加一个 AI。\nRLHF 的偏好标签来自人: 贵、慢, 不同标注员的标准还不一致。有害内容也得有人一条条看。\nConstitutional AI 的做法分四步:\n- 写条文: 先把「什么是好回答」写成一份条文, 叫宪法。\n- AI 反馈: 让一个 AI 反馈者 (judge) 按条文检查回复, 指出哪里违规, 并改写出一个合规版本。\n- 成对: 改写前后两个版本天然就是一个偏好对, 改写后是 chosen (好的那条), 改写前是 rejected (差的那条)。\n- 训练: 之后照旧用 DPO 在这些偏好对上训练。\n本仓库的 judge 是两条规则函数, 不是 LLM; 真实做法里批评和改写都由 LLM 生成。' },
        { title: '一轮怎么走', body: '结论: 本仓库跑的是一轮离线流程: 采样、批评、改写、成对、DPO。\n- 采样: 起点是「只求有用」的模型, 它学的是 copy 任务, 什么都照抄, 包括 prompt 里的脏话 (token 15) 和电话号码 (token 13)。用温度 1 采样 4096 条回复。\n- 批评: judge 逐条原则检查每条回复, 返回「违反哪条, 在哪个位置」。4096 条里 2539 条被指出问题。\n- 改写: 按批评把 15、13 打码成 3 (代表 ***)。例子: 回复 [4, 8, 12, 15, 13, 15] 改写成 [4, 8, 12, 3, 3, 3]。\n- 成对: 只有被批评的回复才成对, 2539 条批评得到 2539 个偏好对。没问题的回复没有「更好的版本」可比, 直接丢掉。\n- 训练: 现成的 DPO 跑 150 步, 每步从偏好对池子里随机取 64 对。' },
        {
          title: 'judge 没查的, DPO 学不到',
          key: true,
          body: '结论: 训练信号只来自 judge 实际检查的东西, 条文写了但规则没查的, 偏好对里没有信号。\n宪法就是那份成文规则。「无害」这条条文写着「不说脏话 15, 以及它的变体 14」, 但 judge 的检查函数只查了 15。于是遇到 14, judge 不批评, 改写也不动它。14 在 chosen 和 rejected 里同时出现, 两边一样, DPO 的差值里没有「去掉 14」这个方向, 训练对它没有直接压力。\n- 实测: 含变体 14 的回复比例从 0.418 降到 0.320。\n- 降的那点从哪来: DPO 的副作用。第一个回复 token 常被带偏, 偶尔把 14 也打了码。模型没有学会原则。脚本断言这个比例至少还剩训练前的 60%。\n条文是给人看的, 生效的是规则。换成 LLM 做 judge, 漏检不会消失, 只是更难被发现。',
        },
        { title: 'judge 的偏差也会被学走', body: '结论: judge 造的偏好对带着系统性偏差时, DPO 学的是偏差, 不是原则。\n调参时试过给宪法加第三条「不啰嗦: 不要连说两遍」。改写就是删掉重复, 所以 chosen 总比 rejected 短。\nDPO 从这些对里学到的是「早点说 EOS」, 不是「别重复」: 模型只说 1 个 token 就结束。\n- 违规率: 从 0.605 降到 0.000, 看起来完美。\n- 理想回复 (照抄且打码正确) 的比例: 从 0.395 降到 0.000, 什么都不说了。\n- 那组设置: lr=3e-4、200 步、1024 条采样。lr 降到 1e-4 也一样。\n最终版的宪法只保留不改长度的原则。' },
        { title: '有用性要一起看', body: '结论: 只看违规率会被「什么都不说」骗过去, 必须同时量有用性。\n本脚本在留出 prompt 上用贪心回复量了两项有用性:\n- 非敏感位置照抄率: 该照抄的位置 (不是 15 也不是 13 的) 抄对了几成。从 1.000 降到 0.856, 错误集中在第一个回复 token, 这是 DPO 一章说的 likelihood displacement 的老毛病。\n- 整条 = 理想回复: 整条回复和「照抄 + 打码」的理想答案完全一致的比例。从 0.352 升到 0.488。\n边界: lr 要比 dpo 脚本小。用 3e-4 / 200 步时违规率同样降到 0, 但理想回复从 0.352 掉到 0.051, 有用性几乎全毁。脚本用的 1e-4 / 150 步是在「守规矩」和「还有用」之间挑的。' },
      ],
      links: [
        { from: 'CONSTITUTION', to: 'critique / revise', body: 'CONSTITUTION 是宪法的代码形式。每条原则是一个四元组 (名字, 条文, 检查函数, 修改函数)。critique 调每条的检查函数产出批评, revise 调每条的修改函数产出改写。条文只给人看, 训练信号只来自后两个函数。' },
        { from: 'build_preference_pairs', to: 'DPOLoss', body: 'build_preference_pairs 把 (改写后, 改写前) 整理成和 PreferenceDataGenerator 一样的输出格式, 所以能直接接 PairwiseForward 和现成的 DPOLoss, 一行不改。' },
        { from: 'DPO 的 likelihood displacement', to: 'RLAIF 的有用性损失', body: '照抄率掉在第一个回复 token 上, 和 DPO 一章里 log π(chosen) 下降是同一件事: DPO 只拉大差值, chosen 自己的概率可能一起掉。这不是 RLAIF 特有的。' },
        { from: '一轮离线 DPO', to: '真实 RLAIF', body: '真实做法和这里的差别 (这里都没做):\n- 轮数: 真实 RLAIF 常迭代多轮, 或者用 AI 给的偏好训 RM 再做 PPO。Constitutional AI 论文本身分两段: 先在改写后的回复上 SFT, 再让 AI 比较成对回复打标签, 训 RM 做 RL。\n- 生成: 批评和改写也由 LLM 生成, 改写本身会出错。这里两步都是确定性的规则。' },
      ],
      sourceRows: [
        { concept: '宪法', code: 'llm_finetune/methods/rlaif.py: CONSTITUTION', takeaway: '两条原则: 「无害」不说 15 (条文还写了变体 14), 「隐私」不复述 13, 都要求打码成 3。规则函数只查 15 和 13。' },
        { concept: '批评', code: 'llm_finetune/methods/rlaif.py:critique', takeaway: '逐条原则检查, 返回类似「违反「无害」: 位置 [3, 5]」的批评。空列表就是没问题。' },
        { concept: '改写', code: 'llm_finetune/methods/rlaif.py:revise', takeaway: '依次套用每条原则的 fix, 长度补 PAD 回 R。示例: [4, 8, 12, 15, 13, 15] 变成 [4, 8, 12, 3, 3, 3]。' },
        { concept: '造偏好对', code: 'llm_finetune/methods/rlaif.py:build_preference_pairs', takeaway: '只对被批评的回复造 (chosen = revise(y), rejected = y)。EOS 之后的 token 先清成 PAD, 不算回复。' },
        { concept: '离线池', code: 'llm_finetune/methods/rlaif.py:AIPreferenceData', takeaway: '偏好对只造一次, DPO 每步从池子里有放回地随机取 64 对。' },
        { concept: '验收', code: 'llm_finetune/run_finetune/rlaif/train_rlaif.py:evaluate', takeaway: '在留出 prompt 上贪心生成, 报四类数: judge 查得到的违规率, judge 看不见的变体 14 的比例, 非敏感位置的照抄率, 整条等于理想回复的比例。' },
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
      subtitle: '同样的解、同样的训练步数, 只把标签从「答案对不对」换成「每一步对不对」。读完你能说清它为什么能多挑对几道题, 还能指出错在哪一步。',
      tldr: '多步推理的解怎么打分? ORM (outcome reward model, 结果奖励模型) 只给整条解打一个分: 最终答案对不对。PRM (process reward model, 过程奖励模型) 给解的每一步各打一个分: 这一步对不对。\n同样的解、同样的训练步数, 逐步标签好学得多, 还能指出第一个错步。best-of-8 (每题 8 个候选解, 挑分最高的那条) 的答案正确率: ORM 0.488, PRM 0.648。',
      question: '只看最终答案的奖励模型差在哪? 过程错、答案蒙对的解, 谁会被骗?',
      evolution: {
        title: '奖励从终点挪到每一步',
        subtitle: '根问题: 多步推理只在终点判对错, 模型得从一个 0/1 里自己猜是哪一步坏的',
        steps: [
          { name: 'ORM', pain: '(原点) 多步解最容易判的只有最终答案', fix: '只在终点打一个分: 标签便宜, 但错在哪要自己猜, 蒙对的错过程也算 "好"' },
          { name: 'PRM', year: 2023, pain: '一个 0/1 摊到 K 步上, 信号稀疏; 过程错、答案蒙对照样高分', fix: '每一步各标对错 (PRM800K 由人逐步标): 标签多 K 倍, 能指出第一个错步' },
          { name: '解分取 min', pain: 'best-of-N 要给整条解一个分, 各步取平均会被对的步稀释', fix: '取各步概率的最小值 (论文主方案连乘, min 是备选): 一步错整条错' },
        ],
      },
      code: 'llm_finetune/methods/prm.py · llm_finetune/run_finetune/prm/train_prm.py',
      points: [
        { title: 'ORM 得自己猜错在哪', body: '结论: ORM 的监督信号太稀, 同样的预算下几乎没学到东西。\n一条 4 步的解, ORM 只拿到 1 个标签「最终答案对不对」。答案错了, 它得自己从这一个 0/1 里推出是哪一步坏的: 4 步里任何一步错都会让答案错, 标签不告诉它是哪一步。信号稀疏, 学得慢。\n实验: 玩具任务是 4 步模 10 算术链 (每步做一次 + / − / × 再对 10 取余), 训练 600 步, 看 BCE (二分类交叉熵, 越低越准):\n- ORM: 停在 0.682。$\\ln 2 = 0.693$ 是瞎猜的 loss, 几乎没学。\n- PRM: 降到 0.359。\n另一个问题: 过程错、答案蒙对的解 (候选里占 4.5%), 在 ORM 的训练数据里就是「好」, ORM 会给它高分。' },
        { title: 'PRM 每步一个标签', body: '结论: PRM 的标签多 K 倍, 而且每个标签只要求一个局部判断。\n- 标签数: N 条解, ORM 拿到 N 个标签, PRM 拿到 $N\\cdot K$ 个 (K 是步数, 这里是 4)。\n- 局部判断: 每个标签只问「这一行算式对不对」, 不问整条解对不对。\n第 $i$ 步算「对」的定义: $v_i = \\mathrm{op}_i(v_{i-1}, k_i)$, 其中 $v_{i-1}$ 是解里写的上一步的值 (不是真值)。这个式子在说: 拿解里写下的上一个结果, 做这一步的运算, 得到的数和解里写的 $v_i$ 一样就算对。\n所以前面错了、这一步照着错的值算对了, 这一步仍然算对。像批改真人的草稿: 只看这一行有没有算错。' },
        {
          title: '同样预算, 差距来自标签粒度',
          key: true,
          body: '结论: 打分器、初始化、训练数据、步数全部相同, 只有标签粒度不同, PRM 挑解的能力明显强于 ORM。\n设置: 同一个主干加标量头 (和 RM 一章同一个结构)、同一初始化、同一串训练数据 (600 步, 每步 96 条解)。评测在留出集 512 题上, 每题 8 个候选解, 用打分器挑分最高的那条。\n- best-of-8 答案对: 随机挑 0.473 / ORM 0.488 / PRM 0.648\n- best-of-8 每步都对: 随机挑 0.426 / ORM 0.422 / PRM 0.627\n- 定位第一个错步 (只看至少错一步的候选, PRM 打分最低的那一步是不是第一个错步): PRM 0.575, 最好的常数猜法「总说第 1 步错」0.341。ORM 只有一个分, 无从定位。\nORM 几乎等于随机挑; PRM 把答案正确率从 0.47 提到 0.65。',
        },
        { title: '解分取 min', body: '结论: PRM 给整条解打分时, 取各步概率里最小的那个。\n理由: 一步错, 整条就错。取各步的平均, 一个错步会被其余对的步稀释, 整条还是高分。取最小值就是看短板。\n边界: 论文 (Lightman et al.) 的主方案是各步概率连乘, min 是它比较过的备选。取 min 的好处是分数不随步数 K 变多而变小。' },
        { title: '这些数字的边界', body: '- 候选解的来源: 一个带噪的程序求解器, 每步以 20% 的概率写错一个数, 之后照着错的值往下算。不是 LM 采样, 错误分布比真实简单。\n- ORM 学得慢, 不是学不会: 脚本外另跑同配置 ORM 3000 步, 约 2000 步才开始降, 3000 步 BCE 约 0.55。这个数不在脚本里, readme 已注明。这里比的是「同样的训练预算」。\n- 标签成本: 逐步标签由程序判, 不花钱。真实 PRM 的主要成本正是标签 (PRM800K 由人逐步标), 这里体现不了。\n- PRM 也没训满: 8 个候选里至少一个答案对的比例是 0.992, PRM 只挑中 0.648, 离上限还远。\n- 同时错多步时, 「分最低的一步」可能指向后面那个错步, 定位准确率会被低估一点。' },
      ],
      links: [
        { from: 'RewardModel', to: 'PRM / ORM', body: 'PRM 和 ORM 用同一个「主干 + 标量头」, token_scores 给每个位置都算一个分 (RewardModel.forward 只取末位置)。差别只在 BCE 落在哪几个位置:\n- ORM: 只落在 EOS 位置, 标签是答案对不对。\n- PRM: 落在 4 个 $v_i$ 的位置, 标签是每步对不对。' },
        { from: 'PRM 解分', to: 'infer · 推理时计算', body: '推理章的 best-of-N 和 PRM beam search 用的是程序化的 PRM (直接按规则判每步)。这一章讲的是怎么把 PRM 训出来。' },
        { from: 'PRM 的逐步分', to: 'PPO 的逐 token 优势', body: 'PRM 的逐步分可以当作稠密奖励 (不只是末尾一个分)。有了逐步奖励, PPO 的 critic 和 GAE 才有东西可以做逐 token 的信用分配, 见下一章。' },
        { from: 'step_positions', to: 'v_i 所在位置', body: '第 $i$ 步的分取自 $v_i$ 这个 token 所在的位置, 也就是模型读完 $v_i$ 之后。取在前一个位置, 模型还没看到 $v_i$, 判断不了它对不对。' },
      ],
      sourceRows: [
        { concept: '玩具数据', code: 'llm_finetune/methods/prm.py:ArithChain', takeaway: '题目写成 a0 op1 k1 … op4 k4 SEP, 解每步写成「op k 结果」三个 token。留出集 = $(a_0 + \\sum k) \\equiv 0 \\pmod 5$ 的题, 和训练集不相交。' },
        { concept: '逐位置取分', code: 'llm_finetune/methods/prm.py:token_scores', takeaway: '给序列每个位置算一个 logit。RewardModel.forward 只取末位置, PRM 要每个位置都有。' },
        { concept: 'PRM loss', code: 'llm_finetune/methods/prm.py:prm_loss', takeaway: '在 4 个 $v_i$ 位置上做 BCE, 标签是 step_ok (每步对不对)。' },
        { concept: 'ORM loss', code: 'llm_finetune/methods/prm.py:orm_loss', takeaway: '只在 EOS 位置做一次 BCE, 标签是 outcome_ok (答案对不对)。' },
        { concept: '整条解的分', code: 'llm_finetune/methods/prm.py:solution_scores', takeaway: '- PRM: 各步概率取 min。\n- ORM: 末位置概率。\nbest-of-8 取分最高的那条候选。' },
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
      subtitle: '把 critic 装回去, 和 GRPO 用同样的采样预算比一比。读完你能手算 GAE, 也能说清在什么规模下 critic 值不值它的显存。',
      tldr: 'PPO (proximal policy optimization, RLHF 里常用的强化学习算法) 比 GRPO 多一个 critic: 另外训练的一个网络, 在每个 token 位置估「从这里往后还能拿多少分」, 当 baseline (参照分) 用。学不好时, 它就是个昂贵的常数。\n玩具规模实测: PPO 的留出集 pass@1 (按策略采样一次就答对的比例) 高于 GRPO (0.381 对 0.287), 但把 critic 冻住不训照样有 0.389。\n高出的那截来自「一题一采」的采样方式, 不是 critic。',
      question: 'GRPO 省掉的 critic, 到底买到了什么?',
      evolution: {
        title: '末尾一个分, 怎样分给每个 token',
        subtitle: '根问题: 奖励只落在最后一个 token, 要把功劳分给前面每一步, 又不能让梯度方差爆掉',
        steps: [
          { name: 'REINFORCE', year: 1992, pain: '(原点) 整条回复只在末尾拿一个分, 要推出哪些 token 该加概率', fix: '每个 token 都拿整条回报乘 $\\nabla\\log\\pi$: 无偏, 但方差大, 好坏 token 一起奖' },
          { name: '减去 V(s_t)', pain: '回报全为正时每个 token 都被加概率, 方差大', fix: 'critic 学 $V(s_t)$ 估 "从这往后能拿多少" 并减掉: 逐 token 优势, 多养一个网络' },
          { name: 'TD(0)', pain: '回报要等到结尾, 后面每一步的随机性都累加进来', fix: '只看一步: $\\delta_t = r_t + \\gamma V(s_{t+1}) - V(s_t)$; 方差小, 但 V 不准就有偏' },
          { name: 'GAE', year: 2015, pain: 'MC (直接用整条回报) 无偏但方差大, TD(0) 方差小但有偏, 只能二选一', fix: '把 δ 按 $(\\gamma\\lambda)^k$ 折扣累加, λ 在两端之间调; 本章取 λ=0.95' },
        ],
      },
      code: 'llm_finetune/methods/ppo.py · llm_finetune/run_finetune/ppo/train_ppo.py',
      points: [
        { title: '为什么要 baseline', body: '结论: 策略梯度要减一个 baseline 来降方差, GRPO 和 PPO 的 baseline 来源不同。\n策略梯度的做法是: 采一条回复, 拿它的回报 (奖励) 当权重, 回报高就把这条回复的概率抬高。问题是回报全为正时每条都被抬, 方差很大。减一个参照分 $b$ 不改变期望梯度, 只降方差: 高于 $b$ 的抬, 低于 $b$ 的压。减掉之后的量叫优势 $A$。\n- GRPO: $b$ 是同一道题 G 条回复的均值。免费, 但必须一题多采, 而且整条回复共用一个 $A$。\n- PPO: $b$ 是 critic 学出来的 $V(s_t)$, 也就是「读到第 t 个 token 时, 预计往后还能拿多少分」。一题采 1 条就够, 每个 token 各有一个 $A_t$。代价是多养一个和 policy 同尺寸的网络 (权重、梯度、Adam 状态都要)。' },
        { title: 'GAE 在 TD(0) 和 MC 之间插值', body: '结论: GAE (generalized advantage estimation, 广义优势估计) 用一个参数 λ 在「只信 critic」和「只信真实回报」之间调。\n优势 $A_t$ 回答「第 t 个 token 比 critic 预期的好多少」。两步算:\n- 先算每一步的意外 $\\delta_t = r_t + \\gamma V(s_{t+1}) - V(s_t)$。这个式子在说: 这一步实际拿到的奖励, 加上 critic 对下一个状态的估值 (打 $\\gamma$ 折), 减去 critic 对当前状态的估值。差值是正的, 说明这一步比预期好。\n- 再把后面各步的意外按 $(\\gamma\\lambda)^k$ 打折加起来: $A_t = \\sum_k (\\gamma\\lambda)^k \\delta_{t+k}$。这个式子在说: 第 t 步的优势不只看自己这一步的意外, 还看之后每一步的意外, 越远的权重越小。\n两个极端:\n- λ=0 (TD(0), 只往后看一步): $A_t = \\delta_t$, 只信下一步的 $V$。方差小, 但 $V$ 不准时有偏。\n- λ=1 (Monte-Carlo, 一直看到结尾): $A_t = \\sum_k \\gamma^k r_{t+k} - V(s_t)$, 只用真实回报减当前估值。无偏, 但方差大。\n脚本用随机数单测了两端: λ=0 的 A 与 TD(0) 误差 0.0; λ=1 的回报目标 $A_t + V(s_t)$ 与 MC 回报误差 4.8e-7。本章训练取 λ=0.95, γ=1 (LLM 场景惯例, 不折扣)。' },
        { title: '手算一遍 GAE', body: '用一组真实数字走一遍。数字取自 check_gae 的第 3 条序列 (种子 0, 长度 3, $\\gamma = 0.9$), 保留两位小数: 逐 token 奖励 $r = [-0.14, -0.11, -0.61]$, critic 估值 $V = [1.71, 2.38, -1.13]$。序列结束后是终止态, $V = 0$。\n从后往前算每一步的意外 $\\delta$:\n- $\\delta_2 = -0.61 + 0 - (-1.13) = 0.52$ (最后一步, 下一个状态是终止态)\n- $\\delta_1 = -0.11 + 0.9 \\times (-1.13) - 2.38 = -3.507$\n- $\\delta_0 = -0.14 + 0.9 \\times 2.38 - 1.71 = 0.292$\n第 0 个 token 的优势:\n- λ=0: $A_0 = \\delta_0 = 0.292$, 只看自己这一步。\n- λ=1: $A_0 = \\delta_0 + 0.9\\,\\delta_1 + 0.81\\,\\delta_2 = -2.443$, 把后面两步按 0.9、0.81 打折加进来。\n- 对一下 MC 回报: $-0.14 + 0.9 \\times (-0.11) + 0.81 \\times (-0.61) = -0.733$, 减 $V(s_0) = 1.71$ 也是 $-2.443$, 和 λ=1 的结果一致。\n这里的 $V$ 是随机数, 和真实回报对不上, 所以两端差得远, 连符号都相反。critic 准不准, 直接决定 λ 小的时候 A 靠不靠谱。' },
        { title: '奖励落在哪', body: '结论: verifier 的分只给最后一个 token, 其余每个 token 只有一份 KL 惩罚。\n- verifier 分: 排序全对得 1 分, 否则 0, 只加在回复的末 token 上。\n- KL 惩罚: 每个 token 再扣 $\\beta(\\log\\pi_{\\text{old}} - \\log\\pi_{\\text{ref}})$。这个式子在说: 这个 token 在当前策略下的 log 概率比在 ref (冻结的 SFT 起点) 下高多少, 高得越多扣得越多, 防止 policy 离起点太远。本仓库 β=0.02。\n两种放 KL 的方式:\n- PPO (本仓库, 沿用 InstructGPT 的做法): KL 写进逐 token 的奖励, 之后会被 critic 当成回报的一部分去学, 也会被 GAE 折扣累加。\n- GRPO: KL 不进奖励, 直接加在 loss 上。' },
        {
          title: '玩具规模上 critic 几乎没用',
          key: true,
          body: '结论: 本规模下 PPO 赢 GRPO 的那 0.09, 来自采样方式, 不来自 critic。\n设置: 同一个 sort 任务、同一个 SFT 起点 (留出集贪心 EM 0.562, pass@1 0.186; 贪心 EM 是每步取最大概率 token 生成时整条全对的比例)、每步 256 条回复、60 步, lr=3e-4。留出集 pass@1: PPO 0.381, GRPO 0.287。\n- critic 学没学会: 用「baseline 解释的方差」量, 也就是减掉 baseline 之后回报的方差少了几成。critic 的 $V(s_0)$ 只解释 2.5%, 免费的组均值解释 36.5%。「critic 方差更小」在这里不成立。\n- 冻结对照: 把 critic 的 lr 设成 0, 它停在初始状态 (V 约等于 0), 剩下的只有批内白化减掉的那个常数。pass@1 0.389, 和 PPO 的 0.381 在噪声内。\n- 高出 GRPO 0.09 的来源: PPO 是 256 道不同的题各采 1 条。GRPO 是 32 道题各采 8 条, 其中约 32% 的组全对或全错, $A\\equiv 0$, 白占预算。\n- 为什么 critic 难学: 它要判断「这个排序前缀还能不能全对」, 和 policy 学排序本身一样难。组均值不用学, 直接把「这道题难不难」平均出来。\n- 逐 token 信用分配 (critic 看出哪个前缀已经没救了): 在 60 步、64 维的 critic 上也没显出来。\n边界: 大模型上 critic 从 RM 或 SFT 初始化并先预热, 能学到有用的 V, 结论可能不同。',
        },
        { title: '代价是实打实的', body: '- 要训的参数: 从 99,648 涨到 199,360 (两倍): critic 和 policy 同尺寸, 梯度和 Adam 状态也翻倍。\n- 常驻参数: 从 99,648 涨到 299,008 (三倍): policy、ref、critic 三份都要放在内存里。\n- 60 步耗时: 本机两次实测 GRPO 5.7–6.1 s, PPO 11.8–12.0 s, 约两倍。\n这就是 DeepSeek 用 GRPO 去掉 critic 的动机。' },
      ],
      links: [
        { from: 'RewardModel', to: 'critic', body: 'critic 就是 RM 那个「主干 + 标量头」, 从 SFT 权重起步。token_values 取每个位置的分, 而不只是末位置。位置 $P-1+t$ 的隐状态读完了 prompt 和前 t 个回复 token $o_{\\lt t}$, 还没生成 $o_t$, 正是状态 $s_t$ 的估值 $V(s_t)$。' },
        { from: '组均值', to: 'critic', body: '两者都是 baseline, 作用一样, 来源不同。\n- 组均值: G=1 时组里只有样本自己, 均值就是它, $A\\equiv 0$, 所以 GRPO 必须一题多采。\n- critic: 与组无关, 一题一采也有信号。' },
        { from: 'PPO clipped surrogate', to: 'GRPO 进阶', body: '两者的目标函数外形一样: 新旧策略的概率比 ratio 乘优势, 再 clip, 同一批 rollout 更新 μ=2 次。只是 $A$ 的来源不同。clip 和 ratio 的细节见 GRPO 进阶一章。' },
        { from: 'PRM 逐步分', to: '逐 token 优势', body: '只有末 token 一个分时, critic 得自己学出「哪个前缀已经没救了」。有了 PRM 的逐步奖励, 每一步都有分, GAE 才有东西可分。' },
      ],
      sourceRows: [
        { concept: '逐位置估值', code: 'llm_finetune/methods/ppo.py:token_values', takeaway: '复用 RewardModel, 取位置 $P-1+t$ 的分当 $V(s_t)$, 与 completion_logprobs 用同一个错位 (位置 t 的输出对应第 t+1 个 token)。' },
        { concept: 'GAE', code: 'llm_finetune/methods/ppo.py:gae', takeaway: '从后往前扫一遍, 每步 A_t = δ_t + γλ·A_{t+1}。mask 外 (EOS 之后) 视为终止, $r = V = 0$; 否则 GAE 会从 EOS 之后的垃圾 token 上自举, 把它们的估值也算进来。' },
        { concept: '逐 token 奖励', code: 'rewards[arange(N), last] += score', takeaway: 'verifier 分只加在末 token 上; 其余 token 的奖励只有 $-\\beta\\cdot$KL。' },
        { concept: '一步', code: 'llm_finetune/methods/ppo.py:PPOTrainer', takeaway: 'PPOTrainer.step 的流程: rollout 采样, 冻结 old_logp / ref_logp / old_values 三组读数, 算 GAE, 批内白化 (A 减均值除标准差), 再做 μ=2 轮 clip 更新加 value loss。policy 与 critic 各有一个优化器。' },
        { concept: '两端单测', code: 'llm_finetune/run_finetune/ppo/train_ppo.py:check_gae', takeaway: '用 4 条长度不等的随机序列, γ=0.9: λ=0 与 TD(0) 误差 0.0e+00, λ=1 与 MC 回报误差 4.8e-07。' },
        { concept: '解释方差', code: 'llm_finetune/run_finetune/ppo/train_ppo.py:baseline_explained_var', takeaway: '$1 - \\mathrm{Var}(R-b)/\\mathrm{Var}(R)$: 减掉 baseline 之后回报的方差少了几成, 1 是完美, 0 和常数一样。在留出集 64 题 × 8 条上量。组均值包含样本自己, 这个数略偏乐观。' },
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
      title: '训练脚本与落盘 · 16 种方法的代价结构',
      subtitle: '读完你能按「有什么数据、有多少显存、能不能在线采样」三个问题直接点名该用的方法, 还能说出它每步跑几次前向、常驻几份权重、最后落盘 (训完存下来交付出去) 的是什么。',
      tldr: '16 种微调方法之间的区别, 不在 loss 写得好不好看, 而在代价结构: 训练时额外要付出什么。\n- 要不要常驻 reference model (ref, 冻结不动的起点模型副本, 用来拉住训练别跑偏)。多一份 ref 就多占一份权重、每步多跑一次前向。\n- 要不要在线采样: 训练中途让模型自己现场生成回复, 再拿来训练。\n- 落盘是全量权重还是 adapter (LoRA 旁挂的小矩阵, 本仓库 76 KiB, 整模型 389 KiB)。\n实测: DPO (用成对偏好直接训练) 每步 2 次 LM 前向 (policy 和 ref 各 1 次, policy 是正在训练的模型)、常驻 778 KiB、200 步 5.4 s; SimPO 与 ORPO 去掉了 ref, 1 次前向、389 KiB、4.0 s。',
      question: '手上只有 (问, 答) 而且显存紧张, 该用哪个? 换成成对偏好 (同一个问题一好一坏两个回答) 呢? 换成「答案能被程序判对错」呢?',
      evolution: {
        title: '数据、显存、采样, 一道道筛',
        subtitle: '根问题: 方法的差别主要在代价结构; 手上有什么数据、多少显存、能否在线采样, 先筛掉一大半',
        steps: [
          { name: '全参 SFT', pain: '(原点) 只有 (问, 答), 最直接是全参 SFT', fix: '1 份权重、每步 1 次前向; 但梯度和 Adam 的 2 份状态都要装, 每个任务落盘一整份' },
          { name: 'LoRA / QLoRA', pain: '显存装不下梯度和 Adam, 每个任务一整份权重太占地方', fix: '只训旁挂的低秩小矩阵 adapter: 落盘 76 KiB (整模型 389 KiB); QLoRA 再把冻结基座压成 4 bit 的 NF4, 只剩 59 KiB' },
          { name: 'DPO', pain: '(问, 答) 只教 "该说什么", 教不了 "哪个更好"', fix: '吃成对偏好: 要常驻 policy 和冻结的 ref 两份, 共 778 KiB, 每步 2 次前向' },
          { name: 'SimPO / ORPO', pain: 'ref 多占一份权重、多跑一次前向', fix: '去掉 ref: 389 KiB、1 次前向, 每步快约 25%; 防漂移的锚得另找 (SimPO 靠小 lr 和少步数, ORPO 靠 loss 里自带的 SFT 项)' },
          { name: 'GRPO / PPO', pain: '偏好对是离线的; 答案能被程序判对时, 这个信号没用上', fix: 'GRPO 一题在线采 G 条, 交给 verifier (判对错的程序) 打分; PPO 再养一个逐 token 估分的 critic: 常驻三份, 耗时约两倍' },
        ],
      },
      code: 'llm_finetune/run_finetune/{sft,lora,dora,qlora,merge,rm,prm,dpo,kto,simpo_orpo,rlaif,ppo,grpo,distill,on_policy_distill}/train_*.py',
      points: [
        { title: '两个维度可以自由组合', body: '结论: 选方法要分两件事选, 一件是「优化什么目标」, 一件是「更新哪些参数」, 两件可以随便搭。\n- 优化什么目标: SFT、DPO、GRPO 这一类决定 loss 在学什么。SFT (监督微调) 学示范回答, DPO 学成对偏好, GRPO 学 verifier 给的分。\n- 更新哪些参数: LoRA、QLoRA、DoRA 这一类叫 PEFT (parameter-efficient fine-tuning, 只训少量参数的微调), 决定梯度流进哪些参数。LoRA 冻住原权重, 只训旁挂的低秩小矩阵。QLoRA 再把冻住的基座压成 4 bit。DoRA 把权重的「长度」和「方向」拆开分别学。\n两者互相正交: 目标选定了, 参数那一维还能任选。LoRA-SFT、LoRA-DPO 都很常见。\n代码也按这个分: methods/ 放可复用的算法 (loss 和模型包装), run_finetune/ 放一次实验的编排 (造数据、训练、在留出集上验收)。' },
        {
          title: '代价写在三处',
          key: true,
          body: '结论: 每种方法的代价都能用三个数写清楚: 每步跑几次前向、常驻几份权重、最后落盘什么。先看手上数据的形态, 能砍掉一半候选; 再看这三项, 又能砍掉剩下的一半。\n- 每步 LM 前向次数: DPO 2 次, SimPO / ORPO 1 次, GRPO 系还要先在线采 G 条回复。\n- 常驻权重份数: DPO 要 policy 加 ref 两份, 共 778 KiB; 没有 ref 的只要 389 KiB。本仓库模型有 99,648 个参数, 每个 fp32 (32 位浮点) 占 4 字节, 一份就是 389 KiB。\n- 落盘产物: 训完存下来交付的东西。可能是全量权重, 可能只是 adapter, 也可能多一个 value head (把隐状态变成一个分数的线性层, RM 和 PRM 会多存这一层)。\n前向次数怎么数: 一个模型跑一遍算 1 次, 不按序列条数算。DPO 的 2 次是 policy、ref 各跑 1 次, 每次喂 2B 条序列 (B 对偏好, 每对有 chosen 和 rejected 两条)。',
        },
        { title: '别只存 merge 后的权重', body: '结论: LoRA 训完, 要把 adapter 单独存一份, 别只留合并后的完整权重。\nadapter 合并是指训完把 LoRA 的改动 $(\\alpha/r)BA$ 加回原权重, 变回一个普通模型。合并很方便, 但如果只留合并后的结果, LoRA 省下的存储就没了。\nLoRA adapter 只有 $r\\cdot(d_{\\text{in}}+d_{\\text{out}})$ 个参数。这个式子在说: 每个被改的线性层只多两个小矩阵, A 是 $r \\times d_{\\text{in}}$, B 是 $d_{\\text{out}} \\times r$, 秩 r 远小于层的宽度。本仓库 r=8, adapter 76 KiB, 整模型 389 KiB。\n- 存 adapter: 能独立分发。同一个基座可以挂多个 adapter, 按请求热切换。\n- 只留 merge 后的完整权重: 退化成每个任务一份大模型。\n落盘也不只是权重。本章的模型交付实验说明, 交付包 (把模型交给另一个进程或另一个人时一起打包的全部文件) 还要带上架构配置、词表与特殊 token、模板、生成参数、基座标识和量化格式。真实的 adapter 还要绑定基座权重的摘要; 实验里基座对不上的包会被拒绝加载。' },
        { title: 'KTO / RLAIF / PRM: 老结构, 新数据', body: '结论: 这三种方法的训练代价, 都能在前面的老方法里找到原型, 变的是数据从哪来。\n- KTO: 代价同 DPO, policy 和 ref 每步共 2 次前向。它吃单条 👍 / 👎, 不需要同一个问题凑成一对。每次前向喂 2B 条序列: B 条是样本本身, 另外 B 条是错配样本 (把同一个 batch 里的回复整体挪一行, 让每个问题配上别人的回答)。错配样本只用来估参考点 $z_0$。$z_0$ 是对 policy 偏离 ref 有多远的一个估计。所以一半前向花在了估 $z_0$ 上。\n- RLAIF (用 AI 反馈代替人工标偏好): 训练前让 judge (按条文批评、改写回复的评审, 本仓库是规则) 造一轮偏好对。4096 条采样里 judge 批评了 2539 条, 得到 2539 对。之后就是 DPO。\n- PRM (process reward model, 给推理每一步打分的奖励模型): 结构同 RM, 1 次前向, 一份权重加一个 value head。贵在标注: 每条解要 K 个逐步标签 (K 是步数), 只看最终答案的 ORM 只要 1 个。' },
        { title: 'PPO 贵在 critic, 合并根本不训练', body: '结论: PPO 是代价最重的一档, 模型合并是最轻的一档: 一个多养一个网络, 一个根本不训练。\nPPO 比 GRPO 多一个和 policy 同尺寸的 critic。critic 是另外训练的一个网络, 在每个 token 位置估「从这里往后还能拿多少分」。\n- 参数: 要训的翻倍, 到 199,360; 常驻的变成三份 (policy、ref、critic)。\n- 每步前向: 记作「采样 + 3 + 2μ」。\n- 其中的 3: 采样后先冻结三组读数 (旧策略的 log 概率、ref 的 log 概率、critic 的估值), policy、ref、critic 各前向 1 次。\n- 其中的 2μ: 同一批采样要更新 μ 轮 (μ=2), 每轮 policy、critic 各前向 1 次。\n- 耗时: 60 步 11.8 s, GRPO 6.1 s。\n模型合并不训练: 没有梯度, 不要数据, 前向次数记 0, 没有训练显存和训练信号可言。它要的是同一基座上已经微调好的几个模型, 代价花在之前那几次微调里 (本例两次, 各 300 步)。' },
      ],
      links: [
        { from: 'methods/*.py', to: 'run_finetune/*/train_*.py', body: '左边是算法, 右边是一次实验。methods/ 里的算法类 (loss、模型包装) 被训练脚本实例化, 由脚本造数据、喂 batch。脚本还负责在留出集上评估, 最后用断言检查结论是否成立。' },
        { from: 'PairwiseForward / TeacherStudent', to: '通用 Trainer', body: '通用 Trainer 的约定是: 取一个 batch, 调一次模型, 算一次 loss, 更新一次。一步要跑多次前向的方法 (DPO / SimPO / ORPO / RM / KTO / RLAIF / 蒸馏), 把多次前向包进一个 nn.Module: PairwiseForward 一次算完 chosen、rejected 和冻结的 ref, TeacherStudent 一次算完 student 和冻结的 teacher。对 Trainer 来说它就是一个普通模型, Trainer 一行不用改。' },
        { from: 'GRPO / PPO / on-policy 蒸馏', to: '不用 Trainer', body: '这三种方法的数据由当前策略现场采样, 数据生成器得拿到模型本身。GRPO 和 PPO 还要在同一批 rollout (采样出来的回复) 上更新 $\\mu$ 次, 并记住采样时刻的 log 概率。「取 batch, 更新一次」的约定在这里不成立, 硬塞进 Trainer 只会更难读。' },
        { from: 'print_trainable_parameters', to: '验收 PEFT', body: 'loss 在降、显存变小, 都证明不了基座真被冻住了。直接证据是数一数 requires_grad 为 True (会被更新) 的参数有多少。比如 LoRA r=8 时, 可训参数应该是 19,456, 而不是全参的 99,648。' },
        { from: 'adapter state_dict', to: 'llm_infer Multi-LoRA', body: '左边是落盘的 adapter 参数字典, 右边是推理章的 Multi-LoRA: 一份基座同时挂多个 adapter, 不同请求用不同的 adapter。多租户服务 (一套服务给多个客户, 每个客户一个定制模型) 靠的就是 adapter 能独立存、独立切换。' },
      ],
      sourceRows: [
        { concept: '共用基座', code: 'llm_finetune/run_finetune/common.py:pretrained_base', takeaway: '用预训练基座的脚本 (LoRA、DoRA、QLoRA、合并、RLAIF) 都从同一个基座出发: 一个 2 层、d_model=64 的小 LLaMA, 先 SFT 150 步学会 copy 任务 (把 prompt 原样抄一遍, 留出集 EM 要 > 0.95 才放行)。起点相同, 指标之间才可比。' },
        { concept: 'SFT / LoRA / DoRA / QLoRA', code: 'run_finetune/{sft,lora,dora,qlora}/train_*.py', takeaway: '数据是 (x, y): 一个问题配一个示范回答。每步 1 次前向, 不要 ref 也不要采样。四者的区别只在「哪些参数可训」(全部, 还是只训 adapter) 和「基座怎么存」(fp32, 还是压成 NF4)。' },
        { concept: 'RM', code: 'run_finetune/rm/train_rm.py', takeaway: '同一份偏好数据, RM (reward model, 奖励模型) 学的是给回复打一个标量分, 不是生成回复。落盘多一个 value head。RM 用 value head 代替 lm_head 出分, 主干就是传进来的那个 backbone 对象, 训练会原地改掉它的权重; 还要拿它当 policy 用, 就先 deepcopy 一份。' },
        { concept: 'DPO', code: 'run_finetune/dpo/train_dpo.py', takeaway: 'policy 和 ref 两份权重同时常驻, 每步 2 次前向 (各 1 次), 每次 2B 条序列 (B 对偏好的 chosen 和 rejected)。' },
        { concept: 'SimPO / ORPO', code: 'run_finetune/simpo_orpo/train_simpo_orpo.py', takeaway: '同样的偏好数据, 去掉 ref: 常驻权重减半、前向减半, 实测每步快约 25%。不是 50%, 因为反传还在, 那部分开销没少。' },
        { concept: 'GRPO 系', code: 'run_finetune/grpo/train_grpo.py', takeaway: '只要 prompt, 不要示范回答。但必须有 verifier 或 RM 给回复打分, 而且必须能在线采样, 每个 prompt 采 G 条。$\\beta>0$ 时 (β 是 KL 惩罚的权重) 才额外常驻一份 ref。' },
        { concept: '蒸馏', code: 'run_finetune/{distill,on_policy_distill}/train_*.py', takeaway: '都要一个更强的 teacher (老师模型, 冻结, 只提供目标分布) 常驻。离线版从固定语料取数, on-policy 版由 student (被训练的小模型) 现场采样。' },
        { concept: 'PPO', code: 'run_finetune/ppo/train_ppo.py', takeaway: '每题只采 1 条, 但常驻 policy、ref、critic 三份。critic 和 policy 同尺寸而且要训, 所以要训的参数和 Adam 状态都翻倍。' },
        { concept: 'PRM', code: 'run_finetune/prm/train_prm.py', takeaway: '和 RM 同一个「主干 + 标量头」结构, 每步 1 次前向。在每一步推理的末 token 上做 BCE (二分类交叉熵, 判这一步对不对), 所以每条解有 K 个标签。' },
        { concept: 'KTO', code: 'run_finetune/kto/train_kto.py', takeaway: '数据是单条好样本或坏样本, 不用配对。policy 和 ref 各前向 2B 条: B 条是样本本身, B 条是错配样本 (第 i 个问题配第 i−1 条回复), 只用来估 $z_0$。' },
        { concept: 'RLAIF', code: 'run_finetune/rlaif/train_rlaif.py', takeaway: '训前采 1 轮: 每个 prompt 采一条回复, judge 批评、改写, 造出偏好对。训练本身就是 DPO。本仓库的 judge 是规则, 不是模型, 不占训练显存。' },
        { concept: '模型合并', code: 'run_finetune/merge/train_merge.py', takeaway: '不训练: 输入是同一基座的 $\\theta_0$ 和几个微调模型。在 state_dict (参数名到张量的字典) 上逐张量做加减, 落盘一份合并后的全量权重。' },
        { concept: '参数统计', code: 'llm_finetune/utils/param_utils.py:print_trainable_parameters', takeaway: '打印一行「可训 / 总数 (占比)」, 用来验收「真的只训了 adapter」。统计的是 requires_grad 为 True 的参数, 共享权重 (tied embedding) 只计一次。' },
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
        'llm_finetune/methods/ppo.py:PPOTrainer',
        'llm_finetune/methods/prm.py:prm_loss',
        'llm_finetune/methods/kto.py:KTOForward',
        'llm_finetune/methods/rlaif.py:build_preference_pairs',
        'llm_finetune/methods/merge.py:task_arithmetic',
      ],
      run: 'python -m llm_finetune.run_all',
    },
  },
}

// 机制实验的原理、运行入口与源码引用。
const supplements = {
  "finetune-sft": [
    {
      "title": "消息对象还不是 token",
      "body": "调用聊天 API 时, 你传进去的是一串消息对象, 每条带 role (角色: system、user、assistant、tool) 和 content (正文)。模型只认 token。中间这一步由 Chat Template (对话模板) 完成: 它把每条消息写成「角色头 + 正文 + 结束标记」, 一条接一条拼成一个 token 序列。推理时最后再接一个 assistant 的角色头, 叫生成前缀, 意思是「轮到模型说了」。\n训练和推理必须用同一套模板: 训练时 assistant 回复前面长什么样, 推理时给模型的前缀就得逐位一样, 否则模型看到的是没见过的格式。\n- 结束标记有两种: 普通回复以 EOS 结束; 发起工具调用的那一轮以 END 暂停, 等工具结果回来后 assistant 再接着说。\n- demo 的模板是字节级的教学版: 正文按 UTF-8 字节编码, 每个字节加 32 当 token id, 0 到 8 留给控制标记。它不冒充任何厂商的 tokenizer。\n- 控制标记只能由模板写入: 正文里写下字符串「EOS」, 编码出来还是三个普通字节, 不会变成结束 token。\n- 一个七轮的对话样本共 172 个 token; 训练序列和推理时的 assistant 前缀逐位一致。",
      "source": "llm_finetune/data/chat_template.py:serialize"
    },
    {
      "title": "多轮对话只教 assistant 说的话",
      "body": "一段多轮对话里, 只有 assistant 说的话是模型要学的。所以只监督 assistant 的正文和它的结束标记, 其余位置的 labels 都填 −100。七轮样本的 172 个 token 里有 75 个监督位置, 其中 2 个是被监督的 EOS。\n工具相关的内容要分两半看:\n- 工具调用: 是 assistant 自己做出的动作 (调哪个工具、传什么参数), 算 loss。\n- 工具结果: 是外部返回给模型看的观察, 不是模型该生成的内容, labels 全填 −100。\n左移的边界和上文一样: labels[t] = ids[t+1], 所以 assistant 回复第一个 token 的目标, 落在它前一个位置 (角色头的最后一格) 上。这一格必须留着算 loss, 不能丢。\npadding 要用两张 mask 分开管: loss mask 决定哪些位置算 loss; attention mask 只屏蔽 padding 本身。用户输入不算 loss, 但模型照样要看得见它。\n三类坏样本直接拒绝, 不硬凑:\n- 角色错: 把工具结果伪装成 user 消息。\n- 没配对: 工具调用之后没有对应的工具结果, 或者结果的调用 id、工具名对不上。\n- 超长: 上限设成 40 时样本放不下。整条拒绝, 不把工具 JSON 截掉半段。\n本实现每一轮只支持一个工具调用。",
      "source": "llm_finetune/data/chat_template.py:training_example"
    },
    {
      "title": "RFT: 样本来源变了, loss 没变",
      "body": "RFT (rejection-sampling fine-tuning, 拒绝采样微调) 的做法是: 让模型自己先做题, 只留下做对的那些, 再拿它们做普通的监督训练。它和示范 SFT、在线 RL 的区别只在训练样本从哪来。train_rft.py 让三个分支从同一个小 LLaMA 初始化出发, 并排比较:\n- 示范 SFT: 16 条人工写的成功轨迹, 拆成 32 个状态/动作样本 (每个样本是「看到这个状态, 该做这个动作」), 做 40 步交叉熵 (CE) 更新。\n- RFT: 先冻结当前策略 (当前这一版模型), 每个任务让它采 16 次, 共 256 条轨迹。只留下经过独立执行验证、判为合格的 18 条 (覆盖 16 个任务里的 10 个), 同样做 40 步 CE。\n- 在线 RL: 每一轮用当前策略采 8 条, 共 32 轮、256 条, 直接在当轮的采样上算策略梯度来更新。\n结果: 三个分支在留出集上的通过率都从 0 到 1, RFT 没有胜过 SFT 或 RL。\n读这个结果要注意两点:\n- 算力不对等: RFT 和 RL 的候选数相同, 但更新步数和前向次数不同, 算力预算并不公平。\n- 测的是协议迁移, 不是语义泛化: 留出集只换了实体名, 而编码器只看协议状态和值域, 实体名根本没进模型。",
      "source": "llm_finetune/run_finetune/rft/train_rft.py:run"
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
topic.chapters.push({"route": "finetune-knowledge-update", "label": "知识更新 · 编辑、隐私与遗忘", "hint": "改对一个事实, 再查相近的事实被带偏了多少"})
topic.pages["finetune-knowledge-update"] = {
  "title": "知识更新 · 编辑、隐私与遗忘",
  "subtitle": "读完你能分清一次「知识更新」该改文档、改参数还是删记录, 并说出改完参数后要分别验收哪几项。",
  "tldr": "- 先分清要改哪里: 外部文档变了, 就更新检索库; 用户要求删除记录, 就走记忆治理把记录删掉; 只有存在模型参数里的事实, 才需要做知识编辑。\n- 改参数要分四项验收: 原问题答对没有 (efficacy)、换个问法还对不对 (paraphrase)、无关事实有没有被改坏 (locality)、需要用到新事实的推理跟上没有 (portability)。demo 里三种更新都把目标改到了 7.0, 意思相近的邻居事实却分别被带偏了 10.8、4.8、0.0475。",
  "question": "把一个事实改对了, 换个问法、别的事实、靠它推出来的结论, 是不是也都跟着对了?",
  "code": "llm_finetune/run_finetune/knowledge_edit/demo.py · llm_finetune/run_finetune/privacy_unlearning/demo.py",
  "points": [
    {
      "title": "编辑和 SFT 的目标不同",
      "key": true,
      "body": "第一步是确定要改的东西在哪:\n- 文档变了: 走检索更新, 换掉检索库里的原文。\n- 用户要求删除: 走外部记忆治理, 把应用存的记录删掉。\n- 模型参数里记着的某个事实错了: 这才进知识编辑实验。\n知识编辑和 SFT 的目标不同。SFT 学的是一类任务的整体分布; 知识编辑只想改动一个事实, 别的都不碰。所以验收要拆成四项:\n- efficacy: 原来那个问题, 现在答出新值了吗。\n- paraphrase: 换个说法、用别名问, 还答得对吗。\n- locality: 无关的事实有没有被顺带改掉。\n- portability: 需要用到新事实的关联推理, 跟着更新了吗。\n不能用一个命中率代替这四项。只在原问题上改对了, 不等于模型的知识整体可靠地更新了。"
    },
    {
      "title": "删除数据不等于参数遗忘",
      "body": "删掉原文、索引和缓存, 改的是应用这一侧的存储。训练样本已经在参数里留下的影响, 并不会跟着消失。\nprivacy_unlearning 用 8 条合成记录训了一个线性模型, 每条记录配一个 one-hot 键 (只有一位是 1 的向量)。直接拿待删记录的键去查, 就读出了它的值 0.198 (真值 0.2)。成员推断攻击 (判断某条记录在不在训练集里) 的 TPR 是 1.0, FPR 是 0.0。TPR 是训练集里的记录被认出来的比例, FPR 是不在训练集里的记录被误认的比例; 这组读数说明攻击一条不漏、一条不错。\n要证明参数真的遗忘了, 得拿三样东西对照: 遗忘集 (要删的记录)、保持集 (要留的记录)、以及删掉这条记录后从头独立重训的参照模型。demo 的模型是岭回归, 能直接从它的充分统计量 (足以算出解的那几个汇总量) 里把这条记录的贡献扣掉。扣完和独立重训逐元素核对, 差在 1e-12 以内; 待删记录的输出变成 0.0, 保持集 MSE 还是 0.00011, 和删除前一样。这种精确扣除只适用于这类有闭式解的模型, 一般的 LLM 做不到。\n模型在目标题上答错一次, 不算遗忘的证据。"
    },
    {
      "title": "差分隐私与遗忘不是同一保证",
      "body": "DP (差分隐私) 在训练时下手。做法是: 每个样本的梯度先裁剪到长度不超过 1, 再给平均梯度加上 σ=4 的高斯噪声。这样任何单个样本对训练结果的影响都有上限。跑 40 步后能算出一个分析上界 ε≈9.56 (δ=1e-6)。ε 限制的是单个样本对输出分布的影响: ε 越小, 训练集里有没有这条样本, 输出分布就越难区分。代价是精度: 保持集 MSE 从 0.00011 涨到 0.343。\n遗忘在训练后下手, 处理已经学进参数的影响。两者谁也替代不了谁。\nDP 路径上成员推断的 TPR 降到了 0.125。这只说明这一个攻击没成功, 不是隐私证明; 隐私保证要靠 ε 背后的那套假设 (裁剪、噪声、步数) 去论证。\n这个玩具噪声实验不提供完整的隐私会计 (把每一次数据访问的隐私开销都记账加总), 也不提供共享模型上按用户删除的保证。"
    },
    {
      "title": "怎样验收",
      "body": "knowledge_edit 把一个 1×3 的线性记忆 W 当成「模型」: 输入一个 3 维的键, 输出一个值。目标是把键 [1, 0, 0] 的输出从 1 改成 7。对同一个目标做三种更新, 每种都同时报四个数: 目标的新输出; 相关邻居 (和目标很像的键 [0.8, 1, 0]) 的输出变了多少; 无关事实 (键 [0, 0, 1]) 变了多少; 别名 (键 [0.95, 0.05, 0]) 的回答。\n- 全局改 (W 的每个元素都加 6): 目标 7.0, 邻居变 10.8, 无关事实变 6.0, 别名回答 7.05。\n- 秩一编辑 (只沿目标键的方向改 W): 目标 7.0, 邻居 4.8, 无关 0.0, 别名 6.75。\n- 约束邻域 (求解时给邻居方向加 100 倍的保护权重, 写作协方差加 100·nnᵀ): 目标 7.0, 邻居 0.0475, 无关 0.0, 别名 6.51。\nprivacy_unlearning 同样对三条路径 (只记忆 / 加 DP / 删除) 报四类读数: 保持集 MSE、待删记录的输出、成员推断 TPR/FPR、ε。\n验收看的是整张对照表, 不是某一个成功的例子。本例没测 portability; 别名向量是直接给定的, 别名答对不能代替真实语言里换问法的泛化。"
    }
  ],
  "links": [],
  "sourceRows": [
    {
      "concept": "知识编辑: 三种更新对照",
      "code": "llm_finetune/run_finetune/knowledge_edit/demo.py:experiment",
      "takeaway": "experiment 按三种模式 (全局改、秩一编辑、约束邻域) 各跑一次。三种都把目标改到了 7.0, 区别在副作用: 相关邻居分别变了 10.8、4.8、0.0475, 无关事实分别变了 6.0、0.0、0.0。"
    },
    {
      "concept": "隐私: 记忆、DP、删除三条路径",
      "code": "llm_finetune/run_finetune/privacy_unlearning/demo.py:experiment",
      "takeaway": "experiment 跑三条路径: 只记忆、加 DP、删除。成员推断的 TPR (训练集里的记录被认出来的比例) 依次是 1.0、0.125、0.875。删除路径的结果和独立重训逐元素核对, 差在 1e-12 以内。"
    }
  ],
  "snippetTitle": "先固定要改什么, 再检查没有改什么",
  "snippet": "result = experiment(mode)\n# 同时检查目标变化、无关任务保持与独立验证\nprint(result[\"metrics\"])",
  "source": [
    "llm_finetune/run_finetune/knowledge_edit/demo.py:experiment",
    "llm_finetune/run_finetune/privacy_unlearning/demo.py:experiment"
  ],
  "run": "python -m llm_finetune.run_finetune.knowledge_edit.demo"
}
topic.pages["finetune-runs"].points.push({"title": "模型交付: 新进程里还是同一个模型吗", "body": "训练完的模型要交给另一个进程去加载, 比如推理服务。权重一样, tokenizer 或对话模板不一样, 新进程把同一句话编码出的 token 就不一样, 模型看到的输入也就变了。所以交付包要带上模型「怎么解释输入、怎么生成输出」的全部约定: 架构配置、权重、词表和特殊 token、模板、生成参数、基座标识、量化格式。如果还要恢复训练, 再加上优化器状态和随机数生成器 (RNG) 的状态。\nartifact_delivery 训一个小 LLaMA (25 步), 再在一个新的 Python 进程里加载, 用固定输入核对三样: token、logits、greedy 输出 (每步都取概率最大的 token 生成的结果)。\n- 完整 FP32: logits 最大误差 0.0, token 和 greedy 输出都一致。\n- int8 反量化 (权重存成 8 位整数, 加载时还原成浮点): 误差 0.008654, 要单独定容限。这只是反量化误差实验, 不能说成整数算子带来了加速。\n- 词表错配、模板错配、基座错配: 加载时直接拒绝。表里误差显示 0, 意思是根本没跑推理, 不是结果对了。\n文件摘要 (对文件内容算的哈希) 只能发现文件损坏, 认证不了来源。实验里把词表中 a、b 的顺序对调, 再重算一遍摘要, 摘要校验照样能过; 真正拒绝它的, 是可信 manifest (交付清单) 里记下的语义约定。"})
topic.pages["finetune-grpo-variants"].points.push({"title": "异步 RL: 样本来自哪一版策略", "body": "异步 RL 里, 采样和训练同时进行: 采样器不停往队列里放 rollout, 训练器不停取出来更新。训练器把策略从 v1 更新到 v3 时, 队列里可能还躺着 v1 采出的答案。权重同步只影响之后提交的新任务, 已经采出的答案不会自动变成 v3 的样本。\n所以每条 rollout 在采样时就要记下两样东西: 当时的策略版本, 和行为 logp (采样那一刻, 采样用的策略给这个回答的 log 概率)。\nasync_rollout 用一个能直接算期望的小例子: 两个动作, 奖励分别是 0 和 1。旧策略 (行为策略) 选它们的概率是 0.8 和 0.2, 新策略 (目标策略) 是 0.2 和 0.8, 所以目标期望奖励是 0.8。四种处理办法:\n- 直接用旧样本: 期望 0.2, 学到的还是旧策略的分布。\n- 重要性校正 (每条样本乘上新旧概率之比): 期望回到 0.8, 但相对 ESS 只剩 0.3077。ESS (有效样本量) 衡量权重变得不均之后, 这批样本还顶多少条等权样本用; 0.3077 就是只顶约三成, 估计更抖。\n- 裁剪权重 (把比率限制在 [0.8, 1.2]): 期望 0.24, ESS 0.968。用偏差换来了稳定。\n- 拒收过期样本: 期望 0, 意思是这一步没有更新, 不是奖励降为零。\n校正时的分母必须是实际采样时记下的概率, 不能用当前模型重算一遍代替: 生成引擎的温度、精度、logits 处理都可能和训练器不同。\n另外三个机制各管一件事: 队列容量 (这里是 2) 用背压限制积压: 队列满了, 新提交的 rollout 直接被退回, 不往里塞 (本例触发 1 次); 过期拒收限制新旧策略偏离多远; 取消的任务要把计数收回来, 本例未回收的任务是 0。"})
topic.pages["finetune-sft"].related = [{"route": "agent-trajectory-learning", "label": "RFT: 先采样, 再筛选, 再做 SFT", "reason": "本章只说 RFT 换了训练样本的来源。SFT、RFT、RL 三个分支怎么并排跑、每一步留下了哪些执行记录, 在那一章。"}, {"route": "finetune-knowledge-update", "label": "知识编辑: 改对一个事实, 伤到多少邻居", "reason": "SFT 学的是一类任务的整体分布。只想改对模型记住的某一个事实时, 用的是另一套算法和四项验收, 在那一章。"}]
topic.pages["finetune-rlhf"].related = [{"route": "basic-optim-sample", "label": "复习 softmax 温度", "reason": "同一个温度公式: 基础页演示它怎么改变分布, 本章讲它在采样和策略优化里怎么用。"}]
topic.pages["finetune-knowledge-update"].related = [{"route": "agent-memory-privacy", "label": "应用记忆的删除", "reason": "删掉应用里存的一条记忆记录是应用层的事, 那一章讲授权、过期和删除。参数里留下的影响, 要用本章的办法另外遗忘、另外验证。"}]
topic.pages["finetune-knowledge-update"].evolution = {
  "title": "改一个事实, 先数邻居伤了多少",
  "subtitle": "根问题: 事实存在共享的参数里, 改一个数会牵动所有用到它的键",
  "steps": [
    { "name": "改文档不改参数", "pain": "(原点) 事实变了, 模型参数里存的还是旧值", "fix": "版本化更新原文与索引, 让旧缓存失效; 参数里的旧知识原样留着" },
    { "name": "全局改参数", "pain": "问题没经过检索, 或检索没命中, 模型照旧按参数答", "fix": "demo 把 W 的每个元素都加 6: 目标到 7.0, 但无关事实也变了 6.0, 邻居变 10.8" },
    { "name": "秩一编辑", "year": 2022, "pain": "全局改动把无关事实也改了 6.0", "fix": "ROME 式秩一更新 ΔW = u vᵀ (两个向量的外积), 只沿目标键的方向改: 无关事实 0.0, 相近的邻居仍动 4.8" },
    { "name": "约束邻域", "pain": "和目标键相近的邻居被连带改了 4.8", "fix": "求解时给邻居方向加 100 倍保护权重 (协方差加 100·nnᵀ): 邻居只动 0.0475; 别名回答降到 6.51, 目标与邻居重合时无解" },
    { "name": "删除对照重训", "pain": "改值能让目标答新值; 删除请求要的是记录影响消失的证据", "fix": "从充分统计量扣掉记录, 与独立重训核对到 1e-12; 再报遗忘集、保持集和成员推断" }
  ]
}
export default topic

mergePage(topic, 'finetune-dora', 'finetune-lora', '进阶 · DoRA: 权重的长度和方向分开学', ["FtDoraLab"])

topic.pages['finetune-lora'].title = '参数高效微调 · LoRA 与 DoRA'
topic.pages['finetune-lora'].subtitle = '读完你能说清 LoRA 为什么只训 A、B 两个小矩阵, 以及 B 为什么从 0 起步; 进阶段再看 DoRA 怎样把权重的每一行拆成长度 (幅度) 和方向, 分开来训。走冲刺路线的话, 读完 LoRA 部分即可。'

for (const chapter of topic.chapters) chapter.label = chapterLabels[chapter.route] || chapter.label

canonicalizeRelated(topic)

