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
      subtitle: '监督微调沿用预训练的交叉熵损失, 关键改动是把用户提问区间屏蔽为 -100; 一旦在错位后的标签上多盖一格, 模型就会丢失开口说第一个词的监督信号。',
      tldr: 'SFT (监督微调) 在 (prompt, response) 拼接序列上继续训练下一词预测, 但只对回复区间计入交叉熵损失:\n- 屏蔽规则: 将提问内部位置的 labels 置为 -100, 交叉熵自动跳过这些位置的梯度与分母计数。\n- 错位边界: 由于 labels 相对输入已整体左移一格, 长度为 $P$ 的 prompt 只能屏蔽前 $P-1$ 格; 若误盖满 $P$ 格会把首个回复 token 一并抹去, 导致留出集整条匹配率 (EM) 从 1.000 跌至 0.000。',
      question: 'labels 左移错位之后, prompt 掩码的右边界为什么是 $P-1$ 而不是 $P$? 为什么多盖一格从训练 loss 曲线上几乎看不出异常?',
      evolution: {
        title: '从接龙到只答不问',
        subtitle: "根问题: 预训练仅对全序列做无差别的下一 token 预测, 未区分用户输入区间与模型应当生成的回复区间。",
        steps: [
          { name: '整条序列接着训', pain: "(原点) 预训练基座只学习文本续写, 输入单条问题时可能继续续写下一个问题而非作答", fix: "将问答对拼接成单条序列继续做下一 token 交叉熵训练: 模型能学会问答格式, 但提问区间同样参与了 loss 计算" },
          { name: 'prompt 置 -100', pain: "提问区间的 loss 在训练模型拟合用户提问分布, 稀释了回复区间分配到的有效梯度", fix: "将 prompt 内部位置的 labels 设为 -100, 使交叉熵既不计算该处 loss, 也不将其计入平均分母" },
          { name: '回复带上 EOS', pain: "若仅监督回复正文内容, 模型缺乏终止信号, 生成时无法自主决定何时停止", fix: "将结束符 EOS 纳入回复区间一并监督: 含 EOS 在内长度为 R 的回复共有 R 个位置参与 loss 计算" },
          { name: '边界 P-1 + 断言', pain: "由于 labels 已相对输入左移一格, 若屏蔽满前 P 格会连带抹去首个回复 token 的监督, 且从 loss 曲线上难以察觉", fix: "在 make_labels 中仅屏蔽前 $P-1$ 格以保留首词监督, 并添加断言校验有效监督位置数等于回复长度" },
        ],
      },
      code: 'llm_finetune/data/tasks.py · llm_finetune/data/instruction_data.py · llm_finetune/methods/sft.py',
      points: [
        { title: 'SFT 复用预训练交叉熵, 差别全在标签掩码', body: 'SFT 并没有引入新的损失函数: 代码里 SFTLoss 直接复用带 ignore_index=-100 的 StandardLMLoss。\n预训练对整段文本的每个位置都计算预测损失, 模型学到的是无差别续写; SFT 则把用户提问部分的标签置为 -100, 让交叉熵同时在分子和分母中忽略这些位置, 把梯度全部集中在「读完问题后如何组织回答与何时输出结束符 EOS」上。' },
        {
          title: '标签左移后, 提问掩码边界必须停在 P−1',
          key: true,
          body: '长度为 $P$ 的 prompt 只能屏蔽前 $P-1$ 个标签位置, 第 $P-1$ 格必须保留用于监督首个回复词。\n原因来自自回归错位: 模型在下标 $t$ 处的输出预测的是原序列下标 $t+1$ 的 token, 数据构造函数已提前令 labels[t] = x[t+1]。以 [x1, x2, SEP, y1, y2, EOS] ($P=3$) 为例:\n- 下标 0 预测 x2, 下标 1 预测 SEP (均属提问内部转移, 置 -100)。\n- 下标 2 (SEP 所在位置) 预测的恰好是第一个回复词 y1, 必须保留计算损失。\n因此掩码切片应写为 labels[:, :P-1] = -100。make_labels 末尾专门加了断言 assert (labels != IGNORE).sum(1) == n_response, 一旦误写成 :P 少算一个位置便会当场报错。',
        },
        { title: '多盖一格不报异常, 却废掉了首词生成', body: '若没有长度断言而误盖满前 $P$ 格, 训练曲线往往看起来一切正常: 回复共有 $R$ 个 token (含 EOS), 剩下的 $R-1$ 个位置依然在正常降 loss。然而模型从未在「刚读完分隔符 SEP」的位置收到过监督梯度, 推理生成第一个词时只能随机乱猜——首词一错, 后续接龙再顺畅整道题也会判错。\n在 6-token 序列反转任务上同配置对照: 正确掩码 (:P-1) 在留出集上整条匹配率 (EM) 为 1.000, 多盖一格 (:P) 直接跌至 0.000; 若遇上单 token 简答题, 唯一监督位被抹掉更会导致分母归零算出 nan。' },
      ],
      links: [
        { from: '预训练 CE', to: 'SFTLoss', body: 'SFTLoss 与预训练交叉熵是同一个类, 仅通过把 labels 中的提问区间填为 -100 切换训练目标。' },
        { from: 'make_labels', to: 'DPO / SimPO / ORPO', body: '后续偏好对齐算法均依赖这套 labels 汇总整条回复的对数似然 $\\log\\pi(y\\mid x)$: 一旦错盖掉下标 $P-1$, 好坏两条回复在首词上的概率差异 $\\log p(y_1\\mid x)$ 也会一并丢失。' },
        { from: 'SFT 终态', to: 'policy / reference', body: 'DPO 与 RLHF 训练需要一份具备指令遵循基础的起点模型: 在训的 policy 与冻结的参照模型 ref 均从 SFT 检查点复制初始化。' },
      ],
      sourceRows: [
        { concept: '错位', code: 'llm_finetune/data/tasks.py:make_labels', takeaway: '通过 seq[:, 1:] 将标签整体左移一格并在末尾补 -100, 数据侧一步到位完成输入与目标的对齐。' },
        { concept: 'prompt mask', code: 'labels[:, :P-1] = -100', takeaway: '仅屏蔽提问内部的 token 转移, 保留下标 $P-1$ 处对首个回复词的预测监督。' },
        { concept: '差一位的守卫', code: 'assert (labels != IGNORE).sum(1) == n_response', takeaway: '强制核对有效监督位数等于含 EOS 在内的回复长度, 第一时间拦截差一错误 (off-by-one)。' },
        { concept: 'ignore_index', code: 'llm_finetune/methods/sft.py: SFTLoss = StandardLMLoss', takeaway: '交叉熵遇到 -100 时既不累加损失也不计入平均分母。' },
        { concept: '验收', code: 'llm_finetune/data/tasks.py:exact_match', takeaway: '在按 sum(prompt) mod 5 == 0 严格隔离的留出集上自回归生成完整回复, 逐 token 全对才计 1 分。' },
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
      subtitle: '全参数微调需要为每个权重配备梯度与优化器状态, 多任务分发还要保存多份完整模型。LoRA 冻结原始权重并在旁路并联两个极瘦的低秩矩阵, 大幅压缩训练状态与落盘体积。',
      tldr: 'LoRA (Low-Rank Adaptation) 保持预训练权重 $W$ 冻结不变, 在旁路并联低秩增量 $y = Wx + (\\alpha/r)BAx$, 仅更新小矩阵 $A$ 与 $B$:\n- 核心收益: 省去基座权重的梯度与双份 Adam 动量显存, 单个下游任务只需保存轻量的 adapter 文件, 推理前还可合并回 $W$ 实现零额外延迟。\n- 能力折中: 同样步数下 LoRA 收敛更慢; 在本库 copy -> sort 300 步对照中, 全参微调留出集 EM 达 0.809, 而 LoRA (r=8) 调大步长后为 0.352。模型太小, 这组差距说明不了低秩容量够不够用。',
      question: 'LoRA 真正省下的是哪几块显存与存储开销? 为什么初始化时必须令 B 全零而 A 随机?',
      evolution: {
        title: '从改整张矩阵到挂一条旁路',
        subtitle: "根问题: 全参数微调需为每个参数分配梯度与两份 Adam 优化器状态, 多任务部署时还需为每个任务保存整份权重。",
        steps: [
          { name: '全参微调', pain: "(原点) 将预训练模型适配到新任务时, 最直接的做法是放开全部参数参与更新", fix: "每个参数均需配备 1 份梯度与 Adam 优化器的 2 份状态 (本仓库占 778 KiB), 服务 N 个任务还需落盘 N 份完整模型" },
          { name: 'Adapter', year: 2019, pain: "训练状态显存与落盘存储均随总参数量线性增长, 降低开销只能只训练极小比例的参数", fix: "在每层主干内串联轻量瓶颈 MLP 并仅训练该模块; 但串联模块留存在前向路径中无法合并消除, 推理时每层都需多算一段" },
          { name: 'Prefix Tuning', year: 2021, pain: "串联的 adapter 模块无法合并移除, 在小 batch 推理时会直接抬高逐层延迟", fix: "保持网络层结构不变, 仅优化拼接在输入前端的可学习前缀向量; 但前缀参数较难收敛, 且直接挤占有效上下文长度" },
          { name: 'LoRA', year: 2021, pain: "前缀微调挤占上下文窗口且较难优化, 串联 adapter 又会引入额外的推理延迟", fix: "在冻结权重 W 旁并联低秩支路 $BA$ 并仅训练 A 与 B; 训练完成后将旁路合并回 W, 线上推理零额外开销" },
        ],
      },
      code: 'llm_finetune/methods/lora.py · llm_finetune/methods/qlora.py · llm_finetune/utils/param_utils.py',
      points: [
        { title: '低秩假设针对权重增量 ΔW, 而非基座 W 本身', body: 'LoRA 并不要求预训练权重 $W$ 低秩, 而是假设下游适配所需的改变量 $\\Delta W$ 集中在一个低维子空间内, 可分解为两个窄矩阵的乘积 $B\\,(d_{\\text{out}}\\times r) \\cdot A\\,(r\\times d_{\\text{in}})$, 其中秩 $r \\ll \\min(d_{\\text{in}}, d_{\\text{out}})$。\n单层可训练参数量由 $d_{\\text{in}}d_{\\text{out}}$ 缩减为 $r(d_{\\text{in}}+d_{\\text{out}})$: 例如在 $d=4096, r=8$ 的真实规模下, 单层可训参数从 16M 降至 65K (仅占 0.39%)。不过当目标任务与预训练分布跨度较大时, 所需更新往往偏离低秩子空间, 此时过小的 $r$ 就会成为容量瓶颈。' },
        { title: 'B 置零搭配 A 随机初始化, 实现无损冷启动', body: '旁路采用不对称初始化——$B$ 全零、$A$ 按 Kaiming 均匀分布随机初始化, 同时兼顾起点等价与梯度破缺:\n- 第 0 步输出不变: 因 $B=0$ 故 $BA=0$, 注入旁路后的初始前向结果与原基座模型逐位相等, 不会在起步时用随机噪声冲坏预训练表征。\n- 第 1 步即可更新: 反向求导时 $B$ 的梯度正比于非零的 $Ax$, 首步更新后 $B$ 脱离零点, 紧接着下一步 $A$ 也获得非零梯度。若将 $A, B$ 同时置零, 双方梯度互乘对方零矩阵将陷入鞍点永远无法更新; 若同时随机初始化, 则第 0 步便破坏了基座输出。' },
        {
          title: '省下的是优化器显存与多任务存储, 而非收敛步数',
          key: true,
          body: 'LoRA 的核心工程价值在于降低优化器状态显存与模型分发成本, 并不保证在相同步数下比全参微调学得更快或更准。将同一个已掌握 copy 任务的基座迁移到 sort 排序任务、各训练 300 步对比:\n- 全参微调: 可训参数 99,648 个, Adam 动量状态占 778 KiB, 留出集 EM 达 0.809。\n- LoRA (r=8): 可训参数降至 19,456 个, Adam 状态压到 152 KiB; 即使把学习率从 3e-3 专门调大到 1e-2, 留出集 EM 为 0.352 (沿用 3e-3 时为 0.281)。\n同时需留意显存账的边界: LoRA 省掉了冻结权重的梯度与 Adam 状态, 但由于旁路挂在各层内部, 前向激活与反向传播流经主干的中间张量依然无法省去。',
        },
      ],
      links: [
        { from: 'SFT loss', to: 'LoRA SFT', body: '训练损失与标签掩码无需任何修改, 区别仅在于冻结主干权重后优化器只更新带 lora_ 前缀的旁路参数。' },
        { from: 'target_modules', to: '模型结构', body: 'apply_lora 按模块属性名后缀 (如 w_q, w_v) 匹配注入目标层; 若目标名在模型中一处都未命中则立即抛出 ValueError 防呆。' },
        { from: 'methods/lora.py', to: 'methods/qlora.py', body: '下一章 QLoRA 原样保留这套高精度 A/B 旁路, 进一步把只读不写的冻结基座压缩为 4-bit NF4 格式以削减静态权重显存。' },
        { from: 'adapter', to: 'llm_infer Multi-LoRA', body: '训完若不把 BA 合并进基座, 线上推理服务即可常驻一份共享基座并按请求动态挂载多个不同业务的轻量 adapter; 本章脚本只导出了 adapter, 没有演示多 adapter 切换。' },
      ],
      sourceRows: [
        { concept: 'LoRALinear', code: 'llm_finetune/methods/lora.py:LoRALinear', takeaway: '将线性层包装为冻结的 base 分支与可训的 lora_A、lora_B 旁路之和。' },
        { concept: '注入', code: 'llm_finetune/methods/lora.py:apply_lora', takeaway: '遍历模块树把匹配 target_modules 的 nn.Linear 原地替换为 LoRALinear。' },
        { concept: '只训 adapter', code: 'llm_finetune/methods/lora.py:mark_only_lora_as_trainable', takeaway: '关闭所有非 lora_ 参数的 requires_grad, 并配合 print_trainable_parameters 核验可训参数数量。' },
        { concept: 'merge', code: 'llm_finetune/methods/lora.py:merge_lora_weights', takeaway: '按 $W \\leftarrow W + (\\alpha/r)BA$ 将旁路合入原权重并换回普通线性层, 并断言合并前后输出 logits 一致 (阈值 1e-4, 实测最大误差 6.4e-06)。' },
        { concept: '落盘', code: 'llm_finetune/methods/lora.py:get_lora_state_dict', takeaway: '仅导出旁路权重字典: 本库单个 adapter 仅 76 KiB, 相比 389 KiB 的整模型缩减近五分之四。' },
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
      subtitle: 'LoRA 消除了基座的梯度与优化器状态后, 常驻显存的大头就只剩下只读的基座权重本身。QLoRA 用贴合正态分布分位数的 4-bit NF4 存储基座, 在前向计算时按需反量化。',
      tldr: 'QLoRA 在保留高精度 LoRA 旁路的同时, 将冻结基座压缩为 4-bit NormalFloat (NF4) 格式:\n- 分位数码本: 神经网络权重近似服从零均值正态分布, NF4 将 16 个码点摆在标准正态分布的等概率分位点上 (零附近密、两端疏), 配合每 64 个权重一块的缩放因子压低量化失真。\n- 整网压缩账: 本库模型权重体积从 389 KiB 压至 59 KiB (压缩 6.57 倍), 基座在原任务上的留出集 EM 仍保持 1.000, 下游适配梯度全程流经 fp32 旁路。',
      question: '同样只有 16 个离散码点, 为什么按正态分位数摆放的 NF4 比等间距 INT4 误差更低? 为什么词嵌入和输出头通常跳过 4-bit 量化?',
      evolution: {
        title: '显存大头一块块往下挪',
        subtitle: "根问题: 微调显存由模型权重、梯度、优化器状态与中间激活共同组成, 消除其中一项后, 剩余最大项便成为新的显存瓶颈。",
        steps: [
          { name: 'LoRA + fp32 基座', pain: "(原点) LoRA 已省去冻结基座的梯度与 Adam 状态, 但基座权重本身仍需常驻显存", fix: "冻结基座仍按 fp32 精度整份常驻: 训练全程只读不写, 却成为静态显存的主要部分 (本仓库占 389 KiB)" },
          { name: '整层 INT4', pain: "基座权重在微调期间全程只读, 继续用 32 bit 存储每个参数造成显存浪费", fix: "整层权重除以最大绝对值 absmax 缩放到 [-1, 1], 再映射到 16 个等间距码点; 但单个离群值便会撑大整层量程" },
          { name: '分 block 缩放', pain: "整层量程被离群值撑大后, 绝大多数中小幅值权重被迫挤入零点附近的少数几个码点", fix: "改为每 64 个权重独立维护一个 absmax 缩放因子: 离群值仅影响所属 block, 平均每参数仅增加 0.0625 字节开销" },
          { name: 'NF4 + LoRA', year: 2023, pain: "预训练权重近似服从正态分布, 均匀分布的等间距码点在稀疏两端几乎闲置", fix: "按标准正态分布等概率分位数选取码点; 前向计算时按需反量化, 梯度仅流经高精度 A/B 旁路, 权重体积从 389 KiB 降至 59 KiB" },
        ],
      },
      code: 'llm_finetune/methods/qlora.py · llm_finetune/run_finetune/qlora/train_qlora.py',
      points: [
        { title: 'NF4 让 16 个码点顺着正态权重的密度分布', body: '4-bit 量化只能用 16 个离散代表值 (码点) 近似连续浮点数。预训练线性层权重普遍呈钟形正态分布: 绝大多数权重密集簇拥在 0 附近, 尾部大值很稀疏。\n等间距 INT4 在 [-1, 1] 上均匀切分, 导致两端码点利用率极低而零点附近分辨率不足; NF4 则取标准正态分布的 16 个等概率分位数 (显式包含精确的 0) 构成固定码表 NF4_CODEBOOK, 使每个码点被命中的概率近似同为 $1/16$, 对近似正态的权重, 期望量化误差比等间距 INT4 更小。' },
        {
          title: '计入分块缩放与未量化层, 核算真实的整模型压缩比',
          key: true,
          body: '评估量化收益必须统计整个模型的实占字节数, 而非仅看单个线性层的理论位宽:\n- 单层理论上限: 每个参数占 4 bit (0.5 字节), 每 64 个参数共享一个 fp32 缩放因子 scale (摊到每参数 $4/64 = 0.0625$ 字节), 合计 0.5625 字节/参数, 相对 fp32 (4 字节) 的理论上限为 $4 / 0.5625 = 7.11$ 倍。\n- 整模型实测: embedding、lm_head 与 RMSNorm 保持 fp32 不做 4-bit 量化, 因此本库整模型从 389 KiB 降至 59 KiB, 实际压缩比为 6.57 倍; 在词表高达十余万的真实大模型中, 未量化的嵌入层占比更高, 整网压缩比还会相应回落一些。',
        },
        { title: '4-bit 只负责静态存储, 前向计算与旁路反传仍走浮点', body: 'QLoRA 量化的是只读基座存储, 而非低精度训练本身:\n- 基座层 (NF4Linear): 将打包后的 uint8 索引与每块 scale 注册为无梯度 buffer, 每次前向时动态查表反量化回浮点权重参与矩阵乘。省下的只是存储: 前向那一刻, 这一层的 fp32 权重照样在内存里。\n- 旁路层 (lora_A, lora_B): 权重与梯度全程保持 fp32 精度。单层约 9.0% 的静态量化误差相当于微调起点函数的轻微偏移, 可由旁路在训练中顺带补偿: 适配 sort 任务 300 步后留出集 EM 为 0.402, 与全精度 LoRA 多随机种子均值 (0.382) 处于同一区间。' },
      ],
      links: [
        { from: 'methods/lora.py', to: 'methods/qlora.py', body: 'QLoRA 完整复用了 LoRA 的 A/B 旁路结构与冻结逻辑, 仅将底层冻结的 nn.Linear 替换为按块压缩的 NF4Linear。' },
        { from: 'nf4_quantize', to: 'nf4_dequantize', body: '量化按 64 元素分块归一化到 [-1, 1], 查最近邻码点并将两个 4-bit 索引拼入单个 uint8; 反量化则拆字节查码表再乘回块 scale。' },
        { from: 'block_size', to: '误差 / 开销', body: '分块大小 $B$ 越小越能隔离局部离群值, 但每参数分摊的缩放因子开销 $4/B$ 字节也随之上升 (QLoRA 原论文可对 scale 再做一次 8-bit 双重量化进一步省空间)。' },
        { from: 'QLoRA', to: 'llm_infer 权重量化', body: '这种「静态 4-bit 存储 + 前向即时反量化乘浮点激活」的 weight-only 思路, 同样是阶段 5 小批量推理解码提速的主力方案。' },
      ],
      sourceRows: [
        { concept: '码本', code: 'llm_finetune/methods/qlora.py: NF4_CODEBOOK', takeaway: '归一化到 [-1, 1] 的 16 个标准正态等概率分位点常量表, 显式保留精确零值。' },
        { concept: '量化', code: 'llm_finetune/methods/qlora.py:nf4_quantize', takeaway: '每 64 个元素除以块内 absmax, 用 argmin 匹配最近码点并将相邻两个半字节打包进 uint8。' },
        { concept: '反量化', code: 'llm_finetune/methods/qlora.py:nf4_dequantize', takeaway: '通过位运算高低半字节解包还原码表数值, 乘回各块 scale 恢复原始权重形状。' },
        { concept: '前向', code: 'llm_finetune/methods/qlora.py:NF4Linear', takeaway: '通过属性按需反量化基座权重计算主路 $\\mathrm{dequant}(W)x$, 再叠加高精度旁路 $(\\alpha/r)BAx$。' },
        { concept: '哪些层放过', code: 'llm_finetune/methods/qlora.py:apply_qlora', takeaway: '仅替换 Transformer 块内的 14 个投影线性层; 查表性质的 embedding、直接输出分数的 lm_head 与一维 RMSNorm 保持原精度。' },
        { concept: '显存账', code: 'llm_finetune/methods/qlora.py:weight_bytes', takeaway: '精确累加未量化层的参数字节与 NF4 层的打包 buffer + scale 字节; 合并落盘时也需先反量化回浮点再叠加 $\\Delta W$。' },
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
      subtitle: '标准 LoRA 将增量直接加到基座权重上, 每一行的模长与方向被迫同向绑定变化。DoRA 引入逐行归一化与独立可训练的幅度向量, 让低秩旁路专注旋转方向。',
      tldr: 'DoRA (Weight-Decomposed Low-Rank Adaptation) 将权重矩阵每一行显式拆解为「幅度 × 单位方向」: $W^\\prime = m \\odot (W_0 + BA) / \\|W_0 + BA\\|_{\\text{row}}$:\n- 解耦分工: 除以行范数消除了 $BA$ 对行长度的影响, 让有限的低秩容量全部用于调整方向, 而每行的模长交由 $d_{\\text{out}}$ 维向量 $m$ 独立学习。\n- 实验表现: 每层仅多出 $d_{\\text{out}}$ 个标量参数, 在同为 r=8 的 3 种子平均对比中, 留出集 EM 从标准 LoRA 的 0.382 提升至 0.522。',
      question: '在秩 r 完全相同的情况下, 为什么仅每行多加 1 个可学习幅度标量就能缩小与全参微调的差距?',
      evolution: {
        title: '把一个增量拆成长度和方向',
        subtitle: "根问题: 在不增加秩 r 的前提下, 如何让低秩旁路调整权重的方式更贴近全参数微调。",
        steps: [
          { name: '全参微调', pain: "(原点) 线性层每行权重兼具模长与方向两个自由度, 微调时往往需要分别调整二者", fix: "全量更新可独立调节模长与方向 (DoRA 论文测得二者变化呈负相关), 但每个参数均需分配梯度与 Adam 状态" },
          { name: 'LoRA', year: 2021, pain: "全参数微调显存开销过高, 只能改用低秩增量矩阵更新权重", fix: "$W_0 + BA$ 省去了绝大部分可训参数; 但直接相加使行模长与行方向被迫同向变化 (论文测得呈正相关)" },
          { name: 'DoRA', year: 2024, pain: "若想只旋转方向而不改变模长, 低秩增量 $BA$ 必须恰好保持行范数不变, 低秩子空间难以满足这一约束", fix: "仿照 Weight Normalization 改写为 $m \\cdot V / \\|V\\|$: 方向更新交由低秩旁路 BA, 每行模长交由独立标量 m 学习" },
          { name: '范数 detach', pain: "对整行范数分母完整求导时, 反向传播需额外暂存一份 [d_out, d_in] 形状的中间梯度", fix: "按论文 4.3 节对行范数分母截断梯度将其视作常数: 省去这份全尺寸梯度显存, 且微调精度几乎不变" },
        ],
      },
      code: 'llm_finetune/methods/dora.py · llm_finetune/methods/lora.py',
      points: [
        {
          title: '解耦被加法绑死的行模长与行方向',
          key: true,
          body: '将线性层权重的每一行看作特征空间中的一个向量, 它可以分解为行范数 (模长) 与单位向量 (方向)。DoRA 论文统计发现, 全参数微调中各层幅度改变量与方向转角呈负相关 (-0.62, 常需大幅转方向而微调模长, 或反之); 而标准 LoRA 直接令 $W_0 + BA$ 相加, 两者被迫强正相关 (+0.83), 想「只转方向不改模长」要求低秩矩阵 $BA$ 在每一行都精准贴合球面 $\\|W_0 + BA\\| = \\|W_0\\|$, 浪费了宝贵的低秩自由度。\nDoRA 借鉴 Weight Normalization 将权重改写为 $W^\\prime = m \\odot V / \\|V\\|_{\\text{row}}$ (其中 $V = W_0 + (\\alpha/r)BA$), 从结构上把模长调整交给一维向量 $m$, 把方向旋转交给 $BA$。',
        },
        { title: '行范数归一化屏蔽径向扰动, 分母 detach 节省反传显存', body: '由于前向计算前每行 $V$ 都会除以自身二范数变成单位向量, $BA$ 沿着原向量径向拉长多少都不会改变前向输出, 有效作用全部落在切向旋转上。\n反向传播处理分母 $\\|V\\|_{\\text{row}}$ 有两种选择: 若对分母完整求导, 计算图需额外暂存一份全尺寸 [d_out, d_in] 的中间梯度; 本库遵循 DoRA 论文 §4.3 对分母调用 .detach() 将其视作逐前向步更新的缩放常数, 省去这份全尺寸梯度开销的同时保留了绝大多数精度收益。' },
        { title: '参数增量极微、推理零额外开销, 但训练每步需构造完整权重', body: '从资源账看 DoRA 与标准 LoRA 的异同:\n- 参数量几乎不变: 每层仅多一个长度为 $d_{\\text{out}}$ 的向量 lora_magnitude, 本库总可训参数仅从 19,456 微增至 20,736。\n- 训练耗时略增, 推理零开销: 标准 LoRA 前向可按 $W_0 x + B(Ax)$ 分两步小矩阵乘算完; DoRA 因需对整行 $W_0 + BA$ 求范数, 训练每步都要显式拼出 [d_out, d_in] 的组合矩阵。但训练结束后同样可一次性写回普通 nn.Linear, 线上推理与 LoRA 完全一样零延迟。\n在 copy -> sort (r=8, lr=1e-2, 300 步, 3 种子平均) 上, 末步损失由 LoRA 的 0.369 降至 0.296, 留出集 EM 由 0.382 升至 0.522。' },
      ],
      links: [
        { from: 'LoRALinear', to: 'DoRALinear', body: 'DoRALinear 直接继承 LoRALinear, 复用 A/B 初始化与 delta(), 仅新增行幅度参数 m 并重写按行归一化前向。' },
        { from: 'weight normalization', to: 'DoRA', body: '经典 Weight Normalization 将全量权重拆作模长乘单位向量以改善优化条件数, DoRA 则把其中的方向更新分支替换成了参数高效的低秩旁路。' },
        { from: 'DoRA', to: 'merge', body: '训练完毕后计算一次 $m \\odot V / \\|V\\|_{\\text{row}}$ 覆盖原线性层权重即可无损合并。' },
      ],
      sourceRows: [
        { concept: '初始化', code: 'llm_finetune/methods/dora.py:DoRALinear', takeaway: 'lora_magnitude 初始化为基座逐行范数 $\\|W_0\\|_{\\text{row}}$, 配合 $B=0$ 保证第 0 步有效权重与原基座逐位相同。' },
        { concept: '方向分支', code: 'V = W₀ + (α/r)·B·A', takeaway: '方向未归一化前与标准 LoRA 完全一致, 后续通过除以行范数剥离长度变化。' },
        { concept: '归一化', code: 'direction.norm(dim=1, keepdim=True).detach()', takeaway: '计算逐行 L2 范数并截断分母梯度回传, 避免全尺寸中间梯度吃掉低秩省下的显存。' },
        { concept: '幅度', code: 'lora_magnitude: nn.Parameter([d_out])', takeaway: '参数命名保留 lora_ 前缀, 无缝兼容现有的冻结统计与 adapter 导出工具函数。' },
        { concept: '多随机种子对照', code: 'run_finetune/dora/README.md', takeaway: '在 r=8 下 3 个种子平均 EM 从 0.382 提至 0.522。玩具规模下这个优势并不稳定: 把 r 换成 2 时, 3 个种子里有 1 个是 LoRA 胜出 (这个对照不在脚本里); 脚本的断言也只要求 3 个种子的平均 loss DoRA 低于 LoRA。' },
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
      subtitle: '传统 RLHF 需要先训奖励模型再跑在线强化学习。DPO 利用最优策略与奖励函数的解析对应关系, 直接在成对偏好数据上做二分类优化; 训练时除了盯分差还需监控好回复自身的对数似然。',
      tldr: '- DPO (Direct Preference Optimization) 省去了显式奖励模型与在线采样, 直接在 (prompt, chosen, rejected) 三元组上拉大在训模型 policy 相对冻结起点 ref 对好坏回复的对数概率比差值 (初始化 policy == ref 时首步损失恒为 $\\ln 2 \\approx 0.6931$)。\n- 关键监控盲区: DPO 损失只看相对分差, 「好坏概率同时下降、坏回复跌得更快」同样能让偏好准确率涨到 0.996, 却引发概率挤压 (likelihood displacement) 导致生成 EM 下滑; 非成对变体 KTO 的好样本有单独一项把隐式奖励往上推, 本例没有出现这一退化 (KTO 的参考点 $z_0$ 全程被截断为 0, 起作用的不是参考点)。',
      question: '为什么在 DPO 训练里看到偏好分差 (reward margin) 一路上涨, 模型实际生成的回复质量却可能不升反降?',
      evolution: {
        title: '用上 rejected, 又别把模型压坏',
        subtitle: "根问题: 成对偏好数据包含负样本更差的对比信息, 但直接压低负样本概率缺乏天然下界。",
        steps: [
          { name: '只拿 chosen 做 SFT', pain: "(原点) 面对 (x, chosen, rejected) 三元组偏好数据, 最简做法是仅取 chosen 做监督微调", fix: "模型能够学会 chosen 回复分布, 但负样本 rejected 中包含的错误模式对比信号被完全丢弃" },
          { name: '同时压 rejected', pain: "仅使用一半偏好数据导致模型缺乏负向约束, 无法明确区分并抑制错误写法", fix: "在损失中额外加入压低 rejected 对数概率的项; 但下压过程没有下界, 正负回复共享的 token 概率也会被一并压低" },
          { name: 'DPO', year: 2023, pain: "无界下压负样本概率既缺少梯度饱和边界, 也缺少限制策略偏离初始模型的约束", fix: "对两条回复分别计算「policy 概率 / ref 概率」的对数并作差送入 $-\\log\\sigma$: 分差拉开后梯度自动消失, $\\beta$ 控制偏离 ref 的幅度" },
          { name: 'KTO', year: 2024, pain: "仅优化相对分差时好坏概率同时下降同样能降低损失; 且训练数据必须按同一题目配对", fix: "将单条样本各自与参考点 $z_0$ 比较高低, 为正样本提供独立向上的推力: $\\log\\pi(\\text{chosen})$ 从 -4.03 升至 -3.73" },
        ],
      },
      code: 'llm_finetune/methods/dpo.py · llm_finetune/data/preference_data.py',
      points: [
        { title: '用最优策略的解析解把「训 RM + 跑 RL」合并为一步', body: '带 KL 正则约束的 RLHF 目标函数 $\\max_\\pi \\mathbb{E}[r(x,y)] - \\beta\\,\\mathrm{KL}(\\pi\\,\\|\\,\\pi_{\\text{ref}})$ 存在闭式最优解, 反解出奖励与策略概率比的关系为 $r(x,y) = \\beta \\log(\\pi(y\\mid x)/\\pi_{\\text{ref}}(y\\mid x)) + \\beta \\log Z(x)$。\n将其代入 Bradley-Terry 成对胜率公式 $P(y_w \\succ y_l \\mid x) = \\sigma(r(x,y_w) - r(x,y_l))$ 后, 与回复无关的配分项 $\\beta \\log Z(x)$ 恰好相减抵消, 得到只依赖当前策略 $\\pi$ 与冻结参照 $\\pi_{\\text{ref}}$ 对数似然的二分类交叉熵损失——无需训练独立 RM, 也无需在线 rollout 采样。' },
        { title: '对数似然差与冻结 ref 如何抵消长度偏置', body: 'DPO 送入 $-\\log\\sigma(\\beta\\Delta)$ 的核心量是隐式奖励差 $\\Delta = (\\log\\pi_c - \\log\\mathrm{ref}_c) - (\\log\\pi_r - \\log\\mathrm{ref}_r)$, 其中每项 $\\log\\pi$ 都是该条回复全部有效 token 的对数概率之和 ($\\sum_t \\log p_t$)。\n因为每个 token 的对数概率都是负数, 越长的回复累加出来的 $\\sum \\log p$ 天然越负: 若没有 ref, 只要差回复 rejected 比好回复 chosen 多几个词, 什么都不学也能白赚正分差 (长度红利); 引入 ref 作差后, 基础长度带来的负偏置在 policy - ref 中被成对抵消。' },
        {
          title: '警惕概率挤压: 分差拉大不代表好回复概率上升',
          key: true,
          body: 'DPO 损失只约束 chosen 与 rejected 的相对分差 $\\Delta$ 变大, 并不要求 $\\log\\pi(\\text{chosen})$ 本身绝对上升: 「两边对数概率一起跌、只是 rejected 跌得更狠」同样能把分差推高。\n从 100 步半收敛 SFT 起点出发跑 200 步 DPO (β=0.5, lr=3e-4) 实测留出集指标:\n- 偏好排序准确率从 0.965 升至 0.996 (分差确实拉开了)。\n- 然而好回复自身的对数似然 log π(chosen) 从 -4.03 滑落至 -4.18, 贪心生成 EM 从 0.332 退化到 0.137。\n因此实际训练 DPO 时必须同时监控 logp_chosen 与生成指标, 不能只看偏好胜率。',
        },
        { title: '温度系数 β 控制偏离冻结基座的幅度', body: '超参数 $\\beta$ 直接对应原始 RLHF 目标里的 KL 惩罚强度。从损失梯度权重 $\\sigma(-\\beta\\Delta)$ 来看:\n- $\\beta$ 偏大时: 微小的正向分差 $\\Delta$ 乘上大 $\\beta$ 就会让 sigmoid 迅速饱和趋近 1, 梯度提前归零, 策略紧贴 ref 保守微调。\n- $\\beta$ 偏小时: 需要把 $\\Delta$ 推得很大才能让梯度衰减, 策略会大幅偏离初始分布。本脚本只跑了 β=0.5 一组, 概率挤压在其他 β 下有多严重并没有测。' },
        {
          title: 'KTO: 直接消费非成对单边反馈, 顺带稳住好回复似然',
          body: '真实线上日志往往只有单条回复的点赞 👍 或点踩 👎, 凑不成同一 prompt 下的好坏配对。KTO (Kahneman-Tversky Optimization) 不再做两两相减, 而是把每条样本的隐式奖励 $r = \\log(\\pi/\\pi_{\\text{ref}})$ 与批内统一参考点 $z_0$ 单独比高低: 好样本最大化 $\\sigma(\\beta(r - z_0))$, 坏样本最大化 $\\sigma(\\beta(z_0 - r))$。\n由于好样本拥有独立向上的推力项, 相同起点训 200 步后 KTO 偏好准确率达 0.992, 同时 log π(chosen) 从 -4.03 稳步升至 -3.73, 贪心 EM 保持在 0.324 (同脚本对照 DPO 为 0.121)。',
        },
        {
          title: 'KTO 的正负类配平与参考点估计细节',
          body: '使用 KTO 时有两个实现细节值得注意:\n- 正负样本极度失衡时必须配平权重: 当好坏样本比为 1:9 时, 需按论文建议令 $\\lambda_D n_D \\approx \\lambda_U n_U$ (取 $\\lambda_D=1, \\lambda_U=1/9$), 留出集偏好准确率可达 0.988、EM 0.254; 若保持等权 1:1, 9 倍负样本的下压梯度会淹没全局, 令 EM 直接归零。\n- 参考点 $z_0$ 的截断行为: 代码用同批错位拼接样本 (x_i, y_{i-1}) 估计 KL 均值并截断为非负 (clamp(min=0)); 本例训练中错配回复概率被压得比 ref 更低, 均值落入负区后全程被截断为 0, 相当于一半前向算力用在了参考点估计上。因此本例实际训练的是「好样本 $\\sigma(\\beta r)$、坏样本 $\\sigma(-\\beta r)$」, 参考点那一项没有发挥作用。',
        },
      ],
      links: [
        { from: 'SFT 终态', to: 'policy / reference', body: 'policy 与冻结 ref 均从 SFT 检查点拷贝; 代码特意将 ref 装入普通 Python 元组而非注册为 nn.Module 子属性, 避免被优化器误收录或被 .train() 切回训练模式。' },
        { from: 'make_labels', to: 'compute_sequence_logprobs', body: '按非 -100 掩码位置汇总每条回复的对数似然; 保留下标 $P-1$ 能确保好坏回复首词差异进入优化目标。' },
        { from: 'DPOLoss', to: 'SimPO / ORPO', body: '下一章的无参考模型变体沿用同一份三元组数据与 -logsigmoid(z) 框架, 重点改造对数似然归一化方式与防漂移锚点。' },
        { from: 'PairwiseForward', to: '通用 Trainer', body: 'PairwiseForward 将 policy 前向与 no_grad 下的 ref 前向封装进单次模块调用, 使标准 Trainer 无需改动即可驱动成对偏好训练。' },
      ],
      sourceRows: [
        { concept: 'logprob 汇总', code: 'llm_finetune/methods/dpo.py:compute_sequence_logprobs', takeaway: '逐样本对有效回复位置的 token 对数概率求和并返回 [B] 向量, 供成对作差使用。' },
        { concept: 'DPO 的 z', code: 'beta * ((p_c - p_r) - (ref_c - ref_r))', takeaway: '计算缩放后的隐式奖励差; 初始时刻 policy == ref 使 z == 0, 首步损失精确等于 $\\ln 2 = 0.6931$。' },
        { concept: '梯度权重', code: 'llm_finetune/methods/dpo.py:DPOLoss', takeaway: '单样本梯度权重为 $\\sigma(-\\beta\\Delta)$, 已拉开差距的易分样本自动衰减退场, 算力聚焦于排错样本。' },
        { concept: '一步的开销', code: 'llm_finetune/methods/dpo.py:PairwiseForward', takeaway: '常驻双份模型权重 (778 KiB), 每步拼成 2B 跑一次带梯度 policy 加一次无梯度 ref (实测单步耗时仅比单模慢约 35%)。' },
        { concept: '偏好数据', code: 'llm_finetune/data/preference_data.py:PreferenceDataGenerator', takeaway: '好坏回复必须保持格式对称 (如末尾均含 EOS), 防止模型偷学表面格式捷径。' },
        { concept: 'KTO 数据', code: 'llm_finetune/methods/kto.py:UnpairedDataGenerator', takeaway: '将偏好对打散为每个 prompt 仅配单条好或坏回复, 并构造错位配对 (x_i, y_{i-1}) 供估计参考点 $z_0$。' },
        { concept: 'KTO loss', code: 'llm_finetune/methods/kto.py:KTOLoss', takeaway: '好样本令 $v = \\lambda_D\\sigma(\\beta(r - z_0))$, 坏样本令 $v = \\lambda_U\\sigma(\\beta(z_0 - r))$, 最小化 $\\mathbb{E}[\\lambda_y - v]$ 独立推高好回复、压低坏回复。' },
        { concept: '参考点 z0', code: 'kl_est.clamp(min=0).detach()', takeaway: '在错位配对样本上估计对数概率比均值并截断非负、阻断梯度反传。' },
        { concept: '前向复用', code: 'llm_finetune/methods/kto.py:KTOForward', takeaway: '继承 PairwiseForward 统一完成原样本与错配样本的双模型前向。' },
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
      subtitle: 'DPO 中的冻结 ref 模型同时承担着「抵消长度偏置」与「限制策略漂移」两项职责, 但也多占了一份权重显存和前向计算。SimPO 与 ORPO 去掉了 ref, 并用各自的方式接管这两项职责。',
      tldr: '移除冻结 ref 后, 常驻权重从 778 KiB 减半至 389 KiB, 单步 LM 前向从 2 次减为 1 次 (200 步耗时从 5.4 s 缩至 4.0 s), 两条技术路线各用不同手段替代 ref:\n- SimPO: 将整句对数概率和改为「按回复长度平均的对数概率」以消除长度红利, 并引入固定目标间隔 $\\gamma$ 充当停手阈值。\n- ORPO: 在好回复的标准监督交叉熵 (NLL) 之上叠加对数发生比 (log odds ratio) 偏好项, 靠 NLL 锚住生成分布, 可在一个阶段内同时完成指令微调与偏好对齐 (论文省掉的是预训练之后的 SFT 阶段, 起点仍是预训练模型)。',
      question: 'DPO 里的冻结 ref 除了防止模型跑偏还顺手消除了什么偏差? 把 ref 拿掉之后, SimPO 和 ORPO 各自靠什么补位?',
      evolution: {
        title: 'ref 拿掉之后, 它的两份活谁来接',
        subtitle: "根问题: 冻结的 ref 模型多占一份权重显存且每步多跑一次前向, 但它同时承担着抵消长度偏置与限制策略漂移两项职责。",
        steps: [
          { name: 'DPO 的 ref', pain: "(原点) 偏好优化需要基准锚点约束更新幅度, 否则在训策略 policy 会无限制偏离初始分布", fix: "冻结一份 SFT 副本作为 ref: 显存需常驻 2 份模型权重 (共 778 KiB), 且每步多跑 1 次前向计算" },
          { name: '直接去掉 ref', pain: "仅作参照的 ref 额外消耗整份权重显存与每步一次前向开销", fix: "直接对 policy 的序列总对数似然 $\\sum \\log p$ 作差: 省去了 ref 开销, 但更长的 rejected 会天然拉低总和而虚增分差 margin" },
          { name: 'SimPO', year: 2024, pain: "单步对数概率为负使 $\\sum \\log p$ 随回复变长而更负, 长度差异会造成非预期的分差偏置", fix: "改按回复长度取平均对数概率并减去目标间隔 $\\gamma$: 消除了长度偏置, 但因缺乏锚点约束, $\\log \\pi(\\text{chosen})$ 跌至 -5.69" },
          { name: 'ORPO', year: 2024, pain: "缺少正向似然锚点时, chosen 自身的生成概率随分差优化一同下滑, 贪心生成 EM 跌至 0.023", fix: "以 chosen 上的 SFT 负对数似然 (NLL) 为锚点并叠加对数发生比 (odds ratio) 偏好项: 单阶段即可完成对齐, 留出集 EM 升至 0.543" },
        ],
      },
      code: 'llm_finetune/methods/simpo.py · llm_finetune/methods/orpo.py · llm_finetune/methods/dpo.py',
      points: [
        { title: '直接删掉 ref 会留下长度红利', body: '若简单把 DPO 里的 ref 项划掉、只优化 $\\sum \\log p_c - \\sum \\log p_r$, 模型就有长度空子可钻: 因为单步 $\\log p < 0$, 只要负样本 rejected 比正样本长几格, 它的总对数似然就天然更低, 模型无需学习实质偏好也能让损失趋近于 0。DPO 原本靠 policy - ref 成对相减消去了这一长度基线, 去掉 ref 后就必须显式做长度归一化。不过本仓库的偏好对长度最多差 1 个 token, 长度红利在这份数据上几乎没有机会出现。' },
        { title: 'SimPO: 长度平均对数似然搭配目标间隔 γ', body: 'SimPO 将隐式奖励定义为平均每 token 对数概率乘温度 $(\\beta / |y|)\\log\\pi(y\\mid x)$, 直接把序列长度 $|y|$ 约掉, 并且与束搜索解码常用的长度归一打分指标完全对齐; 随后在二分类损失 $-\\log\\sigma(r_c - r_r - \\gamma)$ 中减去正间隔 $\\gamma$, 要求好回复的平均对数概率领先超过 $\\gamma/\\beta$: 领先量正好等于 $\\gamma/\\beta$ 时损失为 $\\ln 2$, 领先越多损失越接近 0, 梯度也随之连续变小。\n注意除以长度后数值幅度大幅缩小, 因此 SimPO 的 $\\beta$ 通常需设在 2.0 左右 (远大于 DPO 惯用的 0.1~0.5), 否则梯度几乎推不动参数。' },
        {
          title: '有没有锚定好回复似然, 直接拉开生成质量差距',
          key: true,
          body: '从同一个 100 步半收敛 SFT 起点 (EM = 0.332) 出发, 在相同偏好集上各训 200 步对比留出集表现:\n- DPO (β=0.5): 偏好准确率 0.996, log π(chosen) 为 -4.25, 贪心 EM 0.121。\n- SimPO (β=2, γ=1, 未精调超参): 偏好准确率 0.992, 但因缺少拉住好回复的显式锚点, log π(chosen) 一路跌至 -5.69, 贪心 EM 仅剩 0.023。\n- ORPO (λ=0.5): 偏好准确率 0.988, 同时损失自带的 chosen NLL 项持续把 log π(chosen) 往上推升至 -2.51, 贪心 EM 不降反升至 0.543。甚至跳过前置 SFT 直接从随机初始化单阶段训 300 步 ORPO, EM 也能达到 0.973 (与纯 SFT 持平且把负样本压得更低)。\n注意三个方法都只跑了一组超参; ORPO 的 NLL 项也不是硬保证, λ 取大时 odds ratio 项仍可能把 chosen 拉低, 脚本只测了 λ=0.5。',
        },
      ],
      links: [
        { from: 'DPOLoss', to: 'SimPO / ORPO', body: '三者共享偏好三元组输入与 -logsigmoid 分类损失框架, 核心区别在于送入 sigmoid 的分差量构造: DPO 用带 ref 的总和差, SimPO 用长度均值差减间隔, ORPO 用对数发生比差 (SFT 锚是在 sigmoid 之外另加的 NLL 项)。' },
        { from: 'compute_sequence_logprobs', to: '平均 log p', body: '传入 average=True 即可将回复总对数似然除以有效监督 token 数, 同时供 SimPO 与 ORPO 调用。' },
        { from: 'ref 的那次前向', to: '省掉', body: '省掉冻结 ref 使常驻模型显存减半, 单步耗时加快约 25% (由于 ref 原本就不建图反传, 故总训练耗时节省约为四分之一而非一半)。' },
        { from: 'ORPO 的 NLL 项', to: 'SFT', body: 'ORPO 总损失写作 nll_chosen + λ * odds_ratio_loss, 前半项本身就是完整的 SFT 目标, 使其天然具备单阶段对齐能力。' },
      ],
      sourceRows: [
        { concept: 'DPO 的 z', code: 'llm_finetune/methods/dpo.py:DPOLoss', takeaway: '$\\beta [(\\pi_c - \\mathrm{ref}_c) - (\\pi_r - \\mathrm{ref}_r)]$, 借助冻结 ref 抵消长句负偏置并约束策略漂移。' },
        { concept: 'SimPO 的 z', code: 'llm_finetune/methods/simpo.py:SimPOLoss', takeaway: '$\\beta (\\log p_c / |y_c| - \\log p_r / |y_r|) - \\gamma$, 按长度平均消偏并用间隔 $\\gamma$ 设定目标领先幅度。' },
        { concept: 'γ = 0 且不归一化', code: 'z = logp_c − logp_r', takeaway: '退化为均匀分布参照下的裸求和差, 长负样本无需学习即白得正 margin。' },
        { concept: 'ORPO 的 odds', code: 'log(p/(1−p)), p = exp(mean logp)', takeaway: '基于几何平均单步概率 $p$ 计算发生比 $p/(1-p)$, 实现中对 $p$ 做边界截断防止 $\\log(1-p)$ 数值下溢。' },
        { concept: 'ORPO 总 loss', code: 'llm_finetune/methods/orpo.py:ORPOLoss', takeaway: '组合好回复的交叉熵锚点与加权对数发生比损失, 兼顾生成流畅度与正负偏好区分。' },
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
      subtitle: '围绕后训练的三类核心信号展开: 奖励模型 (RM) 把成对人类偏好转化为连续标量分, GRPO 用同题组内均值替代独立价值网络完成在线强化学习, 知识蒸馏则用平滑后的教师概率分布向小模型传递暗知识。',
      tldr: '- 奖励模型 (RM): 在语言模型主干末尾接标量线性头, 用 Bradley-Terry 损失学习成对回复的相对分差 (本库 300 步后留出集准确率从 0.549 升至 0.930)。\n- GRPO: 对同一道题在线采样 $G$ 条回复, 直接以组内平均分作为基准线 (baseline) 计算相对优势, 省去 PPO 中与策略同尺寸的 critic 网络。\n- 知识蒸馏: 用高温软化教师模型的词表分布并乘上 $T^2$ 补偿梯度衰减, 在小样本 (128 条) 下将留出集前向 KL 从硬标签的 2.293 压低至 1.860。',
      question: '当人工偏好标注存在 20% 的随机标反率时, RM 学出的最优分差会停在哪个数值? GRPO 砍掉 critic 网络后靠什么降低策略梯度方差?',
      evolution: {
        title: '奖励从哪来, baseline 从哪来',
        subtitle: "根问题: 在线强化学习每采样一条回复都需要即时获取标量奖励, 同时还需估计当前题目的期望得分作为基准线。",
        steps: [
          { name: '人打绝对分', pain: "(原点) 在线强化学习需要对每条采样回复给出标量奖励, 最朴素的做法是由人工直接打分", fix: "由标注员对单条回复打绝对分: 不同标注员的评分尺度难以统一, 且人工反馈速度无法匹配在线采样循环" },
          { name: '排序 + RM', pain: "人工绝对评分噪声大、尺度不一, 且在线训练循环无法逐条阻塞等待人工标注", fix: "改为人工对同题回复排序 (InstructGPT 每次 4~9 条), 用 Bradley-Terry 模型 (将胜率写成分差的 sigmoid) 训出可随时调用的 RM" },
          { name: 'critic 当 baseline', pain: "直接将 RM 原始打分作为策略梯度权重时, 题目本身的难易差异会引入很大的梯度方差", fix: "PPO 额外训练价值网络 critic 估计当前状态的期望回报作为 baseline 减掉; 代价是需多维护一个与 policy 同尺寸的可训网络" },
          { name: 'GRPO', year: 2024, pain: "独立 critic 与 policy 参数规模相当, 显著推高显存与训练开销, 且短程训练下估值未必准确", fix: "对同一道题并行采样 $G$ 条回复, 直接以组内平均奖励作为 baseline 从而省去 critic; 代价是组内全对或全错时优势归零、不产生梯度" },
        ],
      },
      code: 'llm_finetune/methods/reward_model.py · llm_finetune/methods/grpo.py · llm_finetune/methods/distill.py',
      points: [
        {
          title: 'RM 只具有平移不变的相对分差意义, 噪声标签决定分差收敛上界',
          key: true,
          body: '奖励模型在语言模型最后一层归一化隐状态后接一个 [D -> 1] 标量头, 读取每条序列最后一个非填充 token 的输出作为打分 $r$。训练采用 Bradley-Terry 成对损失 $-\\log\\sigma(r_w - r_l)$: 给所有回复分数同时加上任意常数不会改变分差与损失, 因此 RM 分数只刻画相对高低而无绝对零点。\n若人工标注有 $\\varepsilon$ 的概率标反, 期望损失对分差 $\\Delta = r_w - r_l$ 求导令其为零可得 $\\sigma(\\Delta^*) = 1 - \\varepsilon$, 即分差收敛在有限值 $\\Delta^* = \\ln((1-\\varepsilon)/\\varepsilon)$ (例如 $\\varepsilon=0.2$ 时停在 1.39, $\\varepsilon=0.5$ 纯随机时为 0)。本库实测首步损失为 0.6929 ($\\approx \\ln 2$), 训练 300 步后留出集准确率从 0.549 升至 0.930。',
        },
        { title: 'GRPO 用同题组内采样均值替代独立的 critic 价值网络', body: '在线策略梯度若直接拿原始奖励加权对数概率梯度, 题目难易差异会引入巨大方差, 因此需减去「当前题目平均能拿多少分」的基准线 (baseline)。PPO 为此专门维护并训练一个与策略网络同尺寸的 critic 模型; GRPO 则对每个 prompt 并行采样 $G$ 条回复, 直接用组内均值 $\\mu$ 和标准差 $\\sigma$ 构造相对优势 $\\hat A_i = (r_i - \\mu) / \\sigma$。\n组内均值本身就是该题期望回报的无偏蒙特卡洛估计, 无需额外网络参数与显存; 唯一代价是当某道题的 $G$ 条采样恰好全对或全错时, 组内奖励完全相等导致 $\\hat A_i \\equiv 0$, 该组样本不产生有效梯度 (本库训练中每步约占 32%)。' },
        { title: '软标签蒸馏传递次优候选排序, 乘 T² 消除温度对梯度的连带缩放', body: '常规硬标签训练只在正确 token 处给 1、其余词表位置全给 0; 软标签蒸馏将教师与学生的输出 logits 同时除以温度 $T > 1$ 拉平概率分布, 让次优候选词之间的相对合理性 (暗知识) 也能传递给学生。\n由于 $\\mathrm{softmax}(z/T)$ 对 logits 求导会天然带入 $1/T^2$ 的缩小因子, 蒸馏 KL 项必须显式乘回 $T^2$, 才能保证调节温度 $T$ 时软硬两项损失的梯度量级配比仍只由权重 $\\alpha$ 控制。在具备双峰答案 (70% 排序 / 30% 照抄) 的 128 条小样本实验上, 软标签蒸馏将留出集前向 KL 从纯硬标签的 2.293 降至 1.860 (当训练样本无限充足时这一优势消失: 硬标签 0.906、软标签 0.960, 软标签甚至略差)。' },
      ],
      links: [
        { from: 'PreferenceDataGenerator', to: 'RewardModel', body: '同一份 (prompt, chosen, rejected) 偏好数据既可交给 DPO 直接训生成策略, 也可交给 RewardModel 训成独立的标量判分器。' },
        { from: 'RM 分数', to: 'GRPO 的 reward_fn', body: 'GRPO 接收任意签名为 reward_fn(prompts, completions) 的打分源: 开放问答可调 RM, 数学与代码题则可直接调确定性的验证脚本 (即 RLVR)。' },
        { from: 'policy.generate', to: '组内优势', body: '在线 RL 每步训练的序列都由当前最新策略现场生成 (on-policy), 因此独立编写采样与多轮内层更新循环, 不套静态数据集的 Trainer。' },
        { from: 'teacher logits / T', to: 'student KL', body: '温度 $T$ 负责展开教师长尾分布细节, $T^2$ 系数负责抵消求导带来的梯度幅值衰减。' },
      ],
      sourceRows: [
        { concept: 'value head', code: 'llm_finetune/methods/reward_model.py:RewardModel', takeaway: '直接调用主干 forward(..., return_hidden=True) 取末层归一化特征并接 [D -> 1] 线性头, 避免复制主干代码引发掩码或位置编码漂移。' },
        { concept: '读哪一个位置', code: 'attention_mask.sum(1) - 1', takeaway: '右填充批次中必须读取最后一个真实 token 下标 attention_mask.sum(1) - 1: 多垫 5 个 PAD 时正确下标分值偏移仅 5e-06, 误取 [:, -1] 则偏离高达 4.54。' },
        { concept: 'BT loss', code: 'llm_finetune/methods/reward_model.py:BradleyTerryLoss', takeaway: '计算 $-\\log\\sigma(r_w - r_l)$, 小初始化下首步两侧得分接近相等, 损失精确落在 $\\ln 2 \\approx 0.693$ 附近。' },
        { concept: '组内优势', code: 'llm_finetune/methods/grpo.py:group_advantages', takeaway: '在组维度 [B, G] 内计算 (r - mean) / (std + eps); 整组全对或全错时优势归零 (下一章 DAPO 针对此做动态过滤)。' },
        { concept: '重要性比率', code: 'llm_finetune/methods/grpo.py:GRPOTrainer', takeaway: '同一批采样序列复用 $\\mu=2$ 轮更新: 首轮参数未动故新旧概率比恒为 1.000, 第二轮起参数偏移触发比率裁剪。' },
        { concept: 'KL 的 k3 估计', code: 'd.exp() - d - 1', takeaway: '采用恒非负、低方差的 Schulman k3 单样本估计器逐 token 计算策略偏离 ref 的 KL 惩罚。' },
        { concept: '蒸馏损失', code: 'llm_finetune/methods/distill.py:DistillLoss', takeaway: '组合 $\\alpha \\cdot \\mathrm{CE} + (1-\\alpha) T^2 \\mathrm{KL}(p_t^T \\,\\|\\, p_s^T)$, 两项严格共用同一张非 -100 监督位置掩码。' },
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
      subtitle: '为了让高成本的在线采样序列复用多轮更新, GRPO 引入了 PPO 式的重要性比率与区间裁剪; 后续的 DAPO、Dr.GRPO 与 GSPO 则分别处理了对称裁剪压制低概率探索词、长度与难度归一化偏置以及单步比率噪声。',
      tldr: '当同一批采样回复只做 1 次梯度更新时, 新旧策略一致 ($\\rho \\equiv 1$), 裁剪逻辑并不生效; 复用多轮更新 ($\\mu > 1$) 时才需引入重要性比率 $\\rho = \\pi_{\\text{new}}/\\pi_{\\text{old}}$ 与区间裁剪。三项后续改进精准对应公式中的三处偏差:\n- DAPO: 单独放宽上界 (clip-higher) 给低概率探索词留出上升空间以防熵塌缩, 同时剔除零优势的全对/全错组。\n- Dr.GRPO: 移除会放大极端难易题权重的 $\\div\\sigma$, 并将按回复长度平均改为除以固定常数 $C$, 消除「答错时拖长回复能摊薄单步惩罚」的长度偏置。\n- GSPO: 将逐 token 抖动的比率替换为整条回复的几何平均序列比率统一裁剪, 匹配序列级奖励粒度。',
      question: '面对同一条「又长又错」的回答, GRPO、DAPO 和 Dr.GRPO 分配给每个 token 的惩罚力度有何区别? 为什么对称上界 $1+\\epsilon$ 只会卡住原本低概率的探索词?',
      evolution: {
        title: '一批 rollout 多用几遍, 偏置一处处修',
        subtitle: "根问题: 自回归采样是在线强化学习耗时最重的一步; 若让同一批 rollout 复用多轮更新, 就必须校正新旧策略分布偏移与归一化偏置。",
        steps: [
          { name: '一批更新一次', pain: "(原点) 原始策略梯度公式要求梯度评估策略与生成样本的行为策略严格一致", fix: "每批 rollout 仅执行 1 次梯度更新便丢弃: 此时新旧概率比恒满足 $\\rho \\equiv 1$, 估计无偏, 但每步更新都需等待一整轮自回归采样" },
          { name: 'PPO 的 ratio + clip', year: 2017, pain: "在同一批旧样本上连续更新多轮后, 当前策略已偏离采样时刻的旧策略, 直接求导会产生分布偏移偏差", fix: "乘上重要性比率 $\\rho = \\pi_{\\text{new}}/\\pi_{\\text{old}}$ 校正分布偏移, 并在比率超出 $[1-\\epsilon, 1+\\epsilon]$ 时截断梯度以防更新过猛" },
          { name: 'DAPO', year: 2025, pain: "对称上界 $1+\\epsilon$ 主要限制低概率探索 token 的上调幅度, 易压制有效探索并引发策略熵塌缩", fix: "单独将裁剪上界半径从 0.2 放宽至 0.28, 过滤全对或全错的零优势组, 并将损失改为按全批有效 token 等权平均" },
          { name: 'Dr.GRPO', year: 2025, pain: "损失虽改为按 token 平均, 优势计算仍除以组标准差 $\\sigma$, 导致 $\\sigma$ 极小的极端难易题梯度权重被异常放大", fix: "移除优势中的除 $\\sigma$ 归一化并将长度分母改为固定常数: 难题相对中等题的梯度权重比由 0.66 回归至 0.44" },
          { name: 'GSPO', year: 2025, pain: "消除分母偏置后, 逐位置比率 $\\rho_t$ 仍存在单步采样抖动, 与整条回复共享单一序列奖励的粒度不匹配", fix: "整条回复共用几何平均序列比率统一裁剪: $\\mathrm{std}(\\log \\rho)$ 从 0.067 降至 0.027, 本库裁剪半径相应收紧至 0.05" },
        ],
      },
      code: 'llm_finetune/methods/grpo.py · llm_finetune/run_finetune/grpo/train_grpo.py',
      points: [
        { title: '裁剪越界即截断梯度, 对称上界天然压制低概率探索词', body: '代理目标 $\\min(\\rho\\hat A, \\mathrm{clip}(\\rho, 1-\\epsilon_{\\text{lo}}, 1+\\epsilon_{\\text{hi}})\\hat A)$ 在比率超出区间后直接变平, 导数归零停止推动。\n观察上界 $1+\\epsilon$: 由于新概率 $\\pi_{\\text{new}} \\le 1$, 比率上限天然不超过 $1/\\pi_{\\text{old}}$。原本高概率的保守词 (如 $\\pi_{\\text{old}}=0.9$) 最大比率只有 $1.11$, 根本撞不上 1.2 的上界; 反倒是原本概率极低 (如 0.01) 但偶然采中拿了正奖励的探索词, 稍微抬升一点概率就会撞上 1.2 被截断梯度, 久而久之导致策略熵快速塌缩。因此 DAPO 采用非对称裁剪 (clip-higher), 保持下界 0.2 不变而将上界放宽至 0.28。' },
        {
          title: '分母里的两处隐性偏置: 回复长度 1/|o_i| 与组标准差 σ',
          key: true,
          body: '原始 GRPO 目标中含有两个与样本相关的动态分母, 无意间扭曲了梯度权重:\n- 序列长度分母 $1/|o_i|$: 先在单条回复内除以自身长度 $|o_i|$, 意味着好回复越短单步奖得越重、错回复越长单步罚得越轻 (实测 3-token 回复的单步权重是 9-token 的 3.0 倍, 诱导模型把错误推理越写越长)。\n- 组标准差 $\\sigma$: 几乎全对或几乎全错的题目组内奖励方差极小, 除以 $\\sigma$ 会反过来把这些极端题的梯度成倍放大 (实测 8 对 1 难题相对 8 对 4 中等题的权重在 GRPO 下为 0.66, 去掉除 $\\sigma$ 后回归合理的 0.44)。\nDr.GRPO 同时移除 $\\div\\sigma$ 并将长度分母替换为固定最大长度常数 $C$, 从根源消除这两项归一化偏置。',
        },
        { title: 'GSPO: 用整句几何平均比率对齐序列级奖励粒度', body: '验证器给出的奖励针对整条回复, 而逐 token 比率 $\\rho_t$ 只是单个采样位置的局部比值, 一条序列内部往往涨跌互现、方差偏大。GSPO 将比率聚合到序列级几何平均 $s = \\exp(\\frac{1}{|o|}\\sum_t \\log \\rho_t)$ 后再统一裁剪: 实测 $\\mathrm{std}(\\log \\rho)$ 从逐 token 的 0.067 压低至 0.027 (约缩减 $\\sqrt{|o|}$ 倍); 由于波动大幅收窄, 本库把裁剪半径相应收紧到 0.05 (这是本库取的值, 论文用的是 3e-4 量级)。' },
      ],
      links: [
        { from: 'GRPOTrainer.step', to: '多轮内层更新', body: '每轮采样后立即用行为策略温度缓存冻结的 logp_old, 随后在同一批序列上迭代 $\\mu$ 次更新: 第 1 次 $|\\rho - 1| = 0.000$, 第 2 次起比率偏离 1 并激活 clip 防护。' },
        { from: 'PPO clipped surrogate', to: 'GRPO', body: '外层比率乘优势再取最小值的截断结构源自 PPO, 区别仅在于优势估计来自组内统计而非独立 critic 网络。' },
        { from: '组内奖励全相同', to: 'DAPO 动态采样', body: '全对或全错组的优势全为 0 却仍占据平均分母 (本例约占 32.1%); DAPO 将这些零方差组过滤剔除, 避免有效梯度被空转样本稀释。本实现只过滤不补采, batch 会变小; DAPO 原版会继续采样直到凑满 batch。' },
        { from: '采样温度 T', to: 'log-prob 的 T', body: '采样若使用了温度 $T$, 计算分母 logp_old 和分子 logp_new 时也必须除以同一个 $T$, 否则第 1 轮起行为策略分布就发生错配。' },
      ],
      sourceRows: [
        { concept: '四个变体 = 五个开关', code: 'llm_finetune/methods/grpo.py:GRPOConfig', takeaway: '通过 clip_high、std_norm、loss_agg、seq_ratio、dynamic_sampling 五个布尔/数值开关统一表达 GRPO、DAPO、Dr.GRPO 与 GSPO。' },
        { concept: '组内优势', code: 'llm_finetune/methods/grpo.py:group_advantages', takeaway: '计算组内中心化奖励 r - mean, 按 std_norm 开关决定是否除以组标准差。' },
        { concept: '重要性比率', code: 'ratio = exp(logp_new − logp_old.detach())', takeaway: '在对数域作差后再求指数以保证数值稳定, 分母 logp_old 严格冻结自采样时刻。' },
        { concept: 'loss 聚合', code: 'llm_finetune/methods/grpo.py:aggregate', takeaway: '对比 seq_mean (GRPO 先除句长)、token_mean (DAPO 全批有效 token 等权) 与 fixed_len (Dr.GRPO 除固定常数) 三种归约。' },
        { concept: '结果解读边界', code: 'run_finetune/grpo/README.md', takeaway: '小模型 60 步后四变体的采样 pass@1 均从 0.186 升至 0.29~0.33 附近, 贪心 EM 基本没动; 四个变体之间的差别在单种子噪声内, 不能读成谁更强, 实验重点在于用单元测试精确验证各处偏置的消除机制。' },
        { concept: '评测口径', code: '采样 pass@1, 不是贪心 EM', takeaway: '在线 RL 直接塑造采样温度下的概率分布, 验收应看采样 pass@1 而非贪心 argmax。' },
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
      subtitle: '离线蒸馏只在教师生成的固定前缀上做正向 KL 拟合, 容量较小的学生模型既容易在多峰答案间折中串台, 也缺乏在自身真实生成轨迹上的纠偏经验; on-policy 蒸馏通过「学生采样 + 反向 KL」同时解决这两点。',
      tldr: '两类蒸馏的根本差异体现在训练前缀来源与 KL 散度方向上:\n- 离线蒸馏 (off-policy + forward KL): 在教师语料上优化 $\\mathrm{KL}(p_t \\,\\|\\, q_s)$, 迫使小容量学生把概率摊开覆盖教师的所有答案模式 (mode-covering), 容易把概率漏进两峰之间的错误谷底。\n- 在轨蒸馏 (on-policy + reverse KL): 让学生先自回归采样出序列, 教师在学生实际走到的每个位置上给出目标分布并优化 $\\mathrm{KL}(q_s \\,\\|\\, p_t)$, 引导学生集中学透单一主峰 (mode-seeking)。本库双峰任务实测采样合格率从离线的 0.059 跃升至 0.402 (代价是主动放弃了次要答法)。',
      question: '当学生模型容量装不下教师的全部答题风格时, 正向 KL 和反向 KL 分别会把学生推向什么分布形态?',
      evolution: {
        title: '从抄老师的卷子到老师批自己的卷子',
        subtitle: "根问题: 小容量学生模型难以覆盖教师的全部多峰答法, 且离线训练仅接触教师前缀, 推理时一旦偏离便缺乏纠偏监督。",
        steps: [
          { name: '硬标签', pain: "(原点) 让小模型迁移大模型能力时, 最直接的做法是在大模型生成的文本上做监督训练", fix: "在教师生成的答案序列上优化硬标签 CE: 每个位置仅监督单一目标 token, 其余候选词之间的相对概率信息全部丢失" },
          { name: '软标签 KD', year: 2015, pain: "硬标签只在单个 token 上置 1, 无法向学生传递次优候选词之间的相对合理性排序", fix: "以温度 $T$ 平滑后的教师分布为目标最小化 forward KL, 并乘 $T^2$ 补偿梯度衰减: 小样本下留出集前向 KL 从 2.293 降至 1.860" },
          { name: '学生自己采样', year: 2023, pain: "离线蒸馏仅在教师前缀上计算损失, 推理时学生一旦采样偏离便进入未见过的状态并逐级累积误差", fix: "先由学生自回归采样生成回复, 再由教师在学生实际走到的每个位置给出目标分布; 采样过程本身不回传梯度 (即 GKD 范式)" },
          { name: '逐 token reverse KL', pain: "学生容量受限时, forward KL 强迫分布摊开覆盖教师全部答法, 导致解码中途在多峰之间混杂失真", fix: "改用 reverse KL (在学生自身分布上取期望) 集中拟合单一主峰: 采样合格率升至 0.402 (离线基线为 0.059), 代价是放弃次要答法" },
        ],
      },
      code: 'llm_finetune/methods/on_policy_distill.py · llm_finetune/methods/distill.py',
      points: [
        {
          title: 'KL 散度不对称: 正向求全覆盖, 反向求单峰写准',
          key: true,
          body: 'KL 散度 $\\mathrm{KL}(p \\,\\|\\, q) = \\sum_v p(v)\\log(p(v)/q(v))$ 以左侧分布 $p$ 为权重决定在哪里惩罚误差。本实验中的 2 层教师掌握了两种合法答法 (70% 概率排序 sort, 30% 概率照抄 copy), 而 1 层学生容量有限无法同时记住两套规则:\n- 正向 KL $\\mathrm{KL}(p_t \\,\\|\\, q_s)$ (教师在前): 凡是教师给过概率的地方学生若敢置零就会吃无穷大惩罚, 被迫把概率平摊在两个峰以及中间的非法过渡区上, 采样时频繁半路「前半句排序、后半句照抄」导致整条作废。\n- 反向 KL $\\mathrm{KL}(q_s \\,\\|\\, p_t)$ (学生在前): 学生不给概率的分支权重为 0, 放弃次要峰完全不受罚; 反之只要把概率放到教师认为非法的位置就会遭重罚, 从而干净利落地收敛进单一正确主峰。',
        },
        { title: '在学生自生成前缀上批改, 消除训练与推理的暴露偏差', body: '离线蒸馏的前缀全部来自教师或静态数据集 (teacher forcing); 上线推理时一旦学生某步采样稍有偏差, 后续面对的就是训练期从未见过的陌生前缀, 错误会雪球式放大 (exposure bias)。\nOn-policy 蒸馏先让学生在 no_grad 下自己写出完整回复, 再请教师沿着这条真实轨迹逐位置给出词表分布进行对齐, 让训练状态分布严格贴合线上解码分布。' },
        { title: '兼具 RL 的在轨分布与蒸馏的逐位置稠密监督', body: '相比整条序列只拿 1 个末端标量奖励的在线 RL (如 GRPO), on-policy 蒸馏在学生走到的每个 token 位置上都能直接拿到教师完整的 $V$ 维概率向量, 在词表上解析求和算出精确的逐位置反向 KL, 无需高方差的策略梯度估计。\n从同一个预热 200 步的学生模型出发再各训 300 步对比: on-policy 蒸馏的采样合格率达到 0.402 (离线正向 KL 仅 0.059), 反向 KL 降至 0.50 (离线为 2.07), 主峰 sort 答案的对数似然升至 -1.34; 相应代价是次峰 copy 的对数似然被压低到 -33, 多样性让位于单峰准确度。' },
      ],
      links: [
        { from: 'DistillLoss (离线)', to: 'on-policy 蒸馏', body: '从静态语料上的正向 KL 切换为学生在线采样轨迹上的反向 KL, 解决小模型多峰折中串台与暴露偏差。' },
        { from: 'student.generate', to: 'teacher 打分', body: '采样流程与 GRPO 一样在 no_grad 下完成, 随后将整条回复一次前向喂给教师模型提取逐位置目标分布。' },
        { from: 'reverse KL', to: 'GRPO 的 KL 惩罚', body: 'GRPO 锚定起点模型的正则项同样是在当前策略采样分布上计算的反向 KL, 只是对齐目标由更强的教师换成了冻结起点 ref。' },
        { from: '热身', to: 'on-policy', body: '随机初始化的学生采出的全是乱码前缀, 教师在分布外乱码上给出的软标签缺乏指导意义, 因此通常先用少量 SFT 或离线蒸馏热身起步。' },
      ],
      sourceRows: [
        { concept: '离线基线', code: 'llm_finetune/methods/distill.py:DistillLoss', takeaway: '在静态数据集上计算正向 $\\mathrm{KL}(p_t \\,\\|\\, q_s)$ (对照臂设 T=1, α=0 纯测分布对齐方向差异)。' },
        { concept: '学生采样', code: 'llm_finetune/methods/on_policy_distill.py:on_policy_distill_step', takeaway: '遵循 GKD 范式: 先无梯度采样生成序列并视前缀为常数, 再做一次带梯度学生前向与无梯度教师前向对齐每个位置。' },
        { concept: '逐 token reverse KL', code: 'llm_finetune/methods/on_policy_distill.py:token_kl', takeaway: '在全词表上解析求和 $\\sum_v q_s(v)(\\log q_s(v) - \\log p_t(v))$, 仅对 EOS 及之前的有效回复位置求均值。' },
        { concept: '双向指标对读', code: 'run_finetune/on_policy_distill/README.md', takeaway: '离线臂在正向 KL 上占优 (0.88 对 1.36), 在轨臂在反向 KL (0.50 对 2.07) 与实际采样合格率 (0.402 对 0.059) 上大幅胜出。' },
        { concept: '独立训练循环', code: 'run_finetune/on_policy_distill/train_on_policy_distill.py', takeaway: '因每步数据需由当前学生模型动态生成, 与 GRPO 一样采用显式手写训练循环。' },
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
      subtitle: '可验证奖励强化学习 (RLVR) 用确定性程序判分取代易被钻空子的神经网络奖励模型; 设计合成任务或规则奖励前, 第一步是核算「不读题的常数输出最多能拿几分」。',
      tldr: 'RLVR (RL with Verifiable Rewards) 的判分函数必须同时依赖输入题目与模型输出 reward_fn(prompt, completion):\n- 反例警示: 若奖励只检查输出特征而不看输入 (如「输出落在词表后半区即给分」), 策略会迅速坍缩为无视题目的固定常数串, 虽然奖励曲线一路涨到 1.0, 实际却未学到任何条件推理能力。\n- 常数基线体检: 开训前先计算单一固定回复在全集上的最高期望得分; 只有当真实任务把常数基线压低到 1 / 类别数 且训练得分显著越过该线时, 上涨的奖励曲线才代表模型学会了读题作答。',
      question: '看到强化学习的平均 reward 一路涨到 1.0 时, 如何快速排查模型是真的学会了审题, 还是仅仅坍缩到了某几个高分固定 token?',
      evolution: {
        title: '奖励先过常数基线这一关',
        subtitle: "根问题: 强化学习总是沿着最易提升奖励的方向收敛; 一旦奖励函数不依赖输入题目, 策略便会退化为忽略题干的常数输出。",
        steps: [
          { name: 'RM 打分', pain: "(原点) 在线强化学习需要奖励信号, 通用方案是先训练神经网络 RM 为开放域回复打分", fix: "神经网络 RM 可覆盖各类开放任务, 但因打分面由有限样本拟合而成, 策略易针对其分布外盲区刷取虚高分数 (reward hacking)" },
          { name: 'RLVR', year: 2024, pain: "RM 给出的高分并不等价于客观正确, 策略持续优化后期往往偏向利用判分模型的拟合漏洞", fix: "将神经网络 RM 替换为确定性程序验证器 (Tülu 3 命名为 RLVR): 数学题核对精确答案, 代码题执行单元测试" },
          { name: '只看输出的规则', pain: "确定性规则虽消除了模型拟合盲区, 但若规则设计不当仍可能忽略对输入题干的条件依赖", fix: "如「输出落在词表后半区即给分」的区域奖励: 固定输出单一常数即可拿满 1.0 分, 奖励曲线升至 1.0 却未学到任何条件推理" },
          { name: '看题的 verify', pain: "当奖励函数与输入 prompt 脱钩时, 无视题干输出固定高分 token 就是全局最优策略", fix: "强制将判分函数定义为双参数 f(prompt, completion), 并在开训前先核算不读题的常数基线: 真实条件任务下该基线仅为 1 / 类别数" },
        ],
      },
      code: 'llm_finetune/data/tasks.py · llm_finetune/data/prompt_data.py · llm_finetune/methods/grpo.py',
      points: [
        {
          title: '开训前先算「不读题常数基线」',
          key: true,
          body: '强化学习总是沿着最省力的梯度方向最大化奖励。验证一个奖励函数是否靠谱, 最直接的体检指标是常数基线: 假设模型完全忽略输入 prompt、对所有样本永远吐出同一个固定字符串 c, 它能拿到的最高平均分是多少?\n- 若常数基线高达 1.0 (例如只考核回复长度、关键词频率或词表区间), 那么无脑复读固定套话就是全局最优解, 再漂亮的满分曲线也证明不了条件推理能力。\n- 当标准答案随 prompt 唯一变化时, 任意固定输出最多蒙对其中一类, 常数基线掉回 1 / 类别数, 只有越过这条水位线的收益才真正来自读题计算。',
        },
        { title: '与输入脱钩的奖励会迅速诱导策略熵归零塌缩', body: '在只看输出落在词表后半区的反例任务上, 策略很快就会把全部概率质量集中到单个高分 token 上: 生成分布的熵迅速跌至 0, 对任何输入都复读同一串字符, 表面平均奖励却完美停在 1.0——这不是算法出了 bug, 而是奖励函数本身退化成了无上下文的多臂老虎机 (bandit)。' },
        { title: '真实数学与代码验证器天然绑定输入条件', body: '真实的数学答案核对与代码单元测试天然满足 verify(prompt, completion) 双参数依赖。本库 SeqTask 同样按此原则设计: 输入为 6 个随机 token (共 $13^6 \\approx 480$ 万种组合), 目标输出为该输入的复制、反转或排序, 且训练集与留出集按 token 和 mod 5 严格隔离, 迫使模型学习通用变换规则而非背诵样本。' },
      ],
      links: [
        { from: 'SeqTask.verify', to: 'reward_fn(prompts, completions)', body: 'verify 先根据每道 prompt 现算唯一标准答案再与生成序列比对, 杜绝不看输入的投机满分。' },
        { from: 'PromptDataGenerator', to: 'SeqTask', body: '在线 RL 数据源每步只提供题干 prompts 而不含现成标签, 回复由策略网络现场采样并交由 SeqTask 判分。' },
        { from: '组内零方差', to: 'DAPO 动态采样', body: '严格的 0/1 精确判分在训练初期容易遇到整组全错 (零优势), 常配合组大小扩容或 DAPO 动态过滤使用。' },
        { from: 'verify', to: '训练与验收', body: '同一个确定性验证函数同时服务于训练集在线打分与留出集泛化验收。' },
      ],
      sourceRows: [
        { concept: '区域奖励 (反例)', code: 'reward = (completion >= V//2).mean()', takeaway: '仅统计生成 token 落入词表后半区的比例而无视输入题干, 常数复读即可轻松拿满 1.0 分。' },
        { concept: '看题奖励', code: 'llm_finetune/data/tasks.py:SeqTask', takeaway: '根据当前输入序列实时计算 target(prompts) 并要求整条回复 (含 EOS) 逐位匹配才给 1 分。' },
        { concept: '判分器', code: 'llm_finetune/data/tasks.py:verify', takeaway: '实现严苛的二值精确判分; 若目标任务过难导致初期全零奖励过多, 可引入步骤部分分或预热 SFT。' },
        { concept: '留出集怎么切', code: 'llm_finetune/data/tasks.py:sample_prompts', takeaway: '将 sum(prompt) % 5 == 0 的五分之一空间划作留出集, 与训练集零交集。' },
        { concept: '分桶诊断', code: '按 prompt 分桶的准确率', takeaway: '除全局平均分外按不同题目类型分桶核查通过率, 一眼识别只答对单一子类的偏科坍缩。' },
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
      subtitle: '当多个微调模型派生自同一个预训练基座时, 微调前后的权重差即构成了「任务向量」。无需训练数据与反向传播, 在参数空间直接缩放叠加任务向量即可组合多任务能力。',
      tldr: '- 任务算术 (Task Arithmetic) 将第 $t$ 次微调学到的参数偏移记作任务向量 $\\tau_t = \\theta_t - \\theta_0$, 合并模型直接计算 $\\theta = \\theta_0 + \\lambda\\sum_t \\tau_t$, 全程零梯度更新、零训练数据。\n- 本库实测: 分别掌握低段 token +1 和高段 token -1 的两个单任务模型, 单独面对双段同时改写题时 EM 均不超过 0.027; 将两者的任务向量以 $\\lambda=1$ 直接加回基座后, 留出集双段全对 EM 跃升至 1.000 (简单权重平均即 $\\lambda=0.5$ 时仅为 0.039)。',
      question: '为什么两个微调模型直接取权重平均效果很差, 换成任务向量相加并调对缩放系数 λ 却能无损兼顾两项任务? TIES 和 DARE 的修剪在什么规模下才真正需要?',
      evolution: {
        title: '不训练, 在权重空间里做加法',
        subtitle: "根问题: 基于同一基座分别微调出的多个专项模型各自掌握一项能力, 工程上常需在缺乏原始训练数据与重训算力的条件下合并它们。",
        steps: [
          { name: '合数据重训', pain: "(原点) 两个派生自同一基座的专项模型各自擅长单一任务, 需要合成为兼顾两项能力的单一模型", fix: "合并两份下游训练集从头联合微调: 效果最稳妥, 但依赖完整算力开销, 且协作方往往无法开放私有训练数据" },
          { name: '权重平均', pain: "实际交付场景中往往只能拿到各团队导出的模型权重, 无法获取原始训练样本", fix: "对两份微调权重直接求算术平均: 全程零训练开销, 但每个任务的参数增量均被折半稀释, 双段同时改写的留出集 EM 仅为 0.039" },
          { name: 'Task Arithmetic', year: 2022, pain: "算术平均等价于固定缩放系数 $\\lambda = 1/T$ ($T$ 为合并任务数), 任务数越多, 单项任务增量被稀释得越严重", fix: "提取微调前后的权重差作为任务向量 $\\tau = \\theta_{\\text{ft}} - \\theta_0$, 乘上经验证集校准的 $\\lambda$ 后加回基座: 取 $\\lambda = 1$ 时留出集 EM 达 1.000" },
          { name: 'TIES / DARE', year: 2023, pain: "大模型任务向量 $\\tau$ 中存在大量幅值微弱的冗余参数坐标, 不同任务向量在这些坐标上正负符号相反, 直接相加易产生干扰抵消", fix: "TIES 先裁剪小幅坐标并统一符号, DARE 先随机置零部分增量并重缩放后再相加; 但本例模型仅约 10 万参数、缺乏冗余, 修剪后反而不如直接相加" },
        ],
      },
      code: 'llm_finetune/methods/merge.py · llm_finetune/run_finetune/merge/train_merge.py',
      points: [
        { title: '共享基座让参数空间具备可加性组合基础', body: '实际协作中常遇到多个团队基于同一开源基座分别微调出专项模型 (如任务 A 把低段 token 3~8 平移 +1, 任务 B 把高段 token 10~15 平移 -1), 却拿不到彼此的私有训练数据合训。此时单个专项模型只能改对各自负责的半边, 双段同时改写的留出集 EM 仅 0.004 和 0.027; 在权重空间直接合并提供了一条几秒钟即可完成的零训练集成路径。' },
        { title: '任务算术的核心是区分基座起点 θ₀ 与增量 τ, 并校准系数 λ', body: '将微调后权重减去公共基座得到任务向量 $\\tau_t = \\theta_t - \\theta_0$, 再按 $\\theta = \\theta_0 + \\lambda(\\tau_A + \\tau_B)$ 加回基座:\n- 若直接对两份完整权重取算术平均 $(\\theta_A + \\theta_B)/2$, 代数上等价于固定取 $\\lambda = 0.5$: 每个任务的增量都被腰斩了一半, 两段改写都停在半途, 留出集 EM 只有 0.039。\n- 通过小验证集选出 $\\lambda = 1.0$ 将两份完整增量叠加回基座后, 留出集双段全对 EM 直接达到 1.000, 单任务干扰为 0.000。' },
        {
          title: '小模型缺乏参数冗余时, TIES / DARE 修剪反而不如直接相加',
          key: true,
          body: '针对大模型多任务合并时的坐标冲突, 后续工作提出了多种稀疏修剪与球面插值方案:\n- TIES: 仅保留每个任务向量绝对值最大的前 $d$ 比例坐标, 按幅度投票统一正负号后只对同号任务求均值。\n- DARE: 以概率 $p$ 随机将任务向量坐标置零, 幸存坐标乘 $1/(1-p)$ 保持期望不变。\n- SLERP: 在两个完整模型权重张量之间按夹角做球面线性插值。\n在本库约 10 万参数的小模型上各自调优 $\\lambda$ 后实测留出集 EM: Task Arithmetic 为 1.000, DARE (p=0.5) 为 0.875, TIES (d=0.5) 为 0.734, DARE (p=0.9) 降至 0.066, SLERP (t=0.5) 为 0.090。原因很直观: 十亿级大模型的微调增量高度过参数化、丢掉九成微弱坐标无伤大雅; 而紧凑小模型几乎没有冗余参数, 强行剪掉一半或九成坐标直接伤到了有效功能电路。\n这些数字来自单个种子: 另换两个种子 (1、2) 时 Task Arithmetic 的 EM 为 0.973 / 0.617, 仍是最高, 但并不稳定在 1.000。本例也只合并了 2 个任务, 任务更多时 TIES 这类方法的价值才可能显现, 这里没有测。',
        },
        { title: '合并成功的前提是输入输出功能正交, 而非参数坐标不重叠', body: '为什么两份任务向量直接相加不会互相破坏? 核查参数坐标会发现, 两者非零元素中多达 47.4% 符号相反 (接近随机抛硬币的 50%), 参数层面完全重叠; 但展平后的整体余弦相似度仅为 +0.046 (近似正交)。\n真正决定能否合并的是函数空间是否冲突: 任务 A 训练时见过高段 token 并学会保持原样照抄, 因此 $\\tau_A$ 只被要求改变低段 token 的输出, 高段输入经过它之后仍按基座原样输出, 反之亦然, 两个任务在函数上几乎不重叠; 若两个任务对同一输入要求截然相反的输出 (如全序列倒序与全序列排序), 则任何权重合并算法都无法调和。' },
      ],
      links: [
        { from: 'LoRA merge', to: '任务向量', body: 'LoRA 合并时加回基座的低秩矩阵 $(\\alpha/r)BA$ 本身就是一个天然的低秩任务向量; merge.py 直接操作 state_dict 字典, 同时兼容全参微调与 LoRA 产物。' },
        { from: 'cos(τ_A, τ_B)', to: 'sign_conflict', body: '整体余弦 +0.046 反映两个增量向量在高维空间近似正交, 而逐坐标符号冲突率 47.4% 说明微观元素仍有大量正负交错, 两者度量的是不同层面的几何特征。' },
        { from: 'SLERP', to: '简单平均', body: '两模型共享同一基座故整体夹角很小 (0.333 rad), t=0.5 处的球面插值仅比算术平均放大 1.014 倍, 同样面临增量减半问题。' },
        { from: '同一个 θ₀', to: 'task arithmetic', body: '任务向量加减要求所有分支严格派生自同一个基座检查点 $\\theta_0$; 跨不同基座或不同随机初始化的模型相减毫无物理意义。' },
      ],
      sourceRows: [
        { concept: '任务向量', code: 'llm_finetune/methods/merge.py:task_vectors', takeaway: '对浮点参数逐张量计算 $\\tau_t = \\theta_t - \\theta_0$, 整型缓冲区直接沿用基座原值。' },
        { concept: '相加', code: 'llm_finetune/methods/merge.py:task_arithmetic', takeaway: '计算 $\\theta_0 + \\lambda\\sum_t \\tau_t$; TIES 与 DARE 均是先对 $\\tau_t$ 做稀疏化处理后再汇总相加。' },
        { concept: 'TIES', code: 'llm_finetune/methods/merge.py:ties_merge', takeaway: '依次执行按张量保留 top-density 幅值坐标、按总和符号投票定方向、仅对同号非零项求均值三步操作。' },
        { concept: 'DARE', code: 'llm_finetune/methods/merge.py:dare', takeaway: '按固定种子生成伯努利掩码随机丢弃 $p$ 比例增量, 其余坐标除以 $1-p$ 保持期望幅度不变。' },
        { concept: 'SLERP', code: 'llm_finetune/methods/merge.py:slerp', takeaway: '按张量夹角 $\\Omega$ 计算球面正弦权重插值, 当夹角趋近 0 时平滑回退至普通线性插值防除零。' },
        { concept: '干扰度量', code: 'llm_finetune/methods/merge.py:sign_conflict', takeaway: '统计两个任务向量共同非零坐标上正负号相反的比例 (本例实测为 47.4%)。' },
        { concept: '两个任务', code: 'llm_finetune/run_finetune/merge/train_merge.py:EditTask', takeaway: '分别构造低段 3~8 加一与高段 10~15 减一的局部改写任务, 从同一基座出发各微调 300 步。' },
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
      subtitle: '人工标注偏好数据成本高且标准易波动。Constitutional AI / RLAIF 将评判准则写成显式条文, 由评审器自动检查模型采样、改写违规片段并构造偏好对送入 DPO 训练; 对齐上限也随之完全取决于评审器的实际覆盖范围。',
      tldr: 'RLAIF 用「条文准则 (宪法) + 自动评审改写」替代人工偏好标注:\n- 合成闭环: 先从基础模型采样回复, 由评审器 (judge) 按条文定位违规并改写出合规版本, 直接将 (改写后, 改写前) 组成偏好对跑 DPO。\n- 边界清晰可见: 评审器实际检查的违规词发生率从 0.648 压低至 0.031, 而写进条文但评审函数漏查的变体词仅从 0.418 缓降至 0.320; 验收时还必须同时核查正常内容的保留率, 防止模型靠「闭嘴少说」混过违规检查。',
      question: '如果某条原则写进了宪法文档里, 但评审器漏掉了其中一种变体, 后续 DPO 能自己泛化学会拦截它吗?',
      evolution: {
        title: '标注员从人换成一份宪法',
        subtitle: "根问题: 人工逐条标注偏好数据成本高、周期长且标注标准易波动, 安全对齐时还会迫使标注员频繁接触有害文本。",
        steps: [
          { name: '人工排序', pain: "(原点) 偏好对齐依赖大规模成对比较标签, 传统管线只能依赖人工逐条审阅标注", fix: "由人工标注员对同题多条回复排序 (InstructGPT 每次排 4~9 条): 标签质量可控, 但人力成本与耗时随数据规模线性增长" },
          { name: 'Constitutional AI', year: 2022, pain: "人工排序成本高、吞吐低, 且红队安全评测需要标注员持续审阅大量有害输出", fix: "将评判规范写成显式宪法条文: 先由 AI 按条文批评并改写回复做 SFT, 再由 AI 成对比较生成偏好标签以训练 RM 并执行在线 RL" },
          { name: '改写前后直接成对', pain: "两阶段流程需串联 SFT、AI 成对评审、RM 训练与在线 RL, 工程链路冗长且计算开销大", fix: "直接将「改写后回复优于改写前回复」构造成偏好对送入现成 DPO 管线训练; 对齐上限完全受限于评审器 judge 实际执行的检查逻辑" },
        ],
      },
      code: 'llm_finetune/methods/rlaif.py · llm_finetune/run_finetune/rlaif/train_rlaif.py',
      points: [
        { title: '用成文宪法与自动改写替代高成本人工排序', body: '人工收集安全与偏好数据不仅昂贵缓慢, 还会让标注员长期暴露在有害文本中。Constitutional AI 把好回答的规范整理成显式条文 (宪法), 让评审模型对初始回复逐条挑刺 (critique) 并原地修订 (revise), 修订后的合规文本与原始违规文本天然构成一组 (chosen, rejected) 偏好对, 直接接入现成的 DPO 管线即可完成对齐。' },
        { title: '单轮离线合成管线的完整数据流', body: '本库以一个只会无脑原样照抄输入 (含敏感词 15 与隐私号 13) 的基座模型为起点, 完整跑通单轮 RLAIF:\n- 批量采样与批评: 按温度 1 采样 4,096 条回复, 评审规则检出其中 2,539 条存在违规。\n- 自动脱敏成对: 将违规位置的 15 和 13 替换为掩码符 3 (如 [4, 8, 12, 15, 13, 15] -> [4, 8, 12, 3, 3, 3]), 原本就合规的样本因无改写差异直接跳过, 共产出 2,539 对偏好样本送入 DPO 训练 150 步。' },
        {
          title: '对齐天花板锁死在 judge 实际执行的检查逻辑上',
          key: true,
          body: '训练梯度完全来自偏好对里的差异信号——条文里写了但 judge 没查出来的漏洞, DPO 根本学不到。\n实验中「无害」条文文字写明禁止脏话 15 及其变体 14, 但故意让底层检查函数只查 15 漏掉 14。结果 14 在改写前后的 chosen 与 rejected 中原样并存、相减抵消: 训完后被查的违规率从 0.648 骤降至 0.031, 而漏检的变体 14 出现率仍高达 0.320 (残余微降仅来自首词偏移副作用, 脚本专门断言其保留率超过训前的 60%)。换成真实大模型做裁判时, 裁判自身的盲区同样会原封不动遗传给学生。',
        },
        { title: '评审改写若改变系统性长度, DPO 会优先偷学短句捷径', body: '若在宪法中加入「删掉连续重复词」这类会缩短句长的改写规则, 所有 chosen 都会系统性比 rejected 短一截; DPO 会迅速学成「吐 1 个词立刻输出 EOS 闭嘴」, 让表面违规率归零的同时把正常回答也清空了 (理想回复比例从 0.395 降到 0.000)。因此合成偏好对时必须严防改写动作引入非预期的格式或长度偏差。' },
        { title: '安全指标必须与有用性 (helpfulness) 成对验收', body: '单看违规率下降极易被「拒绝回答一切」的模型欺骗, evaluate 同时在留出集上追踪有用性指标: 训练 150 步 (lr=1e-4) 后非敏感词正确照抄率从 1.000 降到 0.856 (这是有用性付出的代价), 整条回复与「正常词照抄 + 敏感词打码」理想答案完全匹配的比例从 0.352 提升至 0.488 (若盲目把学习率拉高到 3e-4 训 200 步, 违规率虽降为 0, 理想回复率却会崩塌至 0.051)。' },
      ],
      links: [
        { from: 'CONSTITUTION', to: 'critique / revise', body: 'CONSTITUTION 将每条原则定义为 (名称, 条文说明, 检查函数, 修订函数) 四元组; 自然语言条文仅供人类审阅, 实际训练信号完全取决于后两个函数。' },
        { from: 'build_preference_pairs', to: 'DPOLoss', body: '将 (revise(y), y) 打包为标准成对数据接口, 零代码修改直接复用 PairwiseForward 与 DPOLoss。' },
        { from: 'DPO 的 likelihood displacement', to: 'RLAIF 的有用性损失', body: '非敏感位照抄率从 1.000 降到 0.856 主要集中在首个回复位置, 正是前文 DPO 概率挤压效应在合成偏好集上的复现。' },
        { from: '一轮离线 DPO', to: '真实 RLAIF', body: '工业级 Constitutional AI 通常分「改写后先做一轮 SFT 热身」与「AI 成对盲评训 RM 再跑在线 RL」两阶段或多轮迭代, 且批评与改写均由大模型生成。' },
      ],
      sourceRows: [
        { concept: '宪法', code: 'llm_finetune/methods/rlaif.py: CONSTITUTION', takeaway: '声明「无害」与「隐私」两条原则, 刻意保留条文提及变体 14 但检查函数只查 15 与 13 的对照缺口。' },
        { concept: '批评', code: 'llm_finetune/methods/rlaif.py:critique', takeaway: '遍历原则列表定位违规下标, 返回空列表表示原本合规。' },
        { concept: '改写', code: 'llm_finetune/methods/rlaif.py:revise', takeaway: '按检出位置原地将违规 token 替换为脱敏符 3, 保持序列长度不变以避免引入长度捷径。' },
        { concept: '造偏好对', code: 'llm_finetune/methods/rlaif.py:build_preference_pairs', takeaway: '仅收集触发批评的样本构造 (chosen=revise(y), rejected=y), 并清理 EOS 之后的填充尾部。' },
        { concept: '离线池', code: 'llm_finetune/methods/rlaif.py:AIPreferenceData', takeaway: '封装合成出的偏好对池供 DPO 每步随机抽样训练。' },
        { concept: '验收', code: 'llm_finetune/run_finetune/rlaif/train_rlaif.py:evaluate', takeaway: '四维联合验收: 已查违规率、漏查变体保留率、正常位照抄率与整条理想匹配率。' },
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
      subtitle: '多步推理链只要中间错一步就会导致满盘皆输。结果奖励模型 (ORM) 只在句末给一个总分, 过程奖励模型 (PRM) 则为每一步中间推导单独打分, 既加密了监督信号又能过滤「过程瞎编但答案碰巧蒙对」的伪正例。',
      tldr: '对比多步推理的两种判分训练范式:\n- ORM (结果奖励模型): 整条 $K$ 步推导只在末尾 EOS 处给 1 个答案对错标签, 信用分配稀疏且会把「过程算错、末步碰巧撞对答案」的样本标成正例。\n- PRM (过程奖励模型): 在每一步中间结果处各给 1 个局部对错标签, 同样训练 600 步后, best-of-8 选解准确率从 ORM 的 0.488 (接近随机的 0.473) 大幅提升至 0.648, 并能以 0.575 的准确率直接指出首个出错步。',
      question: '面对同样一批 4 步算术推导题和相同的训练步数预算, 为什么只在终点打分的 ORM 几乎学不动, 而逐步打分的 PRM 却能迅速拉开差距?',
      evolution: {
        title: '奖励从终点挪到每一步',
        subtitle: "根问题: 多步推理若仅在序列终点给出单一的 0/1 结果反馈, 模型便难以从稀疏信号中定位具体出错的中间步骤。",
        steps: [
          { name: 'ORM', pain: "(原点) 在多步推理任务中, 最易自动核验的监督信号只有最终答案是否正确", fix: "仅在序列末尾对最终答案打一个二分类标签: 标注成本低, 但无法指明出错位置, 且中间推导错误却碰巧蒙对答案的样本也会被标为正例" },
          { name: 'PRM', year: 2023, pain: "单一 0/1 结果标签分摊到 $K$ 个推理步上导致信用分配极度稀疏, 且过程出错但答案碰巧正确的伪正例会污染判分器", fix: "在每个中间步骤结果处分别标注局部对错 (如人工逐步标注的 PRM800K): 单条样本监督信号密度提升 $K$ 倍, 且可直接定位首个出错步骤" },
          { name: '解分取 min', pain: "执行 best-of-N 选解时需将逐步得分聚合成整条候选解的总分, 若取算术平均会让单个错步的低分被其余正确步稀释", fix: "取各步通过概率的短板最小值 min (原论文主方案为各步概率连乘, min 为备选方案), 贴合「一步算错则整条推导失效」的逻辑" },
        ],
      },
      code: 'llm_finetune/methods/prm.py · llm_finetune/run_finetune/prm/train_prm.py',
      points: [
        { title: 'ORM 信号稀疏且受「错过程蒙对答案」污染', body: '在 4 步模 10 算术链任务上, 一条错误推导可能坏在第 1 步到第 4 步中的任意位置, ORM 拿到的却只有句末 1 个二值标签, 必须靠大量样本自己反推究竟哪一步出了错——训练 600 步后其二分类交叉熵仍停在 0.682, 与纯随机猜测 ($\ln 2 = 0.693$) 几乎无异。\n更隐蔽的是候选池中约有 4.5% 的样本属于中间步骤算错、后续步骤碰巧抵消蒙对了最终答案, 这些伪正例在 ORM 训练集里被当成满分样本奖励。' },
        { title: 'PRM 将全局判断拆解为 K 个局部单步核验', body: 'PRM 在每一步计算结果 $v_i$ 所在的位置上都施加一次二分类监督, 单条样本提供的有效标签数直接放大 $K=4$ 倍。\n单步真值采用局部条件判定: 只要当前步基于纸面上一步写下的数值 $v_{i-1}$ 正确执行了运算 $v_i = \\mathrm{op}_i(v_{i-1}, k_i)$, 无论前一步本身对错, 当前步都判为局部合法——这样每个位置的判分头只需学会核对眼前这一步四则运算, 学习难度骤降 (600 步后 BCE 降至 0.359)。' },
        {
          title: '同网络结构与同训练预算下, 标签粒度直接决定选解质量',
          key: true,
          body: '固定完全相同的「主干 + 标量头」网络、相同随机初始化与相同 600 步训练流, 在 512 道留出题 (每题 8 条候选解) 上做 best-of-8 挑解评测:\n- 最终答案正确率: 随机盲挑 0.473, ORM 0.488, PRM 达 0.648。\n- 四步过程全对率: 随机盲挑 0.426, ORM 0.422, PRM 达 0.627。\n- 首个错步定位准确率: PRM 取最低分步骤定位首错达 0.575 (猜固定第 1 步基线仅 0.341), 而只输出单一总分的 ORM 根本不具备步骤定位能力。',
        },
        { title: '整条解聚合取短板最小值 (min)', body: '推理选解时需要将 PRM 给出的 $K$ 个步骤通过概率合成为整条候选的总分: 若取算术平均, 1 个致命错步会被其余 3 个正确步的高分稀释; 取短板最小值 $\\min_i p_i$ 则符合「一步错、整条废」的木桶逻辑, 且分值不会随推理步数 $K$ 增多而变小。论文 (Lightman et al.) 的主方案是各步概率连乘, min 是它比较过的备选; 连乘同样遵循木桶逻辑, 但分值会随步数增多而变小。' },
        { title: '实验结论的现实边界', body: '对读实验数字时需注意四点工程差异:\n- ORM 是学得慢, 不是学不会: 脚本外另跑同配置 3,000 步, 约 2,000 步后 BCE 才开始下降, 3,000 步时约 0.55; 本实验展示的是同等有限算力预算下的样本效率差距。\n- 本库步骤标签由算术规则免费生成, 真实复杂推理的逐步标注 (如人工标 PRM800K, 或像 Math-Shepherd 那样用 rollout 估计自动标) 成本远高于结果标签。\n- 候选解来自带噪的程序求解器 (每步以 20% 的概率写错一个数, 之后照着错值往下算), 不是语言模型采样, 错误分布比真实场景简单。\n- 8 条候选的理论上限 pass@8 为 0.992, PRM 的 0.648 仍有继续训练提升的空间。' },
      ],
      links: [
        { from: 'RewardModel', to: 'PRM / ORM', body: '三者共用同一个主干加线性标量头架构, 仅监督位置不同: ORM 只取 EOS 末位, PRM 则取全部 $K$ 个步骤结果位。' },
        { from: 'PRM 解分', to: 'infer · 推理时计算', body: '阶段 5「推理时计算」用 PRM 驱动按步剪枝的束搜索 (prm_beam_search), 在错步刚出现时就地淘汰以节省解码算力。那里的 PRM 是按规则判每步、再加高斯噪声的程序化判分器, 不是本章训出来的模型; 本章讲的是 PRM 怎么训出来。' },
        { from: 'PRM 的逐步分', to: 'PPO 的逐 token 优势', body: '将 PRM 的步骤得分注入强化学习可把末端稀疏奖励转化为稠密过程信号, 发挥下一章 PPO GAE 逐位置信用分配的长处。' },
        { from: 'step_positions', to: 'v_i 所在位置', body: '第 $i$ 步的判分必须读在结果 token $v_i$ 本身的位置上 (此时因果注意力刚读完 $v_i$), 早读一格模型尚未看见该步算出的数字。' },
      ],
      sourceRows: [
        { concept: '玩具数据', code: 'llm_finetune/methods/prm.py:ArithChain', takeaway: '生成 4 步模 10 算术推导序列, 按题干参数和 mod 5 == 0 切出严格不相交的留出测试集。' },
        { concept: '逐位置取分', code: 'llm_finetune/methods/prm.py:token_scores', takeaway: '对整条序列所有位置并行投影输出标量打分 [N, T], 供任意步骤下标切片监督。' },
        { concept: 'PRM loss', code: 'llm_finetune/methods/prm.py:prm_loss', takeaway: '在 4 个中间步骤结果位置上计算二分类交叉熵损失。' },
        { concept: 'ORM loss', code: 'llm_finetune/methods/prm.py:orm_loss', takeaway: '仅在序列末尾 EOS 位置计算单次二分类交叉熵损失。' },
        { concept: '整条解的分', code: 'llm_finetune/methods/prm.py:solution_scores', takeaway: 'PRM 对各步通过概率取 min 短板聚合, ORM 直接取末位概率, 用于 best-of-8 排序选优。' },
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
      subtitle: '标准 PPO 额外维护一个与策略模型同尺寸的价值网络 (critic), 配合广义优势估计 (GAE) 为每个生成位置单独做信用分配。通过与冻结 critic 及 GRPO 同预算对照, 可以清楚看清 critic 的显存代价与生效前提。',
      tldr: 'PPO 相比 GRPO 多养了一个逐位置预测未来期望回报的 critic 网络, 常驻参数涨至 3 份 (policy / ref / critic), 60 步训练耗时翻倍:\n- 机制优势: 支持「每题只采 1 条」的多样化题面覆盖, 并由 GAE ($\\lambda \\in [0, 1]$) 在低方差单步 TD 与无偏蒙特卡洛回报之间平滑插值。\n- 小规模实测真相: 虽然 PPO 留出集 pass@1 (0.381) 高于同总采样数的 GRPO (0.287), 但直接将 critic 学习率置零冻死后依然能跑出 0.389——说明此处领先来自「256 道不同题各采 1 条避免了 32% 全同废组」, 而非未经预热的小 critic 学准了前缀估值 (其方差解释率仅 2.5%)。',
      question: '在小规模短步数的强化学习中, PPO 跑赢 GRPO 的那截分数真的是 critic 逐位置估值立的功吗? 冻住 critic 会发生什么?',
      evolution: {
        title: '末尾一个分, 怎样分给每个 token',
        subtitle: "根问题: 验证奖励仅落在回复末尾最后一个 token 上, 需要将信用合理分配给前序每个生成位置, 同时控制策略梯度的采样方差。",
        steps: [
          { name: 'REINFORCE', year: 1992, pain: "(原点) 整条回复仅在序列末尾获得一个总体奖励分, 需要据此反推前序哪些 token 应当上调生成概率", fix: "令每个位置共用整条序列总回报乘上 $\\nabla \\log \\pi$: 梯度估计虽无偏, 但方差很大, 且同一条回复内的好坏 token 会被无差别同奖同罚" },
          { name: '减去 V(s_t)', pain: "当奖励恒为非负时所有采样序列的 token 都会被上调概率, 缺乏基准线中心化导致梯度方差居高不下", fix: "引入价值网络 critic 预测当前前缀往后的期望回报 $V(s_t)$ 并作为基线减去: 获得了逐 token 相对优势, 代价是需额外训练一套同尺寸网络" },
          { name: 'TD(0)', pain: "蒙特卡洛回报需等待序列生成结束才能结算, 后续每一步的采样随机性都会叠加进当前位置的优势估计", fix: "改为单步时序差分残差 $\\delta_t = r_t + \\gamma V(s_{t+1}) - V(s_t)$ (用下一状态估值替代后续真实回报): 显著压低了采样方差, 但 $V$ 拟合不准时会引入偏差" },
          { name: 'GAE', year: 2015, pain: "蒙特卡洛估计 (直接使用完整序列回报) 无偏但方差高, 单步 TD(0) 方差低但受限于估值偏差, 难以兼顾", fix: "将后续各步残差 $\\delta_{t+k}$ 按 $(\\gamma\\lambda)^k$ 指数衰减累加, 由 $\\lambda \\in [0, 1]$ 在无偏与低方差间平滑插值; 本章取 $\\lambda = 0.95$" },
        ],
      },
      code: 'llm_finetune/methods/ppo.py · llm_finetune/run_finetune/ppo/train_ppo.py',
      points: [
        { title: '两种基准线路线: 同题多采组均值 vs 跨题单采价值网络', body: '为降低策略梯度方差, 每个动作都需要减去当前状态下的期望得分基准 $b$:\n- GRPO 路线: 同一道题采样 $G$ 条回复取均值, 零额外参数但绑定「少题多采」且整条序列共享单一优势标量。\n- PPO 路线: 用独立训练的 critic 网络在每个位置预测状态价值 $V(s_t)$, 每道题只需采 1 条即可算出逐 token 优势 $A_t$, 代价是可训参数与优化器状态直接翻倍。' },
        { title: 'GAE 通过折扣参数 λ 权衡估值偏差与采样方差', body: '广义优势估计 (GAE) 分两步构造逐位置优势 $A_t$: 先计算单步时序差分残差 $\\delta_t = r_t + \\gamma V(s_{t+1}) - V(s_t)$ (衡量当前步实际进展超出 critic 预期的幅度), 再从序列末尾向前按 $(\\gamma\\lambda)^k$ 指数衰减累加 $A_t = \\sum_{k \\ge 0} (\\gamma\\lambda)^k \\delta_{t+k}$:\n- 当 $\\lambda = 0$ 时退化为单步 TD(0) ($A_t = \\delta_t$), 方差最小但完全受制于 $V$ 的预测偏差。\n- 当 $\\lambda = 1$ 时展开望远镜求和恰好还原无偏蒙特卡洛回报减基线 ($\\sum \\gamma^k r_{t+k} - V(s_t)$), 无偏但累加了后续所有步的采样噪声。\ncheck_gae 对两端做了精确单测验证; LLM 训练通常取 $\\gamma=1.0, \\lambda=0.95$。' },
        { title: '三步序列手算核对 GAE 的两端行为', body: '取 check_gae 中长度为 3、$\\gamma=0.9$ 的测试序列直观核对: 设逐位置奖励 $r = [-0.14, -0.11, -0.61]$, 未校准的 critic 估值 $V = [1.71, 2.38, -1.13]$, 终止态价值为 0。从后往前逆推单步残差:\n- $\\delta_2 = -0.61 + 0 - (-1.13) = 0.52$\n- $\\delta_1 = -0.11 + 0.9 \\times (-1.13) - 2.38 = -3.507$\n- $\\delta_0 = -0.14 + 0.9 \\times 2.38 - 1.71 = 0.292$\n对比位置 0 的优势: 取 $\\lambda=0$ 时 $A_0 = \\delta_0 = +0.292$; 取 $\\lambda=1$ 时 $A_0 = \\delta_0 + 0.9\\delta_1 + 0.81\\delta_2 = -2.443$ (与真实折扣总回报 $-0.733$ 减去 $V_0=1.71$ 完全一致)。可见当 critic 尚未学准时, 调小 $\\lambda$ 甚至会把优势的正负号都带反。' },
        { title: '末端验证分叠加逐位置 KL 惩罚构成稠密奖励流', body: '遵循 InstructGPT 经典配方, PPO 将正则项直接写进逐 token 奖励向量: 每个生成位置先扣去 $\\beta(\\log\\pi_{\\text{old}} - \\log\\pi_{\\text{ref}})$ (β=0.02; 这个 token 在当前策略下的对数概率比在 ref 下高得越多, 扣得越多), 而验证器给出的 0/1 成败分仅加在回复的最后一个有效 token 上, 随后统一交给 GAE 向前传播信用。' },
        {
          title: '冻结消融揭示真相: 短程小规模下收益来自题面覆盖而非 critic',
          key: true,
          body: '从同一 SFT 起点出发、每步固定采样 256 条回复跑 60 步: PPO 留出集 pass@1 为 0.381, 高于 GRPO (32 题 × 8 条) 的 0.287。但进一步做两项诊断便能看清真实归因:\n- 方差解释率极低: 训练结束时 critic 的 $V(s_0)$ 仅能解释回报方差的 2.5%, 远不如 GRPO 免费组均值的 36.5%。\n- 冻结 critic 消融: 将 critic 学习率直接设为 0 (全程输出接近 0 的初始随机值, 仅保留批内优势白化), 跑出的 pass@1 达 0.389, 与正常训练 critic 的 0.381 完全打平。\n真正拉开 0.09 差距的是采样配比: PPO 每步覆盖 256 道不同题, 而 GRPO 32 道题里有约 32% 的组因全对或全错沦为零梯度废组。大规模训练中 critic 通常从 RM 或 SFT 初始化并先单独预热, 能学到有用的 V, 那时逐 token 价值估计是否值回它翻倍的显存与耗时 (常驻参数 299K 对 99K, 60 步耗时 11.8 s 对 6.1 s), 结论可能不同。',
        },
      ],
      links: [
        { from: 'RewardModel', to: 'critic', body: 'critic 复用 RewardModel 的主干加标量头结构, 区别在于读取全部生成步骤的状态估值 V(s_t) 而非仅取句末一位。' },
        { from: '组均值', to: 'critic', body: 'GRPO 组均值在单题单采 (G=1) 时恒等于自身回报导致优势归零; critic 跨题共享参数, 支持 G=1 的高题面多样性采样。' },
        { from: 'PPO clipped surrogate', to: 'GRPO 进阶', body: '内层多轮复用样本的比率截断目标在两者间完全通用, 仅优势张量 A 的计算路径不同。' },
        { from: 'PRM 逐步分', to: '逐 token 优势', body: '若将末端稀疏验证分升级为上一章 PRM 的逐步过程奖励, GAE 便能直接利用每步真实反馈做细粒度信用分配。' },
      ],
      sourceRows: [
        { concept: '逐位置估值', code: 'llm_finetune/methods/ppo.py:token_values', takeaway: '在与对数似然对齐的错位下标 $P-1+t$ 处提取生成第 $t$ 个词之前的状态估值 $V(s_t)$。' },
        { concept: 'GAE', code: 'llm_finetune/methods/ppo.py:gae', takeaway: '逆序递推 $A_t = \\delta_t + \\gamma\\lambda A_{t+1}$, 并将 EOS 之后的填充位严格掩码归零以防从无效尾部自举。' },
        { concept: '逐 token 奖励', code: 'rewards[arange(N), last] += score', takeaway: '全序列铺底逐位置 KL 惩罚, 仅在每条回复最后一个有效位置叠加验证器得分。' },
        { concept: '一步', code: 'llm_finetune/methods/ppo.py:PPOTrainer', takeaway: '依次执行在线采样、冻结旧策略/ref/旧价值三组读数、GAE 计算与批内白化、$\\mu=2$ 轮策略裁剪加价值均方误差联合更新。' },
        { concept: '两端单测', code: 'llm_finetune/run_finetune/ppo/train_ppo.py:check_gae', takeaway: '断言 $\\lambda=0$ 时与单步 TD 一致、$\\lambda=1$ 时与蒙特卡洛折扣回报一致 (容差 1e-6 / 1e-5), 实测误差分别为 0.0 与 4.8e-07。' },
        { concept: '解释方差', code: 'llm_finetune/run_finetune/ppo/train_ppo.py:baseline_explained_var', takeaway: '通过 $1 - \\mathrm{Var}(R - b)/\\mathrm{Var}(R)$ 量化基准线消减了多少成回报波动。' },
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
      subtitle: '把本阶段全部 16 种后训练与适配方法摆在同一张工程账单上: 先按手里的数据形态筛定优化目标, 再按显存份数、每步前向次数与是否允许在线采样选定具体方案, 最后把模型权重与分词/模板契约完整打包交付。',
      tldr: '选型微调与对齐方案主要核对三笔资源账:\n- 常驻模型份数与前向次数: 无参考模型的 SFT / SimPO / ORPO 只需 1 份权重 (389 KiB) 和单步 1 次前向; DPO / KTO 需额外常驻冻结 ref (778 KiB, 2 次前向); PPO 则需同时常驻 policy / ref / critic 三份网络。\n- 离线读取还是在线采样: 离线方法直接消费静态数据集, GRPO / PPO / on-policy 蒸馏则要求每步调用当前模型自回归生成候选序列。\n- 交付产物完整性: LoRA 类方案应独立保留轻量 adapter (76 KiB) 以支持多租户动态挂载, 且任何模型落盘交付都必须附带架构配置、词表与对话模板清单 (manifest) 并在独立进程中核验一致性。',
      question: '当手头只有示范问答对且显存吃紧时该怎么搭组合? 换成成对偏好日志、单边点赞点踩或可由程序自动判题的任务时, 选型又该如何切换?',
      evolution: {
        title: '数据、显存、采样, 一道道筛',
        subtitle: "根问题: 后训练各方法的本质差异体现在资源代价结构上; 手头的数据形态、显存预算与是否支持在线采样直接决定了可行方案集合。",
        steps: [
          { name: '全参 SFT', pain: "(原点) 手头仅有成对的 (prompt, response) 示范数据时, 最直接的起步方案是全参数 SFT", fix: "常驻 1 份模型权重、每步仅跑 1 次前向; 但显存需同时容纳全量梯度与 Adam 的 2 份动量状态, 且每个下游任务都要独立落盘整份权重" },
          { name: 'LoRA / QLoRA', pain: "显存难以容纳全参梯度与 Adam 优化器状态, 且多任务部署保存多份完整模型存储开销过高", fix: "冻结主干仅训练旁路低秩 adapter: 单个任务落盘仅 76 KiB (整模型为 389 KiB); QLoRA 进一步将冻结基座压缩为 4-bit NF4, 静态权重降至 59 KiB" },
          { name: 'DPO', pain: "单向示范问答只监督「应该输出什么」, 无法利用负例区分「哪条回复更优、哪条应当规避」", fix: "直接消费成对偏好三元组训练策略: 需同时常驻在训 policy 与冻结 ref 两份权重 (共 778 KiB), 每步执行 2 次模型前向" },
          { name: 'SimPO / ORPO', pain: "冻结的 ref 额外占用整份静态权重显存, 并使每步多出一次前向计算开销", fix: "移除 ref 后权重降至 389 KiB、每步 1 次前向, 单步提速约 25%; 防漂移锚点需另行补位 (SimPO 依赖小 lr 与短步数, ORPO 依赖损失自带的 SFT 项)" },
          { name: 'GRPO / PPO', pain: '偏好对是离线的; 答案能被程序判对时, 离线数据用不上这个信号', fix: 'GRPO 一题在线采 G 条交给 verifier 打分; PPO 再养一个逐 token 估分的 critic: 常驻三份, 耗时约两倍' },
        ],
      },
      code: 'llm_finetune/run_finetune/{sft,lora,dora,qlora,merge,rm,prm,dpo,kto,simpo_orpo,rlaif,ppo,grpo,distill,on_policy_distill}/train_*.py',
      points: [
        { title: '「优化什么损失」与「放开哪些参数」是正交的两个维度', body: '梳理十几套名字时, 先把它们拆成正交的两层就不会眼花:\n- 训练目标层 (SFT / DPO / KTO / SimPO / ORPO / GRPO / PPO / 蒸馏): 由手头的数据形态决定——示范问答用 SFT, 成对偏好用 DPO/ORPO, 单边好坏标签用 KTO, 可验证推理题用 GRPO/PPO, 跨尺寸能力迁移用蒸馏。\n- 参数更新层 (全参 / LoRA / QLoRA / DoRA): 由显存预算和部署形态决定——两层可自由正交组合 (如 QLoRA + DPO 或 LoRA + GRPO), 代码目录也对应拆分为算法积木 methods/ 与实验编排 run_finetune/。' },
        {
          title: '用「常驻份数、单步前向次数、落盘产物」三项指标核算工程成本',
          key: true,
          body: '确定适用数据形态后, 不同算法的计算与显存账单可以直接按三栏对齐 (本库基座参数量 99,648, 单份 fp32 权重为 389 KiB):\n- 单步模型前向次数: SFT / SimPO / ORPO / RM / PRM 每步仅跑 1 次主干前向; DPO / KTO / 离线蒸馏每步需跑 2 次 (在训模型与冻结参照/教师各 1 次); 在线 RL 与在轨蒸馏则需先执行多步自回归生成。\n- 常驻模型份数: 无参照方法仅占 1 份 (QLoRA 把这份基座压到 59 KiB, 另加 76 KiB 的 LoRA 旁路), 带冻结参照占 2 份 (778 KiB), 标准 PPO 加上可训 critic 占满 3 份。\n- 落盘交付物: 全参训练导出完整权重, PEFT 导出轻量 adapter, RM / PRM 额外包含线性打分头 value_head。',
        },
        { title: 'PEFT 训完务必独立归档原始 adapter, 勿只留合并后的大权重', body: '虽然合并进基座 (merge_lora_weights) 便于单模型测试, 但归档交付时务必单独保存未合并的原始 adapter 权重 (本库 r=8 仅 76 KiB): 只有保留独立的小矩阵字典, 下游推理引擎才能在一份共享基座上动态挂载数十个业务分支 (Multi-LoRA), 并与特定基座版本摘要绑定校验。' },
        { title: 'KTO、RLAIF 与 PRM 复用了现成计算骨架, 差异集中在数据侧', body: '这三项进阶方法的训练循环其实都是前面基础算法的熟面孔:\n- KTO: 前向与显存结构同 DPO (2 份权重、每步 2B 条序列前向, 其中一半用于错配样本估计参考点 $z_0$), 只是把输入从成对比较放宽为单条好/坏标签。\n- RLAIF: 训练期本身就是标准 DPO, 只是前置了一轮按宪法条文自动批评改写的偏好合成管线。\n- PRM: 前向结构与 RM 完全一致, 只是把二分类监督从句末 1 个点铺开到了中间全部 $K$ 个推理步骤位上, 主要成本转移到了步骤级数据构造。' },
        { title: '代价谱系的两端: 最重的三模型 PPO 与零训练的模型合并', body: '站在整个阶段的资源谱系两端看:\n- 最重一端是 PPO: 可训参数翻倍 (199,360)、常驻三套模型, 每步历经「在线采样 + 3 次基准前向 + $2\\mu$ 次内层前向更新」, 60 步耗时约为同采样数 GRPO 的两倍。\n- 最轻一端是模型合并 (Task Arithmetic / TIES / DARE): 训练期前向次数为 0、无需任何样本与梯度, 直接对同一基座派生出的 state_dict 逐张量做加减与合成 (TIES、DARE 还会先修剪或随机丢弃坐标), 导出一份合并后的全量权重。' },
      ],
      links: [
        { from: 'methods/*.py', to: 'run_finetune/*/train_*.py', body: 'methods/ 封装纯粹的损失函数与模块包装器, run_finetune/ 各子脚本负责准备合成数据、驱动训练并在严格隔离的留出集上执行 assert 验收。' },
        { from: 'PairwiseForward / TeacherStudent', to: '通用 Trainer', body: '离线多模型方法将主模型与冻结参照/教师的前向打包进同一个包装模块中, 使简洁的单步 Trainer 无需感知内部前向次数。' },
        { from: 'GRPO / PPO / on-policy 蒸馏', to: '不用 Trainer', body: '在线方法每步的数据依赖当前最新权重现场自回归生成, 且常需缓存采样时刻对数概率做多轮内层迭代, 直接手写清晰的外内双层循环反而比硬套回调钩子更易读。' },
        { from: 'print_trainable_parameters', to: '验收 PEFT', body: '检验低秩冻结是否生效不能靠看 loss 下降, 必须直接清点 requires_grad=True 的元素总数 (且对共享词表嵌入去重计数)。' },
        { from: 'adapter state_dict', to: 'llm_infer Multi-LoRA', body: '阶段 4 导出的轻量 lora_A / lora_B 权重字典, 正是阶段 5 多租户推理服务同批混跑不同业务定制模型时加载的那类资产 (推理章的演示用的是自己构造的 adapter, 并不读取这里的导出)。' },
      ],
      sourceRows: [
        { concept: '共用基座', code: 'llm_finetune/run_finetune/common.py:pretrained_base', takeaway: '统一构建 2 层、d_model=64 的小 LLaMA 并预训练学会 copy 任务 (留出集 EM > 0.95), 为 LoRA、DoRA、QLoRA、模型合并与 RLAIF 五个脚本的横向对比提供一致起点。' },
        { concept: 'SFT / LoRA / DoRA / QLoRA', code: 'run_finetune/{sft,lora,dora,qlora}/train_*.py', takeaway: '同吃 (x, y) 示范对、单步 1 次前向, 依次对比全量更新、低秩更新、幅度方向解耦与 4-bit NF4 基座压缩。' },
        { concept: 'RM', code: 'run_finetune/rm/train_rm.py', takeaway: '消费偏好对训练标量打分头; 注意构建 RM 时若后续还需原语言模型生成, 应先对 backbone 做深拷贝以防原地修改权重。' },
        { concept: 'DPO', code: 'run_finetune/dpo/train_dpo.py', takeaway: '双份模型常驻 (778 KiB), 单步拼接好坏回复为 2B 序列各跑一次 policy 与冻结 ref。' },
        { concept: 'SimPO / ORPO', code: 'run_finetune/simpo_orpo/train_simpo_orpo.py', takeaway: '移除冻结 ref 将常驻权重减半至 389 KiB, 单步耗时缩短约 25%。' },
        { concept: 'GRPO 系', code: 'run_finetune/grpo/train_grpo.py', takeaway: '仅输入题干 prompt, 每题在线采样 $G$ 条并交确定性验证器判分, 通过五个配置开关切换四种主流变体。' },
        { concept: '蒸馏', code: 'run_finetune/{distill,on_policy_distill}/train_*.py', takeaway: '常驻冻结教师网络, 对照静态语料正向 KL 与学生自采样轨迹反向 KL。' },
        { concept: 'PPO', code: 'run_finetune/ppo/train_ppo.py', takeaway: '常驻 policy / ref / critic 三份模型并维护双优化器, 包含冻结 critic 的归因消融实验。' },
        { concept: 'PRM', code: 'run_finetune/prm/train_prm.py', takeaway: '与 ORM 同构但将二分类交叉熵监督施加在全部 $K$ 个中间推理步骤结果位上。' },
        { concept: 'KTO', code: 'run_finetune/kto/train_kto.py', takeaway: '消费非成对单边好/坏反馈, 利用批内错位拼接样本估计参考点 $z_0$。' },
        { concept: 'RLAIF', code: 'run_finetune/rlaif/train_rlaif.py', takeaway: '先按宪法规则对采样结果执行一轮批评与脱敏改写合成偏好池, 再交由 DPO 训练并做四维验收。' },
        { concept: '模型合并', code: 'run_finetune/merge/train_merge.py', takeaway: '零训练直接在 state_dict 上执行 Task Arithmetic、TIES、DARE 与 SLERP 四种权重合成。' },
        { concept: '参数统计', code: 'llm_finetune/utils/param_utils.py:print_trainable_parameters', takeaway: '去重统计可训练参数量与总参数量占比, 用来验收是否只训了 adapter; 本仓库只有 SFT 脚本直接调用它, PEFT 脚本用同文件的 count_parameters 做同样的计数。' },
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
      "title": "对话模板 (Chat Template): 把结构化消息压成严格对齐的 token 序列",
      "body": "上层应用传递的是带角色 (system / user / assistant / tool) 的消息列表, 底层语言模型只接收一维整数序列。chat_template.py 将每条消息格式化为「角色控制头 + UTF-8 字节偏移正文 + 轮次结束符」拼接成完整训练串, 并在推理末尾补上 assistant 角色头引导生成。\n训练与线上推理必须复用逐字节一致的模板实现: 普通回复以 EOS 收尾, 发起工具调用的轮次以暂停符 END 收尾等待外部观察; 同时特殊控制符仅由模板外层注入, 用户正文即便包含字面字符串 \"EOS\" 也只会编码成三个普通文本字节, 无法伪造控制边界 (7 轮示例对话序列化为 172 个 token, 训推前缀严格逐位相等)。",
      "source": "llm_finetune/data/chat_template.py:serialize"
    },
    {
      "title": "多轮与工具轨迹掩码: 只监督 assistant 决策, 屏蔽外部工具返回",
      "body": "在包含工具交互的多轮轨迹中, 只有 assistant 产出的自然语言回复与发起的工具调用 (tool_call JSON 及对应结束标记) 属于模型应当学习的动作, 而 system、user 以及外部环境返回的 tool 结果全部将 labels 置为 -100 (172 个 token 的七轮样本中仅 75 个有效监督位)。\n这里同样要区分两张掩码: loss_mask 屏蔽非 assistant 区间的梯度, 而 attention_mask 仅屏蔽末尾填充位, 确保模型前向能完整读到用户提问与工具返回结果。此外数据管线对角色伪造、未配对的孤立工具结果以及截断破坏 JSON 结构的超长样本直接报错拒收, 不把残缺协议喂进训练集。",
      "source": "llm_finetune/data/chat_template.py:training_example"
    },
    {
      "title": "拒绝采样微调 (RFT): 用执行验证筛出的自生成轨迹做 SFT",
      "body": "RFT (Rejection-Sampling Fine-Tuning) 保持标准交叉熵损失不变, 只把训练数据来源换成「模型自己多次采样并通过独立验证器筛选的成功轨迹」。train_rft.py 让三个分支从同一个小 LLaMA 起点并排对照:\n- 人工示范 SFT: 直接在 16 条标准成功轨迹 (32 个状态动作步) 上训练 40 步。\n- RFT: 冻结初始策略对每题采样 16 次共 256 条候选, 经执行器筛出 18 条合格轨迹 (覆盖 10 道题) 后做 40 步交叉熵更新。\n- 在线 RL: 分 32 轮共采样 256 条, 按当轮策略梯度实时更新。\n在本协议任务上三条路径留出集通过率均从 0 升至 1 (需注意 RFT 与在线 RL 虽然总采样条数相同, 但前向与梯度更新步数预算并不完全对等, 且实验聚焦协议状态迁移而非开放语义泛化)。",
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
  "subtitle": "面对事实过期或用户数据删除请求时, 先分清该改外部检索库、应用侧存储还是模型权重本身; 一旦直接修改共享参数, 就必须同时核验目标生效度、同义泛化、邻域局部性与隐私遗忘边界。",
  "tldr": "- 选对治理层级: 外部事实变更优先更新检索索引与缓存, 应用会话删除走外部记忆权限清理; 只有固化在模型权重内部的错误记忆或训练痕迹, 才涉及知识编辑与机器遗忘 (unlearning)。\n- 多维联合验收: 修改参数不能只看原题改对没有 (efficacy), 还必须检查换问法 (paraphrase)、相近与无关事实是否被误伤 (locality) 以及关联推理 (portability)。在 knowledge_edit 中三种编辑都把目标值改到了 7.0, 但相近邻居事实的连带偏移却相差悬殊 (10.8、4.8 与加邻域约束后的 0.0475)。",
  "question": "把模型参数里的某一条事实成功改对之后, 为什么相近概念常常跟着被带偏? 删掉数据库里的原始文件为什么不等于完成了模型参数遗忘?",
  "code": "llm_finetune/run_finetune/knowledge_edit/demo.py · llm_finetune/run_finetune/privacy_unlearning/demo.py",
  "points": [
    {
      "title": "先区分外部数据治理与参数级知识编辑, 按四项指标联合验收",
      "key": true,
      "body": "神经网络将大量事实叠加编码在同一组共享稠密矩阵里, 动一处权重往往牵动一片表征。因此工程上遇到信息更新需求时先做分层路由: 业务文档修订走 RAG 索引版本替换, 用户撤回历史记录走应用记忆库硬删除, 仅当模型内化知识出错时才执行参数编辑。\n任何参数级知识编辑都必须拆成四个维度独立打分: 原问题是否输出新答案 (efficacy)、同义改写或别名提问是否同步生效 (paraphrase)、相近及无关事实是否保持原样 (locality)、依赖新事实的下游推理是否跟进 (portability), 绝不能用单一命中率掩盖副作用。"
    },
    {
      "title": "删除应用层原始记录不等于消除模型权重中的记忆痕迹",
      "body": "清空外部数据库与检索索引只能阻止系统再次把该文本塞进上下文, 无法抹去训练阶段已经刻进模型参数里的统计印记。privacy_unlearning 在 8 条合成记录上训练线性模型后, 直接用待删样本的特征键查询仍能读出 0.198 (真值 0.2), 成员推断攻击识别训练集样本的真阳性率 (TPR) 为 1.0、假阳性率 (FPR) 为 0.0。\n严格的机器遗忘需要将「遗忘集表现、保持集精度、以及剔除该样本后从头独立重训的基准模型」三方并排核对: 在具备闭式充分统计量的岭回归上直接扣除目标样本贡献后, 参数与从头重训误差小于 1e-12, 待删键输出归零且保持集 MSE 稳定在 0.00011; 而非凸深度大模型没有这种精确闭式扣除公式, 不能仅凭目标题答错一次就宣称完成了遗忘。"
    },
    {
      "title": "训练期差分隐私 (DP) 与训练后机器遗忘解决的是两类不同承诺",
      "body": "差分隐私 (DP-SGD) 在训练阶段介入: 先把单样本梯度范数裁剪到不超过 1.0, 再向平均梯度注入高斯噪声 (噪声倍数 $\\sigma=4$, 折算到 8 条记录的平均梯度上, 标准差为 $\\sigma \\cdot 2C/n = 1$), 从而从数学上限制任意单条训练记录对最终权重分布的最大影响上界 (本例 40 步对应分析上界 $\\varepsilon \\approx 9.56, \\delta=10^{-6}$, 代价是保持集 MSE 从 0.00011 上升到 0.343)。成员推断 TPR 随之降到 0.125, 但这只说明这一个攻击没有打中, 只能当旁证, 不是隐私证明; 这个玩具实验也不提供完整的隐私会计。\n简言之, 差分隐私限制「单条数据从一开始能被记住多少」, 机器遗忘则试图在训练结束后「定向擦除特定已学记录的影响」, 二者机制与适用阶段互不替代。"
    },
    {
      "title": "邻域协方差约束如何压低编辑对相近事实的误伤",
      "body": "knowledge_edit 用 1×3 线性层将目标键 [1, 0, 0] 的输出从 1.0 修改为 7.0, 同时监测高相似邻居键 [0.8, 1, 0]、正交无关键 [0, 0, 1] 与别名键 [0.95, 0.05, 0] 的变化:\n- 全局粗暴平移 (全元素加 6): 目标达 7.0, 但无关事实偏离 6.0、相近邻居剧烈偏离 10.8。\n- 无约束秩一更新 (只沿目标键方向更新): 正交无关事实偏移降为 0.0, 但与目标键非正交的相近邻居仍被连带拉偏 4.8。\n- 邻域协方差约束编辑 (在协方差项中对邻居方向施加 100 倍保护权重, 思路对应 ROME/MEMIT 的协方差约束, 但这里的 C 是手写的): 目标精准命中 7.0、无关事实 0.0, 相近邻居连带偏移被压低两个数量级至 0.0475, 别名提问仍输出 6.51。\n本例只测了前三项 (目标键测 efficacy、别名键测 paraphrase、无关键测 locality), 没有测 portability; 别名向量是手工给定的, 别名答对不能代替真实语言里换问法的泛化。"
    }
  ],
  "links": [],
  "sourceRows": [
    {
      "concept": "知识编辑: 三种更新对照",
      "code": "llm_finetune/run_finetune/knowledge_edit/demo.py:experiment",
      "takeaway": "对比全局平移、无约束秩一编辑与带邻域保护权重的约束编辑: 三者均将目标键改至 7.0, 相近邻居连带偏移依次为 10.8、4.8 与 0.0475。"
    },
    {
      "concept": "隐私: 记忆、DP、删除三条路径",
      "code": "llm_finetune/run_finetune/privacy_unlearning/demo.py:experiment",
      "takeaway": "并排测量保持集 MSE、待删样本响应与成员推断 TPR, 验证闭式充分统计量扣除结果与独立重训误差小于 1e-12。"
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
topic.pages["finetune-runs"].points.push({"title": "模型交付包: 跨进程加载必须绑定词表与模板契约", "body": "训练产出的权重张量离开配套的输入输出约定就无法独立工作: 哪怕权重逐位一致, 只要线上服务换了不同顺序的词表或少一个换行符的对话模板, 模型收到的整数序列就会面目全非。因此完整的模型交付包必须将架构参数、权重文件、词表与特殊控制符、Chat Template、默认解码超参、基座标识与量化配置统一写入可信清单 (manifest)。\nartifact_delivery 训练一个小 LLaMA 并在全新独立进程中做加载核验: 完整 FP32 包在新进程中的 token、logits 与贪心输出误差全为 0.0; int8 存储反量化包需按实测误差 (0.008654) 设定明确容差; 而词表错配、模板错配或基座标识不符 (实验里换的是基座名, 没有核对基座权重摘要) 的包在加载入口即被直接拦截拒收 (注意单纯核对文件 SHA 哈希只能防传输损坏, 拦不住有人调换词表顺序后重算哈希, 必须由可信 manifest 校验语义契约)。"})
topic.pages["finetune-grpo-variants"].points.push({"title": "异步 RL 流水线: 追踪策略版本号与行为概率校正陈旧样本", "body": "工业级强化学习常将自回归采样器与梯度训练器异步解耦并行运行: 当训练器已将策略推进到 v3 时, 经验队列里送来的可能仍是 v1 采出的旧轨迹。因此每条 rollout 在生成当下就必须绑定当时的策略版本号与真实行为对数概率 behavior_logp (绝不能事后用训练器重算代替, 因为推理引擎的量化精度或采样截断往往与训练端存在细微差异)。\nasync_rollout 在真实期望奖励已知为 0.8 的双动作模型上对比四种陈旧样本处理策略: 直接当新样本用期望跌回旧策略的 0.2; 乘重要性比率无偏校正可恢复期望 0.8, 但有效样本量比例 (ESS) 缩水至 0.3077; 将比率裁剪在 [0.8, 1.2] 以有偏 (0.24) 换回低方差 (ESS = 0.968); 超出版本容忍度则直接拒收跳过更新, 配合有界队列背压与取消任务计数回收保障异步管线稳定。"})
topic.pages["finetune-sft"].related = [{"route": "agent-trajectory-learning", "label": "RFT: 先采样, 再筛选, 再做 SFT", "reason": "本章仅从训练样本来源角度介绍 RFT 如何复用交叉熵损失; SFT、RFT 与在线 RL 三个分支的并行对照实验及逐步执行记录详见该章。"}, {"route": "finetune-knowledge-update", "label": "知识编辑: 改对一个事实, 伤到多少邻居", "reason": "SFT 用于拟合一类任务的整体输出分布; 若只需定向修改模型参数记住的某一条事实, 则需采用参数级知识编辑算法并按四项指标联合验收, 详见该章。"}]
topic.pages["finetune-rlhf"].related = [{"route": "basic-optim-sample", "label": "复习 softmax 温度", "reason": "两章共用同一套温度缩放公式 $\\mathrm{softmax}(z/T)$: 基础篇演示温度如何调节解码分布的锐度, 本章进一步将其用于在线采样探索与软标签知识蒸馏。"}]
topic.pages["finetune-knowledge-update"].related = [{"route": "agent-memory-privacy", "label": "应用记忆的删除", "reason": "清除应用侧存储的单条记忆记录属于应用层数据治理, 关联章节专门讨论记忆的授权、过期与删除机制; 而训练阶段留在模型参数中的统计印记无法随之自动消失, 必须按本章方法单独执行参数遗忘与独立验证。"}]
topic.pages["finetune-knowledge-update"].evolution = {
  "title": "改一个事实, 先数邻居伤了多少",
  "subtitle": "根问题: 大量事实叠加编码在同一组共享参数矩阵中, 修改任一权重都会连带改变所有经由该参数映射的特征键输出。",
  "steps": [
    { "name": "改文档不改参数", "pain": "(原点) 外部客观事实发生变更后, 模型权重中固化的仍是训练期的旧值", "fix": "按版本更新外部文档与检索索引并使旧缓存失效; 代价是模型参数内部的旧知识仍原样保留" },
    { "name": "全局改参数", "pain": "一旦用户提问未走外部检索或检索未命中, 模型仍会直接输出参数中残留的旧答案", "fix": "demo 将权重矩阵 W 的全部元素统一加 6: 目标键输出升至 7.0, 但无关事实偏移达 6.0, 相近邻居偏移达 10.8" },
    { "name": "秩一编辑", "year": 2022, "pain": "无差别修改全局参数会导致正交的无关事实也被连带拉偏 6.0", "fix": "采用 ROME 式秩一外积更新 $\\Delta W = u v^\\top$, 仅沿目标键方向修正权重: 无关事实偏移降为 0.0, 但非正交的相近邻居仍偏移 4.8" },
    { "name": "约束邻域", "pain": "与目标键特征方向相近的邻居事实仍被连带拉偏 4.8", "fix": "求解时在协方差项中对邻居方向施加 100 倍保护权重 (加 $100 \\cdot n n^\\top$): 邻居偏移压至 0.0475; 代价是别名输出降至 6.51, 且目标与邻居完全重合时无解" },
    { "name": "删除对照重训", "pain": "参数编辑只能让目标键输出新值, 而隐私删除请求要求证明特定训练记录的参数影响已被消除", "fix": "从充分统计量中扣除待删记录贡献, 参数与独立重训误差小于 1e-12, 并联合报告遗忘集、保持集与成员推断指标" }
  ]
}
export default topic

mergePage(topic, 'finetune-dora', 'finetune-lora', '进阶 · DoRA: 权重的长度和方向分开学', ["FtDoraLab"])

topic.pages['finetune-lora'].title = '参数高效微调 · LoRA 与 DoRA'
topic.pages['finetune-lora'].subtitle = '冻结预训练基座、只在旁路更新两个极瘦的低秩矩阵 A 与 B, 大幅降低优化器显存与多任务落盘开销; 进阶小节进一步介绍 DoRA 如何拆解行权重的模长与方向独立优化。若走冲刺路线, 先掌握前半部分标准 LoRA 即可。'

for (const chapter of topic.chapters) chapter.label = chapterLabels[chapter.route] || chapter.label

canonicalizeRelated(topic)
