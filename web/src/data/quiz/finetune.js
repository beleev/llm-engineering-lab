// 阶段 4 · llm_finetune 自测题: 每章 3 题, 考取舍与"为什么", 错误选项是常见误解。
export default {
  finetune: [
    {
      q: 'SFT、LoRA、DPO 三者的关系, 哪种说法是对的?',
      options: ['三者互相替代: 数据少用 LoRA, 有偏好对用 DPO, 其余用 SFT', 'SFT/DPO 定"优化什么目标", LoRA 定"更新哪些参数", 可自由组合', 'LoRA 是一种专为小数据设计的正则 loss, 可叠加在 SFT 的 loss 上', 'DPO 要求 ref 与 policy 结构完全一致, 所以只能全参, 不能配 LoRA'],
      answer: 1,
      why: 'LoRA/QLoRA/DoRA 是参数化方式, SFT/DPO/GRPO 是目标函数; LoRA-SFT、LoRA-DPO 都很常见。',
    },
    {
      q: 'PPO 式 RLHF 要同时驻留 4 个模型, DPO 只要 2 个。被省掉的是哪两个?',
      options: ['reward model 和 critic', 'reference 和 critic', 'policy 和 reference', 'reward model 和 reference'],
      answer: 0,
      why: 'DPO 把奖励隐式写成 $\\beta \\cdot \\log(\\pi/\\pi_{\\text{ref}})$, 不需要显式 RM; 没有在线 RL 也就不需要 critic。reference 仍然保留。',
    },
    {
      q: '下面哪一项最适合用 GRPO 这类在线 RL, 而不是 SFT 或 DPO?',
      options: ['有大量人工写好的标准回答, 想让模型学会同样的格式', '有成对的 chosen / rejected 人工偏好标注, 但不能在线采样', '有一个强 teacher, 能给出每个位置的完整输出分布', '答案能被程序判对错 (数学、代码), 但没有标准解题过程'],
      answer: 3,
      why: '可验证奖励只告诉你"对不对", 不告诉你"怎么写"。让模型自己采样、用结果当信号, 正是在线 RL 的用武之地。',
    },
  ],
  'finetune-sft': [
    {
      q: 'labels = x[:, 1:] 已经左移一格, prompt 占 x 的前 P 个 token。正确的 prompt mask 是?',
      options: ['labels[:, :P] = -100', 'labels[:, :P+1] = -100', 'labels[:, :P-1] = -100', 'labels[:, 1:P] = -100'],
      answer: 2,
      why: 'labels[t] = x[t+1], 第一个 response token x[P] 落在 labels 的第 $P-1$ 格, 所以只能 mask 前 $P-1$ 格。',
    },
    {
      q: '如果错写成 labels[:, :P] = -100, 训练时最可能观察到什么?',
      options: ['第一步就报 shape 不匹配: labels 比 logits 少一格, CE 直接抛错', 'loss 照常下降, 但回复首 token 从未被监督; 单 token 回复时 loss 为 nan', 'loss 明显比正确写法高一截, 训练几十步就能从曲线上看出来', 'prompt 末尾 token 被当成监督目标, 模型学会复述 prompt 的结尾'],
      answer: 1,
      why: '多 mask 一格只是少了一个监督位置, 曲线看不出异常。只有当 response 只有 1 个 token 时, 标签才会全部被 ignore, loss 得到 nan。',
    },
    {
      q: '为什么 SFT 不让 prompt token 贡献 loss?',
      options: ['要教的是"看到指令怎么回答", 在 prompt 上算 loss 会把梯度花在学提问上', '省显存和计算: 被 mask 的位置不做前向, 长 prompt 时能省下一大半激活', '防止泄题: 不 mask 的话训练时模型能看到答案, 推理时就学不会回答', 'prompt 在预训练里见过, 它的 loss 本来就接近 0, 算进来只会把平均 loss 拉低'],
      answer: 0,
      why: 'mask 不省显存也不省计算 (前向照样要算), 它只是把监督信号集中到 response 上。',
    },
  ],
  'finetune-lora': [
    {
      q: 'LoRA 敢用 $r = 8$ 去替代 $d = 4096$ 的全量更新, 依据是什么?',
      options: ['任何矩阵都能被秩 8 的矩阵足够精确地近似', '预训练权重 $W$ 本身就是低秩的, 秩 8 足以装下它', '低秩矩阵的梯度方差更小, 同步数下能追平全参', '微调带来的权重变化 $\\Delta W$ 本身内在秩就很低'],
      answer: 3,
      why: '低秩的是"变化量"而不是 $W$ 本身。如果目标任务离预训练分布很远, $\\Delta W$ 不再低秩, LoRA 就会明显落后于全参。',
    },
    {
      q: '$B$ 初始化为 0、$A$ 随机初始化。如果反过来两个都随机初始化会怎样?',
      options: ['梯度全为 0, 训练根本启动不了', '没区别: 只要 lr 够小, 随机的 $BA$ 很快被训练抵消', '第 0 步 $BA \\neq 0$, 输出立刻偏离基座, 等于先破坏再修复', '起点不变, 只是 merge 时要给 $BA$ 乘一个修正系数'],
      answer: 2,
      why: '$B = 0$ 保证起点与原模型完全一致; 同时 $A$ 非零让 $B$ 的梯度不为零, 训练能正常启动。两个都为 0 则梯度全为 0。',
    },
    {
      q: '同一个基座、同一份数据、同样 300 步, 本仓库实测全参留出集 EM 0.809, LoRA(r=8) 只有 0.352。这说明 LoRA 买到的是什么?',
      options: ['更快的收敛: 参数少, 同样的 loss 用更少步数就能达到', '更好的泛化: 低秩约束起正则作用, 留出集上不易过拟合', '显存、存储和多租户可插拔; 上限不高于全参, 步数也不省', '更高的上限: 300 步还没训完, 步数够了会反超全参'],
      answer: 2,
      why: '"LoRA 收敛更快"这个流行说法, 多半来自病态初始化 (初始 loss ≈ 250) + 只背几条样本 + 10× 学习率。初始化正常、改看留出集之后, 同步数下全参更快也更好。',
    },
  ],
  'finetune-dpo': [
    {
      q: 'DPO 训练第 0 步 (policy 刚从 ref 复制), loss 应该是多少?',
      options: ['$\\ln 2 \\approx 0.693$', '0, 因为 policy = ref', '约 $\\beta \\ln 2$, 随 β 变', '$\\ln V$ (V 是词表大小)'],
      answer: 0,
      why: 'policy = ref 时隐式奖励差 $\\Delta = 0$, $-\\log \\sigma(0) = \\ln 2$, 与 $\\beta$ 无关。初始 loss 不是 0.693 通常意味着 ref/policy 没对齐或 mask 有 bug。',
    },
    {
      q: '本仓库 200 步 DPO: 留出集偏好准确率 0.965 → 0.996, $\\log \\pi(\\text{chosen})$ −4.03 → −4.18, 贪心 EM 0.332 → 0.137。怎么解释?',
      options: ['评估脚本有 bug: 准确率 0.996 与 EM 0.137 不可能同时成立', 'loss 只看两者之差: 两边一起降、rejected 降得更快, loss 照样变小', '步数不够: 200 步只拉开了 margin, 再训 log π(chosen) 就会回升', 'ref 误进了 optimizer 跟着漂, 隐式奖励失去参照, 生成随之退化'],
      answer: 1,
      why: 'likelihood displacement: 优化目标里没有任何一项要求 chosen 自己的概率上升。所以盯 margin 之外还要盯 $\\log \\pi(\\text{chosen})$; ORPO 的 NLL 项正是为此而设。',
    },
    {
      q: '把 $\\beta$ 调大, 效果是?',
      options: ['每对样本推得更用力, policy 被允许离 ref 更远', '等价于把学习率等比放大, 终点不变只是更快', '同样的 $\\Delta$ 更快饱和, policy 被约束得更贴近 ref', '$\\sigma$ 更晚饱和, 要把 $\\Delta$ 拉得更大 loss 才停手'],
      answer: 2,
      why: '$\\beta$ 对应 RLHF 目标里的 KL 惩罚系数: $\\beta$ 大 → 小小的 $\\Delta$ 就让 loss 饱和、停止推动; $\\beta$ 小 → 要把 $\\Delta$ 拉得很大才停。',
    },
    {
      q: 'KTO 的数据里好:坏 = 1:9, $\\lambda_D = 1$。按论文建议 $\\lambda_U$ 取多少? 不调会怎样?',
      options: ['9; 坏样本多, 权重也该跟着放大', '1; KTO 按样本平均, 比例失衡不影响方向', '1/9; 不调则偏好方向丢失、生成崩溃', '1/9; 不调只是收敛慢些, 终点差不多'],
      answer: 2,
      why: '让 $\\lambda_D n_D \\approx \\lambda_U n_U$。调了准确率 0.988; 不调只有 0.492, EM 0.000。不调时坏样本的梯度压倒一切, 模型把所有回复一起往下压。log π(chosen) 掉到 −30.02, 偏好方向丢了, 生成全崩。',
    },
    {
      q: '同样 200 步, KTO 脚本里同条件重跑的 DPO 对照, log π(chosen) 从 −4.03 掉到 −4.25, KTO (1:1) 却升到 −3.73。原因是?',
      options: ['好样本有自己的一项, 直接推高 $r$', 'KTO 不用 ref, 没有 KL 拖住 chosen', 'KTO 的 β 更小, policy 离 ref 更远', '参考点 $z_0 > 0$, 把好样本往上拉'],
      answer: 0,
      why: 'DPO 只看 chosen 与 rejected 的差, 两边可以一起降。KTO 的好样本要高过参考点, 自己就有向上的力。本例 $z_0$ 全程为 0, 所以不是 $z_0$ 的功劳。单独跑 train_dpo 得到的是 −4.18: 两次独立运行, 结论相同。',
    },
    {
      q: 'KTO 的参考点 $z_0$ 用什么估?',
      options: ['本条样本自己的 log-ratio, 随 $r$ 一起更新', '错配的 $(x_i, y_{i+1})$ 上的平均 log-ratio', 'ref 模型在同批样本上的平均 log-likelihood', '不估计, 作为超参固定取 0'],
      answer: 1,
      why: '错配估计再 clamp 到 ≥ 0, 且不回传梯度。用本条样本估的话, $z_0$ 随 $r$ 一起动, 参考点失去意义。本例错配估计从 0 降到 −2.52, clamp 后 $z_0$ 恒为 0: 参考点实际没发挥作用。',
    },
  ],
  'finetune-rlhf': [
    {
      q: '偏好标注有 $\\varepsilon = 20\\%$ 的概率标反。训练充分的 Bradley–Terry RM 在这类样本对上学到的分差趋近于?',
      options: ['$+\\infty$, 训得越久越大', '$\\ln(0.8/0.2) \\approx 1.39$', '$1 - 2\\varepsilon = 0.6$', '0.8, 即标对的概率'],
      answer: 1,
      why: '期望 loss 在 $\\sigma(\\Delta) = 1-\\varepsilon$ 处梯度为零, 即 $\\Delta^* = \\ln((1-\\varepsilon)/\\varepsilon)$。无噪声时才会趋于无穷, 纯噪声 ($\\varepsilon = 0.5$) 时为 0。',
    },
    {
      q: 'GRPO 去掉了 PPO 的 critic, baseline 从哪来?',
      options: ['同一 prompt 采 G 条, 用组内平均奖励', 'reward model 对 prompt 本身打的分', '上一步整批奖励的滑动平均', '不需要 baseline, 奖励直接当优势'],
      answer: 0,
      why: '组内均值是对"这道题的期望得分"的蒙特卡洛估计, 正是 value 网络想预测的东西。用采样换掉了一个模型。',
    },
    {
      q: '蒸馏 loss 的软标签项为什么要乘 $T^2$?',
      options: ['让软标签项的 loss 与硬标签项同量级, 便于画在一张图上', 'KL 本身与 $T^2$ 成正比, 乘上它让高温时蒸馏信号更强', '放大回传给 teacher 的梯度, 让 teacher 跟着一起微调', '软标签项的梯度按 $1/T^2$ 衰减, 不补偿则升温等于关掉蒸馏'],
      answer: 3,
      why: '梯度 $= (1/T)(q^T - p^T)$, 而 $q^T - p^T$ 在高温下又 $\\propto 1/T$; 乘 $T^2$ 后软硬两项的相对权重才只由 $\\alpha$ 决定。',
    },
  ],
  'finetune-merge': [
    {
      q: 'Task Arithmetic 取 $\\lambda = 0.5$, 合并两个任务向量, 结果等于?',
      options: ['两个微调模型的权重平均', '基座加上两个任务向量的和', 'SLERP $t=0.5$ 的球面插值', '基座加两个任务向量均值的一半'],
      answer: 0,
      why: '$\\theta_0 + 0.5(\\tau_A + \\tau_B) = (\\theta_A + \\theta_B)/2$。每个任务向量只加了一半, 实测两段全对 EM 只有 0.039; $\\lambda = 1$ 时是 1.000。SLERP $t=0.5$ 还要再乘 $1/\\cos(\\Omega/2)$ (本例 1.014), 不等于简单平均。',
    },
    {
      q: 'DARE 以概率 $p$ 丢弃任务向量的坐标后, 为什么要把剩下的除以 $(1-p)$?',
      options: ['让 $\\tilde\\tau$ 的范数与 $\\tau$ 相等, 否则合并后任务效果整体变弱', '让 $\\tilde\\tau$ 的方差与 $\\tau$ 相同, 单次随机丢弃的结果才稳定', '保持 $\\mathbb{E}[\\tilde\\tau] = \\tau$; 不除相当于 $\\lambda$ 缩成 $(1-p)$ 倍', '把小坐标推到阈值以下, 让合并结果更稀疏、冲突更少'],
      answer: 2,
      why: '丢弃是随机的, 放大是为了期望不变。但单次实现仍有方差: 本例 p=0.9 时两段全对 EM 掉到 0.066。',
    },
    {
      q: '本例 $\\cos(\\tau_A, \\tau_B) = +0.046$。能说明两个任务向量没有冲突吗?',
      options: ['能: 正交就不冲突', '不能: 逐坐标符号冲突仍有 47.4%', '能: cos > 0 说明方向一致', '不能: cos 要接近 −1 才无冲突'],
      answer: 1,
      why: '整体正交 ≠ 逐坐标无冲突。随机符号的冲突率就是 50%。cos 是对所有坐标求和后的结论, 正负可以互相抵消。TIES 的 "选符号" 就是冲着逐坐标冲突去的。',
    },
    {
      q: '各方法在验证集上挑过 λ 后, 留出集两段全对 EM: Task Arithmetic 1.000, DARE p=0.5 0.875, TIES d=0.5 0.734。TIES 为什么没赢?',
      options: ['$\\cos(\\tau_A, \\tau_B) > 0$, 本来就没有符号冲突可修', 'TIES 没调 λ, 固定用 0.5, 每个 τ 只加了一半', '选符号时 B 总输给 A, B 段的更新被整段丢掉', 'τ 只有约 10 万参数, 冗余少, 修剪一半就伤到有用坐标'],
      answer: 3,
      why: 'TIES 要修的是大量小幅、冗余的坐标互相冲突。这里 τ 是约 10 万参数的全参更新, 修剪掉一半就伤到了有用的坐标。\n- 符号冲突有 47.4%, 并非没有。\n- TIES 挑到的 λ 是 1.5。\n- A、B 段准确率 0.947 / 0.948, 两段都在。\nSLERP 0.090。换两个种子仍是 Task Arithmetic 最高。TIES/DARE 的收益依赖大模型任务向量的冗余, 玩具规模上看不到。',
    },
  ],
  'finetune-rlaif': [
    {
      q: '宪法条文写着 "变体 14 也算脏话", 但 judge 规则只查 15。训练后变体 14 只从 0.418 降到 0.320, 为什么?',
      options: ['14 在采样里出现得太少, 凑不出足够多的偏好对来压它', 'judge 不查 14, 它在 chosen 和 rejected 里都出现', 'β 太小, DPO 推不动 14 这种不在规则里的低频 token', '只训了 150 步, 再多训几百步 14 也会自然降到 0'],
      answer: 1,
      why: '训练信号只来自 judge 检查了的东西。降的那点是 DPO 副作用 (首个回复 token 被带偏, 偶尔把 14 也打码), 不是学会了原则。',
    },
    {
      q: '加了一条 "不啰嗦" 原则后, 改写总让回答变短。DPO 学到了什么?',
      options: ['学会了不重复, 回答仍然完整', '什么也没学到, 违规率基本没变化', '回答变长: 改写补回了被删的内容', '早点输出 EOS, 回答越来越短'],
      answer: 3,
      why: '实测违规 0.605 → 0.000, 理想回复 0.395 → 0.000, 只说 1 个 token 就结束。chosen 总比 rejected 短, "短" 本身就是最容易学的区分特征。judge 造出的偏好对带着系统性偏差, DPO 学的是偏差, 不是原则。',
    },
    {
      q: 'RLAIF 相比 RLHF 省掉了什么?',
      options: ['逐条人工标注的成本', '偏好优化这一步', 'reference model', '定义 "什么是好回答" 的责任'],
      answer: 0,
      why: '标签成本从 "每条一个人" 降到 "每条一次 judge 调用"。但 judge 漏检什么, 模型就学不到什么。',
    },
    {
      q: '只看违规率 0.648 → 0.031, 还缺哪个数才能判断对齐成功?',
      options: ['训练 loss 的下降幅度', '有用性: 该照抄的位置还照不照抄', '偏好对的数量 (2539)', 'judge 批评了多少条'],
      answer: 1,
      why: '什么都不说也能 0 违规。实测非敏感位置照抄率 1.000 → 0.856, 理想回复 0.352 → 0.488。lr 提到 3e-4 时违规同样降到 0, 理想回复却掉到 0.051。',
    },
  ],
  'finetune-prm': [
    {
      q: '同样 N 条 K 步的解, ORM 与 PRM 各拿到多少个标签?',
      options: ['N / N', 'N / NK', 'NK / N', 'NK / NK'],
      answer: 1,
      why: 'ORM 每条解只有 "最终答案对不对" 1 个标签; PRM 每步 1 个。标签稠密 K 倍, 每个只要求局部判断。',
    },
    {
      q: 'PRM 给整条解打分时, 各步概率该怎么汇总?',
      options: ['取平均', '取最后一步', '取最小值', '取最大值'],
      answer: 2,
      why: '一步错整条错。取平均会被其余正确的步稀释。取 min (或乘积) 才是短板。',
    },
    {
      q: '训练 600 步后 ORM 的 BCE 是 0.682 (ln 2 = 0.693)。最合理的解释是?',
      options: ['ORM 的上限就这么低, 永远学不会', 'ORM 与 PRM 初始化不同, 起点就差一截', 'lr 太大, loss 在 ln 2 附近来回震荡', 'ORM 学得慢: 同配置约 2000 步才开始降'],
      answer: 3,
      why: '0.682 ≈ ln 2 说明它几乎还在猜。ORM 要从 1 个 0/1 里自己做信用分配, 信号稀疏。训更久它也会开始下降 (3000 步 ≈ 0.55)。',
    },
    {
      q: '第 2 步写错了, 第 3 步照着写错的值算对了。按 ArithChain 的定义, 第 3 步的标签是?',
      options: ['对: 只判这一行, $v_{i-1}$ 取解里写的值', '错: 前面错了, 这一步结果必然也错', '看最终答案: 答案对就算对, 否则算错', '不打标签: 首个错步之后一律 mask'],
      answer: 0,
      why: '像批改真人草稿: 每步只判局部。这样 PRM 最低分的那一步才能指向第一个错步 (实测定位准确率 0.575, 常数猜 0.341)。',
    },
  ],
  'finetune-ppo': [
    {
      q: '同样每步 256 条回复, PPO 的留出集 pass@1 比 GRPO 高 0.09。主要原因是?',
      options: ['critic 降低了优势估计的方差', 'KL 写进奖励, 由 GAE 逐 token 分配', '每题只采 1 条, 题数多 8 倍', 'clip 区间更紧, 更新更稳'],
      answer: 2,
      why: 'critic 冻结 (lr=0) 的对照一样是 0.389。差别来自采样方式: 256 道不同的题各采 1 条。GRPO 的 32 道题里约 1/3 组全对或全错, $A\\equiv 0$, 白占预算。',
    },
    {
      q: 'GAE 取 $\\lambda = 1$ 时, 回报目标 $G_t = A_t + V(s_t)$ 退化成什么?',
      options: ['TD(0) 目标 $r_t + \\gamma V(s_{t+1})$, 只信下一步', 'Monte-Carlo 回报 $\\sum_k \\gamma^k r_{t+k}$', '组内均值, 与 GRPO 的 baseline 相同', '$V(s_t)$ 本身, 优势恒为 0'],
      answer: 1,
      why: '$\\lambda = 1$ 时 $V$ 的项逐项相消, 只剩真实回报: 无偏, 方差大。$\\lambda = 0$ 时 $A_t = \\delta_t$, 只信下一步的 $V$。脚本单测两端误差 0 与 4.8e-7。',
    },
    {
      q: 'critic 的 $V(s_0)$ 只解释了 2.5% 的回报方差, 组均值解释 36.5%。这说明?',
      options: ['解释方差越低越好: 说明 baseline 没有吃掉策略梯度的信号', '两者估的不是同一个量, 这两个数不能直接比', 'critic 只是没训够, 再多训会超过组均值', '这个规模下 critic 近乎常数 baseline, 却要多养一个网络'],
      answer: 3,
      why: 'critic 要判断 "这个前缀还能不能全对", 和 policy 学排序一样难。组均值不用学。代价是要训参数 ×2、常驻 ×3、耗时约 ×2。',
    },
    {
      q: 'GRPO 如果每题只采 1 条 (G=1), 会怎样?',
      options: ['组均值就是样本自己, $A\\equiv 0$, 学不到东西', '等价于一题一采的 PPO, 效果与 PPO 相同', '方差更小: 组内样本不再互相干扰', 'σ 为 0, 除法报错, 必须改用 critic'],
      answer: 0,
      why: '组内相对优势需要组。PPO 的 baseline 来自 critic (或批内白化), 与组无关, 所以一题一采也有信号。',
    },
  ],
  'finetune-runs': [
    {
      q: '一个 LoRA 实验只保存了 merge 后的完整权重, 没存 adapter。丢掉了什么?',
      options: ['精度: merge 时浮点舍入会吃掉一部分 LoRA 更新', '推理速度: merge 后每层要多算一次 $BA$', '多 adapter 共享基座热切换、只分发小文件的能力', '什么也没丢: 完整权重可直接当 adapter 按请求加载'],
      answer: 2,
      why: 'adapter 只有 $r \\cdot (d_{\\text{in}}+d_{\\text{out}})$ 个参数, 可以独立分发、按请求加载; 只留完整权重就退化成每个任务一份大模型。',
    },
    {
      q: '检查 PEFT 训练脚本"真的只训了 adapter", 最直接的办法是?',
      options: ['看 loss: base 没冻住的话 loss 会明显震荡', '统计 requires_grad 参数占比, 并核对 optimizer 只拿到它们', '看显存: 只训 adapter 时显存应降到全参训练的零头', '看 checkpoint: 只有几十 KiB 就说明只训了 adapter'],
      answer: 1,
      why: 'loss 下降和显存都不能证明 base 被冻结; 参数统计 (本仓库的 print_trainable_parameters) 才是直接证据。',
    },
    {
      q: '手上只有 (问, 答), 显存只够常驻一份权重加上 adapter 的优化器状态。下面哪一组还能用?',
      options: ['DPO + LoRA', 'GRPO 系 (β=0, 不带 ref)', 'LoRA / QLoRA / DoRA', 'SimPO / ORPO (无 ref)'],
      answer: 2,
      why: '- DPO: 要常驻 policy + ref 两份 (778 KiB, 加全参 Adam 合计 1556 KiB)。\n- GRPO: 还得有 verifier, 并能在线采样。\n- SimPO / ORPO: 要的是成对偏好。\n只有 (问, 答) 且显存紧, 剩下的就是 PEFT 三兄弟。',
    },
    {
      q: '线上日志只有单条回复的 👍 / 👎, 同一个 prompt 从没出现过两条回复。下面哪个能直接拿来训?',
      options: ['DPO: 把 👍 当 chosen, 另找一条 👎 配成一对', 'KTO: 每条样本和参考点 $z_0$ 比, 好的推高、坏的压低', 'PRM: 把 👍 / 👎 当成这条回复每一步的对错标签', 'SimPO: 去掉 ref 之后就不再需要成对的数据'],
      answer: 1,
      why: '- DPO / SimPO: 都比同一个 prompt 下的两条回复, 跨 prompt 配对没有意义。\n- PRM: 要的是逐步标签, 一个 👍 说不出哪一步对。\n- KTO: 代价结构同 DPO (policy + ref, 每步 2 次前向), 一半前向花在估 $z_0$ 的错配样本上。',
    },
    {
      q: '选型表里「模型合并」的前向次数记 0, 显存记 "—"。它的代价实际花在哪?',
      options: ['合并时要对 $\\tau$ 求梯度, 只是不存 Adam 状态', '要常驻一份 ref, 防止合并结果漂离基座', '同一基座上先微调出几个模型, 代价在那几次微调', '不用微调模型: $\\tau$ 能直接从基座权重里算出'],
      answer: 2,
      why: '$\\tau = \\theta_{\\text{ft}} - \\theta_0$, 先得有微调好的 $\\theta_{\\text{ft}}$。本例是两次各 300 步的全参微调。合并本身零梯度、零数据, 只在 state_dict 上逐张量加减。',
    },
  ],
  'finetune-qlora': [
    {
      q: 'NF4 的 16 个码点为什么按正态分位数摆, 而不是等间距?',
      options: ['按分位数摆就不用再给每个 block 存 absmax scale', '分位数码点在两端更密, 专门用来保护少数 outlier 权重', '为了和 INT4 kernel 共用同一套整数矩阵乘', '权重近似正态, 按分位数摆让每个码被用到的概率接近相等'],
      answer: 3,
      why: '码点密度跟着数据密度走; 等间距码本把好几个码浪费在几乎没有权重的两端。',
    },
    {
      q: '某个 block 里出现一个很大的 outlier, 会发生什么?',
      options: ['只有 outlier 被截断到最大码点, 其余权重不受影响', '整层共用一个 scale, 整层权重的误差都跟着变大', '该 block 的 absmax 被撑大, 其余权重挤进少数几个码点', 'outlier 超出 [-1, 1] 无法编码, 量化这一步直接报错'],
      answer: 2,
      why: 'scale 是按 block 取的 absmax, 所以伤害范围恰好是一个 block。这也是 block 不能取太大的原因。',
    },
    {
      q: 'block 大小从 64 缩到 16, 代价是什么?',
      options: ['精度下降: block 越小, absmax 越容易被噪声主导', 'scale 开销变大: 每参数从 4.5 bit 涨到 6 bit', 'scale 开销变大: 每参数从 4.5 bit 涨到 8 bit', '反量化要查 4 倍大的码本, 训练慢 4 倍'],
      answer: 1,
      why: '$4 + 32/B$: $B = 64$ → 4.5 bit, $B = 16$ → 6 bit。QLoRA 的双重量化就是把 scale 再量化一次来压这部分开销。',
    },
  ],
  'finetune-dora': [
    {
      q: 'DoRA 的权重参数化是?',
      options: ['$W = m \\cdot (W_0 + BA) / \\|W_0 + BA\\|$, $m$ 每行一个', '$W = W_0 + m \\cdot BA$, $m$ 为每行一个可训标量', '$W = m \\cdot W_0 / \\|W_0\\| + BA$, $m$ 每行一个', '$W = (W_0 + BA) / m$, $m$ 为每层一个可训标量'],
      answer: 0,
      why: '低秩更新只决定方向, 归一化后乘上单独学习的幅度 $m$; 初始 $m = \\|W_0\\|$、$B = 0$, 起点与原模型一致。',
    },
    {
      q: 'DoRA 论文指出 LoRA 与全参微调在学习行为上的关键差别是?',
      options: ['LoRA 只能改幅度、改不了方向, 全参两者都能改', '全参只转方向不改幅度, LoRA 两者都在改', 'LoRA 的 lr 必须比全参小一个量级才稳定', 'LoRA 的幅度与方向变化被绑在一起, 全参中两者近乎独立'],
      answer: 3,
      why: '想"只转方向不改长度", 加性更新必须精确落在一个点上。解耦后低秩容量可以专心用于方向。',
    },
    {
      q: '相对同秩 LoRA, DoRA 的额外代价是?',
      options: ['参数量翻倍: $m$ 与 $BA$ 同样大, 落盘时要单独存一份', '推理时每层多一次归一化, 因为 $m$ 没法写回 $W$ 里', '每层多 d 个幅度参数、训练时多算一次范数; merge 后推理无差别', '要常驻一份 reference 模型来锚定幅度 $m$'],
      answer: 2,
      why: '幅度向量只有 $d_{\\text{out}}$ 个元素 (每行一个), 相对 $2dr$ 可以忽略; $m \\cdot V/\\|V\\|$ 算出来就是一个普通矩阵, 可以写回 $W$。',
    },
  ],
  'finetune-simpo-orpo': [
    {
      q: '去掉 reference、又不做长度归一化, 直接用 $\\sum \\log \\pi(\\text{chosen}) - \\sum \\log \\pi(\\text{rejected})$ 当 margin, 会出什么问题?',
      options: ['$\\sum \\log p$ 随长度变大, 长样本梯度爆炸', 'rejected 更长就白得 margin, 学到"短的好"', 'chosen 更长就白得 margin, 学到"越长越好"', '没问题: 这就是 SimPO, $\\gamma$ 只是可选项'],
      answer: 1,
      why: 'DPO 里 ref 对同一条长序列也给出同样低的 $\\sum \\log p$, 相减后抵消; 拿掉 ref 必须另想办法消除长度红利。',
    },
    {
      q: 'SimPO 里的 $\\gamma$ 起什么作用?',
      options: ['目标间隔: chosen 平均 $\\log p$ 须领先至少 $\\gamma/\\beta$', '长度惩罚: 每多一个 token 从奖励里扣掉 $\\gamma$', 'KL 系数: 代替 ref 约束 policy 别离 SFT 太远', '温度: 平均 $\\log p$ 先除以 $\\gamma$ 再进 sigmoid'],
      answer: 0,
      why: '没有 ref 提供"从哪出发"的参照, $\\gamma$ 明确规定要拉开多大差距才算够, 类似 SVM 的 margin。',
    },
    {
      q: 'ORPO 为什么可以直接从 base 模型起训, 不需要先做 SFT?',
      options: ['odds ratio 项把 chosen 概率推向 1, 等效 SFT', '学习率更大, 一步跨过 SFT 阶段', '内置隐式 reward model, 不需要示范数据', 'loss 里自带 chosen 的 NLL 项, 即 SFT'],
      answer: 3,
      why: '$\\mathrm{loss} = \\mathrm{NLL}(\\text{chosen}) + \\lambda \\cdot (-\\log \\sigma(\\log \\text{odds ratio}))$: 前一项就是 SFT, 同时充当防漂移的锚; 后一项压低 rejected。',
    },
  ],
  'finetune-grpo-variants': [
    {
      q: 'clip 的上界 $1+\\varepsilon$ 实际上约束的是哪类 token?',
      options: ['所有 token 一视同仁, 按 $\\rho$ 统一截断', '高概率 token: 它们梯度最大, $\\rho$ 每轮涨得最快', '低概率 token: $\\rho \\le 1/\\pi_{\\text{old}}$, 高概率的碰不到上界', '只约束 $\\hat A < 0$ 的 token, 正优势不受上界影响'],
      answer: 2,
      why: '- $\\pi_{\\text{old}} = 0.9$: $\\rho$ 最大 $1.11 < 1.2$。\n- $\\pi_{\\text{old}} = 0.01$: 每轮最多涨到 0.012。\n所以 DAPO 的 clip-higher 只放宽上界。\n另外: 一批样本只更新一次时 $\\rho \\equiv 1$ (实测第 1 个 epoch $|\\rho-1| = 0.000$), clip 根本没上场。',
    },
    {
      q: 'GRPO 的 loss 先对每条回答内部按长度平均 ($1/|o_i|$)。它带来的偏置是?',
      options: ['又长又对的回答每个 token 奖励更多, 鼓励越写越长', '又长又错的回答每 token 罚得更轻, 鼓励错就写长', '没有偏置: 按长度平均正好让长短回答一样重', '长回答的奖励被放大 $|o_i|$ 倍, 模型越写越长'],
      answer: 1,
      why: '同样的负优势被摊到更多 token 上: 实测 3-token 回复的单 token 权重是 9-token 的 3.0 倍。Dr.GRPO 用常数分母, DAPO 用 token 级平均, 两者都把这个比值压回 1.0。',
    },
    {
      q: '一个 prompt 的 G 条回答全部正确。对这一组, 哪种处理是 DAPO 的做法?',
      options: ['丢弃该组继续采样, 直到每组都有对有错', '给全组正优势, 强化这道已会的题', '给 $\\sigma$ 加一个大 $\\varepsilon$ 后照常训练', '只保留最短的一条, 奖励简洁的回答'],
      answer: 0,
      why: '零方差的组不提供任何学习信号, 却占着 batch 名额。实测这类 prompt 占 32%。\n开了动态采样后, 进 loss 的 $A=0$ 样本从 32.1% 降到 0.0%。\n- 本仓库: 只过滤不补采, batch 会变小。\n- DAPO 原版: 继续采到凑满。',
    },
  ],
  'finetune-onpolicy-distill': [
    {
      q: '单峰的学生去拟合双峰的老师。最小化 forward $\\mathrm{KL}(p_{\\text{teacher}} \\,\\|\\, q_{\\text{student}})$ 会得到?',
      options: ['缩进其中一个峰, 另一个峰完全放弃', '与老师完全一致, KL 能降到 0', '两峰梯度互相抵消, 退化成均匀分布', '摊开盖住两个峰, 连峰间低谷也放上概率'],
      answer: 3,
      why: 'forward KL 在老师的样本上求期望。老师有质量而学生没有的地方, 惩罚趋于无穷, 所以必须全覆盖 (mode-covering)。',
    },
    {
      q: 'on-policy 蒸馏相对离线蒸馏 (在老师写的文本上训练), 解决的核心问题是?',
      options: ['老师太大: on-policy 训练时不用老师常驻显存', '词表不一致: on-policy 能在不同词表间蒸馏', '分布错配: 离线只见老师的前缀, 推理时却走自己的前缀', 'KL 太慢: on-policy 对 KL 采样估计, 不遍历词表'],
      answer: 2,
      why: '学生自己采样、老师在学生到达的状态上给分布, 训练分布 = 推理分布, 这正是 on-policy 的含义。',
    },
    {
      q: '相对 GRPO 这类 RL, on-policy 蒸馏的监督信号有什么不同?',
      options: ['更稀疏: 只在 EOS 处比较学生和老师', '每个 token 有老师的完整分布; RL 每条序列一个标量', '没区别: 都是每条序列一个 KL 或奖励标量', '要用 REINFORCE 估 KL 的梯度, 方差比 RL 还大'],
      answer: 1,
      why: '同样是 on-policy 采样, 蒸馏把 "整条序列一个分" 换成 "每个位置一个分布"。样本效率高得多, 前提是有一个好老师。\n- 实测样本合格率: 离线 0.059 → on-policy 0.402。\n- 代价: 老师的少数派答法被压到 $\\log \\pi = -33$。',
    },
  ],
  'finetune-rlvr': [
    {
      q: '奖励函数是"completion 中落在词表后半区的 token 比例"。训练后 reward 达到 1.0, 能得出什么结论?',
      options: ['管线能跑通; 但常数输出就能满分, 证明不了会读题', '模型学会了理解 prompt: 满分说明每题都答对了', '奖励设计得好: 曲线平滑升到 1.0, 没被 hack', 'KL 系数太小, policy 过拟合了训练集的 prompt'],
      answer: 0,
      why: '奖励不依赖 prompt 时, 常数策略就能拿满分, RL 会沿最短路径塌缩到它。',
    },
    {
      q: '"常数基线"指的是什么, 为什么训练前要先算它?',
      options: ['学习率恒定时 loss 的平稳值, 用来判断何时停训', 'ref 模型在留出集上的平均奖励, 用来衡量 RL 带来的提升', '随机均匀输出时的平均奖励, 即 1/类别数的那条线', '所有 prompt 输出同一 completion 时的最高平均奖励'],
      answer: 3,
      why: '常数基线 = 100% 的任务验证不了条件行为。看题任务里它通常只有 1/类别数, 超过它才说明模型在读 prompt。',
    },
    {
      q: '换成看题奖励后, 训练早期经常出现"整组 G 条全错"。这时 GRPO 的梯度是?',
      options: ['很大且为负: 全错时每条都被狠罚', '为负, 整组 token 概率被一起压低', '为零: 组内奖励相同 → 优势全为 0', '照常: 除以 σ 会把 0 放大成 ±1'],
      answer: 2,
      why: '组内相对优势只看差异; 全错与全对一样没有信号。对策是降低任务难度 / 增大 G / 动态采样 / 给部分分。',
    },
  ],
}
