# 章节预览

这些图片截自本仓库的教程站。点击章节打开交互页面；模型结构图位于阶段 2 的模型选择器中。

| 章节 | 截图 |
| --- | --- |
| [主线总览](https://beleev.github.io#/) | [查看截图](home.png) |
| [阶段 1.0 · 阶段总览](https://beleev.github.io#/basic) | [查看截图](basic.png) |
| [BPE 分词实验台 — 看着词表一条一条长出来 llm_basic/bpe.py](https://beleev.github.io#/basic/data) | [查看截图](basic-data-1.png) |
| [形状流水线 — 一批 token 从 ids [B,T] 走到 logits [B,T,V] 的每一步 llm_basic/model.py](https://beleev.github.io#/basic/forward) | [查看截图](basic-forward-1.png) |
| [反向传播 — 点一个节点, 看 loss 的梯度怎么一步步传回它 llm_basic/model.py](https://beleev.github.io#/basic/backward) | [查看截图](basic-backward-1.png) |
| [gradcheck 的 ε — 检查梯度时, 为什么步长不是越小越好 llm_basic/gradcheck.py](https://beleev.github.io#/basic/backward) | [查看截图](basic-backward-2.png) |
| [优化器轨迹 — 同一个学习率, SGD / 动量 / Adam 三种走法 llm_basic/optim.py](https://beleev.github.io#/basic/optim-sample) | [查看截图](basic-optim-sample-1.png) |
| [温度实验台 — 一根滑杆把同一组 logits 调尖或压平 llm_infer/m10](https://beleev.github.io#/basic/optim-sample) | [查看截图](basic-optim-sample-2.png) |
| [阶段 2.0 · 阶段总览](https://beleev.github.io#/models) | [查看截图](models.png) |
| [注意力掩码实验台 — 每个位置能看见哪些位置 llm_infer/m16](https://beleev.github.io#/attention) | [查看截图](attention-1.png) |
| [位置编码](https://beleev.github.io#/position) | [查看截图](position.png) |
| [残差流 — norm 挪一个位置, 几十层的网络才训得动 llm_models/layers/core/blocks.py](https://beleev.github.io#/blocks) | [查看截图](blocks-1.png) |
| [初始化 — 刚建好的模型, 第一步 loss 该是多少? llm_models/utils/init.py](https://beleev.github.io#/blocks) | [查看截图](blocks-2.png) |
| [MoE 路由](https://beleev.github.io#/moe) | [查看截图](moe.png) |
| [注意力掩码实验台 — 每个位置能看见哪些位置 llm_infer/m16](https://beleev.github.io#/models/swa-mtp) | [查看截图](models-mtp-1.png) |
| [MTP 多 token 预测 — 一次前向, 每个位置多预测几步 llm_models/mtp](https://beleev.github.io#/models/swa-mtp) | [查看截图](models-mtp-2.png) |
| [线性注意力 — 固定大小的状态矩阵 vs 越读越长的 KV cache llm_models/layers/sparse/linear_attention.py](https://beleev.github.io#/models/swa-mtp) | [查看截图](models-mtp-3.png) |
| [扩散生成](https://beleev.github.io#/diffusion) | [查看截图](diffusion.png) |
| [KV cache 与上下文上限 — 窗口一满, cache 作废, 之后每一步都整段重算 llm_models/utils/generation.py](https://beleev.github.io#/models/generation) | [查看截图](models-generation-1.png) |
| [QK-Norm — 把注意力的打分关进笼子里 llm_models/layers/core/attention.py](https://beleev.github.io#/models/qknorm-yarn) | [查看截图](models-qknorm-yarn-1.png) |
| [RoPE 长度外推 — 哪些频率该压缩, 哪些该放着不动 llm_models/layers/core/position_encoding.py](https://beleev.github.io#/models/qknorm-yarn) | [查看截图](models-qknorm-yarn-2.png) |
| [Mamba 选择性扫描 — 每个 token 自带的步长 Δ 决定它被记住还是被略过 llm_models/layers/sparse/ssm.py](https://beleev.github.io#/models/mamba) | [查看截图](models-mamba-1.png) |
| [Aux-loss-free 均衡 — 一个不收梯度的偏置, 只改「选谁」, 不改「信多少」 llm_models/models/moe/deepseekV3.py](https://beleev.github.io#/models/moe-balance) | [查看截图](models-moe-balance-1.png) |
| [DSA 稀疏注意力 — 先用便宜的 indexer 粗选 top-k 个 key, 再只对它们精算注意力 llm_models/layers/core/attention.py](https://beleev.github.io#/models/dsa) | [查看截图](models-dsa-1.png) |
| [滑窗与全注意力交替 — 远处的信息怎么一层一层接力传过来 llm_models/models/moe/gpt_oss.py](https://beleev.github.io#/models/gptoss) | [查看截图](models-gptoss-1.png) |
| [Attention sink — 让一个注意力头可以「谁都不看」 llm_models/layers/core/attention.py](https://beleev.github.io#/models/gptoss) | [查看截图](models-gptoss-2.png) |
| [LLaDA 扩散语言模型 — 一次预测全部空位, 只留有把握的, 没把握的遮回去再想 llm_models/models/language_models/llada.py](https://beleev.github.io#/models/llada) | [查看截图](models-llada-1.png) |
| [VAR 逐尺度生成 — 自回归的一步是「下一个分辨率」, 不是「下一个像素」 llm_models/models/generative/var.py](https://beleev.github.io#/models/var) | [查看截图](models-var-1.png) |
| [阶段 3.0 · 阶段总览](https://beleev.github.io#/train) | [查看截图](train.png) |
| [一个 batch 的三种切法 — 梯度累积、DDP 算出来的都是同一个梯度, 付的账不同 llm_train/m01+m02](https://beleev.github.io#/train/batch-ddp) | [查看截图](train-batch-ddp-1.png) |
| [流水线并行调度 — GPipe 和 1F1B 的时间表: 气泡在哪, 显存差在哪 llm_train/m04](https://beleev.github.io#/train/model-parallel) | [查看截图](train-model-parallel-1.png) |
| [张量并行 — 一层矩阵切给两张卡, 通信到底该插在哪? llm_train/m03](https://beleev.github.io#/train/model-parallel) | [查看截图](train-model-parallel-2.png) |
| [ZeRO 显存账 — 每参数 2 + 2 + 12 字节, 分三档切给多张卡, 先切哪一块? llm_train/m05](https://beleev.github.io#/train/memory) | [查看截图](train-memory-1.png) |
| [激活重算 — 只存段边界、反向时重跑, 段长 k 怎么选? llm_train/m07](https://beleev.github.io#/train/memory) | [查看截图](train-memory-2.png) |
| [浮点数轴 — 一批梯度落在哪个格式的范围里, loss scale 把它们挪进去 llm_train/m06](https://beleev.github.io#/train/precision-stability) | [查看截图](train-precision-stability-1.png) |
| [MoE 专家并行路由 — 路由一偏, 哪张卡最忙、哪些 token 被丢 llm_train/m11](https://beleev.github.io#/train/moe-seq-parallel) | [查看截图](train-moe-seq-1.png) |
| [Ring Attention — 序列切给 D 张卡, KV 块沿环传一圈, 每张卡都算出精确的注意力 llm_train/m12](https://beleev.github.io#/train/moe-seq-parallel) | [查看截图](train-moe-seq-2.png) |
| [Zigzag 序列切分 — 因果注意力分给多张卡时, 谁在等谁? llm_train/m12](https://beleev.github.io#/train/moe-seq-parallel) | [查看截图](train-moe-seq-3.png) |
| [交错 1F1B — 每张卡拿 v 个不相邻的小段, 把流水线气泡再除以 v llm_train/m04](https://beleev.github.io#/train/pipeline-schedules) | [查看截图](train-pipeline-schedules-1.png) |
| [学习率调度 Cosine vs WSD — 训到一半想加训, 要回退多少步重训? llm_train/m10](https://beleev.github.io#/train/lr-schedule) | [查看截图](train-lr-schedule-1.png) |
| [Block scaling — 低位宽量化时, 一个 outlier 会连累多少邻居? llm_train/m13 · m15](https://beleev.github.io#/train/low-precision) | [查看截图](train-low-precision-1.png) |
| [Muon 的正交化 — Newton–Schulz 迭代把梯度的奇异值谱拉平, 让每个方向迈一样大的步 llm_train/m14](https://beleev.github.io#/train/muon) | [查看截图](train-muon-1.png) |
| [Muon vs Adam — 同一个病态问题, 把病态方向转一下, 谁还修得动? llm_train/m14](https://beleev.github.io#/train/muon) | [查看截图](train-muon-2.png) |
| [Ulysses 序列并行 — 用 all-to-all 把「切序列」换成「切头」, 注意力核一行不改 llm_train/m16](https://beleev.github.io#/train/ulysses) | [查看截图](train-ulysses-1.png) |
| [MinHash-LSH — 近似去重时, 哪两篇文档会被拿来比较? llm_train/m17](https://beleev.github.io#/train/data-packing) | [查看截图](train-data-packing-1.png) |
| [数据源温度配比 — 把小语料抬上来, 要付多少重复的代价? llm_train/m17](https://beleev.github.io#/train/data-packing) | [查看截图](train-data-packing-2.png) |
| [Packing — 几篇文档拼进同一行训练, 它们会互相看见吗? llm_train/m18](https://beleev.github.io#/train/data-packing) | [查看截图](train-data-packing-3.png) |
| [IsoFLOP — 算力预算定死了, 模型该做多大? llm_train/m19](https://beleev.github.io#/train/scaling) | [查看截图](train-scaling-1.png) |
| [μP — 在小模型上调好的学习率, 能直接搬到大模型吗? llm_train/m20](https://beleev.github.io#/train/scaling) | [查看截图](train-scaling-2.png) |
| [pass@k 的两种算法 — 把 c/n 直接代进公式, 为什么平均总是算低? llm_train/m21](https://beleev.github.io#/train/eval) | [查看截图](train-eval-1.png) |
| [LLM 裁判的位置偏差 — 先出现的回答占便宜吗? 交换顺序能治吗? llm_train/m21](https://beleev.github.io#/train/eval) | [查看截图](train-eval-2.png) |
| [Ring all-reduce — 梯度沿环接力相加, 卡数翻倍, 每张卡发的字节数却几乎不变 llm_train/core](https://beleev.github.io#/train/collectives-loop) | [查看截图](train-collectives-loop-1.png) |
| [阶段 4.0 · 阶段总览](https://beleev.github.io#/finetune) | [查看截图](finetune.png) |
| [SFT 标签对齐 — mask 边界差一格, 第一个回答 token 就丢了 llm_finetune/data](https://beleev.github.io#/finetune/sft) | [查看截图](finetune-sft-1.png) |
| [LoRA — 微调要学的 ΔW 本来就低秩, 所以两个瘦矩阵 B·A 就能拟合它 llm_finetune/methods/lora.py](https://beleev.github.io#/finetune/lora) | [查看截图](finetune-lora-1.png) |
| [DPO — 一条 logsigmoid 曲线, 看哪些偏好对还在产生梯度 llm_finetune/methods/dpo.py](https://beleev.github.io#/finetune/dpo) | [查看截图](finetune-dpo-1.png) |
| [KTO — 单条 👍/👎 也能训, 但好坏 1:9 时 λ 必须跟着调 llm_finetune/methods/kto.py](https://beleev.github.io#/finetune/dpo) | [查看截图](finetune-dpo-2.png) |
| [GRPO — 同一道题采 G 条回复互相比较, 顶替 PPO 的 critic llm_finetune/methods/grpo.py](https://beleev.github.io#/finetune/rlhf-grpo-distill) | [查看截图](finetune-rlhf-1.png) |
| [温度实验台 — 一根滑杆把同一组 logits 调尖或压平 llm_infer/m10](https://beleev.github.io#/finetune/rlhf-grpo-distill) | [查看截图](finetune-rlhf-2.png) |
| [Bradley–Terry — 奖励模型学到的只是两条回答的「分差」 llm_finetune/methods/reward_model.py](https://beleev.github.io#/finetune/rlhf-grpo-distill) | [查看截图](finetune-rlhf-3.png) |
| [NF4 vs INT4 — 4 bit 只有 16 个码点, 该摆在哪 llm_finetune/methods/qlora.py](https://beleev.github.io#/finetune/qlora) | [查看截图](finetune-qlora-1.png) |
| [DoRA — 把权重向量拆成「长度」和「方向」, 各给一个旋钮 llm_finetune/methods/dora.py](https://beleev.github.io#/finetune/dora) | [查看截图](finetune-dora-1.png) |
| [模型合并 — 把两个任务向量相加、TIES 修剪、DARE 随机丢弃 llm_finetune/methods/merge.py](https://beleev.github.io#/finetune/merge) | [查看截图](finetune-merge-1.png) |
| [拿掉 reference 之后 — SimPO 与 ORPO 靠什么抵消回答长度 llm_finetune/methods/simpo.py · orpo.py](https://beleev.github.io#/finetune/simpo-orpo) | [查看截图](finetune-simpo-orpo-1.png) |
| [RLAIF — judge 没查的, DPO 学不到 llm_finetune/methods/rlaif.py](https://beleev.github.io#/finetune/rlaif) | [查看截图](finetune-rlaif-1.png) |
| [PRM vs ORM — 同一条解, 结果奖励看到 1 个标签, 过程奖励看到 4 个 llm_finetune/methods/prm.py](https://beleev.github.io#/finetune/prm) | [查看截图](finetune-prm-1.png) |
| [GAE — 优势估计从 TD(0) 滑到 Monte-Carlo, λ 就是那根滑杆 llm_finetune/methods/ppo.py](https://beleev.github.io#/finetune/ppo) | [查看截图](finetune-ppo-1.png) |
| [PPO vs GRPO — 同样 256 条回复, baseline 用组均值还是 critic llm_finetune/methods/ppo.py](https://beleev.github.io#/finetune/ppo) | [查看截图](finetune-ppo-2.png) |
| [ratio clip — 一批样本更新多轮时, 哪些 token 这一步已经不许再推 llm_finetune/methods/grpo.py](https://beleev.github.io#/finetune/grpo-variants) | [查看截图](finetune-grpo-variants-1.png) |
| [GRPO、Dr.GRPO、DAPO — 同一组回答, 每个 token 分到多少梯度 llm_finetune/methods/grpo.py](https://beleev.github.io#/finetune/grpo-variants) | [查看截图](finetune-grpo-variants-2.png) |
| [GSPO — 重要性比率该按 token 算, 还是按整条序列算 llm_finetune/methods/grpo.py](https://beleev.github.io#/finetune/grpo-variants) | [查看截图](finetune-grpo-variants-3.png) |
| [forward KL vs reverse KL — 容量不够的学生该盖住所有峰, 还是钻进一个峰 llm_finetune/methods/on_policy_distill.py](https://beleev.github.io#/finetune/onpolicy-distill) | [查看截图](finetune-onpolicy-distill-1.png) |
| [蒸馏为什么要乘 T² — 温度一高, 软标签的梯度就悄悄消失 llm_finetune/methods/distill.py](https://beleev.github.io#/finetune/onpolicy-distill) | [查看截图](finetune-onpolicy-distill-2.png) |
| [可验证奖励要「看题」— 否则 RL 只学会输出一个常数 llm_finetune/data/tasks.py](https://beleev.github.io#/finetune/rlvr) | [查看截图](finetune-rlvr-1.png) |
| [选型计算器 — 16 种微调方法, 各自要你付什么 llm_finetune/README.md](https://beleev.github.io#/finetune/runs) | [查看截图](finetune-runs-1.png) |
| [阶段 5.0 · 阶段总览](https://beleev.github.io#/infer) | [查看截图](infer.png) |
| [注意力掩码实验台 — 每个位置能看见哪些位置 llm_infer/m16](https://beleev.github.io#/infer/kv-memory) | [查看截图](infer-kv-memory-1.png) |
| [KV cache — 没有它, 每生成一个 token 都要把前面的全部重算一遍 llm_infer/m01](https://beleev.github.io#/infer/kv-memory) | [查看截图](infer-kv-memory-2.png) |
| [PagedAttention — 把 KV 显存切成 block 按需分配, 看浪费少了多少 llm_infer/m02](https://beleev.github.io#/infer/kv-memory) | [查看截图](infer-kv-memory-3.png) |
| [连续批处理 — 静态 batch 整批进出, 连续批每一步重组 llm_infer/m03](https://beleev.github.io#/infer/scheduler) | [查看截图](infer-scheduler-1.png) |
| [Chunked prefill — 长 prompt 切块混进 decode, 用 token 预算封顶卡顿 llm_infer/m06](https://beleev.github.io#/infer/scheduler) | [查看截图](infer-scheduler-2.png) |
| [P/D 分离 — 把 prefill 搬到另一台机器, 用一次 KV 传输换掉 decode 卡顿 llm_infer/m15](https://beleev.github.io#/infer/scheduler) | [查看截图](infer-scheduler-3.png) |
| [温度实验台 — 一根滑杆把同一组 logits 调尖或压平 llm_infer/m10](https://beleev.github.io#/infer/decode-control) | [查看截图](infer-decode-control-1.png) |
| [投机解码 — 小模型连猜 K 个, 大模型一次验完, 看每次调用能多出几个 token llm_infer/m07](https://beleev.github.io#/infer/decode-control) | [查看截图](infer-decode-control-2.png) |
| [采样流水线 — 重复惩罚、温度、top-k、top-p、min-p 按顺序各砍一刀 llm_infer/m10](https://beleev.github.io#/infer/decode-control) | [查看截图](infer-decode-control-3.png) |
| [Beam search — 更可能的序列, 不等于更好的序列 llm_infer/m24](https://beleev.github.io#/infer/decode-control) | [查看截图](infer-decode-control-4.png) |
| [Test-time compute — 采 N 条之后靠什么挑, 决定能涨到多高 llm_infer/m23](https://beleev.github.io#/infer/test-time-compute) | [查看截图](infer-test-time-compute-1.png) |
| [Budget forcing — 想多久由 token 预算 B 决定, 预算不够一次重写就白给 llm_infer/m23](https://beleev.github.io#/infer/test-time-compute) | [查看截图](infer-test-time-compute-2.png) |
| [FlashAttention — 分块算注意力, online softmax 让分块结果能精确拼起来 llm_infer/m11](https://beleev.github.io#/infer/compute) | [查看截图](infer-compute-1.png) |
| [Flash-Decoding — decode 的 Q 只有 1 行切不动, 就把 KV 切成段并行 llm_infer/m26](https://beleev.github.io#/infer/compute) | [查看截图](infer-compute-2.png) |
| [Radix cache — 公共前缀的 KV 只算一次, 任意长度都能命中 llm_infer/m05](https://beleev.github.io#/infer/prefix-radix) | [查看截图](infer-prefix-radix-1.png) |
| [结构化输出 — 用状态机限制每一步能选哪些 token, 并把它预编译成表 llm_infer/m14](https://beleev.github.io#/infer/structured-output) | [查看截图](infer-structured-output-1.png) |
| [INT4 权重量化 — RTN 直接舍入 vs AWQ 先放大重要通道再舍入 llm_infer/m08](https://beleev.github.io#/infer/quant-awq) | [查看截图](infer-quant-awq-1.png) |
| [KV cache 量化 — 谁和谁共用一个 scale, 决定离群值害到谁 llm_infer/m08](https://beleev.github.io#/infer/quant-awq) | [查看截图](infer-quant-awq-2.png) |
| [SmoothQuant — 把激活里的离群值挪一部分给权重, 让两边都好量化 llm_infer/m25](https://beleev.github.io#/infer/quant-awq) | [查看截图](infer-quant-awq-3.png) |
| [KV 体积 — MHA / GQA / MQA / MLA 每个 token 各占多少显存, 一张卡能同时服务几条 llm_infer/m18](https://beleev.github.io#/infer/kv-footprint) | [查看截图](infer-kv-footprint-1.png) |
| [Attention sinks — 为什么滑动窗口不能扔掉开头几个 token llm_infer/m16](https://beleev.github.io#/infer/attention-sinks) | [查看截图](infer-attention-sinks-1.png) |
| [树形投机 — draft 每层留几个候选长成树, target 用 tree mask 一次验完所有分支 llm_infer/m19](https://beleev.github.io#/infer/tree-speculation) | [查看截图](infer-tree-speculation-1.png) |
| [分层 KV offload — GPU 装不下的历史 KV 降级到 CPU 和磁盘, 再决定搬回来还是重算 llm_infer/m20](https://beleev.github.io#/infer/kv-offload) | [查看截图](infer-kv-offload-1.png) |
| [MoE 专家并行 — 热点专家把一张卡压满, EPLB 用冗余副本把它拆开 llm_infer/m21](https://beleev.github.io#/infer/moe-serving) | [查看截图](infer-moe-serving-1.png) |
| [稀疏注意力 decode — 只读打分最高的 k 个 KV block, 看丢了多少注意力 llm_infer/m22](https://beleev.github.io#/infer/sparse-decode) | [查看截图](infer-sparse-decode-1.png) |
| [多副本路由 — 送到有前缀的副本能省 prefill, 但别把一个副本压垮 llm_infer/m27](https://beleev.github.io#/infer/multi-replica-routing) | [查看截图](infer-multi-replica-routing-1.png) |
| [引擎主循环 — 三条队列、一个 block 池、一个调度分支, 看调度器每一步怎么选 llm_infer/full_engine](https://beleev.github.io#/infer/engine) | [查看截图](infer-engine-1.png) |
| [阶段 6.0 · 阶段总览](https://beleev.github.io#/agent) | [查看截图](agent.png) |
| [Agent loop — 消息列表怎么一块一块长出来, 每问一次模型为什么都要重发全部历史 llm_agent/m01](https://beleev.github.io#/agent/loop) | [查看截图](agent-loop-1.png) |
| [权限门 — 模型发出的一次工具调用, 是怎么被裁决成放行、问人或拒绝的 llm_agent/m03](https://beleev.github.io#/agent/tools-permissions) | [查看截图](agent-tools-permissions-1.png) |
| [中文检索 — 分词器不认中文时, 检索为什么整个失效 llm_agent/m08](https://beleev.github.io#/agent/context-memory) | [查看截图](agent-context-memory-1.png) |
| [RAG — 同样 20 道题, 四条检索路线各把正确答案排到第几名 llm_agent/m16](https://beleev.github.io#/agent/rag) | [查看截图](agent-rag-1.png) |
| [Hook 流水线 — 改写之后, 谁来再查一遍? llm_agent/m05](https://beleev.github.io#/agent/extensibility) | [查看截图](agent-extensibility-1.png) |
| [Skills — 装了一堆任务手册, 渐进式披露让常驻上下文只剩每本一行 llm_agent/m05](https://beleev.github.io#/agent/extensibility) | [查看截图](agent-extensibility-2.png) |
| [Resume — 进程重启后从磁盘恢复会话, 哪些东西回来了, 哪些没有 llm_agent/m06 · m07](https://beleev.github.io#/agent/state-subagents) | [查看截图](agent-state-subagents-1.png) |
| [MCP — 住在另一个进程里的工具是怎么接进 agent 的, 一次调用走了哪些报文 llm_agent/m09](https://beleev.github.io#/agent/mcp) | [查看截图](agent-mcp-1.png) |
| [A2A — 一个任务怎么在两个 agent 之间来回, 结束后为什么不能再改 llm_agent/m18](https://beleev.github.io#/agent/a2a) | [查看截图](agent-a2a-1.png) |
| [Plan 模式 — 计划批准之前「只许看不许动」, 是权限门强制的, 不是模型自觉的 llm_agent/m10](https://beleev.github.io#/agent/planning) | [查看截图](agent-planning-1.png) |
| [Orchestrator–workers — 把调研拆给几个子 agent, 三本账怎么变: lead 上下文 / 总 token / 墙钟时间 llm_agent/m11](https://beleev.github.io#/agent/orchestrator) | [查看截图](agent-orchestrator-1.png) |
| [Prompt injection — 假设模型已经照着注入做了, 四层防御里哪几层还拦得住? llm_agent/m12](https://beleev.github.io#/agent/guardrails) | [查看截图](agent-guardrails-1.png) |
| [Computer use — 看完再点, 中间页面挪了, 坐标会点到谁 llm_agent/m17](https://beleev.github.io#/agent/computer-use) | [查看截图](agent-computer-use-1.png) |
| [pass@k 与 pass^k — 「能做到」和「每次都做到」是两个指标 llm_agent/m13](https://beleev.github.io#/agent/evals) | [查看截图](agent-evals-1.png) |
| [上下文预算 — 塞不下的时候, 丢什么? llm_agent/m14](https://beleev.github.io#/agent/context-engineering) | [查看截图](agent-context-engineering-1.png) |
| [Prompt caching — 同一段会话 12 次模型调用的账单, 缓存命中和没命中差多少 llm_agent/m19](https://beleev.github.io#/agent/prompt-caching) | [查看截图](agent-prompt-caching-1.png) |
| [把每一层单独关掉 — 它原本挡住了什么 llm_agent/m12](https://beleev.github.io#/agent/full-loop) | [查看截图](agent-full-loop-1.png) |
| [总览对照表](https://beleev.github.io#/compare) | [查看截图](compare.png) |
| [Transformer 模型结构与运行态](https://beleev.github.io/llm-engineering-lab/#/models) | [查看截图](architecture-transformer.png) |
| [LLaMA 模型结构与运行态](https://beleev.github.io/llm-engineering-lab/#/models) | [查看截图](architecture-llama.png) |
| [Mamba 模型结构与运行态](https://beleev.github.io/llm-engineering-lab/#/models) | [查看截图](architecture-mamba.png) |
| [Mixtral 模型结构与运行态](https://beleev.github.io/llm-engineering-lab/#/models) | [查看截图](architecture-mixtral.png) |
| [DeepSeek‑V3 模型结构与运行态](https://beleev.github.io/llm-engineering-lab/#/models) | [查看截图](architecture-deepseek_v3.png) |
| [Qwen2‑VL 模型结构与运行态](https://beleev.github.io/llm-engineering-lab/#/models) | [查看截图](architecture-qwen2_vl.png) |
| [DiT 模型结构与运行态](https://beleev.github.io/llm-engineering-lab/#/models) | [查看截图](architecture-dit.png) |

## Agent 验证与改进扩展

下列图片由浏览器验收脚本截取真实实验台, 不是示意图。

| 章节 | 截图 |
|---|---|
| [首错诊断与轨迹回归](https://beleev.github.io/llm-engineering-lab/#/agent/evals) | [查看截图](agent-evals-extension-1.png) |
| [记忆更新 · 来源、范围与版本](https://beleev.github.io/llm-engineering-lab/#/agent/memory-lifecycle) | [查看截图](agent-memory-lifecycle-extension-1.png) |
| [完成验收 · 说完不等于做完](https://beleev.github.io/llm-engineering-lab/#/agent/completion-verification) | [查看截图](agent-completion-verification-extension-1.png) |
| [故障恢复 · 重试、幂等与取消 · 重试](https://beleev.github.io/llm-engineering-lab/#/agent/runtime-reliability) | [查看截图](agent-runtime-reliability-extension-1.png) |
| [故障恢复 · 重试、幂等与取消 · 取消](https://beleev.github.io/llm-engineering-lab/#/agent/runtime-reliability) | [查看截图](agent-runtime-reliability-extension-2.png) |
| [上下文成本 · 压缩与缓存一起算](https://beleev.github.io/llm-engineering-lab/#/agent/context-cost) | [查看截图](agent-context-cost-extension-1.png) |
| [检索证据 · 从相关到足够回答](https://beleev.github.io/llm-engineering-lab/#/agent/evidence-chain) | [查看截图](agent-evidence-chain-extension-1.png) |
| [轨迹学习 · 先验证, 再造训练数据](https://beleev.github.io/llm-engineering-lab/#/agent/trajectory-learning) | [查看截图](agent-trajectory-learning-extension-1.png) |
| [持续改进 · 学会、迁移、变化与保持](https://beleev.github.io/llm-engineering-lab/#/agent/learning-loop) | [查看截图](agent-learning-loop-extension-1.png) |

## Agent 专题实验

以下为真实浏览器中的 Python 证据回放实验, 展示配置选择、事件步进与终态读数。由 `python web/scripts/check-agent-coverage-ui.py --screenshots` 更新。

- [工具发现 · 按需加载与提示设计](agent-tool-discovery-coverage.png) · [打开实验](https://beleev.github.io/llm-engineering-lab/#/agent/tool-discovery)
- [知识组织 · 树状摘要与关系图](agent-knowledge-organization-coverage.png) · [打开实验](https://beleev.github.io/llm-engineering-lab/#/agent/knowledge-organization)
- [Coding Agent · 修改、测试与修复](agent-coding-agent-coverage.png) · [打开实验](https://beleev.github.io/llm-engineering-lab/#/agent/coding-agent)
- [异步运行 · 事件、取消与恢复](agent-async-runtime-coverage.png) · [打开实验](https://beleev.github.io/llm-engineering-lab/#/agent/async-runtime)
- [多智能体 · 独立审查与共享冲突](agent-collaboration-coverage.png) · [打开实验](https://beleev.github.io/llm-engineering-lab/#/agent/collaboration)
- [评估体系 · 用户环境与裁判校准](agent-eval-suite-coverage.png) · [打开实验](https://beleev.github.io/llm-engineering-lab/#/agent/eval-suite)
- [Agent 训练 · 轨迹到参数更新](agent-agent-training-coverage.png) · [打开实验](https://beleev.github.io/llm-engineering-lab/#/agent/agent-training)
- [Skill 进化 · 提案、发布与整理](agent-skill-evolution-coverage.png) · [打开实验](https://beleev.github.io/llm-engineering-lab/#/agent/skill-evolution)
- [语音交互 · 打断与快慢响应](agent-voice-interaction-coverage.png) · [打开实验](https://beleev.github.io/llm-engineering-lab/#/agent/voice-interaction)
- [视觉操作 · 截图、定位与核对](agent-visual-grounding-coverage.png) · [打开实验](https://beleev.github.io/llm-engineering-lab/#/agent/visual-grounding)
- [具身控制 · 反馈与模拟偏差](agent-embodied-control-coverage.png) · [打开实验](https://beleev.github.io/llm-engineering-lab/#/agent/embodied-control)
- [中期训练 · 新知识与旧能力](agent-midtraining-coverage.png) · [打开实验](https://beleev.github.io/llm-engineering-lab/#/agent/midtraining)
- [程序化行动 · 规则、适配器与生成界面](agent-programmatic-actions-coverage.png) · [打开实验](https://beleev.github.io/llm-engineering-lab/#/agent/programmatic-actions)
- [元优化 · 发现树与回放](agent-meta-optimization-coverage.png) · [打开实验](https://beleev.github.io/llm-engineering-lab/#/agent/meta-optimization)
- [智能体社会 · 激励与重复博弈](agent-agent-society-coverage.png) · [打开实验](https://beleev.github.io/llm-engineering-lab/#/agent/agent-society)
- [系统选择 · 工作流、状态与缓存](agent-system-design-coverage.png) · [打开实验](https://beleev.github.io/llm-engineering-lab/#/agent/system-design)
- [个人记忆 · 同意、过期与删除](agent-memory-privacy-coverage.png) · [打开实验](https://beleev.github.io/llm-engineering-lab/#/agent/memory-privacy)
- [向量索引 · 精确扫描与近似搜索](agent-vector-index-coverage.png) · [打开实验](https://beleev.github.io/llm-engineering-lab/#/agent/vector-index)
- [在轨自蒸馏 · 特权信息与部署边界](agent-self-distillation-coverage.png) · [打开实验](https://beleev.github.io/llm-engineering-lab/#/agent/self-distillation)
