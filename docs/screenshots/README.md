# 章节预览

这些图片截自本仓库的教程站。点击章节打开交互页面；模型结构图位于阶段 2 的模型选择器中。

| 章节 | 截图 |
| --- | --- |
| [主线总览](https://beleev.github.io#/) | [查看截图](home.png) |
| [阶段 1.0 · 阶段总览](https://beleev.github.io#/basic) | [查看截图](basic.png) |
| [BPE 分词实验台 — 看着词表长出来 llm_basic/bpe.py](https://beleev.github.io#/basic/data) | [查看截图](basic-data-1.png) |
| [形状流水线 — ids [B,T] 到 logits [B,T,V] 的一路 llm_basic/model.py](https://beleev.github.io#/basic/forward) | [查看截图](basic-forward-1.png) |
| [反向传播 — 点一个节点, 看梯度怎么传到它 llm_basic/model.py](https://beleev.github.io#/basic/backward) | [查看截图](basic-backward-1.png) |
| [gradcheck 的 ε — 为什么不是越小越好 llm_basic/gradcheck.py](https://beleev.github.io#/basic/backward) | [查看截图](basic-backward-2.png) |
| [优化器轨迹 — 同一个学习率, 三种走法 llm_basic/optim.py](https://beleev.github.io#/basic/optim-sample) | [查看截图](basic-optim-sample-1.png) |
| [温度实验台 — 一根滑杆连接采样与蒸馏 llm_infer/m10](https://beleev.github.io#/basic/optim-sample) | [查看截图](basic-optim-sample-2.png) |
| [阶段 2.0 · 阶段总览](https://beleev.github.io#/models) | [查看截图](models.png) |
| [注意力掩码实验台 — 谁能看见谁 llm_infer/m16](https://beleev.github.io#/attention) | [查看截图](attention-1.png) |
| [位置编码](https://beleev.github.io#/position) | [查看截图](position.png) |
| [残差流 — norm 挪一个位置, 深层网络就能开训 llm_models/layers/core/blocks.py](https://beleev.github.io#/blocks) | [查看截图](blocks-1.png) |
| [初始化 — 第一步 loss 该是多少? llm_models/utils/init.py](https://beleev.github.io#/blocks) | [查看截图](blocks-2.png) |
| [MoE 路由](https://beleev.github.io#/moe) | [查看截图](moe.png) |
| [注意力掩码实验台 — 谁能看见谁 llm_infer/m16](https://beleev.github.io#/models/swa-mtp) | [查看截图](models-mtp-1.png) |
| [MTP 实验台 — 一次前向, 多步预测 llm_models/mtp](https://beleev.github.io#/models/swa-mtp) | [查看截图](models-mtp-2.png) |
| [固定大小的状态 vs 越长越大的 KV llm_models/layers/sparse/linear_attention.py](https://beleev.github.io#/models/swa-mtp) | [查看截图](models-mtp-3.png) |
| [扩散生成](https://beleev.github.io#/diffusion) | [查看截图](diffusion.png) |
| [窗口满了 — KV cache 作废, 之后每一步都整段重算 llm_models/utils/generation.py](https://beleev.github.io#/models/generation) | [查看截图](models-generation-1.png) |
| [QK-Norm — 把注意力 logit 关进笼子 llm_models/layers/core/attention.py](https://beleev.github.io#/models/qknorm-yarn) | [查看截图](models-qknorm-yarn-1.png) |
| [YaRN / NTK — 每个频率该不该被压缩 llm_models/layers/core/position_encoding.py](https://beleev.github.io#/models/qknorm-yarn) | [查看截图](models-qknorm-yarn-2.png) |
| [Mamba 选择性扫描 — Δ 决定记谁、忘谁 llm_models/layers/sparse/ssm.py](https://beleev.github.io#/models/mamba) | [查看截图](models-mamba-1.png) |
| [Aux-loss-free 均衡 — 偏置只管选人, 不管权重 llm_models/models/moe/deepseekV3.py](https://beleev.github.io#/models/moe-balance) | [查看截图](models-moe-balance-1.png) |
| [DSA — 先粗选 top-k, 再精算注意力 llm_models/layers/core/attention.py](https://beleev.github.io#/models/dsa) | [查看截图](models-dsa-1.png) |
| [滑窗 × 全注意力交替 — 信息怎么一层层传过来 llm_models/models/moe/gpt_oss.py](https://beleev.github.io#/models/gptoss) | [查看截图](models-gptoss-1.png) |
| [Attention sink — 给 softmax 一个 “弃权” 选项 llm_models/layers/core/attention.py](https://beleev.github.io#/models/gptoss) | [查看截图](models-gptoss-2.png) |
| [LLaDA — 先填有把握的, 没把握的遮回去再想 llm_models/models/language_models/llada.py](https://beleev.github.io#/models/llada) | [查看截图](models-llada-1.png) |
| [VAR — 下一个尺度, 而不是下一个像素 llm_models/models/generative/var.py](https://beleev.github.io#/models/var) | [查看截图](models-var-1.png) |
| [阶段 3.0 · 阶段总览](https://beleev.github.io#/train) | [查看截图](train.png) |
| [一个 batch, 三种切法 — 梯度都是同一个 llm_train/m01+m02](https://beleev.github.io#/train/batch-ddp) | [查看截图](train-batch-ddp-1.png) |
| [流水线调度 — GPipe vs 1F1B llm_train/m04](https://beleev.github.io#/train/model-parallel) | [查看截图](train-model-parallel-1.png) |
| [张量并行 — 通信到底该插在哪? llm_train/m03](https://beleev.github.io#/train/model-parallel) | [查看截图](train-model-parallel-2.png) |
| [ZeRO 显存账 — 2 + 2 + 12 字节, 先切哪一块? llm_train/m05](https://beleev.github.io#/train/memory) | [查看截图](train-memory-1.png) |
| [激活重算 — 段长 k 怎么选? llm_train/m07](https://beleev.github.io#/train/memory) | [查看截图](train-memory-2.png) |
| [浮点数轴 — 梯度落在哪个格式的范围里? llm_train/m06](https://beleev.github.io#/train/precision-stability) | [查看截图](train-precision-stability-1.png) |
| [MoE 路由实验台 — 倾斜、溢出与 all-to-all llm_train/m11](https://beleev.github.io#/train/moe-seq-parallel) | [查看截图](train-moe-seq-1.png) |
| [Ring Attention 步进器 — KV 块沿环传递 llm_train/m12](https://beleev.github.io#/train/moe-seq-parallel) | [查看截图](train-moe-seq-2.png) |
| [Zigzag — 因果注意力下谁在等谁? llm_train/m12](https://beleev.github.io#/train/moe-seq-parallel) | [查看截图](train-moe-seq-3.png) |
| [交错 1F1B — 把气泡再除以 v llm_train/m04](https://beleev.github.io#/train/pipeline-schedules) | [查看截图](train-pipeline-schedules-1.png) |
| [Cosine vs WSD — 训到一半想加训怎么办? llm_train/m10](https://beleev.github.io#/train/lr-schedule) | [查看截图](train-lr-schedule-1.png) |
| [Block scaling — 一个 outlier 会连累多少邻居? llm_train/m13 · m15](https://beleev.github.io#/train/low-precision) | [查看截图](train-low-precision-1.png) |
| [Muon — Newton–Schulz 把奇异值谱拉平 llm_train/m14](https://beleev.github.io#/train/muon) | [查看截图](train-muon-1.png) |
| [Muon vs Adam — 把病态方向转一下会怎样? llm_train/m14](https://beleev.github.io#/train/muon) | [查看截图](train-muon-2.png) |
| [Ulysses — all-to-all 把「切序列」换成「切头」 llm_train/m16](https://beleev.github.io#/train/ulysses) | [查看截图](train-ulysses-1.png) |
| [MinHash-LSH — 谁会被拿来比较? llm_train/m17](https://beleev.github.io#/train/data-packing) | [查看截图](train-data-packing-1.png) |
| [数据源温度配比 — 小语料抬多高才合适? llm_train/m17](https://beleev.github.io#/train/data-packing) | [查看截图](train-data-packing-2.png) |
| [Packing — 拼在一行的文档会互相看见吗? llm_train/m18](https://beleev.github.io#/train/data-packing) | [查看截图](train-data-packing-3.png) |
| [IsoFLOP — 钱一定, 模型做多大? llm_train/m19](https://beleev.github.io#/train/scaling) | [查看截图](train-scaling-1.png) |
| [μP — 小模型调好的 lr 能直接搬到大模型吗? llm_train/m20](https://beleev.github.io#/train/scaling) | [查看截图](train-scaling-2.png) |
| [pass@k — 朴素公式为什么偏低? llm_train/m21](https://beleev.github.io#/train/eval) | [查看截图](train-eval-1.png) |
| [LLM 裁判 — 先出现的回答占便宜吗? llm_train/m21](https://beleev.github.io#/train/eval) | [查看截图](train-eval-2.png) |
| [Ring all-reduce — 每张卡每步只给右邻居发一块 llm_train/core](https://beleev.github.io#/train/collectives-loop) | [查看截图](train-collectives-loop-1.png) |
| [阶段 4.0 · 阶段总览](https://beleev.github.io#/finetune) | [查看截图](finetune.png) |
| [SFT 标签对齐 — mask 边界差一格会丢掉什么 llm_finetune/data](https://beleev.github.io#/finetune/sft) | [查看截图](finetune-sft-1.png) |
| [LoRA — 用 B·A 拟合一个低秩的 ΔW llm_finetune/methods/lora.py](https://beleev.github.io#/finetune/lora) | [查看截图](finetune-lora-1.png) |
| [DPO — 一条 logsigmoid 曲线上的 6 个偏好对 llm_finetune/methods/dpo.py](https://beleev.github.io#/finetune/dpo) | [查看截图](finetune-dpo-1.png) |
| [KTO: 好坏 1:9 时, λ 为什么必须跟着调 llm_finetune/methods/kto.py](https://beleev.github.io#/finetune/dpo) | [查看截图](finetune-dpo-2.png) |
| [GRPO 入门 — 组内排名替代 critic llm_finetune/methods/grpo.py](https://beleev.github.io#/finetune/rlhf-grpo-distill) | [查看截图](finetune-rlhf-1.png) |
| [温度实验台 — 一根滑杆连接采样与蒸馏 llm_infer/m10](https://beleev.github.io#/finetune/rlhf-grpo-distill) | [查看截图](finetune-rlhf-2.png) |
| [Bradley–Terry — 奖励模型学到的只是「差」 llm_finetune/methods/reward_model.py](https://beleev.github.io#/finetune/rlhf-grpo-distill) | [查看截图](finetune-rlhf-3.png) |
| [NF4 vs INT4 — 16 个码点该摆在哪 llm_finetune/methods/qlora.py](https://beleev.github.io#/finetune/qlora) | [查看截图](finetune-qlora-1.png) |
| [DoRA — 把「转方向」和「改长度」拆成两个旋钮 llm_finetune/methods/dora.py](https://beleev.github.io#/finetune/dora) | [查看截图](finetune-dora-1.png) |
| [合并两个任务向量: 相加、TIES、DARE llm_finetune/methods/merge.py](https://beleev.github.io#/finetune/merge) | [查看截图](finetune-merge-1.png) |
| [拿掉 reference 之后 — SimPO 与 ORPO 靠什么不被长度骗 llm_finetune/methods/simpo.py · orpo.py](https://beleev.github.io#/finetune/simpo-orpo) | [查看截图](finetune-simpo-orpo-1.png) |
| [judge 没查的, DPO 学不到 llm_finetune/methods/rlaif.py](https://beleev.github.io#/finetune/rlaif) | [查看截图](finetune-rlaif-1.png) |
| [一条解, ORM 看到 1 个标签, PRM 看到 4 个 llm_finetune/methods/prm.py](https://beleev.github.io#/finetune/prm) | [查看截图](finetune-prm-1.png) |
| [GAE: 从 TD(0) 滑到 Monte-Carlo llm_finetune/methods/ppo.py](https://beleev.github.io#/finetune/ppo) | [查看截图](finetune-ppo-1.png) |
| [同样 256 条回复: 组均值 vs critic llm_finetune/methods/ppo.py](https://beleev.github.io#/finetune/ppo) | [查看截图](finetune-ppo-2.png) |
| [ratio clip — 哪些 token 这一步已经不许再推了 llm_finetune/methods/grpo.py](https://beleev.github.io#/finetune/grpo-variants) | [查看截图](finetune-grpo-variants-1.png) |
| [GRPO → Dr.GRPO → DAPO: 每个 token 到底分到多少梯度 llm_finetune/methods/grpo.py](https://beleev.github.io#/finetune/grpo-variants) | [查看截图](finetune-grpo-variants-2.png) |
| [GSPO — 比率该按 token 算, 还是按整条序列算 llm_finetune/methods/grpo.py](https://beleev.github.io#/finetune/grpo-variants) | [查看截图](finetune-grpo-variants-3.png) |
| [forward KL vs reverse KL — 学生该盖住所有峰, 还是钻进一个峰 llm_finetune/methods/on_policy_distill.py](https://beleev.github.io#/finetune/onpolicy-distill) | [查看截图](finetune-onpolicy-distill-1.png) |
| [蒸馏为什么要乘 T² — 温度一高, 软标签梯度就悄悄消失 llm_finetune/methods/distill.py](https://beleev.github.io#/finetune/onpolicy-distill) | [查看截图](finetune-onpolicy-distill-2.png) |
| [可验证奖励要「看题」— 否则 RL 只学会一个常数 llm_finetune/data/tasks.py](https://beleev.github.io#/finetune/rlvr) | [查看截图](finetune-rlvr-1.png) |
| [选型计算器 — 16 种方法, 各自要你付什么 llm_finetune/README.md](https://beleev.github.io#/finetune/runs) | [查看截图](finetune-runs-1.png) |
| [阶段 5.0 · 阶段总览](https://beleev.github.io#/infer) | [查看截图](infer.png) |
| [注意力掩码实验台 — 谁能看见谁 llm_infer/m16](https://beleev.github.io#/infer/kv-memory) | [查看截图](infer-kv-memory-1.png) |
| [KV cache — 每一步到底重算了多少 token llm_infer/m01](https://beleev.github.io#/infer/kv-memory) | [查看截图](infer-kv-memory-2.png) |
| [PagedAttention — 把 KV 显存切成 block 按需分配 llm_infer/m02](https://beleev.github.io#/infer/kv-memory) | [查看截图](infer-kv-memory-3.png) |
| [连续批处理 — 静态 batch vs 逐步重组 llm_infer/m03](https://beleev.github.io#/infer/scheduler) | [查看截图](infer-scheduler-1.png) |
| [Chunked prefill — 用 token 预算封顶卡顿 llm_infer/m06](https://beleev.github.io#/infer/scheduler) | [查看截图](infer-scheduler-2.png) |
| [P/D 分离 — 拿 KV 传输换掉 decode 卡顿 llm_infer/m15](https://beleev.github.io#/infer/scheduler) | [查看截图](infer-scheduler-3.png) |
| [温度实验台 — 一根滑杆连接采样与蒸馏 llm_infer/m10](https://beleev.github.io#/infer/decode-control) | [查看截图](infer-decode-control-1.png) |
| [投机解码 — draft 猜 K 个, target 一次验 K+1 个 llm_infer/m07](https://beleev.github.io#/infer/decode-control) | [查看截图](infer-decode-control-2.png) |
| [采样流水线 — 重复惩罚 → 温度 → top-k → top-p → min-p llm_infer/m10](https://beleev.github.io#/infer/decode-control) | [查看截图](infer-decode-control-3.png) |
| [Beam search — 更可能, 不等于更好 llm_infer/m24](https://beleev.github.io#/infer/decode-control) | [查看截图](infer-decode-control-4.png) |
| [Test-time compute — 采 N 条之后靠什么挑 llm_infer/m23](https://beleev.github.io#/infer/test-time-compute) | [查看截图](infer-test-time-compute-1.png) |
| [Budget forcing — 想多久由预算 B 决定 llm_infer/m23](https://beleev.github.io#/infer/test-time-compute) | [查看截图](infer-test-time-compute-2.png) |
| [FlashAttention — 分块 + online softmax llm_infer/m11](https://beleev.github.io#/infer/compute) | [查看截图](infer-compute-1.png) |
| [Flash-Decoding — Q 切不动, 就切 KV llm_infer/m26](https://beleev.github.io#/infer/compute) | [查看截图](infer-compute-2.png) |
| [Radix cache — 公共前缀只算一次 llm_infer/m05](https://beleev.github.io#/infer/prefix-radix) | [查看截图](infer-prefix-radix-1.png) |
| [结构化输出 — 把 FSM 预编译成 token mask 表 llm_infer/m14](https://beleev.github.io#/infer/structured-output) | [查看截图](infer-structured-output-1.png) |
| [INT4 权重量化 — RTN vs AWQ 激活感知缩放 llm_infer/m08](https://beleev.github.io#/infer/quant-awq) | [查看截图](infer-quant-awq-1.png) |
| [KV 量化 — per-tensor / per-token / per-channel 谁背离群值的锅 llm_infer/m08](https://beleev.github.io#/infer/quant-awq) | [查看截图](infer-quant-awq-2.png) |
| [SmoothQuant — 把离群值从激活挪给权重 llm_infer/m25](https://beleev.github.io#/infer/quant-awq) | [查看截图](infer-quant-awq-3.png) |
| [KV 体积 — MHA / GQA / MQA / MLA 各占多少显存 llm_infer/m18](https://beleev.github.io#/infer/kv-footprint) | [查看截图](infer-kv-footprint-1.png) |
| [Attention sinks — 为什么滑动窗口不能扔掉开头几个 token llm_infer/m16](https://beleev.github.io#/infer/attention-sinks) | [查看截图](infer-attention-sinks-1.png) |
| [树形投机 — tree attention mask 一次验整棵树 llm_infer/m19](https://beleev.github.io#/infer/tree-speculation) | [查看截图](infer-tree-speculation-1.png) |
| [分层 KV offload — GPU 装不下的历史, 降级而不是丢弃 llm_infer/m20](https://beleev.github.io#/infer/kv-offload) | [查看截图](infer-kv-offload-1.png) |
| [MoE 专家并行 — 热点专家与 EPLB 冗余副本 llm_infer/m21](https://beleev.github.io#/infer/moe-serving) | [查看截图](infer-moe-serving-1.png) |
| [稀疏注意力 decode — 只读 top-k 个 KV block llm_infer/m22](https://beleev.github.io#/infer/sparse-decode) | [查看截图](infer-sparse-decode-1.png) |
| [多副本路由 — 命中率 vs 负载均衡 llm_infer/m27](https://beleev.github.io#/infer/multi-replica-routing) | [查看截图](infer-multi-replica-routing-1.png) |
| [引擎主循环 — 三条队列、一个 block 池、一个调度分支 llm_infer/full_engine](https://beleev.github.io#/infer/engine) | [查看截图](infer-engine-1.png) |
| [阶段 6.0 · 阶段总览](https://beleev.github.io#/agent) | [查看截图](agent.png) |
| [Agent loop — 消息列表是怎么一块一块长出来的 llm_agent/m01](https://beleev.github.io#/agent/loop) | [查看截图](agent-loop-1.png) |
| [权限门 — 一次工具调用是怎么被裁决的 llm_agent/m03](https://beleev.github.io#/agent/tools-permissions) | [查看截图](agent-tools-permissions-1.png) |
| [中文检索 — 为什么要字符 bigram llm_agent/m08](https://beleev.github.io#/agent/context-memory) | [查看截图](agent-context-memory-1.png) |
| [RAG — 20 道题, 四条检索路线 llm_agent/m16](https://beleev.github.io#/agent/rag) | [查看截图](agent-rag-1.png) |
| [Hook 流水线 — 改写之后, 谁来再查一遍? llm_agent/m05](https://beleev.github.io#/agent/extensibility) | [查看截图](agent-extensibility-1.png) |
| [Skills — 渐进式披露省下多少上下文 llm_agent/m05](https://beleev.github.io#/agent/extensibility) | [查看截图](agent-extensibility-2.png) |
| [Resume — 哪些东西回来了, 哪些没有 llm_agent/m06 · m07](https://beleev.github.io#/agent/state-subagents) | [查看截图](agent-state-subagents-1.png) |
| [MCP — 一次外部工具调用的完整报文 llm_agent/m09](https://beleev.github.io#/agent/mcp) | [查看截图](agent-mcp-1.png) |
| [A2A — 一个会反问的任务 llm_agent/m18](https://beleev.github.io#/agent/a2a) | [查看截图](agent-a2a-1.png) |
| [Plan 模式 — 只读是门强制的, 不是模型自觉的 llm_agent/m10](https://beleev.github.io#/agent/planning) | [查看截图](agent-planning-1.png) |
| [Orchestrator–workers — 三本账: lead 上下文 / 总 token / 墙钟时间 llm_agent/m11](https://beleev.github.io#/agent/orchestrator) | [查看截图](agent-orchestrator-1.png) |
| [Prompt injection — 假设模型已经上当, 哪几层还拦得住? llm_agent/m12](https://beleev.github.io#/agent/guardrails) | [查看截图](agent-guardrails-1.png) |
| [Computer use — 看完再点, 中间页面挪了 llm_agent/m17](https://beleev.github.io#/agent/computer-use) | [查看截图](agent-computer-use-1.png) |
| [pass@k vs pass^k — 能力上限, 还是可靠性? llm_agent/m13](https://beleev.github.io#/agent/evals) | [查看截图](agent-evals-1.png) |
| [上下文预算 — 塞不下的时候, 丢什么? llm_agent/m14](https://beleev.github.io#/agent/context-engineering) | [查看截图](agent-context-engineering-1.png) |
| [Prompt caching — 12 次调用的账单 llm_agent/m19](https://beleev.github.io#/agent/prompt-caching) | [查看截图](agent-prompt-caching-1.png) |
| [把每一层单独关掉 — 它原本挡住了什么 llm_agent/m12](https://beleev.github.io#/agent/full-loop) | [查看截图](agent-full-loop-1.png) |
| [总览对照表](https://beleev.github.io#/compare) | [查看截图](compare.png) |
| [Transformer 模型结构与运行态](https://beleev.github.io/llm-engineering-lab/#/models) | [查看截图](architecture-transformer.png) |
| [LLaMA 模型结构与运行态](https://beleev.github.io/llm-engineering-lab/#/models) | [查看截图](architecture-llama.png) |
| [Mamba 模型结构与运行态](https://beleev.github.io/llm-engineering-lab/#/models) | [查看截图](architecture-mamba.png) |
| [Mixtral 模型结构与运行态](https://beleev.github.io/llm-engineering-lab/#/models) | [查看截图](architecture-mixtral.png) |
| [DeepSeek‑V3 模型结构与运行态](https://beleev.github.io/llm-engineering-lab/#/models) | [查看截图](architecture-deepseek_v3.png) |
| [Qwen2‑VL 模型结构与运行态](https://beleev.github.io/llm-engineering-lab/#/models) | [查看截图](architecture-qwen2_vl.png) |
| [DiT 模型结构与运行态](https://beleev.github.io/llm-engineering-lab/#/models) | [查看截图](architecture-dit.png) |
