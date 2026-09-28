# PPO (带 critic) — GRPO 省掉的那个 critic 到底买到了什么

```bash
python -m llm_finetune.run_finetune.ppo.train_ppo      # ~33 s
```

## 直觉
策略梯度的回报方差大, 要减一个 baseline。GRPO 的 baseline 是 "同一题 G 条回复的均值", 免费, 但必须一题多采, 整条回复共用一个 A。
PPO 的 baseline 是一个**学出来的** critic V(s_t): 一题采 1 条就够, 每个 token 各有自己的 A_t。代价是多养一个和 policy 同尺寸的网络 (权重 + 梯度 + Adam 状态)。
奖励与 GRPO 相同: `task.verify` (排序全对得 1 分), 落在末 token; 另外每个 token 扣 β·(log π − log π_ref), InstructGPT 式的 KL 惩罚写进奖励。

## 核心数据结构与控制流
- `methods/ppo.py:PPOTrainer.step` — rollout → 冻结 `old_logp` / `ref_logp` / `old_values` → 拼逐 token 奖励 → `gae` → 白化 A → μ=2 轮 clipped surrogate + value loss。
- `methods/ppo.py:token_values` — critic 就是 `RewardModel` (主干 + 标量头), 只是取**每个**位置的分, 位置 P−1+t 对应 V(s_t)。
- `methods/ppo.py:gae` — 从后往前一次扫描; mask 外视为终止 (r = V = 0)。
- `run_finetune/ppo/train_ppo.py:check_gae` — λ=0 / λ=1 两端的单测。

## 核心公式
`r_t = −β·(log π_old(o_t) − log π_ref(o_t)) + [t=末 token]·R`
`δ_t = r_t + γV(s_{t+1}) − V(s_t)`，`A_t = Σ_k (γλ)^k δ_{t+k}`，`G_t = A_t + V(s_t)`
λ=0 → `A_t = δ_t` (TD(0)); λ=1 → `G_t = Σ_k γ^k r_{t+k}` (Monte-Carlo 回报)
`L_π = −E[min(ρ_t A_t, clip(ρ_t, 1±ε) A_t)]`，`L_V = ½E[(V_θ(s_t) − G_t)²]`

## 运行后应该看到什么
GAE 两端 (随机 r / V, 4 条长度不等的序列, γ=0.9): λ=0 与 TD(0) 误差 **0.0**, λ=1 的回报目标与 MC 回报误差 **4.8e-07**。

训练: SFT 150 步起点 (留出集贪心 EM 0.562, 采样 pass@1 0.186); 每步 **256 条回复** × 60 步, lr=3e-4, 全部指标在留出集:

| | 贪心 EM | 采样 pass@1 | 训练奖励 前 10 步 → 后 10 步 | baseline 解释的方差 | 常驻参数 | 要训的参数 | 耗时 |
|---|---|---|---|---|---|---|---|
| GRPO (32 题 × G=8) | 0.531 | 0.287 | 0.205 → 0.295 | **0.365** (组均值) | 99,648 | 99,648 | 6.1 s |
| PPO (256 题 × 1) | 0.570 | **0.381** | 0.200 → 0.356 | 0.025 (V(s_0)) | 299,008 | 199,360 | 11.8 s |
| PPO, critic 冻结 (lr=0) | 0.586 | **0.389** | 0.204 → 0.357 | 0.004 | 299,008 | 199,360 | 12.0 s |

"baseline 解释的方差" = `1 − Var(R − b)/Var(R)`, 在留出集 64 题 × 8 条上量。

**如实解读**:
- PPO 确实学到了东西: 训练奖励 0.20 → 0.36, 留出集 pass@1 0.186 → 0.381。
- 同样的采样预算, PPO 的 pass@1 比 GRPO 高 0.09 —— 但**不是 critic 的功劳**: critic 冻结 (V≈0, 退化成 "批内白化的 REINFORCE + clip") 的对照一样高 (0.389)。差别来自采样方式: 256 道不同的题各采 1 条, 而 GRPO 的 32 道题里约 1/3 组内全对或全错, A≡0, 白占预算。
- critic 在 60 步里几乎没学会: V(s_0) 只解释 **2.5%** 的回报方差, 免费的组均值解释 **36.5%**。"方差更小" 在这个规模上**不成立**。critic 要判断 "这个排序前缀还能不能全对", 这和 policy 学排序本身一样难; 组均值却不用学, 直接把 "这道题难不难" 平均出来。
- 代价是实打实的: 要训的参数 ×2, 常驻 ×3 (policy + ref + critic), 每步耗时 ×2。这就是 DeepSeek 用 GRPO 去掉 critic 的动机。

## 诚实边界
- 大模型上 PPO 的 critic 通常从 RM 或 SFT 初始化并先单独预热, 能学到有用的 V; 这里 60 步、64 维的 critic 学不到, 所以 "逐 token 优势带来更好的信用分配" 在本脚本里**看不到**。
- 组均值的解释方差略偏乐观: 均值里包含样本自己, 残差方差被乘了 (1 − 1/G)。
- critic 冻结那一行的 "要训的参数" 照算 (梯度仍在算, 只是 lr=0)。

## 常见误区
- V(s_t) 取错位置: 位置 P−1+t 读完的是 prompt + o_<t, 正好是 "还没生成 o_t" 的状态, 与 `completion_logprobs` 同一个错位。
- 末 token 之后不当终止: EOS 后的 V 必须记 0, 否则 GAE 会从垃圾 token 上自举。
- KL 写进奖励和写进 loss 不是一回事: 写进奖励 (这里) 它会被 critic 学、被 GAE 折扣累加; GRPO 是直接加在 loss 上。
- 以为 critic 一定降方差: 学不好的 critic 等于常数 baseline, 上表就是。

## 自测题
1. GAE 的 λ=0 和 λ=1 各退化成什么? 各自的偏差 / 方差特点? — TD(0) 只信下一步 V, 方差小但 V 不准时有偏; MC 只用真实回报, 无偏但方差大。
2. 为什么 GRPO 一题只采 1 条就学不到东西, PPO 可以? — G=1 时组均值就是自己, A≡0; PPO 的 baseline 来自 critic (或批内白化), 与组无关。
3. 本脚本里 PPO 比 GRPO 的 pass@1 高, 能说明 critic 有用吗? — 不能。critic 冻结的对照一样高; 差别来自每步 256 道不同的题。
