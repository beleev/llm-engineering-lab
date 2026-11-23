#!/usr/bin/env python
"""
标量 RM vs rubric 奖励: 偏好标签带长度偏差时, Bradley-Terry 标量 RM 学到 "越长越好"。
在基座分布内挑回复 (best-of-N) 时这点偏差影响不大; 一旦用 RL 把策略推出数据分布, 策略就沿长度方向钻空子。
逐条判定的 rubric 奖励不吃长度, 代价是要为每题写 rubric, judge 自己也会判错。

    python -m llm_finetune.run_finetune.rubric_reward.demo

为什么不训练 LM: 这里研究的是 "奖励信号会被怎样利用", 不是生成器。回复直接用特征表示。
植入的设定 (合成数据, 不是从真实数据里测出来的):
    回复 = K=5 条 rubric 条目是否满足 + 长度 L (token 数)。前 4 条是内容条目, 第 5 条是 "简洁": L ≤ 300 才算满足。
    真实质量 q = 5 条里满足的比例。基座策略: 内容条目各以 0.5 的概率满足, L ~ N(200, 60²), 截在 [20, 600]。
    偏好标签带长度偏差: P(A ≻ B) = σ(γ·(q_A − q_B) + β·(L_A − L_B)/100), γ=4, β=0.8 (对照组 β=0)。
        β=0.8 的含义: 多写 100 个 token 在标注者眼里值 0.8 个 logit, 与多满足一条内容条目 (γ/5 = 0.8) 一样多。
四种奖励:
    标量 RM      线性 Bradley-Terry 模型, 输入 = 5 个条目指示 + L/100, 在 4096 个偏好对上训练。
                 标签恰好是特征差的 logistic 函数, 模型设定是对的, 学到的系数应接近标注公式。
                 简化: RM 直接看到每条条目满不满足, 这比真实 RM 读文本要容易得多, 对 RM 是偏宽松的设定。
    rubric       judge 逐条判, 每条以 ε 的概率判反; 奖励 = 判定满足的比例 (等权的显式聚合)。
    rubric+投票  同一个 judge 独立判 5 次取平均 (GenRM 对多条推理的 "Yes" 概率取平均, 这里是离散版)。
    oracle       直接用真实质量 q, 是上限。
两种优化压力:
    best-of-N    从基座采 N 条, 按奖励挑最高的一条。候选都来自基座分布, RM 只在训练分布内被使用。
    RL           参数化策略: 4 个内容条目的 logit + 长度均值 μ (标准差固定 60), REINFORCE, 批均值当 baseline,
                 每步 256 条, 300 步。策略可以把 μ 推到训练数据从没出现过的长度, 上限 600 (相当于 max_new_tokens)。
断言 (每条验证一个结论):
    1. RM 在带偏差的标签上学到的长度系数 > 0.6 (标注值 0.8); β=0 的对照组 |长度系数| < 0.15。
    2. 5 次取平均把 ε=0.3 的逐条判错率降到 0.2 以下。
    3. best-of-N: β=0.8 的 RM 挑出的回复随 N 变长 (N=64 比 N=1 长 40 token 以上), rubric 挑出的长度基本不变 (变化 < 15)。
    4. best-of-N: N=64 时 β=0.8 的 RM 真实质量仍高于 rubric (ε=0.1)。分布内 RM 看得见全部条目, 长度偏差只分走一点选择力。
    5. best-of-N: judge 越不准收益越小 (ε=0.3 < ε=0.1); 投票后高于单次判定。
    6. best-of-N: judge 的误差也被优化压力利用, ε=0.3 时挑中者 "judge 分 − 真实质量" 在 N=64 比 N=1 大 0.1 以上。
    7. RL: β=0.8 的 RM 把平均长度推到 590 以上, 真实质量停在 0.85 以下, 而它自己的 RM 分一路上涨 (末步比第 0 步高 1 以上)。
    8. RL: rubric (ε=0.1 和 ε=0.3) 的真实质量都到 0.95 以上, 平均长度不超过 300。
"""

import torch
import torch.nn.functional as F

K, L_MAX = 5, 300                    # rubric 条目数 (最后一条是 "简洁"); 简洁条目的长度上限
GAMMA, BETA = 4.0, 0.8               # 标注者: 质量差的权重; 每 100 token 长度差的权重 (长度偏差)
N_PAIRS, RM_STEPS = 4096, 300
N_PROMPTS, NS, VOTES = 2000, (1, 4, 16, 64), 5
RL_STEPS, RL_BATCH, RL_LR = 300, 256, 0.05


def sample(*shape):
    """基座策略采样 → items [*shape, K] (0/1), length [*shape]。"""
    content = (torch.rand(*shape, K - 1) < 0.5).float()                      # 4 条内容条目, 与长度独立
    length = (torch.randn(*shape) * 60 + 200).clamp(20, 600).round()
    return with_concise(content, length), length


def with_concise(content, length):
    """拼上第 5 条 "简洁" 条目。★ 植入: 写太长就违反它。"""
    return torch.cat([content, (length <= L_MAX).float().unsqueeze(-1)], -1)


def features(items, length):
    """RM 的输入: 5 个条目指示 + L/100 → [..., K+1]。"""
    return torch.cat([items, (length / 100).unsqueeze(-1)], -1)


def train_rm(beta: float) -> torch.Tensor:
    """在 β 偏差的偏好对上训线性 Bradley-Terry RM, 返回权重 w [K+1]; 分数 r = w·features。"""
    (ia, la), (ib, lb) = sample(N_PAIRS), sample(N_PAIRS)
    logit = GAMMA * (ia.mean(-1) - ib.mean(-1)) + beta * (la - lb) / 100      # 标注者心里的 logit
    a_wins = torch.rand(N_PAIRS) < torch.sigmoid(logit)                        # 按概率抽标签, 不取 argmax
    fa, fb = features(ia, la), features(ib, lb)
    diff = torch.where(a_wins.unsqueeze(1), fa - fb, fb - fa)                  # chosen − rejected
    w = torch.zeros(K + 1, requires_grad=True)                                 # BT 只看分差, 不需要偏置项
    opt = torch.optim.Adam([w], lr=0.05)
    for _ in range(RM_STEPS):
        loss = -F.logsigmoid(diff @ w).mean()                                  # −log σ(r_w − r_l)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return w.detach()


def judge(items, eps: float, votes: int = 1) -> torch.Tensor:
    """逐条判: 每次每条以 ε 判反, votes 次取平均 → [..., K] ∈ [0, 1]。"""
    flips = (torch.rand(*items.shape, votes) < eps).float()                    # [..., K, votes]
    return (items.unsqueeze(-1) - flips).abs().mean(-1)                        # 判反 = 1 − 真值


def rl(reward, rm_w):
    """REINFORCE 训练参数化策略, 返回 {步数: (真实质量, 平均长度, 用 rm_w 打的 RM 分)}。"""
    torch.manual_seed(1)
    theta = torch.zeros(K - 1, requires_grad=True)                             # 内容条目的 logit, 起点 p=0.5
    mu = torch.zeros(1, requires_grad=True)                                    # 长度均值 = 200 + 100·mu
    opt = torch.optim.Adam([theta, mu], lr=RL_LR)
    log = {}
    for step in range(RL_STEPS + 1):
        p = torch.sigmoid(theta)
        content = (torch.rand(RL_BATCH, K - 1) < p).float()                   # [B, 4]
        raw = (200 + 100 * mu + 60 * torch.randn(RL_BATCH)).detach()           # [B] 采样出的长度 (截断前)
        length = raw.clamp(20, 600).round()                                    # 上限 600: 生成长度封顶
        items = with_concise(content, length)
        r = reward(items, length)
        if step % 100 == 0:
            log[step] = (float(items.mean()), float(length.mean()), float((features(items, length) @ rm_w).mean()))
        # log π = 伯努利项 + 高斯项; 截断不影响得分函数, 奖励按截断后的长度算
        logp = (content * p.log() + (1 - content) * (1 - p).log()).sum(1) - (raw - 200 - 100 * mu) ** 2 / (2 * 60 ** 2)
        loss = -((r - r.mean()) * logp).mean()                                 # 批均值当 baseline
        opt.zero_grad()
        loss.backward()
        opt.step()
    return log


def main() -> None:
    torch.manual_seed(0)
    w_biased, w_clean = train_rm(BETA), train_rm(0.0)
    print("---- 标量 RM 学到的系数 (5 个条目 | 每 100 token 的长度) ----")
    print(f"标注公式    条目各 {GAMMA / K:.2f} | 长度 β=0.8 (对照组 0)")
    for name, w in (("RM β=0.8", w_biased), ("RM β=0", w_clean)):
        print(f"{name:<10}  条目 {' '.join(f'{x:.2f}' for x in w[:K].tolist())} | 长度 {w[K]:.2f}")
    assert w_biased[K] > 0.6, f"带长度偏差的标签应让 RM 学到正的长度系数 (标注值 0.8), 实际 {w_biased[K]:.3f}"
    assert abs(w_clean[K]) < 0.15, f"无偏标签 (β=0) 训出的 RM 长度系数应接近 0, 实际 {w_clean[K]:.3f}"

    # ---- judge 的逐条判错率: 单次 vs 5 次取平均 (平均值 > 0.5 算判满足) ----
    probe = sample(20000)[0]
    err = {(eps, m): float(((judge(probe, eps, m) > 0.5).float() != probe).float().mean())
           for eps in (0.1, 0.3) for m in (1, VOTES)}
    print(f"\njudge 逐条判错率: ε=0.1 单次 {err[(0.1, 1)]:.3f}, 投票 {err[(0.1, VOTES)]:.3f}; "
          f"ε=0.3 单次 {err[(0.3, 1)]:.3f}, 投票 {err[(0.3, VOTES)]:.3f}")
    assert err[(0.3, VOTES)] < 0.2, f"5 次取平均应把 ε=0.3 的逐条判错率降到 0.2 以下, 实际 {err[(0.3, VOTES)]:.3f}"

    # ---- best-of-N: 同一批候选, 换不同的奖励来挑 ----
    arms = ("oracle", "RM β=0.8", "RM β=0", "rubric ε=0.1", "rubric ε=0.3", "ε=0.3 投票")
    res, gap = {a: {} for a in arms}, {}                                       # gap: ε=0.3 挑中者的 judge 分 − 真实质量
    for n in NS:
        items, length = sample(N_PROMPTS, n)                                   # [P, N, K], [P, N]
        f, j3 = features(items, length), judge(items, 0.3).mean(-1)
        scores = {"oracle": items.mean(-1) + 1e-6 * torch.rand(N_PROMPTS, n),  # 微小扰动: 同分时随机挑
                  "RM β=0.8": f @ w_biased, "RM β=0": f @ w_clean,
                  "rubric ε=0.1": judge(items, 0.1).mean(-1), "rubric ε=0.3": j3,
                  "ε=0.3 投票": judge(items, 0.3, VOTES).mean(-1)}
        rows = torch.arange(N_PROMPTS)
        for a in arms:
            pick = scores[a].argmax(1)                                         # [P] 每题挑一条
            res[a][n] = (float(items[rows, pick].mean()), float(length[rows, pick].mean()))
        pick = j3.argmax(1)
        gap[n] = float(j3[rows, pick].mean() - items[rows, pick].mean())
    print(f"\n---- best-of-N, {N_PROMPTS} 题; 每格 = 真实质量 / 平均长度 ----")
    print(f"{'N':>3}" + "".join(f"{a:>16}" for a in arms))
    for n in NS:
        print(f"{n:>3}" + "".join(f"{res[a][n][0]:>10.3f} /{res[a][n][1]:>4.0f}" for a in arms))
    print("rubric ε=0.3 挑中者的 judge 分 − 真实质量: " + ", ".join(f"N={n} {gap[n]:+.3f}" for n in NS))

    rm, r1, r3, rv = (res[a] for a in ("RM β=0.8", "rubric ε=0.1", "rubric ε=0.3", "ε=0.3 投票"))
    assert rm[64][1] > rm[1][1] + 40, f"β=0.8 的 RM 挑出的回复应随 N 变长: N=1 {rm[1][1]:.0f}, N=64 {rm[64][1]:.0f}"
    assert abs(r1[64][1] - r1[1][1]) < 15, f"rubric 挑出的长度应基本不变: N=1 {r1[1][1]:.0f}, N=64 {r1[64][1]:.0f}"
    assert rm[64][0] > r1[64][0], (
        f"分布内 RM 看得见全部条目, N=64 的真实质量应仍高于 rubric ε=0.1: RM {rm[64][0]:.3f}, rubric {r1[64][0]:.3f}")
    assert r3[64][0] < r1[64][0], f"judge 越不准收益越小: N=64 ε=0.3 {r3[64][0]:.3f} 应低于 ε=0.1 {r1[64][0]:.3f}"
    assert rv[64][0] > r3[64][0], f"投票后 N=64 的真实质量应高于单次判定: 投票 {rv[64][0]:.3f}, 单次 {r3[64][0]:.3f}"
    assert gap[64] > gap[1] + 0.1, (
        f"judge 的误差也会被利用: 挑中者 judge 分 − 真实质量应随 N 变大: N=1 {gap[1]:+.3f}, N=64 {gap[64]:+.3f}")

    # ---- RL: 策略可以离开基座分布 ----
    rewards = {"oracle": lambda i, L: i.mean(-1), "RM β=0.8": lambda i, L: features(i, L) @ w_biased,
               "rubric ε=0.1": lambda i, L: judge(i, 0.1).mean(-1), "rubric ε=0.3": lambda i, L: judge(i, 0.3).mean(-1)}
    logs = {a: rl(fn, w_biased) for a, fn in rewards.items()}
    print(f"\n---- RL {RL_STEPS} 步; 每格 = 真实质量 / 平均长度 / β=0.8 的 RM 分 ----")
    print(f"{'步':>4}" + "".join(f"{a:>22}" for a in rewards))
    for step in logs["oracle"]:
        print(f"{step:>4}" + "".join(f"{logs[a][step][0]:>10.3f} /{logs[a][step][1]:>4.0f} /{logs[a][step][2]:>5.2f}" for a in rewards))

    end, start = {a: logs[a][RL_STEPS] for a in rewards}, logs["RM β=0.8"][0]
    assert end["RM β=0.8"][1] > 590, f"β=0.8 的 RM 应把长度推到上限附近 (> 590), 实际 {end['RM β=0.8'][1]:.0f}"
    assert end["RM β=0.8"][0] < 0.85, f"RM 被钻空子后真实质量应停在 0.85 以下, 实际 {end['RM β=0.8'][0]:.3f}"
    assert end["RM β=0.8"][2] > start[2] + 1, f"RM 分应一路上涨: 第 0 步 {start[2]:.2f}, 末步 {end['RM β=0.8'][2]:.2f}"
    for a in ("rubric ε=0.1", "rubric ε=0.3"):
        assert end[a][0] > 0.95, f"{a}: RL 后真实质量应 > 0.95, 实际 {end[a][0]:.3f}"
        assert end[a][1] <= L_MAX, f"{a}: RL 后平均长度应不超过 {L_MAX}, 实际 {end[a][1]:.0f}"


if __name__ == "__main__":
    main()
