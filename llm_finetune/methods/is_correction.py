"""
训练-推理不一致的重要性采样校正: TIS 与 MIS (Yao et al. 2025-08; Liu et al. 2025-09)

是什么: RL 里 rollout 由推理引擎采样, 梯度由训练器算。两边加载的是同一份权重, 但推理引擎用融合算子、低精度、
        不同的 softmax 实现, 算出的 token 概率 π_infer 和训练器重算的 π_train 不一样。
        样本其实来自 π_infer, 按同策略公式算的梯度就成了异策略梯度, 有偏。
        这和异步 RL 的 "策略版本陈旧" (run_finetune/async_rollout) 不是一回事: 那边是 v1 与 v3 两个版本的差,
        这边版本完全相同, 差在数值路径。
解决什么: 把每个 token (或每条序列) 乘一个重要性权重 π_train / π_infer, 把行为分布从 π_infer 换回 π_train。
核心公式 (r_t = π_train(o_t) / π_infer(o_t), 由两边的 log-prob 相减再取 exp):
    TIS (截断 IS):           w_t = min(r_t, C)                       权重有上限, 方差有界, 截断部分留下一点偏差
    token 级 MIS (屏蔽 IS):  w_t = r_t · 1[r_t ≤ C]                  比率过大的 token 整个丢掉, 不截断
    序列级 MIS:              w_t = ρ · 1[ρ ≤ C],  ρ = Π_t r_t        比率过大的序列整条丢掉 (Liu et al. 认为最稳的一种)
    权重乘在 PPO / GRPO 的裁剪目标外面: loss_t = −w_t · min(ρ_t A, clip(ρ_t) A), 其中 ρ_t = π_θ / π_train_old 不变。
与论文 / 真实系统的差异:
    简化: 只实现上面三种。Liu et al. 还提出几何平均版 (Geo-MIS, ρ^{1/T} 双侧阈值), 这里没有实现。
    简化: C 取 2 (verl 中 TIS 阈值的默认值); 论文里各实验的 C 不尽相同。
    简化: 本库的 "推理引擎" 是训练脚本里模拟出来的数值路径 (bf16 / fp16 / 量化权重), 不是 vLLM / SGLang。
读代码时盯住: 权重全部 detach —— 它只改变每个样本的分量, 不参与求导。
"""

import torch

MODES = ("none", "tis", "mis_token", "mis_seq")


def is_weights(train_logp: torch.Tensor, infer_logp: torch.Tensor, mask: torch.Tensor,
               mode: str, cap: float = 2.0) -> torch.Tensor:
    """
    train_logp / infer_logp / mask [N, C] → 每个 token 的校正权重 [N, C] (无梯度)。
    mode="none": 全 1, 即把推理引擎的样本当成训练器自己采的 (多数框架的默认行为)。
    """
    if mode not in MODES:
        raise ValueError(f"未知校正方式 {mode}, 可选 {MODES}")
    log_r = (train_logp - infer_logp).detach() * mask                 # [N, C] EOS 之后置 0, 即 r = 1
    if mode == "none":
        return torch.ones_like(log_r)
    if mode == "tis":
        return log_r.exp().clamp(max=cap)
    if mode == "mis_token":
        r = log_r.exp()
        return r * (r <= cap)
    seq_r = log_r.sum(dim=1, keepdim=True).exp()                      # [N, 1] ρ = Π_t r_t, 在 log 域累加
    return (seq_r * (seq_r <= cap)).expand_as(log_r)                  # 整条序列共用一个权重


def mismatch_stats(train_logp: torch.Tensor, infer_logp: torch.Tensor, mask: torch.Tensor,
                   cap: float = 2.0) -> dict:
    """
    逐 token 概率比的分布摘要。只统计 mask 内的 token。
    返回 {"mean_abs": |log r| 的均值, "max_abs": 最大值, "out_frac": r 落在 [1/C, C] 之外的比例,
          "seq_out_frac": 序列级 ρ 超过 C 的序列比例}。
    """
    log_r = (train_logp - infer_logp) * mask
    tok = log_r[mask > 0]
    seq = log_r.sum(dim=1)
    return {"mean_abs": float(tok.abs().mean()), "max_abs": float(tok.abs().max()),
            "out_frac": float((tok.abs() > torch.log(torch.tensor(cap))).float().mean()),
            "seq_out_frac": float((seq > torch.log(torch.tensor(cap))).float().mean())}
