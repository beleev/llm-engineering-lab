#!/usr/bin/env python
"""
训练-推理不一致: 同一份权重, 推理引擎和训练器算出的 token 概率不一样; 按同策略公式算的梯度其实是异策略的。

    python -m llm_finetune.run_finetune.rollout_mismatch.train_rollout_mismatch

模拟的 "推理引擎" (Engine): 每次采样前从训练器拷贝最新权重 (版本完全相同), 只换数值路径:
    fp32       与训练器同一条路径, 当对照
    fp16       权重和激活用 fp16 算, logits 再转回 fp32 做 softmax
    bf16       同上, 换成 bf16 (尾数 7 位, fp16 有 10 位)
    bf16+int4  bf16 之外, 线性层权重再按 INT4 (g=32) 取整, 模拟低精度 rollout
训练器始终用 fp32 重算 log-prob。
实验设计:
    第一部分 概率比分布: sort 任务 SFT 150 步的 policy, 留出集 256 题 × 8 条, 用各引擎采样,
        比较 r = π_train / π_infer: |log r| 的均值与最大值、r 落在 [1/2, 2] 之外的 token 比例、序列级 ρ > 2 的比例,
        以及引擎认为不太可能 (π_infer < 0.1) 的 token 与很有把握 (π_infer > 0.9) 的 token 的 |log r| 之比。
    第二部分 梯度偏差 (精确, 无采样噪声): 留出集 512 题, 沿正确答案的前缀, 在回复的每个位置做一次单步决策:
        下一个 token 采对得 1 分。对词表求和得到每种估计量的期望梯度, 与真实同策略梯度比:
            真实    Σ_v π_train(v)·(r − b_train)·∇log π_train(v)
            估计    Σ_v π_infer(v)·w(v)·(r − b_infer)·∇log π_train(v),  w 由 is_weights 给出 (b 是组均值的期望)
        报相对偏差 ‖E[ĝ] − g‖ / ‖g‖。C=2 之外再用 C=1.1 看截断 / 屏蔽本身留下的偏差。
    第三部分 GRPO 训练: 同一个 SFT 起点, 每步 32 题 × G=8, 共 60 步, GRPO 默认配置 (μ=2, ε=0.2, lr=3e-4)。
        五臂: fp32 不校正 (对照) / bf16 不校正 / bf16+int4 不校正 / bf16+int4 + TIS / bf16+int4 + 序列级 MIS, C=2。
        每臂开始前重置种子。报训练器 (fp32) 在留出集上的采样 pass@1 和贪心 EM。
断言 (每条验证一个结论):
    1. 不一致从小到大: fp32 为 0, fp16 < bf16 < bf16+int4 (|log r| 均值)。fp16 比 bf16 小 4 倍以上。
    2. 不一致集中在低概率 token: bf16 下 π_infer < 0.1 的 token 的 |log r| 是 π_infer > 0.9 的 10 倍以上。
    3. 不校正时梯度有偏: 偏差 fp16 < bf16 < bf16+int4, 且 bf16+int4 的相对偏差 > 0.1。
    4. C=2 时 TIS / MIS 的偏差都不到不校正的 1/5; fp16 / bf16 下没有 r > 2 的 token, TIS 就是完整 IS, 偏差为 0。
    5. C=1.1 时 TIS 留下偏差但仍小于不校正; MIS 屏蔽掉的 token 太多, 偏差反而超过不校正 (阈值要放在真正的离群值上)。
    6. 训练: 每臂的留出集 pass@1 都比起点高 0.04 以上 (这个规模下不一致没有让训练失败), 其余差别只报告不断言。
简化: Engine 用整段前向算 infer log-prob, 真实引擎是增量解码 (KV cache) 时顺手记下的, 两者的数值路径还会再不同。
简化: 不一致来自 dtype 和权重取整; 真实系统里还有融合算子、不同的归约顺序和并行切分, 这里没有模拟。
依赖 llm_models: Engine 继承 GenerationMixin, 只需要 layers / max_len / forward(idx, attention_mask, cache)。
"""

import copy
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_finetune import (
    PromptDataGenerator, SeqTask, aggregate, completion_logprobs, completion_mask, group_advantages, make_config,
)
from llm_finetune.methods.is_correction import is_weights, mismatch_stats
from llm_finetune.methods.lora import ALL_LINEARS
from llm_finetune.methods.qat import quantize_dequantize
from llm_finetune.run_finetune.common import make_model, sft_warmup
from llm_models.utils.generation import GenerationMixin

SFT_STEPS, RL_STEPS, PROMPTS_PER_STEP, G, LR, CAP = 150, 60, 32, 8, 3e-4, 2.0
ENGINES = {"fp32": (torch.float32, None), "fp16": (torch.float16, None),
           "bf16": (torch.bfloat16, None), "bf16+int4": (torch.bfloat16, 4)}


class Engine(GenerationMixin, nn.Module):
    """模拟的推理引擎: 与训练器同一份权重, 换一条数值路径。forward 输出 fp32 logits (softmax 在 fp32 里做)。"""

    def __init__(self, policy: nn.Module, dtype: torch.dtype, weight_bits: Optional[int] = None) -> None:
        super().__init__()
        self.net = copy.deepcopy(policy).to(dtype)
        self.weight_bits = weight_bits
        self.layers, self.max_len = self.net.layers, self.net.max_len     # GenerationMixin 要用这两个
        self.sync(policy)

    @torch.no_grad()
    def sync(self, policy: nn.Module) -> None:
        """每次采样前调用: 拷贝训练器的最新权重 (load_state_dict 顺带转 dtype), 版本差为 0。"""
        self.net.load_state_dict(policy.state_dict())
        if self.weight_bits:
            for name, m in self.net.named_modules():
                if name.split(".")[-1] in ALL_LINEARS:
                    m.weight.copy_(quantize_dequantize(m.weight.float(), self.weight_bits, 32))

    def forward(self, idx: torch.Tensor, attention_mask=None, cache=None) -> torch.Tensor:
        return self.net(idx, attention_mask, cache=cache).float()


def ratio_report(policy, task: SeqTask) -> Dict[str, dict]:
    print("---- 1. 同一份权重, 两条数值路径的逐 token 概率比 r = π_train / π_infer ----")
    print(f"{'引擎':<11}{'|log r| 均值':>12}{'最大':>8}{'r∉[1/2,2]':>11}{'序列 ρ>2':>10}{'低概率/高概率':>14}")
    out, P = {}, task.prompt_len
    for name, (dtype, bits) in ENGINES.items():
        engine = Engine(policy, dtype, bits)
        torch.manual_seed(5)
        prompts = task.sample_prompts(256, "test").repeat_interleave(G, 0)
        seqs = engine.generate(prompts, task.response_len + 2, temperature=1.0)
        mask = completion_mask(seqs[:, P:]).float()
        with torch.no_grad():
            train_lp, infer_lp = completion_logprobs(policy, seqs, P), completion_logprobs(engine, seqs, P)
        s = mismatch_stats(train_lp, infer_lp, mask, CAP)
        gap, p = (train_lp - infer_lp).abs()[mask > 0], infer_lp.exp()[mask > 0]
        s["low_high"] = float(gap[p < 0.1].mean() / gap[p > 0.9].mean()) if name != "fp32" else float("nan")
        out[name] = s
        print(f"{name:<11}{s['mean_abs']:>12.5f}{s['max_abs']:>8.3f}{s['out_frac']:>11.2%}{s['seq_out_frac']:>10.2%}"
              f"{s['low_high']:>14.1f}")
    return out


def bias_report(policy, task: SeqTask) -> Dict[float, Dict[str, Dict[str, float]]]:
    print("\n---- 2. 梯度偏差 ‖E[ĝ] − g‖/‖g‖ (沿正确前缀逐位置精确求和, 无采样噪声) ----")
    torch.manual_seed(7)
    prompts = task.sample_prompts(512, "test")
    seqs = torch.cat([prompts, task.target(prompts)], dim=1)                       # [n, P+R] 正确前缀
    P, R = task.prompt_len, task.response_len
    r = F.one_hot(seqs[:, P:], task.vocab_size).float()                            # [n, R, V] 下一个 token 采对得 1 分

    def logits_of(model):
        return model(seqs[:, :-1])[:, P - 1:]                                      # [n, R, V] 每个回复位置的分布

    def grad_of(coef):
        policy.zero_grad()
        (coef * F.log_softmax(logits_of(policy), -1)).sum().backward()             # coef 无梯度: 期望梯度
        return torch.cat([p.grad.flatten() for p in policy.parameters() if p.grad is not None])

    with torch.no_grad():
        p = F.softmax(logits_of(policy), -1)
    g_true = grad_of(p * (r - (p * r).sum(-1, keepdim=True)))
    res: Dict[float, Dict[str, Dict[str, float]]] = {}
    for cap in (CAP, 1.1):
        res[cap] = {}
        for name in ("fp16", "bf16", "bf16+int4"):
            with torch.no_grad():
                q = F.softmax(logits_of(Engine(policy, *ENGINES[name])), -1)
            adv = r - (q * r).sum(-1, keepdim=True)
            res[cap][name] = {}
            for mode in ("none", "tis", "mis_token"):
                # 每个候选 token 一个权重 [n, R, V]。单步决策里 token 级与序列级相同, 所以只比 token 级的三种
                w = is_weights(p.log(), q.log(), torch.ones_like(p), mode, cap)
                g = grad_of(q * w * adv)
                res[cap][name][mode] = float((g - g_true).norm() / g_true.norm())
        print(f"C={cap}: " + " | ".join(f"{n} 不校正 {v['none']:.4f} TIS {v['tis']:.4f} MIS {v['mis_token']:.4f}"
                                       for n, v in res[cap].items()))
    return res


def grpo_arm(sft, task: SeqTask, engine_name: str, mode: str) -> Dict[str, float]:
    """一臂 GRPO: rollout 由 Engine 采样, 训练器 fp32 重算 old log-prob, 校正权重乘在裁剪目标外面。"""
    cfg = make_config("grpo", lr=LR, max_new=task.response_len + 2, group_size=G)
    torch.manual_seed(1)
    policy = copy.deepcopy(sft)
    engine = Engine(policy, *ENGINES[engine_name])
    opt = torch.optim.AdamW(policy.parameters(), lr=cfg.lr)
    prompts_gen, P = PromptDataGenerator(task, PROMPTS_PER_STEP), task.prompt_len
    gaps, masked = [], []
    for _ in range(RL_STEPS):
        engine.sync(policy)                                                         # 同版本: 每步先同步权重
        expanded = prompts_gen.generate_batch()["prompts"].repeat_interleave(G, dim=0)
        seqs = engine.generate(expanded, cfg.max_new, temperature=cfg.temperature)  # 样本来自 π_infer
        mask = completion_mask(seqs[:, P:]).float()
        adv = group_advantages(task.verify(expanded, seqs[:, P:]), G, cfg.std_norm).unsqueeze(1)
        with torch.no_grad():
            old_lp = completion_logprobs(policy, seqs, P, cfg.temperature)          # 训练器重算的 π_train
            infer_lp = completion_logprobs(engine, seqs, P, cfg.temperature)        # 引擎自己的 π_infer
        w = is_weights(old_lp, infer_lp, mask, mode, CAP)                           # ★ 不一致的校正
        gaps.append(float(((old_lp - infer_lp).abs() * mask).sum() / mask.sum()))
        masked.append(float(((w == 0).float() * mask).sum() / mask.sum()))
        for _ in range(cfg.inner_epochs):
            ratio = (completion_logprobs(policy, seqs, P, cfg.temperature) - old_lp).exp()
            surrogate = torch.min(ratio * adv, ratio.clamp(1 - cfg.clip_low, 1 + cfg.clip_high) * adv)
            loss = aggregate(-w * surrogate, mask, cfg.loss_agg)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
    return {"em": task.exact_match(policy), "pass1": task.exact_match(policy, n=512, temperature=1.0),
            "gap": sum(gaps) / len(gaps), "masked": sum(masked) / len(masked)}


def main() -> None:
    torch.manual_seed(0)
    task = SeqTask("sort")
    sft = make_model(task)
    em_sft = sft_warmup(sft, task, SFT_STEPS)
    pass1_sft = task.exact_match(sft, n=512, temperature=1.0)
    stats = ratio_report(sft, task)
    bias = bias_report(sft, task)

    print(f"\n---- 3. GRPO: {PROMPTS_PER_STEP} 题 × G={G} × {RL_STEPS} 步; 起点 贪心 EM {em_sft:.3f}, pass@1 {pass1_sft:.3f} ----")
    print(f"{'':<24}{'贪心 EM':>8}{'pass@1':>8}{'训练中 |log r|':>15}{'被屏蔽 token':>13}")
    arms = {}
    for engine_name, mode in [("fp32", "none"), ("bf16", "none"), ("bf16+int4", "none"),
                              ("bf16+int4", "tis"), ("bf16+int4", "mis_seq")]:
        r = arms[engine_name, mode] = grpo_arm(sft, task, engine_name, mode)
        print(f"{engine_name + ' ' + mode:<24}{r['em']:>9.3f}{r['pass1']:>8.3f}{r['gap']:>15.4f}{r['masked']:>13.2%}")

    m = {k: v["mean_abs"] for k, v in stats.items()}
    assert m["fp32"] == 0, f"fp32 引擎与训练器同一条路径, 概率比应恒为 1: |log r| 均值 {m['fp32']}"
    assert m["fp16"] < m["bf16"], f"fp16 的不一致应小于 bf16: {m['fp16']:.5f} vs {m['bf16']:.5f}"
    assert m["bf16"] < m["bf16+int4"], f"权重再量化, 不一致应更大: {m['bf16']:.5f} vs {m['bf16+int4']:.5f}"
    assert m["bf16"] > 4 * m["fp16"], f"fp16 多 3 位尾数, 不一致应小 4 倍以上: fp16 {m['fp16']:.5f}, bf16 {m['bf16']:.5f}"
    assert stats["bf16"]["low_high"] > 10, (
        f"不一致应集中在引擎认为不太可能的 token: 低概率 / 高概率 token 的 |log r| 之比 {stats['bf16']['low_high']:.1f}")
    b2 = bias[CAP]
    assert b2["fp16"]["none"] < b2["bf16"]["none"], f"不校正的偏差应随不一致变大: fp16 {b2['fp16']['none']:.4f}, bf16 {b2['bf16']['none']:.4f}"
    assert b2["bf16"]["none"] < b2["bf16+int4"]["none"], (
        f"不校正的偏差应随不一致变大: bf16 {b2['bf16']['none']:.4f}, bf16+int4 {b2['bf16+int4']['none']:.4f}")
    assert b2["bf16+int4"]["none"] > 0.1, f"bf16+int4 不校正的相对偏差应 > 0.1: {b2['bf16+int4']['none']:.4f}"
    for name, v in b2.items():
        for mode in ("tis", "mis_token"):
            assert v[mode] < v["none"] / 5, f"C={CAP} 时 {mode} 应把偏差压到不校正的 1/5 以下 ({name}): {v[mode]:.4f} vs {v['none']:.4f}"
    for name in ("fp16", "bf16"):
        assert b2[name]["tis"] < 1e-6, f"{name} 下没有 r > {CAP} 的 token, TIS 就是完整 IS, 偏差应为 0: {b2[name]['tis']:.2e}"
    for name in ("bf16", "bf16+int4"):
        v = bias[1.1][name]
        assert v["tis"] < v["none"], f"C=1.1 时 TIS 截断留下偏差, 但仍小于不校正 ({name}): {v['tis']:.4f} vs {v['none']:.4f}"
        assert v["mis_token"] > v["none"], (
            f"C=1.1 时 MIS 屏蔽掉太多 token, 偏差反而超过不校正 ({name}): {v['mis_token']:.4f} vs {v['none']:.4f}")
    for k, r in arms.items():
        assert r["pass1"] > pass1_sft + 0.04, f"{k}: 留出集 pass@1 应高于起点: {r['pass1']:.3f} vs {pass1_sft:.3f}"


if __name__ == "__main__":
    main()
