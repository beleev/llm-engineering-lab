#!/usr/bin/env python
"""
推理模型的训练面: 长 CoT 蒸馏、思考 / 非思考模式融合、思考长度控制。任务是 K 步模 10 算术链 (与 prm 同一种算术)。

    python -m llm_finetune.run_finetune.reasoning.train_reasoning

任务与格式 (vocab 24, 序列最长 26 个 token):
    prompt:  a0 op1 k1 … opK kK MODE SEP                MODE ∈ {/think, /no_think, B0 … BK}
    思考:    op1 k1 v1 … opK kK vK </think> ans EOS     每步一行 "op k 结果", 老师 (程序求解器) 写的完整轨迹
    非思考:  </think> ans EOS                           空的思考块, 与 Qwen3 非思考模式的格式一致
    预算 b:  前 b 步 </think> ans EOS                   只写前 b 步, 剩下 K−b 步要在一次前向里算完
    ans = vK。留出集 = (a0 + Σk) ≡ 0 (mod 5) 的题, 与训练集不相交。
实验设计 (所有模型同结构: make_model 的 2 层 LLaMA; batch 64, lr=3e-3, 种子相同):
    1. 只答案 vs 长轨迹: 同样的题, 各从头 SFT 1500 步。一个模型只学 "</think> ans EOS",
       一个学完整轨迹 (R1 的蒸馏做法, 小模型直接学老师的推理过程)。
    2. 模式融合: 在长轨迹模型上继续 SFT 500 步, 数据是两种格式各一半, 靠 prompt 里的 /think 或 /no_think 切换
       (Qwen3 的第 3 阶段也是在推理模型上继续 SFT)。
    3. 长度控制, 两种做法:
        - 预算 token: 在长轨迹模型上继续 SFT 500 步, b 从 0…K 均匀取, prompt 里写 B_b, 回复只写前 b 步
          (思路同 L1 的 "Think for N tokens", 但用 SFT 学)。
        - 强制截断 (s1 的 budget forcing): 拿实验 1 的长轨迹模型, 推理时写完 b 步就硬插 </think>, 让它直接作答。
    指标都在留出集上, 贪心解码: 正确率 = </think> 后面那个 token 等于 vK; 输出 token 数 = 到 EOS (含) 为止。
断言 (每条验证一个结论):
    1. 长轨迹 SFT 的正确率比只答案 SFT 高 0.3 以上; 代价是输出 token 数多 4 倍以上。
    2. 融合模型: /think 下正确率与单独的长轨迹模型相差不到 0.1; /no_think 下平均输出恰好 3 个 token。
    3. 融合模型: /think 下的正确率比 /no_think 高 0.3 以上 (同一组权重, 开关决定算多少)。
    4. 预算 token 模型: 每档都按预算写 (平均长度 = 3b + 3); 正确率在 b=K 时比 b=0 高 0.3 以上。
    5. 预算 token 模型: 0 < b < K 的每一档都不比 b=0 高出 0.05。这个规模下取舍是台阶: 少写一步就和一步不写差不多
       (MIX_STEPS 加到 3000 结果不变, 不是训练不够)。
    6. 强制截断: b=K 时等于长轨迹模型本身; b < K 时正确率 < 0.15 (接近随机猜 0.1), 也低于同一档的预算 token 模型。
简化: 老师是程序, 轨迹无噪声; 真实蒸馏用大模型采样的长 CoT, 还要过滤错误答案。
简化: 长度控制用 SFT 学预算 token; L1 用 RL 学, 奖励 = 答对 − α·|目标长度 − 实际长度|。
依赖 llm_models: `LLaMA.generate(prompts, max_new_tokens, temperature=0)` 贪心解码; make_model 的 max_len=32 放得下 26 个 token。
"""

import copy
from typing import Dict, Optional

import torch

from llm_finetune.data.tasks import EOS, PAD, SEP, make_labels
from llm_finetune.methods.prm import DIGIT0, MOD, PLUS, apply_op
from llm_finetune.methods.sft import SFTLoss
from llm_finetune.run_finetune.common import fit, make_model
from llm_models.training.data import SyntheticDataGenerator

K = 4                                             # 算术链步数
THINK, NO_THINK, END_THINK = 16, 17, 18           # /think  /no_think  </think>
BUDGET0 = 19                                      # 预算 token B_b = 19 + b, b ∈ 0…K
STEPS, MIX_STEPS, LR = 1500, 500, 3e-3       # 从头 SFT 的步数; 在长轨迹模型上继续混训的步数


class ArithCoT:
    """K 步模 10 算术链, 带模式 token。vocab_size / prompt_len 供 make_model 和 make_labels 用。"""

    vocab_size = BUDGET0 + K + 1                  # 24
    prompt_len = 2 * K + 3                        # a0, (op, k)×K, MODE, SEP
    response_len = 3 * K + 3                      # 最长回复: 3K 个思考 token + </think> ans EOS

    def sample(self, n: int, split: str = "train") -> Dict[str, torch.Tensor]:
        """n 道题 → {"a0" [n], "ops" [n, K], "ks" [n, K], "vals" [n, K]}。vals 是老师逐步写下的正确值。"""
        keep = []
        while sum(len(x) for x in keep) < n:
            prob = torch.cat([torch.randint(0, MOD, (4 * n, 1)), torch.randint(1, MOD, (4 * n, K)),
                              torch.randint(0, 3, (4 * n, K))], dim=1)          # [4n, 1+2K]: a0 | ks | ops
            is_test = (prob[:, 0] + prob[:, 1:K + 1].sum(1)) % 5 == 0
            keep.append(prob[is_test if split == "test" else ~is_test])
        prob = torch.cat(keep)[:n]
        a0, ks, ops = prob[:, 0], prob[:, 1:K + 1], prob[:, K + 1:]
        vals, v = torch.zeros(n, K, dtype=torch.long), a0
        for i in range(K):
            v = apply_op(ops[:, i], v, ks[:, i])
            vals[:, i] = v
        return {"a0": a0, "ops": ops, "ks": ks, "vals": vals}

    def prompts(self, prob: Dict[str, torch.Tensor], mode: torch.Tensor) -> torch.Tensor:
        """mode [n] 模式 token → [n, P]。"""
        cols = [prob["a0"] + DIGIT0]
        for i in range(K):
            cols += [prob["ops"][:, i] + PLUS, prob["ks"][:, i] + DIGIT0]
        return torch.stack(cols + [mode, torch.full_like(mode, SEP)], dim=1)

    def responses(self, prob: Dict[str, torch.Tensor], shown: torch.Tensor) -> torch.Tensor:
        """shown [n] 每条写出前几步 (0…K) → [n, 3K+3] 右 pad: 前 shown 步, </think>, ans, EOS。"""
        n = len(shown)
        steps = torch.stack([prob["ops"] + PLUS, prob["ks"] + DIGIT0, prob["vals"] + DIGIT0], dim=2).view(n, 3 * K)
        out = torch.full((n, self.response_len), PAD)
        for i in range(n):                                       # 每条长度不同, 逐条拼; n ≤ 64, 不是瓶颈
            b = int(shown[i])
            tail = torch.tensor([END_THINK, int(prob["vals"][i, -1]) + DIGIT0, EOS])
            out[i, : 3 * b + 3] = torch.cat([steps[i, : 3 * b], tail])
        return out


class ChainData(SyntheticDataGenerator):
    """
    每步一批 SFT 数据。budgets: 这一批里每条从中均匀抽一个 b; mode_of(b) 给出 prompt 里的模式 token。
    例: 只答案 = budgets (0,), 长轨迹 = (K,), 融合 = (0, K), 预算 token = (0, 1, …, K)。
    """

    fixed = False

    def __init__(self, task: ArithCoT, budgets, mode_of, batch_size: int = 64) -> None:
        self.task, self.budgets, self.mode_of, self.batch_size = task, torch.tensor(budgets), mode_of, batch_size

    def _sample(self) -> Dict[str, torch.Tensor]:
        prob = self.task.sample(self.batch_size)
        b = self.budgets[torch.randint(0, len(self.budgets), (self.batch_size,))]       # [B]
        mode = torch.tensor([self.mode_of(int(x)) for x in b])
        seq = torch.cat([self.task.prompts(prob, mode), self.task.responses(prob, b)], dim=1)
        idx, labels = make_labels(seq, self.task.prompt_len)
        return {"idx": idx, "labels": labels}


def think_mode(b: int) -> int:
    return THINK if b == K else NO_THINK


def budget_mode(b: int) -> int:
    return BUDGET0 + b


@torch.no_grad()
def evaluate(model, task: ArithCoT, mode: int, force_after: Optional[int] = None, n: int = 512) -> Dict[str, float]:
    """
    留出集贪心解码 → {"acc", "tokens"}。force_after=b: s1 式强制截断, 先只让模型写 3b 个 token, 再硬插 </think>, 让它作答。
    """
    state = torch.get_rng_state()
    torch.manual_seed(1234)                                       # 每次评同一批题
    prob = task.sample(n, "test")
    torch.set_rng_state(state)
    P, R = task.prompt_len, task.response_len
    prompts = task.prompts(prob, torch.full((n,), mode))
    if force_after is None:
        out = model.generate(prompts, R, temperature=0)[:, P:]                          # [n, R]
    else:
        head = model.generate(prompts, 3 * force_after, temperature=0) if force_after else prompts
        forced = torch.cat([head, torch.full((n, 1), END_THINK)], dim=1)               # ★ 硬插 </think>
        out = model.generate(forced, 2, temperature=0)[:, P:]                           # 再写 ans EOS
    end = out == END_THINK
    pos = end.float().argmax(1)                                                         # 第一个 </think> 的位置
    ans = out.gather(1, (pos + 1).clamp(max=out.size(1) - 1).unsqueeze(1)).squeeze(1)
    ok = end.any(1) & (ans == prob["vals"][:, -1] + DIGIT0)
    eos = out == EOS
    tokens = torch.where(eos.any(1), eos.float().argmax(1) + 1, torch.full_like(pos, out.size(1)))
    return {"acc": float(ok.float().mean()), "tokens": float(tokens.float().mean())}


def train(task: ArithCoT, budgets, mode_of, init: Optional[torch.nn.Module] = None) -> torch.nn.Module:
    """init=None: 从头训 STEPS 步; 否则在 init 的副本上继续训 MIX_STEPS 步 (Qwen3 的模式融合也是在推理模型上继续 SFT)。"""
    torch.manual_seed(0)
    model = make_model(task) if init is None else copy.deepcopy(init)
    steps = STEPS if init is None else MIX_STEPS
    fit(model, ChainData(task, budgets, mode_of), SFTLoss(), steps, LR, log_interval=steps)
    return model


def main() -> None:
    task = ArithCoT()
    print(f"任务: {K} 步模 10 算术链; 只答案 / 长轨迹模型从头 SFT {STEPS} 步, 融合 / 预算模型在长轨迹模型上再混训 {MIX_STEPS} 步; "
          f"batch 64, 指标是留出集贪心解码")

    # ---- 1. 只答案 vs 长轨迹 ----
    ans_only, cot = train(task, (0,), think_mode), train(task, (K,), think_mode)
    r_ans, r_cot = evaluate(ans_only, task, NO_THINK), evaluate(cot, task, THINK)
    print(f"\n---- 1. 只答案 SFT vs 长轨迹 SFT ----")
    print(f"{'':<14}{'正确率':>8}{'输出 token':>12}")
    for name, r in (("只答案", r_ans), ("长轨迹", r_cot)):
        print(f"{name:<14}{r['acc']:>9.3f}{r['tokens']:>12.1f}")

    # ---- 2. 模式融合 ----
    fused = train(task, (0, K), think_mode, init=cot)
    r_on, r_off = evaluate(fused, task, THINK), evaluate(fused, task, NO_THINK)
    print(f"\n---- 2. 一个模型混训 /think 与 /no_think ----")
    print(f"{'':<14}{'正确率':>8}{'输出 token':>12}")
    for name, r in (("/think", r_on), ("/no_think", r_off)):
        print(f"{name:<14}{r['acc']:>9.3f}{r['tokens']:>12.1f}")

    # ---- 3. 长度控制 ----
    budget = train(task, tuple(range(K + 1)), budget_mode, init=cot)
    curve = {b: evaluate(budget, task, budget_mode(b)) for b in range(K + 1)}
    forced = {b: evaluate(cot, task, THINK, force_after=b) for b in range(K + 1)}
    print(f"\n---- 3. 长度控制: 预算 b 步 (输出 3b+3 个 token) ----")
    print(f"{'b':<4}{'预算 token 正确率':>16}{'输出 token':>12}{'强制截断正确率':>16}{'输出 token':>12}")
    for b in range(K + 1):
        print(f"{b:<4}{curve[b]['acc']:>16.3f}{curve[b]['tokens']:>12.1f}{forced[b]['acc']:>16.3f}{forced[b]['tokens']:>12.1f}")

    assert r_cot["acc"] > r_ans["acc"] + 0.3, f"长轨迹 SFT 应明显更准: 长轨迹 {r_cot['acc']:.3f} vs 只答案 {r_ans['acc']:.3f}"
    assert r_cot["tokens"] > 4 * r_ans["tokens"], (
        f"长轨迹的代价是输出变长 (4 倍以上): 长轨迹 {r_cot['tokens']:.1f} vs 只答案 {r_ans['tokens']:.1f} 个 token")
    assert abs(r_on["acc"] - r_cot["acc"]) < 0.1, (
        f"融合模型的 /think 应接近单独的长轨迹模型: 融合 {r_on['acc']:.3f} vs 长轨迹 {r_cot['acc']:.3f}")
    assert r_off["tokens"] == 3.0, f"/no_think 应只输出 </think> ans EOS 三个 token: 平均 {r_off['tokens']:.2f}"
    assert r_on["acc"] > r_off["acc"] + 0.3, (
        f"同一组权重, /think 应比 /no_think 准: /think {r_on['acc']:.3f}, /no_think {r_off['acc']:.3f}")
    for b in range(K + 1):
        assert abs(curve[b]["tokens"] - (3 * b + 3)) < 0.05, (
            f"预算 token 模型应按预算写: b={b} 平均 {curve[b]['tokens']:.2f} 个 token, 预算 {3 * b + 3}")
    assert curve[K]["acc"] > curve[0]["acc"] + 0.3, (
        f"预算写满时应比不写准: b={K} {curve[K]['acc']:.3f} vs b=0 {curve[0]['acc']:.3f}")
    for b in range(1, K):
        assert curve[b]["acc"] < curve[0]["acc"] + 0.05, (
            f"这个规模下取舍是台阶, 不是斜坡: 少写一步就和一步不写差不多, b={b} {curve[b]['acc']:.3f} vs b=0 {curve[0]['acc']:.3f}")
    assert forced[K] == r_cot, f"b=K 的强制截断应等于不截断: {forced[K]} vs {r_cot}"
    for b in range(K):
        assert forced[b]["acc"] < 0.15, f"b={b}: 强制截断后应接近随机猜 (0.1): {forced[b]['acc']:.3f}"
        assert forced[b]["acc"] < curve[b]["acc"], (
            f"b={b}: 没学过写一半就作答的模型, 截断后应不如预算 token 模型: "
            f"截断 {forced[b]['acc']:.3f} vs 预算 token {curve[b]['acc']:.3f}")

if __name__ == "__main__":
    main()
