#!/usr/bin/env python
"""
Off-policy 蒸馏 (forward KL): teacher 学的是一个**有两种正确答案**的任务 (70% 排序 / 30% 照抄),
同样步数下比较 "只学采样出来的硬标签" 和 "学 teacher 的整个分布" —— 分别在 128 条固定数据 和 无限新数据 两种条件下。

    python -m llm_finetune.run_finetune.distill.train_distill

实验设计:
    数据: 每个 prompt 以 0.7 / 0.3 的概率配 sort / copy 的回复。同一个 prompt 有两个都算对的答案。
    teacher: 2 层 d=64, 在这份混合数据上训 800 步。
    student: 1 层 d=48, 比 teacher 小。每臂 300 步, 开始前重置种子 → 四个 student 初始化相同。
    2 × 2 共四臂:
                          硬标签 (α=1, 只有 CE)      软标签 (T=2, α=0.3)
        128 条固定数据     对照组                     实验组
        无限新数据         对照组                     实验组
    "128 条固定" = MixtureData(128, fixed=True): 第一个 batch 缓存下来, 每步都用这 128 条。
    硬标签那两臂也走 TeacherStudent + DistillLoss, 只是 α=1 让 KD 项的权重变成 0。
    验收 (evaluate) 全在留出 prompt 上:
        样本合格率    student 采样出的回复是两种正确答案之一的比例
        forward KL   在 teacher 的样本上量 KL(teacher‖student)
        reverse KL   在 student 自己的样本上量 KL(student‖teacher)
        两个 KL 都是回复段逐 token 的平均, 只算第一个 EOS (含) 之前的位置 (completion_kl)。
        logπ(sort 答案) / logπ(copy 答案)   student 给两种答案各分了多少概率
断言 (每条验证一个结论):
    1. 把 teacher 在 prompt 位置 (label = -100) 的 logits 全换成随机数, KD loss 不变。KD 项和 CE 项用的是同一个 mask。
    2. 128 条固定数据下, 软标签的留出集 forward KL 比硬标签低 0.05 以上。
       数据有限时, 一条软标签顶得上同一前缀的许多条采样硬标签。
    3. 把 student 样本里第一个 EOS 之后的 token 全换成随机数, evaluate 用的 KL 不变。EOS 之后的位置不进平均。
    不设断言的一格: 无限新数据下两者的差距。硬标签自己就能把分布采出来, 两臂都卡在 student 的容量上, 表里列出数值。
依赖 llm_models: student 直接用 LLaMA 的构造参数搭; train_on_policy_distill.py 会 import 本文件的 teacher / student / evaluate。
"""

from typing import Dict

import torch

from llm_models.models.language_models.llama import LLaMA
from llm_models.training.data import SyntheticDataGenerator

from llm_finetune import (
    DistillLoss, SeqTask, SFTLoss, TeacherStudent, completion_kl, completion_mask, compute_sequence_logprobs,
    count_parameters, make_labels,
)
from llm_finetune.run_finetune.common import fit, make_model

MAJOR, MINOR, P_MAJOR = SeqTask("sort"), SeqTask("copy"), 0.7
TEACHER_STEPS, STUDENT_STEPS, LR = 800, 300, 3e-3


class MixtureData(SyntheticDataGenerator):
    """每条样本以 0.7 / 0.3 的概率取 sort / copy 作为回复 —— 同一个 prompt 有两个都 "对" 的答案。"""

    def __init__(self, batch_size: int = 64, fixed: bool = False) -> None:
        self.batch_size, self.fixed = batch_size, fixed

    def _sample(self) -> Dict[str, torch.Tensor]:
        prompts = MAJOR.sample_prompts(self.batch_size)                                       # [B, P]
        pick_major = torch.rand(self.batch_size, 1) < P_MAJOR                                 # [B, 1] 整条回复一起选
        response = torch.where(pick_major, MAJOR.target(prompts), MINOR.target(prompts))      # [B, R]
        idx, labels = make_labels(torch.cat([prompts, response], dim=1), MAJOR.prompt_len)
        return {"idx": idx, "labels": labels}


def train_teacher() -> LLaMA:
    """在混合数据上训出 teacher, 返回时已置 eval。它学到的是 "两种答案各有多大概率"。"""
    teacher = make_model(MAJOR)                                # 2 层, d=64
    fit(teacher, MixtureData(), SFTLoss(), TEACHER_STEPS, LR, log_interval=TEACHER_STEPS)
    return teacher.eval()


def make_student() -> LLaMA:
    """比 teacher 小的 student: 1 层, d_model=48。两个蒸馏脚本比的是: 容量不够时, 它保住什么、放弃什么。"""
    return LLaMA(vocab_size=MAJOR.vocab_size, d_model=48, n_heads=4, num_kv_heads=2, num_layers=1, max_len=32)


@torch.no_grad()
def evaluate(student: LLaMA, teacher: LLaMA, n: int = 512) -> Dict[str, float]:
    """全部在留出 prompt 上。forward_kl 在 teacher 的样本上量, reverse_kl 在 student 自己的样本上量。"""
    state = torch.get_rng_state()
    torch.manual_seed(99)
    prompts = MAJOR.sample_prompts(n, "test")
    P, R = MAJOR.prompt_len, MAJOR.response_len
    own = student.generate(prompts, R, temperature=1.0)                       # [n, P+R] student 的样本
    ref = teacher.generate(prompts, R, temperature=1.0)                       # [n, P+R] teacher 的样本
    torch.set_rng_state(state)

    completion = own[:, P:]                                                   # [n, R] student 的回复
    # [n] 回复与 sort 答案或 copy 答案整条一致
    valid = (completion == MAJOR.target(prompts)).all(1) | (completion == MINOR.target(prompts)).all(1)
    # 两个 KL 都只在回复段上算, 且只算第一个 EOS (含) 之前的位置: 回复提前说了 EOS, 之后采到的 token 不算回复
    out = {
        "valid": float(valid.float().mean()),                                 # student 采样出来的回复是两种正确答案之一
        "reverse_kl": completion_kl(student, teacher, own, P),
        "forward_kl": completion_kl(teacher, student, ref, P),
    }
    for name, task in (("logp_major", MAJOR), ("logp_minor", MINOR)):         # student 给两种答案各分了多少概率
        idx, labels = make_labels(torch.cat([prompts, task.target(prompts)], dim=1), P)
        out[name] = float(compute_sequence_logprobs(student(idx), labels).mean())
    return out


def report(rows: Dict[str, Dict[str, float]]) -> None:
    """把 {臂的名字: evaluate 的结果} 打成一张表。"""
    print(f"\n{'':<22}{'样本合格率':>8}{'forward KL':>12}{'reverse KL':>12}{'logπ(sort 答案)':>16}{'logπ(copy 答案)':>16}")
    for name, m in rows.items():
        print(f"{name:<22}{m['valid']:>12.3f}{m['forward_kl']:>12.3f}{m['reverse_kl']:>12.3f}"
              f"{m['logp_major']:>16.2f}{m['logp_minor']:>16.2f}")


def main() -> None:
    torch.manual_seed(0)
    teacher = train_teacher()

    # ---- KD 项必须和 CE 项用同一个 mask: 把 teacher 在 prompt 位置的 logits 全打乱, loss 不应变化 ----
    batch = MixtureData(8).generate_batch()
    loss_fn = DistillLoss(temperature=2.0, alpha=0.3)
    # s_logits: 随机的 student 输出 [8, T, V=16]; t_logits: teacher 的真实输出 [8, T, V]
    s_logits, t_logits = torch.randn(8, batch["idx"].size(1), 16), teacher(batch["idx"]).detach()
    # label = -100 的位置 ([8, T] → [8, T, 1] 广播到词表维) 换成随机数, 其余位置不动
    scrambled = torch.where((batch["labels"] == -100).unsqueeze(-1), torch.randn_like(t_logits), t_logits)
    kd = [float(loss_fn.compute({"student": s_logits, "teacher": t}, batch["labels"])["kd_loss"]) for t in (t_logits, scrambled)]
    assert abs(kd[0] - kd[1]) < 1e-6, (
        f"KD 项泄漏到了 label = -100 的位置: 打乱这些位置的 teacher logits 后 KD loss 变了 {abs(kd[0] - kd[1]):.1e}")

    rows = {"teacher": evaluate(teacher, teacher)}
    for data_name, make_data in [("128 条固定", lambda: MixtureData(128, fixed=True)), ("无限新数据", MixtureData)]:
        for name, loss in [("硬标签", DistillLoss(1.0, alpha=1.0)), ("软标签 T=2", loss_fn)]:
            torch.manual_seed(1)
            student = make_student()
            fit(TeacherStudent(student, teacher), make_data(), loss, STUDENT_STEPS, LR, log_interval=STUDENT_STEPS)
            rows[f"{data_name} / {name}"] = evaluate(student, teacher)
    report(rows)
    print(f"参数量: teacher {count_parameters(teacher)['total']:,} → student {count_parameters(student)['total']:,}")

    # 数据无限时硬标签自己就能把分布采出来, 两者都卡在 student 的容量上 —— 差距消失 (数值在表里, 不设断言)。
    assert rows["128 条固定 / 软标签 T=2"]["forward_kl"] < rows["128 条固定 / 硬标签"]["forward_kl"] - 0.05, (
        f"数据有限时, 一条软标签顶得上同一前缀的许多条采样硬标签, 留出集上应更接近 teacher "
        f"(forward KL 低 0.05 以上): 软标签 {rows['128 条固定 / 软标签 T=2']['forward_kl']:.3f}, "
        f"硬标签 {rows['128 条固定 / 硬标签']['forward_kl']:.3f}")

    # ---- evaluate 的 KL 只算回复里第一个 EOS (含) 之前: 把 EOS 之后的 token 换成随机数, KL 不应变化 ----
    # 放在表格之后, 不扰动上面各臂的随机流。用最后一个 student: 它常提前说 EOS, 样本里有 EOS 之后的位置可测
    P = MAJOR.prompt_len
    seqs = student.generate(MAJOR.sample_prompts(256, "test"), MAJOR.response_len, temperature=1.0)   # [256, P+R]
    after_eos = ~completion_mask(seqs[:, P:])                                                         # [256, R]
    noisy = seqs.clone()
    # EOS 之后换成随机内容 token (取 [3, V)), 之前的不动
    noisy[:, P:] = torch.where(after_eos, torch.randint_like(seqs[:, P:], 3, MAJOR.vocab_size), seqs[:, P:])
    kl = [completion_kl(student, teacher, s, P) for s in (seqs, noisy)]
    assert after_eos.any(), "student 的样本里应有提前说 EOS 的回复, 否则这条检查测不到东西"
    assert abs(kl[0] - kl[1]) < 1e-6, (
        f"evaluate 的 KL 泄漏到了 EOS 之后的位置: 打乱这些 token 后 KL 变了 {abs(kl[0] - kl[1]):.1e}")


if __name__ == "__main__":
    main()
