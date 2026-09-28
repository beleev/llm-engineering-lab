"""
run_finetune 下所有脚本共用的五件事: 小模型、通用 Trainer 的一行封装、SFT 热身、会 copy 的预训练基座、留出集偏好准确率。

依赖 llm_models 的三处细节, 训练脚本的断言都建立在它们之上:
    - Trainer 的第 1 步 lr = 0 (线性 warmup 从 0 起) → `fit` 返回的 hist[0] 是 "还没更新过" 的模型的 loss。
      所以脚本可以断言: 第 1 步 loss ≈ ln V (SFT) 或 ≈ ln 2 (RM / DPO)。
    - Trainer 每步 `model(**batch)`: batch 里除 "labels" 外的键必须正好是 model.forward 的形参名。
    - `LLaMA.generate` 在 no_grad 下采样, 结束时恢复调用前的 train / eval 状态。
"""

from typing import Dict, List

import torch
import torch.nn as nn

from llm_models.models.language_models.llama import LLaMA
from llm_models.training import Trainer, TrainingConfig
from llm_models.training.data import SyntheticDataGenerator
from llm_models.training.loss import LossComputer

from llm_finetune.data import InstructionDataGenerator, SeqTask
from llm_finetune.methods.sft import SFTLoss

# d_model=64 的模型每个 matmul 只有几微秒, 多线程的同步开销反而比计算大: 单线程实测快约 2 倍
torch.set_num_threads(1)


def make_model(task: SeqTask, d_model: int = 64, num_layers: int = 2) -> LLaMA:
    """
    全章共用的小 LLaMA: 默认 2 层, d_model=64, 4 个注意力头 (KV 头 2 个, 即 GQA)。
    task 只用到 task.vocab_size。max_len=32: 本章最长的序列是 PRM 的算术链 (23 个 token), 放得下。
    """
    return LLaMA(vocab_size=task.vocab_size, d_model=d_model, n_heads=4, num_kv_heads=2,
                 num_layers=num_layers, max_len=32)


def fit(model: nn.Module, data: SyntheticDataGenerator, loss: LossComputer, steps: int,
        lr: float, log_interval: int = 100) -> List[Dict[str, float]]:
    """
    通用 Trainer 跑 steps 步。model 可以是裸 LM, 也可以是 PairwiseForward / TeacherStudent 这类包装。

    返回按 log_interval 采样的指标列表: hist[0] 是第 1 步, hist[-1] 是最后一步 (每项是 {指标名: float})。
    log_interval=steps 时列表只有这两项。warmup 占前 5% 的步数, 至少 1 步。
    """
    cfg = TrainingConfig(learning_rate=lr, num_steps=steps, warmup_steps=max(1, steps // 20),
                         log_interval=log_interval)
    return Trainer(model, cfg, data, loss).train()


def sft_warmup(model: nn.Module, task: SeqTask, steps: int, lr: float = 3e-3) -> float:
    """偏好对齐 / RL 都从 SFT 过的模型出发 (真实流水线也是 SFT → DPO / GRPO)。返回留出集 exact-match。"""
    fit(model, InstructionDataGenerator(task), SFTLoss(), steps, lr, log_interval=steps)   # 原地训练 model
    return task.exact_match(model)


def pretrained_base(steps: int = 150) -> LLaMA:
    """PEFT 脚本共用的 "预训练基座": 已经会 copy 任务 (留出集 EM = 1), 接下来要被适配到 sort。"""
    task = SeqTask("copy")
    base = make_model(task)
    em = sft_warmup(base, task, steps)
    assert em > 0.95, f"基座没训好: copy EM {em:.3f}"
    return base


@torch.no_grad()
def preference_accuracy(policy: nn.Module, task: SeqTask, n: int = 256, seed: int = 4321) -> Dict[str, float]:
    """
    留出集偏好对上, policy 给正确回复的概率高于损坏回复的比例 (直接比 log π(y|x), 不依赖任何 ref)。

    返回 {"accuracy": 上述比例, "logp_chosen": 正确回复的平均 log π, "logp_rejected": 损坏回复的平均 log π}。
    后两个是整条回复的 Σ_t log p (不除长度), 单位 nat。
    """
    # 函数内 import: methods.dpo 与本文件不互相依赖, 只有算这个指标时才用到
    from llm_finetune.data import PreferenceDataGenerator
    from llm_finetune.methods.dpo import compute_sequence_logprobs

    state = torch.get_rng_state()
    torch.manual_seed(seed)                                   # 固定种子: 每次评的是同一批留出偏好对
    batch = PreferenceDataGenerator(task, n, split="test").generate_batch()
    torch.set_rng_state(state)
    was_training = policy.training
    policy.eval()                                             # 评估时关 dropout; 算完恢复原状态
    # logp[k]: [n] 每条序列回复段的 Σ_t log p
    logp = {k: compute_sequence_logprobs(policy(batch[f"{k}_input_ids"]), batch["labels"][k])
            for k in ("chosen", "rejected")}
    policy.train(was_training)
    return {"accuracy": float((logp["chosen"] > logp["rejected"]).float().mean()),
            "logp_chosen": float(logp["chosen"].mean()), "logp_rejected": float(logp["rejected"].mean())}
