"""按学习路径顺序跑完所有脚本; 任何一个 assert 失败都会让整体失败。 python -m llm_finetune.run_all"""
from __future__ import annotations

import importlib
import time

# 顺序 = 学习路径。每个名字对应 run_finetune/<name>/train_<name>.py 里的 main()
SCRIPTS = ["sft", "lora", "dora", "qlora", "merge", "rm", "prm", "dpo", "kto", "simpo_orpo", "rlaif", "ppo", "grpo", "distill", "on_policy_distill"]


def main() -> None:
    for name in SCRIPTS:
        t0 = time.perf_counter()
        print(f"\n{'#' * 20} {name} {'#' * 20}")
        importlib.import_module(f"llm_finetune.run_finetune.{name}.train_{name}").main()
        print(f"[{name}] 通过, {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
