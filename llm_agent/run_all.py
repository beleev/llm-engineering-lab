"""按学习路径顺序跑完所有默认 demo。每个 demo 都用 assert 自检, 任何断言失败都会让本脚本非零退出。

m15 (真实模型) 需要可选依赖和 API key, 不在此列: python -m llm_agent.m15_claude_api.demo
排除它还有一个原因: 环境里有 key 时 m15 会发真实请求并计费。这里列出的 demo 都不联网。
默认路径使用标准库与 numpy。--with-training 还运行 m32 / m37 / m44 / m45 / m46 的 CPU 训练，需要 torch。
"""

from __future__ import annotations

import importlib
import argparse

DEMOS = [
    "llm_agent.m01_agent_loop.demo",
    "llm_agent.m02_tool_use.demo",
    "llm_agent.m03_permissions.demo",
    "llm_agent.m04_context_memory.demo",
    "llm_agent.m05_extensibility.demo",
    "llm_agent.m06_persistence_resume.demo",
    "llm_agent.m07_subagents.demo",
    "llm_agent.m08_retrieval.demo",
    "llm_agent.m09_mcp.demo",
    "llm_agent.m10_planning.demo",
    "llm_agent.m11_orchestrator.demo",
    "llm_agent.m12_guardrails.demo",
    "llm_agent.m13_evals.demo",
    "llm_agent.m14_context_engineering.demo",
    "llm_agent.m16_rag.demo",
    "llm_agent.m17_computer_use.demo",
    "llm_agent.m18_a2a.demo",
    "llm_agent.m19_prompt_caching.demo",
    "llm_agent.full_loop.demo",
    "llm_agent.m15_claude_api.stream_demo",
    "llm_agent.m20_memory_lifecycle.demo",
    "llm_agent.m21_completion_verification.demo",
    "llm_agent.m22_runtime_reliability.demo",
    "llm_agent.m23_context_cache.demo",
    "llm_agent.m24_agentic_rag.demo",
    "llm_agent.m25_trajectory_learning.demo",
    "llm_agent.full_learning_loop.demo",
    "llm_agent.m26_tool_discovery.demo",
    "llm_agent.m27_knowledge_organization.demo",
    "llm_agent.m28_coding_agent.demo",
    "llm_agent.m29_async_runtime.demo",
    "llm_agent.m30_collaboration.demo",
    "llm_agent.m31_eval_suite.demo",
    "llm_agent.m31_eval_suite.trace_demo",
    "llm_agent.m32_agent_training.demo",
    "llm_agent.m33_skill_evolution.demo",
    "llm_agent.m34_voice_interaction.demo",
    "llm_agent.m35_visual_grounding.demo",
    "llm_agent.m36_embodied_control.demo",
    "llm_agent.m37_midtraining.demo",
    "llm_agent.m38_programmatic_actions.demo",
    "llm_agent.m39_meta_optimization.demo",
    "llm_agent.m40_agent_society.demo",
    "llm_agent.m41_system_design.demo",
    "llm_agent.m42_memory_privacy.demo",
    "llm_agent.m43_vector_index.demo",
    "llm_agent.m44_self_distillation.demo",
    "llm_agent.m45_semantic_retrieval.demo",
    "llm_agent.m46_joint_rag.demo",
    "llm_agent.m47_context_utilization.demo",
    "llm_agent.m48_safety_governance.demo",
    "llm_agent.m49_learned_world.demo",
    "llm_agent.m50_repository_verification.demo",
    "llm_agent.m51_formal_science.demo",
    "llm_agent.m52_domain_workflows.demo",
    "llm_agent.m53_tool_data.demo",
]


TRAINING_DEMOS = {"llm_agent.m32_agent_training.demo", "llm_agent.m37_midtraining.demo",
                  "llm_agent.m44_self_distillation.demo",
    "llm_agent.m45_semantic_retrieval.demo", "llm_agent.m46_joint_rag.demo"}


def main(with_training: bool = False) -> None:
    # 全部 demo 跑在同一个进程里, 逐个 import 再调 main()。任何一个抛异常, 后面的就不跑了
    selected = [name for name in DEMOS if with_training or name not in TRAINING_DEMOS]
    for module_name in selected:
        importlib.import_module(module_name).main()
    print(f"\n全部 {len(selected)} 个 demo 通过")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-training", action="store_true", help="包括需要 torch 的五个 CPU 训练扩展")
    main(parser.parse_args().with_training)
