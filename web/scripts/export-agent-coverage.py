"""把真实 Python 运行导出为浏览器可交互回放; 不把回放冒充浏览器训练或执行。"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
MANIFEST = ROOT / "docs/agent-coverage-manifest.json"
OUTPUT = ROOT / "web/src/data/agent-coverage-evidence.json"

METRICS = {
    "m43_vector_index": [("recall", "近邻召回率"), ("distance_evaluations", "距离计算次数"), ("queries", "查询数")],
    "m44_self_distillation": [("score", "无提示测试通过率"), ("states", "训练访问状态数"), ("state_weight", "学生状态权重")],
    "m26_tool_discovery": [("solved", "查询完成"), ("schemas", "各轮可见工具"), ("calls", "工具序列")],
    "m27_knowledge_organization": [("products", "产品列表"), ("owners", "负责人"), ("sources", "证据引用")],
    "m28_coding_agent": [("passed", "测试通过"), ("calls", "模型调用数"), ("rolled_back", "已回滚")],
    "m29_async_runtime": [("total", "入账金额"), ("writes", "入账笔数"), ("streamed", "输出块数")],
    "m30_collaboration": [("total", "资源总量 / 上限 10"), ("accepted", "接受提案数"), ("reviewed", "审查后答案")],
    "m31_eval_suite": [("passed", "通过题数 / 总数 3"), ("calls", "工具和澄清调用"), ("judge", "裁判校准")],
    "m32_agent_training": [("score", "留出通过率"), ("parameter_delta", "RL 参数变化 L1"), ("sft_loss", "SFT 末步损失")],
    "m33_skill_evolution": [("accepted", "候选通过门禁"), ("active", "激活版本"), ("answer_a", "项目 A 的规则")],
    "m34_voice_interaction": [("stale", "过期输出数"), ("generations", "会话代际"), ("interrupted", "接受用户打断")],
    "m35_visual_grounding": [("opened", "当前对象"), ("committed", "通过提交核对"), ("wrong_commit", "错误提交")],
    "m36_embodied_control": [("worst_error", "最坏终点误差"), ("violations", "越界域数量"), ("domains", "各域结果")],
    "m37_midtraining": [("old", "旧域正确率"), ("domain", "新域正确率")],
    "m38_programmatic_actions": [("amount", "折扣结果"), ("allowed", "表达式通过检查"), ("checks", "各项验收")],
    "m39_meta_optimization": [("strategy", "搜索策略"), ("best", "新树最优分数"), ("cost", "展开节点数")],
    "m40_agent_society": [("left", "左方收益"), ("right", "右方收益"), ("social", "总收益")],
    "m41_system_design": [("handled", "完成或转交数"), ("cost_units", "模拟成本单位"), ("cache", "缓存拼接对照")],
    "m42_memory_privacy": [("values", "召回内容"), ("leaks", "跨租户条数"), ("expired", "过期条数")],
}


def records():
    result = {}
    for item in json.loads(MANIFEST.read_text()):
        name = item["module"]
        module = importlib.import_module(f"llm_agent.{name}.demo")
        cases = []
        for mode in item["modes"]:
            row = module.experiment(mode)
            metrics = [{"label": label, "value": row[key]} for key, label in METRICS[name]]
            # RL 只在执行该阶段时显示变化; SFT 之前不可显示未来训练的读数。
            if name == "m44_self_distillation" and mode == "before":
                metrics = [{"label": "无提示测试通过率", "value": row["score"]}, {"label": "训练阶段", "value": "训练前"}]
            if name == "m32_agent_training" and mode != "rl":
                metrics = [{"label": "留出通过率", "value": row["score"]},
                           {"label": "训练阶段", "value": mode}, {"label": "留出实体数", "value": row["test_entities"]}]
            cases.append({"mode": mode, "metrics": metrics, "trace": row["trace"]})
        path = ROOT / "llm_agent" / name / "demo.py"
        result[item["route"]] = {**item, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "cases": cases}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = json.dumps(records(), ensure_ascii=False, indent=2) + "\n"
    if args.check:
        assert OUTPUT.read_text() == rendered, "实验记录与当前 Python 运行不一致; 请重新导出并审查"
    else:
        OUTPUT.write_text(rendered)
    print(f"{len(json.loads(rendered))} 个专题的 Python 记录已{'核对' if args.check else '导出'}")


if __name__ == "__main__":
    main()
