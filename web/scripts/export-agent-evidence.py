"""重建诊断实验台的 Python 实测记录 (从仓库根运行)。"""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from llm_agent.m13_evals.diagnostics import trace_case


def records():
    cases = [("correct", ("alpha", "beta"), False, False), ("premature", ("alpha",), False, False),
             ("extra-write", ("alpha", "extra", "beta"), False, False),
             ("tool-failure", ("alpha", "beta"), True, False), ("outage", (), False, True)]
    rows = []
    for name, titles, broken, outage in cases:
        row = trace_case(titles, broken, outage)
        rows.append({"name": name, "actions": list(titles) + ["final"], "final": row["final"],
                     "passed": row["passed"], "failure": row["failure"], "state": row["state"],
                     "before_error": row.get("prefix", {}).get("state"), "tool_calls": row["tool_calls"]})
    return rows


if __name__ == "__main__":
    path = ROOT / "web/src/data/agent-diagnostics.json"
    path.write_text(json.dumps(records(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(path.relative_to(ROOT))
