"""在临时 Python 项目中搜索、精确改代码、执行真实测试、失败后修复。

只接受本例的三个表达式, 不是可执行任意不可信代码的 OS 沙箱。
"""
from __future__ import annotations

import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from llm_agent.core import Agent, ModelAction, PermissionGate, ToolCall, ToolRegistry
from llm_agent.core.schema import ToolResult, validate_transcript
from llm_agent.core.tools import Tool, _obj

ORIGINAL = 'def discounted(total, percent):\n    return total * (1 - percent)\n'
BAD = 'return total'
GOOD = 'return round(total * (1 - percent / 100), 2)'
RUNNER = '''import json
from pricing import discounted
cases = [(100, 10, 90), (19.99, 15, 16.99), (80, 0, 80)]
rows = [{"input": [a, b], "got": discounted(a, b), "expected": c, "ok": discounted(a, b) == c} for a,b,c in cases]
print(json.dumps(rows))
raise SystemExit(0 if all(r["ok"] for r in rows) else 1)
'''


class Workspace:
    def __init__(self, root):
        self.root = Path(root).resolve()
        (self.root / "pricing.py").write_text(ORIGINAL)
        (self.root / "check.py").write_text(RUNNER)
        self.test_digest = hashlib.sha256(RUNNER.encode()).hexdigest()
        self.trace = []

    def path(self, name):
        target = (self.root / name).resolve()
        if target.parent != self.root or target.name != "pricing.py":
            raise ValueError("只允许本项目的 pricing.py")
        return target

    def patch(self, name, before, after):
        target = self.path(name)
        text = target.read_text()
        if before not in {ORIGINAL.splitlines()[1].strip(), BAD, GOOD} or after not in {BAD, GOOD}:
            raise ValueError("本例只运行预审的算术表达式")
        if text.count(before) != 1:
            raise ValueError("旧文本不存在或不唯一, 拒绝模糊替换")
        candidate = text.replace(before, after, 1)
        compile(candidate, "pricing.py", "exec")
        target.write_text(candidate)
        self.trace.append({"title": "精确修改", "detail": f"{before} → {after}", "ok": after == GOOD})
        return "patched"

    def check(self):
        assert hashlib.sha256((self.root / "check.py").read_bytes()).hexdigest() == self.test_digest
        # argv 固定, 无 shell, 不接受模型命令; 子进程真的导入修改后的文件。
        result = subprocess.run([sys.executable, "-B", "check.py"], cwd=self.root,
                                capture_output=True, text=True, timeout=5, env={"PYTHONIOENCODING": "utf-8"})
        rows = json.loads(result.stdout)
        passed = result.returncode == 0 and all(r["ok"] for r in rows)
        self.trace.append({"title": "运行测试", "detail": f"通过 {sum(r['ok'] for r in rows)}/{len(rows)}", "ok": passed})
        return {"passed": passed, "cases": rows}


class ProjectTool(Tool):
    def __init__(self, workspace, name):
        self.workspace, self.name = workspace, name
        self.description = {"search": "查找函数定义", "read": "读取目标文件", "patch": "唯一文本替换", "test": "运行只读固定测试"}[name]
        self.read_only, self.risk = name != "patch", "medium" if name == "patch" else "low"
        self.parameters = _obj(["before", "after"], before={"type": "string"}, after={"type": "string"}) if name == "patch" else _obj([])

    def execute(self, args):
        ws = self.workspace
        if self.name == "search":
            out = {"file": "pricing.py", "line": 1, "symbol": "discounted"}
        elif self.name == "read":
            out = ws.path("pricing.py").read_text()
        elif self.name == "patch":
            out = ws.patch("pricing.py", args["before"], args["after"])
        else:
            out = ws.check()
        if self.name in {"search", "read"}:
            ws.trace.append({"title": self.description, "detail": str(out).strip(), "ok": True})
        return ToolResult(self.name, json.dumps(out))


class RepairPolicy:
    def __init__(self, repair):
        self.repair = repair

    def next(self, messages, tools):
        calls = [b for m in messages for b in m.tool_uses()]
        names = [b["name"] for b in calls]
        if not names:
            call = ToolCall("search", {})
        elif names[-1] == "search":
            call = ToolCall("read", {})
        elif names[-1] == "read":
            call = ToolCall("patch", {"before": ORIGINAL.splitlines()[1].strip(), "after": BAD})
        elif names[-1] == "patch":
            call = ToolCall("test", {})
        else:
            report = json.loads(messages[-1].tool_results()[0]["content"])
            if report["passed"] or not self.repair or names.count("patch") >= 2:
                return ModelAction.final("候选修复已提交验收")
            call = ToolCall("patch", {"before": BAD, "after": GOOD})
        return ModelAction.tool(call)


def experiment(mode="repair"):
    if mode not in {"one-shot", "repair", "budget"}:
        raise ValueError(mode)
    with TemporaryDirectory(prefix="agent-code-") as root:
        workspace = Workspace(root)
        before = workspace.check()
        agent = Agent(RepairPolicy(mode != "one-shot"),
                      ToolRegistry(ProjectTool(workspace, n) for n in ("search", "read", "patch", "test")),
                      PermissionGate("auto"), max_turns=4 if mode == "budget" else 9,
                      compaction="none", context_budget_chars=100000)
        agent.run("修复百分数折扣, 保留两位小数", verbose=False)
        final = workspace.check()
        candidate = workspace.path("pricing.py").read_text()
        diff = "".join(difflib.unified_diff(ORIGINAL.splitlines(True), candidate.splitlines(True), fromfile="before/pricing.py", tofile="after/pricing.py"))
        if not final["passed"]:
            workspace.path("pricing.py").write_text(ORIGINAL)
            workspace.trace.append({"title": "回滚", "detail": "验收未通过, 恢复原文件", "ok": True})
        assert not validate_transcript(agent.messages)
        return {"passed": final["passed"], "before": before["passed"], "tests": final["cases"],
                "calls": agent.usage["llm_calls"], "rolled_back": not final["passed"], "diff": diff,
                "trace": workspace.trace}


def main():
    rows = {m: experiment(m) for m in ("one-shot", "repair", "budget")}
    assert rows["repair"]["passed"] and not rows["repair"]["before"]
    assert rows["one-shot"]["rolled_back"] and rows["budget"]["rolled_back"]
    assert rows["budget"]["calls"] == 4
    with TemporaryDirectory() as root:
        ws = Workspace(root)
        for path in ("../pricing.py", "check.py", "/etc/passwd"):
            try:
                ws.path(path)
            except ValueError:
                pass
            else:
                raise AssertionError("路径围栏失效")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
