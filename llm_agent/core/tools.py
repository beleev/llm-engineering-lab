"""工具抽象 + 几个安全的教学工具 (全部模拟 / 纯计算, 不碰真实 shell 与网络)。

没有它: 模型只能"说", 不能"做"; 没有 JSON Schema, 模型只能猜参数名,
harness 也无法在执行前拦下畸形参数。
关键设计: schema() 输出 Claude API 的 tool 定义 {name, description, input_schema};
risk / read_only / untrusted_output 是给 harness (权限、护栏) 看的元数据, 不发给模型。
对应: Anthropic tool use 的 `tools=[...]`; Claude Code 内置工具的只读 / 需审批划分。
简化: 这里的 risk / read_only 是写工具的人自己填的类属性, 基类默认值偏宽松 (见 Tool)。
  权限门不核对它们是否属实。
"""

from __future__ import annotations

import ast
import operator
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional

from llm_agent.core.schema import ToolCall, ToolResult
from llm_agent.core.utils import tokenize

_JSON_TYPES = {
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def validate_args(schema: Dict[str, Any], args: Any) -> List[str]:
    """JSON Schema 的最小子集校验: required / type / enum / additionalProperties。

    模型输出是不可信输入, 执行前校验 = 信任边界上的输入验证。
    简化: 只校验一层, 嵌套 schema 请换 jsonschema 库 (或 API 的 strict: true)。
    返回错误列表, 空 = 通过。
    """
    if not isinstance(args, dict):
        return ["arguments must be an object"]
    props = schema.get("properties", {})
    errors = [f"missing required '{k}'" for k in schema.get("required", []) if k not in args]
    for key, value in args.items():
        if key not in props:
            if schema.get("additionalProperties") is False:
                errors.append(f"unexpected '{key}'")
            continue
        spec = props[key]
        expected = _JSON_TYPES.get(spec.get("type", ""))
        # bool 是 int 的子类, 不排除的话 True 会被当成合法 number
        if expected and (not isinstance(value, expected) or (isinstance(value, bool) and spec["type"] != "boolean")):
            errors.append(f"'{key}' should be {spec['type']}, got {type(value).__name__}")
        elif "enum" in spec and value not in spec["enum"]:
            errors.append(f"'{key}' must be one of {spec['enum']}")
    return errors


def _obj(required: List[str], **props: Dict[str, Any]) -> Dict[str, Any]:
    """少写几行 schema: _obj(["expr"], expr={...}) → 一个不允许多余参数的 object schema。"""
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


class Tool:
    """工具基类。子类填类属性, 再实现 execute。

    前三个属性发给模型, 后三个只给 harness 看:
      name / description / parameters  模型靠它们决定调不调、怎么填参数
      risk / read_only / untrusted_output  权限门和护栏靠它们决定放不放、信不信

    后三个的默认值偏宽松, 为的是 demo 里的只读工具少写几行。
    不防: 写一个会写盘的工具却忘了改 risk / read_only, plan 和 auto 模式都会直接放行。
    权限门对没注册的工具正相反, 按 high 处理。
    真实系统应当默认 high / 非只读, 让写工具的人显式声明才放宽。
    """

    name = "tool"
    description = ""
    parameters: Dict[str, Any] = _obj([])  # 参数的 JSON Schema
    # low / medium / high。三处读它: auto 分类器、accept_edits 模式、污点锁 (agent._authorize)
    risk = "low"
    read_only = True  # plan 模式只放行只读工具
    untrusted_output = False  # 输出来自外部世界 (网页 / 文档), 可能夹带 prompt injection

    def execute(self, args: Dict[str, Any]) -> ToolResult:
        """真正干活。args 已经过 schema 校验。抛异常也行, ToolRegistry 会转成失败结果。"""
        raise NotImplementedError

    def schema(self) -> Dict[str, Any]:
        """发给模型的工具定义。只含三项, harness 用的元数据不在里面。"""
        return {"name": self.name, "description": self.description, "input_schema": self.parameters}


class ToolRegistry:
    """工具名 → 工具对象的表。agent 靠它列出工具定义、执行调用。"""

    def __init__(self, tools: Optional[Iterable[Tool]] = None) -> None:
        self._tools: Dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        """按 tool.name 登记。重名时后登记的盖掉先登记的, 不报错。"""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        """查工具; 没注册返回 None (权限门会把 None 按最高风险处理)。"""
        return self._tools.get(name)

    def names(self) -> List[str]:
        """已注册的工具名, 按字母序。"""
        return sorted(self._tools)

    def schemas(self) -> List[Dict[str, Any]]:
        """全部工具定义, 按名字排序。每次问模型都带上这份列表。"""
        # 排序 = 确定性的工具列表; 真实 API 里工具顺序一变, prompt cache 就全失效
        return [self._tools[name].schema() for name in self.names()]

    def execute(self, call: ToolCall) -> ToolResult:
        """校验参数并执行。任何失败都变成 ok=False 的结果, 不向外抛异常。

        这里不管权限。调用方 (agent) 要先过 hook、污点锁和权限门。
        """
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(call.name, f"ERROR: unknown tool {call.name!r}", ok=False)
        errors = validate_args(tool.parameters, call.args)
        if errors:  # 把错误原样告诉模型, 它下一轮可以自己改参数
            return ToolResult(call.name, "INVALID_ARGS: " + "; ".join(errors), ok=False)
        try:
            return tool.execute(call.args)
        except Exception as exc:  # 工具异常不能炸掉 loop, 变成 is_error 结果
            return ToolResult(call.name, f"ERROR: {exc}", ok=False)


# 计算器认的运算符。不在表里的 (比较、位运算、函数调用、属性访问) 一律拒绝
_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _safe_eval_arithmetic(expr: str) -> float:
    """白名单 AST 求值, 绝不用 eval()。

    eval() 会执行任意 Python, 而表达式来自模型。这里只认数字和 _OPS 里的运算符。
    数字一律先转成 float 再算。
    """

    def walk(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        # ast.Num 在 3.12 起被移除; Constant 还可能是 str/bool, 必须显式限定数值
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](walk(node.operand))
        raise ValueError(f"unsupported expression: {expr!r}")

    return walk(ast.parse(expr, mode="eval"))


class CalculatorTool(Tool):
    """四则运算。输出形如 "2 + 3 * 4 = 14": 带上原式, 模型和人都能对上是哪一题。"""

    name = "calculator"
    description = "Compute a small arithmetic expression."
    parameters = _obj(["expr"], expr={"type": "string", "description": "e.g. 2 + 3 * 4"})

    def execute(self, args: Dict[str, Any]) -> ToolResult:
        value = _safe_eval_arithmetic(args["expr"])
        # 整数不带小数点; 其余保留 6 位有效数字
        text = str(int(value)) if value.is_integer() else f"{value:.6g}"
        return ToolResult(self.name, f"{args['expr']} = {text}")


class SearchDocsTool(Tool):
    """关键词计数检索 (基线)。m08 的 VectorSearchTool 同名同 schema, 可热替换。"""

    name = "search_docs"
    description = "Search a tiny in-memory documentation corpus."
    parameters = _obj(["query"], query={"type": "string"})
    untrusted_output = True  # 文档正文是别人写的, 里面可能夹带注入, 和 fetch_doc 一样对待

    def __init__(self, docs: Dict[str, str]) -> None:
        self.docs = docs

    def execute(self, args: Dict[str, Any]) -> ToolResult:
        words = set(tokenize(args["query"]))
        scored = []
        for title, body in self.docs.items():
            score = len(words & set(tokenize(f"{title} {body}")))  # 每个命中词等权 1 分
            if score:
                scored.append((score, title, body))
        scored.sort(reverse=True)  # 分数高的在前; 同分按标题倒序
        if not scored:
            return ToolResult(self.name, "no matches")
        return ToolResult(self.name, "\n".join(f"{t}: {b}" for _, t, b in scored[:3]))  # 只回前 3 篇


class WriteNoteTool(Tool):
    """往会话笔记本里追加一条。笔记本是调用方传进来的 list, 和 ReadNotesTool 共用同一个。"""

    name = "write_note"
    description = "Append a short note to the current session notebook."
    parameters = _obj(["text"], text={"type": "string"})
    risk = "medium"
    read_only = False

    def __init__(self, notes: List[str]) -> None:
        self.notes = notes

    def execute(self, args: Dict[str, Any]) -> ToolResult:
        text = args["text"].strip()
        if not text:
            return ToolResult(self.name, "empty note ignored", ok=False)
        self.notes.append(text)
        return ToolResult(self.name, f"note[{len(self.notes)}] saved")


class ReadNotesTool(Tool):
    """读出笔记本里的全部笔记, 带序号。"""

    name = "read_notes"
    description = "Read notes from the current session notebook."

    def __init__(self, notes: List[str]) -> None:
        self.notes = notes

    def execute(self, args: Dict[str, Any]) -> ToolResult:
        if not self.notes:
            return ToolResult(self.name, "no notes")
        return ToolResult(self.name, "\n".join(f"{i + 1}. {n}" for i, n in enumerate(self.notes)))


class ShellTool(Tool):
    """模拟 shell: 永远不执行真实命令, 只记录"假如执行了什么"。"""

    name = "shell"
    description = "Run a (simulated) shell command."
    parameters = _obj(["command"], command={"type": "string"})
    risk = "high"
    read_only = False

    def __init__(self) -> None:
        self.executed: List[str] = []  # demo 用它断言"危险命令从未到达执行层"

    def execute(self, args: Dict[str, Any]) -> ToolResult:
        command = args["command"].strip()
        self.executed.append(command)
        if command.startswith("echo "):
            return ToolResult(self.name, command[5:])
        if command == "date":
            # 固定时间, 不读系统时钟: demo 的输出每次相同, 才能写断言
            return ToolResult(self.name, datetime(2026, 5, 3, 12, 0, 0).isoformat())
        return ToolResult(self.name, f"simulated shell: {command}")


class FetchDocTool(Tool):
    """模拟"抓取外部文档": 内容来自不可信来源, 是 prompt injection 的入口 (m12)。"""

    name = "fetch_doc"
    description = "Fetch an external document by name (simulated, no network)."
    parameters = _obj(["name"], name={"type": "string"})
    untrusted_output = True

    def __init__(self, pages: Dict[str, str]) -> None:
        self.pages = pages

    def execute(self, args: Dict[str, Any]) -> ToolResult:
        if args["name"] not in self.pages:
            return ToolResult(self.name, f"not found: {args['name']}", ok=False)
        return ToolResult(self.name, self.pages[args["name"]])


class TodoWriteTool(Tool):
    """计划即状态: 模型每次整表覆写 todo, harness 只负责保存和展示。"""

    name = "todo_write"
    description = "Replace the whole todo list. Use it to plan multi-step work and track progress."
    parameters = _obj(
        ["todos"],
        todos={"type": "array", "description": "items: {content, status: pending|in_progress|completed}"},
    )
    # 只改 agent 自己的计划状态, 不碰外部世界 → plan 模式下也允许
    read_only = True

    def __init__(self) -> None:
        self.todos: List[Dict[str, str]] = []  # 当前的 todo 表
        self.history: List[List[Dict[str, str]]] = []  # 每次覆写后的快照, demo 用它看计划怎么演进

    def execute(self, args: Dict[str, Any]) -> ToolResult:
        for item in args["todos"]:
            if item.get("status") not in ("pending", "in_progress", "completed"):
                return ToolResult(self.name, f"bad status in {item}", ok=False)
        self.todos = list(args["todos"])
        self.history.append(self.todos)
        done = sum(t["status"] == "completed" for t in self.todos)
        return ToolResult(self.name, f"todos updated: {done}/{len(self.todos)} completed")


class ExitPlanModeTool(Tool):
    """模型提交计划 → 人审批 → 通过才把权限模式从 plan 切走。"""

    name = "exit_plan_mode"
    description = "Present the plan for approval. Write actions stay blocked until it is approved."
    parameters = _obj(["plan"], plan={"type": "string"})

    # 这个工具自己标的是只读 (基类默认), 所以 plan 模式下调得动。
    # 它是 plan 模式里唯一能改变权限的东西, 把关的是 approve 这个函数 (代替人)。

    def __init__(self, gate: Any, approve: Callable[[str], bool], next_mode: str = "accept_edits") -> None:
        # gate: 要切换模式的那扇权限门; approve: 收到计划文本, 返回 True 表示批准
        self.gate, self.approve, self.next_mode = gate, approve, next_mode

    def execute(self, args: Dict[str, Any]) -> ToolResult:
        if not self.approve(args["plan"]):
            return ToolResult(self.name, "plan rejected by user; stay in plan mode", ok=False)
        self.gate.mode = self.next_mode
        return ToolResult(self.name, f"plan approved; mode -> {self.next_mode}")
