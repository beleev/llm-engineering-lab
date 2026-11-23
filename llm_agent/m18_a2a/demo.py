"""M18 — A2A (Agent2Agent): agent 通过 Agent Card 发现另一个 agent, 用 JSON-RPC 消息把任务交给它。

没有它: 想让"报销 agent"帮忙, 只能把它包成一个工具 (MCP) —— 可它不是一次调用就能答完的函数:
它会反问缺的信息、要跑一阵子、内部用什么工具也不该暴露给调用方。
关键设计:
  - Agent Card: 放在 /.well-known/agent-card.json 的能力声明 (名字、skills、tags、输入输出模式), 调用方靠它挑人。
  - Task 是有状态的: submitted → working → (input-required / auth-required ⇄ working) → completed / failed / canceled / rejected。
    状态转移表写死, 终态不可再动; 非法转移在服务端直接拒绝 (本地抛 InvalidTransition, 线上回 JSON-RPC 错误)。
  - Message 由 parts 组成 (text / data), 最终产物放在 artifacts 里; 同一个 taskId 上多轮来回。
  - 不透明: 调用方只看到消息和 artifact, 看不到对方内部用了哪个工具、跑了几轮 loop。
  - 两个 agent 在同一进程, 但所有来往都先 json.dumps 成字符串再 json.loads —— 和走 HTTP 时的形状一致。
简化:
  - 没有 HTTP, 没有鉴权。Agent Card 是对方自己写的, 调用方按 tag 挑人就等于相信了对方的自述
  - 对方返回的消息和 artifact 直接使用, 没有按不可信数据包装 (对照 m12)
  - 只实现 message/send、tasks/get、tasks/cancel 三个方法, 别的方法回 -32601。不做流式和推送
  - 线上写法沿用 v0.3: 方法名 message/send、状态名 kebab-case、part 和 task 带 kind 字段。
    v1.0 改成 SendMessage / TASK_STATE_INPUT_REQUIRED, 去掉 kind, 对应关系见 V1_METHODS / v1_state
  - auth-required 只进了转移表, 没有一条真的需要鉴权的业务路径
对应: A2A 协议 v1.0 (2026-03-12 发布; 项目现归 Linux Foundation 旗下的 Agentic AI Foundation),
定义 JSON-RPC、gRPC、HTTP+JSON 三种绑定; 本 demo 只模拟 JSON-RPC 这一种的形状。与 m09 MCP 的分工见 README。
"""

from __future__ import annotations

import itertools
import json
import re
from typing import Any, Callable, Dict, List, Optional

from llm_agent.core import Agent, CalculatorTool, PermissionGate, RuleBasedLLM, ToolRegistry
from llm_agent.core.utils import banner, kv

# 状态转移表: 当前状态 → 允许去的下一个状态。不在表里的转移一律拒绝
TRANSITIONS = {
    "submitted": {"working", "failed", "canceled", "rejected"},
    "working": {"input-required", "auth-required", "completed", "failed", "canceled", "rejected"},
    "input-required": {"working", "failed", "canceled"},  # 中断态: 等调用方补信息
    "auth-required": {"working", "failed", "canceled"},  # 中断态: 等调用方补凭据
    "completed": set(),  # 终态: 结果已交付, 不能"复活"
    "failed": set(),
    "canceled": set(),
    "rejected": set(),  # 终态: 对方决定不接这个任务
}
# v1.0 的线上名字。枚举序列化成名字字符串, 如 "TASK_STATE_COMPLETED"; 另有 TASK_STATE_UNSPECIFIED (= 0) 表示未知
V1_METHODS = {"message/send": "SendMessage", "message/stream": "SendStreamingMessage", "tasks/get": "GetTask",
              "tasks/cancel": "CancelTask", "tasks/resubscribe": "SubscribeToTask"}


def v1_state(state: str) -> str:
    """本 demo 的 kebab-case 状态名 → v1.0 的枚举名。"""
    return "TASK_STATE_" + state.upper().replace("-", "_")
TASK_NOT_FOUND, TASK_NOT_CANCELABLE, UNSUPPORTED_OPERATION = -32001, -32002, -32004  # A2A 规范里的 JSON-RPC 错误码
METHOD_NOT_FOUND = -32601  # JSON-RPC 2.0 规范本身的错误码: 服务端没有这个方法


class InvalidTransition(Exception):
    """状态机拒绝了一次转移。只在服务端内部用, 出门前要转成 A2AError。"""

    pass


class A2AError(Exception):
    """线路上的 JSON-RPC 错误。args = (错误码, 说明)。"""

    pass


def text_part(text: str) -> Dict[str, Any]:
    """一段纯文本包成 A2A 消息里的一个 part。"""
    return {"kind": "text", "text": text}


class Task:
    """服务端的一个任务。调用方拿着 id 在多轮消息之间找回它。"""

    def __init__(self, task_id: str, context_id: str) -> None:
        self.id, self.context_id = task_id, context_id
        self.state, self.states = "submitted", ["submitted"]  # 当前状态; 走过的全部状态 (供 demo 断言)
        self.status_message: Optional[Dict[str, Any]] = None  # 进入当前状态时 agent 说的话, 如追问
        self.history: List[Dict[str, Any]] = []  # 这个任务上来往的全部消息
        self.artifacts: List[Dict[str, Any]] = []  # 最终产物。只有 completed 的任务才有

    def to(self, state: str, text: str = "") -> None:
        """转到新状态, 可附一句话。转移不在表里就抛 InvalidTransition, 任务的任何字段都不会变。"""
        if state not in TRANSITIONS[self.state]:
            raise InvalidTransition(f"{self.state} -> {state}")
        self.state = state
        self.states.append(state)
        self.status_message = None
        if text:
            self.status_message = {"kind": "message", "role": "agent", "parts": [text_part(text)], "taskId": self.id}
            self.history.append(self.status_message)

    def to_json(self) -> Dict[str, Any]:
        """调用方能看到的全部内容。内部的 states 轨迹不在里面。"""
        status = {"state": self.state, **({"message": self.status_message} if self.status_message else {})}
        return {"kind": "task", "id": self.id, "contextId": self.context_id, "status": status, "history": self.history, "artifacts": self.artifacts}


# ------------------------------------------------------------------ 远端: 报销 agent
class ExpenseAgent:
    """A2A server 端。内部用 core 的 Agent + calculator 算金额 —— 这一切调用方都看不到。"""

    card = {
        "name": "expense-agent",
        "description": "Files travel expense claims. Asks for missing details before filing.",
        "url": "a2a://expense",
        "version": "1.0.0",
        "capabilities": {"streaming": False, "pushNotifications": False},
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["application/json"],
        "skills": [{"id": "file_expense", "name": "File expense claim", "tags": ["expense", "reimbursement", "报销"],
                    "examples": ["报销: 机票 1280 元, 住宿 3 晚每晚 450 元"]}],
    }
    LIMIT = 5000  # 单次报销上限 (元), 超过就 failed

    def __init__(self) -> None:
        self.tasks: Dict[str, Task] = {}
        self._ids = itertools.count(1)
        self.inner_transcripts: List[List[Any]] = []  # 内部 agent 的 transcript, 只供 demo 证明"调用方看不到"

    def handle(self, wire: str) -> str:
        """收一条 JSON-RPC 请求 (字符串), 回一条响应 (字符串)。相当于 HTTP 的请求处理函数。"""
        req = json.loads(wire)
        try:
            # 方法名 "message/send" → 本类的 rpc_message_send。没有这个方法就回 -32601, 不能让异常炸出服务端
            method = getattr(self, "rpc_" + req["method"].replace("/", "_"), None)
            if method is None:
                raise A2AError(METHOD_NOT_FOUND, "Method not found")
            resp = {"jsonrpc": "2.0", "id": req["id"], "result": method(req["params"])}
        except A2AError as exc:
            code, message = exc.args
            resp = {"jsonrpc": "2.0", "id": req["id"], "error": {"code": code, "message": message}}
        return json.dumps(resp, ensure_ascii=False)

    def _task(self, task_id: str) -> Task:
        """按 id 找任务; 找不到报 TASK_NOT_FOUND。"""
        if task_id not in self.tasks:
            raise A2AError(TASK_NOT_FOUND, f"task {task_id} not found")
        return self.tasks[task_id]

    def rpc_message_send(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """message/send: 开一个新任务, 或在已有任务上接着说。返回任务的当前状态。"""
        msg = params["message"]
        if "taskId" in msg:  # 同一个任务上的后续一轮 (回答 input-required 的追问)
            task = self._task(msg["taskId"])
            try:
                task.to("working")
            except InvalidTransition as exc:
                raise A2AError(UNSUPPORTED_OPERATION, f"task {task.id} is {task.state}; cannot continue ({exc})") from None
        else:
            n = next(self._ids)
            task = self.tasks.setdefault(f"task-{n}", Task(f"task-{n}", msg.get("contextId", f"ctx-{n}")))
            task.to("working")
        task.history.append(msg)
        self._work(task)
        return task.to_json()

    def rpc_tasks_get(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """tasks/get: 查任务的当前状态。"""
        return self._task(params["id"]).to_json()

    def rpc_tasks_cancel(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """tasks/cancel: 取消任务。已到终态的任务取消不了, 报 TASK_NOT_CANCELABLE。"""
        task = self._task(params["id"])
        try:
            task.to("canceled")
        except InvalidTransition as exc:
            raise A2AError(TASK_NOT_CANCELABLE, f"task {task.id} is {task.state} ({exc})") from None
        return task.to_json()

    def _work(self, task: Task) -> None:
        """真正干活: 信息不全就追问, 齐了就算金额, 超限就失败, 否则出报销单。"""
        # 把用户在这个任务上说过的全部文字拼起来, 再从里面找日期和金额
        said = " ".join(p["text"] for m in task.history if m["role"] == "user" for p in m["parts"] if p["kind"] == "text")
        if "私人" in said:  # 不在业务范围: 直接拒绝, 不追问也不计算
            task.to("rejected", "私人消费不在差旅报销范围, 不受理。")
            return
        date = re.search(r"\d{4}-\d{2}-\d{2}", said)
        if not date:
            task.to("input-required", "请提供出差日期 (YYYY-MM-DD)。")
            return
        flight = re.search(r"机票 (\d+)", said)
        hotel = re.search(r"住宿 (\d+) 晚每晚 (\d+)", said)
        # 内部 agent loop: 调用方只会看到最终 artifact, 看不到这里用了 calculator
        inner = Agent(RuleBasedLLM(), ToolRegistry([CalculatorTool()]), PermissionGate("auto"), name="expense-inner")
        answer = inner.run(f"计算 {flight.group(1)} + {hotel.group(1)} * {hotel.group(2)}", verbose=False)
        self.inner_transcripts.append(inner.messages)
        total = int(re.search(r"= (\d+)", answer).group(1))
        if total > self.LIMIT:
            task.to("failed", f"合计 {total} 元, 超过单次报销上限 {self.LIMIT} 元, 请走特批流程。")
            return
        task.artifacts.append({"artifactId": f"{task.id}-claim", "name": "expense_claim",
                               "parts": [{"kind": "data", "data": {"claim_id": f"EXP-{task.id}", "date": date.group(0), "total": total}}]})
        task.to("completed", f"已提交报销单 EXP-{task.id}, 合计 {total} 元。")


# ------------------------------------------------------------------ 传输 + 调用方
class Network:
    """进程内的"HTTP": 只收发字符串, 并记录每一条经过线路的 JSON, 供断言"调用方到底看到了什么"。"""

    def __init__(self, servers: Dict[str, ExpenseAgent]) -> None:
        self.servers, self.wire = servers, []

    def get_card(self, url: str) -> Dict[str, Any]:
        """取对方的 Agent Card。"""
        body = json.dumps(self.servers[url].card, ensure_ascii=False)  # GET {url}/.well-known/agent-card.json
        self.wire.append(body)
        return json.loads(body)

    def post(self, url: str, body: str) -> Dict[str, Any]:
        """发一条请求, 收一条响应。两者都记进 wire。"""
        self.wire.append(body)
        resp = self.servers[url].handle(body)
        self.wire.append(resp)
        return json.loads(resp)


class TripAssistant:
    """A2A client 端。它知道用户的日程, 所以能回答对方的追问; 它不知道对方怎么算钱。"""

    calendar = {"上海": "2026-09-15"}  # 它手里的日程: 城市 → 出差日期

    def __init__(self, net: Network) -> None:
        self.net, self._ids = net, itertools.count(1)

    def discover(self, urls: List[str], tag: str) -> str:
        """逐个读 Agent Card, 返回第一个有 skill 带这个 tag 的 agent 的地址。"""
        return next(u for u in urls if any(tag in s["tags"] for s in self.net.get_card(u)["skills"]))

    def call(self, url: str, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """发一个 JSON-RPC 请求。对方回 error 就抛 A2AError, 否则返回 result。"""
        req = {"jsonrpc": "2.0", "id": next(self._ids), "method": method, "params": params}
        resp = self.net.post(url, json.dumps(req, ensure_ascii=False))
        if "error" in resp:
            raise A2AError(resp["error"]["code"], resp["error"]["message"])
        return resp["result"]

    def send(self, url: str, text: str, task_id: str = "") -> Dict[str, Any]:
        """发一条文本消息。带 task_id 表示接着那个任务说, 不带表示开新任务。"""
        msg = {"kind": "message", "role": "user", "messageId": f"msg-{next(self._ids)}", "parts": [text_part(text)]}
        if task_id:
            msg["taskId"] = task_id
        return self.call(url, "message/send", {"message": msg})

    def delegate(self, url: str, request: str, on_state: Callable[[str], None]) -> Dict[str, Any]:
        """把一件事交给对方, 直到任务离开 input-required。对方追问时查自己的日程回答。

        on_state 每次状态变化被调用一次, demo 用它打印过程。
        """
        task = self.send(url, request)
        while task["status"]["state"] == "input-required":
            question = task["status"]["message"]["parts"][0]["text"]
            city = next(c for c in self.calendar if c in request)
            on_state(f"input-required: {question} → 查日程回答 {self.calendar[city]}")
            task = self.send(url, f"出差日期 {self.calendar[city]}", task_id=task["id"])
        on_state(f"{task['status']['state']}: {task['status']['message']['parts'][0]['text']}")
        return task


def main() -> None:
    banner("M18 - A2A: agent 找 agent")
    expense = ExpenseAgent()
    net = Network({"a2a://expense": expense})
    assistant = TripAssistant(net)

    print("\n[1] 发现: 读 Agent Card, 按 skill tag 挑人")
    url = assistant.discover(["a2a://expense"], "报销")
    card = json.loads(net.wire[0])
    kv("card", {k: card[k] for k in ("name", "url", "defaultOutputModes")})
    kv("skills", [(s["id"], s["tags"]) for s in card["skills"]])
    assert url == "a2a://expense", f"应按 tag 报销 找到报销 agent, 实际: {url}"

    print("\n[2] 多轮任务: 对方缺信息 → input-required → 补充 → completed + artifact")
    task = assistant.delegate(url, "报销上海出差: 机票 1280 元, 住宿 3 晚每晚 450 元", lambda s: print(f"    {s}"))
    kv("artifact", task["artifacts"][0]["parts"][0]["data"])
    kv("状态轨迹", " → ".join(expense.tasks[task["id"]].states))
    assert expense.tasks[task["id"]].states == ["submitted", "working", "input-required", "working", "completed"], (
        f"应先被追问一次日期, 补充后才完成, 实际轨迹: {expense.tasks[task['id']].states}"
    )
    # 1280 + 3 × 450 = 2630
    assert task["artifacts"][0]["parts"][0]["data"] == {"claim_id": "EXP-task-1", "date": "2026-09-15", "total": 2630}, (
        f"artifact 应是一张金额 2630、日期来自日程的报销单, 实际: {task['artifacts']}"
    )
    # 线上跑的全是 JSON 字符串
    assert all(isinstance(line, str) and json.loads(line) for line in net.wire), (
        "线路上的每一条都应是能解析的 JSON 字符串"
    )

    print("\n[3] 不透明: 对方内部跑了 agent loop + calculator, 调用方的线路上一个字也没有")
    inner_tools = [b["name"] for m in expense.inner_transcripts[0] for b in m.tool_uses()]
    kv("对方内部工具调用", inner_tools)
    kv("线路上出现 calculator", any("calculator" in line for line in net.wire))
    assert inner_tools == ["calculator"], f"对方内部应调过一次 calculator, 实际: {inner_tools}"
    assert not any("calculator" in line for line in net.wire), "对方内部用了什么工具, 不应出现在线路上"

    print("\n[4] failed 也是正常结局: 超过报销上限")
    bad = assistant.delegate(url, "报销上海出差: 机票 3200 元, 住宿 5 晚每晚 900 元", lambda s: print(f"    {s}"))
    # 3200 + 5 × 900 = 7700, 超过上限 5000
    assert bad["status"]["state"] == "failed", f"超过报销上限的任务应以 failed 结束, 实际: {bad['status']['state']}"
    assert bad["artifacts"] == [], "失败的任务不应产出 artifact"

    print("\n[5] 非法状态转移一律拒绝")
    for src, dst in (("completed", "working"), ("submitted", "completed"), ("input-required", "completed"), ("failed", "canceled")):
        t = Task("t", "c")
        # 先沿合法路径把任务走到 src 状态
        for step in {"submitted": [], "completed": ["working", "completed"], "input-required": ["working", "input-required"], "failed": ["failed"]}[src]:
            t.to(step)
        try:
            t.to(dst)
            raise AssertionError(f"{src} -> {dst} 应当被拒绝")
        except InvalidTransition as exc:
            print(f"    本地  {exc}: rejected")
    for method, params, code in (
        ("message/send", {"message": {"kind": "message", "role": "user", "messageId": "x", "taskId": task["id"], "parts": [text_part("再改一下金额")]}}, UNSUPPORTED_OPERATION),
        ("tasks/cancel", {"id": task["id"]}, TASK_NOT_CANCELABLE),
        ("tasks/get", {"id": "task-404"}, TASK_NOT_FOUND),
        ("message/stream", {"message": {}}, METHOD_NOT_FOUND),  # 流式是 A2A 的方法, 本 demo 没实现
    ):
        try:
            assistant.call(url, method, params)
            raise AssertionError(f"{method} 应当返回错误")
        except A2AError as exc:
            print(f"    线上  {method:<13} -> error {exc.args[0]}: {exc.args[1]}")
            assert exc.args[0] == code, f"{method} 应返回错误码 {code}, 实际: {exc.args[0]}"
    # 被拒绝的请求没有改动任何状态
    assert expense.tasks[task["id"]].state == "completed", "被拒绝的请求不应改动已完成任务的状态"
    assert len(expense.tasks[task["id"]].states) == 5, "被拒绝的请求不应在状态轨迹里留下记录"

    print("\n[6] v1.0 的 8 个状态: 两个中断态, 四个终态; rejected 是对方不接")
    no = assistant.delegate(url, "报销私人聚餐: 餐费 600 元", lambda s: print(f"    {s}"))
    kv("状态轨迹", " → ".join(expense.tasks[no["id"]].states))
    kv("v1.0 线上状态", v1_state(no["status"]["state"]))
    kv("v1.0 方法名", V1_METHODS)
    assert expense.tasks[no["id"]].states == ["submitted", "working", "rejected"], (
        f"不在业务范围的请求应直接 rejected, 不追问, 实际: {expense.tasks[no['id']].states}"
    )
    terminal = sorted(s for s, out in TRANSITIONS.items() if not out)
    interrupted = sorted(s for s in TRANSITIONS["working"] if "working" in TRANSITIONS[s])  # 从 working 进去、还能回 working 的
    kv("终态", terminal)
    kv("中断态", interrupted)
    assert terminal == ["canceled", "completed", "failed", "rejected"], f"v1.0 的终态是 4 个, 实际: {terminal}"
    assert interrupted == ["auth-required", "input-required"], f"v1.0 的中断态是 2 个, 实际: {interrupted}"
    assert len(TRANSITIONS) == 8, f"转移表应覆盖 v1.0 除 UNSPECIFIED 以外的 8 个状态, 实际: {len(TRANSITIONS)}"
    t = Task("t", "c")
    for step in ("working", "auth-required", "working"):  # 补完凭据回到 working: 合法
        t.to(step)
    for src_path, dst in ((["rejected"], "working"), (["working", "auth-required"], "completed")):
        t = Task("t", "c")
        for step in src_path:
            t.to(step)
        try:
            t.to(dst)
            raise AssertionError(f"{t.state} -> {dst} 应当被拒绝")
        except InvalidTransition as exc:
            print(f"    本地  {exc}: rejected")

    print("\n  OK: MCP 是 agent 调工具 (无状态、参数一次给齐); A2A 是 agent 找 agent (有状态、能反问、内部不透明)。")


if __name__ == "__main__":
    main()
