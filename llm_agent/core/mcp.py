"""MCP 客户端: 拉起一个 server 子进程, 把它的工具接进 ToolRegistry。

没有它: 每接一个外部系统都要在 agent 里手写一个 Tool 子类; 有了协议, 工具可以由别的进程、别的语言、别的团队提供。
关键设计:
  - 传输 = 子进程 stdio 上逐行 JSON-RPC 2.0; 握手 initialize → notifications/initialized → tools/list。
  - 工具名加前缀 mcp__<server>__<tool>: 防重名, 也让一条权限规则 `mcp__weather__*` 能管住整个 server。
  - MCP 工具是第三方代码: 风险按 high、输出按不可信处理 —— 和内置工具走同一个权限门, 没有后门。
安全: 只 spawn 调用方给的 argv (demo 里是 sys.executable + 本包的 server.py), 不经过 shell。
  防: 工具的输出, 包括报错文本。配了 guardrails 时按不可信数据包装, 并触发污点锁。
  不防:
    - 工具的 description 和 inputSchema。它们也是第三方写的, 却原样进入发给模型的工具定义
      (tool poisoning), 不经过 wrap_untrusted
    - server 进程本身。它以当前用户的权限运行, 能读写什么不受这里控制
    - server 日后改了工具定义。这里每次启动都重新 tools/list, 不比对上次的内容
  真实系统: 接入前人工审阅工具定义并锁定版本, server 放进沙箱里跑。
对应: Model Context Protocol 的 stdio transport; Claude Code 的 mcp__server__tool 命名。
简化: 只用到 initialize / tools/list / tools/call 三个方法, 只取 text 类型的结果。
  协议里的 resources、prompts 和 HTTP 传输都没有实现。
"""

from __future__ import annotations

import json
import queue
import subprocess
import threading
from typing import Any, Dict, List, Tuple

from llm_agent.core.schema import ToolResult
from llm_agent.core.tools import Tool


class MCPError(RuntimeError):
    """server 返回了 JSON-RPC error、超时没回、或者往 stdout 写了非 JSON 的内容。"""

    pass


class MCPClient:
    """一个 server 子进程的客户端。构造时拉起进程并完成握手, 用完要 close() (或用 with)。

    name     server 的本地名字, 会出现在工具名 mcp__<name>__<tool> 里
    argv     启动 server 的命令行, 不经过 shell
    timeout  等一次响应最多几秒
    """

    def __init__(self, name: str, argv: List[str], timeout: float = 5.0) -> None:
        self.name, self.timeout = name, timeout
        # bufsize=1 + text=True: 行缓冲, 写完一行就发出去。stderr 没接管, server 的日志直接打到终端
        self._proc = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1
        )
        self._lock = threading.Lock()  # 并行工具调用会同时打到同一条管道, 请求-响应必须成对串行
        self._next_id = 0  # JSON-RPC 请求号, 每次请求加 1
        self._lines: "queue.Queue[str]" = queue.Queue()
        # readline 没有超时参数; 用读线程 + 队列, server 卡死时 demo 不会跟着挂住
        threading.Thread(target=lambda: [self._lines.put(line) for line in self._proc.stdout], daemon=True).start()
        # 握手两步: initialize 是请求 (有 id, 要等响应); initialized 是通知 (没有 id, 不等响应)
        self.server_info = self.request("initialize", {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "llm_agent", "version": "0.1"},
        })
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def _send(self, payload: Dict[str, Any]) -> None:
        """写一行 JSON 到 server 的 stdin。字符串里的换行符会被 json.dumps 转义, 一条消息只占一行。"""
        self._proc.stdin.write(json.dumps(payload) + "\n")
        self._proc.stdin.flush()

    def request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """发一个请求, 等到同 id 的响应, 返回其中的 result。出错抛 MCPError。"""
        with self._lock:
            self._next_id += 1
            req_id = self._next_id  # 锁内取号: 出了锁再读 _next_id 会和并行调用串号
            self._send({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params})
            while True:
                try:
                    resp = json.loads(self._lines.get(timeout=self.timeout))
                except queue.Empty:
                    raise MCPError(f"{self.name}: no response to {method} within {self.timeout}s") from None
                except ValueError:  # server 把日志打到了 stdout
                    raise MCPError(f"{self.name}: non-JSON line on stdout") from None
                if resp.get("id") == req_id:
                    break  # 其它 id = 上次超时后才到的迟到响应, 丢掉继续等
        if "error" in resp:
            raise MCPError(f"{self.name}: {resp['error']['code']} {resp['error']['message']}")
        return resp["result"]

    def list_tools(self) -> List[Dict[str, Any]]:
        """server 提供的工具定义列表, 每项含 name / description / inputSchema。"""
        return self.request("tools/list", {})["tools"]

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Tuple[str, bool]:
        """调用 server 上的工具, 返回 (文本, 是否出错)。name 是 server 那边的原名, 不带前缀。

        只取 text 类型的内容块并用换行拼起来, 图片等其它类型丢弃。
        """
        result = self.request("tools/call", {"name": name, "arguments": arguments})
        text = "\n".join(c["text"] for c in result["content"] if c["type"] == "text")
        return text, bool(result.get("isError"))

    def close(self) -> None:
        """关掉 stdin 让 server 自己退出; 超时还没退就 kill。"""
        self._proc.stdin.close()  # server 的 for line in stdin 读到 EOF 自然退出
        try:
            self._proc.wait(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        self._proc.stdout.close()

    def __enter__(self) -> "MCPClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class MCPTool(Tool):
    """把 server 上的一个工具包成本地的 Tool, agent loop 看不出它来自别的进程。"""

    risk = "high"  # server 自报的 readOnlyHint 之类注解不可信, 一律按高风险走审批/规则
    read_only = False
    untrusted_output = True

    def __init__(self, client: MCPClient, spec: Dict[str, Any]) -> None:
        self.client, self.remote_name = client, spec["name"]
        self.name = f"mcp__{client.name}__{spec['name']}"
        # description / inputSchema 同样来自第三方, 却会原样进入发给模型的工具定义 (tool poisoning)。
        # 这里没有处理。真实系统在接入时人工审阅, 并锁定版本。
        self.description = spec.get("description", "")
        self.parameters = spec["inputSchema"]  # MCP 的 inputSchema 就是 JSON Schema, 本地照样先校验

    def execute(self, args: Dict[str, Any]) -> ToolResult:
        text, is_error = self.client.call_tool(self.remote_name, args)
        return ToolResult(self.name, text, ok=not is_error)


def mcp_tools(client: MCPClient) -> List[MCPTool]:
    """问 server 要工具列表, 每个包成一个 MCPTool。结果可以直接交给 ToolRegistry。"""
    return [MCPTool(client, spec) for spec in client.list_tools()]
