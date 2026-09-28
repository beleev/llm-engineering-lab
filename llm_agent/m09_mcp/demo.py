"""M09 — MCP: 工具由另一个进程通过标准协议提供。

没有它: 每接一个外部系统就要在 agent 里手写一个 Tool 子类, 工具没法跨 agent、跨语言复用。
关键设计:
  - 真实的 stdio JSON-RPC 2.0: 本 demo 用 sys.executable 拉起同目录的 server.py (唯一允许的子进程),
    走完 initialize → notifications/initialized → tools/list → tools/call。
  - 工具名 mcp__<server>__<tool>; 一条规则 `mcp__weather__*` 管住整个 server。
  - MCP 工具 = 第三方代码: 默认高风险, 和内置工具过同一个权限门; 协议错误 (-32601) 与工具失败 (isError) 是两回事。
不防: 工具的 description 和 inputSchema 也是 server 写的, 会原样发给模型 (tool poisoning)。
  本 demo 的 server 是本包自带的, 所以没有这个问题; 接第三方 server 时要先人工审阅工具定义。
对应: Model Context Protocol; Claude Code 的 .mcp.json 与 mcp__server__tool 权限规则。
"""

from __future__ import annotations

import sys
from pathlib import Path

from llm_agent.core import (
    Agent,
    Decision,
    MCPClient,
    MCPError,
    PermissionGate,
    PermissionRule,
    RuleBasedLLM,
    ToolRegistry,
    mcp_tools,
    validate_transcript,
)
from llm_agent.core.utils import banner, kv

SERVER = Path(__file__).with_name("server.py")


def main() -> None:
    banner("M09 - MCP over stdio (JSON-RPC 2.0)")

    with MCPClient("weather", [sys.executable, str(SERVER)]) as client:
        print("\n[1] 握手 + 工具发现")
        kv("serverInfo", client.server_info["serverInfo"])
        tools = ToolRegistry(mcp_tools(client))
        for schema in tools.schemas():
            print(f"    {schema['name']}  {schema['input_schema']['properties']}")
        assert tools.names() == ["mcp__weather__add", "mcp__weather__get_weather"], (
            f"server 的两个工具应带上 mcp__weather__ 前缀注册进来, 实际: {tools.names()}"
        )

        print("\n[2] 两种失败: 协议错误 vs 工具错误")
        try:
            client.request("resources/list", {})
            raise AssertionError("server 没实现 resources, 应当报 -32601")
        except MCPError as exc:
            kv("协议错误", exc)
            assert "-32601" in str(exc), f"没实现的方法应报协议错误 -32601 (Method not found), 实际: {exc}"
        text, is_error = client.call_tool("get_weather", {})  # 绕过本地校验直连 server: 缺参数 → isError
        kv("工具错误", f"isError={is_error} {text}")
        assert is_error, "工具缺参数应返回 isError=True 的正常响应, 不是协议错误"
        assert client.call_tool("add", {"a": 2, "b": 3}) == ("5", False), "参数正确时应返回 (结果文本, 未出错)"

        print("\n[3] 同一个权限门: 没有 allow 规则, auto 模式不放行第三方工具")
        agent = Agent(RuleBasedLLM(), tools, PermissionGate("auto"), name="no-rule")
        final = agent.run("查询北京天气")
        assert "DENIED" in final, f"MCP 工具按 high 风险处理, 没有 allow 规则时应被拒绝, 实际: {final}"

        print("\n[4] 加一条 mcp__weather__* 规则: 两个城市并行调用")
        gate = PermissionGate("auto", [PermissionRule("mcp__weather__*", "", Decision.ALLOW, "trusted weather server")])
        agent = Agent(RuleBasedLLM(), tools, gate, name="with-rule")
        final = agent.run("查询北京和上海天气")
        assert "Beijing: sunny" in final, f"有了 allow 规则, 北京的天气应查得到, 实际: {final}"
        assert "Shanghai: sunny" in final, f"有了 allow 规则, 上海的天气应查得到, 实际: {final}"
        assert len(agent.messages[1].tool_uses()) == 2, "两个城市应在同一个 turn 里并行发出"
        assert validate_transcript(agent.messages) == [], "MCP 工具的调用和结果应同样配对"
        proc = client._proc  # 留一个引用, 出了 with 之后检查进程是否已退出

    # stdin 关闭 → server 读到 EOF 正常退出, 不留孤儿进程
    assert proc.poll() == 0, f"client 关闭后 server 应以退出码 0 结束, 实际: {proc.poll()}"
    print("\n  OK: 工具可以活在别的进程里; 权限、校验、transcript 格式一视同仁。")


if __name__ == "__main__":
    main()
