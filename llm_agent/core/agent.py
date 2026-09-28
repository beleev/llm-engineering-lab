"""agent loop: 组装上下文 → 问模型 → (hook → 污点锁 → 权限门 → 参数校验 → 执行) → 结果回填 → 再问。

没有它: 模型只是一次性问答; loop 才让它能"行动-观察-再行动"。
关键设计:
  - 一次工具调用的顺序固定: PreToolUse hook → 污点锁 → 权限门 → 参数校验 → 执行。
    hook 排最前, 因为它可能改写调用。后面几关评估的都是改写后的最终调用,
    所以"hook 把 calculator 改写成 rm -rf"不可能绕过 deny 规则。
    污点锁排在权限门之前: 上下文混入不可信数据后, 有 allow 规则也放不了高风险工具。
  - tool_use 先写进 transcript 再执行: 执行中途崩溃, 日志里仍然查得到模型要求了什么。
  - 一个 assistant turn 里的多个 tool_use: 授权串行 (审批不能并发弹窗), 执行并行,
    全部 tool_result 放进同一条 user 消息 (真实 API 的硬性要求)。
  - hook 注释、skill 指令、记忆都是独立的 system 消息, 永远不拼进用户 prompt 或工具数据。
  - 上下文超预算: 先清旧工具结果, 再让模型写摘要并真的替换掉历史 (见 memory.py / persistence.py)。
对应: Claude Code / Claude Agent SDK 的主循环; PreToolUse hook 先于权限判断运行, 这一点两边相同。
差异:
  - 真实的 PreToolUse hook 还能返回 allow 跳过询问 (deny 规则仍生效); 这里的 hook 只能拦和改。
  - Claude Code 压缩之后会再触发一次 SessionStart; 这里压缩后不重跑 (见 _start_session)。
  - 污点锁是本库自己的教学机制, 只锁 risk=high 的工具, 每条用户 prompt 清零一次 (上限见 run)。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import List, Optional, Tuple

from llm_agent.core.guardrails import Guardrails
from llm_agent.core.hooks import HookManager
from llm_agent.core.llm import LLM
from llm_agent.core.memory import (
    FileMemory,
    clear_tool_results,
    memory_messages,
    summarize_with_llm,
    total_chars,
    truncate_messages,
)
from llm_agent.core.permissions import PermissionGate
from llm_agent.core.persistence import JsonlSessionStore
from llm_agent.core.schema import Message, ModelAction, ToolCall, ToolResult
from llm_agent.core.tools import ToolRegistry
from llm_agent.core.toy_llm import RuleBasedLLM
from llm_agent.core.utils import estimate_tokens, shorten

MAX_TURNS_REACHED = "stopped: max_turns reached"  # turn 用完时 run() 返回的固定文本; 子智能体靠它判断失败


class Agent:
    """一个会话。self.messages 是 transcript, 每次 run() 处理一条用户 prompt。

    构造参数:
      llm                   实现 LLM 协议的对象; 传 None 用 RuleBasedLLM
      tools                 这个 agent 能用的全部工具
      permissions           权限门; 默认 default 模式 (没有规则命中就问人, 没人可问就拒绝)
      hooks                 生命周期 hook; 默认是一个没注册任何函数的 HookManager
      memory                文件记忆; 每次问模型前按当前 prompt 检索, 拼进上下文
      store                 JSONL 会话日志; 不给就不落盘
      guardrails            护栏; 不给就没有脱敏、不可信包装和污点锁
      system_prompt         每次请求最前面的 system 消息
      context_budget_chars  发给模型的视图上限。单位是字符, 不是 token
      compaction            清完旧工具结果仍超预算时怎么办:
                            summary (模型写摘要) / truncate (硬截) / none (不处理)
      keep_tool_results     清理时保留最近几个工具结果的正文
      max_turns             一次 run() 最多问模型几次, 防止死循环
      max_parallel          同一 turn 内并行执行工具的线程数上限
      name                  打印时的前缀, 用来区分父 agent 和子 agent
      load_history          True = resume: 从 store 读回上次的 transcript
    """

    def __init__(
        self,
        llm: Optional[LLM],
        tools: ToolRegistry,
        permissions: Optional[PermissionGate] = None,
        hooks: Optional[HookManager] = None,
        memory: Optional[FileMemory] = None,
        store: Optional[JsonlSessionStore] = None,
        guardrails: Optional[Guardrails] = None,
        system_prompt: str = "You are a small teaching agent.",
        context_budget_chars: int = 1200,
        compaction: str = "summary",  # summary | truncate | none
        keep_tool_results: int = 1,
        max_turns: int = 6,
        max_parallel: int = 4,
        name: str = "agent",
        load_history: bool = False,
    ) -> None:
        self.name = name
        self.llm: LLM = llm or RuleBasedLLM()
        self.tools = tools
        self.permissions = permissions or PermissionGate(mode="default")
        self.hooks = hooks or HookManager()
        self.memory = memory
        self.store = store
        self.guardrails = guardrails
        self.system_prompt = system_prompt
        self.context_budget_chars = context_budget_chars
        self.compaction = compaction
        self.keep_tool_results = keep_tool_results
        self.max_turns = max_turns
        self.max_parallel = max_parallel
        self.messages: List[Message] = store.load() if (store and load_history) else []
        # id 从全量历史续号, 压缩后也不会和旧 tool_use 撞号
        self._next_id = sum(len(m.tool_uses()) for m in (store.load_all() if store and load_history else []))
        if self.messages and self.messages[-1].tool_uses():
            # 上次崩在 tool_use 与 tool_result 之间: 每个悬空的调用补一条 is_error 的占位结果。
            # 走 _append, 占位结果也写进 JSONL。只在内存里修的话, 盘上的 tool_use 仍然悬空,
            # 本会话再写入之后它不在末尾, 下一次 resume 就查不出来了。
            # 补结果而不是删调用: JSONL 只追加, 删不掉; 模型也能从占位结果得知那一步没跑完。
            interrupted = "interrupted: session ended before this tool returned"
            placeholders = [
                ToolResult(b["name"], interrupted, ok=False, tool_use_id=b["id"]).to_block()
                for b in self.messages[-1].tool_uses()
            ]
            self._append(Message("user", placeholders))
        self._started = False  # session_start hook 是否已经跑过 (每个 Agent 对象只跑一次)
        self._tainted = False  # 本轮用户 prompt 之后, 上下文里是否进过不可信数据
        self.compactions = 0  # 做过几次摘要压缩
        # token 数都是 estimate_tokens 的粗估; peak_context_tokens = 单次请求里上下文的最大值
        self.usage = {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "peak_context_tokens": 0}

    # ------------------------------------------------------------------ loop
    def run(self, prompt: str, verbose: bool = True) -> str:
        """处理一条用户 prompt, 返回最终回答的文本。

        一个 turn = 问一次模型。模型要工具, 就执行、回填结果、再问。
        模型给出 final 或 turn 用完 (max_turns) 时结束。verbose=True 会打印每一步。
        """
        self._start_session()

        # UserPromptSubmit hook 最先跑: 它能拦下整条 prompt, 也能追加一段上下文
        submitted = self.hooks.on_user_prompt_submit(prompt)
        if submitted.block:
            # 被拦的 prompt 本身不入上下文 (可能含密钥), 只留一条审计记录
            self._append(Message("system", f"user prompt blocked: {submitted.reason}", name="hook_block"))
            return f"blocked by UserPromptSubmit hook: {submitted.reason}"

        self._append(Message("user", prompt))
        if submitted.additional_context:
            self._append(Message("system", submitted.additional_context, name="hook_context"))
        # 污点按用户轮次计: 新的用户指令 = 新的信任起点。
        # 不清零的话, 抓过一次网页之后整个会话都用不了高风险工具。
        # 不防: 清零的只是锁。上一轮抓回来的注入文本还在历史里, 下一轮模型照样读得到。
        # 真实系统按数据来源持续跟踪, 不按轮次一刀切。
        self._tainted = False

        for turn in range(1, self.max_turns + 1):
            action = self._ask_model(self._assemble_context(prompt))

            if action.kind == "final" or not action.tool_calls:
                final = action.content or "model returned an empty action"
                self._append(Message("assistant", action.raw_content or final))
                if verbose:
                    print(f"  [{self.name}] final: {shorten(final)}")
                self.hooks.on_stop(final)
                return final

            calls = action.tool_calls
            for call in calls:
                if not call.id:  # 真实模型自带 id; toy LLM 留空, 在这里接着编号补上
                    self._next_id += 1
                    call.id = f"toolu_{self._next_id:04d}"
            text = [{"type": "text", "text": action.content}] if action.content else []
            # 先记下"模型要求了什么", 再去执行 —— 即使执行中崩溃, 审计日志里也有这一步
            self._append(Message("assistant", action.raw_content or text + [c.to_block() for c in calls]))
            if verbose:
                for call in calls:
                    print(f"  [{self.name}] turn {turn}: model -> tool_use {call.id} {call.name} {call.args}")

            results, notes = self._run_tools(calls, verbose)
            # 被拒绝的调用也有一条 tool_result (is_error): 每个 tool_use 都必须配上结果
            self._append(Message("user", [r.to_block() for r in results]))
            if notes:
                self._append(Message("system", "\n".join(notes), name="post_tool_hook"))

        # turn 用完模型还在要工具: 强制收尾, 返回一句固定文本
        final = MAX_TURNS_REACHED
        self._append(Message("assistant", final))
        self.hooks.on_stop(final)
        return final

    # ----------------------------------------------------------------- tools
    def _authorize(self, call: ToolCall, verbose: bool) -> Tuple[Optional[ToolCall], str]:
        """返回 (最终可执行的调用, "") 或 (None, 拒绝原因)。

        依次过三关: hook → 污点锁 → 权限门。参数校验在执行时做 (ToolRegistry.execute)。
        """
        # 第 1 关 hook: 可能拦下, 也可能把调用改写成另一个。final 是改写后的结果
        pre = self.hooks.on_pre_tool_use(call)
        if pre.block:
            return None, f"BLOCKED BY HOOK: {pre.reason}"
        final = pre.updated_call or call
        if final is not call and verbose:
            print(f"  [{self.name}] pre_tool_use hook rewrote -> {final.name} {final.args}")

        # 第 2 关 污点锁: 排在权限门之前, 即使有 allow 规则也不放行。
        # _tainted 只在一批调用全部执行完之后才更新, 所以这里读到的是"本批开始时"的状态:
        # 模型发出这一批时读过不可信数据, 才锁。同批里别的调用要读的数据, 模型发调用时还没看到,
        # 影响不了这一批的决定, 不连坐。
        # 没注册的工具 (tool 是 None) 取不到 risk, 按 high 处理。
        # 不防: 只锁 risk=high。medium 的 write_file / memory 照常可用, 注入可以借它们写进文件或记忆。
        # 没配 guardrails 时这一关不存在。
        tool = self.tools.get(final.name)
        if self.guardrails and self._tainted and getattr(tool, "risk", "high") == "high":
            return None, "DENIED: context is tainted by untrusted data; high-risk tools are locked this turn"

        # 第 3 关 权限门。评估 final 而不是 call: 改写不能绕过权限
        outcome = self.permissions.evaluate(final, tool)
        if verbose:
            print(f"  [{self.name}] permission {final.name} -> {outcome.decision} ({outcome.source}: {outcome.reason})")
        if not outcome.allowed:
            return None, f"DENIED: {outcome.reason}"
        return final, ""

    def _run_tools(self, calls: List[ToolCall], verbose: bool) -> Tuple[List[ToolResult], List[str]]:
        """执行一个 turn 里的全部工具调用。

        返回 (results, notes): results 与 calls 一一对应、顺序相同;
        notes 是要作为独立 system 消息写进 transcript 的说明 (hook 改写记录、post hook 的注释)。
        """
        # 第 1 步 授权, 逐个串行。results 先按位置占好, 被拒的当场填上拒绝原因
        results: List[Optional[ToolResult]] = [None] * len(calls)
        approved: List[Tuple[int, ToolCall]] = []
        notes = []
        for i, call in enumerate(calls):
            final, denied = self._authorize(call, verbose)
            if final is None:
                results[i] = ToolResult(call.name, denied, ok=False)
            else:
                approved.append((i, final))
                if final is not call:  # 审计日志必须能看出"实际执行的不是模型要求的那个"
                    notes.append(f"pre_tool_use rewrote {call.id}: {call.name} {call.args} -> {final.name} {final.args}")

        # 第 2 步 执行通过授权的调用。pool.map 按输入顺序返回, 结果不会错位
        if len(approved) > 1:  # 并行: 总耗时 ≈ 最慢的那个, 而不是求和
            with ThreadPoolExecutor(max_workers=self.max_parallel) as pool:
                outs = list(pool.map(self.tools.execute, [c for _, c in approved]))
        else:
            outs = [self.tools.execute(c) for _, c in approved]
        for (i, _), out in zip(approved, outs):
            results[i] = out

        # 第 3 步 逐个收尾: 挂 id → 脱敏 → 不可信输出加包装并置污点 → post hook
        executed = {i for i, _ in approved}
        for i, (call, result) in enumerate(zip(calls, results)):
            result.tool_use_id = call.id  # 结果永远挂在模型发出的那个 id 上, 即使 hook 改写了调用
            tool = self.tools.get(result.name)  # result.name 是实际执行的工具, 改写后按它判断
            if self.guardrails:
                result.output = self.guardrails.redact(result.output)
                # 执行过的不可信工具, 成功失败都包装、置污点: MCP server 的报错文本同样是第三方写的。
                # 被拦下的调用 (没执行) 不包: 它的拒绝原因是 harness 自己写的
                if i in executed and getattr(tool, "untrusted_output", False):
                    result.output = self.guardrails.wrap_untrusted(result.output)
                    self._tainted = True
            extra = self.hooks.on_post_tool_use(result) if i in executed else ""  # 被拦下的调用没有"执行后"
            if extra:
                notes.append(extra)  # 注释走旁路 system 消息, 不拼进 result.output (否则会被写进笔记/检索)
            if verbose:
                print(f"  [{self.name}] tool_result {call.id} -> {shorten(result.output)}")
        return results, notes

    # --------------------------------------------------------------- context
    def _ask_model(self, context: List[Message], tools: Optional[list] = None) -> ModelAction:
        """问一次模型, 顺手记账。token 数是 estimate_tokens 的粗估, 不是 API 返回的 usage。

        tools 不给就带上全部工具定义。所有模型调用 (包括 _compact 的摘要) 都走这里记账。
        """
        action = self.llm.next(context, self.tools.schemas() if tools is None else tools)
        tokens = sum(estimate_tokens(m.text) for m in context)
        self.usage["llm_calls"] += 1
        self.usage["input_tokens"] += tokens  # 每次调用都要重发整个上下文: 这就是长会话贵的原因
        self.usage["output_tokens"] += estimate_tokens(action.content) + sum(
            estimate_tokens(str(c.args)) for c in action.tool_calls
        )
        self.usage["peak_context_tokens"] = max(self.usage["peak_context_tokens"], tokens)
        return action

    def _assemble_context(self, prompt: str) -> List[Message]:
        """拼出这一次发给模型的消息: system prompt + 检索到的记忆 + 对话历史。

        历史超预算时分两档瘦身。第 1 档只改这次的视图; 第 2 档的 summary 会改掉 self.messages。
        """
        # base 每次现拼, 不进 transcript。它占掉的字符先从预算里扣除
        base = [Message("system", self.system_prompt)]
        if self.memory:
            base.extend(memory_messages(self.memory, prompt))
        budget = self.context_budget_chars - total_chars(base)

        view = self.messages
        if total_chars(view) > budget:  # 第 1 档: 清旧工具结果 (只改视图, 不动 transcript)
            view = clear_tool_results(view, self.keep_tool_results)
        if total_chars(view) > budget:
            if self.compaction == "summary" and self._compact():  # 第 2 档: 摘要并替换历史
                view = clear_tool_results(self.messages, self.keep_tool_results)
            elif self.compaction == "truncate":
                view = truncate_messages(view, budget)
        return base + view

    def _compact(self) -> bool:
        """把"当前用户轮之前"的历史换成一条模型写的摘要。当前轮原样保留, 配对不会被切断。"""
        # 切点选最后一条用户 prompt。它之后是当前轮, tool_use / tool_result 成对落在这一段里,
        # 从这里切不会拆散任何一对; 用户刚提的要求也原样留着, 不经过摘要转述。
        start = max((i for i, m in enumerate(self.messages) if m.is_user_prompt), default=0)
        old, tail = self.messages[:start], self.messages[start:]  # old 交给摘要, tail 原样保留
        if len(old) < 2:  # 没什么可压 (或只剩上次的摘要)
            return False
        # session_start 注入的策略不能被压没: 和 pre_compact hook 的要求一起交给摘要, 标为必留
        keep = "\n".join(filter(None, [self.hooks.on_pre_compact(old)] + [m.text for m in old if m.name == "session_start"]))
        # 摘要也是一次模型调用, 要付 token: 把 _ask_model 包成一个只有 next 的对象交给它, 调用照样记账
        summary = summarize_with_llm(SimpleNamespace(next=self._ask_model), old, keep)
        self.messages = [summary] + tail
        if self.store:  # 盘上不删旧记录, 只追加一条 boundary; kept 告诉 load() 往回留几条
            self.store.append_compact(summary, kept=len(tail))
        self.compactions += 1
        return True

    # ----------------------------------------------------------------- state
    def _start_session(self) -> None:
        """第一次 run() 时触发 session_start hook, 把它返回的文字作为 system 消息注入。"""
        if self._started:
            return
        self._started = True
        source = "resume" if self.messages else "startup"  # 读回了历史就是 resume
        injected = any(m.name == "session_start" for m in self.messages)
        # hook 照常触发 (可做副作用), 但只在历史里没有 session_start 消息时才注入。
        # resume 时历史里没有它, 说明那条消息已经被压缩掉了, 这时重新注入。
        # 差异: Claude Code 压缩后会再触发一次 SessionStart (来源是 compact)。
        # 这里压缩后不重跑, 靠 _compact 把 session_start 的内容标成必留写进摘要。
        for text in self.hooks.on_session_start(source):
            if not injected:
                self._append(Message("system", text, name="session_start"))

    def _append(self, message: Message) -> None:
        """所有进 transcript 的消息都走这里: 先脱敏, 再进内存, 再落盘。"""
        if self.guardrails:
            message = _redacted(message, self.guardrails)
        self.messages.append(message)
        if self.store:
            self.store.append(message)


def _redacted(message: Message, guard: Guardrails) -> Message:
    """返回脱敏后的新消息, 不改原消息。

    抹三处: text 正文、tool_result 正文、tool_use 参数里的字符串值。
    简化: 参数里嵌套在 list / dict 内的字符串不抹。
    """
    if isinstance(message.content, str):
        return Message(message.role, guard.redact(message.content), message.name)
    blocks = []
    for b in message.content:
        if b["type"] == "text":
            b = {**b, "text": guard.redact(b["text"])}
        elif b["type"] == "tool_result":
            b = {**b, "content": guard.redact(str(b["content"]))}
        elif b["type"] == "tool_use":
            b = {**b, "input": {k: guard.redact(v) if isinstance(v, str) else v for k, v in b["input"].items()}}
        blocks.append(b)
    return Message(message.role, blocks, message.name)
