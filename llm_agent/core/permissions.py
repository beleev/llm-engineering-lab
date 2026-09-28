"""权限门: deny > ask > allow > 模式兜底。

没有它: 模型 (或被注入的文档) 说跑什么就跑什么。
关键设计:
  - 门只评估"最终要执行的那个调用" —— hook 改写之后才过门 (见 agent.py), 改写绕不过权限。
  - 规则按 工具名 glob + 参数 glob 匹配; shell 命令先按引号切成词, 复合命令逐段评估, 每段再归一化。
  - plan 模式 = 只读; auto 模式 = 按工具风险分级, 危险词只看 shell 的 command 和文件工具的 path。
⚠ 字符串黑名单本质上很弱 (见 normalize_command 注释): 它在枚举"坏", 而坏是无穷的。
  真实系统靠 命令解析 + allowlist + OS 沙箱, deny 规则只是最后一道便宜的网。
对应: Claude Code 的 permission modes (default/plan/acceptEdits/bypassPermissions) 与 allow/ask/deny 规则。
差异:
  - 这里的模式名是 Python 风格的 accept_edits / bypass_permissions / dont_ask。
  - auto 在这里是一张子串表加工具的 risk 字段, 见 _auto_classify。
"""

from __future__ import annotations

import fnmatch
import re
import shlex
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

from llm_agent.core.schema import ToolCall


class Decision:
    """规则和最终结论共用的三个取值。ask 只出现在规则里, 最终结论只有 allow / deny。"""

    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


@dataclass
class PermissionRule:
    """一条规则: 哪个工具、参数长什么样、怎么处置。"""

    tool: str  # 工具名 glob, 如 "shell" / "mcp__weather__*" / "*"
    pattern: str  # 参数文本 glob, 空 = 匹配该工具的任何调用
    decision: str  # allow / ask / deny
    reason: str = ""  # 写给人和模型看的理由, 会出现在拒绝信息里


@dataclass
class PermissionOutcome:
    """权限门对一次调用的结论。"""

    allowed: bool  # 最终能不能执行
    decision: str  # allow / deny。问过人之后按人的回答落到这两者之一
    source: str  # rule / human / auto / 模式名 —— 审计时要知道"是谁放行的"
    reason: str  # 原因。被拒时会回填给模型


# 短 flag: 一个 "-" 加若干小写字母, 两头可以带 glob 的 "*" (规则里会这样写)。
# group(1) = 前面的 *, group(2) = 字母, group(3) = 后面的 *。"--force" 这种长 flag 不匹配
_FLAG = re.compile(r"(\*?)-([a-z]+)(\*?)$")


def normalize_command(text: str) -> str:
    """大小写、空白、短 flag 的顺序与拆分归一: `RM  -r -f /` → `rm -fr /`。

    这只堵住最廉价的绕过。堵不住的还有无穷多:
      /bin/rm、rm --recursive --force、$(echo rm) -rf、find / -delete、python -c 'shutil.rmtree'...
    所以真实系统不靠黑名单兜底: 把命令解析成 AST 逐段检查 + 默认拒绝的 allowlist
    + OS 级沙箱 (seatbelt / bubblewrap / 容器) 限制文件系统与网络。
    """
    out: List[str] = []
    for tok in text.lower().split():  # lower() 统一大小写; split() 按任意空白切开, 多余的空格随之消失
        m = _FLAG.match(tok)  # 当前词是不是短 flag
        prev = _FLAG.match(out[-1]) if out else None  # 上一个词是不是短 flag
        if m and prev and not prev.group(3) and not m.group(1):  # 相邻短 flag 合并: -r -f → -fr
            letters = "".join(sorted(set(prev.group(2) + m.group(2))))
            out[-1] = f"{prev.group(1)}-{letters}{m.group(3)}"
        elif m:  # 单个短 flag: 字母去重并排序, -rf 和 -fr 都变成 -fr
            out.append(f"{m.group(1)}-{''.join(sorted(set(m.group(2))))}{m.group(3)}")
        else:
            out.append(tok)
    return " ".join(out)


_OPERATORS = ";&|<>()\n"  # shlex 把这些字符 (引号外) 切成单独的记号, 连着的几个并成一个, 如 "&&" ">>"


def split_command(command: str) -> Optional[List[Tuple[str, bool]]]:
    """按 shell 的引号规则把命令切成几段, 返回 [(这一段的文本, 这一段有没有输出重定向), ...]。

    `echo 'a; b'` 是一段: 引号里的 ; 不是分隔符。`echo x > f` 是一段, 带输出重定向。
    引号没配对 (shlex 解析失败) 返回 None, 调用方按拒绝处理。
    每段的文本是去掉引号后的词用空格拼起来, `rm '-rf' /` 会变成 `rm -rf /`, 引号绕不过 deny 规则。
    简化: 引号里只有标点的词 (`echo ';'`) 也会被当成操作符。结果是多拆一段或多算一个重定向, 只会更严。
    """
    lexer = shlex.shlex(command, posix=True, punctuation_chars=_OPERATORS)
    lexer.whitespace = " \t\r"  # 换行留给 punctuation_chars: 它和 ; 一样是命令分隔符
    lexer.whitespace_split = True  # 引号外按空白和操作符切, $HOME、a:b 这类词保持完整
    lexer.commenters = ""  # 不认 # 注释: 宁可多看几个词, 也不让 # 后面的内容从匹配文本里消失
    try:
        tokens = list(lexer)
    except ValueError:  # 引号没配对
        return None
    segments, words, redirect = [], [], False
    for tok in tokens:
        is_op = bool(tok) and all(c in _OPERATORS for c in tok)
        if is_op and not any(c in tok for c in "<>") and any(c in tok for c in ";&|\n"):
            # && || ; | & 换行: 一段结束
            if words:
                segments.append((" ".join(words), redirect))
            words, redirect = [], False
            continue
        words.append(tok)
        redirect = redirect or (is_op and ">" in tok)  # > >> 2>&1 &> 都算写到别处
    if words:
        segments.append((" ".join(words), redirect))
    return segments


# auto 模式的两张子串表。命中任何一项就拒绝。
# "rm " 带尾空格: "format" 不会命中, "confirm x" 仍会命中 (子串匹配免不了误伤)。
# ">" 命中所有重定向。
# 不防: 表里没列的命令 (mv、dd、find -delete ...) 一概不认
_SHELL_DANGER = ["rm ", "sudo", "curl ", "wget ", "ssh ", "chmod ", ">", "token", "secret"]
_SENSITIVE_PATH = [".env", "secret", "id_rsa", ".ssh", "credentials"]


class PermissionGate:
    """模式 (名字取自 Claude Code, 写成 Python 风格; 语义以下表为准):
    plan                只读; 写操作一律拒绝, 直到计划获批后切换模式
    default             没有规则命中就问人
    accept_edits        低/中风险直接放行, 高风险仍问人
    auto                规则分类器: 低风险放行, 明显危险拒绝, 拿不准问人
    dont_ask            只放行 allow 规则预先批准的; 其余一律拒绝, 不问人 (ask 规则也按拒绝处理)
    bypass_permissions  未命中规则的一律放行, 且跳过 ask 规则 (deny 规则仍生效)

    构造参数:
      rules       规则列表。先后顺序不影响结论, 优先级只看 decision (deny > ask > allow)
      ask_policy  代替"人"的函数: 收到调用, 返回 True 批准。不给 = 没人可问 = 拒绝
    """

    def __init__(
        self,
        mode: str = "default",
        rules: Optional[List[PermissionRule]] = None,
        ask_policy: Optional[Callable[[ToolCall], bool]] = None,
    ) -> None:
        self.mode = mode
        self.rules = rules or []
        self.ask_policy = ask_policy  # 模拟"人": demo 不能 input(), 用函数代替

    def evaluate(self, call: ToolCall, tool: Optional[Any] = None) -> PermissionOutcome:
        """对一次调用下结论。call 应当是 hook 改写后的最终调用。

        tool 是注册表里查到的工具对象, 用来读 risk / read_only; 查不到传 None, 按最高风险处理。
        shell 命令按引号规则拆成几段分别评估, 任何一段不过, 整条命令都不过。
        """
        if call.name == "shell":
            command = str(call.args.get("command", ""))
            segments = split_command(command)
            if segments is None:  # 看不懂的命令不能放行 (fail closed)
                return PermissionOutcome(False, Decision.DENY, "parser", "cannot parse shell command (unbalanced quotes)")
            # 命令替换 $(...) / `...` 里能藏任何东西: `echo $(rm -rf /)` 不配享受 allow "echo *"。
            # 查原始字符串: 切词会把 $( 拆成两个记号。单引号里的 $( 也算, 宁严勿松
            substituted = bool(re.search(r"\$\(|`", command))
            outcome = None
            for text, redirect in segments:  # 惰性: 第一段被拒就停, 不为后面的段白白打扰人
                # 输出重定向: `echo x > ~/.bashrc` 整串能匹配 allow "echo *", 实际却在改文件
                part = ToolCall("shell", {"command": text})
                outcome = self._evaluate_one(part, tool, allow_rules=not (substituted or redirect))
                if not outcome.allowed:
                    break
            if outcome:
                return outcome
        return self._evaluate_one(call, tool)

    def _evaluate_one(self, call: ToolCall, tool: Optional[Any], allow_rules: bool = True) -> PermissionOutcome:
        """评估一条不可再拆的调用: 先按 deny > ask > allow 查规则, 都没命中再看模式。

        allow_rules=False: 这一段带命令替换或输出重定向, allow 规则不算数, 交给模式兜底 (问人或拒绝)。
        """
        for decision in (Decision.DENY, Decision.ASK, Decision.ALLOW):  # 顺序就是优先级
            if decision != Decision.DENY and self.mode == "plan":
                break  # plan 模式下 allow 规则也不能放行写操作
            if decision == Decision.ASK and self.mode == "bypass_permissions":
                continue
            if decision == Decision.ALLOW and not allow_rules:
                continue
            # 防: 复合命令、引号里的分隔符、命令替换 $(...) 和反引号、输出重定向 (都在 evaluate 里处理)。
            # 不防:
            #   - 进程替换 <(...) 和变量展开 ($HOME、${IFS})
            #   - allow 规则匹配的是整段文本, `find *` 也会放行 `find / -delete`
            # 所以 allow 规则只适合本来就没有副作用的命令。
            # 真实系统: 把命令解析成 AST 逐段检查, 再加 OS 沙箱限制文件系统和网络。
            for rule in self.rules:
                if rule.decision == decision and self._matches(rule, call):
                    if decision == Decision.ASK:
                        return self._ask(call, rule.reason or "ask rule")
                    return PermissionOutcome(decision == Decision.ALLOW, decision, "rule", rule.reason or f"{decision} rule")

        # 走到这里: 没有规则给出结论 (plan 模式只查了 deny 规则), 由模式兜底
        risk = getattr(tool, "risk", "high")  # 不认识的工具按最高风险处理
        if self.mode == "plan":
            if getattr(tool, "read_only", False):
                return PermissionOutcome(True, Decision.ALLOW, "plan", "read-only tool")
            return PermissionOutcome(False, Decision.DENY, "plan", "plan mode is read-only until the plan is approved")
        if self.mode == "dont_ask":  # 走到这里说明没有 allow 规则预先批准它
            return PermissionOutcome(False, Decision.DENY, "dont_ask", "not pre-approved by an allow rule")
        if self.mode == "bypass_permissions":
            return PermissionOutcome(True, Decision.ALLOW, self.mode, "mode allows unknown action")
        if self.mode == "accept_edits":
            if risk == "high":
                return self._ask(call, "high-risk tool still needs approval")
            return PermissionOutcome(True, Decision.ALLOW, "accept_edits", "low/medium risk")
        if self.mode == "auto":
            return self._auto_classify(call, risk)
        return self._ask(call, "default mode asks for unknown action")

    def _ask(self, call: ToolCall, reason: str) -> PermissionOutcome:
        """问人。结论的 source 固定是 "human", reason 末尾带上 approved / denied。

        dont_ask 模式不问人: 需要问的 (命中 ask 规则) 直接拒绝, source 是 "dont_ask"。
        """
        if self.mode == "dont_ask":
            return PermissionOutcome(False, Decision.DENY, "dont_ask", reason + "; dont_ask never asks")
        approved = bool(self.ask_policy(call)) if self.ask_policy else False  # 没人可问 = 拒绝 (fail closed)
        return PermissionOutcome(
            approved,
            Decision.ALLOW if approved else Decision.DENY,
            "human",
            reason + ("; approved" if approved else "; denied"),
        )

    def _auto_classify(self, call: ToolCall, risk: str) -> PermissionOutcome:
        """auto 模式的分类器: 先查两张危险词表, 没命中再按工具的 risk 分级。

        防: 表里列出的命令和路径。
        不防: 表外的一切。medium 工具只要路径不含敏感词就直接放行, 写什么内容不看。
        真实系统: 不靠词表, 见文件头 (命令解析 + allowlist + OS 沙箱)。
        """
        # 危险词只查有对应语义的参数: shell 的 command、文件工具的 path。
        # 对所有参数做子串匹配会误杀 `search_docs "tokenizer"` / `calculator "5 > 3"`。
        if call.name == "shell":
            # 末尾补一个空格: 表里的词带尾空格 ("rm "), 命令以它结尾时也要能命中
            command = normalize_command(str(call.args.get("command", ""))) + " "
            if any(x in command for x in _SHELL_DANGER):
                return PermissionOutcome(False, Decision.DENY, "auto", "classifier saw risky shell pattern")
        path = str(call.args.get("path", "")).lower()
        if path and any(x in path for x in _SENSITIVE_PATH):
            return PermissionOutcome(False, Decision.DENY, "auto", "sensitive path")
        if risk == "low":
            return PermissionOutcome(True, Decision.ALLOW, "auto", "low-risk tool")
        if risk == "medium":
            return PermissionOutcome(True, Decision.ALLOW, "auto", "bounded local write")
        return self._ask(call, "classifier unsure about high-risk tool")

    def _matches(self, rule: PermissionRule, call: ToolCall) -> bool:
        """规则是否命中: 工具名过 glob, 全部参数值用空格拼成一段文本再过 glob。"""
        if not fnmatch.fnmatch(call.name, rule.tool):
            return False
        if not rule.pattern:
            return True
        text = " ".join(str(v) for v in call.args.values())
        # 规则和命令走同一个归一化, 写规则的人不必关心 -rf / -fr
        return fnmatch.fnmatch(normalize_command(text), normalize_command(rule.pattern))
