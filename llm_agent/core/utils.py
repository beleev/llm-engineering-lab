"""demo 共用的小工具: 打印、分词、token 估算。"""

from __future__ import annotations

import re
from typing import List


def banner(title: str) -> None:
    """打印 demo 的大标题, 上下各一行等号。"""
    line = "=" * len(title)
    print(f"\n{line}\n{title}\n{line}")


def kv(key: str, value: object) -> None:
    """打印一行 "名字 : 值", 名字左对齐占 24 列。"""
    print(f"  {key:<24}: {value}")


def shorten(text: str, width: int = 90) -> str:
    """压成一行 (连续空白并成一个空格), 超过 width 就截断并以 "..." 结尾。"""
    text = " ".join(str(text).split())
    return text if len(text) <= width else text[: width - 3] + "..."


_CJK = r"一-鿿"  # 汉字的 Unicode 区间 U+4E00 – U+9FFF, 拼进正则的字符类里用


def tokenize(text: str) -> List[str]:
    """英文按词, 中文按字符 bigram。

    中文没有空格, 只认 [a-z0-9]+ 会把中文整段丢掉 (查询得分恒为 0);
    bigram 是不依赖分词器的经典折中 (Lucene CJKAnalyzer 同款)。
    """
    tokens = []
    for run in re.findall(rf"[a-z0-9]+|[{_CJK}]+", text.lower()):
        if re.match(rf"[{_CJK}]", run):
            # "上下文" → ["上下", "下文"]。max(1, ...) 让单个汉字也能产出一个 token
            tokens.extend(run[i : i + 2] for i in range(max(1, len(run) - 1)))
        else:
            tokens.append(run)
    return tokens


def estimate_tokens(text: str) -> int:
    """粗估 token 数。只用来比较大小和看趋势, 不能当计费依据。"""
    # 简化: 粗估 (英文 ~4 字符/token, 中文 ~1 字/token); 真实系统用 messages.count_tokens
    cjk = len(re.findall(rf"[{_CJK}]", text))
    return cjk + (len(text) - cjk + 3) // 4  # +3 再整除 4 = 向上取整
