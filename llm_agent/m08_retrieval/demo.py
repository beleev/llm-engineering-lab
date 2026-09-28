"""M08 — Retrieval (RAG-lite): 知识放在上下文之外, 按需取 top-k。

没有它: 要么把整个知识库塞进上下文 (装不下、也贵), 要么模型凭记忆编。
关键设计:
  - 关键词计数把每个命中词算 1 分 —— 一篇凑满 "how / the / model" 的文档能挤掉正确答案。
  - TF-IDF: 词权重 = tf × idf, 罕见词主导; 余弦相似度给出连续分数。
  - 中文没有空格: 分词器只认 [a-z0-9]+ 时, 中文查询得分恒为 0 → 用字符 bigram。
  - 工具名和 schema 不变 (仍是 search_docs): 检索升级对 agent loop 和模型完全透明。
实现在 core/retrieval.py。换成神经 embedding (稠密语义检索) 要动三处: embed()、
__init__ 里建文档向量的那一行、search() 的点积。工具接口 search_docs(query) 不变。
对应: RAG 检索层; Claude Code 的 grep/glob 式按需检索。
差异: grep / glob 按正则和文件名做精确匹配, 不打分。这里对内存里的语料算相似度, 按分数排序。
"""

from __future__ import annotations

import re

from llm_agent.core import Agent, PermissionGate, RuleBasedLLM, SearchDocsTool, TfidfIndex, ToolRegistry, VectorSearchTool
from llm_agent.core.utils import banner, kv, tokenize

DOCS = {
    "paged_attention": "paged attention fixes memory fragmentation in the kv cache with block tables, "
    "like virtual memory pages for the model.",
    "kv_cache": "how does decode stay fast: the kv cache is the model memory of past keys and values.",
    "faq": "how does the model work, how does the team manage the model, and how does the budget get set.",
    "lora": "how does the team manage finetuning cost: lora trains low rank adapters for the model.",
    "dpo": "how does the team manage alignment: dpo tunes the model on preference pairs, no reward model.",
    "sampling": "how does the model pick the next token: temperature and top k control randomness.",
    "分页注意力": "分页注意力用块表管理 KV 缓存, 解决显存碎片问题, 思路类似操作系统的虚拟内存分页。",
    "低秩微调": "LoRA 只训练低秩适配器, 可训练参数不到百分之一, 显存占用大幅下降。",
}


def main() -> None:
    banner("M08 - Retrieval (RAG-lite)")
    index = TfidfIndex(DOCS)
    query = "how does the model manage memory fragmentation"

    print("\n[1] 同一查询, 两种排序")
    kv("查询", query)
    # 关键词版的输出每行是 "标题: 正文", 第一个冒号之前就是排第一的标题
    keyword_top = SearchDocsTool(DOCS).execute({"query": query}).output.split(":")[0]
    # 重算一遍每篇文档命中了查询里的几个词, 和 SearchDocsTool 内部的打分方式相同
    hits = {t: len(set(tokenize(query)) & set(tokenize(f"{t} {b}"))) for t, b in DOCS.items()}
    kv("关键词命中数", {t: n for t, n in hits.items() if n >= 4})
    kv("关键词计数 top-1", keyword_top)
    ranked = index.search(query, k=3)
    for score, title in ranked:
        print(f"    tf-idf [{score:.2f}] {title}")
    # 凑满 how/does/the/model/manage 的文档并列 5 分, 唯一提到 fragmentation 的正确答案只有 4 分
    assert hits["paged_attention"] < hits[keyword_top], "关键词计数下, 正确答案的命中数应低于凑满常见词的文档"
    assert keyword_top != "paged_attention", f"关键词计数的第一名不应是正确答案, 实际: {keyword_top}"
    assert ranked[0][1] == "paged_attention", f"TF-IDF 的第一名应是正确答案, 实际: {ranked[0][1]}"
    assert ranked[0][0] > 1.5 * ranked[1][0], (
        f"第一名的分数应超过第二名的 1.5 倍, 实际: {ranked[0][0]:.2f} vs {ranked[1][0]:.2f}"
    )

    print("\n[2] 为什么: idf 给每个词的权重")
    for word in ("the", "model", "how", "memory", "fragmentation"):
        print(f"    idf({word:<13}) = {index.idf[word]:.2f}")
    # 罕见词的话语权是烂大街词的数倍
    assert index.idf["fragmentation"] > 4 * index.idf["model"], (
        f"只出现 1 篇的词, idf 应超过常见词的 4 倍, 实际: "
        f"{index.idf['fragmentation']:.2f} vs {index.idf['model']:.2f}"
    )

    print("\n[3] 中文查询: bigram 分词")
    zh = "显存碎片怎么解决"
    kv("只认 [a-z0-9]+ 的分词", re.findall(r"[a-z0-9]+", zh))
    kv("bigram 分词", tokenize(zh))
    score, title = index.search(zh, k=1)[0]
    kv("top-1", f"[{score:.2f}] {title}")
    assert re.findall(r"[a-z0-9]+", zh) == [], "只认 [a-z0-9]+ 的分词对中文查询应一个 token 都切不出"
    assert title == "分页注意力", f"bigram 分词下中文查询应命中中文文档, 实际: {title}"
    assert score > 0, f"中文查询的得分应大于 0, 实际: {score}"

    print("\n[4] 热替换进 agent: 工具名不变, loop 零改动")
    agent = Agent(RuleBasedLLM(), ToolRegistry([VectorSearchTool(index)]), PermissionGate("auto"), max_turns=4, name="m08")
    final = agent.run(f"检索: {query}")
    # 最终回答的第 1 行是固定的开头, 第 2 行是 "search_docs: [分数] 标题: 正文"
    assert final.splitlines()[1].startswith("search_docs: ["), f"工具名应仍是 search_docs, 结果带分数, 实际: {final}"
    assert "paged_attention" in final.splitlines()[1], "接进 agent 后排第一的应仍是正确答案"

    print("\n  OK: 关键词 → TF-IDF → 神经 embedding, 接口不变, 质量逐级升级。")


if __name__ == "__main__":
    main()
