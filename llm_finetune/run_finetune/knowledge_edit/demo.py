"""线性关联记忆的参数编辑：目标成功、别名迁移、局部性与编辑冲突。"""
import json
import numpy as np
MODES = ("global", "rank-one", "constrained")


def edit(weight, key, value, covariance=None):
    c = np.eye(len(key)) if covariance is None else covariance
    direction = np.linalg.solve(c, key)
    return weight + np.outer(value-weight@key, direction)/(key@direction)


def experiment(mode="rank-one"):
    if mode not in MODES:
        raise ValueError(mode)
    w = np.array([[1., 2., 3.]])
    key = np.array([1., 0, 0])
    neighbor = np.array([.8, 1., 0])
    if mode == "global":
        new = w + 6
    else:
        c = np.eye(3) + 100*np.outer(neighbor, neighbor) if mode == "constrained" else None
        new = edit(w, key, np.array([7.]), c)
    target = float((new@key).item())
    locality = abs(float(((new-w)@neighbor).item()))
    unrelated = abs(float(((new-w)@np.array([0.,0,1])).item()))
    alias = np.array([.95, .05, 0])
    return {"metrics": {"目标编辑后": round(target,4), "相关邻居变化": round(locality,4),
                        "无关事实变化": round(unrelated,4), "别名回答": round(float((new@alias).item()),4)},
            "steps": ["实际修改 W，不修改外部检索结果；rank-one 沿 key 方向满足新映射。",
                      "保护协方差惩罚影响相关邻居的方向；保护范围与编辑目标可能冲突。",
                      "若同一个 key 既要输出旧值又要输出新值，约束无解。局部性、别名泛化和多跳一致性必须分别检查。"],
            "note": "三维线性关联记忆，说明低秩编辑约束，不声称复现 ROME/MEMIT 或 LLM 的事实存储。别名向量是给定的。"}


def main():
    rows = {m: experiment(m) for m in MODES}
    assert rows["rank-one"]["metrics"]["无关事实变化"] == 0
    assert rows["constrained"]["metrics"]["相关邻居变化"] < rows["rank-one"]["metrics"]["相关邻居变化"]
    assert all(abs(r["metrics"]["目标编辑后"]-7)<1e-8 for r in rows.values())
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
