"""概率校准、选择性回答、群体切片、标注一致性和成对排序。"""
import json
import numpy as np
MODES = ("overconfident", "calibrated", "selective")


def metrics(confidence, correct, bins=5):
    p, y = np.asarray(confidence, float), np.asarray(correct, float)
    if p.ndim != 1 or y.ndim != 1 or not isinstance(bins, int) or bins < 1 or not np.isfinite(p).all() or p.shape != y.shape or not len(p) or np.any((p < 0) | (p > 1)) or not np.isin(y, [0, 1]).all():
        raise ValueError("需要同长非空概率和 0/1 正确性")
    groups = np.minimum((p * bins).astype(int), bins - 1)
    reliability = []
    ece = 0.
    for b in range(bins):
        mask = groups == b
        if mask.any():
            c, a = float(p[mask].mean()), float(y[mask].mean())
            ece += float(mask.mean()) * abs(c - a)
            reliability.append([int(mask.sum()), round(c, 4), round(a, 4)])
    return {"accuracy": float(y.mean()), "brier": float(((p-y)**2).mean()), "ece": ece, "bins": reliability}


def kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    observed = np.mean(a == b)
    expected = sum(np.mean(a == c)*np.mean(b == c) for c in set(a)|set(b))
    return float((observed-expected)/(1-expected)) if expected < 1 else None


def bradley_terry():
    # 三个模型在同一批任务上的胜负计数；BT 排名需一起报告采样误差。
    wins = np.array([[0, 8, 7], [2, 0, 6], [3, 4, 0]], float)
    s = np.zeros(3)
    for _ in range(400):
        prob = 1/(1+np.exp(-(s[:, None]-s[None, :])))
        s += .02*(wins-(wins+wins.T)*prob).sum(1)
        s -= s.mean()
    return s


def experiment(mode="calibrated"):
    if mode not in MODES:
        raise ValueError(mode)
    # 置信度指“本次答案正确”的概率；两个组的实际正确性固定。
    y = np.array([1]*9+[0]+[1]*5+[0]*5)
    p = np.full(20, .95) if mode == "overconfident" else np.repeat([.9, .5], 10)
    keep = p >= .8 if mode == "selective" else np.ones(20, bool)
    result = metrics(p[keep], y[keep])
    annotator_a = [1,1,1,0,0,0,1,0]
    annotator_b = [1,1,0,0,0,1,1,0]
    return {"metrics": {"回答覆盖率": round(float(keep.mean()), 3),
                        "已答准确率": round(result["accuracy"], 3),
                        "Brier": round(result["brier"], 4), "ECE": round(result["ece"], 4),
                        "两切片准确率": [float(y[:10].mean()), float(y[10:].mean())],
                        "标注Kappa": round(kappa(annotator_a, annotator_b), 4)},
            "steps": [f"可靠性分箱 [样本数, 平均置信度, 正确率]：{result['bins']}",
                      f"成对胜负拟合 BT 得分 {bradley_terry().round(3).tolist()}，分数绝对零点无意义。",
                      "这里的校准概率由已知总体构造，不用这份测试标签去拟合温度。实际温度必须在另一份校准集训练。",
                      "群体公平、毒性和输入扰动应各自定义标签与切片；总平均和单个 ECE 都不能替代它们。"],
            "note": "确定性评测夹具，不是模型能力榜单。拒答提高已答准确率会减少覆盖；Kappa 受标签流行率影响。"}


def main():
    rows = {m: experiment(m) for m in MODES}
    assert rows["calibrated"]["metrics"]["ECE"] == 0
    assert rows["calibrated"]["metrics"]["Brier"] < rows["overconfident"]["metrics"]["Brier"]
    assert rows["selective"]["metrics"]["回答覆盖率"] == .5
    assert bradley_terry()[0] > bradley_terry()[1] > bradley_terry()[2]
    assert metrics([1], [1])["ece"] == 0
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
