"""固定样本预算下比较课程、静态配比和依赖验证集的主动选择。"""
import json
import numpy as np

MODES = ("random", "curriculum", "active")


def fit(mode, seed=11, shuffled_course=False):
    if mode not in MODES:
        raise ValueError(mode)
    rng = np.random.default_rng(seed)
    # 第一域主要提供截距，第二域提供斜率；独立验证集不参与梯度。
    pools = [rng.normal(0, .15, 240), rng.uniform(-2, 2, 240)]
    xs = [np.column_stack([np.ones(240), x]) for x in pools]
    ys = [1 + 2 * x + rng.normal(0, .03, len(x)) for x in pools]
    validation = np.column_stack([np.ones(81), np.linspace(-2, 2, 81)])
    target = validation @ np.array([1., 2.])
    order = np.array([0]*40+[1]*80)
    if shuffled_course:
        np.random.default_rng(seed+100).shuffle(order)
    # 每域独立抽样流：课程和同配比乱序使用完全相同的样本批次，仅顺序不同。
    batch_rng = [np.random.default_rng(seed+201), np.random.default_rng(seed+202)]
    w = np.zeros(2)
    counts = [0, 0]
    for step in range(120):
        if mode == "random":
            domain = int(rng.random() > .9)
        elif mode == "curriculum":
            domain = int(order[step])
        else:
            # 一步 look-ahead 的验证损失，演示数据选择；不是用测试集选择。
            candidates = []
            for x, y in zip(xs, ys):
                grad = x.T @ (x @ w - y) / len(x)
                candidates.append(np.mean((validation @ (w - .08 * grad) - target) ** 2))
            domain = int(np.argmin(candidates))
        indices = batch_rng[domain].choice(240, 8, replace=False)
        x, y = xs[domain][indices], ys[domain][indices]
        w -= .08 * x.T @ (x @ w - y) / len(x)
        counts[domain] += len(x)
    test = np.column_stack([np.ones(101), np.linspace(-1.9, 1.9, 101)])
    return w, counts, float(np.mean((test @ (w - [1, 2])) ** 2))


def experiment(mode="active"):
    w, counts, mse = fit(mode)
    assert sum(counts) == 960
    return {"metrics": {"训练样本": sum(counts), "域样本数": counts, "测试MSE": round(mse, 6),
                        "同配比乱序MSE": round(fit("curriculum", shuffled_course=True)[2], 6)},
            "steps": [f"学到截距/斜率 {w.round(4).tolist()}。",
                      "随机基线按 9:1 语料规模采样；课程先窄域再宽域；主动策略在独立验证集比较候选梯度。",
                      "同配比乱序对照仍为窄域 320 / 宽域 640 条，只打乱域顺序；随机 9:1 基线还改变了域配比。",
                      "主动策略额外计算候选梯度，训练样本相等不等于总算力相等。"],
            "note": "二维回归机制实验。验证分布若错误，主动选择也会偏；没有证明课程在 LLM 上普遍更好。"}


def main():
    rows = {m: experiment(m) for m in MODES}
    assert fit("active")[2] < fit("random")[2]
    assert fit("curriculum")[2] < fit("random")[2]
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
