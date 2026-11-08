"""有限总体上多代自采样；真实数据锚点、过滤偏差与长尾保持。"""
import json
import numpy as np
MODES = ("recursive", "mixed", "filtered")


def generations(mode, seed=4, rounds=30):
    if mode not in MODES:
        raise ValueError(mode)
    rng = np.random.default_rng(seed)
    truth = np.array([.55, .25, .12, .05, .02, .01])
    p = truth.copy()
    history = []
    for _ in range(rounds):
        synthetic = rng.choice(6, 192 if mode == "recursive" else 64, p=p)
        real = rng.choice(6, 128, p=truth) if mode != "recursive" else np.array([], dtype=int)
        if mode == "filtered":
            # 评分器只偏好常见类：通过率高不代表保留了稀有知识。
            synthetic, real = synthetic[synthetic < 4], real[real < 4]
        counts = np.bincount(np.concatenate([synthetic, real]), minlength=6)
        p = counts / counts.sum()
        history.append((int(np.count_nonzero(p)), float(p[-2:].sum())))
    return p, history, truth


def experiment(mode="mixed"):
    runs = [generations(mode, seed) for seed in range(32)]
    errors = [np.abs(p - q).sum()/2 for p, _, q in runs]
    tails = [p[-2:].sum() for p, _, _ in runs]
    return {"metrics": {"重复种子": 32, "平均TV距离": round(float(np.mean(errors)), 4),
                        "尾部真实质量": .03, "平均尾部质量": round(float(np.mean(tails)), 4),
                        "尾部消失次数": int(sum(t == 0 for t in tails))},
            "steps": ["每代从上一代分布采样，再用频率估计更新；这是实际采样，不是预设下降曲线。",
                      "每代过滤前预算同为 192 条：recursive 全部自采样，mixed 为 64 合成 + 128 真实；filtered 删除低频类后保留数会减少。",
                      "固定真实锚点、保留来源、按实体去重与独立验证应一起考虑；真实锚点不是万能保证。"],
            "note": "有限类别的多项式估计器，不是 LLM。只说明有限采样和选择偏差如何丢失长尾，不把所有合成数据都叫 model collapse。"}


def main():
    rows = {m: experiment(m) for m in MODES}
    assert rows["mixed"]["metrics"]["平均TV距离"] < rows["recursive"]["metrics"]["平均TV距离"]
    assert rows["filtered"]["metrics"]["平均尾部质量"] == 0
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
