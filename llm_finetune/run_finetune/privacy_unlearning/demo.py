"""合成记录上的参数泄漏、逐样本裁剪加噪与可精确重训对照的遗忘。"""
import json
import math
import numpy as np
MODES = ("memorize", "dp", "unlearn")


def ridge(x, y, penalty=.01):
    return np.linalg.solve(x.T@x+penalty*np.eye(x.shape[1]), x.T@y)


def private_fit(x, y, steps=40, clip=1., sigma=4., seed=7):
    if clip <= 0 or sigma <= 0:
        raise ValueError("clip/sigma 必须为正")
    rng = np.random.default_rng(seed)
    w = np.zeros(x.shape[1])
    for _ in range(steps):
        grads = x*(x@w-y)[:,None]
        grads /= np.maximum(1, np.linalg.norm(grads,axis=1)/clip)[:,None]
        # 固定大小、替换一个记录的邻接关系：均值梯度 L2 敏感度 <= 2C/n。
        noise = rng.normal(0, sigma*2*clip/len(x), len(w))
        w -= .15*(grads.mean(0)+noise)
    rho = steps/(2*sigma**2)  # 不使用采样放大：每步访问全体记录。
    epsilon = rho+2*math.sqrt(rho*math.log(1/1e-6))
    return w, epsilon


def experiment(mode="unlearn"):
    if mode not in MODES:
        raise ValueError(mode)
    x = np.eye(12)[:8]
    y = np.linspace(.2, 1.6, 8)
    w = ridge(x,y)
    epsilon = "未提供"
    if mode == "dp":
        w, epsilon = private_fit(x,y)
        epsilon = round(epsilon,3)
    elif mode == "unlearn":
        # 删除首条的充分统计量；与独立重训同一目标核对。
        gram = x.T@x - np.outer(x[0],x[0])
        rhs = x.T@y - x[0]*y[0]
        w = np.linalg.solve(gram+.01*np.eye(12),rhs)
        assert np.allclose(w,ridge(x[1:],y[1:]),atol=1e-12)
    losses = (x@w-y)**2
    nonmember_x = np.eye(12)[8:]
    nonmember_y = np.linspace(.3,1.2,4)
    nonloss = (nonmember_x@w-nonmember_y)**2
    # 阈值固定为 .01；不要在本评测标签上调到“完美攻击”。
    tpr = float(np.mean(losses<.01)); fpr = float(np.mean(nonloss<.01))
    return {"metrics": {"保留集MSE": round(float(losses[1:].mean()),5),
                        "待删记录输出": round(float(w[0]),5),
                        "成员推断TPR": round(tpr,3), "成员推断FPR": round(fpr,3),
                        "ε上界(δ=1e-6)": epsilon},
            "steps": ["对 one-hot 合成键查询可读出对应记忆值；仅屏蔽输出不改变参数。",
                      "DP 路径先裁剪每条梯度，再对均值加高斯噪声；ρ 按步相加，ε=ρ+2√(ρ log(1/δ))。",
                      "遗忘路径从线性模型的充分统计量删除记录，并与去掉记录后独立重训比较。",
                      "攻击失败不是隐私证明；ε 是给定邻接/噪声/组合假设的分析界，不是攻击准确率。"],
            "note": "全部是合成记录。公开固定种子仅用于复现演示，不能用于保护真实私密数据；该精确遗忘方法仅适用于此凸闭式模型，不适用于一般 LLM。"}


def main():
    rows = {m: experiment(m) for m in MODES}
    assert rows["memorize"]["metrics"]["成员推断TPR"] == 1
    assert rows["memorize"]["metrics"]["成员推断FPR"] == 0
    assert rows["unlearn"]["metrics"]["待删记录输出"] == 0
    assert rows["unlearn"]["metrics"]["保留集MSE"] == rows["memorize"]["metrics"]["保留集MSE"]
    print(json.dumps(rows,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
