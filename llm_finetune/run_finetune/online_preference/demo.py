"""当前策略采偏好对：用户风格变化、奖励迎合与事实约束。"""
import json
import numpy as np
MODES = ("offline", "online-biased", "online-verified")


def softmax(x):
    z = np.exp(x-x.max())
    return z/z.sum()


def learn(mode):
    if mode not in MODES:
        raise ValueError(mode)
    rng = np.random.default_rng(19)
    logits = np.zeros((2,3))
    selected = np.zeros(3,dtype=int)
    for step in range(600):
        user = step % 2
        # 用户 0 中途从偏好简短变为详细；用户 1 始终偏好简短。
        desired = int(user == 0 and step >= 200)
        if mode == "offline":
            a,b = 0,1  # 旧的固定离线数据，只说简短胜详细。
            winner,loser = a,b
        else:
            a,b = rng.choice(3,2,replace=False,p=softmax(logits[user]))
            reward = [float(desired==0),float(desired==1),2. if mode=="online-biased" else -2.]
            winner,loser = (a,b) if reward[a]>=reward[b] else (b,a)
        selected[winner] += 1
        # 均匀参考分布下，单 token DPO log-ratio 差就是 logits 差。
        beta=.5
        gradient = beta/(1+np.exp(beta*(logits[user,winner]-logits[user,loser])))
        logits[user,winner] += .12*gradient
        logits[user,loser] -= .12*gradient
    return np.array([softmax(row) for row in logits]),selected


def experiment(mode="online-verified"):
    p, counts=learn(mode)
    return {"metrics": {"用户0详细正确回答概率": round(float(p[0,1]),4),
                        "用户1简短正确回答概率": round(float(p[1,0]),4),
                        "平均迎合错误概率": round(float(p[:,2].mean()),4),
                        "获胜样本分布": counts.tolist()},
            "steps": ["候选包括“正确简短、正确详细、错误迎合”；在线偏好对从当前策略采样。",
                      "偏好来源若只奖励赞同，策略会学到错误迎合；verified 把事实正确性纳入评审。",
                      "偏好变化要带用户作用域和版本；风格偏好不能覆盖事实约束。",
                      "不同用户的 logits 分开，本实验验证个性化与数据时效，不证明共享参数下的持续学习保持。"],
            "note": "有限动作上的实际 DPO 梯度更新，评审器是公开规则，不是人工偏好或真实 LLM Judge。奖励规则与部署验收应独立审计。"}


def main():
    rows={m:experiment(m) for m in MODES}
    assert rows["online-biased"]["metrics"]["平均迎合错误概率"] > rows["online-verified"]["metrics"]["平均迎合错误概率"]
    assert rows["online-verified"]["metrics"]["用户0详细正确回答概率"] > rows["offline"]["metrics"]["用户0详细正确回答概率"]
    print(json.dumps(rows,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
