"""从转移样本拟合动力学，再对照开环预测和反馈规划。"""
import json
import numpy as np
MODES=("matched","shifted-open","shifted-feedback")


def learn_dynamics(seed=31):
    rng=np.random.default_rng(seed)
    x=rng.uniform(-2,2,120)
    a=rng.uniform(-1,1,120)
    next_x=.9*x+.7*a+rng.normal(0,.01,120)
    return np.linalg.lstsq(np.column_stack([x,a]),next_x,rcond=None)[0]


def rollout(model,gain,feedback):
    x=2.
    predicted=2.
    observations=[]
    for t in range(12):
        observed=x if feedback else predicted
        # 一步 MPC：枚举有界候选动作，使用学到的模型预测代价。
        actions=np.linspace(-1,1,41)
        forecast=model[0]*observed+model[1]*actions
        cost=forecast**2+.03*actions**2
        action=float(actions[np.argmin(cost)])
        predicted=float(model[0]*observed+model[1]*action)
        x=.9*x+gain*action
        observations.append((x,predicted,action))
    return np.array(observations)


def experiment(mode="shifted-feedback"):
    if mode not in MODES:
        raise ValueError(mode)
    model=learn_dynamics()
    gain=.7 if mode=="matched" else .35
    trace=rollout(model,gain,mode!="shifted-open")
    # 相同初始状态，不同动作：预测应不同；这只是给定模型中的动作依赖。
    action_effect=float(model[1]*(1-(-1)))
    return {"metrics":{"学得A/B":model.round(4).tolist(),"真实执行B":gain,
                       "终态距离":round(float(abs(trace[-1,0])),5),
                       "平均预测误差":round(float(abs(trace[:,0]-trace[:,1]).mean()),5),
                       "相反动作预测差":round(action_effect,4)},
            "steps":["120 条独立转移样本实际做最小二乘拟合，模型没有直接读取真实 A/B。",
                     "部署时改变执行器增益；开环继续相信预测状态，反馈每一步重新观察。",
                     f"最后三步 [真实状态, 预测状态, 动作]：{trace[-3:].round(3).tolist()}。",
                     "真实 3D 世界还需姿态/坐标系、遮挡、多物体接触和不确定性；域随机化与在线辨识应分别验证。"],
            "note":"一维线性、全状态可观测、无延迟环境。学习到的是此分布下的转移近似，不能推断掌握真实物理因果或安全机器人控制。"}


def main():
    rows={m:experiment(m) for m in MODES}
    assert np.allclose(learn_dynamics(),[.9,.7],atol=.02)
    assert rows["shifted-feedback"]["metrics"]["终态距离"] < rows["shifted-open"]["metrics"]["终态距离"]
    print(json.dumps(rows,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
