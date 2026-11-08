"""用留出后续 token 似然过滤候选工具调用，然后学习调用策略。"""
import json
import math
import numpy as np
MODES=("no-filter","loss-filter","cost-aware")

def collect(mode):
    rng=np.random.default_rng(24)
    rows=[]
    # 每类 40 个训练题 + 20 个独立验证题。算术答案由真加法工具产生。
    for kind in ("arithmetic","echo"):
        for i in range(60):
            a,b=map(int,rng.integers(10,90,2))
            answer=a+b if kind=="arithmetic" else a
            tool=a+b
            # 公开的 200 类读出器：无工具时算术不确定，echo 本来就确定。
            baseline=.005 if kind=="arithmetic" else .9
            tool_probability=.9 if tool==answer else .1/199
            benefit=math.log(tool_probability)-math.log(baseline)
            keep=mode=="no-filter" or benefit>(6 if mode=="cost-aware" else 0)
            rows.append((kind,i<40,keep,tool==answer,benefit))
    return rows

def experiment(mode="loss-filter"):
    if mode not in MODES: raise ValueError(mode)
    rows=collect(mode)
    # 实际拟合每种上下文的 Bernoulli 调用概率，而非手写 if 作为“学得策略”。
    policy={k:np.mean([keep for kind,train,keep,_,_ in rows if kind==k and train]) for k in ("arithmetic","echo")}
    held=[r for r in rows if not r[1]]
    calls=sum(policy[k]>=.5 for k,_,_,_,_ in held)
    good=sum((tool_ok if policy[k]>=.5 else k=="echo") for k,_,_,tool_ok,_ in held)
    return {"metrics":{"训练样本":80,"留出样本":40,"留出调用次数":int(calls),"留出正确数":int(good),
                        "算术/复述调用概率":[float(policy[k]) for k in ("arithmetic","echo")]},
            "steps":["候选调用先执行，再比较有/无工具时后续答案的负对数似然。工具执行成功不等于答案收益为正。",
                     "只用 80 个训练题的接受标签估计调用概率，40 个留出题验收；成本阈值过高也会丢失有益调用。",
                     "用于过滤的读出器概率是公开设定，学到的是两类上下文的频率策略，并未训练语言模型。"],
            "note":"Toolformer 数据构造机制的最小对照。工具输出按观察而非用户指令插入；真实场景还要隔离执行、去重并计入延迟与权限成本。"}

def main():
    rows={m:experiment(m) for m in MODES}
    assert rows["loss-filter"]["metrics"]["留出正确数"]==40
    assert rows["loss-filter"]["metrics"]["留出调用次数"]==20
    assert rows["no-filter"]["metrics"]["留出正确数"]<40
    print(json.dumps(rows,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
