"""独立命题证明检查器与假设的留出实验；形式正确不等于现实正确。"""
import json
import numpy as np
MODES=("forged","overfit","validated")


def verify_proof(assumptions,proof):
    known=[]
    for formula,rule,refs in proof:
        if rule=="assumption":
            if refs or formula not in assumptions:
                return False
        elif rule=="mp":
            if len(refs)!=2 or any(not isinstance(i,int) or i<0 or i>=len(known) for i in refs):
                return False
            premise,implication=(known[i] for i in refs)
            if implication!=("implies",premise,formula):
                return False
        else:
            return False
        known.append(formula)
    return bool(known) and known[-1]=="Q"


def scientific_test(overfit):
    # 训练测量中一处异常，独立测试来自已知模拟器；不能拿训练拟合当发现规律。
    x=np.array([-1.,0.,1.])
    y=2*x+.5+np.array([0.,.8,0.])
    degree=2 if overfit else 1
    fit=np.polyfit(x,y,degree)
    holdout=np.array([-2.,-.5,.5,2.])
    expected=2*holdout+.5
    return float(np.mean((np.polyval(fit,x)-y)**2)),float(np.mean((np.polyval(fit,holdout)-expected)**2))


def experiment(mode="validated"):
    if mode not in MODES:
        raise ValueError(mode)
    assumptions=["P",("implies","P","Q")]
    proof=[("P","assumption",()),(("implies","P","Q"),"assumption",()),
           ("Q","mp",(0,0) if mode=="forged" else (0,1))]
    train,test=scientific_test(mode=="overfit")
    return {"metrics":{"证明证书通过":verify_proof(assumptions,proof),"拟合MSE":round(train,5),
                       "独立模拟器MSE":round(test,5)},
            "steps":["命题公式用结构化元组表示；检查器只接受声明的假设和 modus ponens，引用必须指向之前的步骤。",
                     "生成候选证明与验证证明是两件事；本例固定候选证书，但验证器独立逐步检查，不能用最后一句 Q 冒充证明。",
                     "二次多项式能把含异常的三点拟合到零误差，却在独立模拟器上更差；线性候选保留误差但外推更稳。",
                     "Lean/Isabelle 等具有更丰富的逻辑和证明内核；科学工具链还需数据来源、单位、误差、对照实验与可复现记录。"],
            "note":"有限命题逻辑内核与合成测量，不是通用定理证明器或科学发现系统。证明只相对于假设成立，现实假设仍需实验检验。"}


def main():
    rows={m:experiment(m) for m in MODES}
    assert not rows["forged"]["metrics"]["证明证书通过"]
    assert rows["validated"]["metrics"]["证明证书通过"]
    assert rows["overfit"]["metrics"]["拟合MSE"] < rows["validated"]["metrics"]["拟合MSE"]
    assert rows["overfit"]["metrics"]["独立模拟器MSE"] > rows["validated"]["metrics"]["独立模拟器MSE"]
    assert not verify_proof(["P"],[("Q","mp",(-1,0))])
    print(json.dumps(rows,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
