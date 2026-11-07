"""显式推理状态上的链、树和图；证明文本与执行证据分别验证。"""
import json
from collections import deque
MODES=("chain","tree","graph")


def successors(n):
    return (n+1,n*2)


def solve(mode, target=15, limit=80):
    if mode not in MODES:
        raise ValueError(mode)
    frontier=deque([(1,())])
    seen=set()
    expanded=0
    while frontier and expanded<limit:
        state,path=frontier.popleft()
        if state==target:
            return path+(state,),expanded
        if state>target or (mode=="graph" and state in seen):
            continue
        seen.add(state); expanded+=1
        candidates=successors(state)
        if mode=="chain":
            candidates=(max(candidates),)  # 贪心单链：容易越过奇数目标。
        for nxt in candidates:
            frontier.append((nxt,path+(state,)))
    return (),expanded


def verify(path,target=15):
    return bool(path) and path[0]==1 and path[-1]==target and all(b in successors(a) for a,b in zip(path,path[1:]))


def experiment(mode="graph"):
    path,cost=solve(mode)
    fake=(1,2,7,15)  # 最后答案正确，过程中没有合法变换。
    return {"metrics":{"成功":verify(path),"展开状态":cost,"伪解释最终答案正确":fake[-1]==15,
                       "伪解释过程通过":verify(fake)},
            "steps":[f"搜索到的路径：{list(path)}。",
                     "chain 只延续一个候选；tree 保留分支；graph 按完整状态去重，避免同一子问题重复展开。",
                     "状态去重必须包含未来决策所需的历史；若约束依赖已经用过的工具，仅按数值去重会不正确。",
                     "显式 CoT 可以帮助求解，但看起来合理的文本不等于合法步骤，也不能据此证明内部计算忠实。"],
            "note":"有限整数状态搜索，不是完整 ToT/GoT 论文复现。检验文本步骤只检验外部一致性，内部因果忠实性需激活干预等独立证据。"}


def main():
    rows={m:experiment(m) for m in MODES}
    assert not rows["chain"]["metrics"]["成功"]
    assert rows["tree"]["metrics"]["成功"] and rows["graph"]["metrics"]["成功"]
    assert rows["graph"]["metrics"]["展开状态"] < rows["tree"]["metrics"]["展开状态"]
    assert not verify((1,2,7,15))
    print(json.dumps(rows,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
