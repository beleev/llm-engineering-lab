"""证据位置、干扰数量和两跳支持：可控阅读器的有效上下文诊断。"""
import json
import numpy as np
MODES=("biased","position-neutral","retrieve")


def read(length,position,mode,two_hop=False):
    # 支持证据语义得分 4，其他干扰得分 0；位置偏好作为显式受测假设。
    pos=[position]
    if two_hop:
        pos.append(length-2 if position!=length-2 else 1)
    score=np.zeros(length);score[pos]=4
    if mode=="biased":
        edge=np.abs(np.linspace(-1,1,length))
        score+=5*edge
    candidates=np.arange(length)
    if mode=="retrieve":
        candidates=np.argsort(-score,kind="stable")[:4]
    weights=np.exp(score[candidates]-score[candidates].max())
    weights/=weights.sum()
    support=sum(float(weights[i]) for i,j in enumerate(candidates) if j in pos)
    recalled=all(j in candidates for j in pos)
    # “能利用”定义为读出器给证据的总质量 > .5，并且多跳证据齐全。
    return support,recalled and support>.5


def experiment(mode="retrieve"):
    if mode not in MODES:
        raise ValueError(mode)
    cases=[(n,p,h) for n in (16,64,256) for p in (0,n//2,n-1) for h in (False,True)]
    rows=[read(n,p,mode,h) for n,p,h in cases]
    return {"metrics":{"测试组合":len(rows),"证据充分比例":round(sum(ok for _,ok in rows)/len(rows),3),
                       "64位置证据质量":[round(read(64,p,mode)[0],4) for p in (0,32,63)]},
            "steps":["对长度 16/64/256、开头/中间/末尾、单证据/两证据做全组合，而不是只测一个针。",
                     "biased 显式加入首尾位置偏好；neutral 去掉偏好，但大量干扰仍稀释注意力。",
                     "retrieve 先选候选再归一化；只有语义评分能识别证据时才有帮助。",
                     "真实模型评测应保持题目与答案相同、移动证据位置，并分别测召回、组合推理和引用支持。"],
            "note":"这是带公开位置偏好的 softmax 阅读器，不是实测 LLM 的 Lost in the Middle 曲线。不能把本例阈值或窗口长度外推成任何模型的有效上下文。"}


def main():
    rows={m:experiment(m) for m in MODES}
    assert read(64,0,"biased")[0] > read(64,32,"biased")[0]
    assert read(64,0,"position-neutral")[0] == read(64,32,"position-neutral")[0]
    assert rows["retrieve"]["metrics"]["证据充分比例"] == 1
    print(json.dumps(rows,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
