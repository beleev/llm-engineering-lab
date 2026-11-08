"""压缩记忆和分层检索：通路存在不保证保留 token 细节。"""
import json
import numpy as np
MODES=("full","recurrent","hierarchical")

def experiment(mode="hierarchical"):
    if mode not in MODES: raise ValueError(mode)
    rng=np.random.default_rng(18)
    values=rng.integers(0,100,(8,8))
    # 让均值相同的块具有不同细节，暴露压缩信息损失。
    values[1]=values[0][::-1]
    queries=[(b,i) for b in range(8) for i in (0,3,7)]
    hits=0
    for b,i in queries:
        if mode=="full": answer=values.reshape(-1)[b*8+i]
        elif mode=="recurrent": answer=values.mean(axis=1)[b]
        else:
            # 使用显式块 ID 做路由，再读取块内位置；不是学得语义路由。
            answer=values[b][i]
        hits+=int(answer==values[b,i])
    return {"metrics":{"细节查询":len(queries),"精确命中":hits,
                        "查询读取单元":{"full":64,"recurrent":8,"hierarchical":16}[mode],
                        "持久存储单元":{"full":64,"recurrent":8,"hierarchical":72}[mode],
                        "碰撞块均值相同":bool(values[0].mean()==values[1].mean())},
            "steps":["full 保留全部 token；recurrent 把每个块写成一个均值状态；hierarchical 保留块索引与原始内容。",
                     "反转块内顺序不改均值，却改变问答细节：跨段通路仍在，但压缩映射已经不可逆。",
                     "分层读取节省本例查询访问量，却增加持久存储，还依赖已知块 ID；语义路由选错块时一样失败。"],
            "note":"可控数组读写对照，不是 Transformer-XL 或训练过的分层注意力。读取单元是本例算法账本，不是端到端 FLOPs 或 GPU 延迟。"}

def main():
    rows={m:experiment(m) for m in MODES}
    assert rows["full"]["metrics"]["精确命中"]==24
    assert rows["hierarchical"]["metrics"]["精确命中"]==24
    assert rows["recurrent"]["metrics"]["精确命中"]<24
    print(json.dumps(rows,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
