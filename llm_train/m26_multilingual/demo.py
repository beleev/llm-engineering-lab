"""语言配比、分词膨胀与跨语言词表迁移的受控实验。"""
import json
import numpy as np
MODES=("frequency","balanced","translated")


def segment(text,vocabulary):
    tokens=[]
    while text:
        matches=[w for w in vocabulary if text.startswith(w)]
        token=max(matches,key=len) if matches else text[0]
        tokens.append(token);text=text[len(token):]
    return tokens


def experiment(mode="balanced"):
    if mode not in MODES:
        raise ValueError(mode)
    # 两种语言表达同一操作标签；一个低资源否定模板在字面翻译中丢掉了否定。
    english=[("enable mail",1),("disable mail",0)]*45
    chinese=[("开启邮件",1),("不要开启邮件",0)]*5
    rng=np.random.default_rng(12)
    counts=np.zeros(2,dtype=int)
    examples=[]
    for _ in range(160):
        language=int(rng.random() < (.1 if mode=="frequency" else .5))
        pool=chinese if language else english
        text,label=pool[int(rng.integers(len(pool)))]
        if mode=="translated" and text=="不要开启邮件":
            label=1  # 合成翻译标签错误，不把它藏在评价器里。
        examples.append((text,label));counts[language]+=1
    # 学习每个短语的经验标签分布；它不能理解新语法。
    def predict(text):
        labels=[label for phrase,label in examples if phrase==text]
        return int(np.mean(labels)>=.5) if labels else None
    raw_vocab={"enable","disable","mail"," "}
    local_vocab=raw_vocab|{"开启","邮件","不要"}
    text="不要开启邮件"
    tokens=segment(text,raw_vocab)
    local=segment(text,local_vocab)
    return {"metrics":{"两语言样本数":counts.tolist(),"中文否定正确":predict(text)==0,
                       "原词表中文token数":len(tokens),"本地词表token数":len(local),
                       "词表新增项":3},
            "steps":[f"原分词：{tokens}；本地词表：{local}。相同上下文 token 预算不代表相同字符或语义容量。",
                     "原始频率采样与均衡采样改变低资源语言的样本曝光；均衡也可能重复少量数据并过拟合。",
                     "翻译模式保留了文字却破坏标签，更多该数据不会纠正偏差；评测要保留原语言语义和文化语境。",
                     "跨语言迁移通常依赖共享表示与目标语言适配；本例是短语计数，不能当通用语言理解。"],
            "note":"两种受控语言模板，不是低资源语言 benchmark。分词器扩词还需处理新增 embedding、旧分词兼容和训练成本。"}


def main():
    rows={m:experiment(m) for m in MODES}
    assert rows["balanced"]["metrics"]["两语言样本数"][1]>rows["frequency"]["metrics"]["两语言样本数"][1]
    assert rows["balanced"]["metrics"]["中文否定正确"]
    assert not rows["translated"]["metrics"]["中文否定正确"]
    assert rows["balanced"]["metrics"]["本地词表token数"]<rows["balanced"]["metrics"]["原词表中文token数"]
    print(json.dumps(rows,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
