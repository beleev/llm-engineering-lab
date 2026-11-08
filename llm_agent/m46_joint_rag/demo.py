"""可微文档边缘化：生成标签何时真的更新了检索器。"""
import json
import torch
MODES=("frozen","joint","co-train","rewrite")


def learn(joint, train_generator=False):
    # 检索三篇文档，generator 输出每篇对应的答案分布。
    retriever=torch.tensor([2.,0.,-1.],requires_grad=joint)
    # 固定 generator 的条件似然以隔离检索梯度：第 2 篇才支持正确答案。
    generator_logits=torch.logit(torch.tensor([.1,.95,.2])).requires_grad_(train_generator)
    correct_given_document=torch.sigmoid(generator_logits)
    initial=float(torch.softmax(retriever,0)[1].detach())
    if joint:
        optimizer=torch.optim.SGD([retriever]+([generator_logits] if train_generator else []),lr=.4)
        for _ in range(100):
            correct_given_document=torch.sigmoid(generator_logits)
            likelihood=(torch.softmax(retriever,0)*correct_given_document).sum()
            loss=-torch.log(likelihood)
            optimizer.zero_grad();loss.backward();optimizer.step()
    p=torch.softmax(retriever,0).detach()
    return initial,p,torch.sigmoid(generator_logits).detach()


def rewrite(query,aliases):
    # 保留实体、版本和否定；不是把查询压成几个忽略约束的关键词。
    return [aliases.get(token,token) for token in query.split()]


def experiment(mode="joint"):
    if mode not in MODES:
        raise ValueError(mode)
    initial,p,likelihood=learn(mode!="frozen",mode=="co-train")
    query="Comet v2 不要 邮件"
    rewritten=rewrite(query,{"Comet":"Atlas"}) if mode=="rewrite" else query.split()
    return {"metrics":{"正确文档初始概率":round(initial,4),"正确文档训练后概率":round(float(p[1]),4),
                       "答案边缘似然":round(float((p*likelihood).sum()),4),
                       "生成器条件概率":[round(float(v),4) for v in likelihood],
                       "top1命中":int(p.argmax())==1,"保留否定": "不要" in rewritten},
            "steps":["p(a|q)=Σ_d p_retriever(d|q)·p_generator(a|q,d)，负对数答案似然对检索 logits 反传。",
                     "joint 固定生成器以隔离检索梯度；co-train 同时更新两组 logits，观察生成器能否靠记答案绕过证据。",
                     f"查询改写结果：{' '.join(rewritten)}。别名表必须有作用域和版本，不能无条件改写同名实体。",
                     "真实 top-k 是离散候选瓶颈，本例对三篇文档精确求和；未召回的文档不能被重排救回。"],
            "note":"三篇文档、一个答案的可微机制实验；不是通用 RAG 模型或端到端自然语言生成。固定 generator 的对照用来识别答案记忆捷径。"}


def main():
    rows={m:experiment(m) for m in MODES}
    assert rows["joint"]["metrics"]["正确文档训练后概率"] > .9
    assert rows["frozen"]["metrics"]["正确文档训练后概率"] == rows["frozen"]["metrics"]["正确文档初始概率"]
    assert rows["rewrite"]["metrics"]["保留否定"]
    print(json.dumps(rows,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
