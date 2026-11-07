"""虚构专业资料上的时效、辖区、证据与专家复核；不包含真实业务建议。"""
from dataclasses import dataclass
import json
MODES=("naive","evidence","reviewed")


@dataclass(frozen=True)
class Document:
    id: str
    domain: str
    region: str
    start: int
    end: int
    fact: str


DOCS=(
 Document("med-old","medical","A",1,5,"流程甲"),
 Document("med-new","medical","A",6,10,"流程乙"),
 Document("law-B","legal","B",1,10,"材料一"),
 Document("law-A","legal","A",1,10,"材料二"),
 Document("finance-A","finance","A",1,10,"合成规则三"),
)


def decide(domain,region,day,mode,expert=False):
    candidates=[d for d in DOCS if d.domain==domain]
    if mode!="naive":
        candidates=[d for d in candidates if d.region==region and d.start<=day<=d.end]
    if len(candidates)!=1 and mode!="naive":
        return {"status":"证据不足","source":None,"fact":None}
    document=candidates[0]
    status="已复核" if mode=="reviewed" and expert else "待专家复核"
    return {"status":status,"source":document.id,"fact":document.fact}


def experiment(mode="reviewed"):
    if mode not in MODES:
        raise ValueError(mode)
    cases=[("medical","A",7),("legal","A",7),("finance","C",7)]
    rows=[decide(*case,mode,expert=mode=="reviewed") for case in cases]
    expected=["med-new","law-A",None]
    correct=sum(r["source"]==e for r,e in zip(rows,expected))
    return {"metrics":{"正确证据或拒答":correct,"测试场景":3,
                       "已专家复核":sum(r["status"]=="已复核" for r in rows),
                       "证据不足":sum(r["status"]=="证据不足" for r in rows)},
            "steps":[f"{case[0]} / 地区{case[1]} / 日期{case[2]}：{row}" for case,row in zip(cases,rows)]+[
                     "先验证来源、对象、辖区和有效期，再形成带引用的候选；缺证据不能用相似领域材料补。",
                     "专家复核是外部输入，模型不能自行把状态改成通过；这里 expert 标志由测试场景提供。",
                     "领域适配可用继续预训练/SFT/RAG，但验收需领域独立切分、错误代价、校准与人机职责，不只看考试准确率。"],
            "note":"所有地区、日期和规则均为虚构夹具，不提供医疗、法律或投资建议。未实现真实专家系统；这里验证资料和工作流边界。"}


def main():
    rows={m:experiment(m) for m in MODES}
    assert rows["naive"]["metrics"]["正确证据或拒答"]<3
    assert rows["evidence"]["metrics"]["正确证据或拒答"]==3
    assert rows["evidence"]["metrics"]["已专家复核"]==0
    assert decide("medical","A",11,"reviewed",True)["status"]=="证据不足"
    print(json.dumps(rows,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
