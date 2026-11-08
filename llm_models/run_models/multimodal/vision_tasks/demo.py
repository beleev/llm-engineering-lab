"""受控像素任务：字符识别、柱状图读数、定位计数与不确定拒答。"""
import json
import numpy as np
MODES=("clean","occluded","ambiguous")
GLYPHS={
 "0":["111","101","101","101","111"],
 "1":["010","110","010","010","111"],
 "2":["111","001","111","100","111"],
 "3":["111","001","111","001","111"]}


def glyph(name):
    return np.array([[int(v) for v in row] for row in GLYPHS[name]])


def recognize(image,max_error=2):
    if image.shape!=(5,3):
        raise ValueError("本例只支持 5×3 字形")
    scores=sorted((int(np.count_nonzero(image!=glyph(k))),k) for k in GLYPHS)
    error,label=scores[0]
    gap=scores[1][0]-error
    return label if error<=max_error and gap>=2 else None,error,gap


def chart():
    image=np.zeros((10,12),dtype=int)
    for i,h in enumerate([3,6,4]):
        image[10-h:,i*4+1:i*4+3]=1
    occupied=image.any(0)
    starts=np.flatnonzero(occupied & ~np.r_[False,occupied[:-1]])
    heights=[int(image[:,x].sum()) for x in starts]
    return starts,heights


def experiment(mode="clean"):
    if mode not in MODES:
        raise ValueError(mode)
    image=glyph("3")
    if mode=="occluded":
        image=image.copy();image[1:4]=0
    if mode=="ambiguous":
        image=(glyph("2")+glyph("3"))/2
    answer,error,gap=recognize(image)
    starts,heights=chart()
    return {"metrics":{"OCR回答":answer or "证据不足","像素差":error,"前两候选差":gap,
                       "柱子数量":len(heights),"读出高度":heights,
                       "最高柱序号":int(np.argmax(heights))+1},
            "steps":["从实际 5×3 像素计算模板距离；模糊或遮挡时不强行输出第一名。",
                     f"柱图通过前景连通列读出位置 {starts.tolist()}，再计算高度、数量与左右关系。",
                     "图表任务还需坐标轴刻度、单位、图例、截断轴与 OCR；这里每像素代表 1 单位的假设必须显式给出。",
                     "VQA/文档理解应分别测识别、布局、定位和推理，避免文字先验掩盖没有看图。"],
            "note":"已知字体和合成柱图的像素算法，不是通用 VLM 或真实 OCR；距离差只是拒答规则，不是校准后的正确概率。"}


def main():
    rows={m:experiment(m) for m in MODES}
    assert rows["clean"]["metrics"]["OCR回答"]=="3"
    assert rows["ambiguous"]["metrics"]["OCR回答"]=="证据不足"
    assert chart()[1]==[3,6,4]
    print(json.dumps(rows,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
