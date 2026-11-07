"""参数稀疏、结构裁剪与算子融合分别节省什么。"""
import json
import numpy as np
MODES=("dense","unstructured","structured")


def experiment(mode="structured"):
    if mode not in MODES:
        raise ValueError(mode)
    rng=np.random.default_rng(21)
    x=rng.normal(size=(32,16))
    w=rng.normal(size=(16,8))
    w[8:]*=.001
    original=x@w
    if mode=="unstructured":
        kept=abs(w)>=np.quantile(abs(w),.5)
        pruned=w*kept
        output=x@pruned
        dense_macs=32*16*8  # 仍调用 dense matmul，0 不会自动少算。
    elif mode=="structured":
        keep=np.linalg.norm(w,axis=1)>=np.median(np.linalg.norm(w,axis=1))
        pruned=w*keep[:,None]
        output=x[:,keep]@w[keep]
        dense_macs=32*int(keep.sum())*8
        assert np.allclose(output,x@pruned)
    else:
        pruned=w
        output=original
        dense_macs=32*16*8
    bias=np.arange(8)/100
    scale=np.linspace(.8,1.2,8)
    # 三个逐元素算子 vs 一个表达式；只核对数学与理想流量，不冒充 fused kernel。
    eager=np.maximum(output+bias,0)*scale
    fused=np.maximum(output+bias,0)*scale
    assert np.array_equal(eager,fused)
    return {"metrics":{"非零权重":int(np.count_nonzero(pruned)),"dense乘加次数":dense_macs,
                       "输出MSE":round(float(np.mean((output-original)**2)),7),
                       "逐元素未融合字节":int(output.nbytes*6),
                       "理想融合字节":int(output.nbytes*2)},
            "steps":["非结构化剪枝只把一半权重设为零，dense GEMM 仍做原来的乘加。",
                     "结构化剪枝实际删输入通道，缩小矩阵；需把相邻层和残差维度一起处理。",
                     "融合把 add→ReLU→scale 的中间张量留在寄存器；本例核对数学等价并估计读写流量。",
                     "CUDA Graph 减少 host launch，不把这些算子变成一个 kernel；编译器融合还受形状、别名和数值顺序约束。"],
            "note":"numpy 的通道裁剪与理论流量。没有稀疏 GPU kernel 或融合 kernel 的实测加速；模型压缩还需看长尾、推理、安全切片。"}


def main():
    rows={m:experiment(m) for m in MODES}
    assert rows["unstructured"]["metrics"]["dense乘加次数"]==rows["dense"]["metrics"]["dense乘加次数"]
    assert rows["structured"]["metrics"]["dense乘加次数"] < rows["dense"]["metrics"]["dense乘加次数"]
    assert rows["structured"]["metrics"]["输出MSE"]<.001
    print(json.dumps(rows,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
