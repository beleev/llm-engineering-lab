"""相关探针、因果激活修补与实际训练的稀疏自编码器。"""
import json
from functools import lru_cache
import numpy as np
import torch
from torch import nn
MODES=("probe","patch","sae")


@lru_cache(maxsize=1)
def train_sae():
    with torch.random.fork_rng():
        torch.manual_seed(29)
        # 五个稀疏源投影进三维激活；真实语义未知，只能测重建与干预。
        dictionary=torch.nn.functional.normalize(torch.randn(5,3),dim=1)
        source=(torch.rand(128,5)<.12).float()*torch.rand(128,5)
        x=source@dictionary
        encoder=nn.Linear(3,8)
        decoder=nn.Linear(8,3,bias=False)
        opt=torch.optim.Adam(list(encoder.parameters())+list(decoder.parameters()),lr=.02)
        first=float(((decoder(torch.relu(encoder(x)))-x)**2).mean().detach())
        for _ in range(400):
            z=torch.relu(encoder(x))
            loss=((decoder(z)-x)**2).mean()+.025*z.mean()
            opt.zero_grad();loss.backward();opt.step()
            with torch.no_grad():
                decoder.weight.copy_(torch.nn.functional.normalize(decoder.weight,dim=0))
        with torch.no_grad():
            z=torch.relu(encoder(x))
            mse=float(((decoder(z)-x)**2).mean())
            ablated=z.clone();ablated[:,int(z.mean(0).argmax())]=0
            effect=float((decoder(ablated)-decoder(z)).abs().mean())
        assert mse<first
        return first,mse,float((z>1e-4).float().mean()),effect


def experiment(mode="patch"):
    if mode not in MODES:
        raise ValueError(mode)
    signal=np.array([-1.,1.,-1.,1.])
    hidden=np.column_stack([signal,signal])
    # 第二维与标签完美相关，但读出器只使用第一维。
    output=lambda h: h@np.array([2.,0.])
    clean=output(hidden)
    corrupt=hidden.copy();corrupt[:,0]*=-1
    restore=corrupt.copy();restore[:,0]=hidden[:,0]
    irrelevant=corrupt.copy();irrelevant[:,1]=0
    first,mse,active,effect=train_sae()
    return {"metrics":{"相关探针准确率":float(np.mean((hidden[:,1]>0)==(signal>0))),
                       "恢复因果通道后的误差":float(np.abs(output(restore)-clean).mean()),
                       "只改相关通道后的误差":float(np.abs(output(irrelevant)-clean).mean()),
                       "SAE初始MSE":round(first,5),"SAE终态MSE":round(mse,5),
                       "SAE激活比例":round(active,4),"消融特征输出变化":round(effect,5)},
            "steps":[{"probe":"第二通道能完全预测标签，但模型输出并不读取它。",
                      "patch":"先损坏第一通道，再恢复清洁激活；只恢复无因果作用的通道不能修复输出。",
                      "sae":"训练 ReLU 编码器与归一化字典解码器，用重建误差加 L1 稀疏项；随后实际消融一个特征。"}[mode],
                     "字典列归一化防止把 decoder 放大、encoder 缩小来钻 L1 的空子。",
                     "恢复与消融只支持本模型、此输入分布上的因果结论；off-manifold patch、冗余电路和特征多义性仍会误导。"],
            "note":"线性读出与合成激活。SAE 特征没有自动获得语义名称；注意力图、相关探针或可读 CoT 都不能单独证明模型机制。"}


def main():
    rows={m:experiment(m) for m in MODES}
    assert rows["patch"]["metrics"]["恢复因果通道后的误差"]==0
    assert rows["patch"]["metrics"]["只改相关通道后的误差"]>0
    print(json.dumps(rows,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
