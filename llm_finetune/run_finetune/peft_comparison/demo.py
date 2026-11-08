"""冻结同一注意力基座，分别训练输入 soft prompt、层内 KV prefix、Adapter 与 LoRA。"""
import json
import math
import torch
from torch import nn
MODES = ("prompt", "prefix", "adapter", "lora")


class TinyAdaptation(nn.Module):
    def __init__(self, mode, d=8):
        super().__init__()
        if mode not in MODES:
            raise ValueError(mode)
        self.mode = mode
        self.register_buffer("qkv", torch.randn(3, d, d)/math.sqrt(d))
        self.register_buffer("out", torch.randn(d, 1)/math.sqrt(d))
        if mode == "prompt":
            self.prompt = nn.Parameter(torch.randn(2, d)*.02)
        elif mode == "prefix":
            self.prefix_k = nn.Parameter(torch.randn(2, d)*.02)
            self.prefix_v = nn.Parameter(torch.randn(2, d)*.02)
        else:
            self.a = nn.Parameter(torch.randn(d, 2)*.02)
            self.b = nn.Parameter(torch.zeros(2, d))

    def forward(self, x):
        if self.mode == "prompt":
            x = torch.cat([self.prompt.expand(len(x), -1, -1), x], 1)
        q, k, v = (x @ matrix for matrix in self.qkv)
        # 只读最后位置；它能看见全部前缀，没有未来位置泄漏。
        q = q[:, -1:]
        if self.mode == "prefix":
            k = torch.cat([self.prefix_k.expand(len(x), -1, -1), k], 1)
            v = torch.cat([self.prefix_v.expand(len(x), -1, -1), v], 1)
        elif self.mode == "lora":
            v = v + x @ self.a @ self.b
        h = (torch.softmax(q @ k.transpose(1,2)/math.sqrt(k.shape[-1]), -1) @ v)[:,0]
        if self.mode == "adapter":
            h = h + torch.relu(h @ self.a) @ self.b
        return h @ self.out


def experiment(mode="lora"):
    with torch.random.fork_rng():
        torch.manual_seed(8)
        model = TinyAdaptation(mode)
        frozen = model.qkv.clone()
        x = torch.randn(32, 3, 8, generator=torch.Generator().manual_seed(81))
        y = x[:, -1, :2].sum(-1, keepdim=True)
        opt = torch.optim.Adam(model.parameters(), lr=.035)
        first = float(((model(x)-y)**2).mean().detach())
        for _ in range(160):
            loss = ((model(x)-y)**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        last = float(((model(x)-y)**2).mean().detach())
        assert torch.equal(model.qkv, frozen)
        assert last < first
        return {"metrics": {"可训练参数": sum(p.numel() for p in model.parameters()),
                            "初始MSE": round(first,4), "训练后MSE": round(last,4),
                            "冻结基座未变": torch.equal(model.qkv, frozen)},
                "steps": ["prompt 学输入向量，经冻结 Q/K/V 投影；prefix 直接学当前层的 K/V。",
                          "Adapter 在输出残差内插入瓶颈非线性；LoRA 给 V 投影增加低秩分支。",
                          "真实多层 prefix 要为每层提供状态。本例只有一个注意力层；参数量不相等，不能按最终 loss 宣称公平排名。"],
                "note": "固定 32 条合成样本验证参数路径与梯度。没有留出泛化结论；Prompt Tuning 与提示词写作不是一回事。"}


def main():
    print(json.dumps({m: experiment(m) for m in MODES}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
