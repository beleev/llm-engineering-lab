#!/usr/bin/env python
"""
LLaMA 推理示例 — 五个可以用 assert 证伪的性质

    1. KV cache: 有/无 cache 的贪心生成逐 token 相同; GQA 让 cache 比 MHA 小 H/Hkv 倍
    2. 左 padding: pad 行在 "因果 ∩ padding" 后一个 key 都不剩, 不兜底的话 softmax 出 NaN;
       这里断言无 NaN, 且真实位置的 logits 与不 padding 时一致 (RoPE 只看相对位置)
    3. QK-Norm: 把 W_q, W_k 放大 10 倍, 无 QK-Norm 时 logit 放大 100 倍, 有 QK-Norm 时有界 ≤ sqrt(Dh)
    4. attention sink: sink logit → -∞ 时退化为普通 GQA; = 0 时每行注意力之和 < 1; 与 cache 兼容
    5. llm_finetune 依赖的接口: return_hidden 隐状态 / 左 pad 批量生成 / EOS 停 / 返回值可直接训练 /
       Trainer 不收冻结参数、batch 可以没有 labels
"""

import copy
import math

import torch
import torch.nn as nn

from llm_models.layers.core.attention import GroupedQueryAttention
from llm_models.models.language_models.llama import LLaMA
from llm_models.training import Trainer, TrainingConfig
from llm_models.training.data import SyntheticDataGenerator
from llm_models.training.loss import LossComputer
from llm_models.utils.generation import benchmark_kv_cache


def max_attn_logit(attn: GroupedQueryAttention, x: torch.Tensor) -> float:
    """复现 GQA 内部的 QK^T·scale (不加 RoPE: 旋转不改范数, 不影响上界)。"""
    B, T, _ = x.shape
    Q = attn.w_q(x).view(B, T, attn.num_heads, attn.head_dim).transpose(1, 2)      # [B, H, T, Dh]
    K = attn.w_k(x).view(B, T, attn.num_kv_heads, attn.head_dim).transpose(1, 2)   # [B, Hkv, T, Dh]
    if attn.q_norm is not None:
        Q, K = attn.q_norm(Q), attn.k_norm(K)
    K = K.repeat_interleave(attn.num_groups, dim=1)                                # [B, Hkv, T, Dh] → [B, H, T, Dh]
    return float((Q @ K.transpose(-2, -1) * attn.scale).abs().max())


@torch.inference_mode()
def main():
    torch.manual_seed(42)
    vocab_size = 1000
    cfg = dict(vocab_size=vocab_size, d_model=256, n_heads=8, num_kv_heads=2,
               num_layers=4, max_len=256, dropout=0.0)
    model = LLaMA(**cfg).eval()
    print(f"LLaMA Mini | 参数量: {sum(p.numel() for p in model.parameters()):,} "
          f"({cfg['n_heads']} Q heads / {cfg['num_kv_heads']} KV heads)")

    # ---- 1) KV cache ----
    idx = torch.randint(1, vocab_size, (2, 16))                          # [B=2, T=16]
    assert model(idx).shape == (2, 16, vocab_size), "logits 形状应为 [B, T, V]"
    n_gen = 200
    speedup = benchmark_kv_cache(model, idx[:, :8], max_new_tokens=n_gen)
    a = model.layers[0].attn                                             # 第 0 层的 GQA, 用来读 Hkv / Dh
    print(f"[1] KV cache: 生成 {n_gen} token 输出完全一致, 加速 {speedup:.1f}x; "
          f"每 token 每层缓存 2·Hkv·Dh = {2 * a.num_kv_heads * a.head_dim} 个数 "
          f"(MHA 要 {2 * a.num_heads * a.head_dim})")

    # ---- 2) 左 padding ----
    P = 5                                                                # 左边补的 pad 个数
    real = idx[:1]                                                       # [1, 16]
    padded = torch.cat([torch.zeros(1, P, dtype=torch.long), real], dim=1)  # [1, P + 16], pad id = 0
    attn_mask = torch.cat([torch.zeros(1, P), torch.ones(1, 16)], dim=1)  # [1, P + 16], 0 = pad
    out_pad = model(padded, attention_mask=attn_mask)
    assert not out_pad.isnan().any(), "左 padding 产生 NaN: 全屏蔽行没有被兜底"
    # [:, P:] 跳过 pad 段, 拿真实 token 的 logits 与 "不 padding 直接前向" 比
    diff = float((out_pad[:, P:] - model(real)).abs().max())
    assert diff < 1e-4, f"padding 改变了真实位置的输出: {diff}"
    print(f"[2] 左 padding {P} 位: 无 NaN, 真实位置 logits 最大偏差 {diff:.1e}")

    # ---- 3) QK-Norm ----
    #      同一个 x 算两次 max|logit|: 放大权重之前一次, W_q、W_k 各 ×10 之后一次
    x = torch.randn(2, 16, 256)                                          # [B, T, D]
    k = 10                                                               # W_q、W_k 各放大的倍数
    for qk_norm in (False, True):
        attn = GroupedQueryAttention(256, 8, 2, qk_norm=qk_norm)
        before = max_attn_logit(attn, x)
        attn.w_q.weight.mul_(k)
        attn.w_k.weight.mul_(k)
        after = max_attn_logit(attn, x)
        print(f"[3] qk_norm={qk_norm!s:5}: max|logit| {before:8.2f} → 权重×{k} → {after:8.2f}")
        if qk_norm:
            bound = math.sqrt(attn.head_dim)       # |q̂|=|k̂|=sqrt(Dh) → |q̂·k̂|/sqrt(Dh) ≤ sqrt(Dh)
            assert after <= bound + 1e-3, "QK-Norm 后 |logit| 应 ≤ sqrt(Dh)"
            assert abs(after - before) < 1e-3, "QK-Norm 下把权重放大 10 倍不应改变 logit (范数被归一化掉了)"
        else:
            # 100× 放大 → softmax 饱和成 one-hot, 梯度消失; 阈值取 50× 留余量
            assert after > 50 * before, "无 QK-Norm 时 W_q, W_k 各放大 10 倍, logit 应放大约 100 倍"

    # ---- 4) attention sink ----
    plain = GroupedQueryAttention(256, 8, 2)
    sink = GroupedQueryAttention(256, 8, 2, use_sink=True)
    sink.load_state_dict(plain.state_dict(), strict=False)                    # 两者共用同一套权重, 只差 sink 参数
    causal = torch.ones(16, 16).tril().bool()                                 # [T, T] 下三角
    out_sink0 = sink(x, mask=causal)                                          # sink logit 初始为 0
    assert not torch.allclose(out_sink0, plain(x, mask=causal), atol=1e-4), \
        "sink = 0 时应分走一部分概率质量, 输出应与普通 GQA 不同"
    sink.sink.fill_(-1e4)                                                     # sink → -∞: 退化为普通 GQA
    assert torch.allclose(sink(x, mask=causal), plain(x, mask=causal), atol=1e-5), \
        "sink logit → -∞ 时应退化为普通 GQA"
    sink.sink.zero_()
    cache, steps = {}, []
    for t in range(16):                                                       # 逐 token 带 cache 解码
        # 第 t 步只送 1 个 query; mask 取第 t 行的前 t+1 列, 对应 cache 里已有的 key 加上自己
        steps.append(sink(x[:, t:t + 1], mask=causal[t:t + 1, : t + 1], cache=cache))
    assert torch.allclose(torch.cat(steps, dim=1), out_sink0, atol=1e-5), \
        "带 sink 的逐 token cache 解码应与整段前向一致"
    print("[4] attention sink: sink→-∞ 等价普通 GQA; sink=0 改变输出; 逐 token cache 解码与整段前向一致")


def check_finetune_api():
    """不在 inference_mode 下跑: 要验证 generate 的返回值能直接参与带梯度的前向。"""
    torch.manual_seed(0)
    V = 50
    model = LLaMA(vocab_size=V, d_model=64, n_heads=4, num_kv_heads=2, num_layers=2, max_len=64).eval()
    with torch.no_grad():   # 初始化的小权重下输出几乎只看最后一个 token, 测不出 mask 有没有生效; 放大 10 倍让它依赖上下文
        for p in model.parameters():
            if p.dim() == 2:
                p.mul_(10)

    # 5a) return_hidden 与 "把 lm_head 换成 Identity" 逐元素一致
    idx = torch.randint(1, V, (3, 10))
    mask = torch.ones(3, 10)
    mask[1, 7:] = 0                                                      # 右 pad 也要一致
    old = copy.deepcopy(model)                                           # 对照组: 同一套权重, 只把 lm_head 换掉
    old.lm_head = nn.Identity()
    with torch.no_grad():
        h = model(idx, mask, return_hidden=True)                         # [B, T, D]
        assert h.shape == (3, 10, 64), "return_hidden 应返回 [B, T, d_model] 的隐状态"
        assert torch.equal(h, old(idx, mask)), "return_hidden 与 Identity 方式不一致"
        assert torch.equal(model.lm_head(h), model(idx, mask)), "隐状态再过 lm_head 应正好得到 logits"

    # 5b) 左 pad + attention_mask 的批量贪心生成 == 逐条单独生成 (有 / 无 cache)
    prompts = [torch.randint(1, V, (n,)) for n in (3, 7, 5)]             # 三条长度不同的 prompt
    P = max(len(q) for q in prompts)                                     # 补齐后的长度
    batch = torch.stack([torch.cat([torch.zeros(P - len(q), dtype=torch.long), q]) for q in prompts])   # [3, P]
    # 每行最后 len(q) 个位置是 1, 左边的 pad 是 0 → [3, P]
    bmask = (torch.arange(P) >= P - torch.tensor([len(q) for q in prompts]).unsqueeze(1)).long()
    for use_cache in (True, False):
        out = model.generate(batch, 12, temperature=0, attention_mask=bmask, use_cache=use_cache)
        for i, q in enumerate(prompts):
            single = model.generate(q[None], 12, temperature=0, use_cache=use_cache)   # q[None]: [1, len(q)]
            # 两边都只比新生成的 12 个 token: 批量的从第 P 位起, 单条的从第 len(q) 位起
            assert torch.equal(out[i, P:], single[0, len(q):]), f"左 pad 批量生成第 {i} 条与单独生成不一致"
    assert not torch.equal(model.generate(batch, 12, temperature=0), out), "不给 mask 时 pad 被当成真 token, 结果本应不同"

    # 5c) EOS: 取第 0 条贪心输出的第 3 个新 token 当 EOS → 这条在那里停, 之后全是 pad; 全部结束就提前退出
    plain = model.generate(batch, 12, temperature=0, attention_mask=bmask)
    eos = int(plain[0, P + 2])                                           # P + 2 = 第 3 个新 token 的下标
    stopped = model.generate(batch, 12, temperature=0, attention_mask=bmask, eos_token_id=eos, pad_token_id=0)
    for i in range(len(prompts)):
        # stopped 可能比 plain 短 (全部结束就提前退出), ref 截到同样长度再比
        gen, ref = stopped[i, P:], plain[i, P:P + stopped.size(1) - P]
        hit = (ref == eos).nonzero()
        k = int(hit[0]) + 1 if len(hit) else len(ref)                   # 含 EOS 的有效长度
        assert torch.equal(gen[:k], ref[:k]), f"第 {i} 条: 到 EOS 为止的 token 应与不设 EOS 时相同"
        assert (gen[k:] == 0).all(), f"第 {i} 条 EOS 之后没有停: 后面应全是 pad (0)"
    k0 = int((plain[0, P:] == eos).nonzero()[0]) + 1                    # 第 0 条首次出现 EOS (≤ 3)
    solo = model.generate(batch[:1], 12, temperature=0, attention_mask=bmask[:1], eos_token_id=eos)
    assert solo.size(1) == P + k0, "唯一一条序列遇到 EOS 应提前退出: 总长应为 prompt 加上到 EOS 为止的 token 数"
    assert solo[0, -1] == eos, "提前退出时最后一个 token 应是 EOS"

    # 5d) 返回的是普通张量: 不 clone 就能做带梯度的前向
    assert not stopped.is_inference(), "generate 返回了 inference tensor: 不 clone 就没法参与带梯度的前向"
    model.train()
    model(stopped).sum().backward()                                      # 不报错即通过

    # 5e) Trainer: optimizer 只收可训参数; batch 没有 labels 时传 None 给 LossComputer
    #     先全部冻结, 再只放开最后一层: 模拟微调时只训一部分参数
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.layers[-1].parameters():
        p.requires_grad_(True)

    class NoLabels(SyntheticDataGenerator):
        """只给 idx、不给 labels 的数据: 用来验证 batch 里可以没有 labels。"""

        def _sample(self):
            return {"idx": idx}

    class MeanLogit(LossComputer):
        """不需要 labels 的 loss: 直接取 logits 的均值, 只为让反传能跑。"""

        def compute(self, model_output, labels, **kwargs):
            assert labels is None, "batch 里没有 labels 时, Trainer 应把 None 传给 LossComputer"
            return {"total_loss": model_output.mean()}

    trainer = Trainer(model, TrainingConfig(num_steps=1, warmup_steps=1), NoLabels(), MeanLogit())
    in_opt = {id(p) for g in trainer.optimizer.param_groups for p in g["params"]}   # optimizer 实际收到的参数
    assert in_opt == {id(p) for p in model.parameters() if p.requires_grad}, "optimizer 里混进了冻结参数"
    trainer.train_step()
    # 调度按 config.num_steps 建, 走满之后余弦掉头、lr 回升, 所以再走一步必须报错
    try:
        trainer.train_step()
        overran = True
    except ValueError:
        overran = False
    assert not overran, "走满 config.num_steps 后 train_step 应抛 ValueError, 否则 lr 会回升"
    print(f"[5] finetune 接口: return_hidden ≡ Identity 方式; 左 pad 批量 ≡ 逐条生成; EOS 后补 pad 并提前退出; "
          f"返回值可直接反传; optimizer 只含 {len(in_opt)} 个可训参数; 无 labels 的 batch 可训练")


if __name__ == "__main__":
    main()
    check_finetune_api()
