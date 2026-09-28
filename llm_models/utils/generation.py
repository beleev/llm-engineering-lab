"""
自回归生成 + KV cache — 全库 decoder-only LM 共用的一份 generate()

是什么: `KVCache` (每层一个可变 dict) + `GenerationMixin.generate()`。
解决什么: 朴素生成每步把整个前缀重算一遍, 生成 C 个 token 要 O(C·T²);
          过去 token 的 K/V 不会再变, 缓存后每步只算 1 个新 token → O(C·T)。
协议:
    - attention 模块收到 `cache: dict` 时, 把本步 K/V (RoPE 之后) 追加进去, 再对全部缓存做注意力
      GQA: cache["k"], cache["v"] [B, Hkv, S, Dh];  MLA: cache["c_kv"], cache["k_rope"];
      线性注意力: cache["state"] (O(1) 递推状态)
    - 模型 forward(idx, ..., cache=None): past = cache.pos; position_ids = arange(past, past+T);
      mask 取因果 mask 的行 [past:past+T]、列 [:past+T]; 结尾 cache.pos += T
读代码时盯住: `cache.pos` —— 它同时决定 RoPE 位置和 mask 切片, 错一位输出就和无 cache 不一致。
"""

import time
from typing import List, Optional

import torch
import torch.nn.functional as F


class KVCache:
    """layers[i] 是第 i 层 attention 自己读写的 dict; pos 是已缓存的 token 数。"""

    def __init__(self, num_layers: int):
        self.layers: List[dict] = [{} for _ in range(num_layers)]
        self.pos: int = 0


def _logits_of(out) -> torch.Tensor:
    """forward 可能返回 Tensor / tuple (logits, aux...) / dict {"logits": ...}。"""
    if isinstance(out, dict):
        return out["logits"]
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


class GenerationMixin:
    """
    要求宿主模型有: `layers` (len = 层数), `max_len`, `forward(idx, cache=None)`。
    """

    # no_grad 而不是 inference_mode: 后者返回的 inference tensor 不能参与带梯度的前向,
    # RL (GRPO / PPO / on-policy 蒸馏) 拿采样结果直接训练时就得先 .clone()
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,                # [B, P] prompt; 批量长短不一时左 pad
        max_new_tokens: int,
        temperature: float = 1.0,         # 0 ⇒ 贪心 argmax
        top_k: Optional[int] = None,
        use_cache: bool = True,
        attention_mask: Optional[torch.Tensor] = None,   # [B, P] 1=真 token 0=pad; 需要宿主 forward 接受该参数
        eos_token_id: Optional[int] = None,              # 某条序列生成 EOS 后只补 pad; 全部结束提前退出
        pad_token_id: Optional[int] = None,              # EOS 之后补的 token, 默认 = eos_token_id
    ) -> torch.Tensor:                    # [B, P + 实际生成步数 (≤ max_new_tokens)]
        """
        左 pad + attention_mask: pad 仍占绝对位置, 但 RoPE 只看相对位置, 所以真 token 的输出与不 pad 一致;
        学习式绝对位置编码 (GPT3) 不满足这一点。
        """
        was_training = self.training
        self.eval()
        max_len = self.max_len
        cache = KVCache(len(self.layers)) if use_cache else None
        if pad_token_id is None:
            pad_token_id = eos_token_id
        finished = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)

        for _ in range(max_new_tokens):
            if cache is None:
                inp = idx[:, -max_len:]                       # 无 cache: 每步重算整个 (裁剪后的) 前缀
            elif cache.pos == 0 or cache.pos + 1 > max_len:
                # prefill; 或上下文已满: 窗口左移后所有绝对位置都变了, 旧 cache 作废, 重新 prefill
                cache = KVCache(len(self.layers))
                inp = idx[:, -max_len:]
            else:
                inp = idx[:, -1:]                             # decode: 只喂 1 个新 token

            kwargs = {}
            if attention_mask is not None:
                kwargs["attention_mask"] = attention_mask[:, -max_len:]   # 三个分支下都恰好是 [B, past+T]
            if cache is not None:
                kwargs["cache"] = cache
            logits = _logits_of(self(inp, **kwargs))[:, -1, :]            # [B, V] 只要最后一个位置

            if temperature == 0:
                idx_next = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                idx_next = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            if eos_token_id is not None:
                idx_next = idx_next.masked_fill(finished.unsqueeze(1), pad_token_id)
                finished |= idx_next.squeeze(1) == eos_token_id
            idx = torch.cat([idx, idx_next], dim=1)           # [B, P+1]
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask, attention_mask.new_ones(idx.size(0), 1)], dim=1)
            if eos_token_id is not None and bool(finished.all()):
                break

        self.train(was_training)
        return idx


def benchmark_kv_cache(model, prompt: torch.Tensor, max_new_tokens: int) -> float:
    """贪心生成两遍 (无 cache / 有 cache), 断言逐 token 完全相同, 返回加速比 t_无 / t_有。"""
    t0 = time.perf_counter()
    slow = model.generate(prompt, max_new_tokens, temperature=0, use_cache=False)
    t1 = time.perf_counter()
    fast = model.generate(prompt, max_new_tokens, temperature=0, use_cache=True)
    t2 = time.perf_counter()
    assert torch.equal(slow, fast), "KV cache 改变了输出: 检查 position_ids / mask 切片是否和 cache.pos 对齐"
    return (t1 - t0) / (t2 - t1)
