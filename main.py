#!/usr/bin/env python
"""
LLM Models - 主入口文件

模块化大语言模型教学库的总览入口:
1. 列出当前库支持的架构 (按 llm_models.models.MODEL_REGISTRY 的四类打印)
2. 列出可直接运行的示例脚本
3. 用几个最小前向做 "冒烟测试", 确保依赖 & 模块导入都正确

并非训练入口; 可运行示例在 llm_models/run_models/<category>/<model>/{infer_X,train_X}.py。
"""

import math
from pathlib import Path

import torch
import torch.nn.functional as F

from llm_models import (
    MultiHeadAttention,
    GPT3,
    LLaMA,
    Mixtral,
    Mamba,
    BERT,
    CLIPModel,
    Whisper,
    DiT,
    ImageVAE,
    DDPMScheduler,
)
from llm_models.models import MODEL_REGISTRY

def _check_lm(name: str, m, vocab: int = 100) -> None:
    """LM 冒烟断言: 初始 CE ≈ ln V (初始化健康) + 有/无 KV cache 贪心生成逐 token 相同。"""
    torch.manual_seed(0)
    m.eval()
    idx = torch.randint(1, vocab, (2, 9))                           # [B=2, 9]: 前 8 个当输入, 后 8 个当标签
    with torch.inference_mode():
        out = m(idx[:, :-1])                                        # [B, 8, V]
        logits = out[0] if isinstance(out, tuple) else out          # MoE 返回 (logits, routing)
        # 标签是输入右移一位: 位置 t 的目标是第 t+1 个 token
        ce = F.cross_entropy(logits.reshape(-1, vocab), idx[:, 1:].reshape(-1)).item()
    # 未训练的模型在 V 个 token 里近似均匀瞎猜, 交叉熵 ≈ ln V; 0.5 是给随机初始化留的余量
    assert abs(ce - math.log(vocab)) < 0.5, f"{name}: 初始 CE {ce:.2f} 偏离 ln V {math.log(vocab):.2f}"
    # temperature=0 是贪心解码, 没有随机性, 两条路径才能逐 token 比较
    a = m.generate(idx[:, :4], max_new_tokens=8, temperature=0, use_cache=True)
    b = m.generate(idx[:, :4], max_new_tokens=8, temperature=0, use_cache=False)
    assert torch.equal(a, b), f"{name}: KV cache 与无 cache 输出不一致"
    print(f"  {name:<8} 初始 CE {ce:.2f} ≈ ln V {math.log(vocab):.2f} | KV cache 一致  ✓")


def _smoke_attention() -> None:
    """MHA 前向一次, 打印输入输出形状。"""
    torch.manual_seed(42)
    attn = MultiHeadAttention(d_model=64, num_heads=4)
    x = torch.randn(1, 10, 64)
    out = attn(x)
    print(f"  MHA:     输入 {tuple(x.shape)} -> 输出 {tuple(out.shape)}  ✓ ")


def _smoke_gpt() -> None:
    """GPT-3: 初始 CE 与 KV cache 一致性 (见 _check_lm)。"""
    m = GPT3(vocab_size=100, d_model=64, n_heads=4, num_layers=2, max_len=32)
    _check_lm("GPT-3", m)


def _smoke_llama() -> None:
    """LLaMA (4 个 Q 头 / 2 个 KV 头的 GQA): 同上。"""
    m = LLaMA(vocab_size=100, d_model=64, n_heads=4, num_kv_heads=2,
              num_layers=2, max_len=32)
    _check_lm("LLaMA", m)


def _smoke_mixtral() -> None:
    """Mixtral (4 个专家, 每 token 选 2 个): 同上。forward 返回 (logits, routing)。"""
    m = Mixtral(vocab_size=100, d_model=64, n_heads=4, num_kv_heads=2,
                num_layers=2, num_experts=4, top_k=2, max_len=32)
    _check_lm("Mixtral", m)


def _smoke_mamba() -> None:
    """Mamba: 只查前向输出形状 [B, T, V]。"""
    m = Mamba(vocab_size=100, d_model=64, num_layers=2).eval()
    with torch.inference_mode():
        logits = m(torch.randint(0, 100, (1, 8)))
    print(f"  Mamba:   logits {tuple(logits.shape)}  ✓")


def _smoke_bert() -> None:
    """BERT: 只查前向输出形状 [B, T, V]。"""
    m = BERT(vocab_size=100, d_model=64, n_heads=4, num_layers=2, max_len=32).eval()
    with torch.inference_mode():
        logits = m(torch.randint(0, 100, (1, 8)))
    print(f"  BERT:    logits {tuple(logits.shape)}  ✓")


def _smoke_clip() -> None:
    """CLIP: 2 张 32×32 的图和 2 条文本, 打印两路特征的形状 [B, embed_dim]。"""
    m = CLIPModel(
        embed_dim=64, vocab_size=100,
        text_d_model=64, text_n_heads=4, text_num_layers=2, text_max_len=16,
        image_size=32, patch_size=8,
        vision_d_model=64, vision_n_heads=4, vision_num_layers=2,
    ).eval()
    with torch.inference_mode():
        out = m(torch.randn(2, 3, 32, 32), torch.randint(0, 100, (2, 8)))
    print(f"  CLIP:    img {tuple(out['image_features'].shape)} "
          f"txt {tuple(out['text_features'].shape)}  ✓")


def _smoke_whisper() -> None:
    """Whisper: 输入 mel [B, 80, 20] 和 4 个文本 token, 打印 logits 形状。"""
    m = Whisper(vocab_size=100, n_mels=80, d_model=64, n_heads=4,
                encoder_layers=1, decoder_layers=1,
                max_source_len=50, max_target_len=16).eval()
    with torch.inference_mode():
        logits = m(torch.randn(1, 80, 20), torch.randint(0, 100, (1, 4)))
    print(f"  Whisper: logits {tuple(logits.shape)}  ✓")


def _smoke_dit() -> None:
    """DiT: 给 x_0 加噪得到 x_t, 前向一次预测噪声 ε̂, 形状应与 x_t 相同。"""
    m = DiT(latent_channels=4, image_size=4, patch_size=2,
            d_model=32, n_heads=4, num_layers=2, num_classes=2).eval()
    scheduler = DDPMScheduler(num_train_timesteps=100)
    x0 = torch.randn(1, 4, 4, 4)
    t = scheduler.sample_timesteps(1, x0.device)                    # 随机抽一个时间步
    noised = scheduler.add_noise(x0, t)                             # .noisy 是 x_t, .t_norm 是喂给模型的 t
    with torch.inference_mode():
        pred = m(noised.noisy, noised.t_norm, torch.tensor([0]))
    print(f"  DiT:     x_t {tuple(noised.noisy.shape)} -> ε̂ {tuple(pred.shape)}  ✓")


def _smoke_vae() -> None:
    """ImageVAE: 前向一次, 打印重建 recon 和 latent z 的形状。"""
    m = ImageVAE(base_channels=16, levels=1).eval()
    with torch.inference_mode():
        out = m(torch.randn(1, 3, 16, 16))
    print(f"  VAE:     recon {tuple(out['recon'].shape)} z {tuple(out['z'].shape)}  ✓")


def _smoke_conditional_depth() -> None:
    """MoD / MoR: 初始 CE ≈ ln V。它们没有 generate() (跳过某层的 token 在那层没有 K/V), 不查 KV cache。"""
    from llm_models.models.language_models.conditional_depth import MoDLLaMA, MoRLLaMA
    torch.manual_seed(0)
    for name, m in (("MoD", MoDLLaMA(capacity=0.5, vocab_size=100, d_model=64, n_heads=4, num_layers=2, max_len=32)),
                    ("MoR", MoRLLaMA(num_recursions=2, vocab_size=100, d_model=64, n_heads=4, max_len=32))):
        x = torch.randint(0, 100, (4, 32))
        with torch.inference_mode():
            ce = F.cross_entropy(m.eval()(x[:, :-1]).flatten(0, 1), x[:, 1:].flatten()).item()
        assert abs(ce - math.log(100)) < 0.5, f"{name}: 初始 CE {ce:.2f} 偏离 ln V"
        print(f"  {name:<8} 初始 CE {ce:.2f} ≈ ln V {math.log(100):.2f}  ✓")


def _smoke_block_diffusion() -> None:
    """BD3-LM: 块因果掩码 + 训练掩码前向, 初始 ELBO ≈ ln V。"""
    from llm_models.models.language_models.block_diffusion import BlockDiffusionLM
    torch.manual_seed(0)
    m = BlockDiffusionLM(block_size=4, vocab_size=100, d_model=64, n_heads=4, num_layers=2, max_len=32)
    with torch.inference_mode():
        ce = m.elbo_loss(torch.randint(0, 99, (16, 16)), generator=torch.Generator().manual_seed(0)).item()
    assert abs(ce - math.log(100)) < 0.5, f"BD3-LM 初始 ELBO {ce:.2f} 偏离 ln V"
    print(f"  {'BD3-LM':<8} 初始 ELBO {ce:.2f} ≈ ln V {math.log(100):.2f}  ✓")


def _smoke_blt() -> None:
    """BLT: 按固定 4 字节切 patch 前向一次, 初始 CE ≈ ln 256。"""
    from llm_models.models.language_models.byte_latent import ByteLatentLM, fixed_patch_ids
    torch.manual_seed(0)
    m = ByteLatentLM(d_model=32, n_heads=2, global_layers=1, max_len=32).eval()
    x = torch.randint(0, 256, (2, 32))
    with torch.inference_mode():
        logits = m(x, fixed_patch_ids(2, 32, 4))
    ce = F.cross_entropy(logits[:, :-1].flatten(0, 1), x[:, 1:].flatten()).item()
    assert abs(ce - math.log(256)) < 0.5, f"BLT: 初始 CE {ce:.2f} 偏离 ln 256"
    print(f"  {'BLT':<8} 初始 CE {ce:.2f} ≈ ln 256 {math.log(256):.2f}  ✓")


def _smoke_transfusion() -> None:
    """Transfusion: 7 个位置里 4 个是图像 patch, 打印文本 logits 和图像速度预测的形状。"""
    from llm_models.models.multimodal.transfusion import TransfusionModel
    torch.manual_seed(0)
    m = TransfusionModel(vocab_size=10, patch_dim=12, d_model=32, n_heads=4, num_layers=1, max_len=16).eval()
    image_id = torch.tensor([[-1, -1, 0, 0, 0, 0, -1]])
    with torch.inference_mode():
        out = m(torch.randint(0, 10, (1, 7)), torch.randn(1, 7, 12), image_id, torch.tensor([500.0]))
    print(f"  Transfusion: logits {tuple(out['logits'].shape)} velocity {tuple(out['velocity'].shape)}  ✓")


def _smoke_hyper_connections() -> None:
    """HC / mHC: 加宽残差流在初始化时是恒等改动, 三种残差的初始 CE 都应 ≈ ln V。"""
    from llm_models.models.language_models.hyper_llama import HyperLLaMA
    cfg = dict(vocab_size=100, d_model=64, n_heads=4, num_layers=2, max_len=32)
    x = torch.randint(0, 100, (4, 32))
    for name in ("standard", "hc", "mhc"):
        torch.manual_seed(0)
        m = HyperLLaMA(residual=name, n_streams=4, **cfg).eval()
        with torch.inference_mode():
            logits = m(x[:, :-1])
        ce = F.cross_entropy(logits.flatten(0, 1), x[:, 1:].flatten()).item()
        assert abs(ce - math.log(100)) < 0.5, f"{name}: 初始 CE {ce:.2f} 偏离 ln V"
        print(f"  {name:<8} 初始 CE {ce:.2f} ≈ ln V {math.log(100):.2f}  ✓")


def _smoke_hybrid() -> None:
    """HybridLM: 三种排布各跑一次最小前向, 初始 CE ≈ ln V; 顺手核对缓存算式。"""
    from llm_models.models.language_models.hybrid import HybridLM
    from llm_models.utils.generation import KVCache
    torch.manual_seed(0)
    x = torch.randint(0, 100, (2, 32))
    for layout in ("kda kda kda mla_nope", "gdn gdn gdn gated_gqa", "gqa_chunk gqa_chunk gqa_chunk gqa_nope"):
        m = HybridLM(layout=layout, vocab_size=100, d_model=64, n_heads=4, max_len=64,
                     chunk_size=8, latent_dim=32, qk_rope_head_dim=16).eval()
        cache = KVCache(len(m.layers))
        with torch.inference_mode():
            logits = m(x[:, :-1])
            for t in range(31):
                m(x[:, t:t + 1], cache=cache)
        ce = F.cross_entropy(logits.flatten(0, 1), x[:, 1:].flatten()).item()
        got = sum(sum(v.numel() for v in c.values()) for c in cache.layers) // 2
        assert abs(ce - math.log(100)) < 0.5, f"{layout}: 初始 CE {ce:.2f} 偏离 ln V"
        assert got == m.cache_elements(31)["total"], f"{layout}: 缓存元素数与算式不符"
        print(f"  {'Hybrid':<8} {layout:<40} 初始 CE {ce:.2f} | 缓存 {got} 个元素  ✓")


def _smoke_deepseek_v4() -> None:
    """DeepSeek-V4: CSA / HCA 交错, 初始 CE ≈ ln V; 再查一次增量生成与整段前向一致。"""
    from llm_models.models.moe.deepseek_v4 import DeepSeekV4
    from llm_models.utils.generation import KVCache
    torch.manual_seed(0)
    m = DeepSeekV4(vocab_size=100, d_model=64, n_heads=4, num_layers=4, max_len=64, dropout=0.0,
                   num_routed_experts=4, num_shared_experts=1, top_k=2,
                   latent_dim=32, qk_rope_head_dim=16, csa_block=4, hca_block=16,
                   window=8, csa_top_k=4).eval()
    x = torch.randint(0, 100, (2, 32))
    with torch.inference_mode():
        full = m(x[:, :-1])[0]
        cache = KVCache(len(m.layers))
        step = torch.cat([m(x[:, t:t + 1], cache=cache)[0] for t in range(31)], dim=1)
    ce = F.cross_entropy(full.flatten(0, 1), x[:, 1:].flatten()).item()
    gap = (full - step).abs().max().item()
    assert abs(ce - math.log(100)) < 0.5, f"DeepSeek-V4: 初始 CE {ce:.2f} 偏离 ln V"
    assert gap < 1e-4, f"DeepSeek-V4: 增量生成与整段前向差 {gap:.1e}"
    print(f"  {'V4':<8} 层排布 {m.pattern} 初始 CE {ce:.2f} ≈ ln V {math.log(100):.2f} | 增量差 {gap:.1e}  ✓")


def main():
    print("=" * 70)
    print("LLM Models - 模块化大语言模型 & 生成模型教学库")
    print("=" * 70)

    # 架构清单不写死: 从模型注册表生成, 加模型只改 MODEL_REGISTRY
    print(f"\n架构 (共 {sum(len(m) for m in MODEL_REGISTRY.values())} 个):")
    for category, models in MODEL_REGISTRY.items():
        print(f"  [{category}] " + " | ".join(models))

    print("\n示例脚本 (python -m llm_models.run_models.<category>.<model>.<script>):")
    # 示例清单不写死: 扫 run_models/ 下的两级目录 (类别 / 模型), 跳过 _ 开头的
    root = Path(__file__).parent / "llm_models" / "run_models"
    for cat in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        models = sorted(p.name for p in cat.iterdir() if p.is_dir() and not p.name.startswith("_"))
        print(f"  {cat.name}: " + " | ".join(models))

    print("\n冒烟测试 (每个核心模型跑一次最小前向):")
    _smoke_attention()
    _smoke_bert()
    _smoke_gpt()
    _smoke_llama()
    _smoke_mixtral()
    _smoke_deepseek_v4()
    _smoke_mamba()
    _smoke_conditional_depth()
    _smoke_block_diffusion()
    _smoke_blt()
    _smoke_hyper_connections()
    _smoke_hybrid()
    _smoke_clip()
    _smoke_whisper()
    _smoke_transfusion()
    _smoke_vae()
    _smoke_dit()

    print("\n安装:  pip install -e .")
    print("更多: 见 README.md")


if __name__ == "__main__":
    main()
