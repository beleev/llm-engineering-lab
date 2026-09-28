"""
扩散 / 流匹配: 调度器 (正向加噪 + loss target) + 采样器 (反向去噪) + Loss + CFG

DDPM:          x_t = sqrt(ᾱ_t)·x_0 + sqrt(1-ᾱ_t)·ε,  target = ε,       t ∈ {0..T-1}
Flow Matching: x_t = (1-t)·x_0 + t·ε,                target = ε - x_0, t ∈ [0, 1]
它解决的问题: Scheduler 与 Sampler 解耦 (diffusers 风格), 同一个 DiT 可换目标 / 换采样器。

关键数字: 喂给模型的时间统一是 **[0, T=1000) 量纲**。sinusoidal 频率族为 max_period=10000
的整数位置设计, 若把 t∈[0,1] 直接喂进去, t=0.1 与 t=0.9 的嵌入余弦相似度 0.98 (几乎不可分);
×1000 后降到 0.17。所以 Flow Matching 的 t 在出本文件前一律 ×num_train_timesteps (SD3 同款)。

读代码时盯住: AddNoiseResult.t_norm (给模型看的 t) 与 add_noise 内部用的 t (插值系数) 的区别。
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_models.training.loss import LossComputer


# -----------------------------------------------------------------------------
# Schedulers
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class AddNoiseResult:
    """
    正向加噪 q(x_t | x_0) 的输出包:
        noisy:  x_t
        noise:  ε
        target: loss 的回归目标 (DDPM: ε; Flow Matching: velocity = ε - x_0)
        t_norm: **喂给 model.forward 的时间**, 统一为 [0, T) 量纲
                (DDPM: 原始整数步; Flow Matching: t·T)。
                名字带 norm, 但取值范围不是 [0,1]。
    形状: noisy / noise / target 与 x_0 相同, t_norm 是 [B]。
    """
    noisy: torch.Tensor
    noise: torch.Tensor
    target: torch.Tensor
    t_norm: torch.Tensor


class NoiseScheduler(ABC):
    """
    抽象调度器: 规定怎么采训练用的 t、怎么加噪、loss 的回归目标是什么。
    反向去噪的每一步不在这里, 在 Sampler 里。
    """

    prediction_type: str  # "epsilon" | "velocity"

    @abstractmethod
    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """从调度器定义的分布中采训练时的 timestep。"""

    @abstractmethod
    def add_noise(self, x0: torch.Tensor, t: torch.Tensor) -> AddNoiseResult:
        """正向加噪 q(x_t | x_0), 产出 loss target。"""


class DDPMScheduler(NoiseScheduler):
    """
    DDPM (Ho et al., 2020) cosine β 调度 + ε-prediction

    加噪: x_t = sqrt(ᾱ_t) · x_0 + sqrt(1 - ᾱ_t) · ε
    target = ε

    为什么用 cosine 调度 (Nichol & Dhariwal, 2021)?
        linear β 在 T 较大时端点噪声过大, cosine 调度让 ᾱ_t 更平滑, 生成质量更好。

    符号: β_t = 第 t 步加的噪声方差, α_t = 1 - β_t, ᾱ_t = α_1·…·α_t (x_0 还剩多少信号)。
    s = 0.008 是 cosine 调度论文的取值: 一个小偏移, 让 t=0 附近的 β 不至于太小。
    """

    prediction_type = "epsilon"

    def __init__(self, num_train_timesteps: int = 1000, s: float = 0.008):
        self.num_train_timesteps = num_train_timesteps

        # cosine ᾱ_t = f(t)^2 / f(0)^2, f(t) = cos((t/T + s)/(1+s) · π/2)
        t = torch.arange(num_train_timesteps + 1, dtype=torch.float) / num_train_timesteps  # [T+1], 0..1
        alpha_bar = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]             # 除以 f(0)², 让 ᾱ_0 = 1

        # ᾱ_t = ᾱ_{t-1}·(1 - β_t) ⇒ β_t = 1 - ᾱ_t / ᾱ_{t-1}; 相邻两项相除, [T+1] → [T]
        betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
        # 截到 0.999: t → T 时 ᾱ → 0, 比值会让 β 贴近 1, 最后几步出现奇点
        betas = betas.clamp(max=0.999)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)    # [T] 用截断后的 β 重新累乘出 ᾱ_t

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)

    def to(self, device: torch.device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        return self

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        # DDPM 训练时均匀采 t ∈ [0, T-1], 整数, [B]
        return torch.randint(0, self.num_train_timesteps, (batch_size,), device=device)

    def _broadcast(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """把 [B] 张量扩到 ref 的维度 (广播到 [B, 1, 1, ...]) 以便和空间张量相乘。"""
        while x.dim() < ref.dim():
            x = x.unsqueeze(-1)
        return x

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor) -> AddNoiseResult:
        """x0 [B, ...], t [B] 整数步 → AddNoiseResult。一步到位算出 x_t, 不用逐步加噪。"""
        noise = torch.randn_like(x0)
        # 按每个样本的 t 查表, 再补维度好和 x0 相乘: [B] → [B, 1, 1, ...]
        sqrt_ab = self._broadcast(self.sqrt_alphas_cumprod[t], x0)                 # sqrt(ᾱ_t)
        sqrt_1mab = self._broadcast(self.sqrt_one_minus_alphas_cumprod[t], x0)     # sqrt(1 - ᾱ_t)
        noisy = sqrt_ab * x0 + sqrt_1mab * noise
        # DDPM 的 t 本来就是 [0, T) 的整数步, 不需要再缩放
        return AddNoiseResult(noisy=noisy, noise=noise, target=noise, t_norm=t.float())


class FlowMatchingScheduler(NoiseScheduler):
    """
    Rectified Flow / Flow Matching (SD3, FLUX)

    线性路径: x_t = (1 - t) · x_0 + t · ε,   t ∈ [0, 1]
    target = velocity = dx_t/dt = ε - x_0

    一句话差异对比:
        DDPM: 学噪声 ε, 用 cosine 调度加噪, 推理数十步
        Flow Matching: 学 velocity v, 用线性直线路径, 推理数步即可 (轨迹更直)

    训练时 t 的分布:
        原论文用 logit-normal (偏重中段 t), 教学用均匀分布, 实现更简单。
    """

    prediction_type = "velocity"

    def __init__(self, num_train_timesteps: int = 1000):
        self.num_train_timesteps = num_train_timesteps

    def to(self, device: torch.device):
        return self

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        # 直接采 [0, 1) 的连续值, [B]; 线性路径对任意实数 t 都有定义, 不需要离散成整数步
        return torch.rand(batch_size, device=device)

    @staticmethod
    def _broadcast(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """[B] → [B, 1, 1, ...], 补到和 ref 一样多的维度。"""
        while x.dim() < ref.dim():
            x = x.unsqueeze(-1)
        return x

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor) -> AddNoiseResult:
        """x0 [B, ...], t [B] ∈ [0, 1] → AddNoiseResult。"""
        noise = torch.randn_like(x0)
        t_b = self._broadcast(t, x0)                # [B] → [B, 1, 1, ...]
        noisy = (1 - t_b) * x0 + t_b * noise        # 插值系数用 t ∈ [0, 1]
        velocity = noise - x0                       # dx_t/dt, 与 t 无关 (直线路径)
        # 给模型看的 t 要 ×T: sinusoidal 嵌入在 [0,1] 上几乎不动 (见文件头)
        return AddNoiseResult(noisy=noisy, noise=noise, target=velocity,
                              t_norm=t * self.num_train_timesteps)


# -----------------------------------------------------------------------------
# Loss
# -----------------------------------------------------------------------------


class DiffusionLoss(LossComputer):
    """
    MSE(pred, target)。target 是 ε 还是 velocity 由 scheduler 在造数据时就定了
    (labels = AddNoiseResult.target), 所以这里不需要按 prediction_type 分支。

    可选 kwargs["loss_mask"]: 可广播到 labels 形状, 1=计入 / 0=排除
    (如 inpainting 只对被遮住的区域算 loss); 按有效元素数取平均。

    sanity: FinalLayer 零初始化 → 初始 pred=0 → 初始 loss = E[target²]
            (DDPM ≈ 1.0, Flow Matching ≈ Var(ε)+Var(x_0) ≈ 2.0)。
    """

    def compute(
        self,
        model_output: torch.Tensor,
        labels: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        loss_mask = kwargs.get("loss_mask")
        if loss_mask is None:
            loss = F.mse_loss(model_output, labels)
        else:
            m = loss_mask.expand_as(labels).to(labels.dtype)   # 0/1 掩码, 扩到和 labels 同形
            # 只对 m=1 的元素求平均。clamp(min=1): mask 全 0 时分母是 0, 垫到 1 防除零
            loss = ((model_output - labels) ** 2 * m).sum() / m.sum().clamp(min=1)
        return {"total_loss": loss, "diffusion_loss": loss}


# -----------------------------------------------------------------------------
# Samplers
# -----------------------------------------------------------------------------


class DDIMSampler:
    """
    DDIM (Song et al., 2021) 确定性采样器 — DDPM 的少步推理版

    从 T-1 走到 0, 每步根据模型预测的 ε 做:
        x_0 = (x_t - sqrt(1-ᾱ_t) · ε) / sqrt(ᾱ_t)
        x_{t-1} = sqrt(ᾱ_{t-1}) · x_0 + sqrt(1 - ᾱ_{t-1}) · ε

    本实现纯确定性 (η=0)。

    clip_x0: cosine 调度下 ᾱ_999 ≈ 2.4e-9, 第一步除以 sqrt(ᾱ_t) 会把 ε 的任何误差放大 ~2 万倍;
             把 x0_pred 截到 [-clip_x0, clip_x0] 是标准补救 (diffusers 的 clip_sample)。
             像素空间用 1.0; 近似 N(0,1) 的 latent 用 3 左右; None = 不截断。
    """

    def __init__(
        self, scheduler: DDPMScheduler, num_inference_steps: int = 50,
        clip_x0: Optional[float] = None,
    ):
        self.scheduler = scheduler
        self.num_inference_steps = num_inference_steps
        self.clip_x0 = clip_x0

    @torch.inference_mode()
    def sample(
        self,
        model: nn.Module,
        shape,
        device: torch.device,
        class_labels: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
        null_class_id: Optional[int] = None,
    ) -> torch.Tensor:
        """从纯噪声采样, 返回 x_0 的估计, 形状 = shape。

        model: 可调用对象, 签名 (x, t, class_labels) → 与 x 同形的 ε 预测。
               DiT / VideoDiT 可直接传; MMDiT 要包一层, 把文本条件闭包进去。
        shape: 输出形状, 第 0 维是 batch。
        喂给模型的 t 是 [B], [0, T) 量纲。
        class_labels [B] 和 guidance_scale ≠ 1 同时给出时才做 CFG, 此时必须给 null_class_id。
        """
        self.scheduler.to(device)
        x = torch.randn(shape, device=device)            # x_T: 纯噪声

        # 从 T-1 到 0 等间隔取 num_inference_steps 个整数步 (降序)
        step_ids = torch.linspace(
            self.scheduler.num_train_timesteps - 1, 0,
            self.num_inference_steps, device=device,
        ).long()                                         # [num_inference_steps]

        for i in range(self.num_inference_steps):
            t = step_ids[i]
            t_batch = t.expand(shape[0])                 # 标量 → [B], 整个 batch 用同一个 t

            pred = _apply_cfg(model, x, t_batch, class_labels, guidance_scale, null_class_id)

            # ab_t = ᾱ_t (当前步), ab_prev = ᾱ 在下一个要去的步上的值
            ab_t = self.scheduler.alphas_cumprod[t]
            if i < self.num_inference_steps - 1:
                ab_prev = self.scheduler.alphas_cumprod[step_ids[i + 1]]
            else:
                ab_prev = torch.tensor(1.0, device=device)   # 最后一步: ᾱ = 1 即没有噪声, x 直接等于 x0_pred

            # 把加噪公式 x_t = sqrt(ᾱ_t)·x_0 + sqrt(1-ᾱ_t)·ε 反过来解出 x_0
            x0_pred = (x - (1 - ab_t).sqrt() * pred) / ab_t.sqrt()
            if self.clip_x0 is not None:
                x0_pred = x0_pred.clamp(-self.clip_x0, self.clip_x0)
            # 用同一个 ε 预测, 按下一步的 ᾱ 重新加噪 (η=0: 不加新的随机噪声)
            x = ab_prev.sqrt() * x0_pred + (1 - ab_prev).sqrt() * pred

        return x


class EulerFlowSampler:
    """
    Rectified Flow 的 Euler ODE 求解 — 线性路径, 步长 Δt = 1/N

    从 x_1 = ε (纯噪声) 开始, 每步:
        v = model(x_t, t)
        x_{t-Δt} = x_t - Δt · v
    直到 t → 0, 即得 x_0。

    少步即可收敛 (教学默认 20 步; SD3 推理 28 步左右).

    time_scale 必须等于训练时 FlowMatchingScheduler.num_train_timesteps:
    积分用 t ∈ [0,1], 但喂给模型的是 t·time_scale (与训练时的 t_norm 同量纲)。
    """

    def __init__(self, num_inference_steps: int = 20, time_scale: float = 1000.0):
        self.num_inference_steps = num_inference_steps
        self.time_scale = time_scale

    @torch.inference_mode()
    def sample(
        self,
        model: nn.Module,
        shape,
        device: torch.device,
        class_labels: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
        null_class_id: Optional[int] = None,
    ) -> torch.Tensor:
        """从纯噪声采样, 返回 x_0 的估计, 形状 = shape。

        model: 可调用对象, 签名 (x, t, class_labels) → 与 x 同形的 velocity 预测。
               DiT / VideoDiT 可直接传; MMDiT 要包一层, 把文本条件闭包进去
               (写法见 run_models/generative/mmdit/infer_mmdit.py 的 denoiser)。
        喂给模型的 t 是 [B], 值为 t·time_scale: 第一步 time_scale, 最后一步 time_scale / N。
        其余参数同 DDIMSampler.sample。
        """
        x = torch.randn(shape, device=device)            # x_1 = ε: 纯噪声
        # N 步需要 N+1 个时间点: t 从 1 线性递减到 0
        ts = torch.linspace(1.0, 0.0, self.num_inference_steps + 1, device=device)
        for i in range(self.num_inference_steps):
            t = ts[i]
            dt = ts[i] - ts[i + 1]                       # 步长 Δt = 1/N > 0
            t_batch = (t * self.time_scale).expand(shape[0])   # [B], 模型量纲

            v = _apply_cfg(model, x, t_batch, class_labels, guidance_scale, null_class_id)
            x = x - dt * v                               # v = dx/dt 指向噪声; 往 t 变小的方向走, 所以是减
        return x


# -----------------------------------------------------------------------------
# Classifier-Free Guidance (CFG)
# -----------------------------------------------------------------------------


def classifier_free_guidance(
    cond_pred: torch.Tensor,
    uncond_pred: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    """
    CFG 线性外插:
        pred = uncond + guidance_scale · (cond - uncond)

    guidance_scale = 1  → 只用条件 (不引导)
    guidance_scale > 1  → 推模型更靠近条件, 典型 7.5 (SD 系列默认)
    """
    return uncond_pred + guidance_scale * (cond_pred - uncond_pred)


def _apply_cfg(
    model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    class_labels: Optional[torch.Tensor],
    guidance_scale: float,
    null_class_id: Optional[int],
) -> torch.Tensor:
    """
    sampler 内部用的 CFG 封装:
        - class_labels 为 None 或 guidance_scale == 1 时直接单次前向
        - 否则额外跑一次 null class, 线性外插
    model 按位置参数调用: model(x, t, class_labels) → 与 x 同形。x [B, ...], t [B], class_labels [B] 或 None。
    """
    if class_labels is None or guidance_scale == 1.0:
        return model(x, t, class_labels)

    if null_class_id is None:
        raise ValueError("CFG 需要 null_class_id (训练时用 class_dropout 制造的 null)")

    uncond_labels = torch.full_like(class_labels, null_class_id)   # [B] 全填 "空类别"
    cond_pred = model(x, t, class_labels)       # 有条件的预测
    uncond_pred = model(x, t, uncond_labels)    # 无条件的预测 (每步要跑两次前向)
    return classifier_free_guidance(cond_pred, uncond_pred, guidance_scale)
