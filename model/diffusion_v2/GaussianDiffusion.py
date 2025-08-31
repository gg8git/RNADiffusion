import math
from collections import namedtuple
from collections.abc import Callable
from functools import partial
from random import random
from typing import Literal, TypeVar

import lightning as L
import torch
import torch.nn.functional as F
from einops import rearrange, reduce
from torch import Tensor, nn
from torch.nn import Module, ModuleList
from tqdm.auto import tqdm

LMIN = -10
LMAX = 10

PredType = Literal["pred_noise", "pred_x0", "pred_v"]


class Unet1D(Module):
    def __init__(
        self,
        channels: int,
        dim: int,
        dim_mults: tuple[int, ...] = (1, 2, 4, 8),
        self_condition: bool = False,
        attn_dim_head: int = 32,
        attn_heads: int = 4,
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        self.init_conv = nn.Conv1d(input_channels, dim, 7, padding=3)

        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time embeddings

        time_dim = dim * 4

        sinu_pos_emb = SinusoidalPosEmb(dim)
        fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb, nn.Linear(fourier_dim, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim)
        )

        resnet_block = partial(ResnetBlock, time_emb_dim=time_dim)

        # layers

        self.downs = ModuleList([])
        self.ups = ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                ModuleList(
                    [
                        resnet_block(dim_in, dim_in),
                        resnet_block(dim_in, dim_in),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Downsample(dim_in, dim_out) if not is_last else nn.Conv1d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = resnet_block(mid_dim, mid_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim, dim_head=attn_dim_head, heads=attn_heads)))
        self.mid_block2 = resnet_block(mid_dim, mid_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                ModuleList(
                    [
                        resnet_block(dim_out + dim_in, dim_out),
                        resnet_block(dim_out + dim_in, dim_out),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Upsample(dim_out, dim_in) if not is_last else nn.Conv1d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        self.out_dim = channels

        self.final_res_block = resnet_block(dim * 2, dim)
        self.final_conv = nn.Conv1d(dim, self.out_dim, 1)

    def forward(self, x: Tensor, time: Tensor, x_self_cond: Tensor | None = None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:  # type: ignore
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:  # type: ignore
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)


class GaussianDiffusion1D(L.LightningModule):
    def __init__(
        self,
        model: Unet1D,
        *,
        seq_length: int,
        timesteps: int = 1000,
        objective: PredType = "pred_noise",
    ):
        super().__init__()
        self.model = model
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition

        self.seq_length = seq_length

        self.objective = objective

        assert objective in {"pred_noise", "pred_x0", "pred_v"}, (
            "objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])"
        )

        betas = cosine_beta_schedule(timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)

        self.betas = nn.Buffer(betas.to(torch.float32))
        self.alphas_cumprod = nn.Buffer(alphas_cumprod.to(torch.float32))
        self.alphas_cumprod_prev = nn.Buffer(alphas_cumprod_prev.to(torch.float32))

        # calculations for diffusion q(x_t | x_{t-1}) and others

        self.sqrt_alphas_cumprod = nn.Buffer(torch.sqrt(alphas_cumprod).to(torch.float32))
        self.sqrt_one_minus_alphas_cumprod = nn.Buffer(torch.sqrt(1.0 - alphas_cumprod).to(torch.float32))
        self.log_one_minus_alphas_cumprod = nn.Buffer(torch.log(1.0 - alphas_cumprod).to(torch.float32))
        self.sqrt_recip_alphas_cumprod = nn.Buffer(torch.sqrt(1.0 / alphas_cumprod).to(torch.float32))
        self.sqrt_recipm1_alphas_cumprod = nn.Buffer(torch.sqrt(1.0 / alphas_cumprod - 1).to(torch.float32))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        self.posterior_variance = nn.Buffer(posterior_variance.to(torch.float32))

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        self.posterior_log_variance_clipped = nn.Buffer(
            torch.log(posterior_variance.clamp(min=1e-20)).to(torch.float32)
        )
        self.posterior_mean_coef1 = nn.Buffer(
            (betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)).to(torch.float32)
        )
        self.posterior_mean_coef2 = nn.Buffer(
            ((1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)).to(torch.float32)
        )

        # calculate loss weight

        snr = alphas_cumprod / (1 - alphas_cumprod)
        self.snr = snr

        if objective == "pred_noise":
            loss_weight = torch.ones_like(snr)
        elif objective == "pred_x0":
            loss_weight = snr
        elif objective == "pred_v":
            loss_weight = snr / (snr + 1)
        else:
            raise ValueError(f"unknown objective {objective}")

        self.loss_weight = nn.Buffer(loss_weight.to(torch.float32))

    def predict_start_from_noise(self, x_t: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t: Tensor, t: Tensor, x0: Tensor) -> Tensor:
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / extract(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape
        )

    def predict_v(self, x_start: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise
            - extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t: Tensor, t: Tensor, v: Tensor) -> Tensor:
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start: Tensor, x_t: Tensor, t: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(
        self, x: Tensor, t: Tensor, x_self_cond: Tensor | None = None, clip_x_start: bool = False
    ) -> tuple[Tensor, Tensor]:
        model_output = self.model(x, t, x_self_cond)

        maybe_clip = partial(torch.clamp, min=LMIN, max=LMAX) if clip_x_start else identity

        if self.objective == "pred_noise":
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

        elif self.objective == "pred_x0":
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == "pred_v":
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)
        else:
            raise ValueError(f"unknown objective {self.objective}")

        return pred_noise, x_start

    @torch.no_grad()
    def ddim_sample(
        self,
        shape: tuple[int, ...],
        sampling_steps: int,
        clip_denoised: bool = True,
        eta: float = 0.0,
    ):
        batch, total_timesteps = (shape[0], self.num_timesteps)

        times = torch.linspace(
            -1, total_timesteps - 1, steps=sampling_steps + 1
        )  # [-1, 0, 1, 2, ..., T-1] when sampling_steps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device=self.device)

        x_start = None

        for time, time_next in tqdm(time_pairs, desc="sampling loop time step", leave=False):
            time_cond = torch.full((batch,), time, device=self.device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(
                img,
                time_cond,
                x_self_cond=self_cond,
                clip_x_start=clip_denoised,
            )

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma**2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

        return img

    def q_sample(self, x_start: Tensor, t: Tensor, noise: Tensor | None = None) -> Tensor:
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start: Tensor, t: Tensor, noise: Tensor | None = None) -> Tensor:
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample

        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            x_self_cond = self.model_predictions(x, t, clip_x_start=True)[1]
            x_self_cond.detach_()

        # predict and take gradient step

        model_out = self.model(x, t, x_self_cond)

        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = x_start
        elif self.objective == "pred_v":
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f"unknown objective {self.objective}")

        loss = F.mse_loss(model_out, target, reduction="none")
        loss = reduce(loss, "b ... -> b", "mean")

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, seq: Tensor) -> Tensor:
        B, _, N = seq.shape
        assert self.seq_length == N, f"seq length must be {self.seq_length}"

        t = torch.randint(0, self.num_timesteps, (B,), device=self.device).long()

        return self.p_losses(seq, t)


class Residual(Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim: int, dim_out: int | None = None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv1d(dim, default(dim_out, dim), 3, padding=1)
    )


def Downsample(dim: int, dim_out: int | None = None):
    return nn.Conv1d(dim, default(dim_out, dim), 4, 2, 1)


class RMSNorm(Module):
    def __init__(self, dim: int):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x: Tensor):
        return F.normalize(x, dim=1) * self.g * (x.shape[1] ** 0.5)


class PreNorm(Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x: Tensor):
        x = self.norm(x)
        return self.fn(x)


# sinusoidal positional embeds


class SinusoidalPosEmb(Module):
    def __init__(self, dim: int, theta: float = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x: Tensor):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Block(Module):
    def __init__(self, dim: int, dim_out: int):
        super().__init__()
        self.proj = nn.Conv1d(dim, dim_out, 3, padding=1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()

    def forward(self, x: Tensor, scale_shift: tuple[Tensor, Tensor] | None = None):
        x = self.proj(x)
        x = self.norm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(Module):
    def __init__(self, dim: int, dim_out: int, *, time_emb_dim: int | None = None):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2)) if time_emb_dim is not None else None

        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x: Tensor, time_emb: Tensor | None = None):
        scale_shift = None
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1")
            scale_shift = time_emb.chunk(2, dim=1)  # type: ignore

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


class LinearAttention(Module):
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv1d(hidden_dim, dim, 1), RMSNorm(dim))

    def forward(self, x: Tensor):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) n -> b h c n", h=self.heads), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale

        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c n -> b (h c) n", h=self.heads)
        return self.to_out(out)


class Attention(Module):
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x: Tensor):
        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        q = rearrange(q, "b (h d) n -> b h n d", h=self.heads).contiguous()
        k = rearrange(k, "b (h d) n -> b h n d", h=self.heads).contiguous()
        v = rearrange(v, "b (h d) n -> b h n d", h=self.heads).contiguous()

        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, "b h n d -> b (h d) n")
        return self.to_out(out)


T = TypeVar("T")
ModelPrediction = namedtuple("ModelPrediction", ["pred_noise", "pred_x_start"])


def exists(x: object | None) -> bool:
    return x is not None


def default(val: T | None, d: T | Callable[[], T]) -> T:
    if val is not None:
        return val

    if isinstance(d, Callable):
        return d()

    return d


def identity(t: Tensor, *args, **kwargs):
    return t


def extract(a: Tensor, t: Tensor, x_shape: tuple[int, ...]):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps: int):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps: int, s: float = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)
