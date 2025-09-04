import math
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

T = TypeVar("T")

LMIN = -10
LMAX = 10

PredType = Literal["pred_noise", "pred_x0", "pred_v"]


class GaussianDiffusion1D(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
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

    def q_sample_dist(self, x_start: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        std = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, std

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
            clip_x_start = not self.training
            x_self_cond = self.model_predictions(x, t, clip_x_start=clip_x_start)[1]
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
        B, N, _ = seq.shape
        assert self.seq_length == N, f"seq length must be {self.seq_length}"

        t = torch.randint(0, self.num_timesteps, (B,), device=self.device).long()

        return self.p_losses(seq, t)


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


class Unet1D(Module):
    def __init__(self, in_dim: int, hdim: int, dim_ff: int, num_layers: int, ntime: int):
        super().__init__()

        self.in_proj = nn.Sequential(
            nn.Linear(in_dim * 2, hdim),
            nn.LayerNorm(hdim),
        )
        self.out_proj = nn.Sequential(
            nn.LayerNorm(hdim),
            nn.Linear(hdim, in_dim),
        )

        self.time_emb = nn.Sequential(
            PositionalEncoding(
                d_model=hdim,
                max_len=ntime,
            ),
            nn.Linear(hdim, hdim * 4),
            nn.SiLU(),
            nn.Linear(hdim * 4, hdim),
        )

        self.blocks = ModuleList([Block(hdim, dim_ff) for _ in range(num_layers)])

        self.channels = in_dim
        self.self_condition = True

    # @torch.compile
    def forward(self, x: Tensor, time: Tensor, x_self_cond: Tensor | None = None):
        if x_self_cond is None:
            x_self_cond = torch.zeros_like(x)

        x = torch.cat([x, x_self_cond], dim=-1)
        x = self.in_proj(x)

        time_emb = self.time_emb(time)

        for block in self.blocks:
            x = block(x, time_emb)

        x = self.out_proj(x)

        return x


class Block(Module):
    def __init__(self, dim: int, dim_ff: int):
        super().__init__()

        self.transformer = TransformerEncoderLayer(
            d_model=dim,
            dim_feedforward=dim_ff,
            nhead=8,
        )

        self.time_proj = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, 2 * dim),
        )

    def forward(self, x: Tensor, time_emb: Tensor):
        x = self.transformer(x)

        scale, shift = self.time_proj(time_emb).chunk(2, dim=-1)
        x = x * (scale + 1) + shift

        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
    ) -> None:
        super().__init__()

        self.self_attn = SelfAttention(d_model, nhead)

        self.ff_block = FFNSwiGLU(d_model, dim_feedforward)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self._sa_block(self.norm1(x))
        x = x + self.ff_block(self.norm2(x))

        return x

    def _sa_block(self, x: Tensor) -> Tensor:
        return self.self_attn(x)


class SelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        heads: int,
    ):
        super().__init__()

        self.embed_size = embed_dim
        self.num_heads = heads
        self.head_dim = embed_dim // heads

        assert self.head_dim * heads == embed_dim, "Embedding size needs to be divisible by heads"

        self.qkv_proj = nn.Linear(self.embed_size, self.embed_size * 3)

        self.out_proj = nn.Linear(heads * self.head_dim, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)

        q = rearrange(q, "... n (h d) -> ... h n d", h=self.num_heads)
        k = rearrange(k, "... n (h d) -> ... h n d", h=self.num_heads)
        v = rearrange(v, "... n (h d) -> ... h n d", h=self.num_heads)

        attn = F.scaled_dot_product_attention(q, k, v)

        attn = rearrange(attn, "... h n d -> ... n (h d)")
        return self.out_proj(attn)


class FFNSwiGLU(nn.Module):
    """
    GLU Variants Improve Transformer
    https://arxiv.org/abs/2002.05202
    """

    def __init__(
        self,
        dim: int,
        dim_feedforward: int,
        out_dim: int | None = None,
        use_bias: bool = True,
    ):
        super().__init__()

        out_dim = out_dim or dim

        self.ff1 = nn.Linear(dim, dim_feedforward * 2, bias=use_bias)
        self.ff2 = nn.Linear(dim_feedforward, out_dim, bias=use_bias)

    def forward(self, x: Tensor) -> Tensor:
        y, gate = self.ff1(x).chunk(2, dim=-1)
        x = y * F.silu(gate)
        return self.ff2(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 128):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.pe = nn.Buffer(pe)

    def forward(self, x: Tensor) -> Tensor:
        return self.pe[x]


class ExtinctPredictor(L.LightningModule):
    def __init__(
        self,
        hdim: int,
        n_bn: int = 8,
        d_bn: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.trunk = nn.Sequential(
            nn.Unflatten(dim=-1, unflattened_size=(n_bn, d_bn)),
            nn.Linear(d_bn, hdim),
            nn.GELU(),
            nn.Linear(hdim, hdim),
            nn.GELU(),
            nn.Linear(hdim, hdim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hdim, d_bn),
            nn.GELU(),
            nn.Flatten(1),
        )

        self.head = nn.Sequential(
            nn.Linear(n_bn * d_bn, hdim // 2),
            nn.GELU(),
            nn.Linear(hdim // 2, hdim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hdim // 2, 1),
        )

    # @torch.compile
    def forward(self, x: Tensor) -> Tensor:
        x = self.trunk(x.flatten(1))
        x = self.head(x)
        return x

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        seq, extinct = batch
        extinct_hat = self(seq)

        bce = F.binary_cross_entropy_with_logits(extinct_hat.flatten(), extinct.float().flatten())

        metrics = calc_acc(extinct_hat, extinct)
        metrics = {f"train/{k}": v for k, v in metrics.items()}

        self.log("train/bce", bce, prog_bar=True, on_step=True, on_epoch=False)
        self.log_dict(metrics, prog_bar=True, on_step=True, on_epoch=True)

        return bce

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        seq, extinct = batch
        extinct_hat = self(seq)

        bce = F.binary_cross_entropy_with_logits(extinct_hat.flatten(), extinct.float().flatten())
        metrics = calc_acc(extinct_hat, extinct)
        metrics = {f"val/{k}": v for k, v in metrics.items()}

        self.log("val/bce", bce, prog_bar=True, on_step=False, on_epoch=True)
        self.log_dict(metrics, prog_bar=True, on_step=False, on_epoch=True)

        return bce

    def configure_optimizers(self):  # type: ignore
        opt = torch.optim.AdamW(self.parameters(), lr=1e-4)
        # 2048 step warmup
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda step: min(step / 2048, 1))
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


@torch.no_grad()
def calc_acc(logits: Tensor, targets: Tensor) -> dict[str, Tensor]:
    preds = logits.flatten().sigmoid() > 0.5
    acc = (preds == targets.flatten()).float().mean()

    zero_mask = targets.flatten() == 0
    zero_acc = (preds[zero_mask] == targets.flatten()[zero_mask]).float().mean()
    one_acc = (preds[~zero_mask] == targets.flatten()[~zero_mask]).float().mean()

    return {"acc": acc, "zero_acc": zero_acc, "one_acc": one_acc}
