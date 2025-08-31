import math
from collections.abc import Callable

import lightning as L
import torch
from torch import Tensor
from tqdm.auto import tqdm

from model.diffusion_v2.GaussianDiffusion import GaussianDiffusion1D, PredType, Unet1D, extract
from model.diffusion_v2.vae import BaseVAE

torch.set_float32_matmul_precision("medium")


class DiffusionModel(L.LightningModule):
    def __init__(
        self,
        data_type: str,
        pred_type: PredType,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.data_type = data_type
        self.vae = BaseVAE.load_from_checkpoint(f"./data/{data_type}_vae.ckpt")

        self.n_bn = self.vae.n_acc
        self.d_bn = self.vae.d_bnk

        model = Unet1D(
            channels=self.d_bn,
            dim=64,
            self_condition=True,
        )

        self.diffusion = GaussianDiffusion1D(
            model,
            seq_length=self.n_bn,
            timesteps=1000,
            objective=pred_type,
        )

    def forward(self, z: Tensor) -> Tensor:
        z = z.reshape(z.shape[0], self.n_bn, self.d_bn).transpose(-2, -1)
        return self.diffusion(z)

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        loss = self.forward(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        loss = self.forward(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):  # type: ignore
        opt = torch.optim.Adam(self.diffusion.parameters(), lr=3e-4, betas=(0.9, 0.99))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda step: min((step + 1) / 2048, 1))

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "interval": "step",
                "frequency": 1,
            },
        }

    @torch.no_grad()
    def ddim_sample(
        self,
        batch_size: int,
        sampling_steps: int = 50,
        guidance_scale: float = 1.0,
        cond_fn: Callable | None = None,
    ) -> torch.Tensor:
        device = self.device
        assert self.diffusion.objective == "pred_v", (
            f"This DDIM sampler only supports pred_v, not {self.diffusion.objective}"
        )

        # time schedule: [T-1, ... 1, 0, -1]
        times = torch.linspace(-1, self.diffusion.num_timesteps - 1, steps=sampling_steps + 1)
        times = list(reversed(times.long().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        x_t = torch.randn((batch_size, self.d_bn, self.n_bn), device=device)
        x_start = None
        for t, t_next in tqdm(time_pairs, desc="DDIM Sampling", leave=False):
            t_vec = torch.full((batch_size,), t, device=device, dtype=torch.long)

            # 1) predict v_t
            v_hat = self.diffusion.model(
                x=x_t,
                time=t_vec,
                x_self_cond=x_start if self.diffusion.self_condition else None,
            )

            alpha_bar_t = extract(self.diffusion.alphas_cumprod, t_vec, x_t.shape)
            A = alpha_bar_t.sqrt()
            B = (1.0 - alpha_bar_t).sqrt()

            if cond_fn is not None and t <= 900:
                _x0_hat = A * x_t - B * v_hat

                grad = cond_fn(_x0_hat, t_vec)
                grad = clip_max_grad(grad, 6 * math.sqrt(self.n_bn * self.d_bn)) # Keep gradient in [-6, 6] ball

                v_hat = v_hat - B * grad * guidance_scale

            x0_hat = A * x_t - B * v_hat
            eps_hat = B * x_t + A * v_hat

            if t_next < 0:
                x_t = x0_hat
                continue

            # 6) DDIM update (eta=0): x_{t_next} = sqrt(abar_{t_next}) * x0_hat + sqrt(1-abar_{t_next}) * eps_hat
            t_next_vec = torch.full((batch_size,), t_next, device=device, dtype=torch.long)
            alpha_bar_next = extract(self.diffusion.alphas_cumprod, t_next_vec, x_t.shape)

            x_t = alpha_bar_next.sqrt() * x0_hat + (1.0 - alpha_bar_next).sqrt() * eps_hat
            x_start = x0_hat

        return x_t.transpose(1, 2).flatten(1)

    @torch.no_grad()
    def ddim_sample_v1(
        self,
        batch_size: int,
        sampling_steps: int = 50,
        guidance_scale: float = 1.0,
        cond_fn: Callable | None = None,
        clip_denoised: bool = True,
    ) -> torch.Tensor:
        device = self.device

        # time schedule: [T-1, ..., 1, 0, -1]
        times = torch.linspace(-1, self.diffusion.num_timesteps - 1, steps=sampling_steps + 1)
        times = list(reversed(times.long().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        x_t = torch.randn((batch_size, self.d_bn, self.n_bn), device=device)
        x_start = None  # for self-conditioning (we'll set this to x0_hat each step)

        for t, t_next in tqdm(time_pairs, desc="DDIM Sampling Loop Time Step"):
            t_vec = torch.full((batch_size,), t, device=device, dtype=torch.long)

            # 1) predict v_t
            v_hat = self.diffusion.model(
                x=x_t,
                time=t_vec,
                x_self_cond=x_start if self.diffusion.self_condition else None,
            )

            # 2) compute ᾱ_t terms
            alpha_bar_t = extract(self.diffusion.alphas_cumprod, t_vec, x_t.shape)  # (B,1,1)
            A = alpha_bar_t.sqrt()
            B = (1.0 - alpha_bar_t).sqrt()

            # 3) convert v -> x0_hat and eps (no divisions)
            #    x0 = A * x_t - B * v,   eps = B * x_t + A * v
            x0_hat = A * x_t - B * v_hat
            eps = B * x_t + A * v_hat

            # 4) classifier guidance in eps-space: eps_hat = eps - sqrt(1-ᾱ_t) * ∇ log p(y|x_t)
            if cond_fn is not None:
                grad = cond_fn(x_t, t_vec)
                grad = clip_max_grad(grad, math.sqrt(self.d_bn * self.n_bn))
                eps_hat = eps - B * A.reciprocal() * (guidance_scale * grad)
                x0_hat = (x_t - B * eps_hat) * A.reciprocal()  # Rederive x0_hat
                if clip_denoised:
                    x0_hat = x0_hat.clamp(-6.0, 6.0)  # (swap for dynamic thresholding if you prefer)

            else:
                eps_hat = eps

            # 5) last step -> return x0
            if t_next < 0:
                break

            # 6) DDIM update (η=0): x_{t_next} = √ᾱ_{t_next} x0_hat + √(1-ᾱ_{t_next}) eps_hat
            t_next_vec = torch.full((batch_size,), t_next, device=device, dtype=torch.long)
            alpha_bar_next = extract(self.diffusion.alphas_cumprod, t_next_vec, x_t.shape)
            x_t = alpha_bar_next.sqrt() * x0_hat + (1.0 - alpha_bar_next).sqrt() * eps_hat

            # 7) update self-conditioning input to be current x0 estimate
            x_start = x0_hat

        # x_t holds x0 at this point
        return x_t.transpose(1, 2).flatten(1)


def clip_max_grad(x: Tensor, max_norm: float) -> Tensor:
    shape = x.shape
    x = x.flatten(1)

    norm = x.norm(2, dim=-1, keepdim=True)
    factor = torch.clamp_max(max_norm / (norm + 1e-6), 1.0)

    x = x * factor
    return x.reshape(shape)
