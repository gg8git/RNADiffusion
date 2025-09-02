import math
from collections.abc import Callable

import lightning as L
import torch
import torch.nn.functional as F
from einops import reduce
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
            in_dim=self.d_bn,
            hdim=256,
            dim_ff=2048,
            num_layers=8,
            ntime=1000,
        )

        self.diffusion = GaussianDiffusion1D(
            model,
            seq_length=self.n_bn,
            timesteps=1000,
            objective=pred_type,
        )

    # @torch.compile
    def _train_forward(self, seq: Tensor) -> Tensor:
        x_start = seq.reshape(seq.shape[0], self.n_bn, self.d_bn)

        B, N, _ = x_start.shape

        t = torch.randint(0, self.diffusion.num_timesteps, (B,), device=self.device).long()

        noise = torch.randn_like(x_start)

        # noise sample

        x = self.diffusion.q_sample(x_start=x_start, t=t, noise=noise)

        with torch.no_grad():
            x_self_cond = self.diffusion.model_predictions(x, t, clip_x_start=False)[1]

        mask = torch.randn_like(t, dtype=torch.float32) < 0.5
        x_self_cond = x_self_cond * mask[:, None, None]

        # predict and take gradient step

        model_out = self.diffusion.model(x, t, x_self_cond)

        if self.diffusion.objective == "pred_noise":
            target = noise
        elif self.diffusion.objective == "pred_x0":
            target = x_start
        elif self.diffusion.objective == "pred_v":
            v = self.diffusion.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f"unknown objective {self.objective}")

        loss = F.mse_loss(model_out, target, reduction="none")
        loss = reduce(loss, "b ... -> b", "mean")

        loss = loss * extract(self.diffusion.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, z: Tensor) -> Tensor:
        z = z.reshape(z.shape[0], self.n_bn, self.d_bn)
        return self.diffusion(z)

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        loss = self._train_forward(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        loss = self._train_forward(batch)
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

        x_t = torch.randn((batch_size, self.n_bn, self.d_bn), device=device)
        x_start = None
        for t, t_next in tqdm(time_pairs, desc="DDIM Sampling", leave=False):
            t_vec = torch.full((batch_size,), t, device=device, dtype=torch.long)

            # 1) predict v_t
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                v_hat = self.diffusion.model(
                    x=x_t,
                    time=t_vec,
                    x_self_cond=x_start if self.diffusion.self_condition else None,
                ).float()

            alpha_bar_t = extract(self.diffusion.alphas_cumprod, t_vec, x_t.shape)
            A = alpha_bar_t.sqrt()
            B = (1.0 - alpha_bar_t).sqrt()

            if cond_fn is not None:
                _x0_hat = A * x_t - B * v_hat

                grad = cond_fn(_x0_hat.flatten(1), t_vec)
                grad = grad.reshape(x_t.shape)
                grad = clip_max_grad(grad, 6 * math.sqrt(self.n_bn * self.d_bn))  # Keep gradient in [-6, 6] ball

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

        return x_t.flatten(1)


def clip_max_grad(x: Tensor, max_norm: float) -> Tensor:
    shape = x.shape
    x = x.flatten(1)

    norm = x.norm(2, dim=-1, keepdim=True)
    factor = torch.clamp_max(max_norm / (norm + 1e-6), 1.0)

    x = x * factor
    return x.reshape(shape)


def calc_minmax(x: Tensor) -> tuple[float, float]:
    x_min = x.min().item()
    x_max = x.max().item()
    return x_min, x_max
