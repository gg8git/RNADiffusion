import math
from collections.abc import Callable

import lightning as L
import torch
import torch.nn.functional as F
from einops import reduce
from torch import Tensor
from torch.distributions import Normal
from tqdm.auto import tqdm

from model.diffusion_v2.GaussianDiffusion import GaussianDiffusion1D, PredType, Unet1D, extract
from model.diffusion_v2.vae import BaseVAE


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

    @torch.no_grad()
    def ddim_repaint(
        self,
        x_known: Tensor,
        mask: Tensor,
        sampling_steps: int = 50,
        u_steps: int = 20,
        tr_center: Tensor | None = None,
        tr_halfwidth: float | Tensor | None = None,
    ) -> Tensor:
        """
        Re-painting with DDIM: https://arxiv.org/abs/2201.09865
        A mask value of 1 indicates the value is known and should be preserved.
        """
        device = self.device
        assert self.diffusion.objective == "pred_v", (
            f"This DDIM sampler only supports pred_v, not {self.diffusion.objective}"
        )
        assert u_steps >= 1, "u_steps must be at least 1"

        B = x_known.shape[0]

        # schedule: [T-1, ... 0, -1]
        times = torch.linspace(-1, self.diffusion.num_timesteps - 1, steps=sampling_steps + 1)
        times = list(reversed(times.long().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        x_t = torch.randn((B, self.n_bn, self.d_bn), device=device)
        x_known = x_known.reshape(B, self.n_bn, self.d_bn).to(device)
        mask = mask.reshape(B, self.n_bn, self.d_bn).to(device)
        x_start = None

        if tr_center is not None:
            tr_center = tr_center.reshape(1, self.n_bn, self.d_bn).to(device)
        if tr_halfwidth is not None and isinstance(tr_halfwidth, float):
            tr_halfwidth = torch.full((1, self.n_bn, self.d_bn), tr_halfwidth, device=device)
        elif tr_halfwidth is not None:
            tr_halfwidth = tr_halfwidth.reshape(1, self.n_bn, self.d_bn).to(device)  # type: ignore

        for t, t_next in tqdm(time_pairs, desc="DDIM RePaint", leave=False):
            t_vec = torch.full((B,), t, device=device, dtype=torch.long)
            t_next_vec = torch.full((B,), t_next, device=device, dtype=torch.long)

            for ustep in range(u_steps):
                # 1) predict v_t
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    v_hat = self.diffusion.model(
                        x=x_t,
                        time=t_vec,
                        x_self_cond=x_start if self.diffusion.self_condition else None,
                    ).float()

                # v -> x0_hat, eps_hat
                alpha_bar_t = extract(self.diffusion.alphas_cumprod, t_vec, x_t.shape)
                A_t = alpha_bar_t.sqrt()
                B_t = (1.0 - alpha_bar_t).sqrt()

                x0_hat = A_t * x_t - B_t * v_hat
                eps_hat = B_t * x_t + A_t * v_hat
                x_start = x0_hat

                if t_next < 0:  # final snap enforce known region
                    x_t = mask * x_known + (1.0 - mask) * x0_hat
                    break

                # 2) DDIM (eta=0): t -> t_next
                alpha_bar_next = extract(self.diffusion.alphas_cumprod, t_next_vec, x_t.shape)
                x_t_unk = alpha_bar_next.sqrt() * x0_hat + (1.0 - alpha_bar_next).sqrt() * eps_hat

                # 3) RePaint mix at t_next
                x_t_knw = self.diffusion.q_sample(x_known, t=t_next_vec, noise=torch.randn_like(x_t) * (t_next > 0))

                x_tm1 = (1.0 - mask) * x_t_unk + mask * x_t_knw

                # 4) If more inner loops forward re-noise x_{t_next} -> x_t
                if ustep < u_steps - 1:
                    ratio = (alpha_bar_t / alpha_bar_next).clamp(min=0.0, max=1.0)
                    _z = torch.randn_like(x_tm1) if t > 0 else torch.zeros_like(x_tm1)
                    x_t = ratio.sqrt() * x_tm1 + (1.0 - ratio).sqrt() * _z
                else:
                    x_t = x_tm1

                if tr_center is not None and tr_halfwidth is not None:
                    # Interpolate between a half-width of 6 at t=T to the given half-width at t=0
                    interpolant = self.diffusion.sqrt_one_minus_alphas_cumprod[t]
                    hw_t = tr_halfwidth * (1 - interpolant) + 6.0 * interpolant

                    eps = 1e-3
                    x_t = tr_center + (hw_t - eps) * torch.tanh((x_t - tr_center) / (hw_t - eps))

        # Finally, clamp
        if tr_center is not None and tr_halfwidth is not None:
            lower_0 = tr_center - tr_halfwidth
            upper_0 = tr_center + tr_halfwidth
            x_t = x_t.clamp(min=lower_0, max=upper_0)

        return x_t.flatten(1)

    @torch.no_grad()
    def ddim_sample_tr(
        self,
        batch_size: int,
        sampling_steps: int = 50,
        guidance_scale: float = 1.0,
        cond_fn: Callable | None = None,
        tr_center: Tensor | None = None,
        tr_halfwidth: float | Tensor | None = None,
    ) -> torch.Tensor:
        device = self.device
        assert self.diffusion.objective == "pred_v", (
            f"This DDIM sampler only supports pred_v, not {self.diffusion.objective}"
        )
        if tr_center is not None:
            tr_center = tr_center.reshape(1, self.n_bn, self.d_bn).to(device)

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

            if tr_center is not None and tr_halfwidth is not None:
                # Interpolate between a half-width of 6 at t=T to the given half-width at t=0
                interpolant = self.diffusion.sqrt_one_minus_alphas_cumprod[t]
                hw_t = tr_halfwidth * (1 - interpolant) + 6.0 * interpolant

                lower_t = tr_center - hw_t
                upper_t = tr_center + hw_t
                # x_t = x_t.clamp(min=lower_t, max=upper_t)
                eps = 1e-3
                x_t = tr_center + (hw_t - eps) * torch.tanh((x_t - tr_center) / (hw_t - eps))

                on_boundary = ((x_t - lower_t).abs() < 1e-5) | ((x_t - upper_t).abs() < 1e-5)
                avg = on_boundary.flatten(1).float().sum(dim=1).mean()
                # tqdm.write(f"t={t}, hw_t={hw_t:.3f} | {avg:.3f} / {x_t.flatten(1).shape[1]} on boundary")

        # Finally, clamp
        if tr_center is not None and tr_halfwidth is not None:
            lower_0 = tr_center - tr_halfwidth
            upper_0 = tr_center + tr_halfwidth
            x_t = x_t.clamp(min=lower_0, max=upper_0)

        return x_t.flatten(1)
    
    @torch.no_grad()
    def ddim_sample_tr_guidance(
        self,
        batch_size: int,
        sampling_steps: int = 50,
        guidance_scale: float = 1.0,
        cond_fn: Callable | None = None,
        tr_center: Tensor | None = None,
        tr_halfwidth: float | Tensor | None = None,
        tr_clamp: bool = True,
        tr_guidance: str | None = None,
        tr_guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        device = self.device
        assert self.diffusion.objective == "pred_v", (
            f"This DDIM sampler only supports pred_v, not {self.diffusion.objective}"
        )
        if tr_center is not None:
            tr_center = tr_center.reshape(1, self.n_bn, self.d_bn).to(device)

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

            gradients = []
            if cond_fn is not None:
                _x0_hat = A * x_t - B * v_hat

                grad = cond_fn(_x0_hat.flatten(1), t_vec)

                grad = grad.reshape(x_t.shape)
                grad = clip_max_grad(grad, 6 * math.sqrt(self.n_bn * self.d_bn))  # Keep gradient in [-6, 6] ball

                # v_hat = v_hat - B * grad * guidance_scale
                gradients.append(B * grad)
            
            if tr_guidance is not None and tr_center is not None and tr_halfwidth is not None:
                if tr_guidance == "midpoint":
                    tr_center_vec = tr_center.repeat(batch_size, 1, 1)
                    mean, std = self.diffusion.q_sample_dist(tr_center_vec, t_vec)
                    
                    with torch.enable_grad():
                        x_t = x_t.detach().requires_grad_(True)
                        logp = Normal(mean, std).log_prob(x_t)

                        s = logp.sum()
                        (grad,) = torch.autograd.grad(s, x_t, retain_graph=False, create_graph=False)

                    grad = grad.reshape(x_t.shape)
                    grad = clip_max_grad(grad.detach(), 6 * math.sqrt(self.n_bn * self.d_bn))

                    # v_hat = v_hat - (B/A) * grad * guidance_scale
                    # gradients.append((B/A) * grad * tr_guidance_scale)
                    gradients.append(B * grad * tr_guidance_scale)
                
                elif tr_guidance == "sampled_point":
                    pass
                    
                elif tr_guidance == "boundary":
                    pass

                elif tr_guidance == "x0_hat":
                    _x0_hat = A * x_t - B * v_hat

                    with torch.enable_grad(): 
                        _x0_hat = _x0_hat.detach().requires_grad_(True)

                        lower = tr_center - tr_halfwidth
                        upper = tr_center + tr_halfwidth

                        lam = 0.1 + (50.0 - 0.1) * (1.0 - t_vec.float() / 1000.0)
                        lam = lam.view(-1, 1).to(_x0_hat.device)

                        low_viol = F.relu(lower - _x0_hat).pow(2)
                        up_viol  = F.relu(_x0_hat - upper).pow(2)
                        penalty = (low_viol + up_viol).flatten(1).sum(dim=1)

                        s = (-lam.squeeze() * penalty).sum()
                        (grad,) = torch.autograd.grad(s, _x0_hat, retain_graph=False, create_graph=False)

                    grad = grad.reshape(x_t.shape)
                    grad = clip_max_grad(grad.detach(), 6 * math.sqrt(self.n_bn * self.d_bn))

                    # v_hat = v_hat - B * grad * guidance_scale
                    gradients.append(B * grad * tr_guidance_scale)
            
            v_hat = v_hat - guidance_scale * torch.stack(gradients).sum(dim=0)

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

            if tr_clamp and tr_center is not None and tr_halfwidth is not None:
                # Interpolate between a half-width of 6 at t=T to the given half-width at t=0
                interpolant = self.diffusion.sqrt_one_minus_alphas_cumprod[t]
                hw_t = tr_halfwidth * (1 - interpolant) + 6.0 * interpolant

                lower_t = tr_center - hw_t
                upper_t = tr_center + hw_t
                # x_t = x_t.clamp(min=lower_t, max=upper_t)
                eps = 1e-3
                x_t = tr_center + (hw_t - eps) * torch.tanh((x_t - tr_center) / (hw_t - eps))

                on_boundary = ((x_t - lower_t).abs() < 1e-5) | ((x_t - upper_t).abs() < 1e-5)
                avg = on_boundary.flatten(1).float().sum(dim=1).mean()
                # tqdm.write(f"t={t}, hw_t={hw_t:.3f} | {avg:.3f} / {x_t.flatten(1).shape[1]} on boundary")

        # Finally, clamp
        if tr_center is not None and tr_halfwidth is not None:
            lower_0 = tr_center - tr_halfwidth
            upper_0 = tr_center + tr_halfwidth
            x_t = x_t.clamp(min=lower_0, max=upper_0)

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


def diffusion_box_bounds_forward(
    lower0: torch.Tensor,
    upper0: torch.Tensor,
    t: torch.Tensor,
    q_sample_dist: Callable,
    joint_coverage: float = 0.99,
):
    d = lower0.numel()
    _, std = q_sample_dist(lower0, t=t)
    sigma_t = std
    alpha_bar_t = 1.0 - sigma_t**2
    alpha_sqrt = torch.sqrt(alpha_bar_t)
    p_1d = joint_coverage ** (1.0 / d)
    z = Normal(0, 1).icdf(torch.tensor((1.0 + p_1d) / 2.0, device=lower0.device))
    lower_t = alpha_sqrt * lower0 - z * sigma_t
    upper_t = alpha_sqrt * upper0 + z * sigma_t
    return lower_t, upper_t


def trust_region_bounds_from_center(
    center0: torch.Tensor,
    halfwidth: torch.Tensor | float,
    t: torch.Tensor,
    q_sample_dist: Callable,
    joint_coverage: float = 0.99,
):
    center0 = center0.detach()
    halfwidth = torch.as_tensor(halfwidth, device=center0.device, dtype=center0.dtype).expand_as(center0)
    mean_t, std_t = q_sample_dist(center0, t=t)
    sigma_t = std_t
    alpha_sqrt = torch.sqrt(torch.clamp(1.0 - sigma_t**2, min=0.0))

    d = center0.numel()
    p_1d = joint_coverage ** (1.0 / d)
    z = Normal(0, 1).icdf(torch.tensor((1.0 + p_1d) / 2.0, device=center0.device, dtype=center0.dtype))

    m_t = mean_t
    h_t = alpha_sqrt * halfwidth + z * sigma_t

    lower_t = m_t - h_t
    upper_t = m_t + h_t
    return lower_t, upper_t, m_t, h_t


def cond_fn_trust(
    x_t: torch.Tensor,
    t: torch.Tensor,
    center: torch.Tensor,
    hw_t: torch.Tensor,
    margin_frac: float = 0.05,
    lam: float = 1.0,
):
    delta = x_t - center
    slack = hw_t - delta.abs()
    margin = (margin_frac * hw_t).clamp(min=1e-3)
    g = (margin - slack).clamp(min=0.0) * torch.sign(center - x_t)
    return lam * g
