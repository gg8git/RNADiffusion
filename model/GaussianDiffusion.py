import math
from collections import namedtuple
from functools import partial
from random import random

import torch
import torch.nn.functional as F
from einops import reduce
from torch import nn
from torch.cuda.amp import autocast
from tqdm.auto import tqdm

from model.UNet1D import KarrasUnet1D

ModelPrediction = namedtuple("ModelPrediction", ["pred_noise", "pred_x_start"])
LMIN = -5.5
LMAX = 5.5


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t, *args, **kwargs):
    return t


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
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


class GaussianDiffusion1D(nn.Module):
    def __init__(
        self,
        model: KarrasUnet1D,
        *,
        seq_length,
        timesteps=1000,
        sampling_timesteps=None,
        objective="pred_noise",
        beta_schedule="cosine",
        ddim_sampling_eta=0.0,
        loss_type="mse",
    ):
        super().__init__()
        self.model = model
        assert hasattr(self.model, "channels"), "Model must have 'channels' attribute"
        self.channels = self.model.channels
        self.self_condition = getattr(model, "self_condition", False)

        self.seq_length = seq_length

        self.loss_type = loss_type

        self.objective = objective

        assert (
            self.objective in {"pred_noise", "pred_x0", "pred_v"}
        ), "objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])"

        if beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"unknown beta schedule {beta_schedule}")

        betas = betas.clamp(1e-6, 1-1e-6)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0).clamp(min=1e-6)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        self.sampling_timesteps = default(
            sampling_timesteps, timesteps
        )  # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        def register_buffer(name, val):
            return self.register_buffer(name, val.to(torch.float32))

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer("posterior_variance", posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        # calculate loss weight

        snr = alphas_cumprod / (1 - alphas_cumprod)

        if objective == "pred_noise":
            loss_weight = torch.ones_like(snr)
        elif objective == "pred_x0":
            loss_weight = snr
        elif objective == "pred_v":
            loss_weight = snr / (snr + 1)

        register_buffer("loss_weight", loss_weight)

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise
            - extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(
        self,
        x,
        t,
        class_labels,
        x_self_cond=None,
        clip_x_start=False,
        rederive_pred_noise=False,
    ):
        assert (
            class_labels is not None or not self.model.needs_class_labels
        ), "class_labels must be provided for model_predictions"

        model_output = self.model(x, t, x_self_cond, class_labels=class_labels)

        maybe_clip = (
            partial(torch.clamp, min=LMIN, max=LMAX) if clip_x_start else identity
        )

        if self.objective == "pred_noise":
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == "pred_x0":
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == "pred_v":
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(
        self, x, t, x_self_cond=None, clip_denoised=True, class_labels=None
    ):
        preds = self.model_predictions(
            x, t, class_labels=class_labels, x_self_cond=x_self_cond
        )
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(min=LMIN, max=LMAX)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_start, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, x_self_cond=None, clip_denoised=True):
        b, *_, _device = *x.shape, x.device
        batched_times = torch.full((b,), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x, t=batched_times, x_self_cond=x_self_cond, clip_denoised=clip_denoised
        )
        noise = torch.randn_like(x) if t > 0 else 0.0  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    def condition_score(self, cond_fn, x, t, x_start, alpha):
        pred_noise = self.predict_noise_from_start(x, t, x_start)
        pred_noise = pred_noise - (1 - alpha).sqrt() * cond_fn(x, t)

        x_start = self.predict_start_from_noise(x, t, pred_noise)
        pred_noise = self.predict_noise_from_start(x, t, x_start)

        return pred_noise, x_start

    @torch.no_grad()
    def ddim_sample(self, shape, cond_fn=None, clip_denoised=True, class_labels=None):
        assert (
            class_labels is not None or not self.model.needs_class_labels
        ), "class_labels must be provided for ddim_sample"

        batch, device, total_timesteps, sampling_timesteps, eta = (
            shape[0],
            self.betas.device,
            self.num_timesteps,
            self.sampling_timesteps,
            self.ddim_sampling_eta,
        )

        times = torch.linspace(
            -1, total_timesteps - 1, steps=sampling_timesteps + 1
        )  # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(
            zip(times[:-1], times[1:])
        )  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device=device)

        x_start = None

        for time, time_next in tqdm(
            time_pairs, desc="sampling loop time step", leave=False
        ):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(
                img,
                time_cond,
                class_labels=class_labels,
                x_self_cond=self_cond,
                clip_x_start=clip_denoised,
            )

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            # guidance
            if cond_fn is not None:
                pred_noise, x_start = self.condition_score(
                    cond_fn,
                    img,
                    time_cond,
                    x_start,
                    alpha,
                )

            sigma = (
                eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            )
            c = (1 - alpha_next - sigma**2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

        return img
    
    def condition(self, cond_fn, x, t, pred_noise, alpha, guidance_scale=1.0):
        return pred_noise - guidance_scale * (1 - alpha).sqrt() * cond_fn(x, t)

    @torch.no_grad()
    def ddim_sample_orig(self, shape, cond_fn=None, clip_denoised=True, class_labels=None):
        assert (
            class_labels is not None or not self.model.needs_class_labels
        ), "class_labels must be provided for ddim_sample"

        batch, device, total_timesteps, sampling_timesteps, eta = (
            shape[0],
            self.betas.device,
            self.num_timesteps,
            self.sampling_timesteps,
            self.ddim_sampling_eta,
        )

        times = torch.linspace(
            -1, total_timesteps - 1, steps=sampling_timesteps + 1
        )  # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(
            zip(times[:-1], times[1:])
        )  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device=device)

        x_start = None

        for time, time_next in tqdm(
            time_pairs, desc="sampling loop time step", leave=False
        ):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(
                img,
                time_cond,
                class_labels=class_labels,
                x_self_cond=self_cond,
                clip_x_start=clip_denoised,
            )

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            if cond_fn is not None:
                pred_noise = self.condition(
                    cond_fn, 
                    img, 
                    time_cond, 
                    pred_noise, 
                    alpha
                )

            sigma = (
                eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            )
            c = (1 - alpha_next - sigma**2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

        return img

    @torch.no_grad()
    def sample(self, batch_size, class_labels: torch.Tensor = None):
        assert (
            class_labels is not None or not self.model.needs_class_labels
        ), "class_labels must be provided for sampling"

        seq_length, channels = self.seq_length, self.channels
        return self.ddim_sample(
            (batch_size, channels, seq_length), class_labels=class_labels
        )

    @autocast(enabled=False)
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, class_labels=None, noise=None):
        assert (
            class_labels is not None or not self.model.needs_class_labels
        ), "class_labels must be provided for p_losses"

        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample

        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            x_self_cond = self.model_predictions(
                x, t, class_labels=class_labels, clip_x_start=True
            ).pred_x_start
            x_self_cond.detach_()

        # predict and take gradient step

        model_out = self.model(x, t, x_self_cond, class_labels=class_labels)

        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = x_start
        elif self.objective == "pred_v":
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f"unknown objective {self.objective}")

        if self.loss_type == "mse":
            loss = F.mse_loss(model_out, target, reduction="none")
        elif self.loss_type == "l1":
            loss = F.l1_loss(model_out, target, reduction="none")
        elif self.loss_type == "huber":
            loss = F.smooth_l1_loss(model_out, target, reduction="none")
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")
        
        loss = reduce(loss, "b ... -> b", "mean")

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img, class_labels=None):
        (
            b,
            _c,
            n,
            device,
            seq_length,
        ) = *img.shape, img.device, self.seq_length
        assert n == seq_length, f"seq length must be {seq_length}"
        assert (
            class_labels is not None or not self.model.needs_class_labels
        ), "class_labels must be provided"

        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        return self.p_losses(img, t, class_labels=class_labels)
