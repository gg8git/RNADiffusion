import json
import time

import gpytorch
import lightning as L
import selfies as sf
import torch
from botorch.acquisition import qExpectedImprovement
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.mlls import PredictiveLogLikelihood
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from datamodules.diffusion_datamodule import DiffusionDataModule, LatentDataset, LatentDatasetDescriptors
from model import GaussianDiffusion1D, GPModelDKL, KarrasUnet1D, VAEFlatWrapper
from utils.guacamol_utils import smiles_to_desired_scores

device = "cuda" if torch.cuda.is_available() else "cpu"
vae = VAEFlatWrapper(path_to_vae_statedict="checkpoints/SELFIES_VAE/epoch=447-step=139328.ckpt").to(device)


def update_surr_model(model, mll, learning_rte, train_z, train_y, n_epochs):
    model = model.train()
    optimizer = torch.optim.Adam([{"params": model.parameters(), "lr": learning_rte}], lr=learning_rte)
    train_bsz = min(len(train_y), 128)
    train_dataset = TensorDataset(train_z.cuda(), train_y.cuda())
    train_loader = DataLoader(train_dataset, batch_size=train_bsz, shuffle=True)
    for _ in tqdm(range(n_epochs), leave=False):
        for inputs, scores in train_loader:
            optimizer.zero_grad()
            output = model(inputs.cuda())
            loss = -mll(output, scores.cuda())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    model = model.eval()

    return model


def get_cond_fn(
    log_prob_fn, guidance_strength: float = 1.0, latent_dim: int = 128, clip_grad=False, clip_grad_max=10.0
):
    """
    log_prob_fn --> maps a latent z of shape (B, 128) into a log probability
    guidance_strength --> the guidance strength of the model
    latent_dim --> the latent dim (always 128)
    clip_grad --> if the model should clip the gradient to +-clip_grad_max

    Returns a cond_fn that evaluastes the grad of the log probability
    """

    def cond_fn(mean, t, **kwargs):
        # mean.shape = (B, 1, 128), so reshape to (B, 128) so predicter can handle it
        mean = mean.detach().reshape(-1, latent_dim)
        mean.requires_grad_(True)

        with torch.enable_grad():
            predicted_log_probability = log_prob_fn(mean)
            gradients = torch.autograd.grad(predicted_log_probability, mean, retain_graph=True)[0]

            if clip_grad:
                gradients = torch.clamp(gradients, -clip_grad_max, clip_grad_max)

            grads = guidance_strength * gradients.reshape(-1, 1, latent_dim)
            return grads

    return cond_fn


def load_diffusion_model(load_model_checkpoint):
    class Wrapper(L.LightningModule):
        def __init__(self):
            super().__init__()
            model = KarrasUnet1D(
                seq_len=8,
                dim=64,
                dim_max=128,
                channels=16,
                num_downsamples=3,
                attn_res=(32, 16, 8),
                attn_dim_head=32,
                self_condition=True,
            )

            self.diffusion = GaussianDiffusion1D(
                model,
                seq_length=8,
                timesteps=1000,
                objective="pred_noise",
            )

    model = Wrapper()

    ckpt = torch.load(load_model_checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    diffusion = model.diffusion
    diffusion.eval()
    return diffusion.cuda()


def score_lambda(function, name, latent_dim, log_qei, max_restarts=5, **kwargs):
    """
    Runs a function that generates latents, computes log_qEI,
    and handles restarts if exceptions occur.
    """
    num_restarts = 0
    log_qei_score = "N/A"

    start = end = time.time() * 1000
    while log_qei_score == "N/A" and num_restarts < max_restarts:
        try:
            latents = function(**kwargs)  # use kwargs properly
            # reshape if needed
            if isinstance(latents, tuple):
                latents = latents[0]
            if len(latents.shape) == 3:
                latents = latents.transpose(1, 2)
            log_qei_score = log_qei(latents.reshape(-1, latent_dim)).detach().cpu().item()
            end = time.time() * 1000
        except Exception as e:
            print(f"Exception: {e}")
            torch.cuda.empty_cache()
            num_restarts += 1

    print(f"log qei {name}: {log_qei_score}")
    torch.cuda.empty_cache()
    return {"log qei score": log_qei_score, "num restarts": num_restarts, "clock time (ms)": int(end - start)}


def evaluate_on_batch(batch_size, diffusion_model, cond_fn, log_qei, bounds):
    curr_summary = {}
    latent_dim = diffusion_model.channels * diffusion_model.seq_length

    # no conditioning
    curr_summary["no cond"] = score_lambda(
        function=diffusion_model.ddim_sample,
        name="no cond",
        latent_dim=latent_dim,
        log_qei=log_qei,
        # kwargs
        batch_size=batch_size,
        cond_fn=None,
        guidance_scale=1.0,
    )

    curr_summary["ddim cond"] = score_lambda(
        function=diffusion_model.ddim_sample,
        name="ddim cond",
        latent_dim=latent_dim,
        log_qei=log_qei,
        # kwargs
        batch_size=batch_size,
        use_self_cond=False,
        cond_fn=cond_fn,
        guidance_scale=25.0,
    )

    curr_summary["ddim self cond"] = score_lambda(
        function=diffusion_model.ddim_sample,
        name="ddim self cond",
        latent_dim=latent_dim,
        log_qei=log_qei,
        # kwargs
        batch_size=batch_size,
        use_self_cond=True,
        cond_fn=cond_fn,
        guidance_scale=25.0,
    )

    curr_summary["ddim cond 50"] = score_lambda(
        function=diffusion_model.ddim_sample,
        name="ddim cond 50",
        latent_dim=latent_dim,
        log_qei=log_qei,
        # kwargs
        batch_size=batch_size,
        sampling_timesteps=50,
        use_self_cond=False,
        cond_fn=cond_fn,
        guidance_scale=25.0,
    )

    curr_summary["ddim self cond 50"] = score_lambda(
        function=diffusion_model.ddim_sample,
        name="ddim self cond 50",
        latent_dim=latent_dim,
        log_qei=log_qei,
        # kwargs
        batch_size=batch_size,
        sampling_timesteps=50,
        use_self_cond=True,
        cond_fn=cond_fn,
        guidance_scale=25.0,
    )

    # curr_summary['dpmpp2m cond 50'] = score_lambda(
    #     function=diffusion_model.dpmpp2m_sample,
    #     name='dpmpp2m cond 50',
    #     latent_dim=latent_dim,
    #     log_qei=log_qei,
    #     # kwargs
    #     batch_size=batch_size,
    #     sampling_timesteps=50,
    #     cond_fn=cond_fn,
    #     guidance_scale=25.0
    # )

    # curr_summary['dpmpp2msde cond 50'] = score_lambda(
    #     function=diffusion_model.dpmpp2msde_sample,
    #     name='dpmpp2msde cond 50',
    #     latent_dim=latent_dim,
    #     log_qei=log_qei,
    #     # kwargs
    #     batch_size=batch_size,
    #     sampling_timesteps=50,
    #     cond_fn=cond_fn,
    #     guidance_scale=25.0
    # )

    curr_summary["optimize acqf"] = score_lambda(
        function=optimize_acqf,
        name="optimize acqf",
        latent_dim=latent_dim,
        log_qei=log_qei,
        # kwargs
        acq_function=log_qei,
        bounds=bounds,
        q=batch_size,
        num_restarts=3,
        raw_samples=1024,
    )

    # curr_summary['optimize acqf seq'] = score_lambda(
    #     function=optimize_acqf,
    #     name='optimize acqf seq',
    #     latent_dim=latent_dim,
    #     log_qei=log_qei,
    #     # kwargs
    #     acq_function=log_qei,
    #     bounds=bounds,
    #     q=batch_size,
    #     num_restarts=3,
    #     raw_samples=1024,
    #     sequential=True,
    # )

    return curr_summary


def get_batch(batch, mode="pdop"):
    if mode == "qed" or mode == "fsp3":
        batch_z, _, qed, fsp3 = batch
        flat_batch_z = batch_z.reshape(batch_z.size(0), -1)
        return flat_batch_z, (qed if mode == "qed" else fsp3)

    with torch.no_grad():
        batch_selfies = vae.latent_to_selfies(batch)
        batch_smiles = [sf.decoder(s) for s in batch_selfies]
        scores = smiles_to_desired_scores(batch_smiles, mode)
    batch_y = torch.tensor(scores, device=device, dtype=torch.float32)
    flat_batch_z = batch.reshape(batch.size(0), -1)
    return flat_batch_z, batch_y


def validate_with_gp(diffusion, mode="pdop", batch_sizes=[64], surr_iters=[16], log_path=None):
    print("=== Conditional Sampling (GP Condition) ===")
    # Placeholder cond_fn – to be replaced with a proper differentiable cond_fn

    data_batch_size = 1024
    sub_data_dir = "descriptors" if (mode == "qed" or mode == "fsp3") else "selfies/selfies_flat"
    dataset = LatentDatasetDescriptors if (mode == "qed" or mode == "fsp3") else LatentDataset
    dm = DiffusionDataModule(
        data_dir=f"data/{sub_data_dir}",
        batch_size=data_batch_size,
        num_workers=0,
        dataset=dataset,
    )

    batch = next(iter(dm.train_dataloader()))  # [b,128]
    batch = batch[0] if isinstance(batch, list) else batch
    inducing_z = batch.cuda().to(device)
    import ipdb

    ipdb.set_trace()

    surrogate_model = GPModelDKL(
        inducing_z.reshape(data_batch_size, -1), likelihood=gpytorch.likelihoods.GaussianLikelihood().cuda()
    ).cuda()
    surrogate_mll = PredictiveLogLikelihood(surrogate_model.likelihood, surrogate_model, num_data=data_batch_size)
    surrogate_model.eval()

    max_score = float("-inf")
    best_z = None

    summary = {}
    for i, batch in enumerate(dm.train_dataloader()):
        batch_z, batch_y = get_batch(batch, mode)
        surrogate_model = update_surr_model(surrogate_model, surrogate_mll, 0.002, batch_z, batch_y, 100)
        batch_max_score, batch_max_idx = batch_y.max(dim=0)
        if batch_max_score.item() > max_score:
            max_score = batch_max_score.item()
            best_z = batch_z[batch_max_idx].detach().clone()

        if i + 1 in surr_iters:
            best_f = torch.tensor(max_score, device=device, dtype=torch.float32)

            lb = torch.full_like(best_z, -3)
            ub = torch.full_like(best_z, 3)
            bounds = torch.stack([lb, ub]).cuda()

            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
            log_qEI = qLogExpectedImprovement(model=surrogate_model.cuda(), best_f=best_f, sampler=sampler)
            qEI = qExpectedImprovement(model=surrogate_model.cuda(), best_f=best_f, sampler=sampler)
            torch.cuda.empty_cache()

            def log_prob_fn_ei(x, eps=1e-8, tau=None):
                vals = qEI(x).clamp_min(eps)
                if tau is None:
                    tau = torch.quantile(vals.detach(), 0.5).clamp_min(eps)
                squashed = vals / (vals + tau)
                return torch.log(squashed + eps)

            cond_fn_ei = get_cond_fn(
                log_prob_fn=log_prob_fn_ei,
                clip_grad=False,
                latent_dim=(diffusion.seq_length * diffusion.channels),
            )

            for batch_size in batch_sizes:
                print(f"processing (iter: {i + 1}, bsz: {batch_size})")

                summary[f"(iter: {i + 1}, bsz: {batch_size})"] = evaluate_on_batch(
                    batch_size=batch_size, diffusion_model=diffusion, cond_fn=cond_fn_ei, log_qei=log_qEI, bounds=bounds
                )

                if log_path is not None:
                    with open(log_path, "w") as file:
                        json.dump(summary, file, indent=2)

        if i > max(surr_iters):
            break

    if log_path is not None:
        with open(log_path, "w") as file:
            json.dump(summary, file, indent=2)
    print(summary)


# === Entry point ===


def main():
    diffusion = load_diffusion_model(load_model_checkpoint="SELFIES_Diffusion/oflvuzyp/checkpoints/last.ckpt")
    validate_with_gp(
        diffusion=diffusion,
        mode="pdop",
        batch_sizes=[4, 16, 64, 256],
        surr_iters=[4, 16, 64],
        log_path=f"results/log_{int(time.time() * 1000)}.json",
    )


if __name__ == "__main__":
    main()
