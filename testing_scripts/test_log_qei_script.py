import json
import time
from typing import List, Union
import selfies as sf

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from botorch.acquisition import qExpectedImprovement
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim import optimize_acqf
import gpytorch
from gpytorch.mlls import PredictiveLogLikelihood

from RNADiffusion.data.diffusion_datamodule import DiffusionDataModule, LatentDatasetDescriptors
from model.surrogate_model.ppgpr import GPModelDKL
from RNADiffusion.model.GaussianDiffusion_deprecated import GaussianDiffusion1D
from model.UNet1D import KarrasUnet1D

device = 'cuda' if torch.cuda.is_available() else 'cpu'


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


def get_cond_fn(log_prob_fn, guidance_strength: float = 1.0, latent_dim: int = 128, clip_grad=False, clip_grad_max=10.0, debug=False):
    '''
        log_prob_fn --> maps a latent z of shape (B, 128) into a log probability
        guidance_strength --> the guidance strength of the model
        latent_dim --> the latent dim (always 128)
        clip_grad --> if the model should clip the gradient to +-clip_grad_max

        Returns a cond_fn that evaluastes the grad of the log probability
    '''

    def cond_fn(mean, t, **kwargs):
        # mean.shape = (B, 1, 128), so reshape to (B, 128) so predicter can handle it
        mean = mean.detach().reshape(-1, latent_dim)
        mean.requires_grad_(True)

        # if debug:
            # print(f"mean: {mean}")
            # print(f"mean.shape: {mean.shape}")
            # print(f"mean.requires_grad: {mean.requires_grad}")


        #---------------------------------------------------------------------------

        with torch.enable_grad():
            predicted_log_probability = log_prob_fn(mean)
            if debug:
                print(f"pred_log_prob: {predicted_log_probability}")
                print(f"pred_log_prob.shape: {predicted_log_probability.shape}")
                print(f"pred_log_prob.requires_grad {predicted_log_probability.requires_grad}")
                
            gradients = torch.autograd.grad(predicted_log_probability, mean, retain_graph=True)[0]

            # if debug:
                # print(f"gradients: {gradients}")
                # print(f"graidents.shape: {gradients.shape}")
                # print(f"gradients.requires_grad {gradients.requires_grad}")
                
            if clip_grad:
                if debug:
                    print(f"Clipping gradients to {-clip_grad_max} to {clip_grad_max}")
                gradients = torch.clamp(gradients, -clip_grad_max, clip_grad_max)
                
            grads = guidance_strength * gradients.reshape(-1, 1, latent_dim)
            if debug:
                # print(f"grads: {grads}")
                print(f"grad_norm: {grads.norm(2)}")
                print(f"grads.shape: {grads.shape}")
                # print(f"grads.requires_grad {grads.requires_grad}")
                
            return grads
        
    return cond_fn


def load_diffusion_model():
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
            )

            self.diffusion = GaussianDiffusion1D(
                model,
                seq_length=8,
                timesteps=1000,
                objective="pred_noise",
            )

        def forward(self, z):
            z = z.transpose(1,2)
            return self.diffusion(z)

        def training_step(self, batch, batch_idx):
            loss = self.forward(batch)
            self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.diffusion.parameters(), lr=3e-4)
    
    model = Wrapper()
    
    ckpt = torch.load("SELFIES_Diffusion/jhqe3fgr/checkpoints/last.ckpt", map_location="cpu")
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
                latents = latents.transpose(1,2)
            log_qei_score = log_qei(latents.reshape(-1, latent_dim)).detach().cpu().item()
            end = time.time() * 1000
        except Exception as e:
            print(f"Exception: {e}")
            torch.cuda.empty_cache()
            num_restarts += 1

    print(f"log qei {name}: {log_qei_score}")
    torch.cuda.empty_cache()
    return {'log qei score': log_qei_score, 'num restarts': num_restarts, 'clock time (ms)': int(end - start)}


def evaluate_on_batch(batch_size, diffusion_model, cond_fn, log_qei, bounds):
    curr_summary = {}
    shape = (batch_size, diffusion_model.channels, diffusion_model.seq_length)
    latent_dim = diffusion_model.channels * diffusion_model.seq_length

    # no conditioning
    curr_summary['no cond'] = score_lambda(
        function=diffusion_model.ddim_sample_haydn,
        name='no cond',
        latent_dim=latent_dim,
        log_qei=log_qei,
        # kwargs
        shape=shape,
        cond_fn=None,
        guidance_scale=1.0
    )

    curr_summary['ddim cond'] = score_lambda(
        function=diffusion_model.ddim_sample_haydn,
        name='ddim cond',
        latent_dim=latent_dim,
        log_qei=log_qei,
        # kwargs
        shape=shape,
        cond_fn=cond_fn,
        guidance_scale=25.0
    )

    curr_summary['ddim cond 50'] = score_lambda(
        function=diffusion_model.ddim_sample_haydn,
        name='ddim cond 50',
        latent_dim=latent_dim,
        log_qei=log_qei,
        # kwargs
        shape=shape,
        sampling_timesteps=50,
        cond_fn=cond_fn,
        guidance_scale=25.0
    )

    # curr_summary['dpmpp2m cond 50'] = score_lambda(
    #     function=diffusion_model.dpmpp2m_sample,
    #     name='dpmpp2m cond 50',
    #     latent_dim=latent_dim,
    #     log_qei=log_qei,
    #     # kwargs
    #     shape=shape,
    #     sampling_timesteps=50,
    #     cond_fn=cond_fn,
    #     guidance_scale=25.0
    # )

    curr_summary['dpmpp2msde cond 50'] = score_lambda(
        function=diffusion_model.dpmpp2msde_sample,
        name='dpmpp2msde cond 50',
        latent_dim=latent_dim,
        log_qei=log_qei,
        # kwargs
        shape=shape,
        sampling_timesteps=50,
        cond_fn=cond_fn,
        guidance_scale=25.0
    )

    # curr_summary['optimize acqf'] = score_lambda(
    #     function=optimize_acqf,
    #     name='optimize acqf',
    #     latent_dim=latent_dim,
    #     log_qei=log_qei,
    #     # kwargs
    #     acq_function=log_qei,
    #     bounds=bounds,
    #     q=batch_size,
    #     num_restarts=3,
    #     raw_samples=1024,
    # )

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


def validate_with_descriptor_gp(diffusion, batch_sizes=[64], surr_iters=[16]):
    print("=== Conditional Sampling (GP Condition) ===")
    # Placeholder cond_fn – to be replaced with a proper differentiable cond_fn

    data_batch_size = 1024
    dm = DiffusionDataModule(
        data_dir="data/descriptors",
        batch_size=data_batch_size,
        num_workers=0,
        dataset=LatentDatasetDescriptors,
    )

    batch = next(iter(dm.train_dataloader())) # [b,8,16]
    inducing_z = batch[0].cuda()
    device = inducing_z.device

    surrogate_model = GPModelDKL(inducing_z.reshape(data_batch_size, -1), likelihood=gpytorch.likelihoods.GaussianLikelihood().cuda()).cuda()
    surrogate_mll = PredictiveLogLikelihood(surrogate_model.likelihood, surrogate_model, num_data=data_batch_size)
    surrogate_model.eval()

    max_score = float('-inf')
    best_z = None

    from collections import defaultdict
    summary = defaultdict(dict)
    for i, batch in enumerate(dm.train_dataloader()):
        batch_z, _, qed, _ = batch
        flat_batch_z = batch_z.reshape(batch_z.size(0), -1)
        surrogate_model = update_surr_model(surrogate_model, surrogate_mll, 0.002, flat_batch_z, qed, 100)

        batch_max_score, batch_max_idx = qed.max(dim=0)
        if batch_max_score.item() > max_score:
            max_score = batch_max_score.item()
            best_z = flat_batch_z[batch_max_idx].detach().clone()

        if i+1 in surr_iters:
            best_f = torch.tensor(max_score, device=device, dtype=torch.float32)
            # best_s = vae_wrapper.latent_to_selfies(best_z.reshape(-1, diffusion.seq_length, diffusion.channels))[0]

            lb = torch.full_like(best_z, -3)
            ub = torch.full_like(best_z,  3)
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

            def cond_fn_ei_reshaped(x, t):
                # x = x.reshape(x.shape[0], -1)  # no transpose
                x = x.transpose(1, 2).reshape(x.shape[0], -1)  # with transpose
                grad = cond_fn_ei(x, t)
                # grad = grad.reshape(-1, diffusion.channels, diffusion.seq_length)  # no transpose
                grad = grad.reshape(-1, diffusion.seq_length, diffusion.channels).transpose(1,2)  # with transpose
                return grad
            
            for batch_size in batch_sizes:
                print(f"processing (iter: {i+1}, bsz: {batch_size})")

                summary[f"(iter: {i+1}, bsz: {batch_size})"] = evaluate_on_batch(
                    batch_size=batch_size,
                    diffusion_model=diffusion,
                    cond_fn=cond_fn_ei_reshaped,
                    log_qei=log_qEI,
                    bounds=bounds
                )

                # num_restarts = 0
                # log_qei_score = "N/A"
                # while log_qei_score == "N/A" and num_restarts < 5:
                #     try:
                #         shape = (batch_size, diffusion.channels, diffusion.seq_length)
                #         latents = diffusion.ddim_sample_haydn(shape, cond_fn=None, guidance_scale=1.0)
                #         latents = latents.transpose(1,2)  # with transpose
                #         log_qei_score = log_qEI(latents.reshape(latents.shape[0], -1)).detach().cpu().item()
                #     except Exception as e:
                #         num_restarts += 1
                # curr_summary['no cond'] = {'log qei score': log_qei_score, 'num restarts': num_restarts}
                # print(f"log qei no cond: {log_qei_score}")
                # torch.cuda.empty_cache()

                # num_restarts = 0
                # log_qei_score = "N/A"
                # while log_qei_score == "N/A" and num_restarts < 5:
                #     try:
                #         shape = (batch_size, diffusion.channels, diffusion.seq_length)
                #         latents = diffusion.ddim_sample_haydn(shape, cond_fn=cond_fn_ei_reshaped, guidance_scale=25.0)
                #         latents = latents.transpose(1,2)  # with transpose
                #         log_qei_score = log_qEI(latents.reshape(latents.shape[0], -1)).detach().cpu().item()
                #     except Exception as e:
                #         num_restarts += 1
                # curr_summary['ddim'] = {'log qei score': log_qei_score, 'num restarts': num_restarts}
                # print(f"log qei ddim: {log_qei_score}")
                # torch.cuda.empty_cache()

                # num_restarts = 0
                # log_qei_score = "N/A"
                # while log_qei_score == "N/A" and num_restarts < 5:
                #     try:
                #         latents, _ = optimize_acqf(log_qEI, bounds=bounds, q=batch_size, num_restarts=10, raw_samples=1024)
                #         latents = latents.reshape(-1, diffusion.seq_length, diffusion.channels)
                #         log_qei_score = log_qEI(latents.reshape(latents.shape[0], -1)).detach().cpu().item()
                #     except Exception as e:
                #         num_restarts += 1
                # curr_summary['optimize acqf'] = {'log qei score': log_qei_score, 'num restarts': num_restarts}
                # print(f"log qei optimize acqf: {log_qei_score}")
                # torch.cuda.empty_cache()

                with open("./log_big_batch_noacqf.json", 'w') as file:
                    json.dump(summary, file, indent=2)
        
        if i > max(surr_iters):
            break
    
    with open("./log_big_batch_noacqf.json", 'w') as file:
        json.dump(summary, file, indent=2)
    print(summary)

# === Entry point ===

def main():
    diffusion = load_diffusion_model()
    validate_with_descriptor_gp(diffusion=diffusion, batch_sizes=[256, 512, 1024, 2048, 4096], surr_iters = [1, 4, 16, 64, 256])

if __name__ == "__main__":
    main()