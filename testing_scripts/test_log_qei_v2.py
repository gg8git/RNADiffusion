import torch
import torch.nn.functional as F
from rdkit import RDLogger

from model.diffusion_v2 import DiffusionModel

torch.set_printoptions(sci_mode=False, precision=3, linewidth=120)
RDLogger.DisableLog("rdApp.*")  # type: ignore

##########################################################
# Log qEI
##########################################################
model = DiffusionModel.load_from_checkpoint("./data/molecule_diffusion.ckpt")
model.cuda()
model.freeze()


import json
import time

import gpytorch
from botorch.acquisition import qExpectedImprovement
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.mlls import PredictiveLogLikelihood
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from datamodules.diffusion_datamodule import DiffusionDataModule, LatentDataset
from model import GPModelDKL
from utils.guacamol_utils import smiles_to_desired_scores

import ipdb; ipdb.set_trace()
DATA_BATCH_SIZE = 1024
MODE = "pdop"
NUM_INIT_ITERS = 10
SURR_ITERS = [0, 1, 4, 16, 64, 256]
ACQ_BATCH_SIZES = [4, 16, 64, 128, 256]

# TODO: improve logging
LOG_NAME = "v2_testing_4"
LOG_PATH = f"./results/log_{LOG_NAME}.json" if LOG_NAME is not None else None


def update_surr_model(surr_model, mll, learning_rte, train_z, train_y, n_epochs):
    surr_model = surr_model.train()
    optimizer = torch.optim.Adam([{"params": surr_model.parameters(), "lr": learning_rte}], lr=learning_rte)
    train_bsz = min(len(train_y), 128)
    train_dataset = TensorDataset(train_z.cuda(), train_y.cuda())
    train_loader = DataLoader(train_dataset, batch_size=train_bsz, shuffle=True)
    for _ in tqdm(range(n_epochs), leave=False):
        for inputs, scores in train_loader:
            optimizer.zero_grad()
            output = surr_model(inputs.cuda())
            loss = -mll(output, scores.cuda())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(surr_model.parameters(), max_norm=1.0)
            optimizer.step()
    surr_model = surr_model.eval()

    return surr_model


def load_dataloader(data_batch_size, split="train"):
    dm = DiffusionDataModule(
        data_dir="./data/selfies/selfies_flat",
        batch_size=data_batch_size,
        num_workers=0,
        dataset=LatentDataset,
    )

    if split == "train":
        return dm.train_dataloader()
    if split == "val":
        return dm.val_dataloader()
    else:
        return dm.test_dataloader()


def get_batch_scores(batch_z, mode="pdop"):
    with torch.no_grad():
        # find a way to use diffusion model vae
        batch_tokens = model.vae.sample(batch_z)
        batch_smiles = model.vae.detokenize(batch_tokens)
        scores = smiles_to_desired_scores(batch_smiles, mode)
    batch_y = torch.tensor(scores, device=model.device, dtype=torch.float32)
    flat_batch_z = batch_z.reshape(batch_z.size(0), -1)
    return flat_batch_z, batch_y


def score_lambda(function, name, latent_dim, log_qei, max_restarts=5, **kwargs):
    """
    Runs a function that generates latents, computes log_qEI,
    and handles restarts if exceptions occur.
    """
    num_restarts = 0
    log_qei_score = best_score = average_score = "N/A"

    start = end = time.time() * 1000
    while log_qei_score == "N/A" and num_restarts < max_restarts:
        try:
            latents = function(**kwargs)
            if isinstance(latents, tuple):
                latents = latents[0]

            log_qei_score = log_qei(latents.reshape(-1, latent_dim)).detach().cpu().item()

            _, scores = get_batch_scores(latents)
            best_score = scores.max().detach().cpu().item()
            average_score = scores.mean().detach().cpu().item()

            end = time.time() * 1000
        except Exception as e:
            print(f"Exception: {e}")
            torch.cuda.empty_cache()
            num_restarts += 1

    print(f"results {name} - (log qei: {log_qei_score}, best pdop: {best_score})")
    torch.cuda.empty_cache()
    return {
        "log qei score": log_qei_score,
        "best score": best_score,
        "average score": average_score,
        "num restarts": num_restarts,
        "clock time (ms)": int(end - start),
    }


def evaluate_on_batch(batch_size, diffusion_model, cond_fn, log_qei, bounds):
    curr_summary = {}
    latent_dim = diffusion_model.vae.n_acc * diffusion_model.vae.d_bnk

    # no conditioning
    curr_summary["no cond"] = score_lambda(
        function=diffusion_model.ddim_sample,
        name="no cond",
        latent_dim=latent_dim,
        log_qei=log_qei,
        # kwargs
        batch_size=batch_size,
        sampling_steps=50,
    )

    curr_summary["ddim cond"] = score_lambda(
        function=diffusion_model.ddim_sample,
        name="ddim cond",
        latent_dim=latent_dim,
        log_qei=log_qei,
        # kwargs
        batch_size=batch_size,
        sampling_steps=50,
        cond_fn=cond_fn,
        guidance_scale=1.0,
    )

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

    return curr_summary


# TODO: haydn verify if this makes any sense at all
def cond_fn_log_ei_generator(log_ei_mod):
    def cond_fn_log_ei(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            log_ei = log_ei_mod(x.unsqueeze(1))
            if log_ei.dim() > 1:
                log_ei = log_ei.sum(dim=tuple(range(1, log_ei.dim())))
            s = log_ei.sum()
            (grad_x,) = torch.autograd.grad(s, x, retain_graph=False, create_graph=False)

        return grad_x.detach()

    return cond_fn_log_ei


# TODO: find dataloader that works across systems
dataloader = load_dataloader(data_batch_size=DATA_BATCH_SIZE)
inducing_z = next(iter(dataloader))  # [b,128]

surrogate_model = GPModelDKL(
    inducing_z.reshape(DATA_BATCH_SIZE, -1).cuda(), likelihood=gpytorch.likelihoods.GaussianLikelihood().cuda()
).cuda()
surrogate_mll = PredictiveLogLikelihood(surrogate_model.likelihood, surrogate_model, num_data=DATA_BATCH_SIZE)
surrogate_model.eval()

max_score = float("-inf")
best_z = None

summary = {}
init_batch = []
for i, batch in enumerate(dataloader):
    if i + 1 < NUM_INIT_ITERS:
        init_batch.append(batch)
        continue
    
    n_surr_epochs = 10
    if i + 1 == NUM_INIT_ITERS:
        init_batch.append(batch)
        batch = torch.cat(init_batch, dim=0)
        n_surr_epochs = 100

    batch_z, batch_y = get_batch_scores(batch, MODE)
    surrogate_model = update_surr_model(surrogate_model, surrogate_mll, 0.002, batch_z, batch_y, n_surr_epochs)
    batch_max_score, batch_max_idx = batch_y.max(dim=0)
    if batch_max_score.item() > max_score:
        max_score = batch_max_score.item()
        best_z = batch_z[batch_max_idx].detach().clone()

    iter = (i + 1) - NUM_INIT_ITERS
    if iter in SURR_ITERS:
        best_f = torch.tensor(max_score, device=model.device, dtype=torch.float32)

        lb = torch.full_like(best_z, -3)
        ub = torch.full_like(best_z, 3)
        bounds = torch.stack([lb, ub]).cuda()

        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
        log_qEI = qLogExpectedImprovement(model=surrogate_model.cuda(), best_f=best_f, sampler=sampler)
        torch.cuda.empty_cache()

        for batch_size in ACQ_BATCH_SIZES:
            print(f"processing (iter: {iter}, bsz: {batch_size})")

            summary[f"(iter: {iter}, bsz: {batch_size})"] = evaluate_on_batch(
                batch_size=batch_size,
                diffusion_model=model,
                cond_fn=cond_fn_log_ei_generator(log_qEI),
                log_qei=log_qEI,
                bounds=bounds,
            )

            if LOG_PATH is not None:
                with open(LOG_PATH, "w") as file:
                    json.dump(summary, file, indent=2)

    if i > max(SURR_ITERS):
        break

if LOG_PATH is not None:
    with open(LOG_PATH, "w") as file:
        json.dump(summary, file, indent=2)
print(summary)
