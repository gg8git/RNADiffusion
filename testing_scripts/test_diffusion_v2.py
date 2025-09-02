import json
import time

import gpytorch
import torch
import torch.nn.functional as F
from botorch.acquisition import qExpectedImprovement
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from datasets import load_dataset
from fcd import get_fcd
from gpytorch.mlls import PredictiveLogLikelihood
from rdkit import Chem, RDLogger
from torch.distributions import Normal
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from datamodules.diffusion_datamodule import DiffusionDataModule, LatentDataset
from model import GPModelDKL
from model.diffusion_v2 import DiffusionModel
from model.diffusion_v2.GaussianDiffusion import ExtinctPredictor
from utils.guacamol_utils import smiles_to_desired_scores

DATA_BATCH_SIZE = 1024
MODE = "pdop"
SURR_ITERS = [1, 4, 16, 64, 256]
ACQ_BATCH_SIZES = [4, 16, 64, 256, 1024, 4096]

# TODO: improve logging
LOG_NAME = "v2_testing"
LOG_PATH = f"./results/log_{LOG_NAME}.json" if LOG_NAME is not None else None

torch.set_printoptions(sci_mode=False, precision=3, linewidth=120)
RDLogger.DisableLog("rdApp.*")  # type: ignore


##########################################################
# Molecule
##########################################################
model = DiffusionModel.load_from_checkpoint("./data/molecule_diffusion.ckpt")
model.cuda()
model.freeze()
MU = -3.0
STD = 0.1


def try_canon(smi: str):
    try:
        return Chem.CanonSmiles(smi)
    except Exception:
        return None


def cond_fn_mvn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    with torch.enable_grad():
        x = x.detach().requires_grad_(True)
        dist = Normal(MU, STD)
        logp = dist.log_prob(x)
        if logp.dim() > 1:
            logp = logp.sum(dim=tuple(range(1, logp.dim())))
        s = logp.sum()
        (grad_x,) = torch.autograd.grad(s, x, retain_graph=False, create_graph=False)
    return grad_x.detach()


z = model.ddim_sample(
    batch_size=1024,
    sampling_steps=50,
    cond_fn=cond_fn_mvn,
    guidance_scale=1.0,
)

print("MVN:")
print(f"Target: {MU:.3f} +/- {STD:.3f}")
print(f"Sample: {z.mean():.3f} +/- {z.std():.3f}")
print()

ddim_z = model.ddim_sample(
    batch_size=2048,
    sampling_steps=50,
)
ddim_tokens = model.vae.sample(ddim_z)
ddim_smiles = model.vae.detokenize(ddim_tokens)

vae_z = torch.randn(2048, model.d_bn * model.n_bn)
vae_tokens = model.vae.sample(vae_z)
vae_smiles = model.vae.detokenize(vae_tokens)


ds_smiles = load_dataset("haydn-jones/Guacamol", split="val")["SMILES"][:2048]  # type: ignore
ds_smiles = [try_canon(smi) for smi in ds_smiles]
ds_smiles = [smi for smi in ds_smiles if smi is not None]

vae_smiles = [try_canon(smi) for smi in vae_smiles]
vae_smiles = [smi for smi in vae_smiles if smi is not None]

ddim_smiles = [try_canon(smi) for smi in ddim_smiles]
ddim_smiles = [smi for smi in ddim_smiles if smi is not None]

print("FCD:")
print(f"VAE-DDIM:  {get_fcd(vae_smiles, ddim_smiles):.3f}")
print(f"VAE-Data:  {get_fcd(vae_smiles, ds_smiles):.3f}")
print(f"DDIM-Data: {get_fcd(ddim_smiles, ds_smiles):.3f}")
print()

##########################################################
# Peptide
##########################################################
model = DiffusionModel.load_from_checkpoint("./data/peptide_diffusion.ckpt")
model.cuda()
model.freeze()

predictor = ExtinctPredictor.load_from_checkpoint("./data/extinct_predictor.ckpt")
predictor.cuda()
predictor.freeze()


def cond_fn_extinct(z: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    with torch.enable_grad():
        z = z.detach().requires_grad_(True)
        logits = predictor(z.view(z.shape[0], -1))
        logp = F.logsigmoid(logits)
        if logp.dim() > 1:
            logp = logp.sum(dim=tuple(range(1, logp.dim())))
        s = logp.sum()
        (grad_z,) = torch.autograd.grad(s, z, retain_graph=False, create_graph=False)
    return grad_z.detach()


ddim_z = model.ddim_sample(
    batch_size=1024,
    sampling_steps=50,
)
guide_z = model.ddim_sample(
    batch_size=1024,
    sampling_steps=50,
    cond_fn=cond_fn_extinct,
    guidance_scale=1.0,
)
rand_z = torch.randn_like(ddim_z)

extinct_preds_ddim = (predictor(ddim_z).sigmoid() > 0.5).float()
extinct_preds_rand = (predictor(rand_z).sigmoid() > 0.5).float()
extinct_preds_guide = (predictor(guide_z).sigmoid() > 0.5).float()
print("Extinct:")
print(f"DDIM   Z: {extinct_preds_ddim.mean():.3f} +/- {extinct_preds_ddim.std():.3f}")
print(f"Random Z: {extinct_preds_rand.mean():.3f} +/- {extinct_preds_rand.std():.3f}")
print(f"Guided Z: {extinct_preds_guide.mean():.3f} +/- {extinct_preds_guide.std():.3f}")


##########################################################
# Log qEI
##########################################################
model = DiffusionModel.load_from_checkpoint("./data/molecule_diffusion.ckpt")
model.cuda()
model.freeze()


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
def cond_fn_gp_generator(qei_fn):
    def predictor_ei(x, eps=1e-8, tau=None):
        vals = qei_fn(x).clamp_min(eps)
        if tau is None:
            tau = torch.quantile(vals.detach(), 0.5).clamp_min(eps)
        squashed = vals / (vals + tau)
        return torch.log(squashed + eps)

    def cond_fn_gp(z: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        with torch.enable_grad():
            z = z.detach().requires_grad_(True)
            logits = predictor_ei(z.view(z.shape[0], -1))
            logp = F.logsigmoid(logits)
            if logp.dim() > 1:
                logp = logp.sum(dim=tuple(range(1, logp.dim())))
            s = logp.sum()
            (grad_z,) = torch.autograd.grad(s, z, retain_graph=False, create_graph=False)
        return grad_z.detach()

    return cond_fn_gp


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
for i, batch in enumerate(dataloader):
    batch_z, batch_y = get_batch_scores(batch, MODE)
    surrogate_model = update_surr_model(surrogate_model, surrogate_mll, 0.002, batch_z, batch_y, 100)
    batch_max_score, batch_max_idx = batch_y.max(dim=0)
    if batch_max_score.item() > max_score:
        max_score = batch_max_score.item()
        best_z = batch_z[batch_max_idx].detach().clone()

    if i + 1 in SURR_ITERS:
        best_f = torch.tensor(max_score, device=model.device, dtype=torch.float32)

        lb = torch.full_like(best_z, -3)
        ub = torch.full_like(best_z, 3)
        bounds = torch.stack([lb, ub]).cuda()

        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
        log_qEI = qLogExpectedImprovement(model=surrogate_model.cuda(), best_f=best_f, sampler=sampler)
        qEI = qExpectedImprovement(model=surrogate_model.cuda(), best_f=best_f, sampler=sampler)
        torch.cuda.empty_cache()

        for batch_size in ACQ_BATCH_SIZES:
            print(f"processing (iter: {i + 1}, bsz: {batch_size})")

            summary[f"(iter: {i + 1}, bsz: {batch_size})"] = evaluate_on_batch(
                batch_size=batch_size,
                diffusion_model=model,
                cond_fn=cond_fn_gp_generator(qEI),
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
