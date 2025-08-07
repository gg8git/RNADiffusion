import json
from typing import List, Union
import selfies as sf

import lightning as L
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, TensorDataset

from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm
from scipy import stats
from fcd.fcd import get_fcd

from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim import optimize_acqf

from data.guacamol_utils import smiles_to_desired_scores, smile_to_guacamole_score
from data.diffusion_datamodule import DiffusionDataModule
from model.surrogate_model.ppgpr import GPModelDKL
from model.GaussianDiffusion import GaussianDiffusion1D
from model.UNet1D import KarrasUnet1D
from model.surrogate_model.wrapper import BoTorchDKLModelWrapper

from model.mol_vae_model.wrapper import VAEWrapper

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# === VAE MODEL ===

vae_wrapper = VAEWrapper(path_to_vae_statedict="checkpoints/SELFIES_VAE/epoch=447-step=139328.ckpt")


# === GP methods ===

def update_surr_model(
    model,
    surr_optim,
    train_z,
    train_y,
    n_epochs,
):
    print("update surr model")
    model = model.train()
    train_bsz = min(len(train_y), 128)
    train_dataset = TensorDataset(train_z.cuda(), train_y.cuda())
    train_loader = DataLoader(train_dataset, batch_size=train_bsz, shuffle=True)
    for _ in tqdm(range(n_epochs), leave=False):
        tot_loss = []
        tot_meandiff = []
        tot_std = []
        for inputs, scores in train_loader:
            output = model(inputs.cuda())
            loss = model.loss(output, scores.cuda())
            surr_optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            surr_optim.step()
            tot_loss.append(loss.item())
            tot_meandiff.append((scores - output.mean).abs().mean().item())
            tot_std.append(output.variance.pow(1 / 2).mean().item())
        tot_loss = sum(tot_loss) / len(tot_loss)
        tot_meandiff = sum(tot_meandiff) / len(tot_meandiff)
        tot_std = sum(tot_std) / len(tot_std)
        print(
            f"surr_loss {tot_loss:.3f}, meandiff {tot_meandiff:.3f}, datadiff {(train_y - train_y.mean()).abs().mean().item():.3f}, std {tot_std:.3f}"
        )
    model = model.eval()

    return

# === Molecule decoding and evaluation ===

with open("data/selfies_vocab.json") as f:
    vocab = json.load(f)

def tokens_to_selfie(tokens, drop_after_stop=True) -> str:
    """ Converts a *single* token sequence to a selfie string """

    # returns single atom in case encoding to selfies fails
    try :
        selfie = sf.encoding_to_selfies(tokens.squeeze().tolist(), {v:k for k,v in vocab.items()}, 'label')
    except:
        selfie = '[C]'
    
    if drop_after_stop and '[stop]' in selfie:
        selfie = selfie[:selfie.find('[stop]')]
    if '[pad]' in selfie:
        selfie = selfie[:selfie.find('[pad]')]
    if '[start]' in selfie:
        selfie = selfie[selfie.find('[start]') + len('[start]'):]
    return selfie

def tokens_to_smiles(tokens) -> str:
    selfie = tokens_to_selfie(tokens)
    smiles = sf.decoder(selfie)
    return smiles

def is_valid_molecule(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

def penalized_logp(smiles: str) -> float:
    """
    Penalized logP score as a sample molecular objective
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return -10.0
    log_p = Descriptors.MolLogP(mol)
    sa_score = 0  # You can use SAScore from the original paper
    return log_p - sa_score

# === Conditional functions ===

def toy_cond_fn(target_token_idx: int, target_value: float, weight=5.0):
    """
    A toy conditional function that pushes the average value of
    a particular token dimension towards a target.
    """
    def fn(x, t):
        grad = torch.zeros_like(x)
        grad[:, target_token_idx, :] = (x[:, target_token_idx, :] - target_value)
        return weight * grad
    return fn

def normal_analytical_cond_fn_grad(mean=0.0, sigma=0.001):
    # analytically computes the gradient of the log probability under
    # a normal distribution with mean=mean, sigma=sigma

    def cond_fn(z, t, **guidance_kwargs):
        z = z.to(device)
        grad = -(z - mean) / (sigma**2)

        # MUST clamp gradient for numerical stability
        # It is REALLY finnicky about how much you clamp
        # not ideal really...

        grad = torch.clamp(grad, min=-1000.0, max=1000.0)
        return grad
    
    return cond_fn

def normal_analytical_cond_fn(mean=0.0, sigma=0.001):
    if sigma <= 0:
        raise ValueError("Sigma must be positive")

    def cond_fn(z, t, **guidance_kwargs):
        z = z.to(device)
        var = sigma ** 2
        log_prob = -((z - mean) ** 2).sum() / (2 * var) - z.numel() * torch.log(torch.sqrt(torch.tensor(2 * np.pi, device=z.device)) * sigma)
        return log_prob

    return cond_fn

def logp_cond_fn(logp_fn, tokenizer):
    """
    Gradient-free conditioning via score estimation.
    """
    def cond_fn(x, t):
        # You can use REINFORCE-style or Langevin steps on logp if you implement differentiable decoding
        # This is a placeholder zero gradient
        return torch.zeros_like(x)
    return cond_fn


# === Evaluation Functions ===

def distribution_comparison(z, z_other, alpha=0.05, do_print=False):
    guided_flat = z.flatten().detach().cpu().numpy()
    baseline_flat = z_other.flatten().detach().cpu().numpy()
    
    # KS test
    _, p_value = stats.ks_2samp(guided_flat, baseline_flat)
    is_different = p_value < alpha
    
    if do_print:
        print(f"Sample of shape: {z.shape} is {'' if is_different else 'not '}different from other with p={p_value:.4f}\n")
    
    return p_value < alpha, p_value

# === Diffusion model loader ===

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


# === Validation functions ===

def validate_unconditional(diffusion, batch_size=64):
    print("=== Unconditional Sampling ===")
    with torch.no_grad():
        for method in ["ddim"]:
            print(f"Sampling with {method.upper()}...")
            if method == "ddpm":
                latents = diffusion.ddpm_sample(batch_size=batch_size, class_labels=None)
            else:
                shape = (batch_size, diffusion.channels, diffusion.seq_length)
                latents = diffusion.ddim_sample(shape, class_labels=None)

            # do get_fcd() from frechet dist github with smiles1 from vae prior+vae and smiles2 from diffusion+vae
            diff_selfies = vae_wrapper.latent_to_selfies(latents)
            vae_selfies = vae_wrapper.sample_selfies_from_prior(batch_size)
            print(f"FCD Score: {get_fcd(vae_selfies, diff_selfies)}")

            valid_count = 0
            for selfie in diff_selfies:
                smiles = sf.decoder(selfie)
                if is_valid_molecule(smiles):
                    valid_count += 1
            print(f"{valid_count}/{batch_size} valid molecules")


def validate_with_normal_analytical_cond(diffusion, batch_size=16):
    print("=== Conditional Sampling (Normal Analytical Function) ===")
    mean = -1.0
    sigma = 0.1
    cond_fn_normal_dist = normal_analytical_cond_fn(mean=mean, sigma=sigma)
    cond_fn_grad_normal_dist = normal_analytical_cond_fn_grad(mean=mean, sigma=sigma)

    for method in ["ddim_orig"]:
        print(f"Sampling with {method.upper()}...")
        if method == "ddpm":
            latents = diffusion.ddpm_sample(batch_size=batch_size, class_labels=None, cond_fn=cond_fn_grad_normal_dist)
        elif method == "ddim":
            shape = (batch_size, diffusion.channels, diffusion.seq_length)
            latents = diffusion.ddim_sample(shape, class_labels=None, cond_fn=cond_fn_normal_dist)
        else:
            shape = (batch_size, diffusion.channels, diffusion.seq_length)
            latents = diffusion.ddim_sample_orig(shape, class_labels=None, cond_fn=cond_fn_grad_normal_dist)

        selfies = vae_wrapper.latent_to_selfies(latents)
        for i, latent in enumerate(latents):
            smiles = sf.decoder(selfies[i])
            print(f"Sample {i+1}: {smiles} | Valid: {is_valid_molecule(smiles)}, Mean Value: {latent.mean():.3f}, STD Deviation: {latent.std():.5f}")
        
        noise = torch.randn_like(latents)
        distribution_comparison(latents, mean + sigma * noise, alpha=0.01, do_print=True)
        

def validate_with_mol_score_predictor(diffusion, batch_size=64):
    print("=== Conditional Sampling (pdop Condition) ===")

def validate_with_gp(diffusion, batch_size=64):
    print("=== Conditional Sampling (GP Condition) ===")
    # Placeholder cond_fn – to be replaced with a proper differentiable cond_fn

    data_batch_size = 1024
    dm = DiffusionDataModule(
        data_dir="data/selfies",
        batch_size=data_batch_size,
        num_workers=0,
    )

    batch = next(iter(dm.train_dataloader())) # [b,8,16]
    device = batch.device

    surrogate_model = GPModelDKL(batch.reshape(data_batch_size, -1)).to(device)
    surrogate_model.eval()
    surr_optim = torch.optim.Adam([{"params": surrogate_model.parameters(), "lr": 0.001}])

    max_score = float('-inf')
    best_z = None

    for i, batch_z in enumerate(dm.train_dataloader()):
        with torch.no_grad():
            batch_selfies = vae_wrapper.latent_to_selfies(batch_z)
            batch_smiles = [sf.decoder(s) for s in batch_selfies]
            scores = smiles_to_desired_scores(batch_smiles, "pdop")

        batch_y = torch.tensor(scores, device=device, dtype=torch.float32)
        flat_batch_z = batch_z.reshape(batch_z.size(0), -1)
        update_surr_model(surrogate_model, surr_optim, flat_batch_z, batch_y, 100)

        batch_max_score, batch_max_idx = batch_y.max(dim=0)
        if batch_max_score.item() > max_score:
            max_score = batch_max_score.item()
            best_z = flat_batch_z[batch_max_idx].detach().clone()

        if i+1 == 16:
            break
    
    best_f = torch.tensor(max_score, device=device, dtype=torch.float32)
    best_s = vae_wrapper.latent_to_selfies(best_z.reshape(-1, diffusion.seq_length, diffusion.channels))[0]
    score = smile_to_guacamole_score("pdop", sf.decoder(best_s))

    tr_ub = best_z + 1.5
    tr_lb = best_z - 1.5
    bounds = torch.stack([tr_lb, tr_ub]).cuda()

    botorch_model = BoTorchDKLModelWrapper(surrogate_model).to(device).eval()
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
    qEI = qLogExpectedImprovement(model=botorch_model, best_f=best_f, sampler=sampler)

    def neg_qei(z, t):
        z = z.transpose(1,2)
        return -1 * qEI(z.reshape(z.shape[0], -1))

    def grad_qei(z, t):
        z = z.detach()
        z.requires_grad_(True)

        with torch.enable_grad():
            out = qEI(z.reshape(z.shape[0], -1))
            gradients = torch.autograd.grad(out, z, retain_graph=True)[0]
            gradients = torch.clamp(gradients, -100, 100)
            return gradients.view_as(z)

    summary = []
    for method in ["ddim_new_mean_cond", "ddim_no_cond", "optimize_acqf"]:
        print(f"Sampling with {method.upper()}...")
        if method == "ddpm_mean_cond":
            latents = diffusion.ddpm_sample(batch_size=batch_size, class_labels=None, cond_fn=grad_qei)
            latents = latents.transpose(1,2)
        elif method == "ddim_new_mean_cond":
            shape = (batch_size, diffusion.channels, diffusion.seq_length)
            latents = diffusion.ddim_sample(shape, class_labels=None, cond_fn=neg_qei, bounds=bounds, grad_scale=10.0, eta=0.1)
            latents = latents.transpose(1,2)
        elif method == "ddim_new_score_cond":
            shape = (batch_size, diffusion.channels, diffusion.seq_length)
            latents = diffusion.ddim_sample(shape, class_labels=None, cond_fn=neg_qei, bounds=bounds, score_cond=True)
            latents = latents.transpose(1,2)
        elif method == "ddim_orig_score_cond":
            shape = (batch_size, diffusion.channels, diffusion.seq_length)
            latents = diffusion.ddim_sample_orig(shape, class_labels=None, cond_fn=grad_qei)
            latents = latents.transpose(1,2)
        elif method == "optimize_acqf":
            latents, _ = optimize_acqf(qEI, bounds=bounds, q=batch_size, num_restarts=10, raw_samples=1024)
            latents = latents.reshape(-1, diffusion.seq_length, diffusion.channels)
        else:
            print(f"Method does not exist, defaulting to no conditioning.")
            shape = (batch_size, diffusion.channels, diffusion.seq_length)
            latents = diffusion.ddim_sample(shape, class_labels=None, cond_fn=None, bounds=bounds, eta=0.1)
            latents = latents.transpose(1,2)
        
        z = latents.reshape(latents.shape[0], -1)
        within_bounds_mask = ((z >= tr_lb.unsqueeze(0)) & (z <= tr_ub.unsqueeze(0))).all(dim=1)
        selfies = vae_wrapper.latent_to_selfies(latents)
        all_scores = []
        errors = 0
        for i, selfie in enumerate(selfies):
            smiles = sf.decoder(selfie)
            score = smile_to_guacamole_score("pdop", smiles)
            if score:
                print(f"Sample {i+1}: {smiles} | Valid: {is_valid_molecule(smiles)}, Score: {score:.2f}, Prev Best Score: {best_f:.2f}, Within Bounds: {within_bounds_mask[i]}")
                all_scores.append(score)
            else:
                print(f"Sample {i+1}: {smiles} | Valid: {is_valid_molecule(smiles)}, Within Bounds: {within_bounds_mask[i]}, error with score")
                errors += 1
        
        best_score = f"{max(all_scores):4f}" if len(all_scores) else "N/A"
        avg_score = f"{(sum(all_scores) / len(all_scores)):4f}" if len(all_scores) else "N/A"

        print(f"Final Best Score: {best_score}")

        qEI_score = qEI(latents.reshape(latents.shape[0], -1))
        summary.append({'method': method, 'best score': best_score, 'average score': avg_score, 'log qei': qEI_score.detach().cpu().item(), 'score errors': errors, 'bounds errors': (~within_bounds_mask).sum().item()})
    print(summary)



# === Entry point ===

def main():
    diffusion = load_diffusion_model()
    
    # validate_unconditional(diffusion, batch_size=2048)
    # validate_with_normal_analytical_cond(diffusion)
    # validate_with_logp_cond(diffusion)
    validate_with_gp(diffusion, 16)

if __name__ == "__main__":
    main()