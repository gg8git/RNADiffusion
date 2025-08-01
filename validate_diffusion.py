import json
from typing import List, Union
import selfies as sf

import lightning as L
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm
from scipy import stats
from fcd.fcd import get_fcd

from model.GaussianDiffusion import GaussianDiffusion1D
from model.UNet1D import KarrasUnet1D

from model.mol_vae_model.wrapper import VAEWrapper

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# === VAE MODEL ===

vae_wrapper = VAEWrapper(path_to_vae_statedict="checkpoints/SELFIES_VAE/epoch=447-step=139328.ckpt")

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
    import ipdb; ipdb.set_trace()
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


# def validate_with_logp_cond(diffusion, batch_size=32):
#     print("=== Conditional Sampling (LogP Condition) ===")
#     # Placeholder cond_fn – to be replaced with a proper differentiable cond_fn
#     cond_fn = logp_cond_fn(logp_fn=penalized_logp, tokenizer=None)
#     shape = (batch_size, diffusion.channels, diffusion.seq_length)
#     samples = diffusion.ddim_sample(shape, class_labels=None, cond_fn=cond_fn)

#     for i, sample in enumerate(samples):
#         smiles = tokens_to_smiles(sample.unsqueeze(0))
#         logp = penalized_logp(smiles)
#         print(f"Sample {i+1}: {smiles} | LogP: {logp:.2f} | Valid: {is_valid_molecule(smiles)}")

# === Entry point ===

def main():
    diffusion = load_diffusion_model()
    
    # validate_unconditional(diffusion, batch_size=2048)
    validate_with_normal_analytical_cond(diffusion)
    # validate_with_logp_cond(diffusion)

if __name__ == "__main__":
    main()