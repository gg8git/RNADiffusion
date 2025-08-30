import json
from typing import List, Union
import selfies as sf

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, TensorDataset

from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm
from scipy import stats
from fcd.fcd import get_fcd

from botorch.acquisition import qExpectedImprovement
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim import optimize_acqf
import gpytorch
from gpytorch.mlls import PredictiveLogLikelihood

from utils.guacamol_utils import smiles_to_desired_scores, smile_to_guacamole_score
from datamodules.diffusion_datamodule import DiffusionDataModule, LatentDatasetClassifier, LatentDatasetDescriptors
from model import GPModelDKL, GaussianDiffusion1D, KarrasUnet1D
from model.mol_score_model.conv_classifier import ConvScoreClassifier
from model.surrogate_model.wrapper import BoTorchDKLModelWrapper

from RNADiffusion.model.mol_vae_model.FlatWrapper import VAEFlatWrapper

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# === VAE MODEL ===

vae_wrapper = VAEFlatWrapper(path_to_vae_statedict="checkpoints/SELFIES_VAE/epoch=447-step=139328.ckpt")


# === GP methods ===

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

# === Molecule decoding and evaluation ===

with open("data/selfies/selfies_vocab.json") as f:
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


def load_diffusion_model_lolbo():
    class Wrapper(L.LightningModule):
        def __init__(self):
            super().__init__()
            model = KarrasUnet1D(
                seq_len=8,
                dim=64,
                dim_max=128,
                channels=32,
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
            return self.diffusion(z)

        def training_step(self, batch, batch_idx):
            batch = batch.reshape(-1,8,32).transpose(1,2)
            loss = self.forward(batch)
            self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.diffusion.parameters(), lr=3e-4)
    
    model = Wrapper()
    
    ckpt = torch.load("SELFIES_Diffusion/jj934sg6/checkpoints/last.ckpt", map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    diffusion = model.diffusion
    diffusion.eval()
    return diffusion.cuda()


def load_qed_classifier():
    class Wrapper(L.LightningModule):
        def __init__(self):
            super().__init__()
            self.save_hyperparameters()
            self.predictor = ConvScoreClassifier()
    
    model = Wrapper()

    ckpt = torch.load("SELFIES_Diffusion/g19ugzc1/checkpoints/last.ckpt", map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    classifier = model.predictor
    classifier.eval()
    return classifier.cuda()


# === Validation functions ===

def validate_unconditional(diffusion, batch_size=64):
    print("=== Unconditional Sampling ===")
    with torch.no_grad():
        for method in ["ddim"]:
            print(f"Sampling with {method.upper()}...")
            if method == "ddpm":
                latents = diffusion.ddpm_sample(batch_size=batch_size, class_labels=None)
                latents = latents.transpose(1,2)
            else:
                shape = (batch_size, diffusion.channels, diffusion.seq_length)
                latents = diffusion.ddim_sample_haydn(shape, cond_fn=None, guidance_scale=1.0)
                latents = latents.transpose(1,2)

            # do get_fcd() from frechet dist github with smiles1 from vae prior+vae and smiles2 from diffusion+vae
            diff_selfies = vae_wrapper.latent_to_selfies(latents)
            vae_selfies, z = vae_wrapper.sample_selfies_from_prior(batch_size)
            import ipdb; ipdb.set_trace()
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

    for method in ["ddim_haydn", "ddim_new_mean_cond", "ddim_no_cond"]:
        print(f"Sampling with {method.upper()}...")
        if method == "ddpm":
            latents = diffusion.ddpm_sample(batch_size=batch_size, class_labels=None, cond_fn=lambda x,t: cond_fn_grad_normal_dist(x.transpose(1,2), t).transpose(1,2))
        elif method == "ddim":
            shape = (batch_size, diffusion.channels, diffusion.seq_length)
            latents = diffusion.ddim_sample(shape, class_labels=None, cond_fn=lambda x,t: cond_fn_normal_dist(x.transpose(1,2), t))
        elif method == "ddim_new_mean_cond":
            shape = (batch_size, diffusion.channels, diffusion.seq_length)
            latents = diffusion.ddim_sample(shape, class_labels=None, cond_fn=lambda x,t: cond_fn_normal_dist(x.transpose(1,2), t), grad_scale=10.0, eta=0.1)
            latents = latents.transpose(1,2)
        elif method == "ddim_haydn":
            shape = (batch_size, diffusion.channels, diffusion.seq_length)
            latents = diffusion.ddim_sample_haydn(shape, cond_fn=lambda x,t: cond_fn_grad_normal_dist(x.transpose(1,2), t).transpose(1,2), guidance_scale=1.0)
            latents = latents.transpose(1,2)
        else:
            print(f"Method does not exist, defaulting to no conditioning.")
            shape = (batch_size, diffusion.channels, diffusion.seq_length)
            latents = diffusion.ddim_sample(shape, class_labels=None, cond_fn=None, eta=0.1)
            latents = latents.transpose(1,2)

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

    surrogate_model = GPModelDKL(batch.reshape(data_batch_size, -1), likelihood=gpytorch.likelihoods.GaussianLikelihood().cuda()).cuda()
    surrogate_mll = PredictiveLogLikelihood(surrogate_model.likelihood, surrogate_model, num_data=data_batch_size)
    surrogate_model.eval()

    max_score = float('-inf')
    best_z = None

    for i, batch_z in enumerate(dm.train_dataloader()):
        with torch.no_grad():
            batch_selfies = vae_wrapper.latent_to_selfies(batch_z)
            batch_smiles = [sf.decoder(s) for s in batch_selfies]
            scores = smiles_to_desired_scores(batch_smiles, "pdop")

        batch_y = torch.tensor(scores, device=device, dtype=torch.float32)
        flat_batch_z = batch_z.reshape(batch_z.size(0), -1)
        surrogate_model = update_surr_model(surrogate_model, surrogate_mll, 0.005, flat_batch_z, batch_y, 50)

        batch_max_score, batch_max_idx = batch_y.max(dim=0)
        if batch_max_score.item() > max_score:
            max_score = batch_max_score.item()
            best_z = flat_batch_z[batch_max_idx].detach().clone()

        if i+1 == 32:
            break
    
    best_f = torch.tensor(max_score, device=device, dtype=torch.float32)
    best_s = vae_wrapper.latent_to_selfies(best_z.reshape(-1, diffusion.seq_length, diffusion.channels))[0]
    score = smile_to_guacamole_score("pdop", sf.decoder(best_s))

    tr_ub = best_z + 1.5
    tr_lb = best_z - 1.5
    bounds = torch.stack([tr_lb, tr_ub]).cuda()

    lb = torch.full_like(best_z, -5.5)
    ub = torch.full_like(best_z,  5.5)
    bounds = torch.stack([lb, ub]).cuda()

    # botorch_model = BoTorchDKLModelWrapper(surrogate_model).to(device).eval()
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
    log_qEI = qLogExpectedImprovement(model=surrogate_model.cuda(), best_f=best_f, sampler=sampler)
    qEI = qExpectedImprovement(model=surrogate_model.cuda(), best_f=best_f, sampler=sampler)

    def cond_fn_log_qei(z, t):
        z = z.transpose(1,2)
        z_flat = z.reshape(z.shape[0], -1)
        return -1 * log_qEI(z_flat)

    def grad_qei(z, t):
        z = z.detach()
        z.requires_grad_(True)

        with torch.enable_grad():
            out = log_qEI(z.reshape(z.shape[0], -1))
            gradients = torch.autograd.grad(out, z, retain_graph=True)[0]
            gradients = torch.clamp(gradients, -100, 100)
            return gradients.view_as(z)

    summary = []
    for method in ["ddim_haydn", "ddim_new_mean_cond", "ddim_no_cond", "optimize_acqf"]:
        print(f"Sampling with {method.upper()}...")
        if method == "ddpm_mean_cond":
            latents = diffusion.ddpm_sample(batch_size=batch_size, class_labels=None, cond_fn=grad_qei)
            latents = latents.transpose(1,2)
        elif method == "ddim_new_mean_cond":
            shape = (batch_size, diffusion.channels, diffusion.seq_length)
            latents = diffusion.ddim_sample(shape, class_labels=None, cond_fn=cond_fn_log_qei, bounds=bounds, grad_scale=10.0, eta=0.1)
            latents = latents.transpose(1,2)
        elif method == "ddim_new_score_cond":
            shape = (batch_size, diffusion.channels, diffusion.seq_length)
            latents = diffusion.ddim_sample(shape, class_labels=None, cond_fn=cond_fn_log_qei, bounds=bounds, score_cond=True)
            latents = latents.transpose(1,2)
        elif method == "ddim_orig_score_cond":
            shape = (batch_size, diffusion.channels, diffusion.seq_length)
            latents = diffusion.ddim_sample_orig(shape, class_labels=None, cond_fn=grad_qei)
            latents = latents.transpose(1,2)
        elif method == "ddim_haydn":
            def log_prob_fn_ei(x, eps=1e-8, tau=None):
                vals = qEI(x).clamp_min(eps)
                if tau is None:
                    tau = torch.quantile(vals.detach(), 0.5).clamp_min(eps)
                squashed = vals / (vals + tau)
                return torch.log(squashed + eps) 
                
            grad_qei_haydn = get_cond_fn(log_prob_fn_ei, clip_grad=False, latent_dim=(diffusion.channels * diffusion.seq_length))
            shape = (batch_size, diffusion.channels, diffusion.seq_length)
            latents = diffusion.ddim_sample_haydn(shape, cond_fn=lambda x, t: grad_qei_haydn(x.transpose(1,2), t).reshape(-1, diffusion.seq_length, diffusion.channels).transpose(1,2), guidance_scale=8.0)
            latents = latents.transpose(1,2)
        elif method == "optimize_acqf":
            latents, _ = optimize_acqf(log_qEI, bounds=bounds, q=batch_size, num_restarts=10, raw_samples=1024)
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

        qEI_score = log_qEI(latents.reshape(latents.shape[0], -1))
        summary.append({'method': method, 'best score': best_score, 'average score': avg_score, 'log qei': qEI_score.detach().cpu().item(), 'score errors': errors, 'bounds errors': (~within_bounds_mask).sum().item()})
    print(summary)


def validate_with_descriptor_classifier(diffusion, classifier, batch_size=64):
    print("=== Conditional Sampling (Descriptor Classifier) ===")
    
    def classify(z: torch.Tensor) -> torch.Tensor:
        return torch.softmax(classifier(z), dim=1)

    def eval_probs(z: torch.Tensor) -> None:
        probs = classify(z)
        print(f"Diffusion Probs: {probs}")
        argmax = torch.argmax(probs, dim=1)
        print(f"Percent Accuracy: {argmax.sum() / len(argmax)}")


    def log_prob_fn_classifier(z: torch.Tensor) -> torch.Tensor:
        z = z.reshape(-1, diffusion.seq_length, diffusion.channels) # B 8 16
        logits = classifier(z)
        log_probs = F.log_softmax(logits, dim=1)
        return log_probs[:, 1].sum()

    cond_fn_classifier = get_cond_fn(
        log_prob_fn=log_prob_fn_classifier,
        clip_grad=False,
        latent_dim=(diffusion.seq_length * diffusion.channels),
    )

    def cond_fn_classifier_reshaped(x, t):
        x = x.transpose(1, 2).reshape(x.shape[0], -1)
        grad = cond_fn_classifier(x, t)
        grad = grad.reshape(-1, diffusion.seq_length, diffusion.channels).transpose(1,2)
        return grad

    shape = (batch_size, diffusion.channels, diffusion.seq_length)
    latents = diffusion.ddim_sample_haydn(shape, cond_fn=cond_fn_classifier_reshaped, guidance_scale=8.0)
    latents = latents.transpose(1,2)

    eval_probs(latents)


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
                curr_summary = {}

                num_restarts = 0
                log_qei_score = "N/A"
                while log_qei_score == "N/A" and num_restarts < 5:
                    try:
                        shape = (batch_size, diffusion.channels, diffusion.seq_length)
                        latents = diffusion.ddim_sample_haydn(shape, cond_fn=None, guidance_scale=1.0)
                        latents = latents.transpose(1,2)  # with transpose
                        log_qei_score = log_qEI(latents.reshape(latents.shape[0], -1)).detach().cpu().item()
                    except Exception as e:
                        num_restarts += 1
                curr_summary['no cond'] = {'log qei score': log_qei_score, 'num restarts': num_restarts}
                print(f"log qei no cond: {log_qei_score}")
                torch.cuda.empty_cache()

                num_restarts = 0
                log_qei_score = "N/A"
                while log_qei_score == "N/A" and num_restarts < 5:
                    try:
                        shape = (batch_size, diffusion.channels, diffusion.seq_length)
                        latents = diffusion.ddim_sample_haydn(shape, cond_fn=cond_fn_ei_reshaped, guidance_scale=25.0)
                        latents = latents.transpose(1,2)  # with transpose
                        log_qei_score = log_qEI(latents.reshape(latents.shape[0], -1)).detach().cpu().item()
                    except Exception as e:
                        num_restarts += 1
                curr_summary['ddim'] = {'log qei score': log_qei_score, 'num restarts': num_restarts}
                print(f"log qei ddim: {log_qei_score}")
                torch.cuda.empty_cache()

                num_restarts = 0
                log_qei_score = "N/A"
                while log_qei_score == "N/A" and num_restarts < 5:
                    try:
                        shape = (batch_size, diffusion.channels, diffusion.seq_length)
                        latents = diffusion.ddim_sample_haydn(shape, sampling_timesteps=50, cond_fn=cond_fn_ei_reshaped, guidance_scale=25.0)
                        latents = latents.transpose(1,2)  # with transpose
                        log_qei_score = log_qEI(latents.reshape(latents.shape[0], -1)).detach().cpu().item()
                    except Exception as e:
                        num_restarts += 1
                curr_summary['ddim 50'] = {'log qei score': log_qei_score, 'num restarts': num_restarts}
                print(f"log qei ddim 50: {log_qei_score}")
                torch.cuda.empty_cache()

                num_restarts = 0
                log_qei_score = "N/A"
                while log_qei_score == "N/A" and num_restarts < 5:
                    try:
                        shape = (batch_size, diffusion.channels, diffusion.seq_length)
                        latents = diffusion.dpmpp2m_sample(shape, sampling_timesteps=50, cond_fn=cond_fn_ei_reshaped, guidance_scale=25.0)
                        latents = latents.transpose(1,2)  # with transpose
                        log_qei_score = log_qEI(latents.reshape(latents.shape[0], -1)).detach().cpu().item()
                    except Exception as e:
                        num_restarts += 1
                curr_summary['dpmpp2m 50'] = {'log qei score': log_qei_score, 'num restarts': num_restarts}
                print(f"log qei dpmpp2m 50: {log_qei_score}")
                torch.cuda.empty_cache()

                num_restarts = 0
                log_qei_score = "N/A"
                while log_qei_score == "N/A" and num_restarts < 5:
                    try:
                        shape = (batch_size, diffusion.channels, diffusion.seq_length)
                        latents = diffusion.dpmpp2msde_sample(shape, sampling_timesteps=50, cond_fn=cond_fn_ei_reshaped, guidance_scale=25.0)
                        latents = latents.transpose(1,2)  # with transpose
                        log_qei_score = log_qEI(latents.reshape(latents.shape[0], -1)).detach().cpu().item()
                    except Exception as e:
                        num_restarts += 1
                curr_summary['dpmppp2msde 50'] = {'log qei score': log_qei_score, 'num restarts': num_restarts}
                print(f"log qei dpmppp2msde 50: {log_qei_score}")
                torch.cuda.empty_cache()

                num_restarts = 0
                log_qei_score = "N/A"
                while log_qei_score == "N/A" and num_restarts < 5:
                    try:
                        latents, _ = optimize_acqf(log_qEI, bounds=bounds, q=batch_size, num_restarts=10, raw_samples=1024)
                        latents = latents.reshape(-1, diffusion.seq_length, diffusion.channels)
                        log_qei_score = log_qEI(latents.reshape(latents.shape[0], -1)).detach().cpu().item()
                    except Exception as e:
                        num_restarts += 1
                curr_summary['optimize acqf'] = {'log qei score': log_qei_score, 'num restarts': num_restarts}
                print(f"log qei optimize acqf: {log_qei_score}")
                torch.cuda.empty_cache()

                summary[f"(iter: {i+1}, bsz: {batch_size})"] = curr_summary
                with open("./log_yimeng.json", 'w') as file:
                    json.dump(summary, file, indent=2)
        
        if i > max(surr_iters):
            break
    
    with open("./log_yimeng.json", 'w') as file:
        json.dump(summary, file, indent=2)
    print(summary)

# === Entry point ===

def main():
    diffusion = load_diffusion_model()
    classifier = load_qed_classifier()
    
    # validate_unconditional(diffusion, batch_size=2048)
    # validate_with_normal_analytical_cond(diffusion)
    # validate_with_logp_cond(diffusion)
    # validate_with_gp(diffusion, 32)
    # validate_with_descriptor_classifier(diffusion=diffusion, classifier=classifier, batch_size=16)
    # validate_with_descriptor_gp(diffusion=diffusion, batch_sizes=[4, 8, 16], surr_iters = [1, 4, 16])
    # import ipdb; ipdb.set_trace()
    validate_with_descriptor_gp(diffusion=diffusion, batch_sizes=[4, 16, 64, 128, 256], surr_iters = [4, 8, 16, 32, 64])

if __name__ == "__main__":
    main()