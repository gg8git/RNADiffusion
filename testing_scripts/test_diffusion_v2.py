import torch
import torch.nn.functional as F
from datasets import load_dataset
from fcd import get_fcd
from rdkit import Chem, RDLogger
from torch import Tensor
from torch.distributions import Normal
from torch.func import grad, vmap

from model.diffusion_v2 import DiffusionModel
from model.diffusion_v2.GaussianDiffusion import ExtinctPredictor

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
