import torch
from torch import Tensor
from torch.distributions import Normal
from torch.func import grad, vmap

from model.diffusion_v2 import DiffusionModel

torch.set_printoptions(sci_mode=False, precision=3, linewidth=120)

model = DiffusionModel.load_from_checkpoint("./DiffuseOpt/djvqxl9n/checkpoints/last.ckpt")
model.cuda()
model.freeze()

MU = -3.0
STD = 0.1


def cond_fn(x: Tensor, t: Tensor) -> Tensor:
    dist = Normal(MU, STD)
    grad_fn = grad(lambda x: dist.log_prob(x).sum())
    return vmap(grad_fn, in_dims=(0,))(x)


z = model.ddim_sample(
    batch_size=1024,
    sampling_steps=50,
    cond_fn=cond_fn,
    guidance_scale=1.0,
)

print(f"Target: {MU:.3f} +/- {STD:.3f}")
print(f"Sample: {z.mean():.3f} +/- {z.std():.3f}")

ddim_z = model.ddim_sample(
    batch_size=1024,
    sampling_steps=50,
)
ddim_tokens = model.vae.sample(ddim_z)
ddim_smiles = model.vae.detokenize(ddim_tokens)
print(ddim_smiles[:3])

vae_z = torch.randn(1024, model.d_bn * model.n_bn)
vae_tokens = model.vae.sample(vae_z)
vae_smiles = model.vae.detokenize(vae_tokens)
print(vae_smiles[:3])
