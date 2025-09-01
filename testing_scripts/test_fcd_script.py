import gzip
from pathlib import Path

import lightning as L
import torch

import selfies as sf
from rdkit import Chem
from fcd import get_fcd

from model import GaussianDiffusion1D, KarrasUnet1D, VAEFlatWrapper
from datamodules import DiffusionDataModule

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


def diffuse_batch(batch_size, diffusion_model):
    return diffusion_model.ddim_sample(
        batch_size=batch_size,
        sampling_timesteps=50,
        use_self_cond=True,
    )


def latent_to_smiles(z):
    vae_wrapper = VAEFlatWrapper(path_to_vae_statedict="checkpoints/SELFIES_VAE/epoch=447-step=139328.ckpt")

    selfies = vae_wrapper.latent_to_selfies(z)
    return [sf.decoder(selfie) for selfie in selfies]


def prior_to_smiles(batch_size):
    vae_wrapper = VAEFlatWrapper(path_to_vae_statedict="checkpoints/SELFIES_VAE/epoch=447-step=139328.ckpt")

    selfies, _ = vae_wrapper.sample_selfies_from_prior(batch_size=batch_size)
    return [sf.decoder(selfie) for selfie in selfies]


# def load_smiles_dataset(dataset_path, split, batch_size=None):
#     selfie_path = Path(dataset_path) / f"{split}_selfie.gz"
#     selfies = []
#     with gzip.open(selfie_path, "rt") as f:
#         if batch_size is None:
#             selfies = [line.strip() for line in f if line.strip()]
#         else:
#             for line in f:
#                 if line.strip():
#                     selfies.append(line.strip())
#     smiles = [sf.decoder(selfie) for selfie in selfies]

#     return {"SMILES": smiles, "SELFIES": selfies}


def load_smiles_dataset(dataset_path, split, batch_size=None):
    batch_size = 10000 if batch_size is None else batch_size

    dm = DiffusionDataModule(
        data_dir=dataset_path,
        batch_size=batch_size,
        num_workers=0,
    )

    if split == "train":
        dm_z = next(iter(dm.train_dataloader()))
    elif split == "val":
        dm_z = next(iter(dm.val_dataloader()))
    else:
        dm_z = next(iter(dm.test_dataloader()))
    
    return {"SMILES": latent_to_smiles(dm_z)}


def try_canon(smi: str):
    try:
        return Chem.CanonSmiles(smi)
    except:
        return None


BATCH_SIZE = 1024

model = load_diffusion_model(load_model_checkpoint="SELFIES_Diffusion/oflvuzyp/checkpoints/last.ckpt")
ddim_z = diffuse_batch(batch_size=BATCH_SIZE, diffusion_model=model)
ddim_smiles = latent_to_smiles(z=ddim_z)

ddim_smiles = [try_canon(smi) for smi in ddim_smiles]
ddim_smiles = [smi for smi in ddim_smiles if smi is not None]


vae_smiles = prior_to_smiles(batch_size=BATCH_SIZE)

vae_smiles = [try_canon(smi) for smi in vae_smiles]
vae_smiles = [smi for smi in vae_smiles if smi is not None]


# ds_smiles = load_smiles_dataset("data/selfies/selfies_new", split="train", batch_size=BATCH_SIZE)["SMILES"][:BATCH_SIZE]
ds_smiles = load_smiles_dataset("data/selfies/selfies_flat", split="train", batch_size=BATCH_SIZE)["SMILES"][:BATCH_SIZE]

ds_smiles = [try_canon(smi) for smi in ds_smiles]
ds_smiles = [smi for smi in ds_smiles if smi is not None]


print(f"VAE-DDIM: {get_fcd(vae_smiles, ddim_smiles):.3f}")
print(f"VAE-Data: {get_fcd(vae_smiles, ds_smiles):.3f}")
print(f"DDIM-Data: {get_fcd(ddim_smiles, ds_smiles):.3f}")