import datasets
import numpy as np
import polars as pl
import torch
from tqdm import tqdm

from model.diffusion_v2 import BaseVAE

torch.set_float32_matmul_precision("medium")


def encode_split(vae: BaseVAE, subset: datasets.Dataset, batch_size: int = 4096):
    print(f"Encoding {len(subset):,} samples...")
    mus = []
    sigmas = []

    niter = len(subset) // batch_size + int(len(subset) % batch_size > 0)
    for batch in tqdm(subset.iter(batch_size=batch_size), total=niter):
        tokens = vae.tokenize(batch["input"])  # type: ignore

        with torch.autocast(device_type=vae.device.type, dtype=torch.bfloat16):
            out = vae(tokens)

        mu = out["mu_ign"]
        sigma = out["sigma_ign"]
        string_acc = out["string_acc"]
        if string_acc < 0.8:
            raise ValueError(f"Low string accuracy: {string_acc}")

        mu = mu.float().cpu().flatten(1)
        sigma = sigma.float().cpu().flatten(1)

        mus.append(mu.numpy())
        sigmas.append(sigma.numpy())

    mus = np.concatenate(mus, axis=0)
    sigmas = np.concatenate(sigmas, axis=0)
    return mus, sigmas


#############################################################
# Peptide
#############################################################
vae = BaseVAE.load_from_checkpoint("./data/peptide_vae.ckpt")
vae.freeze()
vae.cuda()

ds: datasets.Dataset = datasets.Dataset.from_csv("./data/Extinct_labels_10M.csv")  # type: ignore
ds = ds.rename_column("text", "input")  # type: ignore

train, valtest = ds.train_test_split(test_size=0.2).values()  # type: ignore
val, test = valtest.train_test_split(test_size=0.5).values()
ds = datasets.DatasetDict({"train": train, "val": val, "test": test})  # type: ignore

out_ds = []
for split in ds:
    subset = ds[split]
    mus, sigmas = encode_split(vae, subset, batch_size=4096)  # type: ignore
    df = pl.DataFrame({"input": subset["input"], "mu": mus, "sigma": sigmas})
    df.write_parquet(f"./data/latents/peptide_{split}.parquet")

#############################################################
# Molecule
#############################################################
vae = BaseVAE.load_from_checkpoint("./data/molecule_vae.ckpt")
vae.freeze()
vae.cuda()

ds: datasets.DatasetDict = datasets.load_dataset("haydn-jones/Guacamol", num_proc=8)  # type: ignore
ds = ds.rename_column("SMILES", "input")

out_ds = []
for split in ds:
    subset = ds[split]
    mus, sigmas = encode_split(vae, subset, batch_size=4096)  # type: ignore
    df = pl.DataFrame({"input": subset["input"], "mu": mus, "sigma": sigmas})
    df.write_parquet(f"./data/latents/molecule_{split}.parquet")
