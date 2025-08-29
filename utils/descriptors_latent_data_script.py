import pandas as pd
import selfies as sf
import torch
import numpy as np
from pathlib import Path
from model.mol_vae_model.wrapper import VAEWrapper

# Read the parquet file
df = pd.read_parquet("data/descriptors/descriptors.parquet")

# Precompute QED class
if "qed_class" not in df:
    df["qed_class"] = (df["qed"] > 0.5).astype(int)

df.to_parquet("data/descriptors/descriptors.parquet")

# Initialize VAE
device = torch.device("cuda")
vae_wrapper = VAEWrapper(path_to_vae_statedict="checkpoints/SELFIES_VAE/epoch=447-step=139328.ckpt")

# Storage for valid rows
valid_idx = []
mus, sigmas = [], []
errors = 0

smiles_list = df["SMILES"].tolist()
for i, smile in enumerate(smiles_list):
    try:
        selfie = sf.encoder(smile)
        mu, sigma = vae_wrapper.selfies_to_latent_params(selfie)
        mus.append(mu.detach().cpu())
        sigmas.append(sigma.detach().cpu())
        valid_idx.append(i)
    except Exception as e:
        errors += 1
        # print(f"error #{errors} at index {i}: {e}")
    
    if i % (len(df) // 50) == 0:
        print(f"Reached batch {i}/{len(df)}, with errors {errors}/{i}")
        torch.cuda.empty_cache()

print(f"Successful molecules: {len(valid_idx)}, errors: {errors}")

# Build clean dataframe
clean_df = df.iloc[valid_idx].reset_index(drop=True)
mus_tensor = torch.stack(mus).squeeze(1)
sigmas_tensor = torch.stack(sigmas).squeeze(1)

# Precompute QED class
clean_df["qed_class"] = (clean_df["qed"] > 0.5).astype(int)

# Allocate splits AFTER cleaning
np.random.seed(42)
probs = [0.8, 0.15, 0.05]
clean_df["split"] = np.random.choice(["train", "test", "val"], size=len(clean_df), p=probs)

clean_df.to_parquet("data/descriptors/descriptors_clean.parquet")

# Save tensors per split
for split in ["test", "train", "val"]:
    split_df = clean_df[clean_df["split"] == split]
    idxs = split_df.index

    torch.save(
        (
            mus_tensor[idxs],
            sigmas_tensor[idxs],
            torch.tensor(split_df["qed_class"].values, dtype=torch.long),
            torch.tensor(split_df["qed"].values, dtype=torch.float32),
            torch.tensor(split_df["fsp3"].values, dtype=torch.float32),
        ),
        Path("data/descriptors") / f"low_all_{split}.pt",
    )
    print(f"Saved {len(split_df)} entries to low_all_{split}.pt")