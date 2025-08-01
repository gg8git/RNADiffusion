import json
import time
import gzip
from pathlib import Path
from typing import List, Union

import lightning as L
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

import selfies as sf
from model.mol_vae_model.BaseMolVAE import BaseVAE
from model.mol_vae_model.MolVAE import VAEModule
from model.mol_vae_model.wrapper import VAEWrapper

device = torch.device("cuda")

vae_wrapper = VAEWrapper(path_to_vae_statedict="checkpoints/SELFIES_VAE/epoch=447-step=139328.ckpt")

split = "test"

with torch.no_grad():
    selfie_path = Path("data/selfies") / f"{split}_selfie.gz"
    with gzip.open(selfie_path, "rt") as f:
        selfies = [f"[start]{line.strip()}[stop]" for line in f if line.strip()]

    print(f"Processed selfies strings, length {len(selfies)}")

    mus, sigmas = [], []
    for i, selfie in enumerate(selfies):
        mu, sigma = vae_wrapper.selfies_to_latent_params(selfie)
        mus.append(mu)
        sigmas.append(sigma)

        if i % (len(selfies) // 50) == 0:
            print(f"Reached batch {i}/{len(selfies)}")

    mus_tensor = torch.stack(mus).squeeze(1)
    sigmas_tensor = torch.stack(sigmas).squeeze(1)

    output_path = Path("data/selfies") / f"low_all_{split}.pt"
    torch.save((mus_tensor, sigmas_tensor), output_path)

    print(f"Saved {len(mus_tensor)} encodings to {output_path}")

m, s = torch.load(f"data/selfies/low_all_{split}.pt")
print(m.shape, s.shape)