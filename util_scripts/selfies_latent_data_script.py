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

# ====== MOL VAE ORIGINAL =====

from model import VAEFlatWrapper

device = torch.device("cuda")

vae_wrapper = VAEFlatWrapper(path_to_vae_statedict="checkpoints/SELFIES_VAE/epoch=447-step=139328.ckpt")

split = "val"

with torch.no_grad():
    selfie_path = Path("data/selfies/selfies_new") / f"{split}_selfie.gz"
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
            torch.cuda.empty_cache()

    mus_tensor = torch.stack(mus).squeeze(1)
    sigmas_tensor = torch.stack(sigmas).squeeze(1)

    output_dir = Path("data/selfies/selfies_flat")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"low_all_{split}.pt"
    torch.save((mus_tensor, sigmas_tensor), output_path)

    print(f"Saved {len(mus_tensor)} encodings to {output_path}")

m, s = torch.load(f"data/selfies/selfies_flat/low_all_{split}.pt")
print(m.shape, s.shape)


# ===== MOL VAE LOLBO =====

# from data.selfies_dataset_lolbo import SELFIESDataset
# from model.mol_vae_lolbo.mol_vae import InfoTransformerVAE

# device = torch.device("cuda")

# dataobj = SELFIESDataset()
# vae = InfoTransformerVAE(dataset=dataobj)

# state_dict = torch.load("checkpoints/SELFIES_VAE_LOLBO/SELFIES-VAE-state-dict.pt")
# vae.load_state_dict(state_dict, strict=True)
# vae = vae.cuda().eval()

# split = "val"

# with torch.no_grad():
#     selfie_path = Path("data/selfies_lolbo") / f"{split}_selfie.gz"
#     with gzip.open(selfie_path, "rt") as f:
#         selfies = [line.strip() for line in f if line.strip()]

#     print(f"Processed selfies strings, length {len(selfies)}")

#     mus, sigmas = [], []
#     for i, selfie in enumerate(selfies):
#         tokenized_selfie = dataobj.tokenize_selfies([selfie])[0]
#         tokens = dataobj.encode(tokenized_selfie).unsqueeze(0).cuda()
#         mu, sigma = vae.encode(tokens)
#         # import ipdb; ipdb.set_trace()
#         mus.append(mu.detach().cpu())
#         sigmas.append(sigma.detach().cpu())

#         if i % (len(selfies) // 50) == 0:
#             print(f"Reached batch {i}/{len(selfies)}")
#             torch.cuda.empty_cache()

#     mus_tensor = torch.stack(mus).squeeze(1)
#     sigmas_tensor = torch.stack(sigmas).squeeze(1)

#     output_path = Path("data/selfies_lolbo") / f"low_all_{split}.pt"
#     torch.save((mus_tensor, sigmas_tensor), output_path)

#     print(f"Saved {len(mus_tensor)} encodings to {output_path}")

# m, s = torch.load(f"data/selfies_lolbo/low_all_{split}.pt")
# print(m.shape, s.shape)








# device = torch.device("cuda")

# dataobj = SELFIESDataset()
# vae = InfoTransformerVAE(dataset=dataobj)

# state_dict = torch.load("checkpoints/SELFIES_VAE_LOLBO/SELFIES-VAE-state-dict.pt")
# vae.load_state_dict(state_dict, strict=True)
# vae = vae.cuda().eval()

# split = "train"

# with torch.no_grad():
#     selfie_path = Path("data/selfies_lolbo") / f"{split}_selfie.gz"
#     with gzip.open(selfie_path, "rt") as f:
#         selfies = [line.strip() for line in f if line.strip()]

#     print(f"Processed selfies strings, length {len(selfies)}")

#     mus, sigmas = [], []
#     for i, selfie in enumerate(selfies):
#         tokenized_selfie = dataobj.tokenize_selfies([selfie])[0]
#         tokens = dataobj.encode(tokenized_selfie).unsqueeze(0).cuda()
#         mu, sigma = vae.encode(tokens)
#         # import ipdb; ipdb.set_trace()
#         mus.append(mu.detach().cpu())
#         sigmas.append(sigma.detach().cpu())

#         if i % (len(selfies) // 50) == 0:
#             print(f"Reached batch {i}/{len(selfies)}")
        
#         if i == (len(selfies) // 3):
#             mus_tensor = torch.stack(mus).squeeze(1)
#             sigmas_tensor = torch.stack(sigmas).squeeze(1)

#             output_path = Path("data/selfies_lolbo") / f"low_all_{split}_1.pt"
#             torch.save((mus_tensor, sigmas_tensor), output_path)

#             print(f"Saved {len(mus_tensor)} encodings to {output_path} (1)")

#             mus = []
#             sigmas = []

#             torch.cuda.empty_cache()
        
#         if i == 2*(len(selfies) // 3):
#             mus_tensor = torch.stack(mus).squeeze(1)
#             sigmas_tensor = torch.stack(sigmas).squeeze(1)

#             output_path = Path("data/selfies_lolbo") / f"low_all_{split}_2.pt"
#             torch.save((mus_tensor, sigmas_tensor), output_path)

#             print(f"Saved {len(mus_tensor)} encodings to {output_path} (2)")

#             mus = []
#             sigmas = []

#             torch.cuda.empty_cache()

#     mus_tensor = torch.stack(mus).squeeze(1)
#     sigmas_tensor = torch.stack(sigmas).squeeze(1)

#     output_path = Path("data/selfies_lolbo") / f"low_all_{split}_3.pt"
#     torch.save((mus_tensor, sigmas_tensor), output_path)

#     print(f"Saved {len(mus_tensor)} encodings to {output_path} (3)")

# m, s = torch.load(f"data/selfies_lolbo/low_all_{split}_1.pt")
# print(m.shape, s.shape)

# m, s = torch.load(f"data/selfies_lolbo/low_all_{split}_2.pt")
# print(m.shape, s.shape)

# m, s = torch.load(f"data/selfies_lolbo/low_all_{split}_3.pt")
# print(m.shape, s.shape)