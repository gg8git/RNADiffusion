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
from RNADiffusion.model.BaseMolVAE import BaseVAE
from RNADiffusion.model.MolVAE import VAEModule


class SELFIESDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str, vocab: dict[str, int], batch_size: int, num_workers: int, latent: bool = False, path_to_vae_statedict: str = None) -> None:
        super().__init__()

        self.data_dir = data_dir
        self.vocab = vocab

        self.batch_size = batch_size
        self.num_workers = num_workers

        if '[start]' not in self.vocab:
            raise ValueError("Vocab must contain '[start]' token")
        if '[stop]' not in self.vocab:
            raise ValueError("Vocab must contain '[stop]' token")
        if '[pad]' not in self.vocab:
            raise ValueError("Vocab must contain '[pad]' token")

        self.latent = latent
        if self.latent:
            assert path_to_vae_statedict is not None
            self.path_to_vae_statedict = path_to_vae_statedict

    def train_dataloader(self) -> DataLoader:
        ds = SELFIESDataset(self.data_dir, 'train', self.vocab, latent=self.latent, path_to_vae_statedict=self.path_to_vae_statedict)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=ds.get_collate_fn()
        )

    def val_dataloader(self) -> DataLoader:
        ds = SELFIESDataset(self.data_dir, 'val', self.vocab, latent=self.latent, path_to_vae_statedict=self.path_to_vae_statedict)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=ds.get_collate_fn()
        )

    def test_dataloader(self) -> DataLoader:
        ds = SELFIESDataset(self.data_dir, 'test', self.vocab, latent=self.latent, path_to_vae_statedict=self.path_to_vae_statedict)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=ds.get_collate_fn()
        )

class SELFIESDataset(Dataset):
    def __init__(self, data_dir: str, split: str, vocab: dict[str, int], latent: bool = False, path_to_vae_statedict: str = None) -> None:
        super().__init__()

        self.data_dir = data_dir
        self.split = split
        self.vocab = vocab
        self.latent = latent

        path = Path(data_dir) / f'{split}_selfie.gz'

        with gzip.open(path, 'rt') as f:
            self.selfies = [l.strip() for l in f.readlines()]
        
        if self.latent:
            assert path_to_vae_statedict is not None
            self.encoder = VAE(path_to_vae_statedict=path_to_vae_statedict)

    def __len__(self) -> int:
        return len(self.selfies)

    def __getitem__(self, index: int) -> torch.Tensor:
        selfie = f"[start]{self.selfies[index]}[stop]"
        
        if self.latent:
            return self.encoder(selfie)
        else:
            tokens = fast_split(selfie)
            return torch.tensor([self.vocab[tok] for tok in tokens])

    def get_collate_fn(self):
        def collate(batch: List[torch.Tensor]) -> torch.Tensor:
            return pad_sequence(batch, batch_first=True, padding_value=self.vocab['[pad]'])
        return collate

# Faster than sf.split_selfies because it doesn't check for invalid selfies
def fast_split(selfie: str) -> list[str]:
    return [f"[{tok}" for tok in selfie.split("[") if tok]


device = torch.device("cuda")

class VAE(L.LightningModule):
    def __init__(self, path_to_vae_statedict, vocab_path="data/selfies_vocab.json"):
        """Load a VAE model for SELFIES representation"""

        # Load vocabulary
        with open(vocab_path) as f:
            vocab = json.load(f)
        
        # Initialize model with same architecture as your training script
        model = BaseVAE(
            vocab,
            d_bnk=16,
            n_acc=8,
            
            d_dec=64,
            decoder_num_layers=3,
            decoder_dim_ff=256,
            
            d_enc=256,
            encoder_dim_ff=512,
            encoder_num_layers=3,
        )
        
        # Load state dict
        state_dict = torch.load(path_to_vae_statedict, map_location=device)["state_dict"]
        
        print(f"loading model from {path_to_vae_statedict}")
        
        # Remove 'model.' prefix if present (from nn.DataParallel)
        new_state_dict = {}
        for key in state_dict.keys():
            if key.startswith("model."):
                new_key = key[6:]  # remove the 'model.' prefix
            else:
                new_key = key
            new_state_dict[new_key] = state_dict[key]
        
        model.load_state_dict(new_state_dict)
        model = model.to(device)
        model.eval()
        
        # Wrap in VAEModule
        vae = VAEModule(model).eval().to(device)
    
    def collate_selfies_fn(self, batch: List[torch.Tensor], vocab) -> torch.Tensor:
        """Collate function for SELFIES tokens"""
        return pad_sequence(batch, batch_first=True, padding_value=vocab['[pad]'])

    def forward_selfies(self, selfies: Union[str, List[str]], vae):
        """Convert SELFIES string(s) to latent vector(s)
        Also returns the loss
        """
        # Ensure input is a list
        if isinstance(selfies, str):
            selfies = [selfies]
        
        # Convert SELFIES to tokens using VAEModule's method
        tokens = []
        for s in selfies:
            token_tensor = vae.selfie_to_tokens(s).to(device)
            tokens.append(token_tensor)
        
        # Collate tokens
        tokens_batch = self.collate_selfies_fn(tokens, vae.vocab)
        
        with torch.no_grad():
            out = vae.model(tokens_batch)
            z = out["mu_ign"] + out["sigma_ign"] * torch.randn_like(out["sigma_ign"])
            loss = out["loss"]
        
        # Reshape to match expected output format
        return z.reshape(-1, vae.model.n_acc * vae.model.d_bnk), loss
