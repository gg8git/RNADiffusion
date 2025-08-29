import json
from typing import List, Union
import selfies as sf

import lightning as L
import torch
from torch.nn.utils.rnn import pad_sequence

from model.mol_vae_model.BaseMolVAE import BaseVAE
from model.mol_vae_model.MolVAE import VAEModule

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class VAEFlatWrapper(L.LightningModule):
    def __init__(self, path_to_vae_statedict, vocab_path="data/selfies/selfies_vocab.json"):
        """Load a VAE model for SELFIES representation"""

        super().__init__()

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
        self.vae = VAEModule(model).eval().to(device)

    def selfies_to_latent_params(self, selfies: Union[str, List[str]]):
        """Convert SELFIES string(s) to mu and sigma
        """
        _, _, out = self.vae.forward_selfies(selfies)
        return out["mu_ign"].flatten(1), out["sigma_ign"].flatten(1)
    
    def forward_selfies(self, selfies: Union[str, List[str]]):
        """Convert SELFIES string(s) to latent vector(s)
        Also returns the loss
        """
        z, loss, out = self.vae.forward_selfies(selfies)
        return z.flatten(1), loss, out

    def latent_to_selfies_batch(self, z: torch.Tensor, argmax=True, max_len=256):
        """Convert batch of latent vectors to SELFIES strings"""
        z = z.reshape(-1, self.vae.model.n_acc, self.vae.model.d_bnk)
        return self.vae.latent_to_selfies_batch(z, argmax=argmax, max_len=max_len)

    def latent_to_selfies(self, z: torch.Tensor, argmax=True, max_len=256):
        """Convert latent vector(s) to SELFIES string(s)
        Wrapper around latent_to_selfies_batch for consistency
        """
        z = z.reshape(-1, self.vae.model.n_acc, self.vae.model.d_bnk)
        results = self.latent_to_selfies_batch(z, argmax=argmax, max_len=max_len)
        return results

    def sample_selfies_from_prior(self, batch_size=32):
        selfies, z = self.vae.sample_selfies_from_prior(batch_size)
        return selfies, z.flatten(1)