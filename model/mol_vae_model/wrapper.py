import json
from typing import List, Union
import selfies as sf

import lightning as L
import torch
from torch.nn.utils.rnn import pad_sequence

from model.mol_vae_model.BaseMolVAE import BaseVAE
from model.mol_vae_model.MolVAE import VAEModule

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class VAEWrapper(L.LightningModule):
    def __init__(self, path_to_vae_statedict, vocab_path="data/selfies_vocab.json"):
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
    
    def collate_selfies_fn(self, batch: List[torch.Tensor], vocab) -> torch.Tensor:
        """Collate function for SELFIES tokens"""
        return pad_sequence(batch, batch_first=True, padding_value=vocab['[pad]'])

    def selfies_to_latent_params(self, selfies: Union[str, List[str]]):
        """Convert SELFIES string(s) to mu and sigma
        """
        # Ensure input is a list
        if isinstance(selfies, str):
            selfies = [selfies]
        
        # Convert SELFIES to tokens using VAEModule's method
        tokens = []
        for s in selfies:
            token_tensor = self.vae.selfie_to_tokens(s).to(device)
            tokens.append(token_tensor)
        
        # # Collate tokens
        tokens_batch = self.collate_selfies_fn(tokens, self.vae.vocab)
        
        with torch.no_grad():
            out = self.vae.model(tokens_batch)
        
        return out["mu_ign"], out["sigma_ign"]
    
    def forward_selfies(self, selfies: Union[str, List[str]]):
        """Convert SELFIES string(s) to latent vector(s)
        Also returns the loss
        """
        # Ensure input is a list
        if isinstance(selfies, str):
            selfies = [selfies]
        
        # Convert SELFIES to tokens using VAEModule's method
        tokens = []
        for s in selfies:
            token_tensor = self.vae.selfie_to_tokens(s).to(device)
            tokens.append(token_tensor)
        
        # Collate tokens
        tokens_batch = self.collate_selfies_fn(tokens, self.vae.vocab)
        
        with torch.no_grad():
            out = self.vae.model(tokens_batch)
            z = out["mu_ign"] + out["sigma_ign"] * torch.randn_like(out["sigma_ign"])
            loss = out["loss"]
        
        # Reshape to match expected output format
        return z, loss

    def latent_to_selfies_batch(self, z: torch.Tensor, argmax=True, max_len=256):
        """Convert batch of latent vectors to SELFIES strings"""
        z = z.to(device)
        
        with torch.no_grad():
            # Use VAEModule's sample method to generate tokens
            tokens = self.vae.sample(
                z.reshape(-1, self.vae.model.n_acc * self.vae.model.d_bnk),
                argmax=argmax,
                max_len=max_len
            )
        
        # Convert tokens to SELFIES strings
        selfies_list = []
        for token_seq in tokens:
            selfie = self.vae.tokens_to_selfie(token_seq, drop_after_stop=True)
            selfies_list.append(selfie)
        
        return selfies_list

    def latent_to_selfies(self, z: torch.Tensor, argmax=True, max_len=256):
        """Convert latent vector(s) to SELFIES string(s)
        Wrapper around latent_to_selfies_batch for consistency
        """
        z = z.to(device)
        results = self.latent_to_selfies_batch(z, argmax=argmax, max_len=max_len)
        return results

    def sample_selfies_from_prior(self, batch_size=32):
        z = torch.randn(batch_size, self.vae.model.n_acc, self.vae.model.d_bnk)
        return self.latent_to_selfies(z)