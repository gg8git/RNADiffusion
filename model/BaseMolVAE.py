import lightning.pytorch as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import kl_divergence, Normal
from hnn_utils import nn as HNN

from molformers.models.components import Embedding, causal_mask, TransformerDecoderLayer, TransformerEncoderLayer

MIN_STD = 1e-4

class BaseVAE(pl.LightningModule):
    def __init__(self,
        vocab: dict[str, int],
        n_acc: int = 2,
        d_enc: int = 128,
        d_dec: int = 128,
        d_bnk: int = 128,
        kl_factor: float = 0.1,
        encoder_nhead: int = 8,
        encoder_dim_ff: int = 512,
        encoder_dropout: float = 0.1,
        encoder_num_layers: int = 6,
        decoder_nhead: int = 8,
        decoder_dim_ff: int = 256,
        decoder_dropout: float = 0.1,
        decoder_num_layers: int = 6,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.d_enc = d_enc
        self.d_dec = d_dec
        self.d_bnk = d_bnk
        self.n_acc = n_acc

        self.vocab      = vocab
        self.vocab_size = len(self.vocab)

        self.kl_factor = kl_factor

        self.encoder_token_embedding = Embedding(n_tokens=self.vocab_size, d_embed=d_enc, dropout=encoder_dropout, padding_idx=self.pad_tok)
        self.decoder_token_embedding = Embedding(n_tokens=self.vocab_size, d_embed=d_dec, dropout=decoder_dropout, padding_idx=self.pad_tok)

        self.enc_neck = nn.Sequential(
            nn.Linear(d_enc, 4*d_bnk),
            nn.GELU(),
            nn.Linear(4*d_bnk, 2*d_bnk),
        )

        self.dec_neck = nn.Sequential(
            nn.Linear(d_bnk, 4*d_dec),
            nn.GELU(),
            nn.Linear(4*d_dec, d_dec),
        )

        self.dec_tok_deproj = nn.Linear(d_dec, self.vocab_size)

        self.encoder = HNN.TransformerEncoder(
            HNN.TransformerEncoderLayer(
                d_model=d_enc,
                nhead=encoder_nhead,
                dim_feedforward=encoder_dim_ff,
                dropout=encoder_dropout,
            )
            for _ in range(encoder_num_layers)
        )

        self.decoder = HNN.TransformerDecoder(
            HNN.TransformerDecoderLayer(
                d_model=d_dec,
                nhead=decoder_nhead,
                dim_feedforward=decoder_dim_ff,
                dropout=decoder_dropout,
            )
            for _ in range(decoder_num_layers)
        )

        self.add_acc_toks()

    def encode(self, tokens):
        # embed = self.encoder_token_embedding(tokens)
        embed = self.encoder_token_embedding.embedding(tokens)
        embed = torch.cat([self.acc_toks.expand(tokens.shape[0], self.n_acc, self.d_enc), embed], dim=1)

        pad_mask = tokens == self.pad_tok
        pad_mask = torch.cat([torch.full((tokens.shape[0], self.n_acc), False, dtype=torch.bool, device=tokens.device), pad_mask], dim=1)

        encoding = self.encoder(embed, src_pad_mask=pad_mask)[:, :self.n_acc]
        encoding = self.enc_neck(encoding)

        mu, sigma = encoding.chunk(2, dim=-1)
        sigma = F.softplus(sigma) + MIN_STD

        return mu, sigma

    def decode(self, z, tokens):
        z = self.dec_neck(z)
        # embed = self.decoder_token_embedding(tokens)
        embed = self.decoder_token_embedding.embedding(tokens)

        tgt_mask = causal_mask(embed.shape[1], embed.device, embed.dtype)
        # Sometimes we dont use a strictly causal mask
        decoding = self.decoder(tgt=embed, mem=z, tgt_mask=tgt_mask)
        logits = self.dec_tok_deproj(decoding)

        return logits

    def forward(self, tokens):
        mu, sigma = self.encode(tokens)
        z = mu + torch.randn_like(sigma)*sigma

        logits = self.decode(z, tokens)

        # Autoregressive shift
        logits = logits[:, :-1]
        tokens = tokens[:, 1:]

        recon_loss = F.cross_entropy(logits.permute(0, 2, 1), tokens, ignore_index=self.pad_tok)

        if self.global_step < 6250:
            kl_fac = min(self.global_step / 6250, 1.0) * self.kl_factor
        else:
            kl_fac = self.kl_factor

        kldiv = kl_divergence(
            Normal(mu, sigma),
            Normal(0, 1)
        )
        kldiv = kldiv.mean() + (kldiv.mean(dim=0) - kldiv.detach().mean()).abs().mean()

        loss = recon_loss + kl_fac * kldiv
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            mask = tokens != self.pad_tok

            token_acc = (preds[mask] == tokens[mask]).float().mean()
            string_acc = ((preds == tokens) | (tokens == self.pad_tok)).all(dim=1).float().mean()
            sigma_mean = sigma.mean()

        return dict(
            loss=loss,
            recon_loss=recon_loss,
            kldiv=kldiv,
            token_acc=token_acc,
            string_acc=string_acc,
            sigma_mean=sigma_mean,
            mu_ign=mu,
            sigma_ign=sigma,
            kl_factor=kl_fac
        )

    def add_acc_toks(self):
        self.register_parameter('acc_toks', nn.Parameter(torch.randn(1, self.n_acc, self.d_enc)))

    @property
    def start_tok(self):
        return self.vocab['[start]']

    @property
    def stop_tok(self):
        return self.vocab['[stop]']

    @property
    def pad_tok(self):
        return self.vocab['[pad]']
