from math import log
from typing import Any, Literal

import lightning.pytorch as pl
import selfies as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch.distributions import Categorical, Normal, kl_divergence
from torch.nn.utils.rnn import pad_sequence

MIN_STD = 1e-4

ModelType = Literal["molecule", "peptide"]


class BaseVAE(pl.LightningModule):
    def __init__(
        self,
        vocab: dict[str, int],
        model_type: ModelType,
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

        self.vocab = vocab
        self.vocab_size = len(self.vocab)

        self.kl_factor = kl_factor

        self.encoder_token_embedding = Embedding(
            n_tokens=self.vocab_size, d_embed=d_enc, dropout=encoder_dropout, padding_idx=self.pad_tok
        )
        self.decoder_token_embedding = Embedding(
            n_tokens=self.vocab_size, d_embed=d_dec, dropout=decoder_dropout, padding_idx=self.pad_tok
        )

        self.enc_neck = nn.Sequential(
            nn.Linear(d_enc, 4 * d_bnk),
            nn.GELU(),
            nn.Linear(4 * d_bnk, 2 * d_bnk),
        )

        self.dec_neck = nn.Sequential(
            nn.Linear(d_bnk, 4 * d_dec),
            nn.GELU(),
            nn.Linear(4 * d_dec, d_dec),
        )

        self.dec_tok_deproj = nn.Linear(d_dec, self.vocab_size)

        self.encoder = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=d_enc,
                nhead=encoder_nhead,
                dim_feedforward=encoder_dim_ff,
                dropout=encoder_dropout,
            )
            for _ in range(encoder_num_layers)  # type: ignore
        )

        self.decoder = TransformerDecoder(
            TransformerDecoderLayer(
                d_model=d_dec,
                nhead=decoder_nhead,
                dim_feedforward=decoder_dim_ff,
                dropout=decoder_dropout,
            )
            for _ in range(decoder_num_layers)  # type: ignore
        )

        self.acc_toks = nn.Parameter(torch.randn(1, self.n_acc, self.d_enc))

        self.model_type = model_type

    def encode(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        embed = self.encoder_token_embedding.embedding(tokens)
        embed = torch.cat([self.acc_toks.expand(tokens.shape[0], self.n_acc, self.d_enc), embed], dim=1)

        pad_mask = tokens != self.pad_tok
        pad_mask = F.pad(pad_mask, (self.n_acc, 0), mode="constant", value=True)

        encoding = self.encoder(embed, src_pad_mask=pad_mask)[:, : self.n_acc]
        encoding = self.enc_neck(encoding)

        mu, sigma = encoding.chunk(2, dim=-1)
        sigma = F.softplus(sigma) + MIN_STD

        return mu, sigma

    def decode(self, z: Tensor, tokens: Tensor) -> Tensor:
        z = self.dec_neck(z)
        embed = self.decoder_token_embedding.embedding(tokens)

        decoding = self.decoder(tgt=embed, mem=z)
        logits = self.dec_tok_deproj(decoding)

        return logits

    def forward(self, tokens: Tensor) -> dict[str, Any]:
        mu, sigma = self.encode(tokens)
        z = mu + torch.randn_like(sigma) * sigma

        logits = self.decode(z, tokens)

        # Autoregressive shift
        logits = logits[:, :-1]
        tokens = tokens[:, 1:]

        recon_loss = F.cross_entropy(logits.permute(0, 2, 1), tokens, ignore_index=self.pad_tok)

        if self.global_step < 6250:
            kl_fac = min(self.global_step / 6250, 1.0) * self.kl_factor
        else:
            kl_fac = self.kl_factor

        kldiv = kl_divergence(Normal(mu, sigma), Normal(0, 1))
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
            z=z,
            recon_loss=recon_loss,
            kldiv=kldiv,
            token_acc=token_acc,
            string_acc=string_acc,
            sigma_mean=sigma_mean,
            mu_ign=mu,
            sigma_ign=sigma,
            kl_factor=kl_fac,
        )

    @torch.inference_mode()
    def sample(self, z: torch.Tensor, argmax: bool = True, max_len: int = 256) -> torch.Tensor:
        is_tr = self.training
        self.eval()

        z = z.reshape(z.shape[0], self.n_acc, self.d_bnk).to(self.device)

        tokens = torch.full((z.shape[0], 1), fill_value=self.start_tok, dtype=torch.long, device=self.device)
        while True:  # Loop until every molecule hits a stop token
            logits = self.decode(z, tokens)[:, -1:]
            if argmax:
                sample = logits.argmax(dim=-1)
            else:
                sample = Categorical(logits=logits).sample()

            tokens = torch.cat([tokens, sample], dim=-1)

            if (tokens == self.stop_tok).any(dim=-1).all() or tokens.shape[1] > max_len:
                break

        self.train(is_tr)

        return tokens[:, 1:]  # Cut out start token

    @property
    def start_tok(self):
        return self.vocab["[start]"]

    @property
    def stop_tok(self):
        return self.vocab["[stop]"]

    @property
    def pad_tok(self):
        return self.vocab["[pad]"]

    def detokenize(self, tokens: Tensor) -> list[str]:
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)
        elif tokens.ndim > 2:
            raise ValueError("Tokens should be 1D or 2D tensor")

        strings = []
        rev_vocab = {v: k for k, v in self.vocab.items()}
        for row in tokens:
            row = row.tolist()

            if self.stop_tok in row:
                row = row[: row.index(self.stop_tok)]
            if self.pad_tok in row:
                row = row[: row.index(self.pad_tok)]

            toks = [rev_vocab[i] for i in row]

            if self.model_type == "peptide":
                dec = _decode_peptide_tokens(toks)
            elif self.model_type == "molecule":
                dec = _decode_selfies_tokens(toks)
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")

            strings.append(dec)

        return strings

    def tokenize(self, strings: list[str] | str):
        if isinstance(strings, str):
            strings = [strings]

        if self.model_type == "peptide":
            tokens = [_tokenize_peptide(s) for s in strings]
        elif self.model_type == "molecule":
            tokens = [_tokenize_smiles(s) for s in strings]
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Original code: 
        # enc = [[self.vocab[s] for s in s_tok] for s_tok in tokens]
        # enc = [[self.start_tok] + e + [self.stop_tok] for e in enc]
        # Updated to handle KeyErrors (e.g., KeyError: '[=NH1]') when doing self.vocab[s]. 
        enc = []
        for s_tok in tokens:
            encoded_s_tok = [self.start_tok]
            for s in s_tok:
                try:
                    encoded_s_tok.append(self.vocab[s])
                except KeyError:
                    # if selfies token s is not in vocab, skip token 
                    pass 
            encoded_s_tok.append(self.stop_tok)
            enc.append(encoded_s_tok)
        
        enc = [torch.tensor(e) for e in enc]
        enc = pad_sequence(enc, batch_first=True, padding_value=self.pad_tok).to(device=self.device, dtype=torch.long)

        return enc


class SinePosEnc(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()

        self.dim_model = d_model
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.shape[1]
        pos = torch.arange(0, seq_len, device=x.device, dtype=x.dtype).unsqueeze(1).repeat(1, self.dim_model)
        dim = torch.arange(0, self.dim_model, device=x.device, dtype=x.dtype).unsqueeze(0).repeat(seq_len, 1)
        div = torch.exp(-log(10000) * (2 * (dim // 2) / self.dim_model))
        pos *= div
        pos[:, 0::2] = torch.sin(pos[:, 0::2])
        pos[:, 1::2] = torch.cos(pos[:, 1::2])

        output = x.unsqueeze(-1) if x.ndim == 2 else x

        output = output + pos.unsqueeze(0)
        return self.dropout(output)


class Embedding(nn.Module):
    def __init__(self, n_tokens: int, d_embed: int, dropout: float = 0.1, padding_idx: int | None = None):
        super().__init__()

        self.embedding = nn.Embedding(n_tokens, d_embed, padding_idx=padding_idx)
        self.pos_enc = SinePosEnc(d_embed, dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.pos_enc(self.embedding(x))


def _decode_selfies_tokens(tokens: list[str]):
    selfies = "".join(tokens)
    smiles = sf.decoder(selfies)

    return smiles


def _decode_peptide_tokens(tokens: list[str]):
    peptide = "".join(tokens)
    return peptide


def _tokenize_smiles(smiles: str) -> list[str]:
    tokens = sf.encoder(smiles)
    tokens = list(sf.split_selfies(tokens))
    return tokens


def _tokenize_peptide(peptide: str) -> list[str]:
    tokens = list(peptide)
    return tokens


class TransformerEncoder(nn.Module):
    def __init__(self, *encoder_layers: nn.Module) -> None:
        super().__init__()

        self.layers = nn.ModuleList(list(*encoder_layers))

    def forward(self, src: Tensor, src_pad_mask: Tensor) -> Tensor:
        output = src

        for layer in self.layers:
            output = layer(output, src_pad_mask)

        return output


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        *decoder_layers: nn.Module,
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList(list(*decoder_layers))

    def forward(self, tgt: Tensor, mem: Tensor) -> Tensor:
        output = tgt

        for layer in self.layers:
            output = layer(output, mem)

        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.self_attn = SelfAttention(d_model, nhead, dropout=dropout)

        self.ff_block = FFNSwiGLU(d_model, dim_feedforward)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src: Tensor, src_pad_mask: Tensor) -> Tensor:
        x = src
        x = x + self._sa_block(self.norm1(x), src_pad_mask)
        x = x + self.ff_block(self.norm2(x))

        return x

    def _sa_block(self, x: Tensor, pad_mask: Tensor) -> Tensor:
        x = self.self_attn(x, mask=pad_mask, is_causal=False)
        return self.dropout(x)


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.self_attn = SelfAttention(d_model, nhead, dropout=dropout)
        self.cross_attn = CrossAttention(d_model, nhead, dropout=dropout)

        self.ff_block = FFNSwiGLU(d_model, dim_feedforward)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, tgt: Tensor, mem: Tensor) -> Tensor:
        x = tgt
        x = self.norm1(x + self._sa_block(x))
        x = self.norm2(x + self._mha_block(x, mem))
        x = self.norm3(x + self.ff_block(x))
        return x

    def _sa_block(self, x: Tensor) -> Tensor:
        x = self.self_attn(x, is_causal=True)
        return self.dropout1(x)

    def _mha_block(self, x: Tensor, mem: Tensor) -> Tensor:
        x = self.cross_attn(x, mem)
        return self.dropout2(x)


class CrossAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.embed_size = embed_dim
        self.num_heads = heads
        self.head_dim = embed_dim // heads
        self.dropout = dropout

        assert self.head_dim * heads == embed_dim, "Embedding size needs to be divisible by heads"

        self.q_proj = nn.Linear(self.embed_size, self.embed_size)
        self.kv_proj = nn.Linear(self.embed_size, self.embed_size * 2)

        self.out_proj = nn.Linear(heads * self.head_dim, embed_dim)

    def forward(self, query: Tensor, kv: Tensor) -> Tensor:
        q = self.q_proj(query)
        k, v = self.kv_proj(kv).chunk(2, dim=-1)

        q = rearrange(q, "... n (h d) -> ... h n d", h=self.num_heads)
        k = rearrange(k, "... n (h d) -> ... h n d", h=self.num_heads)
        v = rearrange(v, "... n (h d) -> ... h n d", h=self.num_heads)

        dropout = self.dropout if self.training else 0.0
        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout)

        attn = rearrange(attn, "... h n d -> ... n (h d)")
        return self.out_proj(attn)


class SelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.embed_size = embed_dim
        self.num_heads = heads
        self.head_dim = embed_dim // heads
        self.dropout = dropout

        assert self.head_dim * heads == embed_dim, "Embedding size needs to be divisible by heads"

        self.qkv_proj = nn.Linear(self.embed_size, self.embed_size * 3)

        self.out_proj = nn.Linear(heads * self.head_dim, embed_dim)

    def forward(self, x: Tensor, mask: Tensor | None = None, is_causal: bool = False) -> Tensor:
        assert not is_causal or mask is None, "is_causal not supported with padding or attention masks."

        if mask is not None and mask.ndim == 2:
            mask = rearrange(mask, "b n -> b () () n")

        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)

        q = rearrange(q, "... n (h d) -> ... h n d", h=self.num_heads)
        k = rearrange(k, "... n (h d) -> ... h n d", h=self.num_heads)
        v = rearrange(v, "... n (h d) -> ... h n d", h=self.num_heads)

        q, k = apply_rotary_emb(q, k)

        dropout = self.dropout if self.training else 0.0
        attn = F.scaled_dot_product_attention(q, k, v, mask, dropout_p=dropout, is_causal=is_causal)

        attn = rearrange(attn, "... h n d -> ... n (h d)")
        return self.out_proj(attn)


class FFNSwiGLU(nn.Module):
    """
    GLU Variants Improve Transformer
    https://arxiv.org/abs/2002.05202
    """

    def __init__(
        self,
        dim: int,
        dim_feedforward: int,
        out_dim: int | None = None,
        use_bias: bool = True,
    ):
        super().__init__()

        out_dim = out_dim or dim

        self.ff1 = nn.Linear(dim, dim_feedforward * 2, bias=use_bias)
        self.ff2 = nn.Linear(dim_feedforward, out_dim, bias=use_bias)

    def forward(self, x: Tensor) -> Tensor:
        y, gate = self.ff1(x).chunk(2, dim=-1)
        x = y * F.silu(gate)
        return self.ff2(x)


class LayerNorm(nn.Module):
    def __init__(self, features: int):
        super().__init__()

        self.ln = nn.LayerNorm(features)

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype

        x = self.ln(x)
        return x.type(dtype)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    SL, D = xq.shape[-2:]

    freqs_cis = precompute_freqs_cis(
        dim=D,
        sl=SL,
        device=xq.device,
    ).reshape(1, 1, SL, D // 2)

    xq_ = torch.view_as_complex(rearrange(xq.float(), "b h sl (d n) -> b h sl d n", n=2))
    xk_ = torch.view_as_complex(rearrange(xk.float(), "b h sl (d n) -> b h sl d n", n=2))

    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def precompute_freqs_cis(
    dim: int,
    sl: int,
    device: torch.device,
    theta: float = 10000.0,
) -> Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device)[: (dim // 2)].float() / dim))
    t = torch.arange(sl, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis
