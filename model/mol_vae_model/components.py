from math import log
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding


class SinePosEnc(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()

        self.dim_model = d_model
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.shape[1]
        pos = (
            torch.arange(0, seq_len, device=x.device, dtype=x.dtype)
            .unsqueeze(1)
            .repeat(1, self.dim_model)
        )
        dim = (
            torch.arange(0, self.dim_model, device=x.device, dtype=x.dtype)
            .unsqueeze(0)
            .repeat(seq_len, 1)
        )
        div = torch.exp(-log(10000) * (2 * (dim // 2) / self.dim_model))
        pos *= div
        pos[:, 0::2] = torch.sin(pos[:, 0::2])
        pos[:, 1::2] = torch.cos(pos[:, 1::2])

        output = x.unsqueeze(-1) if x.ndim == 2 else x

        output = output + pos.unsqueeze(0)
        return self.dropout(output)

class Embedding(nn.Module):
    def __init__(self, n_tokens: int, d_embed: int, dropout: float = 0.1, padding_idx: Optional[int] = None):
        super().__init__()

        self.embedding = nn.Embedding(n_tokens, d_embed, padding_idx=padding_idx)
        self.pos_enc = SinePosEnc(d_embed, dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.pos_enc(self.embedding(x))

def causal_mask(sz: int, device: torch.device, dtype: torch.dtype):
    return torch.triu(
        torch.full((sz, sz), float('-inf'), device=device, dtype=dtype),
        diagonal=1
    )


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1) -> None:
        super().__init__()

        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.ff_out = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
            self,
            src: Tensor,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: bool = False) -> Tensor:
        x = src
        x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal)
        x = x + self.ff_out(self.norm2(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, is_causal=is_causal)
        return self.dropout1(x)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # Implementation of Feedforward model
        self.ff_out = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = tgt
        x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
        x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
        x = x + self.ff_out(self.norm3(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
        )
        return self.dropout1(x)

    def _mha_block(self, x: Tensor, mem: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=False)[0]
        return self.dropout2(x)

class SwiGLU(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate)*x


class MultiheadAttention(nn.Module):
    def __init__(self, embed_size, heads, dropout=0.0):
        super().__init__()

        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        self.dropout = dropout
        self.batch_first=True

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.q_proj = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.k_proj = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.v_proj = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

        self.rotary_emb = RotaryEmbedding(
            dim=self.head_dim,
            use_xpos=True,
        )

    def forward(self, q, k, v, attn_mask=None, key_padding_mask=None, is_causal=False):
        N, sl = q.shape[0], q.shape[1]

        mask = None
        if attn_mask is not None:
            mask = attn_mask
        if key_padding_mask is not None:
            key_mask = key_padding_mask.view(N, 1, 1, sl).expand(N, self.heads, sl, sl)
            mask = mask + key_mask if mask is not None else key_mask
        
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        q = q.reshape(N, sl, self.heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(N, sl, self.heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(N, sl, self.heads, self.head_dim).permute(0, 2, 1, 3)

        q, k = self.rotary_emb.rotate_queries_and_keys(q, k)

        attn = F.scaled_dot_product_attention(q, k, v, mask, is_causal=is_causal, dropout_p=self.dropout)
        attn = attn.permute(0, 2, 1, 3).reshape(N, sl, self.heads * self.head_dim)
        return self.fc_out(attn)