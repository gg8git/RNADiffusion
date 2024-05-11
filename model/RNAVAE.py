from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import hnn_utils.nn as hnn
import hnn_utils.nn.functional as HNNF
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical, Normal


@dataclass
class Config:
    vocab: Dict[str, int]

    d_model: int = 256
    dim_ff: int = 512
    n_head: int = 8
    n_layers: int = 3
    n_bn: int = 8
    zdim: int = 16

    beta: float = 0.1
    lr: float = 1e-4
    wd: float = 0.0


class RNAVAE(L.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()

        config = get_config(*args, **kwargs)

        self.save_hyperparameters()

        self.config = config

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        self.register_buffer(
            "kl_coeff",
            kl_balancer_coeff(config.n_bn, [config.zdim] * config.n_bn),
        )

        self.reset_parameters()

    def forward(self, tokens: Tensor) -> Tuple[Tensor, Dict]:
        post: Normal = self.encoder(tokens)
        all_kl = HNNF.gaussian_kl_standard_normal(post.loc, post.scale)
        all_kl = all_kl * self.calc_kl_weights(all_kl)

        z = post.rsample()

        logits = self.decoder(tokens, z)

        nll = cross_entropy(logits[:, :-1], tokens[:, 1:], self.pad_idx)
        kl = all_kl.mean(dim=(1, 2))

        stats = compute_stats(tokens, logits, post, self.pad_idx)

        elbo = nll + kl
        elbo = elbo.mean()

        return elbo, stats

    @torch.no_grad()
    def sample(
        self, z: torch.Tensor, argmax: bool = False, p: float = 1.0
    ) -> List[Union[str, None]]:
        tr = self.training
        self.eval()

        if z.ndim == 1:
            z = z.unsqueeze(0)

        N = z.shape[0]
        z = z.reshape(N, -1, self.config.zdim).to(self.device, dtype=self.dtype)

        tokens = torch.full(
            (N, 1), self.start_idx, dtype=torch.long, device=self.device
        )
        while True:
            logits = self.decoder(tokens, z)[:, -1:]
            logits = logits / p  # Temperature scaling

            dist = Categorical(logits=logits)

            if argmax:
                sampled = dist.mode
            else:
                sampled = dist.sample()

            tokens = torch.cat([tokens, sampled], dim=-1)

            if (
                torch.any(tokens == self.stop_idx, dim=-1).all()
                or tokens.shape[1] > 128
            ):
                break

        self.train(tr)
        return tokens

    @torch.no_grad()
    def calc_kl_weights(self, all_kl: torch.Tensor) -> torch.Tensor:
        sample_shape = all_kl.shape[-2:]

        all_kl = all_kl.view(-1, *sample_shape)

        kl_coeff_i = batch_nvae_kl_weights(all_kl, self.kl_coeff, self.config.beta)

        return kl_coeff_i.view(-1, *sample_shape)

    def reset_parameters(self):
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.xavier_uniform_(mod.weight)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)
            elif isinstance(mod, nn.Embedding):
                nn.init.normal_(mod.weight, std=self.config.d_model**-0.5)
                mod._fill_padding_idx_with_zero()

    @property
    def pad_idx(self):
        return self.config.vocab["[PAD]"]

    @property
    def start_idx(self):
        return self.config.vocab["[START]"]

    @property
    def stop_idx(self):
        return self.config.vocab["[STOP]"]
    
    @property
    def unk_idx(self):
        return self.config.vocab["[UNK]"]


class Encoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(
            len(config.vocab), config.d_model, padding_idx=config.vocab["[PAD]"]
        )

        self.blocks = hnn.TransformerEncoder(
            hnn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.n_head,
                dim_feedforward=config.dim_ff,
                dropout=0.1,
            )
            for _ in range(config.n_layers)
        )

        self.enc_neck = nn.Linear(config.d_model, 2 * config.zdim)

        acc = torch.zeros(config.n_bn, config.d_model)
        acc = nn.init.orthogonal_(acc).unsqueeze(0)
        self.register_parameter("acc", nn.Parameter(acc))

    def forward(self, tokens: Tensor):
        emb = torch.cat(
            (
                self.acc.expand(tokens.size(0), -1, -1),
                self.embed(tokens),
            ),
            dim=1,
        )

        pad_mask = tokens == self.config.vocab["[PAD]"]
        pad_mask = F.pad(pad_mask, (self.config.n_bn, 0), value=False)

        hidden = self.blocks(emb, src_pad_mask=pad_mask)[:, : self.config.n_bn]
        loc, log_scale = self.enc_neck(hidden).chunk(2, dim=-1)

        posterior = Normal(loc, exp_lin(log_scale))
        return posterior


class Decoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.embed = nn.Embedding(
            len(config.vocab), config.d_model, padding_idx=config.vocab["[PAD]"]
        )

        self.blocks = hnn.TransformerDecoder(
            hnn.TransformerDecoderLayer(
                d_model=config.d_model,
                nhead=config.n_head,
                dim_feedforward=config.dim_ff,
                dropout=0.1,
            )
            for _ in range(config.n_layers)
        )

        self.dec_neck = nn.Linear(config.zdim, config.d_model)

        self.logit_proj = nn.Linear(config.d_model, len(config.vocab))

    def forward(self, tokens: Tensor, z: Tensor) -> Tensor:
        z = self.dec_neck(z)

        emb = self.embed(tokens)

        hidden = self.blocks(emb, mem=z, tgt_is_causal=True)

        return self.logit_proj(hidden)


def exp_lin(x: Tensor) -> Tensor:
    xpos = torch.clamp_min(x, 0.0)
    xneg = torch.clamp_max(x, 0.0)

    return torch.where(
        x <= 0,
        torch.exp(xneg),
        xpos + 1,
    )


@torch.no_grad()
def compute_stats(
    tokens: Tensor,
    logits: Tensor,
    post: Normal,
    pad_idx: int,
) -> Tuple[Tensor, Dict]:
    ltok = tokens[:, 1:]
    llog = logits[:, :-1]

    nll = cross_entropy(llog, ltok, pad_idx).mean()
    kl = HNNF.gaussian_kl_standard_normal(post.loc, post.scale).mean(dim=0)

    preds = llog.argmax(dim=-1)
    hits = preds == ltok
    tok_acc = hits[ltok != pad_idx].float().mean()
    string_acc = (hits | (ltok == pad_idx)).all(dim=-1).float().mean()

    kl_min = kl.min()
    kl_max = kl.max()
    kl_mean = kl.mean()

    stats = {
        "nll": nll,
        "kl_min": kl_min,
        "kl_max": kl_max,
        "kl_mean": kl_mean,
        "tok_acc": tok_acc,
        "string_acc": string_acc,
    }

    return stats


def kl_balancer_coeff(
    num_scales: int,
    groups_per_scale: List[int],
) -> torch.Tensor:
    """
    NVAE KL balancing coefficients for each level of the hierarchy.
    as the index of z increases the KL is regularized more.
    """
    groups_per_scale = torch.tensor(groups_per_scale)

    exponents = torch.ones(num_scales)
    repeated_groups = groups_per_scale.repeat_interleave(groups_per_scale)
    coeff = exponents.repeat_interleave(groups_per_scale) / repeated_groups
    coeff = coeff / coeff.min()
    return coeff.float()


@torch.no_grad()
def batch_nvae_kl_weights(
    all_kl: torch.Tensor,
    alpha_i: torch.Tensor,
    kl_coeff: float = 1.0,
) -> torch.Tensor:
    """
    NVAE KL balancing coefficients for each level of the hierarchy.
    as the index of z increases the KL is regularized more.
    """
    all_kl = all_kl.flatten(1)
    kl_coeff_i = all_kl.mean(dim=0, keepdim=True)
    total_kl = kl_coeff_i.sum()

    kl_coeff_i = kl_coeff_i / alpha_i * total_kl
    kl_coeff_i = kl_coeff_i / kl_coeff_i.mean(dim=1, keepdim=True)

    return kl_coeff * kl_coeff_i


def cross_entropy(
    logits: Tensor,
    targets: Tensor,
    pad_idx: int,
):
    nll = F.cross_entropy(
        logits.permute(0, 2, 1), targets, ignore_index=pad_idx, reduction="none"
    )
    nll = nll.sum(dim=-1) / (targets != pad_idx).sum(dim=-1)
    return nll


def get_config(*args, **kwargs) -> Config:
    if len(args) == 1 and isinstance(args[0], Config):
        return args[0]
    if "config" in kwargs:
        return kwargs["config"