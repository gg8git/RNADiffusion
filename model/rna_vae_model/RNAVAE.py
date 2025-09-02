from dataclasses import dataclass

import hnn_utils.nn as hnn
import hnn_utils.nn.functional as HNNF
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Normal


@dataclass
class Config:
    vocab: dict[str, int]

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

    def forward(self, tokens: Tensor) -> tuple[Tensor, dict]:
        # Pad with 4 stop tokens
        tokens = F.pad(tokens, (0, 4), value=self.stop_idx)

        post: Normal = self.encoder(tokens)
        all_kl = HNNF.gaussian_kl_standard_normal(post.loc, post.scale)
        all_kl = all_kl * self.calc_kl_weights(all_kl)

        z = post.rsample()

        logits = self.decoder(z)

        nll = cross_entropy(logits, tokens)
        kl = all_kl.mean(dim=(1, 2))

        stats = compute_stats(tokens, logits, post)

        elbo = nll + kl
        elbo = elbo.mean()

        return elbo, stats

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
            elif isinstance(mod, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.xavier_uniform_(mod.weight)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)

    @property
    def start_idx(self):
        return self.config.vocab["[START]"]

    @property
    def stop_idx(self):
        return self.config.vocab["[STOP]"]

    @property
    def unk_idx(self):
        return self.config.vocab["[UNK]"]


class DownConv1DBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=5, padding=2, stride=2),  # 2x downsampling
            nn.SiLU(),
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.transpose(1, 2)
        x = self.block(x)
        return x.transpose(1, 2)


class UpConv1DBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose1d(dim, dim, kernel_size=5, padding=2, stride=2, output_padding=1),
            nn.SiLU(),
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.transpose(1, 2)
        x = self.block(x)
        return x.transpose(1, 2)


class Encoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(len(config.vocab), config.d_model)

        # 400
        self.block1 = hnn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_head,
            dim_feedforward=config.dim_ff,
            dropout=0.0,
        )
        self.conv1 = DownConv1DBlock(config.d_model)

        # 200
        self.block2 = hnn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_head,
            dim_feedforward=config.dim_ff,
            dropout=0.0,
        )
        self.conv2 = DownConv1DBlock(config.d_model)

        # 100
        self.block3 = hnn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_head,
            dim_feedforward=config.dim_ff,
            dropout=0.0,
        )
        self.conv3 = DownConv1DBlock(config.d_model)

        # 50
        self.block4 = hnn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_head,
            dim_feedforward=config.dim_ff,
            dropout=0.0,
        )
        self.conv4 = DownConv1DBlock(config.d_model)

        self.conv_last = nn.Sequential(
            nn.Conv1d(config.d_model, config.d_model, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(config.d_model, config.d_model, kernel_size=1),
        )

        self.enc_neck = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.Tanh(),
            nn.Linear(config.d_model, 2 * config.zdim),
        )

    def forward(self, tokens: Tensor):
        emb = self.embed(tokens)

        emb = self.block1(emb)
        emb = self.conv1(emb)

        emb = self.block2(emb)
        emb = self.conv2(emb)

        emb = self.block3(emb)
        emb = self.conv3(emb)

        emb = self.block4(emb)
        emb = self.conv4(emb)

        # Interpolate to 16
        emb = F.interpolate(emb.transpose(1, 2), size=16, mode="linear")
        emb = self.conv_last(emb).transpose(1, 2)

        loc, log_scale = self.enc_neck(emb).chunk(2, dim=-1)

        posterior = Normal(loc, exp_lin(log_scale))
        return posterior


class Decoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.base = nn.Parameter(torch.randn(1, 25, config.d_model))
        self.z_proj = nn.Linear(config.zdim, config.d_model)

        # 25
        self.block1 = hnn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.n_head,
            dim_feedforward=config.dim_ff,
            dropout=0.0,
        )
        self.conv1 = UpConv1DBlock(config.d_model)

        # 50
        self.block2 = hnn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.n_head,
            dim_feedforward=config.dim_ff,
            dropout=0.0,
        )
        self.conv2 = UpConv1DBlock(config.d_model)

        # 100
        self.block3 = hnn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.n_head,
            dim_feedforward=config.dim_ff,
            dropout=0.0,
        )
        self.conv3 = UpConv1DBlock(config.d_model)

        # 200
        self.block4 = hnn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.n_head,
            dim_feedforward=config.dim_ff,
            dropout=0.0,
        )
        self.conv4 = UpConv1DBlock(config.d_model)

        self.conv_last = nn.Sequential(
            nn.Conv1d(config.d_model, config.d_model, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(config.d_model, config.d_model, kernel_size=1),
        )

        self.logit_proj = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, len(config.vocab)),
        )

    def forward(self, z: Tensor) -> Tensor:
        z = self.z_proj(z)

        hidden = self.base.expand(z.size(0), -1, -1)

        hidden = self.block1(hidden, z)
        hidden = self.conv1(hidden)

        hidden = self.block2(hidden, z)
        hidden = self.conv2(hidden)

        hidden = self.block3(hidden, z)
        hidden = self.conv3(hidden)

        hidden = self.block4(hidden, z)
        hidden = self.conv4(hidden)

        hidden = self.conv_last(hidden.transpose(1, 2)).transpose(1, 2)

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
) -> tuple[Tensor, dict]:
    nll = cross_entropy(logits, tokens).mean()
    kl = HNNF.gaussian_kl_standard_normal(post.loc, post.scale).mean(dim=0)

    preds = logits.argmax(dim=-1)
    hits = preds == tokens
    tok_acc = hits.float().mean()
    string_acc = (hits).all(dim=-1).float().mean()

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
    groups_per_scale: list[int],
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
):
    nll = F.cross_entropy(logits.permute(0, 2, 1), targets, reduction="none")
    nll = nll.mean(dim=-1)
    return nll


def get_config(*args, **kwargs) -> Config:
    if len(args) == 1 and isinstance(args[0], Config):
        return args[0]
    if "config" in kwargs:
        return kwargs["config"]
    # Otherwise construct a new config object
    return Config(*args, **kwargs)
