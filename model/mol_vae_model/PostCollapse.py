import torch
from torch.distributions import Normal, kl_divergence
from torchmetrics.aggregation import Metric, dim_zero_cat


class KLCalc(Metric):
    full_state_update = True

    def __init__(self, dead_cutoff=0.01):
        super().__init__()

        self.dead_cutoff = dead_cutoff

        self.add_state("means", default=[], dist_reduce_fx="cat")
        self.add_state("stds", default=[], dist_reduce_fx="cat")

    def update(self, mu: torch.Tensor, sigma: torch.Tensor) -> None:
        self.means.append(mu.flatten(1).float())
        self.stds.append(sigma.flatten(1).float())

    def compute(self) -> torch.Tensor:
        mu, sigma = self.get_latents()

        mkl = kl_divergence(Normal(mu, sigma), Normal(0, 1)).mean(dim=0)

        alive = mkl[mkl > self.dead_cutoff]

        n_alive = float(alive.numel())
        mean = alive.mean()
        min_ = alive.min()
        max_ = alive.max()

        mean_sigma = sigma.mean(dim=0)[mkl > self.dead_cutoff].mean()

        return n_alive, mean, min_, max_, mean_sigma

    def get_latents(self) -> torch.Tensor:
        means = comb(self.means)
        stds = comb(self.stds)
        return means, stds


def comb(val) -> torch.Tensor:
    if isinstance(val, list) and val:
        return dim_zero_cat(val)
    return val
