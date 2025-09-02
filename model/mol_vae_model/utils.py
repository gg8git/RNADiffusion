import torch


def diag_gauss_kl(mu, sigma):
    sigma2 = sigma.square()
    return 0.5 * sigma2 + mu.square() - 1.0 - torch.log(sigma2)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
