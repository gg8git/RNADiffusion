import math
from dataclasses import dataclass

import torch
from botorch.acquisition import qExpectedImprovement, qLogExpectedImprovement
from botorch.generation import MaxPosteriorSampling
from botorch.optim import optimize_acqf
from torch.quasirandom import SobolEngine


@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = 32
    success_counter: int = 0
    success_tolerance: int = 10
    best_value: float = -float("inf")
    restart_triggered: bool = False

    # def __post_init__(self):
    #     self.failure_tolerance = math.ceil(
    #         max([4.0 / self.batch_size, float(self.dim ) / self.batch_size])
    #     )


def update_state(state, Y_next):
    if max(Y_next) > state.best_value + 1e-3 * math.fabs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1

    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0

    state.best_value = max(state.best_value, max(Y_next).item())
    if state.length < state.length_min:
        state.restart_triggered = True
    return state


def generate_batch(
    state,
    model,  # GP model
    X,  # Evaluated points on the domain [0, 1]^d
    Y,  # Function values
    batch_size,
    n_candidates=None,  # Number of candidates for Thompson sampling
    num_restarts=10,
    raw_samples=256,
    acqf="ts",  # "ei" or "ts" or "ddim"
    diffusion=None,
    dtype=torch.float32,
    device=torch.device("cuda"),
):
    assert torch.all(torch.isfinite(Y))
    if n_candidates is None:
        n_candidates = min(5000, max(2000, 200 * X.shape[-1]))

    x_center = X[Y.argmax(), :].clone()
    weights = torch.ones_like(x_center) * 8  # less than 4 stdevs on either side max
    tr_lb = x_center - weights * state.length / 2.0
    tr_ub = x_center + weights * state.length / 2.0

    if acqf == "ei":
        try:
            ei = qExpectedImprovement(model.cuda(), Y.max().cuda())
            X_next, _ = optimize_acqf(
                ei,
                bounds=torch.stack([tr_lb, tr_ub]).cuda(),
                q=batch_size,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
            )
        except:
            acqf = "ts"

    if acqf == "ts":
        dim = X.shape[-1]
        tr_lb = tr_lb.cuda()
        tr_ub = tr_ub.cuda()
        sobol = SobolEngine(dim, scramble=True)
        pert = sobol.draw(n_candidates).to(dtype=dtype).cuda()
        pert = tr_lb + (tr_ub - tr_lb) * pert
        tr_lb = tr_lb.cuda()
        tr_ub = tr_ub.cuda()
        # Create a perturbation mask
        prob_perturb = min(20.0 / dim, 1.0)
        mask = torch.rand(n_candidates, dim, dtype=dtype, device=device) <= prob_perturb
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1
        mask = mask.cuda()

        # Create candidate points from the perturbations and the mask
        X_cand = x_center.expand(n_candidates, dim).clone()
        X_cand = X_cand.cuda()
        X_cand[mask] = pert[mask]

        # Sample on the candidate points
        thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
        X_next = thompson_sampling(X_cand.cuda(), num_samples=batch_size)

    if acqf == "ddim":
        assert diffusion is not None

        log_ei_mod = qLogExpectedImprovement(
            model=model.cuda(),  # type: ignore
            best_f=Y.max().cuda(),
        )

        def cond_fn_log_ei(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            with torch.enable_grad():
                x = x.detach().requires_grad_(True)
                log_ei = log_ei_mod(x)
                if log_ei.dim() > 1:
                    log_ei = log_ei.sum(dim=tuple(range(1, log_ei.dim())))
                s = log_ei.sum()
                (grad_x,) = torch.autograd.grad(s, x, retain_graph=False, create_graph=False)

            return grad_x.detach()

        X_next = diffusion.ddim_sample(
            batch_size=batch_size,
            sampling_steps=50,
            guidance_scale=1.0,
            cond_fn=cond_fn_log_ei,
        )

    if acqf == "ddim_repaint":
        repaint_candidates = 128

        dim = X.shape[-1]
        tr_lb = tr_lb.cuda()
        tr_ub = tr_ub.cuda()
        sobol = SobolEngine(dim, scramble=True)
        pert = sobol.draw(repaint_candidates).to(dtype=dtype).cuda()
        pert = tr_lb + (tr_ub - tr_lb) * pert
        tr_lb = tr_lb.cuda()
        tr_ub = tr_ub.cuda()
        # Create a perturbation mask
        prob_perturb = min(20.0 / dim, 1.0)
        mask = torch.rand(repaint_candidates, dim, dtype=dtype, device=device) <= prob_perturb
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1
        mask = (~mask.cuda()).float()

        X_cand = x_center.expand(repaint_candidates, dim).clone()

        X_cand = diffusion.ddim_repaint(
            x_known=X_cand.cuda(),
            mask=mask,
            sampling_steps=50,
            u_steps=10,
        )
        thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
        X_next = thompson_sampling(X_cand.cuda(), num_samples=batch_size)

    return X_next
