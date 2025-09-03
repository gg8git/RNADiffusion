import gpytorch
import polars as pl
import torch
import torch.nn.functional as F
from botorch.acquisition import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import PredictiveLogLikelihood
from rdkit import RDLogger
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from model import GPModelDKL
from model.diffusion_v2 import DiffusionModel

torch.set_float32_matmul_precision("highest")

torch.set_printoptions(sci_mode=False, precision=3, linewidth=120)
RDLogger.DisableLog("rdApp.*")  # type: ignore


######################
### Bounds
######################

# set up
model = DiffusionModel.load_from_checkpoint("./data/molecule_diffusion.ckpt")
model.cuda()
model.freeze()

shape = (model.vae.n_acc * model.vae.d_bnk,)
test_x = torch.full(shape, -2.85)

tr_ub = test_x + 0.02
tr_lb = test_x - 0.02
bounds = torch.stack([tr_lb, tr_ub]).cuda()


# cond fns
def cond_fn_bounds_soft_sigmoid(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    with torch.enable_grad():
        x = x.detach().requires_grad_(True)

        lower, upper = bounds[0].to(x.device), bounds[1].to(x.device)

        # normalize t: goes from 1000→0, so t_norm= t/1000 in [0,1]
        t_norm = (t.float() / 1000.0).view(-1, 1).to(x.device)

        # sigma shrinks as t→0
        sigma = 1e-2 + (1.0 - 1e-2) * t_norm
        sigma = sigma.expand_as(x)

        a = torch.sigmoid((x - lower) / sigma)
        b = torch.sigmoid((upper - x) / sigma)
        logp = (torch.log(a + 1e-12) + torch.log(b + 1e-12)).sum(dim=1)

        s = logp.sum()
        (grad_x,) = torch.autograd.grad(s, x)
    return grad_x.detach()

def cond_fn_bounds_proj(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    with torch.enable_grad():
        x = x.detach().requires_grad_(True)

        lower, upper = bounds[0].to(x.device), bounds[1].to(x.device)
        x_clamped = torch.max(lower, torch.min(x, upper))

        # schedule lambda: small at t=1000, large near 0
        lam = 0.1 + (50.0 - 0.1) * (1.0 - t.float() / 1000.0)
        lam = lam.view(-1, 1).to(x.device)

        dist2 = (x - x_clamped).pow(2).sum(dim=1)
        s = (-lam.squeeze() * dist2).sum()

        (grad_x,) = torch.autograd.grad(s, x)
    return grad_x.detach()

def cond_fn_bounds_hinge(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    with torch.enable_grad():
        x = x.detach().requires_grad_(True)

        lower, upper = bounds[0].to(x.device), bounds[1].to(x.device)

        lam = 0.1 + (50.0 - 0.1) * (1.0 - t.float() / 1000.0)
        lam = lam.view(-1, 1).to(x.device)

        low_viol = F.relu(lower - x).pow(2)
        up_viol  = F.relu(x - upper).pow(2)
        penalty = (low_viol + up_viol).sum(dim=1)

        s = (-lam.squeeze() * penalty).sum()
        (grad_x,) = torch.autograd.grad(s, x)
    return grad_x.detach()

# testing
N = 10
for cond_fn_bounds in [cond_fn_bounds_soft_sigmoid, cond_fn_bounds_proj, cond_fn_bounds_hinge]:
    # sampling

    logei_guide_z = model.ddim_sample(
        batch_size=N,
        sampling_steps=50,
        guidance_scale=1.0,
        cond_fn=cond_fn_bounds,
    ).cuda()

    # evaluation
    with torch.no_grad():
        inside = ((logei_guide_z >= bounds[0]) & (logei_guide_z <= bounds[1])).all(dim=1)
        accuracy = inside.float().mean().item()

        dist_low  = F.relu(bounds[0] - logei_guide_z)
        dist_high = F.relu(logei_guide_z - bounds[1])
        loss = (dist_low.pow(2) + dist_high.pow(2)).mean().item()

        print(f"results - acc: {accuracy}, loss: {loss}")



######################
### Bounds w log qEI
######################


def update_surr_model(surr_model, mll, learning_rte, train_z, train_y, n_epochs):  # noqa: ANN001
    surr_model = surr_model.train()
    optimizer = torch.optim.Adam([{"params": surr_model.parameters(), "lr": learning_rte}], lr=learning_rte)
    train_bsz = min(len(train_y), 128)
    train_dataset = TensorDataset(train_z.cuda(), train_y.cuda())
    train_loader = DataLoader(train_dataset, batch_size=train_bsz, shuffle=True)
    for _ in tqdm(range(n_epochs), leave=False):
        for inputs, scores in train_loader:
            optimizer.zero_grad()
            output = surr_model(inputs.cuda())
            loss = -mll(output, scores.cuda())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(surr_model.parameters(), max_norm=1.0)
            optimizer.step()
    surr_model = surr_model.eval()

    return surr_model


model = DiffusionModel.load_from_checkpoint("./data/molecule_diffusion.ckpt")
model.cuda()
model.freeze()

# df = pl.read_csv("./data/guacamol_train_data_first_20k.txt").select("smile", "logp").head(10_000)
df = pl.read_csv("./data/guacamol/guacamol_train_data_first_20k.csv").select("smile", "logp").head(10_000)

latents = []
for subdf in df.iter_slices(4096):
    tokens = model.vae.tokenize(subdf["smile"].to_list())
    mu, sigma = model.vae.encode(tokens.cuda())
    z = mu + sigma * torch.randn_like(sigma)
    latents.append(z.flatten(1))

train_x = torch.vstack(latents).cuda()
train_y = df["logp"].to_torch().cuda()

best_idx = torch.argmax(train_y)
best_x, best_y = train_x[best_idx], train_y[best_idx]

surrogate_model = GPModelDKL(
    train_x[:1024].cuda(),
    likelihood=gpytorch.likelihoods.GaussianLikelihood().cuda(),
).cuda()
surrogate_mll = PredictiveLogLikelihood(surrogate_model.likelihood, surrogate_model, num_data=train_x.shape[-1])

update_surr_model(
    surrogate_model,
    surrogate_mll,
    learning_rte=0.002,
    train_z=train_x,
    train_y=train_y,
    n_epochs=20,
)

log_ei_mod = qLogExpectedImprovement(
    model=surrogate_model,  # type: ignore
    best_f=best_y,
)

tr_ub = best_x + 0.02
tr_lb = best_x - 0.02
bounds = torch.stack([tr_lb, tr_ub]).cuda()


def cond_fn_log_ei(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    with torch.enable_grad():
        x = x.detach().requires_grad_(True)
        log_ei = log_ei_mod(x)
        if log_ei.dim() > 1:
            log_ei = log_ei.sum(dim=tuple(range(1, log_ei.dim())))
        s = log_ei.sum()
        (grad_x,) = torch.autograd.grad(s, x, retain_graph=False, create_graph=False)

    return grad_x.detach()

def cond_fn_bounds_hinge(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    with torch.enable_grad():
        x = x.detach().requires_grad_(True)

        lower, upper = bounds[0].to(x.device), bounds[1].to(x.device)

        lam = 0.1 + (50.0 - 0.1) * (1.0 - t.float() / 1000.0)
        lam = lam.view(-1, 1).to(x.device)

        low_viol = F.relu(lower - x).pow(2)
        up_viol  = F.relu(x - upper).pow(2)
        penalty = (low_viol + up_viol).sum(dim=1)

        s = (-lam.squeeze() * penalty).sum()
        (grad_x,) = torch.autograd.grad(s, x)
    return grad_x.detach()

def cond_fn_joint(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return 0.05 * cond_fn_log_ei(x, t) + 9.95 * cond_fn_bounds_hinge(x, t)


N = 10

logei_guide_z = model.ddim_sample(
    batch_size=N,
    sampling_steps=50,
    guidance_scale=1.0,
    cond_fn=cond_fn_log_ei,
)
with torch.no_grad():
    logei_guide_z_pred = log_ei_mod(logei_guide_z)

    inside = ((logei_guide_z >= bounds[0]) & (logei_guide_z <= bounds[1])).all(dim=1)
    guide_accuracy = inside.float().mean()

    dist_low  = F.relu(bounds[0] - logei_guide_z)
    dist_high = F.relu(logei_guide_z - bounds[1])
    guide_loss = (dist_low.pow(2) + dist_high.pow(2)).mean()

logei_ddim_z = model.ddim_sample(
    batch_size=N,
    sampling_steps=50,
    guidance_scale=1.0,
    cond_fn=cond_fn_bounds_hinge,
)
with torch.no_grad():
    logei_ddim_z_pred = log_ei_mod(logei_ddim_z)

    inside = ((logei_ddim_z >= bounds[0]) & (logei_ddim_z <= bounds[1])).all(dim=1)
    ddim_accuracy = inside.float().mean()

    dist_low  = F.relu(bounds[0] - logei_ddim_z)
    dist_high = F.relu(logei_ddim_z - bounds[1])
    ddim_loss = (dist_low.pow(2) + dist_high.pow(2)).mean()

logei_optim_z, _ = optimize_acqf(
    acq_function=log_ei_mod,
    bounds=bounds,
    q=N,
    num_restarts=10,
    raw_samples=256,
)
with torch.no_grad():
    logei_optim_z_pred = log_ei_mod(logei_optim_z)

    inside = ((logei_optim_z >= bounds[0]) & (logei_optim_z <= bounds[1])).all(dim=1)
    optim_accuracy = inside.float().mean()

    dist_low  = F.relu(bounds[0] - logei_optim_z)
    dist_high = F.relu(logei_optim_z - bounds[1])
    optim_loss = (dist_low.pow(2) + dist_high.pow(2)).mean()

print("=== Evals ===")
print(f"DDIM guided samples:   {logei_guide_z_pred.item():.3f}, {guide_accuracy.item():.3f}, {guide_loss.item():.6f}")
print(f"DDIM unguided samples: {logei_ddim_z_pred.item():.3f}, {ddim_accuracy.item():.3f}, {ddim_loss.item():.6f}")
print(f"Optimized samples:     {logei_optim_z_pred.item():.3f}, {optim_accuracy.item():.3f}, {optim_loss.item():.6f}")