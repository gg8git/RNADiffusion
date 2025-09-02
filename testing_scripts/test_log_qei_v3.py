import gpytorch
import polars as pl
import torch
from botorch.acquisition import LogExpectedImprovement
from gpytorch.mlls import PredictiveLogLikelihood
from rdkit import RDLogger
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from model import GPModelDKL
from model.diffusion_v2 import DiffusionModel

torch.set_printoptions(sci_mode=False, precision=3, linewidth=120)
RDLogger.DisableLog("rdApp.*")  # type: ignore


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


df = pl.read_csv("./data/guacamol_train_data_first_20k.txt").select("smile", "logp").head(10_000)

latents = []
for subdf in df.iter_slices(4096):
    tokens = model.vae.tokenize(subdf["smile"].to_list())
    mu, sigma = model.vae.encode(tokens.cuda())
    z = mu + sigma * torch.randn_like(sigma)
    latents.append(z.flatten(1))

train_x = torch.vstack(latents).cuda()
train_y = df["logp"].to_torch().cuda()

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

log_ei_mod = LogExpectedImprovement(
    model=surrogate_model,  # type: ignore
    best_f=train_y.max(),
)


def cond_fn_log_ei(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    with torch.enable_grad():
        x = x.detach().requires_grad_(True)
        log_ei = log_ei_mod(x.unsqueeze(1))
        if log_ei.dim() > 1:
            log_ei = log_ei.sum(dim=tuple(range(1, log_ei.dim())))
        s = log_ei.sum()
        (grad_x,) = torch.autograd.grad(s, x, retain_graph=False, create_graph=False)

    return grad_x.detach()


logei_guide_z = model.ddim_sample(batch_size=128, sampling_steps=50, guidance_scale=1.0, cond_fn=cond_fn_log_ei)
with torch.no_grad():
    logei_guide_z_pred = log_ei_mod(logei_guide_z.unsqueeze(1))

logei_rand_z = torch.randn_like(logei_guide_z)
with torch.no_grad():
    logei_rand_z_pred = log_ei_mod(logei_rand_z.unsqueeze(1))

logei_ddim_z = model.ddim_sample(
    batch_size=128,
    sampling_steps=50,
)
with torch.no_grad():
    logei_ddim_z_pred = log_ei_mod(logei_ddim_z.unsqueeze(1))

print(f"LogEI guided samples: {logei_guide_z_pred.mean():.3f} ± {logei_guide_z_pred.std():.3f}")
print(f"Random samples:       {logei_rand_z_pred.mean():.3f} ± {logei_rand_z_pred.std():.3f}")
print(f"LogEI DDIM samples:   {logei_ddim_z_pred.mean():.3f} ± {logei_ddim_z_pred.std():.3f}")
