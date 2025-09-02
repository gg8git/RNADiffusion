import gpytorch
import polars as pl
import torch
from botorch.acquisition import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import PredictiveLogLikelihood
from rdkit import RDLogger
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from model import GPModelDKL
from model.diffusion_v2 import DiffusionModel
from utils.guacamol_utils import smiles_to_desired_scores

torch.set_printoptions(sci_mode=False, precision=3, linewidth=120)
RDLogger.DisableLog("rdApp.*")  # type: ignore


#################################
### Setup 
#################################

model = DiffusionModel.load_from_checkpoint("./data/molecule_diffusion.ckpt")
model.cuda()
model.freeze()


#################################
### Acquisition Utils
#################################

def cond_fn_log_ei_generator(log_ei_mod):

    def cond_fn_log_ei(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            log_ei = log_ei_mod(x.unsqueeze(1))
            if log_ei.dim() > 1:
                log_ei = log_ei.sum(dim=tuple(range(1, log_ei.dim())))
            s = log_ei.sum()
            (grad_x,) = torch.autograd.grad(s, x, retain_graph=False, create_graph=False)

        return grad_x.detach()

    return cond_fn_log_ei


def ddim_no_cond_acq(batch_size, log_qei):
    z = model.ddim_sample(
        batch_size=batch_size,
        sampling_steps=50,
    )
    return z

def ddim_cond_acq(batch_size, log_qei):
    z = model.ddim_sample(
        batch_size=batch_size,
        sampling_steps=50,
        guidance_scale=1.0,
        cond_fn=cond_fn_log_ei_generator(log_qei),
    )
    return z

def rand_acq(batch_size, log_qei):
    z = torch.randn_like(batch_size, model.vae.n_acc * model.vae.d_bnk)
    return z

def optimize_acqf_acq(batch_size, log_qei):
    shape = (model.vae.n_acc * model.vae.d_bnk,)
    z, _ = optimize_acqf(
        acq_function=cond_fn_log_ei_generator(log_qei),
        bounds=torch.vstack(
            [torch.full(shape, -3.0), torch.full(shape, 3.0)],
        ),
        q=batch_size,
        num_restarts=10,
        raw_samples=256,
    )
    return z


def get_batch_scores(batch_z, mode="pdop"):
    with torch.no_grad():
        # find a way to use diffusion model vae
        batch_tokens = model.vae.sample(batch_z)
        batch_smiles = model.vae.detokenize(batch_tokens)
        scores = smiles_to_desired_scores(batch_smiles, mode)
    batch_y = torch.tensor(scores, device=model.device, dtype=torch.float32)
    flat_batch_z = batch_z.reshape(batch_z.size(0), -1)
    return flat_batch_z, batch_y


#################################
### Surrogate Update Utils 
#################################

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


#################################
### Setup 
#################################

def init_surr_model(num_init=10_000):
    df = pl.read_csv("./data/guacamol/guacamol_train_data_first_20k.csv").select("smile", "logp").head(num_init)

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

    return surrogate_model, surrogate_mll


######################
### BO Loop 
######################

def bo_loop(batch_size, acq_func, surrogate_model, surrogate_mll, all_y):
    # acquisition
    log_ei_mod = qLogExpectedImprovement(
        model=surrogate_model,  # type: ignore
        best_f=all_y.max(),
    )
    
    acq_batch_z = acq_func(batch_size, log_ei_mod)
    _, acq_batch_y = get_batch_scores(acq_batch_z)

    with torch.no_grad():
        acq_logei = log_ei_mod(acq_batch_z.unsqueeze(1))

    print(f"acq - prev max: {all_y.max().detach().cpu().item()}, batch max: {acq_batch_y.max().detach().cpu().item()}, batch logei: {acq_logei}")

    # update surr model
    update_surr_model(
        surrogate_model,
        surrogate_mll,
        learning_rte=0.002,
        train_z=acq_batch_z,
        train_y=acq_batch_y,
        n_epochs=20,
    )

    return surrogate_model, acq_batch_z, acq_batch_y


######################
### Run BO
######################

def run_bo(acq_batch_size, max_oracle_calls, acq_func):
    surrogate_model, surrogate_mll, all_z, all_y = init_surr_model()

    num_oracle_calls = 0
    while num_oracle_calls < max_oracle_calls:
        print(f"loop - num_oracle_calls: {num_oracle_calls}")
        surrogate_model, acq_batch_z, acq_batch_y = bo_loop(acq_batch_size, acq_func, surrogate_model, surrogate_mll, all_y)
        all_z = torch.cat([all_z, acq_batch_z], dim=0)
        all_y = torch.cat([all_y, acq_batch_y], dim=0)
        num_oracle_calls += acq_batch_size
    
    print(f"finish - max: {all_y.max().detach().cpu().item()}")
    return all_y.max().detach().cpu().item()

ACQ_BATCH_SIZE = 256
MAX_ORACLE_CALLS = 100_000
maxs = []
for acq_func in [ddim_no_cond_acq, ddim_cond_acq, optimize_acqf_acq]:
    maxs.append(run_bo(ACQ_BATCH_SIZE, MAX_ORACLE_CALLS, acq_func))
print(f"summary: {maxs}")







