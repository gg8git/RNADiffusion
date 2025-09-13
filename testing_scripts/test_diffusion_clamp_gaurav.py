import torch
import torch.nn.functional as F
from rdkit import RDLogger

from model.diffusion_v2 import DiffusionModel
from model.diffusion_v2.GaussianDiffusion import ExtinctPredictor

DATA_BATCH_SIZE = 1024
MODE = "pdop"
SURR_ITERS = [1, 4, 16, 64, 256]
ACQ_BATCH_SIZES = [4, 16, 64, 256, 1024, 4096]

# TODO: improve logging
LOG_NAME = "v2_testing"
LOG_PATH = f"./results/log_{LOG_NAME}.json" if LOG_NAME is not None else None

torch.set_printoptions(sci_mode=False, precision=3, linewidth=120)
RDLogger.DisableLog("rdApp.*")  # type: ignore


##########################################################
# Peptide
##########################################################
model = DiffusionModel.load_from_checkpoint("./data/peptide_diffusion.ckpt")
model.cuda()
model.freeze()

predictor = ExtinctPredictor.load_from_checkpoint("./data/extinct_predictor.ckpt")
predictor.cuda()
predictor.freeze()


def cond_fn_extinct(z: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    with torch.enable_grad():
        z = z.detach().requires_grad_(True)
        logits = predictor(z.view(z.shape[0], -1))
        logp = F.logsigmoid(logits)
        if logp.dim() > 1:
            logp = logp.sum(dim=tuple(range(1, logp.dim())))
        s = logp.sum()
        (grad_z,) = torch.autograd.grad(s, z, retain_graph=False, create_graph=False)
    return grad_z.detach()


ddim_z = model.ddim_sample(
    batch_size=1024,
    sampling_steps=50,
)

# fmt: off
TRC = torch.tensor([-0.254,  0.918,  0.553,  0.263, -2.132, -0.162,  2.220,  1.129,  0.346, -0.604,  1.344,  1.299,  1.409,  0.185,
         1.178, -0.041, -1.069, -0.972,  0.571,  0.214,  0.595,  1.887,  0.870,  1.522, -0.034,  0.515, -0.324, -0.173,
        -1.410,  1.013, -0.977,  0.701, -1.629,  0.724,  1.066, -0.083, -0.575,  1.176,  0.227, -0.764,  1.210, -0.216,
         0.772,  0.679,  0.794, -0.788,  0.316,  0.353,  0.450,  1.034,  1.412, -0.347, -0.009, -1.243,  0.232,  1.295,
        -0.825,  1.673, -1.967,  0.703,  0.209, -0.487,  1.683, -0.199, -0.973,  0.390, -0.538, -0.989, -0.358, -0.964,
        -0.719,  1.470, -0.981,  0.032,  0.274,  0.846,  1.124, -0.541, -0.194, -1.170,  1.702,  0.063,  0.152,  1.096,
        -0.180,  0.637,  0.030,  1.018,  0.254,  0.325, -2.034, -0.168,  0.069, -0.046, -0.376, -0.530,  1.787, -0.748,
         1.129,  1.890, -0.684,  0.027, -0.135,  0.231, -0.007, -0.963,  0.179, -0.518, -0.585, -1.516, -1.755,  0.325,
        -0.292, -2.100,  1.447,  0.490, -0.822, -0.718, -1.362,  0.379, -1.315, -0.267,  1.047,  0.308, -1.184,  1.629,
         1.697, -0.923,  0.029,  1.678,  1.308, -0.255,  1.711, -1.266, -1.282,  1.299,  1.497,  1.817, -0.902, -1.216,
        -1.148, -0.396, -0.332, -0.819,  0.621,  0.173,  0.933, -0.519,  1.358, -0.862,  0.677, -0.547, -0.235,  0.630,
         2.191,  0.474, -1.148,  0.981, -0.040, -2.036,  0.405,  1.118, -0.675,  1.153,  0.322, -0.990, -0.464,  0.930,
        -1.928,  0.090,  0.464,  0.894, -1.337, -0.920, -0.390, -1.371,  1.873,  0.743,  0.194, -0.452,  1.681, -2.327,
         0.351, -0.118, -1.383, -0.321,  1.110, -0.941, -0.627, -0.586, -1.119, -0.084, -1.555,  0.775, -2.006, -0.437,
         0.899, -0.718,  0.612, -0.387, -2.764, -0.685,  0.402,  0.190,  0.899, -1.022, -0.535, -1.021,  0.843, -0.164,
         0.265, -1.656,  0.652,  0.426,  0.287,  0.097, -0.621,  0.192,  1.065,  1.524,  1.195, -0.081, -0.602, -0.857,
         0.127,  1.084, -1.237,  0.008,  1.431, -0.037, -1.428,  0.795,  1.708,  1.110,  0.728,  0.880, -0.647,  0.772,
         0.621, -0.883,  0.342, -0.805,  1.078, -0.275, -0.489,  0.637,  0.873, -0.162, -1.488, -0.758,  0.117,  0.884,
        -1.072,  0.374, -0.449, -1.410], device='cuda:0').unsqueeze(0)
# fmt: on

TRHW = 0.2
guide_z = model.ddim_sample_tr(
    batch_size=1024,
    sampling_steps=200,
    cond_fn=cond_fn_extinct,
    guidance_scale=1.0,
    tr_center=TRC,
    tr_halfwidth=TRHW,
)
rand_z = torch.randn_like(ddim_z)

extinct_preds_ddim = (predictor(ddim_z).sigmoid() > 0.5).float()
extinct_preds_rand = (predictor(rand_z).sigmoid() > 0.5).float()
extinct_preds_guide = (predictor(guide_z).sigmoid() > 0.5).float()
print("Extinct:")
print(f"DDIM   Z: {extinct_preds_ddim.mean():.3f} +/- {extinct_preds_ddim.std():.3f}")
print(f"Random Z: {extinct_preds_rand.mean():.3f} +/- {extinct_preds_rand.std():.3f}")
print(f"Guided (tr clamp) Z: {extinct_preds_guide.mean():.3f} +/- {extinct_preds_guide.std():.3f}")

# Check how many points in guide_z are within epsilon of the trust region boundaries
TRLOWER = TRC - TRHW
TRUPPER = TRC + TRHW
on_boundary = ((guide_z - TRLOWER).abs() < 1e-3) | ((guide_z - TRUPPER).abs() < 1e-3)
# Average per example
avg = on_boundary.flatten(1).float().sum(dim=1).mean()
print(f"Average number of dimensions on boundary: {avg:.3f} / {guide_z.shape[1]}")

inside_boundary = (guide_z >= TRLOWER) & (guide_z <= TRUPPER)
count = inside_boundary.flatten(1).all(dim=1).sum()
print(f"Average number of points within boundary: {count:.0f} / {guide_z.shape[0]}")

print(f"Avg dist from center: {(guide_z - TRC).abs().mean():.3f}")

peptides = model.vae.detokenize(model.vae.sample(guide_z))
print("Sample peptides from guided diffusion:")
print(f" {len(set(peptides))} / {len(peptides)} samples are unique\n")

##############################
# TR Guidance
##############################
trials = [
    # (True, "midpoint", 0.1),
    # (True, "midpoint", 0.01),
    # (True, "midpoint_inc", 0.1),
    # (True, "midpoint_inc", 0.01),
    # (True, "midpoint_dec", 0.1),
    # (True, "midpoint_dec", 0.01),
    # (False, "midpoint", 0.1),
    # (False, "midpoint", 0.01),
    # (False, "midpoint_inc", 0.1),
    # (False, "midpoint_inc", 0.01),
    # (False, "midpoint_dec", 0.1),
    # (False, "midpoint_dec", 0.01),
    (True, "x0_hat", 0.1),
    (True, "x0_hat", 0.01),
    (True, "x0_hat_inc", 0.1),
    (True, "x0_hat_inc", 0.01),
    (True, "x0_hat_dec", 0.1),
    (True, "x0_hat_dec", 0.01),
    (False, "x0_hat", 0.1),
    (False, "x0_hat", 0.01),
    (False, "x0_hat_inc", 0.1),
    (False, "x0_hat_inc", 0.01),
    (False, "x0_hat_dec", 0.1),
    (False, "x0_hat_dec", 0.01),
]

for tr_clamp, tr_guidance, tr_guidance_scale  in trials:
    guide_z = model.ddim_sample_tr_guidance(
        batch_size=1024,
        sampling_steps=200,
        cond_fn=cond_fn_extinct,
        guidance_scale=1.0,
        tr_center=TRC,
        tr_halfwidth=TRHW,
        tr_clamp=tr_clamp,
        tr_guidance=tr_guidance,
        tr_guidance_scale=tr_guidance_scale,
    )
    rand_z = torch.randn_like(ddim_z)

    extinct_preds_ddim = (predictor(ddim_z).sigmoid() > 0.5).float()
    extinct_preds_rand = (predictor(rand_z).sigmoid() > 0.5).float()
    extinct_preds_guide = (predictor(guide_z).sigmoid() > 0.5).float()
    print("Extinct:")
    print(f"Guided {('tr_clamp' if tr_clamp else 'no tr clamp', tr_guidance, tr_guidance_scale)} Z: {extinct_preds_guide.mean():.3f} +/- {extinct_preds_guide.std():.3f}")

    # Check how many points in guide_z are within epsilon of the trust region boundaries
    TRLOWER = TRC - TRHW
    TRUPPER = TRC + TRHW
    on_boundary = ((guide_z - TRLOWER).abs() < 1e-3) | ((guide_z - TRUPPER).abs() < 1e-3)
    avg = on_boundary.flatten(1).float().sum(dim=1).mean()
    on_boundary_tight = ((guide_z - TRLOWER).abs() < 1e-5) | ((guide_z - TRUPPER).abs() < 1e-5)
    avg_tight = on_boundary.flatten(1).float().sum(dim=1).mean()
    print(f"Average number of dimensions on boundary: {avg:.3f} / {guide_z.shape[1]} (eps=1e-3), {avg_tight:.3f} / {guide_z.shape[1]} (eps=1e-5)")

    inside_boundary = (guide_z >= TRLOWER) & (guide_z <= TRUPPER)
    count = inside_boundary.flatten(1).all(dim=1).sum()
    print(f"Average number of points within boundary: {count:.0f} / {guide_z.shape[0]}")

    print(f"Avg dist from center: {(guide_z - TRC).abs().mean():.3f}")

    peptides = model.vae.detokenize(model.vae.sample(guide_z))
    print("Sample peptides from guided diffusion:")
    print(f" {len(set(peptides))} / {len(peptides)} samples are unique\n")
