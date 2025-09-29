import torch
import torch.nn.functional as F
from rdkit import RDLogger
from torch.quasirandom import SobolEngine

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


model = DiffusionModel.load_from_checkpoint("./data/peptide_diffusion.ckpt")
model.cuda()
model.freeze()

predictor = ExtinctPredictor.load_from_checkpoint("./data/extinct_predictor.ckpt")
predictor.cuda()
predictor.freeze()

model.predictor = predictor

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

TRC_ONES = torch.ones(256).to(torch.float32).cuda().unsqueeze(0)
TRC_RAND = torch.rand(256, dtype=torch.float32, device="cuda").unsqueeze(0)

N = 1024
x_known = TRC.repeat(N, 1).view(N, model.n_bn, model.d_bn)
mask = (torch.rand_like(x_known) > 0.2).float()

repaint_mask_z = model.ddim_repaint(
    x_known=x_known,
    mask=mask,
    sampling_steps=50,
    u_steps=20,
)

extinct_preds_guide_mask = (predictor(repaint_mask_z).sigmoid() > 0.5).float()
print(f"Guide No Cond: {extinct_preds_guide_mask.mean():.3f} +/- {extinct_preds_guide_mask.std():.3f}")
_tmp = torch.stack([x_known[0].flatten(), repaint_mask_z[0], mask[0].flatten()], dim=0)
print(_tmp[:, :10])


# for guidance_scale in [1.0, 2.0, 4.0, 6.0, 8.0]:
#     for threshold in [0.2, 0.5, 0.8]:
#         mask = (torch.rand_like(x_known) > threshold).float()

#         guide_mask_z = model.ddim_repaint(
#             x_known=x_known,
#             mask=mask,
#             sampling_steps=50,
#             u_steps=20,
#             sample_extinct=True,
#             extinct_guidance_scale=guidance_scale,
#         )

#         extinct_preds_guide_mask = (predictor(guide_mask_z).sigmoid() > 0.5).float()
#         print(f"Guide Extinct (scale: {guidance_scale}, threshold: {threshold}): {extinct_preds_guide_mask.mean():.3f} +/- {extinct_preds_guide_mask.std():.3f}")
#         _tmp = torch.stack([x_known[0].flatten(), guide_mask_z[0], mask[0].flatten()], dim=0)
#         # print(_tmp[:, :10])


################
### TS
################

TRHW = 0.5
repaint_candidates = 128
dtype = torch.float32
# device = torch.cuda

tr_lb = TRC - TRHW
tr_ub = TRC + TRHW

dim = 256
tr_lb = tr_lb.cuda()
tr_ub = tr_ub.cuda()
sobol = SobolEngine(dim, scramble=True)
pert = sobol.draw(repaint_candidates).to(dtype=dtype).cuda()
pert = tr_lb + (tr_ub - tr_lb) * pert
tr_lb = tr_lb.cuda()
tr_ub = tr_ub.cuda()
# Create a perturbation mask
prob_perturb = min(20.0 / dim, 1.0)
mask = (torch.rand(repaint_candidates, dim, dtype=dtype).cuda()) <= prob_perturb
ind = torch.where(mask.sum(dim=1) == 0)[0]
mask[ind, torch.randint(0, dim - 1, size=(len(ind),)).cuda()] = 1
mask = (~mask.cuda()).float()

X_cand = TRC.expand(repaint_candidates, dim).clone()

X_cand = model.ddim_repaint(
    x_known=X_cand.cuda(),
    mask=mask,
    sampling_steps=50,
    u_steps=10,
    tr_center=None,
    tr_halfwidth=None,
)

extinct = (predictor(X_cand).sigmoid() > 0.5).float()
print(f"around extinct + no extinct guidance + no tr: {extinct.mean():.3f} +/- {extinct.std():.3f}")

X_cand = model.ddim_repaint(
    x_known=X_cand.cuda(),
    mask=mask,
    sampling_steps=50,
    u_steps=10,
    tr_center=None,
    tr_halfwidth=None,
    sample_extinct=True,
    extinct_guidance_scale=1.0,
)

extinct = (predictor(X_cand).sigmoid() > 0.5).float()
print(f"around extinct + extinct guidance + no tr: {extinct.mean():.3f} +/- {extinct.std():.3f}")


tr_lb = TRC_ONES - TRHW
tr_ub = TRC_ONES + TRHW

dim = 256
tr_lb = tr_lb.cuda()
tr_ub = tr_ub.cuda()
sobol = SobolEngine(dim, scramble=True)
pert = sobol.draw(repaint_candidates).to(dtype=dtype).cuda()
pert = tr_lb + (tr_ub - tr_lb) * pert
tr_lb = tr_lb.cuda()
tr_ub = tr_ub.cuda()
# Create a perturbation mask
prob_perturb = min(20.0 / dim, 1.0)
mask = (torch.rand(repaint_candidates, dim, dtype=dtype).cuda()) <= prob_perturb
ind = torch.where(mask.sum(dim=1) == 0)[0]
mask[ind, torch.randint(0, dim - 1, size=(len(ind),)).cuda()] = 1
mask = (~mask.cuda()).float()

X_cand = TRC_ONES.expand(repaint_candidates, dim).clone()

X_cand = model.ddim_repaint(
    x_known=X_cand.cuda(),
    mask=mask,
    sampling_steps=50,
    u_steps=10,
    tr_center=None,
    tr_halfwidth=None,
)

extinct = (predictor(X_cand).sigmoid() > 0.5).float()
print(f"around ones + no extinct guidance + no tr: {extinct.mean():.3f} +/- {extinct.std():.3f}")

X_cand = model.ddim_repaint(
    x_known=X_cand.cuda(),
    mask=mask,
    sampling_steps=50,
    u_steps=10,
    tr_center=None,
    tr_halfwidth=None,
    sample_extinct=True,
    extinct_guidance_scale=1.0,
)

extinct = (predictor(X_cand).sigmoid() > 0.5).float()
print(f"around ones + extinct guidance + no tr: {extinct.mean():.3f} +/- {extinct.std():.3f}")


tr_lb = TRC_RAND - TRHW
tr_ub = TRC_RAND + TRHW

dim = 256
tr_lb = tr_lb.cuda()
tr_ub = tr_ub.cuda()
sobol = SobolEngine(dim, scramble=True)
pert = sobol.draw(repaint_candidates).to(dtype=dtype).cuda()
pert = tr_lb + (tr_ub - tr_lb) * pert
tr_lb = tr_lb.cuda()
tr_ub = tr_ub.cuda()
# Create a perturbation mask
prob_perturb = min(20.0 / dim, 1.0)
mask = (torch.rand(repaint_candidates, dim, dtype=dtype).cuda()) <= prob_perturb
ind = torch.where(mask.sum(dim=1) == 0)[0]
mask[ind, torch.randint(0, dim - 1, size=(len(ind),)).cuda()] = 1
mask = (~mask.cuda()).float()

X_cand = TRC_RAND.expand(repaint_candidates, dim).clone()

X_cand = model.ddim_repaint(
    x_known=X_cand.cuda(),
    mask=mask,
    sampling_steps=50,
    u_steps=10,
    tr_center=None,
    tr_halfwidth=None,
)

extinct = (predictor(X_cand).sigmoid() > 0.5).float()
print(f"around rand + no extinct guidance + no tr: {extinct.mean():.3f} +/- {extinct.std():.3f}")

X_cand = model.ddim_repaint(
    x_known=X_cand.cuda(),
    mask=mask,
    sampling_steps=50,
    u_steps=10,
    tr_center=None,
    tr_halfwidth=None,
    sample_extinct=True,
    extinct_guidance_scale=1.0,
)

extinct = (predictor(X_cand).sigmoid() > 0.5).float()
print(f"around rand + extinct guidance + no tr: {extinct.mean():.3f} +/- {extinct.std():.3f}")