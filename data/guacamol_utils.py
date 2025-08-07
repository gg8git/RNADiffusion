import math

import torch
import numpy as np
import pandas as pd
from guacamol import standard_benchmarks
from rdkit import Chem

med1 = standard_benchmarks.median_camphor_menthol()  #'Median molecules 1'
med2 = standard_benchmarks.median_tadalafil_sildenafil()  #'Median molecules 2',
pdop = standard_benchmarks.perindopril_rings()  # 'Perindopril MPO',
osmb = standard_benchmarks.hard_osimertinib()  # 'Osimertinib MPO',
adip = standard_benchmarks.amlodipine_rings()  # 'Amlodipine MPO'
siga = standard_benchmarks.sitagliptin_replacement()  #'Sitagliptin MPO'
zale = standard_benchmarks.zaleplon_with_other_formula()  # 'Zaleplon MPO'
valt = standard_benchmarks.valsartan_smarts()  #'Valsartan SMARTS',
dhop = standard_benchmarks.decoration_hop()  # 'Deco Hop'
shop = standard_benchmarks.scaffold_hop()  # Scaffold Hop'
rano = standard_benchmarks.ranolazine_mpo()  #'Ranolazine MPO'
fexo = standard_benchmarks.hard_fexofenadine()  # 'Fexofenadine MPO'... 'make fexofenadine less greasy'


guacamol_objs = {
    "med1": med1,
    "pdop": pdop,
    "adip": adip,
    "rano": rano,
    "osmb": osmb,
    "siga": siga,
    "zale": zale,
    "valt": valt,
    "med2": med2,
    "dhop": dhop,
    "shop": shop,
    "fexo": fexo,
}


GUACAMOL_TASK_NAMES = ["med1", "pdop", "adip", "rano", "osmb", "siga", "zale", "valt", "med2", "dhop", "shop", "fexo"]


def smile_to_guacamole_score(obj_func_key, smile):
    if smile is None or len(smile) == 0:
        return None
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None
    func = guacamol_objs[obj_func_key]
    score = func.objective.score(smile)
    if score is None:
        return None
    if score < 0:
        return None
    return score


def smiles_to_desired_scores(smiles_list, task_id="logp"):
    scores = []
    for smiles_str in smiles_list:
        score_ = smile_to_guacamole_score(task_id, smiles_str)
        if (score_ is not None) and (math.isfinite(score_)):
            scores.append(score_)
        else:
            scores.append(0)  # np.nan)

    return np.array(scores)


def load_molecule_train_data(
    task_id,
    path_to_vae_statedict,
    num_initialization_points=10_000,
):  
    df = pd.read_csv("data/guacamol/guacamol_train_data_first_20k.csv")
    df = df[0:num_initialization_points]
    train_x_smiles = df["smile"].values.tolist()
    train_x_selfies = df["selfie"].values.tolist()
    train_y = torch.from_numpy(df[task_id].values).float()
    train_y = train_y.unsqueeze(-1)
    train_z = load_train_z(
        num_initialization_points=num_initialization_points, path_to_vae_statedict=path_to_vae_statedict
    )

    return train_x_smiles, train_x_selfies, train_z, train_y

def load_train_z(
    num_initialization_points,
    path_to_vae_statedict,
):
    state_dict_file_type = path_to_vae_statedict.split(".")[-1]  # usually .pt or .ckpt
    path_to_init_train_zs = path_to_vae_statedict.replace(f".{state_dict_file_type}", "-train-zs.csv")
    # if we have a path to pre-computed train zs for vae, load them
    try:
        zs = pd.read_csv(path_to_init_train_zs, header=None).values
        # make sure we have a sufficient number of saved train zs
        assert len(zs) >= num_initialization_points
        zs = zs[0:num_initialization_points]
        zs = torch.from_numpy(zs).float()
    # otherwisee, set zs to None
    except:
        zs = None

    return zs


def compute_train_zs(mol_objective, init_train_x, bsz=64):
    init_zs = []
    # make sure vae is in eval mode
    mol_objective.vae.eval()
    n_batches = math.ceil(len(init_train_x) / bsz)
    for i in range(n_batches):
        xs_batch = init_train_x[i * bsz : (i + 1) * bsz]
        zs, _ = mol_objective.vae_forward(xs_batch)
        init_zs.append(zs.detach().cpu())
    init_zs = torch.cat(init_zs, dim=0)
    # now save the zs so we don't have to recompute them in the future:
    state_dict_file_type = mol_objective.path_to_vae_statedict.split(".")[-1]  # usually .pt or .ckpt
    path_to_init_train_zs = mol_objective.path_to_vae_statedict.replace(f".{state_dict_file_type}", "-train-zs.csv")
    zs_arr = init_zs.cpu().detach().numpy()
    pd.DataFrame(zs_arr).to_csv(path_to_init_train_zs, header=None, index=None)

    return init_zs