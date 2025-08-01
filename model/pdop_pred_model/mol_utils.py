import math

import numpy as np
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