import random
import warnings

import fire
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")
import os

os.environ["WANDB_SILENT"] = "True"
from lolbo.lolbo import LOLBOState
from lolbo.lolbo_constrained import LOLBOStateConstrained

try:
    import wandb

    WANDB_IMPORTED_SUCCESSFULLY = True
except ModuleNotFoundError:
    WANDB_IMPORTED_SUCCESSFULLY = False

from datamodules.kmer_datamodule import compute_peptide_train_zs, load_peptide_train_data
from datamodules.selfies_datamodule import compute_molecule_train_zs, load_molecule_train_data
from lolbo import ConstrainedPeptideObjective, ConstrainedPeptideObjectiveV2, MoleculeObjective, MoleculeObjectiveV2
from utils.guacamol_utils import GUACAMOL_TASK_NAMES


class Optimize:
    """
    Run LOLBO Optimization
    Args:
        task_id: String id for optimization task, by default the wandb project name will be f'optimize-{task_id}'
        seed: Random seed to be set. If None, no particular random seed is set
        track_with_wandb: if True, run progress will be tracked using Weights and Biases API
        wandb_entity: Username for your wandb account (valid username necessary iff track_with_wandb is True)
        wandb_project_name: Name of wandb project where results will be logged (if no name is specified, default project name will be f"optimimze-{self.task_id}")
        minimize: If True we want to minimize the objective, otherwise we assume we want to maximize the objective
        max_n_oracle_calls: Max number of oracle calls allowed (budget). Optimization run terminates when this budget is exceeded
        learning_rte: Learning rate for model updates
        acq_func: Acquisition function, must be either ei or ts (ei-->Expected Imporvement, ts-->Thompson Sampling)
        bsz: Acquisition batch size
        num_initialization_points: Number evaluated data points used to optimization initialize run
        init_n_update_epochs: Number of epochs to train the surrogate model for on initial data before optimization begins
        num_update_epochs: Number of epochs to update the model(s) for on each optimization step
        e2e_freq: Number of optimization steps before we update the models end to end (end to end update frequency)
        update_e2e: If True, we update the models end to end (we run LOLBO). If False, we never update end to end (we run TuRBO)
        k: We keep track of and update end to end on the top k points found during optimization
        verbose: If True, we print out updates such as best score found, number of oracle calls made, etc.
    """

    def __init__(
        self,
        task_id: str,
        seed: int = None,
        track_with_wandb: bool = False,
        wandb_entity: str = "molformers",
        wandb_project_name: str = "lolbo_encoder",
        minimize: bool = False,
        max_n_oracle_calls: int = 200_000,
        learning_rte: float = 0.001,
        acq_func: str = "ts",
        bsz: int = 10,
        num_initialization_points: int = 10_000,
        init_n_update_epochs: int = 20,
        num_update_epochs: int = 2,
        e2e_freq: int = 10,
        update_e2e: bool = True,
        k: int = 1_000,
        verbose: bool = True,
        constrained: bool = False,
        use_dsp: bool = False,
        use_vae_v2: bool = False,
        path_to_vae_statedict: str = "data/molecule_vae.ckpt",
        max_string_length: int = 1024,
        save_results_top_level_dir: str = "results/",  # local directory where run data/results are saved
        repaint_candidates: int = 128,  # number of candidates to use for thompson sampling acquisition # type: ignore
        # add peptide task + constraints if needed
        task_specific_args: list = [],  # list of additional args to be passed into objective funcion
        constraint_function_ids: list = [],  # list of strings identifying the black box constraint function to use
        constraint_thresholds: list = [],  # list of corresponding threshold values (floats)
        constraint_types: list = [],  # list of strings giving correspoding type for each threshold ("min" or "max" allowed)
    ):
        self.path_to_vae_statedict = path_to_vae_statedict
        self.max_string_length = max_string_length
        self.constrained = constrained
        self.repaint_candidates = repaint_candidates

        # add all local args to method args dict to be logged by wandb
        self.method_args = {}
        self.method_args["init"] = locals()
        del self.method_args["init"]["self"]
        self.seed = seed
        self.track_with_wandb = track_with_wandb
        self.wandb_entity = wandb_entity
        self.task_id = task_id
        self.task = "molecule" if self.task_id in GUACAMOL_TASK_NAMES + ["logp"] else "peptide"
        assert not (self.task != "peptide" and self.constrained), (
            "constrained optimization can only be done for peptides"
        )
        self.max_n_oracle_calls = max_n_oracle_calls
        self.verbose = verbose
        self.num_initialization_points = num_initialization_points
        self.e2e_freq = e2e_freq
        self.update_e2e = update_e2e
        self.save_results_top_level_dir = save_results_top_level_dir
        self.set_seed()
        if wandb_project_name:  # if project name specified
            self.wandb_project_name = wandb_project_name
        else:  # otherwise use defualt
            self.wandb_project_name = f"optimize-{self.task_id}"
        if not WANDB_IMPORTED_SUCCESSFULLY:
            assert not self.track_with_wandb, "Failed to import wandb, to track with wandb, try pip install wandb"
        if self.track_with_wandb:
            assert self.wandb_entity, (
                "Must specify a valid wandb account username (wandb_entity) to run with wandb tracking"
            )

        # handle peptide task + constraint args
        if self.task == "peptide":
            self.score_version = task_specific_args[0]
            self.task_specific_args = task_specific_args
        assert len(constraint_function_ids) == len(constraint_thresholds)
        assert len(constraint_thresholds) == len(constraint_types)
        self.constraint_function_ids = (
            constraint_function_ids  # list of strings identifying the black box constraint function to use
        )
        self.constraint_thresholds = constraint_thresholds  # list of corresponding threshold values (floats)
        self.constraint_types = (
            constraint_types  # list of strings giving correspoding type for each threshold ("min" or "max" allowed)
        )

        # initialize train data for particular task
        #   must define self.init_train_x, self.init_train_y, and self.init_train_z
        self.load_train_data()
        # initialize latent space objective (self.objective) for particular task
        assert acq_func not in ["ddim", "ddim_repaint", "ddim_repaint_tr"] or use_vae_v2, (
            "if acq_func is ddim or ddim_repaint, must use vae v2"
        )
        self.initialize_objective(use_vae_v2)
        assert (
            isinstance(self.objective, MoleculeObjective)
            or isinstance(self.objective, MoleculeObjectiveV2)
            or isinstance(self.objective, ConstrainedPeptideObjective)
            or isinstance(self.objective, ConstrainedPeptideObjectiveV2)
        ), "self.objective must be an instance of MoleculeObjective or MoleculeObjectiveV2"
        assert type(self.init_train_x) is list, "load_train_data() must set self.init_train_x to a list of xs"
        assert torch.is_tensor(self.init_train_y), "load_train_data() must set self.init_train_y to a tensor of ys"
        assert torch.is_tensor(self.init_train_z), "load_train_data() must set self.init_train_z to a tensor of zs"
        assert len(self.init_train_x) == self.num_initialization_points, (
            f"load_train_data() must initialize exactly self.num_initialization_points={self.num_initialization_points} xs, instead got {len(self.init_train_x)} xs"
        )
        assert self.init_train_y.shape[0] == self.num_initialization_points, (
            f"load_train_data() must initialize exactly self.num_initialization_points={self.num_initialization_points} ys, instead got {self.init_train_y.shape[0]} ys"
        )
        assert self.init_train_z.shape[0] == self.num_initialization_points, (
            f"load_train_data() must initialize exactly self.num_initialization_points={self.num_initialization_points} zs, instead got {self.init_train_z.shape[0]} zs"
        )

        # initialize lolbo state
        if self.constrained:
            self.lolbo_state = LOLBOStateConstrained(
                objective=self.objective,
                train_x=self.init_train_x,
                train_y=self.init_train_y,
                train_z=self.init_train_z,
                train_c=self.init_train_c,
                minimize=minimize,
                k=k,
                num_update_epochs=num_update_epochs,
                init_n_epochs=init_n_update_epochs,
                learning_rte=learning_rte,
                bsz=bsz,
                acq_func=acq_func,
                use_dsp=use_dsp,
                verbose=verbose,
                task=self.task,
                repaint_candidates=self.repaint_candidates,
            )
        else:
            self.lolbo_state = LOLBOState(
                objective=self.objective,
                train_x=self.init_train_x,
                train_y=self.init_train_y,
                train_z=self.init_train_z,
                minimize=minimize,
                k=k,
                num_update_epochs=num_update_epochs,
                init_n_epochs=init_n_update_epochs,
                learning_rte=learning_rte,
                bsz=bsz,
                acq_func=acq_func,
                use_dsp=use_dsp,
                verbose=verbose,
                task=self.task,
                repaint_candidates=self.repaint_candidates,
            )

        # add args to method args dict to be logged by wandb
        self.method_args["molopt"] = locals()
        del self.method_args["molopt"]["self"]

    def initialize_save_results_dir(
        self,
    ):
        save_dir = f"{self.save_results_top_level_dir}"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_dir = save_dir + f"{self.task_id}/"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_dir = save_dir + f"{self.wandb_run_name}/"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        self.save_results_dir = save_dir
        return self

    def initialize_objective(self, use_vae_v2=False):
        if self.task == "molecule":
            assert hasattr(self, "init_smiles_to_selfies"), (
                "molecule objective must have init smiles to selfies function"
            )

            if use_vae_v2:
                # initialize molecule objective
                self.objective = MoleculeObjectiveV2(
                    task_id=self.task_id,
                    path_to_vae_statedict=self.path_to_vae_statedict,
                    max_string_length=self.max_string_length,
                    smiles_to_selfies=self.init_smiles_to_selfies,
                )
            else:
                # initialize molecule objective
                self.objective = MoleculeObjective(
                    task_id=self.task_id,
                    path_to_vae_statedict=self.path_to_vae_statedict,
                    max_string_length=self.max_string_length,
                    smiles_to_selfies=self.init_smiles_to_selfies,
                )

            # if train zs have not been pre-computed for particular vae, compute them
            #   by passing initialization selfies through vae
            if self.init_train_z is None:
                self.init_train_z = compute_molecule_train_zs(
                    self.objective,
                    self.init_train_x,
                )

        elif self.task == "peptide":
            assert hasattr(self, "task_specific_args"), "molecule objective must have task specific args argument"
            if use_vae_v2:
                self.objective = ConstrainedPeptideObjectiveV2(
                    task_id=self.task_id,
                    task_specific_args=self.task_specific_args,
                    max_string_length=self.max_string_length,
                    path_to_vae_statedict=self.path_to_vae_statedict,
                    # constraints
                    constraint_function_ids=self.constraint_function_ids,  # list of strings identifying the black box constraint function to use
                    constraint_thresholds=self.constraint_thresholds,  # list of corresponding threshold values (floats)
                    constraint_types=self.constraint_types,  # list of strings giving correspoding type for each threshold ("min" or "max" allowed)
                )
            else:
                self.objective = ConstrainedPeptideObjective(
                    task_id=self.task_id,
                    task_specific_args=self.task_specific_args,
                    max_string_length=self.max_string_length,
                    path_to_vae_statedict=self.path_to_vae_statedict,
                    # constraints
                    constraint_function_ids=self.constraint_function_ids,  # list of strings identifying the black box constraint function to use
                    constraint_thresholds=self.constraint_thresholds,  # list of corresponding threshold values (floats)
                    constraint_types=self.constraint_types,  # list of strings giving correspoding type for each threshold ("min" or "max" allowed)
                )

            # if train zs have not been pre-computed for particular vae, compute them
            #   by passing initialization selfies through vae
            if self.init_train_z is None:
                self.init_train_z = compute_peptide_train_zs(
                    self.objective,
                    self.init_train_x,
                )
            self.init_train_c = self.objective.compute_constraints(self.init_train_x)

        return self

    def load_train_data(self):
        """Load in or randomly initialize self.num_initialization_points
        total initial data points to kick-off optimization
        Must define the following:
            self.init_train_x (a list of x's)
            self.init_train_y (a tensor of scores/y's)
            self.init_train_z (a tensor of corresponding latent space points)
        """
        assert self.num_initialization_points <= 20_000

        # fix this shit

        if self.task == "molecule":
            smiles, selfies, zs, ys = load_molecule_train_data(
                task_id=self.task_id,
                num_initialization_points=self.num_initialization_points,
                path_to_vae_statedict=self.path_to_vae_statedict,
            )
            self.init_train_x, self.init_train_z, self.init_train_y = smiles, zs, ys
            if self.verbose:
                print("Loaded initial training data")
                print("train y shape:", self.init_train_y.shape)
                print(f"train x list length: {len(self.init_train_x)}\n")

            # create initial smiles to selfies dict
            self.init_smiles_to_selfies = {}
            for ix, smile in enumerate(self.init_train_x):
                self.init_smiles_to_selfies[smile] = selfies[ix]

        elif self.task == "peptide":
            self.init_train_x, self.init_train_z, self.init_train_y = load_peptide_train_data(
                score_version=self.score_version,
                num_initialization_points=self.num_initialization_points,
                path_to_vae_statedict=self.path_to_vae_statedict,
            )

        return self

    def set_seed(self):
        # torch.set_float32_matmul_precision("highest")
        if self.seed is not None:
            torch.manual_seed(self.seed)
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            os.environ["PYTHONHASHSEED"] = str(self.seed)

        return self

    def create_wandb_tracker(self):
        if self.track_with_wandb:
            self.tracker = wandb.init(
                project=self.wandb_project_name,
                entity=self.wandb_entity,
                config={k: v for method_dict in self.method_args.values() for k, v in method_dict.items()},
            )
            self.wandb_run_name = wandb.run.name
        else:
            self.tracker = None
            self.wandb_run_name = "no-wandb-tracking"

        return self

    def log_data_to_wandb_on_each_loop(self):
        if self.track_with_wandb:
            dict_log = {
                "best_found": self.lolbo_state.best_score_seen,
                "n_oracle_calls": self.lolbo_state.objective.num_calls,
                "total_number_of_e2e_updates": self.lolbo_state.tot_num_e2e_updates,
                "best_input_seen": self.lolbo_state.best_x_seen,
            }
            dict_log["TR_length"] = self.lolbo_state.tr_state.length
            self.tracker.log(dict_log)

        return self

    def run_lolbo(self):
        """Main optimization loop"""
        # creates wandb tracker iff self.track_with_wandb == True
        self.create_wandb_tracker()
        # create directory to periodically save best sequences and scores found during opt:
        self.initialize_save_results_dir()
        # main optimization loop
        while self.lolbo_state.objective.num_calls < self.max_n_oracle_calls:
            print(
                f"new loop, num calls: {self.lolbo_state.objective.num_calls}, curr best: {self.lolbo_state.best_score_seen}"
            )

            self.log_data_to_wandb_on_each_loop()
            # update models end to end when we fail to make
            #   progress e2e_freq times in a row (e2e_freq=10 by default)
            if (self.lolbo_state.progress_fails_since_last_e2e >= self.e2e_freq) and self.update_e2e:
                self.lolbo_state.update_models_e2e()
                self.lolbo_state.recenter()
            else:  # otherwise, just update the surrogate model on data
                self.lolbo_state.update_surrogate_model()
            # generate new candidate points, evaluate them, and update data
            self.lolbo_state.acquisition()
            if self.lolbo_state.tr_state.restart_triggered:
                self.lolbo_state.initialize_tr_state()
            if self.constrained:
                self.lolbo_state.recenter_trs()
            # if a new best has been found, print out new best input and score:
            if self.lolbo_state.new_best_found:
                if self.verbose:
                    print("\nNew best found:")
                    self.print_progress_update()
                self.lolbo_state.new_best_found = False
            self.save_topk_results()  # save topk solutions found so far locally on each step (in case we kill runs early)

        # if verbose, print final results
        if self.verbose:
            print("\nOptimization Run Finished, Final Results:")
            self.print_progress_update()

        # log top k scores and xs in table
        self.log_topk_table_wandb()

        return self

    def print_progress_update(self):
        """Important data printed each time a new
        best input is found, as well as at the end
        of the optimization run
        (only used if self.verbose==True)
        More print statements can be added her as desired
        """
        if self.track_with_wandb:
            print(f"Optimization Run: {self.wandb_project_name}, {wandb.run.name}")
        print(f"Best X Found: {self.lolbo_state.best_x_seen}")
        print(f"Best {self.objective.task_id} Score: {self.lolbo_state.best_score_seen}")
        print(f"Total Number of Oracle Calls (Function Evaluations): {self.lolbo_state.objective.num_calls}")

        return self

    def save_topk_results(
        self,
    ):
        """Save top k solutions found during optimization so far to CSV"""
        results_df = {
            "top_k_scores": self.lolbo_state.top_k_scores,
            "top_k_solutions": self.lolbo_state.top_k_xs,
        }
        results_df = pd.DataFrame.from_dict(results_df)
        results_df.to_csv(f"{self.save_results_dir}top_k_solutions_found.csv", index=False)

        return self

    def log_topk_table_wandb(self):
        """After optimization finishes, log
        top k inputs and scores found
        during optimization"""
        if self.track_with_wandb:
            cols = ["Top K Scores", "Top K Strings"]
            data_list = []
            for ix, score in enumerate(self.lolbo_state.top_k_scores):
                data_list.append([score, str(self.lolbo_state.top_k_xs[ix])])
            top_k_table = wandb.Table(columns=cols, data=data_list)
            self.tracker.log({"top_k_table": top_k_table})
            self.tracker.finish()

        return self

    def done(self):
        return None


def new(**kwargs):
    return Optimize(**kwargs)


if __name__ == "__main__":
    fire.Fire(Optimize)

# python3 diff_bo.py --task_id pdop --num_initialization_points 500 --max_n_oracle_calls 10000 --k 10 --use_vae_v2 True - run_lolbo - done
