import math
import time

import gpytorch
import numpy as np
import torch
from gpytorch.mlls import PredictiveLogLikelihood

from model.diffusion_v2 import DiffusionModel
from model.surrogate_model import GPModelDKL

from .turbo import TurboStateConstrained, generate_batch, update_state
from .update_models_utils import (
    update_constraint_surr_models,
    update_models_end_to_end_with_constraints,
    update_surr_model,
)


class LOLBOStateConstrained:
    def __init__(
        self,
        objective,
        train_x,
        train_y,
        train_z,
        train_c,
        k=1_000,
        minimize=False,
        num_update_epochs=2,
        init_n_epochs=20,
        learning_rte=0.01,
        bsz=10,
        acq_func="ts",
        verbose=True,
        task="molecule",
        repaint_candidates=128,
    ):
        self.objective = objective  # Objective with vae and associated diversity function for particular task
        self.train_x = train_x  # initial train x data
        self.train_y = train_y  # initial train y data
        self.train_c = train_c  # initial train c data (for constraints, see lolrobot subclass)
        self.train_z = train_z  # initial train z data (for latent space objectives, see lolrobot subclasss)
        self.minimize = (
            minimize  # if True we want to minimize the objective, otherwise we assume we want to maximize the objective
        )
        self.k = k  # track and update on top k scoring points found
        self.num_update_epochs = num_update_epochs  # num epochs update models
        self.init_n_epochs = init_n_epochs  # num epochs train surr model on initial data
        self.learning_rte = learning_rte  # lr to use for model updates
        self.bsz = bsz  # acquisition batch size
        self.acq_func = acq_func  # acquisition function (Expected Improvement (ei) or Thompson Sampling (ts))
        self.verbose = verbose
        self.task = task
        self.repaint_candidates = repaint_candidates  # number of candidates to repaint when using ddim with repainting

        assert acq_func in ["ts", "ei", "ddim", "ddim_repaint", "ddim_repaint_tr"]
        if minimize:
            self.train_y = self.train_y * -1

        self.num_new_points = 0  # number of newly acquired points (in acquisiton)
        self.best_score_seen = torch.max(train_y)
        self.best_x_seen = train_x[torch.argmax(train_y.squeeze())]
        self.initial_model_training_complete = (
            False  # initial training of surrogate model uses all data for more epochs
        )
        self.new_best_found = False

        self.initialize_surrogate_model()
        self.initialize_xs_to_scores_dict()
        self.initialize_tr_state()

        self.progress_fails_since_last_e2e = 0
        self.tot_num_e2e_updates = 0
        self.initialize_top_k()  # only track top k for LOL-BO, unnessary for regular opt

        self.initialize_diffusion_model()

    def search_space_data(self):
        return self.train_z

    def initialize_constraint_surrogates(self):
        self.c_models = []
        self.c_mlls = []
        for i in range(self.train_c.shape[1]):
            likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
            n_pts = min(self.train_z.shape[0], 1024)
            c_model = GPModelDKL(self.train_z[:n_pts, :].cuda(), likelihood=likelihood).cuda()
            c_mll = PredictiveLogLikelihood(c_model.likelihood, c_model, num_data=self.train_z.size(-2))
            c_model = c_model.eval()
            # c_model = self.model.cuda()
            self.c_models.append(c_model)
            self.c_mlls.append(c_mll)
        return self

    def initialize_surrogate_model(self):
        likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
        n_pts = min(self.search_space_data().shape[0], 1024)
        self.model = GPModelDKL(self.search_space_data()[:n_pts, :].cuda(), likelihood=likelihood).cuda()
        self.mll = PredictiveLogLikelihood(
            self.model.likelihood, self.model, num_data=self.search_space_data().size(-2)
        )
        self.model = self.model.eval()
        self.model = self.model.cuda()

        if self.train_c is not None:
            self.initialize_constraint_surrogates()
        else:
            self.c_models = None
            self.c_mlls = None

        return self

    def initialize_xs_to_scores_dict(
        self,
    ):
        # put initial xs and ys in dict to be tracked by objective
        init_xs_to_scores_dict = {}
        for idx, x in enumerate(self.train_x):
            init_xs_to_scores_dict[x] = self.train_y.squeeze()[idx].item()
        self.objective.xs_to_scores_dict = init_xs_to_scores_dict

    def initialize_tr_state(self):
        self.tr_state = TurboStateConstrained(  # initialize turbo state
            dim=self.objective.dim,
            batch_size=self.bsz,
            center_point=None,
            # best_value=None, ## TODO: make sure this is right
            # best_constraint_values=best_constraint_values, ## TODO: make sure this is right
        )

        self.recenter_trs()
        return self

    def initialize_top_k(self):
        """Initialize top k x, y, and zs"""
        # if we have constriants, the top k are those that meet constraints!
        if self.train_c is not None:
            bool_arr = torch.all(self.train_c <= 0, dim=-1)  # all constraint values <= 0
            vaid_train_y = self.train_y[bool_arr]
            valid_train_z = self.train_z[bool_arr]
            valid_train_x = np.array(self.train_x)[bool_arr]
            valid_train_c = self.train_c[bool_arr]
        else:
            vaid_train_y = self.train_y
            valid_train_z = self.train_z
            valid_train_x = self.train_x

        if len(vaid_train_y) > 1:
            self.best_score_seen = torch.max(vaid_train_y)
            self.best_x_seen = valid_train_x[torch.argmax(vaid_train_y.squeeze())]

            # track top k scores found
            self.top_k_scores, top_k_idxs = torch.topk(vaid_train_y.squeeze(), min(self.k, vaid_train_y.shape[0]))
            self.top_k_scores = self.top_k_scores.tolist()
            top_k_idxs = top_k_idxs.tolist()
            self.top_k_xs = [valid_train_x[i] for i in top_k_idxs]
            self.top_k_zs = [valid_train_z[i].unsqueeze(-2) for i in top_k_idxs]
            if self.train_c is not None:
                self.top_k_cs = [valid_train_c[i].unsqueeze(-2) for i in top_k_idxs]
        elif len(vaid_train_y) == 1:
            self.best_score_seen = vaid_train_y.item()
            self.best_x_seen = valid_train_x.item()
            self.top_k_scores = [self.best_score_seen]
            self.top_k_xs = [self.best_x_seen]
            self.top_k_zs = [valid_train_z]
            if self.train_c is not None:
                self.top_k_cs = [valid_train_c]
        else:
            print("No valid init data according to constraint(s)")
            self.best_score_seen = None
            self.best_x_seen = None
            self.top_k_scores = []
            self.top_k_xs = []
            self.top_k_zs = []
            if self.train_c is not None:
                self.top_k_cs = []

    def initialize_diffusion_model(self):
        if self.task == "molecule":
            ckpt_path = "./data/molecule_diffusion.ckpt"
        elif self.task == "peptide":
            ckpt_path = "./data/peptide_diffusion.ckpt"
        else:
            self.diffusion = None
            return self

        self.diffusion = DiffusionModel.load_from_checkpoint(ckpt_path)
        self.diffusion.eval().cuda()
        self.diffusion.freeze()

    def update_next(self, z_next_, y_next_, x_next_, c_next_=None, acquisition=False):
        """Add new points (z_next, y_next, x_next) to train data
        and update progress (top k scores found so far)
        """
        # if no progess made on acqusition, count as a failure
        if (len(x_next_) == 0) and acquisition:
            self.progress_fails_since_last_e2e += 1
            return None

        if c_next_ is not None:
            if len(c_next_.shape) == 1:
                c_next_ = c_next_.unsqueeze(-1)
            valid_points = torch.all(c_next_ <= 0, dim=-1)  # all constraint values <= 0
        else:
            valid_points = torch.tensor([True] * len(y_next_))

        z_next_ = z_next_.detach().cpu()
        if len(z_next_.shape) == 1:
            z_next_ = z_next_.unsqueeze(0)
        y_next_ = y_next_.detach().cpu()
        if len(y_next_.shape) > 1:
            y_next_ = y_next_.squeeze()
        progress = False
        for i, score in enumerate(y_next_):
            self.train_x.append(x_next_[i])
            if valid_points[i]:  # if y is valid according to constraints
                if len(self.top_k_scores) < self.k:
                    # if we don't yet have k top scores, add it to the list
                    self.top_k_scores.append(score.item())
                    self.top_k_xs.append(x_next_[i])
                    self.top_k_zs.append(z_next_[i].unsqueeze(-2))
                    if self.train_c is not None:  # if constrained, update best constraints too
                        self.top_k_cs.append(c_next_[i].unsqueeze(-2))
                elif score.item() > min(self.top_k_scores) and (x_next_[i] not in self.top_k_xs):
                    # if the score is better than the worst score in the top k list, upate the list
                    min_score = min(self.top_k_scores)
                    min_idx = self.top_k_scores.index(min_score)
                    self.top_k_scores[min_idx] = score.item()
                    self.top_k_xs[min_idx] = x_next_[i]
                    self.top_k_zs[min_idx] = z_next_[i].unsqueeze(-2)  # .cuda()
                    if self.train_c is not None:  # if constrained, update best constraints too
                        self.top_k_cs[min_idx] = c_next_[i].unsqueeze(-2)
                # if this is the first valid example we've found, OR if we imporve
                if (self.best_score_seen is None) or (score.item() > self.best_score_seen):
                    self.progress_fails_since_last_e2e = 0
                    progress = True
                    self.best_score_seen = score.item()  # update best
                    self.best_x_seen = x_next_[i]
                    self.new_best_found = True
        if (not progress) and acquisition:  # if no progress msde, increment progress fails
            self.progress_fails_since_last_e2e += 1
        y_next_ = y_next_.unsqueeze(-1)
        if acquisition:
            pass  # TODO: check if this is needed, state already updated during acquisition
        self.train_y = torch.cat((self.train_y, y_next_), dim=-2)
        self.train_z = torch.cat((self.train_z, z_next_), dim=-2)
        if c_next_ is not None:
            self.train_c = torch.cat((self.train_c, c_next_), dim=-2)

        return self

    def update_surrogate_model(self):
        if not self.initial_model_training_complete:
            # first time training surr model --> train on all data
            n_epochs = self.init_n_epochs
            X = self.search_space_data()  # this is just self.train_x
            Y = self.train_y.squeeze(-1)
            # considering constraints
            train_c = self.train_c
        else:
            # otherwise, only train on most recent batch of data
            n_epochs = self.num_update_epochs
            X = self.search_space_data()[-self.num_new_points :]
            Y = self.train_y[-self.num_new_points :].squeeze(-1)
            # considering constraints
            if self.train_c is not None:
                train_c = self.train_c[-self.num_new_points :]
            else:
                train_c = None

        self.model = update_surr_model(self.model, self.mll, self.learning_rte, X, Y, n_epochs)

        # considering constraints
        if self.train_c is not None:
            self.c_models = update_constraint_surr_models(
                self.c_models, self.c_mlls, self.learning_rte, X, train_c, n_epochs
            )

        self.initial_model_training_complete = True

    def update_models_e2e(self):
        """Finetune VAE end to end with surrogate model"""
        self.progress_fails_since_last_e2e = 0
        new_xs = self.train_x[-self.num_new_points :]
        new_ys = self.train_y[-self.num_new_points :].squeeze(-1).tolist()
        train_x = new_xs + self.top_k_xs
        train_y = torch.tensor(new_ys + self.top_k_scores).float()

        c_models = []
        c_mlls = []
        train_c = None

        if self.train_c is not None:
            c_models = self.c_models
            c_mlls = self.c_mlls
            new_cs = self.train_c[-self.num_new_points :]
            # Note: self.top_k_cs is a list of (1, n_cons) tensors
            if len(self.top_k_cs) > 0:
                top_k_cs_tensor = torch.cat(self.top_k_cs, -2).float()
                train_c = torch.cat((new_cs, top_k_cs_tensor), -2).float()
            else:
                train_c = new_cs

        self.objective, self.model = update_models_end_to_end_with_constraints(
            train_x=train_x,
            train_y_scores=train_y,
            objective=self.objective,
            model=self.model,
            mll=self.mll,
            learning_rte=self.learning_rte,
            num_update_epochs=self.num_update_epochs,
            train_c_scores=train_c,
            c_models=c_models,
            c_mlls=c_mlls,
        )
        self.tot_num_e2e_updates += 1

        return self

    def recenter(self):
        """Pass SELFIES strings back through
        VAE to find new locations in the
        new fine-tuned latent space
        """
        self.objective.vae.eval()
        self.model.train()
        optimizer1 = torch.optim.Adam(
            [{"params": self.model.parameters(), "lr": self.learning_rte}], lr=self.learning_rte
        )
        new_xs = self.train_x[-self.num_new_points :]
        train_x = new_xs + self.top_k_xs
        max_string_len = len(max(train_x, key=len))
        # max batch size smaller to avoid memory limit
        #   with longer strings (more tokens)
        bsz = max(1, int(2560 / max_string_len))
        num_batches = math.ceil(len(train_x) / bsz)
        for _ in range(self.num_update_epochs):
            for batch_ix in range(num_batches):
                start_idx, stop_idx = batch_ix * bsz, (batch_ix + 1) * bsz
                batch_list = train_x[start_idx:stop_idx]
                z, _ = self.objective.vae_forward(batch_list)
                out_dict = self.objective(z)
                scores_arr = out_dict["scores"]
                valid_zs = out_dict["valid_zs"]
                selfies_list = out_dict["decoded_xs"]
                constraints_list = out_dict["constr_vals"]
                if len(scores_arr) > 0:  # if some valid scores
                    scores_arr = torch.from_numpy(scores_arr)
                    if self.minimize:
                        scores_arr = scores_arr * -1
                    pred = self.model(valid_zs)
                    loss = -self.mll(pred, scores_arr.cuda())
                    optimizer1.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer1.step()
                    with torch.no_grad():
                        z = z.detach().cpu()
                        self.update_next(z, scores_arr, selfies_list, constraints_list)
            torch.cuda.empty_cache()
        self.model.eval()

    def sample_random_searchspace_points(self, N):
        lb, ub = self.objective.lb, self.objective.ub
        if ub is None:
            ub = self.search_space_data().max()
        if lb is None:
            lb = self.search_space_data().max()
        return torch.rand(N, self.objective.dim) * (ub - lb) + lb

    def is_feasible(self, x):
        current_time = time.time()
        # also make sure candidate satisfies constraints
        if isinstance(x, str):
            seq = [x]
        else:
            seq = x
        # This is already a string, so no need to do vae decode
        # import pdb; pdb.set_trace()
        current_time = time.time()
        cvals = self.objective.compute_constraints(seq)
        # print(f"checking candidate for constraints in {time.time() - current_time}")
        # return false if any of cvals are > 0
        if cvals is not None:
            if cvals.max() > 0:
                return False
        return True

    def randomly_sample_feasible_center(self, max_n_samples=1_000):
        """Rare edge case when we run out of feasible evaluated datapoints
        and must randomly sample to find a new feasible center point
        """
        n_samples = 0
        while True:
            if n_samples > max_n_samples:
                raise RuntimeError(
                    f"Failed to find a feasible tr center after {n_samples} random samples, recommend tring use of smaller M or smaller tau"
                )
            center_point = self.sample_random_searchspace_points(N=1)
            center_x = self.objective.vae_decode(center_point)
            n_samples += 1
            if self.is_feasible(center_x):
                out_dict = self.objective(center_point, center_x)
                center_score = out_dict["scores"].item()
                center_cval = out_dict["constr_vals"].item()
                # add new point to existing data
                self.update_next(
                    z_next_=center_point,
                    y_next_=torch.tensor(center_score).float(),
                    x_next_=center_x,
                    c_next_=torch.tensor(center_cval).float(),
                )
                break

        return center_x, center_point, center_score

    def recenter_trs(self):
        _, top_t_idxs = torch.topk(self.train_y.squeeze(), len(self.train_y))
        train_y_idx_num = 0

        while True:
            # if we run out of feasible points in dataset
            if train_y_idx_num >= len(self.train_y):
                # Randomly sample a new feasible point (rare occurance)
                print("Randomly sample feasible center during recenter trs")
                center_x, center_point, center_score = self.randomly_sample_feasible_center()
                break
            # otherwise, finding highest scoring feassible point in remaining dataset for tr center
            center_idx = top_t_idxs[train_y_idx_num]
            center_score = self.train_y[center_idx].item()
            center_point = self.search_space_data()[center_idx]
            center_x = self.train_x[center_idx]
            train_y_idx_num += 1
            if self.is_feasible(center_x):
                break

        self.tr_state.center_point = center_point
        self.tr_state.best_value = center_score
        self.tr_state.best_x = center_x

    def generate_batch_single_tr(self, tr_state):
        z_next = generate_batch(
            state=tr_state,
            model=self.model,
            X=self.search_space_data(),
            Y=self.train_y,
            batch_size=self.bsz,
            acqf=self.acq_func,
            diffusion=self.diffusion,
            absolute_bounds=(self.objective.lb, self.objective.ub),
            # considering constraints
            constraint_model_list=self.c_models,
            repaint_candidates=self.repaint_candidates,
        )

        with torch.no_grad():
            out_dict = self.objective(z_next)
            self.z_next = out_dict["valid_zs"]
            return out_dict["decoded_xs"]

    def remove_infeasible_candidates(self, x_cands):
        feasible_xs = []
        bool_arr = []
        # print(f"{len(x_cands)} candidates to check")

        # Prepare batch for constraint checking
        if isinstance(x_cands[0], str):
            constraint_batch = x_cands
        else:
            constraint_batch = [x.unsqueeze(0) if isinstance(x, torch.Tensor) else x for x in x_cands]

        # Batch constraint check
        current_time = time.time()
        cvals_batch = self.objective.compute_constraints(constraint_batch)
        # print(f"Batch constraint checking completed in {time.time() - current_time} seconds")

        current_time = time.time()
        candidate_counter = 0
        for i, x_cand in enumerate(x_cands):
            # Check constraints
            if cvals_batch is not None and cvals_batch[i].max() > 0:
                bool_arr.append(False)
            else:
                if isinstance(x_cand, torch.Tensor):
                    feasible_xs.append(x_cand.unsqueeze(0))
                else:
                    feasible_xs.append(x_cand)
                bool_arr.append(True)

            # print(f"Checking candidate {candidate_counter} completed in {time.time() - current_time} seconds")
            candidate_counter += 1
            current_time = time.time()

        bool_arr = np.array(bool_arr)
        return feasible_xs, bool_arr

    def get_feasible_cands(self, x_cands):
        self.feasible_xs, bool_arr = self.remove_infeasible_candidates(
            x_cands=x_cands,
        )
        feasible_searchspace_pts = self.z_next[bool_arr]

        return feasible_searchspace_pts

    def compute_scores_remaining_cands(self, feasible_searchspace_pts):
        out_dict = self.objective(
            feasible_searchspace_pts,
            self.feasible_xs,  # pass in to avoid re-decoding the zs to xs
        )
        feasible_xs = out_dict["decoded_xs"]
        feasible_ys = out_dict["scores"]
        feasible_cs = out_dict["constr_vals"]
        feasible_searchspace_pts = out_dict["valid_zs"]
        self.all_feasible_xs = self.all_feasible_xs + feasible_xs.tolist()
        self.all_feasible_cs = self.all_feasible_cs + feasible_cs.tolist()

        return feasible_ys, feasible_searchspace_pts, feasible_cs

    def update_feasible_candidates_and_tr_state(self, state, feasible_searchspace_pts, feasible_ys, feasible_cs):
        if len(feasible_ys) > 0:
            if type(feasible_searchspace_pts) is np.ndarray:
                feasible_searchspace_pts = torch.from_numpy(feasible_searchspace_pts).float()
            if self.minimize:
                feasible_ys = feasible_ys * -1
            self.all_feasible_ys = self.all_feasible_ys + feasible_ys.tolist()
            feasible_searchspace_pts = feasible_searchspace_pts.detach().cpu()
            self.all_feasible_searchspace_pts = torch.cat((self.all_feasible_searchspace_pts, feasible_searchspace_pts))
            # 4. update state of this tr only on the feasible ys it suggested
            # make sure feasible_ys and feasible_cs are tensors
            feasible_ys = torch.tensor(feasible_ys).float()
            feasible_cs = torch.tensor(feasible_cs).float()
            update_state(state, feasible_ys, feasible_cs)

    def compute_scores_and_update_state(self, state, feasible_searchspace_pts):
        if len(feasible_searchspace_pts) > 0:
            # Compute scores for remaining feasible candiates
            feasible_ys, feasible_searchspace_pts, feasible_cs = self.compute_scores_remaining_cands(
                feasible_searchspace_pts
            )
            # Update tr state on feasible candidates
            self.update_feasible_candidates_and_tr_state(state, feasible_searchspace_pts, feasible_ys, feasible_cs)

    def update_data_all_feasible_points(
        self,
    ):
        self.update_next(
            z_next_=self.all_feasible_searchspace_pts,
            y_next_=torch.tensor(self.all_feasible_ys).float(),
            x_next_=self.all_feasible_xs,
            c_next_=torch.tensor(self.all_feasible_cs).float(),
            acquisition=True,
        )

    def acquisition(self):
        """Generate new candidate points,
        asymetrically discard infeasible ones,
        evaluate them, and update data
        """
        # adding constraint support
        if self.train_c is not None:  # if constrained
            constraint_model_list = self.c_models
        else:
            constraint_model_list = None

        self.all_feasible_xs = []  # (used only by LOL-ROBOT when searchspace pts != xs)
        self.all_feasible_ys = []
        self.all_feasible_cs = []
        self.all_feasible_searchspace_pts = torch.tensor([])
        import time

        current_time = time.time()
        counter = 0

        # print(f"starting tr {counter}")
        # 1. Generate a batch of candidates in
        #   trust region using global surrogate model
        x_next = self.generate_batch_single_tr(self.tr_state)
        # print(f"generated batch {counter} in {time.time() - current_time}")
        current_time = time.time()

        # 2. Asymetrically remove infeasible candidates
        feasible_searchspace_pts = self.get_feasible_cands(x_next)
        # print(f"got feasible cands {counter} in {time.time() - current_time}")
        current_time = time.time()

        # 3. Compute scores for feassible cands and update tr statee
        self.compute_scores_and_update_state(self.tr_state, feasible_searchspace_pts)
        # print(f"computed scores and updated state {counter} in {time.time() - current_time}")
        current_time = time.time()

        counter += 1

        # 4. Add all new evaluated points to dataset (update_next)
        if len(self.all_feasible_searchspace_pts) != 0:
            self.num_new_points = len(self.all_feasible_ys)
        self.update_data_all_feasible_points()
