import numpy as np
import torch 
import sys 
import random
from .tasks.objective_functions import OBJECTIVE_FUNCTIONS_DICT 
from .tasks.diversity_functions import DIVERSITY_FUNCTIONS_DICT 
from model.diffusion_v2 import BaseVAE

class PeptideObjectiveV2:

    def __init__(
        self,
        task_id='example', # id of objective funciton you want to maximize 
        task_specific_args=[],
        divf_id="edit_dist",
        path_to_vae_statedict="data/peptide_vae.ckpt",
        xs_to_scores_dict={},
        max_string_length=1024,
        num_calls=0,
        lb=None,
        ub=None,
    ):
        self.dim                    = 256
        self.path_to_vae_statedict  = path_to_vae_statedict # path to trained vae stat dict
        self.task_specific_args     = task_specific_args
        self.max_string_length      = max_string_length # max string length that VAE can generate
        self.divf_id                = divf_id # specify which diversity function to use with string id 
        assert task_id in OBJECTIVE_FUNCTIONS_DICT 
        self.objective_function = OBJECTIVE_FUNCTIONS_DICT[task_id](*self.task_specific_args)
        
        # dict used to track xs and scores (ys) queried during optimization
        self.xs_to_scores_dict = xs_to_scores_dict 
        # track total number of times the oracle has been called
        self.num_calls = num_calls
        # string id for optimization task, often used by oracle
        #   to differentiate between similar tasks (ie for guacamol)
        self.task_id = task_id
        # absolute upper and lower bounds on search space
        self.lb = lb
        self.ub = ub  

        # load in pretrained VAE, store in variable self.vae
        self.vae = None
        self.initialize_vae()
        assert self.vae is not None

    def __call__(self, z, decoded_xs=None):
        ''' Input 
                z: a numpy array or pytorch tensor of latent space points
                decoded_xs: option to pass in list of decoded xs for efficiency if the zs have already been decoded
            Output
                out_dict['valid_zs'] = the zs which decoded to valid xs 
                out_dict['decoded_xs'] = an array of valid xs obtained from input zs
                out_dict['scores']: an array of valid scores obtained from input zs
        '''
        if type(z) is np.ndarray: 
            z = torch.from_numpy(z).float()
        # if no decoded xs passed in, we decode the zs to get xs
        if decoded_xs is None: 
            decoded_xs = self.vae_decode(z)

        out_dict = self.xs_to_valid_scores(decoded_xs)
        valid_zs = z[out_dict['bool_arr']] 
        out_dict['valid_zs'] = valid_zs
        # get valid constraint values for valid decoded xs

        return out_dict
    
    def xs_to_valid_scores(self, xs):
        scores = []
        for idx, x in enumerate(xs):
            # if we have already computed the score, don't 
            #   re-compute (don't call oracle unnecessarily)
            if x in self.xs_to_scores_dict:
                score = self.xs_to_scores_dict[x]
            else: # otherwise call the oracle to get score
                score = self.query_oracle(x)
                # add score to dict so we don't have to
                #   compute it again if we get the same input x
                self.xs_to_scores_dict[x] = score
                # track number of oracle calls 
                #   nan scores happen when we pass an invalid
                #   peptide string and thus avoid calling the
                #   oracle entirely
                if np.logical_not(np.isnan(score)):
                    self.num_calls += 1
            scores.append(score)
        scores_arr = np.array(scores)
        if type(xs) is list: 
            xs = np.array(xs) 
        elif type(xs) is torch.Tensor:
            xs = xs.detach().cpu().numpy() 
        # get valid zs, xs, and scores
        bool_arr = np.logical_not(np.isnan(scores_arr)) 
        valid_xs = xs[bool_arr]
        valid_scores = scores_arr[bool_arr]
        out_dict = {}
        out_dict['scores'] = valid_scores
        out_dict['bool_arr'] = bool_arr
        out_dict['decoded_xs'] = valid_xs

        return out_dict

    def vae_decode(self, z):
        """Input
            z: a tensor latent space points
        Output
            a corresponding list of the decoded input space
            items output by vae decoder
        """
        if type(z) is np.ndarray:
            z = torch.from_numpy(z).float()
        z = z.cuda()
        self.vae = self.vae.eval()
        self.vae = self.vae.cuda()

        tokens = self.vae.sample(z, argmax=False, max_len=self.max_string_length)
        decoded_seqs = self.vae.detokenize(tokens)

        return decoded_seqs

    def query_oracle(self, x):
        ''' Input: 
                a single input space item x
            Output:
                method queries the oracle and returns 
                the corresponding score y,
                or np.nan in the case that x is an invalid input
        '''
        scores_list = self.objective_function([x])
        return scores_list[0]

    def initialize_vae(self):
        ''' Sets self.vae to the desired pretrained vae and 
            sets self.dataobj to the corresponding data class 
            used to tokenize inputs, etc. '''
        self.dataobj = None
        self.vae = BaseVAE.load_from_checkpoint(self.path_to_vae_statedict)
        self.vae = self.vae.cuda()
        self.vae = self.vae.eval()

    def vae_forward(self, xs_batch):
        """Input:
            a list xs
        Output:
            z: tensor of resultant latent space codes
                obtained by passing the xs through the encoder
            vae_loss: the total loss of a full forward pass
                of the batch of xs through the vae
                (ie reconstruction error)
        """
        # assumes xs_batch is a batch of smiles strings
        tokens = self.vae.tokenize(xs_batch)
        out_dict = self.vae(tokens.cuda())
        return out_dict['z'].flatten(1), out_dict['loss']

    def divf(self, x1, x2):
        ''' Compute diversity function between two 
            potential xs/ sequences so we can
            create a diverse set of optimal solutions
            with some minimum diversity between eachother
        '''
        return DIVERSITY_FUNCTIONS_DICT[self.divf_id](x1, x2) 
