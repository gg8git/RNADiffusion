import numpy as np
import selfies as sf
import torch

from utils.guacamol_utils import GUACAMOL_TASK_NAMES, smiles_to_desired_scores
from model.diffusion_v2 import BaseVAE


class MoleculeObjectiveV2:
    """MoleculeObjective class supports all molecule optimization
    tasks and uses the SELFIES VAE by default"""

    def __init__(
        self,
        task_id="pdop",
        path_to_vae_statedict="data/molecule_vae.ckpt",
        xs_to_scores_dict={},
        max_string_length=1024,
        num_calls=0,
        smiles_to_selfies={},
    ):
        assert task_id in GUACAMOL_TASK_NAMES + ["logp"]

        self.dim = 128
        self.path_to_vae_statedict = path_to_vae_statedict  # path to trained vae stat dict
        self.max_string_length = max_string_length  # max string length that VAE can generate
        self.smiles_to_selfies = smiles_to_selfies  # dict to hold computed mappings form smiles to selfies strings

        # dict used to track xs and scores (ys) queried during optimization
        self.xs_to_scores_dict = xs_to_scores_dict

        # track total number of times the oracle has been called
        self.num_calls = num_calls

        # string id for optimization task, often used by oracle
        #   to differentiate between similar tasks (ie for guacamol)
        self.task_id = task_id

        # load in pretrained VAE, store in variable self.vae
        self.vae = None
        self.initialize_vae()
        assert self.vae is not None

    def __call__(self, z):
        """Input
            z: a numpy array or pytorch tensor of latent space points
        Output
            out_dict['valid_zs'] = the zs which decoded to valid xs
            out_dict['decoded_xs'] = an array of valid xs obtained from input zs
            out_dict['scores']: an array of valid scores obtained from input zs
        """
        if type(z) is np.ndarray:
            z = torch.from_numpy(z).float()
        decoded_xs = self.vae_decode(z)
        scores = []
        for x in decoded_xs:
            # if we have already computed the score, don't
            #   re-compute (don't call oracle unnecessarily)
            if x in self.xs_to_scores_dict:
                score = self.xs_to_scores_dict[x]
            else:  # otherwise call the oracle to get score
                score = self.query_oracle(x)
                # add score to dict so we don't have to
                #   compute it again if we get the same input x
                self.xs_to_scores_dict[x] = score
                # track number of oracle calls
                #   nan scores happen when we pass an invalid
                #   molecular string and thus avoid calling the
                #   oracle entirely
                if np.logical_not(np.isnan(score)):
                    self.num_calls += 1
            scores.append(score)

        scores_arr = np.array(scores)
        decoded_xs = np.array(decoded_xs)
        # get valid zs, xs, and scores
        bool_arr = np.logical_not(np.isnan(scores_arr))
        decoded_xs = decoded_xs[bool_arr]
        scores_arr = scores_arr[bool_arr]
        valid_zs = z[bool_arr]

        out_dict = {}
        out_dict["scores"] = scores_arr
        out_dict["valid_zs"] = valid_zs
        out_dict["decoded_xs"] = decoded_xs
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

        tokens = self.vae.sample(z)
        decoded_smiles = self.vae.detokenize(tokens)

        return decoded_smiles

    def query_oracle(self, x):
        """Input:
            a single input space item x
        Output:
            method queries the oracle and returns
            the corresponding score y,
            or np.nan in the case that x is an invalid input
        """
        # method assumes x is a single smiles string
        score = smiles_to_desired_scores([x], self.task_id).item()

        return score

    def initialize_vae(self):
        """Sets self.vae to the desired pretrained vae and
        sets self.dataobj to the corresponding data class
        used to tokenize inputs, etc."""
        self.dataobj = None
        self.vae = BaseVAE.load_from_checkpoint(self.path_to_vae_statedict)
        self.vae = self.vae.cuda()
        self.vae = self.vae.eval()
        # set max string length that VAE can generate
        self.vae.max_string_length = self.max_string_length

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


if __name__ == "__main__":
    # testing molecule objective
    obj1 = MoleculeObjectiveV2(task_id="pdop")
    print(obj1.num_calls)
    dict1 = obj1(torch.randn(10, 256))
    print(dict1["scores"], obj1.num_calls)
    dict1 = obj1(torch.randn(3, 256))
    print(dict1["scores"], obj1.num_calls)
    dict1 = obj1(torch.randn(1, 256))
    print(dict1["scores"], obj1.num_calls)
