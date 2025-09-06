import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F 
import pandas as pd
import itertools
import os 
import numpy as np


class DataModuleKmers(pl.LightningDataModule):
    def __init__(self, batch_size, k, version=1, load_data=True ): 
        super().__init__() 
        self.batch_size = batch_size 
        if version == 1: DatasetClass = DatasetKmers
        else: raise RuntimeError('Invalid data version') 
        self.train  = DatasetClass(dataset='train', k=k, load_data=load_data) 
        self.val    = DatasetClass(dataset='val', k=k, vocab=self.train.vocab, vocab2idx=self.train.vocab2idx, load_data=load_data )
        self.test   = DatasetClass(dataset='test', k=k, vocab=self.train.vocab, vocab2idx=self.train.vocab2idx, load_data=load_data )
    
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, pin_memory=True, shuffle=True, collate_fn=collate_fn, num_workers=10)

    def val_dataloader(self):
        return DataLoader(self.val,   batch_size=self.batch_size, pin_memory=True, shuffle=False, collate_fn=collate_fn, num_workers=10)
    
    def test_dataloader(self):
        return DataLoader(self.test,   batch_size=self.batch_size, pin_memory=True, shuffle=False, collate_fn=collate_fn, num_workers=10)


class DatasetKmers(Dataset): # asssuming train data 
    def __init__(self, dataset='train', data_path=None, k=3, vocab=None, vocab2idx=None, load_data=False):
        self.dataset = dataset
        self.k = k 
        vocab_path = f"data/apex/{k}mer_vocab.csv"
        if data_path is None: 
            path_to_data = "data/apex/uniref-small.csv"
        if (vocab is None) and os.path.exists(vocab_path):
                vocab = pd.read_csv(vocab_path, header=None ).values.squeeze().tolist() 

        if (vocab is None) or load_data:
            df = pd.read_csv(path_to_data )
            train_seqs = df['sequence'].values  # 1_500_000  sequences 
            # SEQUENCE LENGTHS ANALYSIS:  Max = 299, Min = 100, Mean = 183.03 
            regular_data = [] 
            for seq in train_seqs: 
                regular_data.append([token for token in seq]) # list of tokens
        
        # first get initial vocab set 
        if vocab is None:
            self.regular_vocab = set((token for seq in regular_data for token in seq))  # 21 tokens 
            self.regular_vocab.discard(".") 
            if '-' not in self.regular_vocab: 
                self.regular_vocab.add('-')  # '-' used as pad token when length of sequence is not a multiple of k
            self.vocab = ["".join(kmer) for kmer in itertools.product(self.regular_vocab, repeat=k)] # 21**k tokens 
            self.vocab = ['<start>', '<stop>', *sorted(list(self.vocab))] # 21**k + 2 tokens 

            # save vocab for next time... 
            vocab_arr = np.array(self.vocab)
            df = pd.DataFrame(vocab_arr)
            df.to_csv(vocab_path, header=None, index=None)
        else: 
            self.vocab = vocab 

        if vocab2idx is None:
            self.vocab2idx = { v:i for i, v in enumerate(self.vocab) }
        else:
            self.vocab2idx = vocab2idx
        
        self.data = []
        if load_data:
            for seq in regular_data:
                token_num = 0
                kmer_tokens = []
                while token_num < len(seq):
                    kmer = seq[token_num:token_num+k]
                    while len(kmer) < k:
                        kmer += '-' # padd so we always have length k 
                    kmer_tokens.append("".join(kmer)) 
                    token_num += k 
                self.data.append(kmer_tokens) 
        
        num_data = len(self.data) 
        ten_percent = int(num_data/10) 
        five_percent = int(num_data/20) 
        if self.dataset == 'train': # 90 %
            self.data = self.data[0:-ten_percent] 
        elif self.dataset == 'val': # 5 %
            self.data = self.data[-ten_percent:-five_percent] 
        elif self.dataset == 'test': # 5 %
            self.data = self.data[-five_percent:] 
        else: 
            raise RuntimeError("dataset must be one of train, val, test")


    def tokenize_sequence(self, list_of_sequences):   
        ''' 
        Input: list of sequences in standard form (ie 'AGYTVRSGCMGA...')
        Output: List of tokenized sequences where each tokenied sequence is a list of kmers
        '''
        tokenized_sequences = []
        for seq in list_of_sequences:
            token_num = 0
            kmer_tokens = []
            while token_num < len(seq):
                kmer = seq[token_num:token_num + self.k]
                while len(kmer) < self.k:
                    kmer += '-' # padd so we always have length k  
                if type(kmer) == list: kmer = "".join(kmer)
                kmer_tokens.append(kmer) 
                token_num += self.k 
            tokenized_sequences.append(kmer_tokens) 
        return tokenized_sequences 

    def encode(self, tokenized_sequence):
        return torch.tensor([self.vocab2idx[s] for s in [*tokenized_sequence, '<stop>']])

    def decode(self, tokens):
        '''
        Inpput: Iterable of tokens specifying each kmer in a given protien (ie [3085, 8271, 2701, 2686, ...] )
        Output: decoded protien string (ie GYTVRSGCMGA...)
        '''
        dec = [self.vocab[t] for t in tokens]
        # Chop out start token and everything past (and including) first stop token
        stop = dec.index("<stop>") if "<stop>" in dec else None # want first stop token
        protien = dec[0:stop] # cut off stop tokens
        while "<start>" in protien: # start at last start token (I've seen one case where it started w/ 2 start tokens)
            start = (1+dec.index("<start>")) 
            protien = protien[start:]
        protien = "".join(protien) # combine into single string 

        # Remove chars incompatible with ESM vocab
        protien = protien.replace("X", "")
        protien = protien.replace("-", "") 
        protien = protien.replace("U", "") 
        protien = protien.replace("X", "") 
        protien = protien.replace("Z", "") 
        protien = protien.replace("O", "") 
        protien = protien.replace("B", "")

        # Catch empty seq case 
        if len(protien) == 0:
            protien = "AAA" 

        return protien


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.encode(self.data[idx]) 

    @property
    def vocab_size(self):
        return len(self.vocab)


def collate_fn(data):
    # Length of longest peptide in batch 
    max_size = max([x.shape[-1] for x in data])
    return torch.vstack(
        # Pad with stop token
        [F.pad(x, (0, max_size - x.shape[-1]), value=1) for x in data]
    )


# === DATA LOADING ===

import numpy as np
import pandas as pd
import torch
import math


def load_peptide_train_data(
    score_version,
    path_to_vae_statedict,
    num_initialization_points=10_000,
):
    filename_seqs = f"data/apex/init_seqs.csv"
    df = pd.read_csv(filename_seqs, header=None)
    train_x_seqs = df.values.squeeze().tolist()
    filename_scores = f"data/apex/{score_version}_scores.csv"
    df = pd.read_csv(filename_scores, header=None)
    train_y = torch.from_numpy(df.values).float()
    
    num_initialization_points = min(num_initialization_points, len(train_x_seqs))
    train_z = load_train_z(num_initialization_points, path_to_vae_statedict)
    init_train_x = train_x_seqs[0:num_initialization_points]
    train_y = train_y[0:num_initialization_points]
    init_train_y = train_y #.unsqueeze(-1)
    
    return init_train_x, train_z, init_train_y 


def load_train_z(
    num_initialization_points,
    path_to_vae_statedict,
):
    state_dict_file_type = path_to_vae_statedict.split('.')[-1] # usually .pt or .ckpt
    path_to_init_train_zs = path_to_vae_statedict.replace(f".{state_dict_file_type}", '-train-zs.csv')
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

def compute_peptide_train_zs(pep_objective, init_train_x, bsz=64):
        init_zs = []
        # make sure vae is in eval mode 
        pep_objective.vae.eval() 
        n_batches = math.ceil(len(init_train_x)/bsz)
        for i in range(n_batches):
            xs_batch = init_train_x[i*bsz:(i+1)*bsz] 
            zs, _ = pep_objective.vae_forward(xs_batch)
            init_zs.append(zs.detach().cpu())
        init_zs = torch.cat(init_zs, dim=0)
        # now save the zs so we don't have to recompute them in the future:
        state_dict_file_type = pep_objective.path_to_vae_statedict.split('.')[-1] # usually .pt or .ckpt
        path_to_init_train_zs = pep_objective.path_to_vae_statedict.replace(f".{state_dict_file_type}", '-train-zs.csv')
        zs_arr = init_zs.cpu().detach().numpy()
        pd.DataFrame(zs_arr).to_csv(path_to_init_train_zs, header=None, index=None) 

        return init_zs