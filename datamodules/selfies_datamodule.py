import json
import gzip
from pathlib import Path
from typing import List

import torch
import selfies as sf
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


class SELFIESDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size,
        data_root_path,
        vocab_path="data/selfies/selfies_vocab.json",
    ):
        super().__init__()

        with open(vocab_path) as f:
            vocab = json.load(f)

        self.batch_size = batch_size
        self.train = SELFIESDataset(data_root=data_root_path, split="train", vocab=vocab)
        self.val = SELFIESDataset(data_root=data_root_path, split="val", vocab=vocab)

        self.val.vocab = self.train.vocab
        self.val.vocab2idx = self.train.vocab2idx

        # Drop data from val that we have no tokens for
        self.val.data = [smile for smile in self.val.data if False not in [tok in self.train.vocab for tok in smile]]

    def train_dataloader(self):
        return DataLoader(
            self.train, batch_size=self.batch_size, pin_memory=True, shuffle=True, collate_fn=self.train.get_collate_fn, num_workers=10
        )

    def val_dataloader(self):
        return DataLoader(
            self.val, batch_size=self.batch_size, pin_memory=True, shuffle=False, collate_fn=self.val.get_collate_fn, num_workers=10
        )


class SELFIESDataset(Dataset):
    def __init__(
        self,
        fname=None,
        load_data=False,
    ):
        self.data = []
        if load_data:
            assert fname is not None
            with gzip.open(fname, 'rt') as f:
                selfie_strings = [x.strip() for x in f.readlines()]
            for string in selfie_strings:
                self.data.append(list(sf.split_selfies(string)))

        with open("data/selfies/selfies_vocab.json") as f:
            self.vocab2idx = json.load(f)
        self.vocab = self.vocab2idx.keys()

    def tokenize_selfies(self, selfies_list):
        # tokenized_selfies = []
        # for string in selfies_list:
        #     tokenized_selfies.append(list(sf.split_selfies(string)))
        # return tokenized_selfies
        return selfies_list

    def encode(self, selfie):
        selfie = f"[start]{selfie}[stop]"
        enc = sf.selfies_to_encoding(selfie, self.vocab2idx, enc_type='label')
        return torch.tensor(enc, dtype=torch.long)

    def decode(self, tokens, drop_after_stop=True):
        try :
            selfie = sf.encoding_to_selfies(tokens.squeeze().tolist(), {v:k for k,v in self.vocab2idx.items()}, 'label')
        except:
            selfie = '[C]'
        
        if drop_after_stop and '[stop]' in selfie:
            selfie = selfie[:selfie.find('[stop]')]
        if '[pad]' in selfie:
            selfie = selfie[:selfie.find('[pad]')]
        if '[start]' in selfie:
            selfie = selfie[selfie.find('[start]') + len('[start]'):]
        return selfie

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.encode(self.data[idx])

    @property
    def vocab_size(self):
        return len(self.vocab)
    
    def get_collate_fn(self, fix_len=None):
        if fix_len and isinstance(fix_len, int):
            def collate(batch: List[torch.Tensor]) -> torch.Tensor:
                batch = [x.squeeze(0) for x in batch]
                padded_batch = []
                for x in batch:
                    length = x.size(0)
                    if length < fix_len:
                        # Pad to the right
                        pad_size = fix_len - length
                        padded_x = F.pad(x, (0, pad_size), value=self.vocab2idx['[pad]'])
                    else:
                        # Truncate
                        padded_x = x[:fix_len]
                    padded_batch.append(padded_x)
                
                return torch.stack(padded_batch, dim=0)
        else:
            def collate(batch: List[torch.Tensor]) -> torch.Tensor:
                batch = [x.squeeze(0) for x in batch]
                return pad_sequence(batch, batch_first=True, padding_value=self.vocab2idx['[pad]'])
        return collate
