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
        vocab_path="data/selfies_vocab.json",
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
    def __init__(self, data_root: str, split: str, vocab: dict[str, int] = None, vocab_path: str = None) -> None:
        super().__init__()

        self.data_root = data_root
        self.split = split

        assert (not (vocab is None and vocab_path is None))
        if vocab is None:
            with open(vocab_path) as f:
                vocab = json.load(f)

        self.vocab = vocab.keys()
        self.vocab2idx = vocab

        path = Path(data_root) / f'{split}_selfie.gz'

        with gzip.open(path, 'rt') as f:
            self.selfies = [l.strip() for l in f.readlines()]

    def __len__(self) -> int:
        return len(self.selfies)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.encode(self.data[index])
    
    def tokenize_selfies(self, selfies_list):
        tokenized_selfies = []
        for selfie in selfies_list:
            tokenized_selfies.append(fast_split(selfie))
        return tokenized_selfies

    def encode(self, selfie):
        selfie =  f"[start]{selfie}[stop]"

        tokens = fast_split(selfie)
        tokens = torch.tensor([self.vocab2idx[tok] for tok in tokens])

        return tokens
    
    def decode(self, tokens):
        try :
            selfie = sf.encoding_to_selfies(tokens.squeeze().tolist(), {v:k for k,v in self.vocab2idx.items()}, 'label')
        except:
            selfie = '[C]'
        
        if '[stop]' in selfie:
            selfie = selfie[:selfie.find('[stop]')]
        if '[pad]' in selfie:
            selfie = selfie[:selfie.find('[pad]')]
        if '[start]' in selfie:
            selfie = selfie[selfie.find('[start]') + len('[start]'):]
        return selfie
    
    def get_collate_fn(self):
        def collate(batch: List[torch.Tensor]) -> torch.Tensor:
            return pad_sequence(batch, batch_first=True, padding_value=self.vocab2idx['[pad]'])
        return collate

# Faster than sf.split_selfies because it doesn't check for invalid selfies
def fast_split(selfie: str) -> list[str]:
    return [f"[{tok}" for tok in selfie.split("[") if tok]
