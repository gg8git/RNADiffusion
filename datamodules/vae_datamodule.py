import json
import os

import lightning as L
import torch
from torch.utils.data import DataLoader


class VAEDataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size: int, num_workers: int) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.vocab = json.load(open("./data/selfies/vocab.json", "r"))

    def train_dataloader(self) -> DataLoader:
        train_data = torch.load(os.path.join(self.data_dir, "train.pt")).long()
        return DataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        val_data = torch.load(os.path.join(self.data_dir, "val.pt")).long()
        return DataLoader(
            val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
