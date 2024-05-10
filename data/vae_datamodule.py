import json

import lightning as L
from torch.utils.data import DataLoader


class VAEDataModule(L.LightningDataModule):
    def __init__(self, batch_size: int, num_workers: int) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.vocab = json.load(open("./data/vocab.json", "r"))

    def train_dataloader(self) -> DataLoader:
        # Load the training dataset
        ds = None

        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        # Load the val dataset
        ds = None

        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
