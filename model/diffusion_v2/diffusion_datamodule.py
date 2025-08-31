from typing import Literal

import datasets
import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset

DataType = Literal["molecule", "peptide"]


class DiffusionDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_type: DataType,
        batch_size: int,
        num_workers: int,
    ) -> None:
        super().__init__()

        self.data_type = data_type
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dataset = datasets.DatasetDict(
            {
                "train": datasets.Dataset.from_parquet(f"./data/latents/{data_type}_train.parquet"),
                "val": datasets.Dataset.from_parquet(f"./data/latents/{data_type}_val.parquet"),
                "test": datasets.Dataset.from_parquet(f"./data/latents/{data_type}_test.parquet"),
            }  # type: ignore
        )

    def train_dataloader(self) -> DataLoader:
        train_data = LatentDataset(self.dataset["train"])
        return DataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        val_data = LatentDataset(self.dataset["val"])
        return DataLoader(
            val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        test_data = LatentDataset(self.dataset["test"])
        return DataLoader(
            test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class LatentDataset(Dataset):
    def __init__(self, data: datasets.Dataset) -> None:
        self.data = data
        self.data.set_format("torch", columns=["mu", "sigma"])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data[idx]
        mu = row["mu"]
        sigma = row["sigma"]
        z = mu + sigma * torch.randn_like(mu)
        return z

    def __getitems__(self, idxs: int):
        rows = self.data[idxs]
        mus = rows["mu"]
        sigmas = rows["sigma"]
        zs = mus + sigmas * torch.randn_like(mus)
        return (zs,)
