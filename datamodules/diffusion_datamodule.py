import random

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset


class DiffusionDataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size: int, num_workers: int, dataset: Dataset = None) -> None:
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = dataset if dataset else LatentDataset

    def train_dataloader(self) -> DataLoader:
        train_data = self.dataset(self.data_dir, "train")
        return DataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            # pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        val_data = self.dataset(self.data_dir, "val")
        return DataLoader(
            val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            # pin_memory=True,
            drop_last=True,
        )
    
    def test_dataloader(self) -> DataLoader:
        test_data = self.dataset(self.data_dir, "test")
        return DataLoader(
            test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            # pin_memory=True,
            drop_last=True,
        )


class LatentDataset(Dataset):
    def __init__(self, data_dir: str, split: str) -> None:
        self.m, self.s = torch.load(f"{data_dir}/low_all_{split}.pt")

    def __len__(self) -> int:
        return len(self.m)

    def __getitem__(self, idx: int):
        z = self.m[idx] + self.s[idx] * torch.randn_like(self.m[idx])
        return z

class LatentDatasetClassifier(Dataset):
    def __init__(self, data_dir: str, split: str) -> None:
        self.m, self.s, self.cls, _, _ = torch.load(f"{data_dir}/low_all_{split}.pt")

    def __len__(self) -> int:
        return len(self.m)

    def __getitem__(self, idx: int):
        z = self.m[idx] + self.s[idx] * torch.randn_like(self.m[idx])
        cls = self.cls[idx]
        return z, cls

class LatentDatasetDescriptors(Dataset):
    def __init__(self, data_dir: str, split: str) -> None:
        self.m, self.s, self.cls, self.qed, self.fsp3 = torch.load(f"{data_dir}/low_all_{split}.pt")

    def __len__(self) -> int:
        return len(self.m)

    def __getitem__(self, idx: int):
        z = self.m[idx] + self.s[idx] * torch.randn_like(self.m[idx])
        cls, qed, fsp3 = self.cls[idx], self.qed[idx], self.fsp3[idx]
        return z, cls, qed, fsp3


class LatentDatasetCond(Dataset):
    def __init__(self, data_dir: str) -> None:
        lm, ls = torch.load(f"{data_dir}/low.pt")
        hm, hs = torch.load(f"{data_dir}/high.pt")

        self.m = torch.cat(
            [
                lm,
                hm,
            ],
            dim=0,
        )
        self.s = torch.cat(
            [
                ls,
                hs,
            ],
            dim=0,
        )
        self.class_idx = torch.cat(
            [
                torch.zeros(len(lm)),
                torch.ones(len(hm)),
            ],
            dim=0,
        ).long()

    def __len__(self) -> int:
        return len(self.m)

    def __getitem__(self, idx: int):
        z = self.m[idx] + self.s[idx] * torch.randn_like(self.m[idx])
        return z, self.class_idx[idx]


class LatentDatasetOverSample(Dataset):
    def __init__(self, data_dir: str) -> None:
        self.lm, self.ls = torch.load(f"{data_dir}/low.pt")
        self.hm, self.hs = torch.load(f"{data_dir}/high.pt")

    def __len__(self) -> int:
        return len(self.lm) + len(self.hm)

    def __getitem__(self, idx: int):
        if random.random() < 0.5:
            idx = random.randint(0, len(self.lm) - 1)
            class_idx = 0
            mu = self.lm[idx]
            sigma = self.ls[idx]
        else:
            idx = random.randint(0, len(self.hm) - 1)
            class_idx = 1
            mu = self.hm[idx]
            sigma = self.hs[idx]

        z = mu + sigma * torch.randn_like(mu)
        return z, torch.tensor(class_idx)
