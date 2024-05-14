import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset


class DiffusionDataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size: int, num_workers: int) -> None:
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self) -> DataLoader:
        train_data = LatentDataset(self.data_dir)
        return DataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class LatentDataset(Dataset):
    def __init__(self, data_dir: str) -> None:
        data = torch.load(f"{data_dir}/latent.pt")

        self.data = {
            1: (data[0], data[1]),
            2: (data[2], data[3]),
            3: (data[4], data[5]),
            4: (data[6], data[7]),
        }

    def __len__(self) -> int:
        return len(self.data[1][0])

    def get_single(self, col: int, idx: int):
        loc, scale = self.data[col]
        return loc[idx] + torch.randn_like(loc[idx]) * scale[idx]

    def __getitem__(self, idx: int) -> dict:
        c1 = self.get_single(1, idx)
        c2 = self.get_single(2, idx)
        c3 = self.get_single(3, idx)
        c4 = self.get_single(4, idx)

        return torch.vstack([c1, c2, c3, c4])
