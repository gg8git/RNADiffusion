import json, os
import lightning as L
import torch
from torch.utils.data import DataLoader, TensorDataset


class VAEDataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size: int, num_workers: int) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.vocab = json.load(open("./data/vocab.json", "r"))

    def train_dataloader(self) -> DataLoader:
        # Load the training dataset
        train_file = os.path.join(self.data_dir, "train.tsv")
        print(f"Loading training data from {train_file}")
        train_data = []
        with open(train_file) as f:
            for line in f:
                token = [self.vocab['[START]']] + [self.vocab.get(c, self.vocab['[UNK]']) for c in line.strip()] + [self.vocab['[STOP]']]
                train_data.append(token)
        train_data = torch.tensor(train_data, dtype=torch.long)
        ds = TensorDataset(train_data)
        print("Data Loaded")
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        # Load the val dataset
        val_file = os.path.join(self.data_dir, "test.tsv")
        val_data = []
        print(f"Loading validation data from {val_file}")
        with open(val_file) as f:
            for line in f:
                token = [self.vocab['[START]']] + [self.vocab.get(c, self.vocab['[UNK]']) for c in line.strip()] + [self.vocab['[STOP]']]
                val_data.append(token)
        val_data = torch.tensor(val_data, dtype=torch.long)
        ds = TensorDataset(val_data)
        print("Data Loaded")
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
