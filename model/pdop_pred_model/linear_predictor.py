import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearScorePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):  # x: (B, 16, 8)
        x = self.net(x)         # (B, 16, 64)
        x = x.mean(dim=1)       # (B, 64)
        return self.head(x).squeeze(-1)  # (B,)