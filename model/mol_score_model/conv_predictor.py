import torch.nn as nn


class ConvScorePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # (B, 32, 16, 8)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # (B, 64, 16, 8)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # (B, 64, 1, 1)
            nn.Flatten(),  # (B, 64)
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # output ∈ [0, 1]
        )

    def forward(self, x):  # x: (B, 16, 8)
        x = x.unsqueeze(1)  # -> (B, 1, 16, 8)
        return self.net(x).squeeze(-1)  # -> (B,)
