import torch
import torch.nn as nn


class ConvScoreClassifier(nn.Module):
    """
    Input:  z of shape (B, 8, 16)
    Output: logits of shape (B, 2)  (raw, unnormalized)
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),  # 2 classes -> logits[:, 0], logits[:, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z expected: (B, 8, 16)
        if z.dim() != 3 or z.size(1) != 8 or z.size(2) != 16:
            raise ValueError(f"Expected z of shape (B, 8, 16); got {tuple(z.shape)}")
        x = z.unsqueeze(1)  # (B, 1, 8, 16)
        return self.net(x)  # (B, 2)

    @torch.no_grad()
    def prob_class1(self, z: torch.Tensor) -> torch.Tensor:
        """Convenience: returns P(y=1|z) with shape (B,)"""
        return torch.softmax(self.forward(z), dim=1)[:, 1]
