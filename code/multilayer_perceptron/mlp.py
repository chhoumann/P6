import torch.nn as nn


class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            # nn.Flatten(),
            # in_features, out_features, bias=True, device=None, dtype=None
            nn.Linear(3, 6, dtype=float),
            nn.ReLU(),
            nn.Linear(6, 6, dtype=float),
            nn.ReLU(),
            nn.Linear(6, 6, dtype=float),
            nn.ReLU(),
            nn.Linear(6, 6, dtype=float),
            nn.ReLU(),
            nn.Linear(6, 6, dtype=float),
            nn.ReLU(),
            nn.Linear(6, 6, dtype=float),
            nn.ReLU(),
            nn.Linear(6, 1, dtype=float)
        )

    def forward(self, x):
        return self.layers(x)
