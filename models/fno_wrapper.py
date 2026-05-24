import torch.nn as nn
from neuralop.models.fno import FNO


class FNOWrapper(nn.Module):
    def __init__(self, n_modes, in_channels, out_channels, n_layers,
                 hidden_channels, **kwargs):
        super().__init__()
        self.out_channels = out_channels
        self.fno = FNO(
            n_modes=tuple(n_modes),
            in_channels=in_channels,
            out_channels=out_channels,
            n_layers=n_layers,
            hidden_channels=hidden_channels,
        )

    def forward(self, x):
        return self.fno(x) + x[:, :self.out_channels]
