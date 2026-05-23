import torch
import torch.nn as nn


class AuxHead(nn.Module):
    WELL_COORDS = [
        [31, 15], [45, 4], [56, 15], [45, 27], [18, 27],
        [4, 15], [18, 4], [18, 15], [45, 15],
    ]

    def __init__(self, state_channels=4, depth=16, aux_channels=16, hidden_dim=64):
        super().__init__()
        self.register_buffer(
            'well_indices', torch.tensor(self.WELL_COORDS, dtype=torch.long))
        num_wells = len(self.WELL_COORDS)
        per_well_dim = depth * state_channels * 2
        self.well_mlp = nn.Sequential(
            nn.Linear(per_well_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.aux_linear = nn.Linear(num_wells * hidden_dim, aux_channels)

    def forward(self, y_t, y_tp1):
        wx = self.well_indices[:, 0]
        wy = self.well_indices[:, 1]

        cols_t = y_t[:, :, :, wx, wy]      # (B, C, D, num_wells)
        cols_tp1 = y_tp1[:, :, :, wx, wy]

        B, C, D, W = cols_t.shape
        cols_t = cols_t.permute(0, 3, 1, 2).reshape(B, W, C * D)
        cols_tp1 = cols_tp1.permute(0, 3, 1, 2).reshape(B, W, C * D)

        per_well = torch.cat([cols_t, cols_tp1], dim=-1)  # (B, num_wells, C*D*2)
        per_well = self.well_mlp(per_well)                 # (B, num_wells, hidden_dim)
        return self.aux_linear(per_well.reshape(B, -1))    # (B, aux_channels)
