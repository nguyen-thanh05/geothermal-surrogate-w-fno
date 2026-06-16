import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralop.layers.spectral_convolution import SpectralConv
from neuralop.layers.embeddings import GridEmbeddingND


def patchify_3d(x, patch_size):
    B, C, D, H, W = x.shape
    pD, pH, pW = patch_size
    nD, nH, nW = D // pD, H // pH, W // pW
    x = x.reshape(B, C, nD, pD, nH, pH, nW, pW)
    x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()  # (B, nD, nH, nW, C, pD, pH, pW)
    x = x.reshape(B * nD * nH * nW, C, pD, pH, pW)
    return x, (nD, nH, nW)


def unpatchify_3d(x, batch_size, grid_size):
    nD, nH, nW = grid_size
    _, C, pD, pH, pW = x.shape
    x = x.reshape(batch_size, nD, nH, nW, C, pD, pH, pW)
    x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()  # (B, C, nD, pD, nH, pH, nW, pW)
    x = x.reshape(batch_size, C, nD * pD, nH * pH, nW * pW)
    return x


def highfreq_3d(x, kernel_size=4):
    smooth = F.avg_pool3d(x, kernel_size=kernel_size, stride=kernel_size)
    smooth = F.interpolate(smooth, size=x.shape[2:],
                           mode='trilinear', align_corners=False)
    return x - smooth


class MLP_Block(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super(MLP_Block, self).__init__()
        self.fc1 = nn.Conv3d(in_dim, hidden_dim * 2, kernel_size=1)
        self.fc2 = nn.Conv3d(hidden_dim * 2, out_dim, kernel_size=1)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x
    

class ModulationEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_blocks):
        super(ModulationEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks
        self.pool_stream_encoder = MLP_Block(in_dim, hidden_dim, hidden_dim)
        self.pool_stream_out = nn.Linear(hidden_dim, hidden_dim * 2 * n_blocks) # gamma, beta and alpha
        
        self.local_perception = nn.Sequential(
            nn.Conv3d(in_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(hidden_dim, hidden_dim * 2 * n_blocks, kernel_size=1)
        )
        
        # AdaLN-Zero: gamma = beta = alpha = 0 at init, so blocks start with
        # branches gated off; with the zero-init projection the model is
        # identity at init.
        nn.init.zeros_(self.pool_stream_out.weight)
        nn.init.zeros_(self.pool_stream_out.bias)
        nn.init.zeros_(self.local_perception[-1].weight)
        nn.init.zeros_(self.local_perception[-1].bias)

    def forward(self, x):
        """(B, in_dim, D, H, W) to (B, n_blocks, 3, hidden_dim, D, H, W)."""
        B, _, D, H, W = x.shape
        pool_stream = self.pool_stream_encoder(x)
        pool_stream = self.pool_stream_out(pool_stream.mean(dim=[2, 3, 4]))

        local_perception_stream = self.local_perception(x)
        cond = local_perception_stream + pool_stream.view(B, -1, 1, 1, 1)
        return cond.view(B, self.n_blocks, 2, self.hidden_dim, D, H, W)
    

class LOGLO_Block(nn.Module):
    def __init__(self, hidden_dim, patch_size=(8, 8, 8)):
        super(LOGLO_Block, self).__init__()
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size

        # --- Global branch ---
        self.global_spectral = SpectralConv(
            in_channels=hidden_dim, out_channels=hidden_dim,
            n_modes=(4, 16, 8))
        self.global_mlp_inner = MLP_Block(hidden_dim, hidden_dim, hidden_dim)
        self.global_mlp_outer = MLP_Block(hidden_dim, hidden_dim, hidden_dim)
        self.global_mlp_skip = MLP_Block(hidden_dim, hidden_dim, hidden_dim)

        # --- Local branch (all modes retained for patch size) ---
        self.local_spectral = SpectralConv(
            in_channels=hidden_dim, out_channels=hidden_dim,
            n_modes=patch_size)
        self.local_mlp_inner = MLP_Block(hidden_dim, hidden_dim, hidden_dim)
        self.local_mlp_outer = MLP_Block(hidden_dim, hidden_dim, hidden_dim)
        self.local_mlp_skip = MLP_Block(hidden_dim, hidden_dim, hidden_dim)

        # --- High-freq branch (pointwise MLP only) ---
        self.highfreq_mlp = MLP_Block(hidden_dim, hidden_dim, hidden_dim)

        self.norm = nn.GroupNorm(num_groups=hidden_dim, num_channels=hidden_dim, affine=False)
        self.activation = nn.GELU()
    
    def forward(self, z, z_hat, z_prime, gamma=None, beta=None):
        B = z.shape[0]
        D, H, W = z.shape[2], z.shape[3], z.shape[4]
        pD, pH, pW = self.patch_size
        grid_size = (D // pD, H // pH, W // pW)
        
        if gamma is not None:
            z_mod = self.norm(z)
            z_hat_mod = self.norm(z_hat)
            z_mod = z_mod * (1 + gamma) + beta
            z_hat_mod = z_hat_mod * (1 + gamma) + beta
        else:
            z_mod = z
            z_hat_mod = z_hat

        y_global = self.global_mlp_outer(
            self.activation(self.global_spectral(z_mod) + self.global_mlp_inner(z_mod))
        ) + self.global_mlp_skip(z)
        
        z_hat_mod, _ = patchify_3d(z_hat_mod, self.patch_size)
        z_hat, _ = patchify_3d(z_hat, self.patch_size)  # for skip connection
        y_local = self.local_mlp_outer(
            self.activation(self.local_spectral(z_hat_mod) + self.local_mlp_inner(z_hat_mod))
        ) + self.local_mlp_skip(z_hat)
        y_local_full = unpatchify_3d(y_local, B, grid_size)

        y_highfreq = self.highfreq_mlp(z_prime)

        return self.norm(y_global + y_local_full + y_highfreq)
    

class LOGLO_FNO(nn.Module):
    def __init__(self, in_dim=4,
                 out_dim=4,
                 lifting_dim=128,
                 projection_dim=128,
                 hidden_dim=64,
                 n_blocks=4,
                 action_channels=2,
                 patch_size=(8, 8, 8),
                 highfreq_kernel=4,
                 **kwargs):
        super(LOGLO_FNO, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.lifting_dim = lifting_dim
        self.projection_dim = projection_dim
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks
        self.action_channels = action_channels
        self.patch_size = patch_size
        self.highfreq_kernel = highfreq_kernel

        # ---- Global lifting (grid coords + Conv3d) ----
        spatial_grid_boundaries = [[0.0, 1.0]] * 3
        self.grid_embedding = GridEmbeddingND(in_channels=self.in_dim,
                                              dim=3,
                                              grid_boundaries=spatial_grid_boundaries)
        self.global_lifting = MLP_Block(self.in_dim, self.hidden_dim, self.lifting_dim)

        # ---- Local lifting (patches, independent representation) ----
        self.local_lifting = MLP_Block(self.in_dim + 3, self.hidden_dim, self.lifting_dim)

        # ---- High-freq lifting (independent representation) ----
        self.highfreq_lifting = MLP_Block(self.in_dim, self.hidden_dim, self.lifting_dim)
        
        # ---- LOGLO blocks ----
        self.loglo_blocks = nn.ModuleList(
            [LOGLO_Block(hidden_dim=self.hidden_dim, patch_size=self.patch_size)
             for _ in range(self.n_blocks)]
        )

        self.projection = MLP_Block(in_dim=self.hidden_dim, out_dim=self.out_dim,
                                    hidden_dim=self.projection_dim)
        nn.init.zeros_(self.projection.fc2.weight)
        nn.init.zeros_(self.projection.fc2.bias) # zero init, the very first prediction is u_t+1 = u_t

        self.modulation_encoder = ModulationEncoder(
            in_dim=action_channels + self.in_dim,
            hidden_dim=self.hidden_dim,
            n_blocks=self.n_blocks,
        )


    def forward(self, x, action):
        x_input = x

        conditioning = self.modulation_encoder(torch.cat([action, x_input], dim=1))

        z = self.global_lifting(x)

        z_hat = self.local_lifting(self.grid_embedding(x))

        x_h = highfreq_3d(x, kernel_size=self.highfreq_kernel)
        z_prime = self.highfreq_lifting(x_h)

        for i, block in enumerate(self.loglo_blocks):
            z = block(z, z_hat, z_prime,
                      conditioning[:, i, 0], conditioning[:, i, 1])

            if i < self.n_blocks - 1:
                z_hat = z  # block patchifies internally ⇒ v1's z_hat = patchify(z)
                z_prime = highfreq_3d(z, kernel_size=self.highfreq_kernel)

        spatial_out = self.projection(z) + x_input[:, :self.out_dim]
        return spatial_out


class VanillaLOGLO_FNO(nn.Module):
    def __init__(self, in_dim=6,
                 out_dim=4,
                 lifting_dim=128,
                 projection_dim=128,
                 hidden_dim=64,
                 n_blocks=4,
                 patch_size=(8, 8, 8),
                 highfreq_kernel=4,
                 **kwargs):
        super(VanillaLOGLO_FNO, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.lifting_dim = lifting_dim
        self.projection_dim = projection_dim
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks
        self.patch_size = patch_size
        self.highfreq_kernel = highfreq_kernel

        # ---- Global lifting ----
        spatial_grid_boundaries = [[0.0, 1.0]] * 3
        self.grid_embedding = GridEmbeddingND(in_channels=self.in_dim,
                                              dim=3,
                                              grid_boundaries=spatial_grid_boundaries)
        self.global_lifting = MLP_Block(self.in_dim, self.hidden_dim, self.hidden_dim)

        # ---- Local lifting (grid coords: patches lose global position) ----
        self.local_lifting = MLP_Block(self.in_dim + 3, self.hidden_dim, self.hidden_dim)

        # ---- High-freq lifting ----
        self.highfreq_lifting = MLP_Block(self.in_dim, self.hidden_dim, self.hidden_dim)

        # ---- LOGLO blocks ----
        self.loglo_blocks = nn.ModuleList(
            [LOGLO_Block(hidden_dim=self.hidden_dim, patch_size=self.patch_size)
             for _ in range(self.n_blocks)]
        )

        self.projection = MLP_Block(in_dim=self.hidden_dim, out_dim=self.out_dim,
                                    hidden_dim=self.projection_dim)
        nn.init.zeros_(self.projection.fc2.weight)
        nn.init.zeros_(self.projection.fc2.bias) # zero init, the very first prediction is u_t+1 = u_t

    def forward(self, x):
        # x: (B, in_dim, D, H, W) — state + action already concatenated upstream.
        x_input = x

        z = self.global_lifting(x)

        z_hat = self.local_lifting(self.grid_embedding(x))

        x_h = highfreq_3d(x, kernel_size=self.highfreq_kernel)
        z_prime = self.highfreq_lifting(x_h)

        for i, block in enumerate(self.loglo_blocks):
            z = block(z, z_hat, z_prime)  # no AdaLN modulation, no norm

            if i < self.n_blocks - 1:
                z_hat = z
                z_prime = highfreq_3d(z, kernel_size=self.highfreq_kernel)

        spatial_out = self.projection(z) + x_input[:, :self.out_dim]
        return spatial_out


if __name__ == "__main__":
    import torchinfo
    if torch.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    test_data = torch.randn(1, 4, 16, 64, 32).to(device)
    test_action = torch.randn(1, 2, 16, 64, 32).to(device)
    model = LOGLO_FNO(in_dim=4, out_dim=4, lifting_dim=256,
                      projection_dim=256, hidden_dim=64, n_blocks=5,
                      action_channels=2,
                      patch_size=(8, 8, 8)).to(device)
    spatial_out = model(test_data, test_action)
    print(f"LOGLO-FNO v2 — Spatial: {spatial_out.shape}")

    from models.aux_head import AuxHead
    aux = AuxHead(state_channels=4, depth=16, aux_channels=16, hidden_dim=64).to(device)
    aux_out = aux(test_data, spatial_out)
    print(f"AuxHead — Aux: {aux_out.shape}")

    torchinfo.summary(model, input_data=[test_data, test_action])
        