import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralop.layers.spectral_convolution import SpectralConv
from neuralop.layers.embeddings import GridEmbeddingND


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def patchify_3d(x, patch_size):
    """Split a 5-D tensor into non-overlapping 3-D patches.

    Args:
        x: (B, C, D, H, W)
        patch_size: (pD, pH, pW)
    Returns:
        patches: (B * nD * nH * nW, C, pD, pH, pW)
        grid_size: (nD, nH, nW) — number of patches per spatial dim
    """
    B, C, D, H, W = x.shape
    pD, pH, pW = patch_size
    nD, nH, nW = D // pD, H // pH, W // pW
    x = x.reshape(B, C, nD, pD, nH, pH, nW, pW)
    x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()  # (B, nD, nH, nW, C, pD, pH, pW)
    x = x.reshape(B * nD * nH * nW, C, pD, pH, pW)
    return x, (nD, nH, nW)


def unpatchify_3d(x, batch_size, grid_size):
    """Reassemble patches back into the full spatial volume.

    Args:
        x: (B * nD * nH * nW, C, pD, pH, pW)
        batch_size: B
        grid_size: (nD, nH, nW)
    Returns:
        (B, C, nD*pD, nH*pH, nW*pW)
    """
    nD, nH, nW = grid_size
    _, C, pD, pH, pW = x.shape
    x = x.reshape(batch_size, nD, nH, nW, C, pD, pH, pW)
    x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()  # (B, C, nD, pD, nH, pH, nW, pW)
    x = x.reshape(batch_size, C, nD * pD, nH * pH, nW * pW)
    return x


def highfreq_3d(x, kernel_size=4):
    """High-pass: x - upsample(AvgPool(x))."""
    smooth = F.avg_pool3d(x, kernel_size=kernel_size, stride=kernel_size)
    smooth = F.interpolate(smooth, size=x.shape[2:],
                           mode='trilinear', align_corners=False)
    return x - smooth


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

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
    """Encode actions → per-block AdaLN-Zero modulation params (γ, β).

    Structure: Linear → GELU → Conv3d → GELU → Linear
    (pointwise Conv3d k=1 acts as a channel-wise Linear). Final projection
    zero-init ⇒ γ=β=0 at start, so each block reduces to LN(sum); combined
    with the zero-init projection the whole model is identity at init.
    """

    def __init__(self, action_channels, hidden_dim, n_blocks):
        super(ModulationEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks
        self.encoder = nn.Sequential(
            nn.Conv3d(action_channels, hidden_dim, kernel_size=1),            # Linear
            nn.GELU(),
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1),      # Conv3d
            nn.GELU(),
            nn.Conv3d(hidden_dim, hidden_dim * 2 * n_blocks, kernel_size=1),  # Linear
        )
        nn.init.zeros_(self.encoder[-1].weight)
        nn.init.zeros_(self.encoder[-1].bias)

    def forward(self, action):
        """(B, action_channels, D, H, W) → (B, n_blocks, 2, hidden_dim, D, H, W)."""
        B, _, D, H, W = action.shape
        cond = self.encoder(action)
        return cond.view(B, self.n_blocks, 2, self.hidden_dim, D, H, W)


class LOGLO_Block(nn.Module):
    """One Local-Global block (AdaLN-Zero conditioning at the end).

    Per-branch op (transformer-style residual):
        Y = MLP_outer( σ( SpectralConv(z) + MLP_inner(z) ) ) + MLP_skip(z)

    High-freq branch is just a pointwise MLP:
        Y_hf = MLP(z')

    Combination + modulation:
        s   = Y_global + Y_local + Y_highfreq
        out = LN(s) ⊙ (1 + γ(a)) + β(a)   # non-affine LN, no post-activation
    """

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

        # --- End-of-block normalization (non-affine: affine handled by AdaLN γ/β) ---
        self.norm = nn.GroupNorm(num_groups=1, num_channels=hidden_dim, affine=False)
        # activation used inside the global/local branches (not after modulation)
        self.activation = nn.GELU()

    def forward(self, z, z_hat, z_prime, gamma=None, beta=None):
        """
        Args:
            z:       (B, C, D, H, W)        — global hidden state
            z_hat:   (B*nP, C, pD, pH, pW)  — patchified hidden state
            z_prime: (B, C, D, H, W)        — high-freq hidden state
            gamma, beta: (B, C, D, H, W)    — AdaLN modulation from action.
                         If None (e.g. VanillaLOGLO_FNO), modulation is skipped
                         and the block returns the non-affine LN(sum).
        Returns:
            out: (B, C, D, H, W)
        """
        B = z.shape[0]
        D, H, W = z.shape[2], z.shape[3], z.shape[4]
        pD, pH, pW = self.patch_size
        grid_size = (D // pD, H // pH, W // pW)

        # Global: MLP_o(σ(SpectralConv(z) + MLP_i(z))) + MLP_s(z)
        y_global = self.global_mlp_outer(
            self.activation(self.global_spectral(z) + self.global_mlp_inner(z))
        ) + self.global_mlp_skip(z)

        # Local: same structure on patches
        y_local = self.local_mlp_outer(
            self.activation(self.local_spectral(z_hat) + self.local_mlp_inner(z_hat))
        ) + self.local_mlp_skip(z_hat)
        y_local_full = unpatchify_3d(y_local, B, grid_size)

        # High-freq: pointwise MLP
        y_highfreq = self.highfreq_mlp(z_prime)

        # Sum → LayerNorm (GroupNorm-1, non-affine) → AdaLN modulation (no activation)
        s = y_global + y_local_full + y_highfreq
        s = self.norm(s)
        if gamma is not None:
            s = s * (1 + gamma) + beta
        return s


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class LOGLO_FNO(nn.Module):
    """Per-iteration variant with independent input lifts.

    Block 0 receives:
        z       = lifting(grid_embedding(x))     — global, position-aware
        z_hat   = local_lifting(patchify(x))     — independent local lift
        z_prime = highfreq_lifting(highfreq(x))  — independent high-freq lift

    Between blocks (i → i+1) z_hat and z_prime are refreshed from the
    updated hidden state z:
        z_hat   = patchify(z)
        z_prime = highfreq(z)
    """

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
        self.lifting = nn.Conv3d(in_channels=self.in_dim + 3,
                                 out_channels=self.hidden_dim, kernel_size=1)

        # ---- Local lifting (patches, independent representation) ----
        self.local_lifting = nn.Conv3d(in_channels=self.in_dim,
                                       out_channels=self.hidden_dim, kernel_size=1)

        # ---- High-freq lifting (independent representation) ----
        self.highfreq_lifting = nn.Conv3d(in_channels=self.in_dim,
                                          out_channels=self.hidden_dim, kernel_size=1)

        # ---- LOGLO blocks ----
        self.loglo_blocks = nn.ModuleList(
            [LOGLO_Block(hidden_dim=self.hidden_dim, patch_size=self.patch_size)
             for _ in range(self.n_blocks)]
        )

        # ---- Projection (zero-init → identity at init: y(t+1) = y(t) + 0) ----
        self.projection = MLP_Block(in_dim=self.hidden_dim, out_dim=self.out_dim,
                                    hidden_dim=self.projection_dim)
        nn.init.zeros_(self.projection.fc2.weight)
        nn.init.zeros_(self.projection.fc2.bias)

        # ---- Action → AdaLN-Zero modulation (single consolidated encoder) ----
        # γ, β per block (factor 2). Final Conv zero-init so γ=β=0 at start ⇒
        # each block reduces to LN(sum). Combined with the zero-init projection,
        # the model is identity at init.
        self.modulation_encoder = ModulationEncoder(
            action_channels=action_channels,
            hidden_dim=self.hidden_dim,
            n_blocks=self.n_blocks,
        )


    def forward(self, x, action):
        x_input = x

        conditioning = self.modulation_encoder(action)

        z = self.lifting(self.grid_embedding(x))

        x_patches, _ = patchify_3d(x, self.patch_size)
        z_hat = self.local_lifting(x_patches)

        x_h = highfreq_3d(x, kernel_size=self.highfreq_kernel)
        z_prime = self.highfreq_lifting(x_h)

        for i, block in enumerate(self.loglo_blocks):
            z = block(z, z_hat, z_prime,
                      conditioning[:, i, 0], conditioning[:, i, 1])

            if i < self.n_blocks - 1:
                z_hat, _ = patchify_3d(z, self.patch_size)
                z_prime = highfreq_3d(z, kernel_size=self.highfreq_kernel)

        spatial_out = self.projection(z) + x_input[:, :self.out_dim]
        return spatial_out


class VanillaLOGLO_FNO(nn.Module):
    """Vanilla LOGLO_FNO baseline — action concatenated into the input channels
    instead of AdaLN-Zero modulation.

    Identical to LOGLO_FNO (global spectral + local patch spectral + high-freq
    MLP branches, zero-init projection ⇒ identity at init) EXCEPT the action
    modulation encoder is removed. The action is concatenated into ``x`` upstream
    (via SingleTensorAdapter), so ``in_dim`` already includes the rate/mask
    (+ static) channels and the blocks run with no γ/β modulation (each reduces
    to the non-affine LN(sum)).
    """

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

        # ---- Global lifting (grid coords + Conv3d) ----
        spatial_grid_boundaries = [[0.0, 1.0]] * 3
        self.grid_embedding = GridEmbeddingND(in_channels=self.in_dim,
                                              dim=3,
                                              grid_boundaries=spatial_grid_boundaries)
        self.lifting = nn.Conv3d(in_channels=self.in_dim + 3,
                                 out_channels=self.hidden_dim, kernel_size=1)

        # ---- Local lifting (patches, independent representation) ----
        self.local_lifting = nn.Conv3d(in_channels=self.in_dim,
                                       out_channels=self.hidden_dim, kernel_size=1)

        # ---- High-freq lifting (independent representation) ----
        self.highfreq_lifting = nn.Conv3d(in_channels=self.in_dim,
                                          out_channels=self.hidden_dim, kernel_size=1)

        # ---- LOGLO blocks ----
        self.loglo_blocks = nn.ModuleList(
            [LOGLO_Block(hidden_dim=self.hidden_dim, patch_size=self.patch_size)
             for _ in range(self.n_blocks)]
        )

        # ---- Projection (zero-init → identity at init: y(t+1) = y(t) + 0) ----
        self.projection = MLP_Block(in_dim=self.hidden_dim, out_dim=self.out_dim,
                                    hidden_dim=self.projection_dim)
        nn.init.zeros_(self.projection.fc2.weight)
        nn.init.zeros_(self.projection.fc2.bias)

    def forward(self, x):
        # x: (B, in_dim, D, H, W) — state + action already concatenated upstream.
        x_input = x

        z = self.lifting(self.grid_embedding(x))

        x_patches, _ = patchify_3d(x, self.patch_size)
        z_hat = self.local_lifting(x_patches)

        x_h = highfreq_3d(x, kernel_size=self.highfreq_kernel)
        z_prime = self.highfreq_lifting(x_h)

        for i, block in enumerate(self.loglo_blocks):
            z = block(z, z_hat, z_prime)  # no AdaLN modulation

            if i < self.n_blocks - 1:
                z_hat, _ = patchify_3d(z, self.patch_size)
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
