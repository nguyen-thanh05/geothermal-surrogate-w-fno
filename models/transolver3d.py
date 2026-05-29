import math
import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange


ACTIVATION = {
    'gelu': nn.GELU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'relu': nn.ReLU,
    'leaky_relu': lambda: nn.LeakyReLU(0.1),
    'softplus': nn.Softplus,
    'elu': nn.ELU,
    'silu': nn.SiLU,
}


def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class SinusoidalPosEmbed3D(nn.Module):
    """Fixed sinusoidal positional encoding for 3D coordinates.

    For each spatial axis, produces sin/cos pairs at geometrically spaced
    frequencies. Total output dim = 3 axes × num_bands × 2 (sin+cos) + 3 (raw coords).
    A linear projection maps this to the target dimension.
    """

    def __init__(self, out_dim, num_bands=32, max_freq=64.0):
        super().__init__()
        self.num_bands = num_bands
        freqs = torch.exp2(torch.linspace(0.0, math.log2(max_freq), num_bands))
        self.register_buffer('freqs', freqs)
        raw_dim = 3 + 3 * num_bands * 2
        self.proj = nn.Linear(raw_dim, out_dim)

    def forward(self, coords):
        """coords: [B, N, 3] with values in [0, 1]."""
        parts = [coords]
        for i in range(3):
            c = coords[..., i:i+1]                          # [B, N, 1]
            scaled = c * self.freqs[None, None, :] * 2 * math.pi  # [B, N, num_bands]
            parts.extend([scaled.sin(), scaled.cos()])
        return self.proj(torch.cat(parts, dim=-1))


class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        super().__init__()
        if act in ACTIVATION:
            act = ACTIVATION[act]
        else:
            raise NotImplementedError(f"Activation '{act}' not supported")
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList(
            [nn.Sequential(nn.Linear(n_hidden, n_hidden), act()) for _ in range(n_layers)]
        )

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            x = self.linears[i](x) + x if self.res else self.linears[i](x)
        return self.linear_post(x)


class PhysicsAttention3D(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=32,
                 H=32, W=32, D=32, kernel=3):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.H = H
        self.W = W
        self.D = D

        self.in_project_x = nn.Conv3d(dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_fx = nn.Conv3d(dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        nn.init.orthogonal_(self.in_project_slice.weight)
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, self.D, C).permute(0, 4, 1, 2, 3).contiguous()

        fx_mid = (self.in_project_fx(x)
                  .permute(0, 2, 3, 4, 1).contiguous()
                  .reshape(B, N, self.heads, self.dim_head)
                  .permute(0, 2, 1, 3).contiguous())
        x_mid = (self.in_project_x(x)
                 .permute(0, 2, 3, 4, 1).contiguous()
                 .reshape(B, N, self.heads, self.dim_head)
                 .permute(0, 2, 1, 3).contiguous())

        slice_weights = self.softmax(
            self.in_project_slice(x_mid) / torch.clamp(self.temperature, min=0.1, max=5)
        )
        slice_norm = slice_weights.sum(2)
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[..., None])

        q = self.to_q(slice_token)
        k = self.to_k(slice_token)
        v = self.to_v(slice_token)
        attn = self.softmax(torch.matmul(q, k.transpose(-1, -2)) * self.scale)
        attn = self.dropout(attn)
        out_slice = torch.matmul(attn, v)

        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        return self.to_out(out_x)


class TransolverBlock(nn.Module):
    def __init__(self, num_heads, hidden_dim, dropout=0., act='gelu', mlp_ratio=4,
                 last_layer=False, out_dim=1, slice_num=32, H=32, W=32, D=32):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = PhysicsAttention3D(
            hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
            dropout=dropout, slice_num=slice_num, H=H, W=W, D=D,
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx):
        fx = self.attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        return fx


class Transolver3D(nn.Module):
    """
    Transolver for 3D structured meshes.

    Args:
        space_dim:    Spatial coordinate channels (default 3).
        n_layers:     Number of Transolver blocks.
        n_hidden:     Hidden dimension throughout the model.
        n_head:       Number of attention heads.
        dropout:      Dropout rate.
        act:          Activation function key.
        mlp_ratio:    MLP expansion ratio inside each block.
        fun_dim:      Number of input function channels (your C).
        out_dim:      Number of output channels.
        slice_num:    G — number of learned physical states.
        ref:          Reference grid resolution for unified positional encoding.
        unified_pos:  If True, use distance-to-reference-grid positional encoding.
        H, W, D:      Grid dimensions. N = H * W * D.
        time_input:   If True, accept a time embedding input.
    """

    def __init__(self, space_dim=3, n_layers=5, n_hidden=256, dropout=0., n_head=8,
                 time_input=False, act='gelu', mlp_ratio=1, fun_dim=1, out_dim=1,
                 slice_num=32, ref=8, unified_pos=False, H=32, W=32, D=32,
                 spatial_embed=False, num_bands=32, max_freq=64.0):
        super().__init__()
        self.H = H
        self.W = W
        self.D = D
        self.ref = ref
        self.unified_pos = unified_pos
        self.spatial_embed = spatial_embed
        self.time_input = time_input
        self.n_hidden = n_hidden
        self.space_dim = space_dim

        if self.spatial_embed:
            self.pos_embed = SinusoidalPosEmbed3D(n_hidden, num_bands=num_bands, max_freq=max_freq)

        if self.unified_pos:
            self.pos = self._build_ref_pos()
            self.preprocess = MLP(fun_dim + ref ** 3, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)
        else:
            self.preprocess = MLP(fun_dim + space_dim, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)

        if time_input:
            self.time_fc = nn.Sequential(
                nn.Linear(n_hidden, n_hidden), nn.SiLU(), nn.Linear(n_hidden, n_hidden)
            )

        self.blocks = nn.ModuleList([
            TransolverBlock(
                num_heads=n_head, hidden_dim=n_hidden, dropout=dropout, act=act,
                mlp_ratio=mlp_ratio, out_dim=out_dim, slice_num=slice_num,
                H=H, W=W, D=D, last_layer=(i == n_layers - 1),
            )
            for i in range(n_layers)
        ])

        self.placeholder = nn.Parameter(torch.rand(n_hidden) / n_hidden)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _build_ref_pos(self):
        sx, sy, sz = self.H, self.W, self.D
        gx = torch.linspace(0, 1, sx).reshape(1, sx, 1, 1, 1).expand(1, sx, sy, sz, 1)
        gy = torch.linspace(0, 1, sy).reshape(1, 1, sy, 1, 1).expand(1, sx, sy, sz, 1)
        gz = torch.linspace(0, 1, sz).reshape(1, 1, 1, sz, 1).expand(1, sx, sy, sz, 1)
        grid = torch.cat((gx, gy, gz), dim=-1)

        r = self.ref
        rx = torch.linspace(0, 1, r).reshape(1, r, 1, 1, 1).expand(1, r, r, r, 1)
        ry = torch.linspace(0, 1, r).reshape(1, 1, r, 1, 1).expand(1, r, r, r, 1)
        rz = torch.linspace(0, 1, r).reshape(1, 1, 1, r, 1).expand(1, r, r, r, 1)
        grid_ref = torch.cat((rx, ry, rz), dim=-1)

        pos = torch.sqrt(
            ((grid[:, :, :, :, None, None, None, :]
              - grid_ref[:, None, None, None, :, :, :, :]) ** 2).sum(dim=-1)
        ).reshape(1, sx, sy, sz, r ** 3).contiguous()
        return pos

    def forward(self, x, fx=None, t=None):
        """
        Args:
            x:  Coordinates  [B, N, space_dim]  where N = H*W*D,
                OR ignored when unified_pos=True.
            fx: Input field  [B, N, fun_dim].
                Pass None for unconditional generation (placeholder added).
            t:  Optional timestep tensor [B] for time-dependent PDEs.

        Returns:
            [B, N, out_dim]
        """
        coords = x

        if self.unified_pos:
            x = (self.pos
                 .to(x.device)
                 .expand(x.shape[0], -1, -1, -1, -1)
                 .reshape(x.shape[0], self.H * self.W * self.D, self.ref ** 3))

        if fx is not None:
            fx = self.preprocess(torch.cat((x, fx), dim=-1))
        else:
            fx = self.preprocess(x) + self.placeholder[None, None, :]

        if self.spatial_embed:
            fx = fx + self.pos_embed(coords)

        if t is not None and self.time_input:
            te = timestep_embedding(t, self.n_hidden).unsqueeze(1).expand(-1, fx.shape[1], -1)
            fx = fx + self.time_fc(te)

        for block in self.blocks:
            fx = block(fx)

        return fx


def make_coords(H, W, D, device='cpu'):
    """Build a flat coordinate tensor [1, H*W*D, 3] with values in [0, 1]."""
    gx = torch.linspace(0, 1, H, device=device)
    gy = torch.linspace(0, 1, W, device=device)
    gz = torch.linspace(0, 1, D, device=device)
    grid = torch.stack(torch.meshgrid(gx, gy, gz, indexing='ij'), dim=-1)
    return grid.reshape(1, H * W * D, 3)


class TransolverWrapper(nn.Module):
    """Drop-in wrapper matching SingleTensorAdapter (like FNOWrapper/UNet3D).

    Accepts [B, in_channels, H, W, D], returns [B, out_channels, H, W, D].
    Residual + zero-init last projection give identity-at-init for stable
    autoregressive next-step prediction.
    """

    def __init__(self, in_channels, out_channels, hidden_dim=256, n_layers=8,
                 n_head=8, slice_num=32, mlp_ratio=2, H=16, W=64, D=32,
                 spatial_embed=True, num_bands=32, max_freq=64.0, **kwargs):
        super().__init__()
        self.out_channels = out_channels
        self.H, self.W, self.D = H, W, D
        self.core = Transolver3D(
            space_dim=3, n_layers=n_layers, n_hidden=hidden_dim, n_head=n_head,
            fun_dim=in_channels, out_dim=out_channels, slice_num=slice_num,
            mlp_ratio=mlp_ratio, H=H, W=W, D=D, spatial_embed=spatial_embed,
            num_bands=num_bands, max_freq=max_freq,
        )
        last = self.core.blocks[-1]
        nn.init.zeros_(last.mlp2.weight)
        nn.init.zeros_(last.mlp2.bias)
        self.register_buffer('coords', make_coords(H, W, D), persistent=False)

    def forward(self, x):
        B = x.shape[0]
        y_skip = x[:, :self.out_channels]
        fx = rearrange(x, 'b c h w d -> b (h w d) c')
        coords = self.coords.expand(B, -1, -1)
        out = self.core(coords, fx=fx)
        out = rearrange(out, 'b (h w d) c -> b c h w d', h=self.H, w=self.W, d=self.D)
        return out + y_skip


if __name__ == '__main__':
    B, C, H, W, D = 2, 4, 16, 64, 32
    out_dim = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = Transolver3D(
        space_dim=3,
        n_layers=3,
        n_hidden=64,
        n_head=4,
        fun_dim=C,
        out_dim=out_dim,
        slice_num=32,
        mlp_ratio=1,
        H=H, W=W, D=D,
        spatial_embed=True,
        num_bands=32,
        max_freq=64.0,
    ).to(device)

    coords = make_coords(H, W, D, device=device).expand(B, -1, -1)
    field = torch.randn(B, C, H, W, D, device=device)
    fx = rearrange(field, 'b c h w d -> b (h w d) c')

    out = model(coords, fx=fx)
    print(f"Input field: {list(field.shape)}")
    print(f"Output:      {list(out.shape)}")
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters:  {params:,}")
