import torch
import torch.nn.functional as F


beta = 8.8e-4       # thermal expansion coefficient [1/°C]
c_f = 7e-7           # fluid compressibility [1/kPa]
c_w = 4.5e-7         # rock compressibility [1/kPa]
c_t = c_f + c_w      # total compressibility


def extract_physical_porosity(static):
    phi_m = static[:, 0] * (0.07 - 0.03) + 0.03
    phi_frac = static[:, 1] * (0.008 - 0.002) + 0.002
    return phi_m, phi_frac


def compute_mbe_rhs(action_normalized, action_max=5000.0, rho_sc=1000.0, dt=7.0):
    physical_action = action_normalized * action_max
    total_rate = physical_action[:, 0, 5, :, :].sum(dim=(1, 2))
    return total_rate * rho_sc * dt


def compute_mbe_lhs(y_t, y_tp1, phi_m, phi_frac, *,
                    pres_min, pres_max,
                    V=1000.0,
                    temp_min=20.0, temp_max=185.0,
                    rho_ref=1000.11, P_ref=100.0, T_ref=25.0):
    temp_range = temp_max - temp_min
    pres_range = pres_max - pres_min

    T_m = y_t[:, 0] * temp_range + temp_min
    T_f = y_t[:, 1] * temp_range + temp_min
    P_m = y_t[:, 2] * pres_range + pres_min
    P_f = y_t[:, 3] * pres_range + pres_min

    rho_m = rho_ref * torch.exp(c_t * (P_m - P_ref) - beta * (T_m - T_ref))
    rho_f = rho_ref * torch.exp(c_t * (P_f - P_ref) - beta * (T_f - T_ref))

    dP_m = (y_tp1[:, 2] - y_t[:, 2]) * pres_range
    dP_f = (y_tp1[:, 3] - y_t[:, 3]) * pres_range
    dT_m = (y_tp1[:, 0] - y_t[:, 0]) * temp_range
    dT_f = (y_tp1[:, 1] - y_t[:, 1]) * temp_range

    accum = torch.sum(
        rho_m * phi_m * (c_t * dP_m - beta * dT_m) * V +
        rho_f * phi_frac * (c_t * dP_f - beta * dT_f) * V,
        dim=(1, 2, 3),
    )
    return accum


def compute_mbe_loss(y_t, y_tp1, action_normalized, phi_m, phi_frac, *,
                     pres_min, pres_max, char_mass=1e7):
    lhs = compute_mbe_lhs(y_t, y_tp1, phi_m, phi_frac,
                          pres_min=pres_min, pres_max=pres_max)
    rhs = compute_mbe_rhs(action_normalized)
    residual = (lhs - rhs) / char_mass
    return torch.mean(residual ** 2)


def mean_field_pressure_loss(pred, target):
    pred_mean = pred[:, 2:4].mean(dim=(2, 3, 4))
    target_mean = target[:, 2:4].mean(dim=(2, 3, 4))
    return F.mse_loss(pred_mean, target_mean)


def add_adaptive_noise(y, alpha=(0.0025, 0.0025, 0.025, 0.025), eps=1e-8):
    mu = y.mean(dim=(2, 3, 4), keepdim=True)
    sigma = torch.sqrt(((y - mu) ** 2).mean(dim=(2, 3, 4), keepdim=True) + eps)
    N = torch.randn_like(y)
    alpha_t = torch.tensor(alpha, device=y.device, dtype=y.dtype).view(1, -1, 1, 1, 1)
    return y + alpha_t * sigma * N


def radial_binned_spectral_loss(preds, target, iLow=4, iHigh=12):
    B, C, D, H, W = target.shape
    device = target.device

    err_fft = torch.fft.fftn(preds - target, dim=[2, 3, 4])
    power = err_fft.real ** 2 + err_fft.imag ** 2

    dh, hh, wh = D // 2, H // 2, W // 2
    power_h = power[:, :, :dh, :hh, :wh]

    kd = torch.arange(dh, device=device, dtype=torch.float32)
    kh = torch.arange(hh, device=device, dtype=torch.float32)
    kw = torch.arange(wh, device=device, dtype=torch.float32)
    KD, KH, KW = torch.meshgrid(kd, kh, kw, indexing="ij")
    radii = (KD ** 2 + KH ** 2 + KW ** 2).sqrt().floor().long()
    max_radius = int(radii.max())
    radii_flat = radii.reshape(-1)

    power_flat = power_h.reshape(B, C, -1)
    binned = torch.zeros(B, C, max_radius + 1, device=device)
    binned.index_add_(2, radii_flat, power_flat)

    nrm = D * H * W
    spectral_err = torch.sqrt(binned.mean(dim=0)) / nrm

    iLow = min(iLow, max_radius + 1)
    iHigh = min(iHigh, max_radius + 1)

    band_list = []
    for lo, hi in [(0, iLow), (iLow, iHigh), (iHigh, max_radius + 1)]:
        if hi > lo:
            band_list.append(spectral_err[:, lo:hi].mean(dim=1))
        else:
            band_list.append(torch.zeros(C, device=device))

    band_losses = torch.stack(band_list, dim=1)
    per_band = band_losses.mean(dim=0)
    return per_band[1:3].sum(), per_band
