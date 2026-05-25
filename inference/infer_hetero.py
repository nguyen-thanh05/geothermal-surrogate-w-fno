import numpy as np
import torch
import os
import argparse
import yaml
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from losses import LpLoss


# ── Constants (matched to ARFNO_hetero.py ARDataset normalization) ──────────
TEMP_MIN, TEMP_MAX = 20.0, 185.0
PRES_MIN, PRES_MAX = 1300.0, 70000.0
ACTION_MIN, ACTION_MAX = 0.0, 5000.0
ENERGY_MAX = 2.9e12
ENERGY_LOG_DENOM = np.log1p(ENERGY_MAX)
TEMP_RANGE = TEMP_MAX - TEMP_MIN
PRES_RANGE = PRES_MAX - PRES_MIN
ACTION_RANGE = ACTION_MAX - ACTION_MIN

# Aux (BHP + energy rate) — hetero _aux_at normalizes BHP with pres_min/pres_max
AUX_BHP_MIN, AUX_BHP_MAX = 1300.0, 70000.0
AUX_BHP_RANGE = AUX_BHP_MAX - AUX_BHP_MIN

# Static heterogeneity field ranges
POR_MIN_MATRIX, POR_MAX_MATRIX = 0.03, 0.07
POR_MIN_FRAC, POR_MAX_FRAC = 0.002, 0.008
PERM_MIN_MATRIX, PERM_MAX_MATRIX = 0.05, 0.12
PERM_MIN_FRAC, PERM_MAX_FRAC = 3.0, 190.0

CHANNEL_NAMES = ['Temp Formation', 'Temp Frac', 'Pres Formation', 'Pres Frac']
CHANNEL_UNITS = ['°C', '°C', 'kPa', 'kPa']
CHANNEL_SCALES = [TEMP_RANGE, TEMP_RANGE, PRES_RANGE, PRES_RANGE]
CHANNEL_COLORS = ['tab:red', 'tab:orange', 'tab:blue', 'tab:cyan']

WELL_NAMES = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'Inj1', 'Inj2']
ENERGY_WELL_NAMES = WELL_NAMES[:7]   # energy rate: 7 production wells
WELL_COORDS = [
    [31, 15], [45, 4], [56, 15], [45, 27], [18, 27],
    [4, 15], [18, 4], [18, 15], [45, 15],
]

N_STEPS = 156
TEST_INDICES = list(range(350, 400))


# ── Helpers ──────────────────────────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, 'mps') and torch.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_datasets(data_path):
    """Load all dataset arrays (memory-mapped), including static heterogeneity fields."""
    load = lambda name: np.load(os.path.join(data_path, name), mmap_mode='r')
    return {
        'temp': load('all_temp_formation.npy'),
        'pres': load('all_pres_formation.npy'),
        'temp_frac': load('all_temp_frac.npy'),
        'pres_frac': load('all_pres_frac.npy'),
        'action': load('all_action.npy'),
        'aux': load('all_energyrate_bhp.npy'),
        'por_matrix': load('all_por_matrix.npy'),
        'por_frac': load('all_por_frac.npy'),
        'perm_matrix': load('all_perm_matrix.npy'),
        'perm_frac': load('all_perm_frac.npy'),
    }


def normalize_trajectory(datasets, idx):
    """Return normalized GT fields (157, 4, 16, 64, 32) for a trajectory."""
    tf = (datasets['temp'][idx].astype(np.float32) - TEMP_MIN) / TEMP_RANGE
    tfr = (datasets['temp_frac'][idx].astype(np.float32) - TEMP_MIN) / TEMP_RANGE
    pf = (datasets['pres'][idx].astype(np.float32) - PRES_MIN) / PRES_RANGE
    pfr = (datasets['pres_frac'][idx].astype(np.float32) - PRES_MIN) / PRES_RANGE
    return np.stack([tf, tfr, pf, pfr], axis=1)


def get_static(datasets, idx, device):
    """Per-trajectory static heterogeneity tensor, normalized.

    Returns: (1, 4, D, H, W) on `device` — [por_matrix, por_frac, perm_matrix, perm_frac].
    """
    pm = (datasets['por_matrix'][idx].astype(np.float32)  - POR_MIN_MATRIX)  / (POR_MAX_MATRIX  - POR_MIN_MATRIX)
    pf = (datasets['por_frac'][idx].astype(np.float32)    - POR_MIN_FRAC)    / (POR_MAX_FRAC    - POR_MIN_FRAC)
    km = (datasets['perm_matrix'][idx].astype(np.float32) - PERM_MIN_MATRIX) / (PERM_MAX_MATRIX - PERM_MIN_MATRIX)
    kf = (datasets['perm_frac'][idx].astype(np.float32)   - PERM_MIN_FRAC)   / (PERM_MAX_FRAC   - PERM_MIN_FRAC)
    static = np.stack([pm, pf, km, kf], axis=0)  # (4, D, H, W)
    return torch.from_numpy(static).unsqueeze(0).float().to(device)


def load_model(cfg, checkpoint_path, device, use_ema=False):
    """Load LOGLO_FNO v2 model and AuxHead from config and checkpoint (hetero variant)."""
    from models.loglo_fno import LOGLO_FNO
    from models.aux_head import AuxHead
    m = cfg['model']
    model = LOGLO_FNO(
        in_dim=m['in_dim'], out_dim=m['out_dim'],
        lifting_dim=m['lifting_dim'], projection_dim=m['projection_dim'],
        hidden_dim=m['hidden_dim'], n_blocks=m['n_blocks'],
        action_channels=m['action_channels'],
    )
    aux_head_model = AuxHead(
        state_channels=m['out_dim'],
        depth=16,
        aux_channels=m.get('aux_channels', 16),
        hidden_dim=m['hidden_dim'],
    )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    key = 'ema_model' if use_ema else 'model'
    aux_key = 'ema_aux' if use_ema else 'aux_head'
    print(f"Loading {'EMA' if use_ema else 'non-EMA'} weights")
    ckpt[key].pop('_metadata', None)
    model.load_state_dict(ckpt[key])
    aux_head_model.load_state_dict(ckpt[aux_key])
    model.to(device)
    aux_head_model.to(device)
    model.eval()
    aux_head_model.eval()
    return model, aux_head_model


def autoregressive_rollout(model, aux_head_model, gt_all, datasets, idx, device):
    """Run full AR rollout with heterogeneity conditioning.

    Returns:
        pred_all     (157,4,D,H,W) predicted fields
        l2_rel       per-channel L2 relative errors
        pred_aux_all (156,16) predicted aux (normalized): BHP[0:9], energy[9:16]
    """
    l2_loss = LpLoss(d=3, p=2, reduction='none')
    y_t = torch.from_numpy(gt_all[0]).unsqueeze(0).float().to(device)  # (1, 4, D, H, W)
    static = get_static(datasets, idx, device)                         # (1, 4, D, H, W)

    pred_all = [gt_all[0]]
    pred_aux_all = []
    l2_rel = {ch: [] for ch in range(4)}

    with torch.no_grad():
        for t in range(N_STEPS):
            # Prepare action
            action_np = datasets['action'][idx, t].astype(np.float32)
            action_t = torch.from_numpy(action_np.copy()).unsqueeze(0).float()
            action_t = (action_t - ACTION_MIN) / ACTION_RANGE

            # Zero out non-perforated layers
            for wx, wy in WELL_COORDS:
                action_t[:, 0:2, wx, wy] = 0.0

            action_t = action_t.to(device)
            x_t = action_t.unsqueeze(1)                             # (1, 1, D, H, W)
            active_wells = (x_t != 0).float()
            x_t = torch.cat([x_t, active_wells, static], dim=1)     # (1, 6, D, H, W)

            y_input = torch.cat([y_t, static], dim=1)               # (1, 8, D, H, W)
            predicted_y = model(y_input, x_t)
            predicted_aux = aux_head_model(y_t, predicted_y)

            pred_np = predicted_y[0].cpu().numpy()
            pred_all.append(pred_np)
            pred_aux_all.append(predicted_aux[0].cpu().numpy())

            # Per-channel L2 relative error vs GT at t+1
            gt_tp1 = torch.from_numpy(gt_all[t + 1]).unsqueeze(0).float().to(device)
            for ch in range(4):
                err = l2_loss(predicted_y[:, ch:ch + 1], gt_tp1[:, ch:ch + 1]).item()
                l2_rel[ch].append(err)

            y_t = predicted_y

            if (t + 1) % 40 == 0:
                print(f"  Step {t + 1}/{N_STEPS}")

    pred_all = np.array(pred_all)  # (157,4,16,64,32)
    pred_aux_all = np.array(pred_aux_all)  # (156,16)
    return pred_all, l2_rel, pred_aux_all


def extract_well_rates(datasets, idx):
    """Return per-well rates array (9, 156)."""
    rates = np.zeros((len(WELL_COORDS), N_STEPS))
    for t in range(N_STEPS):
        action_np = datasets['action'][idx, t].astype(np.float32)
        for w, (wx, wy) in enumerate(WELL_COORDS):
            rates[w, t] = np.abs(action_np[:, wx, wy]).sum()
    return rates


def load_gt_aux(datasets, idx):
    """Return GT aux for rollout steps 1..156 in physical units.

    Returns (gt_bhp, gt_energy) with shapes (156,9) and (156,7).
    Raw file layout: BHP at [0:9], energy rate at [9:16].
    """
    aux = datasets['aux'][idx][1:N_STEPS + 1].astype(np.float32)  # (156,16)
    gt_bhp = aux[:, 0:9]
    gt_energy = aux[:, 9:16]
    return gt_bhp, gt_energy


# ── Plots ────────────────────────────────────────────────────────────────────
def plot_l2_relative(l2_rel, out_dir, traj_idx):
    """Plot 1: L2 relative error over time for all 4 channels (no action overlay)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    timesteps = np.arange(1, N_STEPS + 1)
    for ch in range(4):
        ax.plot(timesteps, l2_rel[ch], label=CHANNEL_NAMES[ch], color=CHANNEL_COLORS[ch])
    ax.set_xlabel('Timestep')
    ax.set_ylabel('L2 Relative Error')
    ax.set_title(f'Per-Channel L2 Relative Error — Trajectory {traj_idx}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, f'l2_rel_traj{traj_idx}.png')
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close(fig)


def plot_physical_error(pred_all, gt_all, out_dir, traj_idx):
    """Plot: Mean absolute error in physical units — 4 subplots, one per channel."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    timesteps = np.arange(1, N_STEPS + 1)
    for ch in range(4):
        abs_err = np.abs(pred_all[1:, ch] - gt_all[1:, ch]) * CHANNEL_SCALES[ch]
        mean_err = abs_err.mean(axis=(1, 2, 3))
        max_err = abs_err.max(axis=(1, 2, 3))
        axes[ch].plot(timesteps, mean_err, label='Mean', color=CHANNEL_COLORS[ch])
        axes[ch].fill_between(timesteps, mean_err, max_err, alpha=0.2,
                              color=CHANNEL_COLORS[ch], label='Max')
        axes[ch].set_xlabel('Timestep')
        axes[ch].set_ylabel(f'Absolute Error ({CHANNEL_UNITS[ch]})')
        axes[ch].set_title(CHANNEL_NAMES[ch])
        axes[ch].legend(fontsize=8)
        axes[ch].grid(True, alpha=0.3)
    fig.suptitle(f'Absolute Error in Physical Units — Trajectory {traj_idx}', fontsize=14)
    fig.tight_layout()
    path = os.path.join(out_dir, f'physical_err_traj{traj_idx}.png')
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close(fig)


def plot_aux_pred_gt(pred_aux_all, datasets, idx, out_dir, traj_idx):
    """Plot pred vs GT for all 16 aux quantities (9 BHP + 7 energy rate).

    pred_aux_all: (156,16) normalized predictions — BHP[0:9], energy[9:16].
    """
    # Denormalize prediction to physical units
    pred_bhp = pred_aux_all[:, 7:16] * AUX_BHP_RANGE + AUX_BHP_MIN
    pred_energy = np.expm1(pred_aux_all[:, 0:7] * ENERGY_LOG_DENOM)

    gt_bhp, gt_energy = load_gt_aux(datasets, idx)

    timesteps = np.arange(1, N_STEPS + 1)
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    axes = axes.flatten()

    # Subplots 0-8: BHP per well (P1-P7, Inj1, Inj2)
    for w in range(9):
        ax = axes[w]
        ax.plot(timesteps, pred_bhp[:, w], color='tab:blue', label='Pred')
        ax.plot(timesteps, gt_bhp[:, w], color='black', linestyle='--', label='GT')
        ax.set_title(f'BHP — {WELL_NAMES[w]}')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('BHP (kPa)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Subplots 9-15: energy rate per production well (P1-P7)
    for w in range(7):
        ax = axes[9 + w]
        ax.plot(timesteps, pred_energy[:, w], color='tab:red', label='Pred')
        ax.plot(timesteps, gt_energy[:, w], color='black', linestyle='--', label='GT')
        ax.set_title(f'Energy Rate — {ENERGY_WELL_NAMES[w]}')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Energy Rate')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Aux Pred vs GT — Trajectory {traj_idx}', fontsize=14)
    fig.tight_layout()
    path = os.path.join(out_dir, f'aux_pred_gt_traj{traj_idx}.png')
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close(fig)


def make_timestep_gif(pred_all, gt_all, sl, out_dir, traj_idx):
    """Plot 2: GIF of pred vs GT slices over time (slower transition)."""
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))

    vmins, vmaxs = [], []
    for ch in range(4):
        vmin = min(gt_all[:, ch, sl].min(), pred_all[:, ch, sl].min())
        vmax = max(gt_all[:, ch, sl].max(), pred_all[:, ch, sl].max())
        vmins.append(vmin)
        vmaxs.append(vmax)

    im_objects = []
    for ch in range(4):
        im_gt = axes[0, ch].imshow(gt_all[0, ch, sl], vmin=vmins[ch], vmax=vmaxs[ch], aspect='auto')
        im_pred = axes[1, ch].imshow(pred_all[0, ch, sl], vmin=vmins[ch], vmax=vmaxs[ch], aspect='auto')
        im_objects.append((im_gt, im_pred))
        fig.colorbar(im_gt, ax=[axes[0, ch], axes[1, ch]], fraction=0.046, pad=0.04)

    for ch in range(4):
        axes[0, ch].set_title(f'GT {CHANNEL_NAMES[ch]}')
        axes[1, ch].set_title(f'Pred {CHANNEL_NAMES[ch]}')

    fig.suptitle(f'Timestep 0 / {N_STEPS}', fontsize=16)

    def update(frame_t):
        for ch in range(4):
            im_objects[ch][0].set_data(gt_all[frame_t, ch, sl])
            im_objects[ch][1].set_data(pred_all[frame_t, ch, sl])
        fig.suptitle(f'Timestep {frame_t} / {N_STEPS}', fontsize=16)

    ani = animation.FuncAnimation(fig, update, frames=range(0, N_STEPS + 1),
                                  interval=200, repeat=True)
    path = os.path.join(out_dir, f'timestep_traj{traj_idx}.gif')
    ani.save(path, writer='pillow', fps=5)
    print(f"Saved: {path}")
    plt.close(fig)


def make_action_pred_gt_gif(pred_all, gt_all, well_rates, sl, out_dir, traj_idx):
    """Plot 3: Combined GIF — action rates + pred + GT + signed error.

    Layout (4 rows):
      Row 1: 9 subplots — per-well rate vs time (line builds up)
      Row 2: 4 subplots — predicted field slices
      Row 3: 4 subplots — GT field slices
      Row 4: 4 subplots — signed unnormalized error (pred − GT)
    """
    fig = plt.figure(figsize=(28, 22))
    gs = fig.add_gridspec(4, 36, hspace=0.35, wspace=0.4)

    ax_actions = []
    for w in range(9):
        ax = fig.add_subplot(gs[0, w * 4: w * 4 + 4])
        ax_actions.append(ax)

    ax_pred = [fig.add_subplot(gs[1, ch * 9: ch * 9 + 9]) for ch in range(4)]
    ax_gt = [fig.add_subplot(gs[2, ch * 9: ch * 9 + 9]) for ch in range(4)]
    ax_err = [fig.add_subplot(gs[3, ch * 9: ch * 9 + 9]) for ch in range(4)]

    timesteps = np.arange(1, N_STEPS + 1)

    action_lines = []
    action_vlines = []
    for w in range(9):
        ax = ax_actions[w]
        ax.set_xlim(0, N_STEPS + 1)
        ax.set_ylim(0, max(well_rates[w].max() * 1.1, 1.0))
        ax.set_title(WELL_NAMES[w], fontsize=9)
        ax.grid(True, alpha=0.2)
        if w == 0:
            ax.set_ylabel('Rate', fontsize=8)
        ax.tick_params(labelsize=7)
        line, = ax.plot([], [], color='tab:green', linewidth=1.2)
        vline = ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
        action_lines.append(line)
        action_vlines.append(vline)

    vmins_field, vmaxs_field = [], []
    for ch in range(4):
        vmin = min(gt_all[:, ch, sl].min(), pred_all[:, ch, sl].min())
        vmax = max(gt_all[:, ch, sl].max(), pred_all[:, ch, sl].max())
        vmins_field.append(vmin)
        vmaxs_field.append(vmax)

    err_all = (pred_all - gt_all) * np.array(CHANNEL_SCALES).reshape(1, 4, 1, 1, 1)
    vmins_err, vmaxs_err = [], []
    for ch in range(4):
        emin = err_all[:, ch, sl].min()
        emax = err_all[:, ch, sl].max()
        elim = max(abs(emin), abs(emax))
        vmins_err.append(-elim)
        vmaxs_err.append(elim)

    im_pred_objs, im_gt_objs, im_err_objs = [], [], []
    for ch in range(4):
        imp = ax_pred[ch].imshow(pred_all[0, ch, sl], vmin=vmins_field[ch], vmax=vmaxs_field[ch],
                                 aspect='auto', cmap='viridis')
        img = ax_gt[ch].imshow(gt_all[0, ch, sl], vmin=vmins_field[ch], vmax=vmaxs_field[ch],
                               aspect='auto', cmap='viridis')
        ime = ax_err[ch].imshow(err_all[0, ch, sl], vmin=vmins_err[ch], vmax=vmaxs_err[ch],
                                aspect='auto', cmap='RdBu_r')
        im_pred_objs.append(imp)
        im_gt_objs.append(img)
        im_err_objs.append(ime)

        ax_pred[ch].set_title(f'Pred {CHANNEL_NAMES[ch]}', fontsize=10)
        ax_gt[ch].set_title(f'GT {CHANNEL_NAMES[ch]}', fontsize=10)
        ax_err[ch].set_title(f'Error {CHANNEL_NAMES[ch]} ({CHANNEL_UNITS[ch]})', fontsize=10)

        fig.colorbar(imp, ax=ax_pred[ch], fraction=0.046, pad=0.04)
        fig.colorbar(img, ax=ax_gt[ch], fraction=0.046, pad=0.04)
        fig.colorbar(ime, ax=ax_err[ch], fraction=0.046, pad=0.04)

    fig.suptitle(f'Trajectory {traj_idx} — Timestep 0 / {N_STEPS}', fontsize=16)

    def update(frame_t):
        for w in range(9):
            end = min(frame_t, N_STEPS)
            if end > 0:
                action_lines[w].set_data(timesteps[:end], well_rates[w, :end])
            else:
                action_lines[w].set_data([], [])
            action_vlines[w].set_xdata([frame_t])

        for ch in range(4):
            im_pred_objs[ch].set_data(pred_all[frame_t, ch, sl])
            im_gt_objs[ch].set_data(gt_all[frame_t, ch, sl])
            im_err_objs[ch].set_data(err_all[frame_t, ch, sl])

        fig.suptitle(f'Trajectory {traj_idx} — Timestep {frame_t} / {N_STEPS}', fontsize=16)

    ani = animation.FuncAnimation(fig, update, frames=range(0, N_STEPS + 1),
                                  interval=200, repeat=True)
    path = os.path.join(out_dir, f'action_fields_traj{traj_idx}.gif')
    ani.save(path, writer='pillow', fps=5)
    print(f"Saved: {path}")
    plt.close(fig)


# ── Eval-all mode ────────────────────────────────────────────────────────────
def eval_all(model, aux_head_model, datasets, device, sl, out_dir):
    """Run rollout on all test trajectories and produce aggregate plots."""
    n_test = len(TEST_INDICES)
    all_l2 = np.zeros((n_test, 4, N_STEPS))
    all_signed_err_mean = np.zeros((n_test, 4, N_STEPS))

    for i, idx in enumerate(TEST_INDICES):
        print(f"[{i + 1}/{n_test}] Trajectory {idx}")
        gt_all = normalize_trajectory(datasets, idx)
        pred_all, l2_rel, _ = autoregressive_rollout(model, aux_head_model, gt_all, datasets, idx, device)

        for ch in range(4):
            all_l2[i, ch] = l2_rel[ch]

        for ch in range(4):
            signed_err = (pred_all[1:, ch] - gt_all[1:, ch]) * CHANNEL_SCALES[ch]
            all_signed_err_mean[i, ch] = signed_err.mean(axis=(1, 2, 3))

    timesteps = np.arange(1, N_STEPS + 1)

    mean_l2 = all_l2.mean(axis=0)
    std_l2 = all_l2.std(axis=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    for ch in range(4):
        ax.plot(timesteps, mean_l2[ch], label=CHANNEL_NAMES[ch], color=CHANNEL_COLORS[ch])
        ax.fill_between(timesteps,
                        mean_l2[ch] - std_l2[ch],
                        mean_l2[ch] + std_l2[ch],
                        alpha=0.15, color=CHANNEL_COLORS[ch])
    ax.set_xlabel('Timestep')
    ax.set_ylabel('L2 Relative Error')
    ax.set_title(f'Mean L2 Relative Error (n={n_test} test trajectories)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, 'eval_mean_l2_rel.png')
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close(fig)

    mean_signed = all_signed_err_mean.mean(axis=0)
    std_signed = all_signed_err_mean.std(axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for ch in range(4):
        ax = axes[ch]
        ax.plot(timesteps, mean_signed[ch], color=CHANNEL_COLORS[ch])
        ax.fill_between(timesteps,
                        mean_signed[ch] - std_signed[ch],
                        mean_signed[ch] + std_signed[ch],
                        alpha=0.15, color=CHANNEL_COLORS[ch])
        ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax.set_xlabel('Timestep')
        ax.set_ylabel(f'Error ({CHANNEL_UNITS[ch]})')
        ax.set_title(f'{CHANNEL_NAMES[ch]}')
        ax.grid(True, alpha=0.3)
    fig.suptitle(f'Mean Signed Error in Physical Units (n={n_test} test trajectories)', fontsize=14)
    fig.tight_layout()
    path = os.path.join(out_dir, 'eval_mean_signed_error.png')
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='AR-FNO Hetero Inference')
    parser.add_argument('--config', type=str, default='configs/loglo_hetero.yml',
                        help='Path to YAML config (must match hetero training)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--traj_idx', type=int, default=350,
                        help='Trajectory index to visualize (single-sample mode)')
    parser.add_argument('--slice_z', type=int, default=10,
                        help='Z-slice index for spatial visualization')
    parser.add_argument('--use_ema', action='store_true', default=False,
                        help='Use EMA model weights')
    parser.add_argument('--out_dir', type=str, default='inference_out_hetero',
                        help='Output directory for all plots and GIFs')
    parser.add_argument('--eval_all', action='store_true', default=False,
                        help='Evaluate all test trajectories (aggregate plots, no GIFs)')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    data_path = cfg['data']['path']
    device = get_device()
    print(f"Device: {device}")

    os.makedirs(args.out_dir, exist_ok=True)

    datasets = load_datasets(data_path)
    model, aux_head_model = load_model(cfg, args.checkpoint, device, use_ema=args.use_ema)

    if args.eval_all:
        print(f"Running eval-all on {len(TEST_INDICES)} test trajectories...")
        eval_all(model, aux_head_model, datasets, device, args.slice_z, args.out_dir)
    else:
        idx = args.traj_idx
        print(f"Running rollout for trajectory {idx}...")
        gt_all = normalize_trajectory(datasets, idx)
        pred_all, l2_rel, pred_aux_all = autoregressive_rollout(model, aux_head_model, gt_all, datasets, idx, device)
        well_rates = extract_well_rates(datasets, idx)

        sl = args.slice_z
        plot_l2_relative(l2_rel, args.out_dir, idx)
        plot_physical_error(pred_all, gt_all, args.out_dir, idx)
        plot_aux_pred_gt(pred_aux_all, datasets, idx, args.out_dir, idx)
        make_timestep_gif(pred_all, gt_all, sl, args.out_dir, idx)
        make_action_pred_gt_gif(pred_all, gt_all, well_rates, sl, args.out_dir, idx)

    print("Done.")


if __name__ == "__main__":
    main()
