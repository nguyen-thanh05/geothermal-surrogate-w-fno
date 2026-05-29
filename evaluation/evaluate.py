import gc
import json
import os
import sys
import argparse
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from losses import LpLoss
from models.aux_head import AuxHead
from training.loop import create_model, get_out_channels, get_hidden_dim
from training.model_adapters import create_adapter
from evaluation.constants import (
    get_constants, WELL_COORDS, N_STEPS, TEST_INDICES,
    CHANNEL_KEYS, CHANNEL_NAMES,
)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, 'mps') and torch.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model_universal(cfg, checkpoint_path, device, use_ema=True):
    model_cfg = cfg['model']
    model_type = model_cfg['type']
    heterogeneous = cfg['data'].get('heterogeneous', False)

    model = create_model(model_cfg, model_type)
    aux_head = AuxHead(
        state_channels=get_out_channels(model_cfg),
        depth=16,
        aux_channels=model_cfg.get('aux_channels', 16),
        hidden_dim=get_hidden_dim(model_cfg),
    )
    adapter = create_adapter(model_type, heterogeneous)

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model_key = 'ema_model' if use_ema else 'model'
    aux_key = 'ema_aux' if use_ema else 'aux_head'

    sd = ckpt[model_key]
    sd.pop('_metadata', None)
    model.load_state_dict(sd)
    aux_head.load_state_dict(ckpt[aux_key])

    metadata = {
        'epoch': ckpt.get('epoch', -1),
        'global_step': ckpt.get('global_step', -1),
    }
    del ckpt

    model.to(device).eval()
    aux_head.to(device).eval()

    return model, aux_head, adapter, metadata


def load_datasets(data_path, heterogeneous):
    load = lambda name: np.load(os.path.join(data_path, name), mmap_mode='r')
    ds = {
        'temp': load('all_temp_formation.npy'),
        'pres': load('all_pres_formation.npy'),
        'temp_frac': load('all_temp_frac.npy'),
        'pres_frac': load('all_pres_frac.npy'),
        'action': load('all_action.npy'),
        'aux': load('all_energyrate_bhp.npy'),
    }
    if heterogeneous:
        ds['por_matrix'] = load('all_por_matrix.npy')
        ds['por_frac'] = load('all_por_frac.npy')
        ds['perm_matrix'] = load('all_perm_matrix.npy')
        ds['perm_frac'] = load('all_perm_frac.npy')
    return ds


def normalize_trajectory(datasets, idx, nc):
    tf = (datasets['temp'][idx].astype(np.float32) - nc.TEMP_MIN) / nc.TEMP_RANGE
    tfr = (datasets['temp_frac'][idx].astype(np.float32) - nc.TEMP_MIN) / nc.TEMP_RANGE
    pf = (datasets['pres'][idx].astype(np.float32) - nc.PRES_MIN) / nc.PRES_RANGE
    pfr = (datasets['pres_frac'][idx].astype(np.float32) - nc.PRES_MIN) / nc.PRES_RANGE
    return np.stack([tf, tfr, pf, pfr], axis=1)


def get_static(datasets, idx, device, nc):
    pm = (datasets['por_matrix'][idx].astype(np.float32) - nc.POR_MIN_MATRIX) / (nc.POR_MAX_MATRIX - nc.POR_MIN_MATRIX)
    pf = (datasets['por_frac'][idx].astype(np.float32) - nc.POR_MIN_FRAC) / (nc.POR_MAX_FRAC - nc.POR_MIN_FRAC)
    km = (datasets['perm_matrix'][idx].astype(np.float32) - nc.PERM_MIN_MATRIX) / (nc.PERM_MAX_MATRIX - nc.PERM_MIN_MATRIX)
    kf = (datasets['perm_frac'][idx].astype(np.float32) - nc.PERM_MIN_FRAC) / (nc.PERM_MAX_FRAC - nc.PERM_MIN_FRAC)
    static = np.stack([pm, pf, km, kf], axis=0)
    return torch.from_numpy(static).unsqueeze(0).float().to(device)


def autoregressive_rollout(model, aux_head, adapter, gt_all, datasets, idx,
                           device, nc, heterogeneous):
    l2_loss = LpLoss(d=3, p=2, reduction='none')
    y_t = torch.from_numpy(gt_all[0]).unsqueeze(0).float().to(device)

    static = None
    if heterogeneous:
        static = get_static(datasets, idx, device, nc)

    l2_per_step = {ch: [] for ch in range(4)}
    mse_phys_accum = np.zeros((N_STEPS, 4))

    with torch.no_grad():
        for t in range(N_STEPS):
            action_np = datasets['action'][idx, t].astype(np.float32)
            action_t = torch.from_numpy(action_np.copy()).unsqueeze(0).float()
            action_t = (action_t - nc.ACTION_MIN) / nc.ACTION_RANGE

            for wx, wy in WELL_COORDS:
                action_t[:, 0:2, wx, wy] = 0.0

            action_t = action_t.to(device)

            model_input = adapter.build_model_input(y_t, action_t, static)
            predicted_y = adapter.forward(model, model_input)

            gt_tp1 = torch.from_numpy(gt_all[t + 1]).unsqueeze(0).float().to(device)

            for ch in range(4):
                err = l2_loss(predicted_y[:, ch:ch+1], gt_tp1[:, ch:ch+1]).item()
                l2_per_step[ch].append(err)

            pred_np = predicted_y[0].cpu().numpy()
            gt_np = gt_all[t + 1]
            scales = [nc.TEMP_RANGE, nc.TEMP_RANGE, nc.PRES_RANGE, nc.PRES_RANGE]
            for ch in range(4):
                diff_phys = (pred_np[ch] - gt_np[ch]) * scales[ch]
                mse_phys_accum[t, ch] = np.mean(diff_phys ** 2)

            y_t = predicted_y

    return l2_per_step, mse_phys_accum


def compute_metrics(all_l2, all_mse_phys):
    n_traj = all_l2.shape[0]
    timesteps = np.arange(1, N_STEPS + 1)

    mean_l2_per_traj = all_l2.mean(axis=2)
    mean_l2 = mean_l2_per_traj.mean(axis=0)

    final_l2_per_traj = all_l2[:, :, -1]
    final_l2 = final_l2_per_traj.mean(axis=0)

    drift_slopes = np.zeros((n_traj, 4))
    for i in range(n_traj):
        for ch in range(4):
            slope = np.polyfit(timesteps, all_l2[i, ch], 1)[0]
            drift_slopes[i, ch] = slope
    mean_drift = drift_slopes.mean(axis=0)

    rmse_phys = np.sqrt(all_mse_phys.mean(axis=(0, 2)))

    init_l2 = all_l2[:, :, 0]
    amplification = np.where(init_l2 > 1e-10,
                             final_l2_per_traj / init_l2,
                             np.nan)
    mean_amp = np.nanmean(amplification, axis=0)

    metrics = {}
    for ch, key in enumerate(CHANNEL_KEYS):
        metrics[key] = {
            'mean_l2_rel': float(mean_l2[ch]),
            'final_step_l2': float(final_l2[ch]),
            'drift_slope': float(mean_drift[ch]),
            'rmse_physical': float(rmse_phys[ch]),
            'amplification': float(mean_amp[ch]),
        }

    mean_l2_curve = all_l2.mean(axis=0)
    per_timestep = {}
    for ch, key in enumerate(CHANNEL_KEYS):
        per_timestep[key] = mean_l2_curve[ch].tolist()

    return metrics, per_timestep


def evaluate_checkpoint(config_path, checkpoint_path, data_path, out_path,
                        use_ema=True):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    heterogeneous = cfg['data'].get('heterogeneous', False)
    nc = get_constants(heterogeneous)
    device = get_device()
    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"  GPU mem: {torch.cuda.memory_allocated()/1e9:.2f}GB alloc, "
              f"{torch.cuda.memory_reserved()/1e9:.2f}GB reserved")

    model, aux_head, adapter, ckpt_meta = load_model_universal(
        cfg, checkpoint_path, device, use_ema)
    print(f"Loaded model (type={cfg['model']['type']}, "
          f"epoch={ckpt_meta['epoch']}, step={ckpt_meta['global_step']})")

    try:
        datasets = load_datasets(data_path, heterogeneous)

        n_test = len(TEST_INDICES)
        all_l2 = np.zeros((n_test, 4, N_STEPS))
        all_mse_phys = np.zeros((n_test, 4, N_STEPS))

        for i, idx in enumerate(TEST_INDICES):
            print(f"  [{i+1}/{n_test}] Trajectory {idx}")
            gt_all = normalize_trajectory(datasets, idx, nc)
            l2_per_step, mse_phys = autoregressive_rollout(
                model, aux_head, adapter, gt_all, datasets, idx,
                device, nc, heterogeneous)

            for ch in range(4):
                all_l2[i, ch] = l2_per_step[ch]
            all_mse_phys[i] = mse_phys.T

        metrics, per_timestep = compute_metrics(all_l2, all_mse_phys)

        hpc_epochs = cfg['training']['hpc']['num_epochs']
        training_complete = (ckpt_meta['epoch'] + 1) >= hpc_epochs

        result = {
            'config_path': config_path,
            'checkpoint_path': checkpoint_path,
            'model_type': cfg['model']['type'],
            'variant': 'hetero' if heterogeneous else 'homo',
            'use_ema': use_ema,
            'training_epoch': ckpt_meta['epoch'],
            'training_global_step': ckpt_meta['global_step'],
            'expected_epochs': hpc_epochs,
            'training_complete': training_complete,
            'num_test_trajectories': n_test,
            'metrics': metrics,
            'per_timestep_l2': per_timestep,
        }

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Results saved: {out_path}")

        return result
    finally:
        del model, aux_head, adapter
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description='Evaluate single checkpoint')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--out-dir', type=str, default='evaluation_results')
    parser.add_argument('--use-ema', action='store_true', default=True)
    parser.add_argument('--no-ema', action='store_true')
    parser.add_argument('--seed-label', type=str, default=None,
                        help='Seed label for output path (e.g. "seed42")')
    parser.add_argument('--model-label', type=str, default=None,
                        help='Model label for output filename')
    args = parser.parse_args()

    use_ema = not args.no_ema

    if args.seed_label and args.model_label:
        out_path = os.path.join(args.out_dir, args.seed_label,
                                f'{args.model_label}.json')
    else:
        ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
        ckpt_name = ckpt_name.replace('_resume', '')
        parent = os.path.basename(os.path.dirname(args.checkpoint))
        out_path = os.path.join(args.out_dir, parent, f'{ckpt_name}.json')

    evaluate_checkpoint(args.config, args.checkpoint, args.data_path,
                        out_path, use_ema=use_ema)


if __name__ == '__main__':
    main()
