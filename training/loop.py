import copy
import math
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
import matplotlib.pyplot as plt

from losses import H1Loss, LpLoss
from models.unet3d import UNet3D
from models.fno_wrapper import FNOWrapper
from models.loglo_fno import LOGLO_FNO, VanillaLOGLO_FNO
from models.transolver3d import TransolverWrapper
from models.aux_head import AuxHead

from training.physics import (
    extract_physical_porosity, compute_mbe_loss,
    mean_field_pressure_loss, add_adaptive_noise,
    radial_binned_spectral_loss,
)
from training.dataset import ARDataset
from training.utils import (
    set_seed, seed_worker, get_device,
    build_action_for_mbe, update_ema,
    capture_rng_states, restore_rng_states,
)
from training.model_adapters import create_adapter


def _clean_state_dict(module):
    sd = module.state_dict()
    sd.pop('_metadata', None)
    return sd


def _save_weights_checkpoint(path, *, model, ema_model, aux_head, ema_aux,
                             epoch, global_step, rng_states, wandb_run_id):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp_path = path + '.tmp'
    torch.save({
        'model': _clean_state_dict(model),
        'ema_model': _clean_state_dict(ema_model),
        'aux_head': aux_head.state_dict(),
        'ema_aux': ema_aux.state_dict(),
        'epoch': epoch,
        'global_step': global_step,
        'rng_states': rng_states,
        'wandb_run_id': wandb_run_id,
    }, tmp_path)
    os.replace(tmp_path, path)
    print(f"[CHECKPOINT] Weights saved: {path}  (epoch={epoch+1}, step={global_step})")


def _save_optimizer_state(path, *, optimizer, scheduler, epoch):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp_path = path + '.tmp'
    torch.save({
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
    }, tmp_path)
    os.replace(tmp_path, path)
    print(f"[CHECKPOINT] Optimizer saved: {path}  (epoch={epoch+1})")


WELL_COORDS = [
    [31, 15], [45, 4], [56, 15], [45, 27], [18, 27],
    [4, 15], [18, 4], [18, 15], [45, 15],
]


def create_model(model_cfg, model_type):
    if model_type == 'unet3d':
        return UNet3D(
            in_channels=model_cfg['in_channels'],
            out_channels=model_cfg['out_channels'],
            hidden_channels=model_cfg['hidden_channels'],
            depth=model_cfg.get('depth', 3),
            channel_multipliers=model_cfg.get('channel_multipliers', None),
        )
    elif model_type == 'fno':
        return FNOWrapper(
            n_modes=model_cfg['n_modes'],
            in_channels=model_cfg['in_channels'],
            out_channels=model_cfg['out_channels'],
            n_layers=model_cfg['n_layers'],
            hidden_channels=model_cfg['hidden_channels'],
        )
    elif model_type == 'loglo':
        return LOGLO_FNO(
            in_dim=model_cfg['in_dim'],
            out_dim=model_cfg['out_dim'],
            lifting_dim=model_cfg['lifting_dim'],
            projection_dim=model_cfg['projection_dim'],
            hidden_dim=model_cfg['hidden_dim'],
            n_blocks=model_cfg['n_blocks'],
            action_channels=model_cfg['action_channels'],
        )
    elif model_type == 'vanilla_loglo':
        return VanillaLOGLO_FNO(
            in_dim=model_cfg['in_dim'],
            out_dim=model_cfg['out_dim'],
            lifting_dim=model_cfg['lifting_dim'],
            projection_dim=model_cfg['projection_dim'],
            hidden_dim=model_cfg['hidden_dim'],
            n_blocks=model_cfg['n_blocks'],
        )
    elif model_type == 'transolver':
        return TransolverWrapper(
            in_channels=model_cfg['in_channels'],
            out_channels=model_cfg['out_channels'],
            hidden_dim=model_cfg['hidden_dim'],
            n_layers=model_cfg['n_layers'],
            n_head=model_cfg['n_head'],
            slice_num=model_cfg.get('slice_num', 32),
            mlp_ratio=model_cfg.get('mlp_ratio', 2),
            H=model_cfg.get('H', 16),
            W=model_cfg.get('W', 64),
            D=model_cfg.get('D', 32),
            spatial_embed=model_cfg.get('spatial_embed', True),
            num_bands=model_cfg.get('num_bands', 32),
            max_freq=model_cfg.get('max_freq', 64.0),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_out_channels(model_cfg):
    return model_cfg.get('out_channels', model_cfg.get('out_dim', 4))


def get_hidden_dim(model_cfg):
    return model_cfg.get('hidden_channels', model_cfg.get('hidden_dim', 64))


def run_training(cfg, args, resume_path=None):
    seed = args.seed
    set_seed(seed)
    print(f"Seed set to {seed}")

    model_cfg = cfg['model']
    model_type = model_cfg['type']
    heterogeneous = cfg['data'].get('heterogeneous', False)
    variant = 'hetero' if heterogeneous else 'homo'

    train_cfg = cfg['training']['hpc'] if args.hpc else cfg['training']['local']
    batch_size = train_cfg['batch_size']
    lr = train_cfg['lr']
    num_epochs = train_cfg['num_epochs']
    test_batch_size = train_cfg['test_batch_size']

    weight_decay = cfg['training']['weight_decay']
    ema_decay = cfg['training']['ema_decay']
    log_every = cfg['training']['log_every']
    grad_clip_norm = cfg['training']['grad_clip_norm']
    noise_alpha = cfg['training'].get('noise_alpha', [0.0025, 0.0025, 0.025, 0.025])
    k_max = cfg['training'].get('pushforward_k_max', 5)
    k_step_interval = cfg['training'].get('pushforward_k_step_interval', 300)
    warmup_steps = cfg['training'].get('warmup_steps', 0)
    min_lr = cfg['training'].get('min_lr', 1e-5)
    use_pushforward = cfg['training'].get('use_pushforward', True)
    log_scalar_every = cfg['logging'].get('log_scalar_every', 10)

    use_mse = cfg['loss'].get('use_mse', True)
    mse_weight = cfg['loss'].get('mse_weight', 1.0)
    use_h1 = cfg['loss'].get('use_h1', True)
    h1_weight = cfg['loss'].get('h1_weight', 1.0)
    use_aux = cfg['loss'].get('use_aux', True)
    aux_weight = cfg['loss'].get('aux_weight', 0.1)
    channel_weights = cfg['loss']['channel_weights']
    use_mbe = cfg['loss'].get('use_mbe', True)
    mbe_weight = cfg['loss'].get('mbe_weight', 1.0)
    use_spectral = cfg['loss'].get('use_spectral', True)
    spectral_weight = cfg['loss'].get('spectral_weight', 0.0)
    spectral_iLow = cfg['loss'].get('spectral_iLow', 4)
    spectral_iHigh = cfg['loss'].get('spectral_iHigh', 12)
    use_meanfield = cfg['loss'].get('use_meanfield', True)
    meanfield_weight = cfg['loss'].get('meanfield_weight', 1.0)

    WRITER = cfg['logging']['writer']

    running_dir = cfg['checkpoints']['running_dir']
    final_path = cfg['checkpoints']['final_path']
    resume_ckpt_path = cfg['checkpoints'].get('resume_path',
        os.path.join(running_dir, 'resume_checkpoint.pt'))
    optim_ckpt_path = resume_ckpt_path.replace('.pt', '_optim.pt')
    save_every = cfg['checkpoints'].get('save_every', log_every * 5)
    epochs_per_run = cfg['training'].get('epochs_per_run', num_epochs)

    # --- Idempotency guard ---
    # The resume checkpoint is the completion signal, but it's deleted once the
    # final model is saved. Without this, a requeued/re-launched job for an
    # already-finished experiment would find no resume ckpt, start from scratch,
    # and overwrite final_path. final_path is seed-specific, so this is per-seed.
    if os.path.isfile(final_path):
        print(f"[DONE] Final model already exists at {final_path}, skipping.")
        return

    # --- Data loading ---
    mode = None if args.hpc else 'r'
    data_path = cfg['data']['path']
    dataset_temp = np.load(os.path.join(data_path, 'all_temp_formation.npy'), mmap_mode=mode)
    dataset_pres = np.load(os.path.join(data_path, 'all_pres_formation.npy'), mmap_mode=mode)
    dataset_action = np.load(os.path.join(data_path, 'all_action.npy'), mmap_mode=mode)
    dataset_temp_frac = np.load(os.path.join(data_path, 'all_temp_frac.npy'), mmap_mode=mode)
    dataset_pres_frac = np.load(os.path.join(data_path, 'all_pres_frac.npy'), mmap_mode=mode)
    dataset_aux = np.load(os.path.join(data_path, 'all_energyrate_bhp.npy'), mmap_mode=mode)

    extra_kwargs = {}
    if heterogeneous:
        extra_kwargs = dict(
            heterogeneous=True,
            por_matrix=np.load(os.path.join(data_path, 'all_por_matrix.npy'), mmap_mode=mode),
            por_frac=np.load(os.path.join(data_path, 'all_por_frac.npy'), mmap_mode=mode),
            perm_matrix=np.load(os.path.join(data_path, 'all_perm_matrix.npy'), mmap_mode=mode),
            perm_frac=np.load(os.path.join(data_path, 'all_perm_frac.npy'), mmap_mode=mode),
        )

    custom_ds = ARDataset(
        dataset_temp, dataset_temp_frac, dataset_pres, dataset_pres_frac,
        dataset_action, dataset_aux, k_max=k_max, **extra_kwargs,
    )

    print("Dataset:", custom_ds.n_trajectories, "trajectories,",
          "heterogeneous" if heterogeneous else "homogeneous")

    train_index = list(range(300))
    test_index = list(range(350, 400))
    train_ds = torch.utils.data.Subset(custom_ds, train_index)
    test_ds = torch.utils.data.Subset(custom_ds, test_index)

    device = get_device()

    loader_gen = torch.Generator()
    loader_gen.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              generator=loader_gen, worker_init_fn=seed_worker)
    test_loader = DataLoader(test_ds, batch_size=test_batch_size, shuffle=True,
                             generator=loader_gen, worker_init_fn=seed_worker)

    # --- Model setup ---
    model = create_model(model_cfg, model_type).to(device)

    out_channels = get_out_channels(model_cfg)
    hidden_dim = get_hidden_dim(model_cfg)

    aux_head_model = AuxHead(
        state_channels=out_channels,
        depth=16,
        aux_channels=model_cfg.get('aux_channels', 16),
        hidden_dim=hidden_dim,
    ).to(device)

    ema_model = copy.deepcopy(model)
    ema_model.requires_grad_(False)
    ema_model.eval()

    ema_aux = copy.deepcopy(aux_head_model)
    ema_aux.requires_grad_(False)
    ema_aux.eval()

    adapter = create_adapter(model_type, heterogeneous)

    # --- Loss functions ---
    mse_fn = nn.MSELoss()
    h1_loss_fn = H1Loss(d=3, reduction='none',
                        fix_x_bnd=True, fix_y_bnd=True, fix_z_bnd=True,
                        measure=[0.25, 1., 0.5]).abs
    l2_rel = LpLoss(d=3, p=2, reduction='mean', measure=[0.25, 1., 0.5])

    _channel_weights = torch.tensor(channel_weights)

    def calculate_weighted_mse_loss(pred, target):
        diff = (pred - target) ** 2
        weights = _channel_weights.to(pred.device)
        return torch.mean(diff * weights.view(1, -1, 1, 1, 1))

    def calculate_weighted_h1_loss(pred, target):
        per_channel = h1_loss_fn(pred, target)
        weights = _channel_weights.to(pred.device)
        return (per_channel * weights).mean()

    def get_mbe_porosity(static):
        if heterogeneous:
            return extract_physical_porosity(static)
        return 0.05, 0.005

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(aux_head_model.parameters()),
        lr=lr, weight_decay=weight_decay)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * num_epochs

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        min_ratio = min_lr / lr
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # --- Resume detection ---
    _resume_ckpt = None
    _optim_ckpt = None
    start_epoch = 0
    global_step = 0
    wandb_run_id = None

    if resume_path is not None and os.path.isfile(resume_path):
        print(f"[RESUME] Loading checkpoint: {resume_path}")
        _resume_ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        start_epoch = _resume_ckpt['epoch'] + 1
        global_step = _resume_ckpt['global_step']
        wandb_run_id = _resume_ckpt.get('wandb_run_id')
        print(f"[RESUME] Will resume from epoch {start_epoch}, global_step {global_step}")

        if 'optimizer' in _resume_ckpt:
            _optim_ckpt = _resume_ckpt
            print("[RESUME] Legacy checkpoint with embedded optimizer state.")
        elif os.path.isfile(optim_ckpt_path):
            _optim_ckpt = torch.load(optim_ckpt_path, map_location=device, weights_only=False)
            if _optim_ckpt['epoch'] != _resume_ckpt['epoch']:
                print(f"[RESUME] Optimizer stale (epoch {_optim_ckpt['epoch']+1} vs "
                      f"weights epoch {start_epoch}), restarting optimizer.")
                _optim_ckpt = None
            else:
                print("[RESUME] Loaded matching optimizer state.")
        else:
            print("[RESUME] No optimizer checkpoint, restarting optimizer.")
    elif resume_path is not None:
        print(f"[RESUME] No checkpoint at {resume_path}, starting fresh.")

    if start_epoch >= num_epochs:
        print(f"[RESUME] Training already complete ({start_epoch}/{num_epochs} epochs). Exiting.")
        return

    end_epoch = min(start_epoch + epochs_per_run, num_epochs)

    # --- W&B ---
    if WRITER:
        run_tag = cfg['logging'].get('run_tag', '')
        run_name = f"{model_type}{run_tag}-{variant}-seed{seed}"
        init_kwargs = dict(
            project=cfg['logging'].get('wandb_project', f'LOGLOFNO_{variant.upper()}_exp'),
            entity=cfg['logging'].get('wandb_entity', None),
            name=run_name,
            config=cfg,
            tags=[model_type, variant],
            group=model_type,
        )
        if wandb_run_id is not None:
            init_kwargs['id'] = wandb_run_id
            init_kwargs['resume'] = 'must'
        wandb.init(**init_kwargs)
        if wandb_run_id is None:
            wandb.config.update({"seed": seed, "hpc": args.hpc})
            if cfg['logging'].get('watch_model', False):
                watch_freq = cfg['logging'].get('watch_freq', 1000)
                wandb.watch(model, log="all", log_freq=watch_freq)
                wandb.watch(aux_head_model, log="all", log_freq=watch_freq)
        wandb_run_id = wandb.run.id

    # --- Apply resume state ---
    optimizer_restored = False
    if _resume_ckpt is not None:
        _resume_ckpt['model'].pop('_metadata', None)
        _resume_ckpt['ema_model'].pop('_metadata', None)
        model.load_state_dict(_resume_ckpt['model'])
        ema_model.load_state_dict(_resume_ckpt['ema_model'])
        aux_head_model.load_state_dict(_resume_ckpt['aux_head'])
        ema_aux.load_state_dict(_resume_ckpt['ema_aux'])
        restore_rng_states(_resume_ckpt['rng_states'], loader_gen)
        if _optim_ckpt is not None:
            optimizer.load_state_dict(_optim_ckpt['optimizer'])
            scheduler.load_state_dict(_optim_ckpt['scheduler'])
            optimizer_restored = True
            if _optim_ckpt is not _resume_ckpt:
                del _optim_ckpt
        del _resume_ckpt
        if not optimizer_restored and global_step > 0:
            for _ in range(global_step):
                scheduler.step()
            print(f"[RESUME] Scheduler advanced to step {global_step}, "
                  f"LR={scheduler.get_last_lr()[0]:.2e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- Training loop ---
    print(f"Training epochs {start_epoch+1} to {end_epoch} (of {num_epochs} total)")

    for epoch in range(start_epoch, end_epoch):
        model.train()

        for batch_idx, batch in enumerate(train_loader):
            if heterogeneous:
                y_history, action_history, valid_k, y_t, y_tp1, action_t, aux_t, aux_tp1, static = batch
                static = static.to(device)
            else:
                y_history, action_history, valid_k, y_t, y_tp1, action_t, aux_t, aux_tp1 = batch
                static = None

            k_upper = min(k_max, 1 + global_step // k_step_interval)
            k = np.random.randint(1, k_upper + 1)

            for well in WELL_COORDS:
                action_history[:, :, 0:2, well[0], well[1]] = 0.
                action_t[:, 0:2, well[0], well[1]] = 0.

            y_history = y_history.to(device)
            action_history = action_history.to(device)
            valid_k = valid_k.to(device)
            y_t = y_t.to(device)
            y_tp1 = y_tp1.to(device)
            action_t = action_t.to(device)
            aux_t = aux_t.to(device)
            aux_tp1 = aux_tp1.to(device)

            if torch.rand(1).item() < 0.8:
                y_noisy = add_adaptive_noise(y_t, alpha=noise_alpha)
            else:
                y_noisy = y_t

            model_input = adapter.build_model_input(y_noisy, action_t, static)
            predicted_y = adapter.forward(model, model_input)
            predicted_aux = aux_head_model(y_t, predicted_y)

            loss = torch.tensor(0.0, device=device)

            if use_mse:
                loss_mse = calculate_weighted_mse_loss(predicted_y, y_tp1)
                loss = loss + mse_weight * loss_mse
            else:
                loss_mse = torch.tensor(0.0, device=device)

            if use_h1:
                loss_h1 = calculate_weighted_h1_loss(predicted_y, y_tp1)
                loss = loss + h1_weight * loss_h1
            else:
                loss_h1 = torch.tensor(0.0, device=device)

            if use_aux:
                loss_aux = aux_weight * mse_fn(predicted_aux, aux_tp1)
                loss = loss + loss_aux
            else:
                loss_aux = torch.tensor(0.0, device=device)

            if use_mbe:
                phi_m, phi_frac = get_mbe_porosity(static)
                action_mbe = build_action_for_mbe(action_t)
                loss_mbe = mbe_weight * compute_mbe_loss(
                    y_t, predicted_y, action_mbe, phi_m, phi_frac,
                    pres_min=custom_ds.pres_min, pres_max=custom_ds.pres_max)
                loss = loss + loss_mbe
            else:
                loss_mbe = torch.tensor(0.0, device=device)

            if use_meanfield:
                loss_meanfield = meanfield_weight * mean_field_pressure_loss(predicted_y, y_tp1)
                loss = loss + loss_meanfield
            else:
                loss_meanfield = torch.tensor(0.0, device=device)

            if use_spectral and spectral_weight > 0:
                loss_spectral, spectral_bands = radial_binned_spectral_loss(
                    predicted_y, y_tp1, iLow=spectral_iLow, iHigh=spectral_iHigh)
                loss = loss + spectral_weight * loss_spectral
            else:
                loss_spectral = torch.tensor(0.0, device=device)
                spectral_bands = torch.zeros(3, device=device)

            # --- Pushforward loss ---
            loss_pf = torch.tensor(0.0, device=device)
            if use_pushforward and k > 0:
                start_idx = k_max - k
                pf_mask = valid_k[:, start_idx].bool()

                if pf_mask.any():
                    static_pf = static[pf_mask] if static is not None else None

                    # eval() during rollout so dropout (Transolver) doesn't
                    # inject extra noise — we want a clean drift-error estimate.
                    with torch.no_grad():
                        model.eval()
                        y_pf = y_history[pf_mask, start_idx]
                        for i in range(k):
                            pf_input_i = adapter.build_model_input(
                                y_pf, action_history[pf_mask, start_idx + i], static_pf)
                            y_pf = adapter.forward(model, pf_input_i)
                        model.train()

                    y_pf = y_pf.detach()
                    target_pf = y_tp1[pf_mask]

                    pred_pf = adapter.forward(
                        model,
                        adapter.build_model_input(y_pf, action_t[pf_mask], static_pf))
                    pred_pf_aux = aux_head_model(y_pf, pred_pf)

                    if use_mse:
                        loss_pf = loss_pf + mse_weight * calculate_weighted_mse_loss(pred_pf, target_pf)
                    if use_h1:
                        loss_pf = loss_pf + h1_weight * calculate_weighted_h1_loss(pred_pf, target_pf)
                    if use_aux:
                        loss_pf = loss_pf + aux_weight * mse_fn(pred_pf_aux, aux_tp1[pf_mask])
                    if use_mbe:
                        phi_m_pf, phi_frac_pf = get_mbe_porosity(static_pf)
                        action_mbe_pf = build_action_for_mbe(action_t[pf_mask])
                        loss_pf = loss_pf + mbe_weight * compute_mbe_loss(
                            y_pf, pred_pf, action_mbe_pf, phi_m_pf, phi_frac_pf,
                            pres_min=custom_ds.pres_min, pres_max=custom_ds.pres_max)
                    if use_meanfield:
                        loss_pf = loss_pf + meanfield_weight * mean_field_pressure_loss(pred_pf, target_pf)
                    if use_spectral and spectral_weight > 0:
                        loss_pf_spec, _ = radial_binned_spectral_loss(
                            pred_pf, target_pf, iLow=spectral_iLow, iHigh=spectral_iHigh)
                        loss_pf = loss_pf + spectral_weight * loss_pf_spec

                    loss = loss + loss_pf

            with torch.no_grad():
                loss_l2_rel = l2_rel(predicted_y, y_tp1)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(aux_head_model.parameters()),
                max_norm=grad_clip_norm)
            optimizer.step()
            scheduler.step()
            update_ema(ema_model, model, ema_aux, aux_head_model, ema_decay)

            if WRITER:
                if global_step % log_scalar_every == 0:
                    wandb.log({
                        'Loss/MSE': loss_mse.item(),
                        'Loss/H1': loss_h1.item(),
                        'Loss/Aux': loss_aux.item(),
                        'Loss/L2_Rel': loss_l2_rel.item(),
                        'Loss/MBE': loss_mbe.item(),
                        'Loss/Spectral_Low': spectral_bands[0].item(),
                        'Loss/Spectral_Mid': spectral_bands[1].item(),
                        'Loss/Spectral_High': spectral_bands[2].item(),
                        'Loss/MeanField': loss_meanfield.item(),
                        'Loss/Total': loss.item(),
                        'Loss/Pushforward': loss_pf.item(),
                        'Training/k': k,
                        'Training/lr': optimizer.param_groups[0]['lr'],
                    }, step=global_step)

                if epoch % log_every == 0 and batch_idx == 0:
                    _run_validation(
                        model=ema_model, aux_model=ema_aux,
                        test_loader=test_loader, adapter=adapter,
                        mse_fn=mse_fn, h1_fn=calculate_weighted_h1_loss,
                        l2_rel=l2_rel, device=device,
                        heterogeneous=heterogeneous,
                        global_step=global_step,
                    )
                    model.train()
            else:
                print(f"Batch {batch_idx}, k={k}, "
                      f"LossMSE: {loss_mse.item():.5f}, LossH1: {loss_h1.item():.5f}, "
                      f"LossPF: {loss_pf.item():.5f}, LossSpec: {loss_spectral.item():.5f}")

            global_step += 1

        if (epoch + 1) % save_every == 0:
            _save_weights_checkpoint(
                resume_ckpt_path,
                model=model, ema_model=ema_model,
                aux_head=aux_head_model, ema_aux=ema_aux,
                epoch=epoch, global_step=global_step,
                rng_states=capture_rng_states(loader_gen),
                wandb_run_id=wandb_run_id,
            )
            _save_optimizer_state(optim_ckpt_path,
                optimizer=optimizer, scheduler=scheduler,
                epoch=epoch)
            print(f"Epoch {epoch + 1}, k_upper={k_upper}, "
                  f"LossMSE: {loss_mse.item():.5f}, LossH1: {loss_h1.item():.5f}")

    # --- End of segment ---
    training_complete = (end_epoch >= num_epochs)

    if training_complete:
        os.makedirs(os.path.dirname(final_path) or '.', exist_ok=True)
        torch.save({
            'model': model.state_dict(),
            'ema_model': ema_model.state_dict(),
            'aux_head': aux_head_model.state_dict(),
            'ema_aux': ema_aux.state_dict(),
        }, final_path)
        for f in [resume_ckpt_path, optim_ckpt_path]:
            if os.path.isfile(f):
                os.remove(f)
                print(f"[CHECKPOINT] Training complete. Removed {os.path.basename(f)}")
        if WRITER:
            try:
                art = wandb.Artifact(f"{model_type}{run_tag}-{variant}-final", type="model")
                art.add_file(final_path)
                wandb.log_artifact(art)
            except Exception as e:
                print(f"[WARNING] Final artifact upload failed: {e}")
            wandb.finish()
        print(f"[DONE] All {num_epochs} epochs complete.")
    else:
        _save_weights_checkpoint(
            resume_ckpt_path,
            model=model, ema_model=ema_model,
            aux_head=aux_head_model, ema_aux=ema_aux,
            epoch=end_epoch - 1, global_step=global_step,
            rng_states=capture_rng_states(loader_gen),
            wandb_run_id=wandb_run_id,
        )
        _save_optimizer_state(optim_ckpt_path,
            optimizer=optimizer, scheduler=scheduler,
            epoch=end_epoch - 1)
        if WRITER:
            wandb.finish(exit_code=0)
        print(f"[SEGMENT DONE] Epochs {start_epoch+1}-{end_epoch} of {num_epochs} complete.")


def _run_validation(*, model, aux_model, test_loader, adapter,
                    mse_fn, h1_fn, l2_rel, device, heterogeneous, global_step):
    model.eval()
    with torch.no_grad():
        batch = next(iter(test_loader))
        if heterogeneous:
            _, _, _, val_y_t, val_y_tp1, val_act_t, _, val_aux_tp1, val_static = batch
            val_static = val_static.to(device)
        else:
            _, _, _, val_y_t, val_y_tp1, val_act_t, _, val_aux_tp1 = batch
            val_static = None

        val_y_t = val_y_t.to(device)
        val_y_tp1 = val_y_tp1.to(device)
        val_act_t = val_act_t.to(device)
        val_aux_target = val_aux_tp1.to(device)

        for well in WELL_COORDS:
            val_act_t[:, 0:2, well[0], well[1]] = 0.

        val_input = adapter.build_model_input(val_y_t, val_act_t, val_static)
        val_pred = adapter.forward(model, val_input)
        val_pred_aux = aux_model(val_y_t, val_pred)

        loss_mse_val = mse_fn(val_pred, val_y_tp1)
        loss_h1_val = h1_fn(val_pred, val_y_tp1)
        loss_aux_val = mse_fn(val_pred_aux, val_aux_target)
        loss_l2_rel_val = l2_rel(val_pred, val_y_tp1)

        channel_names = ['T_form', 'T_frac', 'P_form', 'P_frac']
        val_log = {
            'Val_Loss/MSE': loss_mse_val.item(),
            'Val_Loss/H1': loss_h1_val.item(),
            'Val_Loss/Aux': loss_aux_val.item(),
            'Val_Loss/L2_Rel': loss_l2_rel_val.item(),
        }

        for i, ch_name in enumerate(channel_names):
            truth = val_y_tp1[0, i, 10, :, :].cpu().numpy()
            pred_np = val_pred[0, i, 10, :, :].cpu().numpy()
            error = pred_np - truth
            vmax = max(abs(error.min()), abs(error.max())) or 1.0

            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            axes[0].imshow(truth); axes[0].set_title(f'True {ch_name}'); axes[0].axis('off')
            axes[1].imshow(pred_np); axes[1].set_title(f'Pred {ch_name}'); axes[1].axis('off')
            im = axes[2].imshow(error, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            axes[2].set_title(f'Error {ch_name}'); axes[2].axis('off')
            fig.colorbar(im, ax=axes[2], fraction=0.046)
            plt.tight_layout()
            val_log[f'Val_Image/{ch_name}'] = wandb.Image(fig)
            plt.close(fig)

        wandb.log(val_log, step=global_step)
