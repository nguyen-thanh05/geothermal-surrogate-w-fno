# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Surrogate modeling for geothermal reservoir simulation using neural operators. Trains autoregressive (AR) models to predict next-timestep reservoir state given current state + well control actions, replacing expensive numerical simulators.

**Domain**: 3D dual-porosity geothermal reservoir (matrix + fracture). Grid: (16, 64, 32) = (depth, height, width).

**State channels** (4): temperature formation, temperature fracture, pressure formation, pressure fracture.

**Actions**: per-cell well injection/production rates (9 wells with fixed coordinates in `WELL_COORDS`).

**Auxiliary head**: predicts per-well BHP (9) and energy production rates (7) = 16 outputs total.

Two data variants: **homogeneous** (fixed porosity/permeability) and **heterogeneous** (per-realization static fields: por_matrix, por_frac, perm_matrix, perm_frac appended as 4 extra input channels).

## Commands

### Training
```bash
# Local (small batch, mmap data)
python training/train.py --config configs/loglo_hetero.yml --seed 42

# HPC (larger batch, data loaded to memory)
python training/train.py --config configs/fno_m8x32x16_h64_homo.yml --hpc true --seed 42
```

### Inference (single trajectory or full test set)
```bash
python inference/infer_hetero.py --config configs/loglo_hetero.yml --checkpoint weights/LOGLO_FNO/42/hetero.pth --use_ema --traj_idx 390

python inference/infer_hetero.py --config configs/loglo_hetero.yml --checkpoint weights/LOGLO_FNO/42/hetero.pth --use_ema --eval_all
```

### HPC (SLURM)
```bash
# Interactive launcher (select models, variants, seeds)
python slurm/launch.py

# CLI mode (for scripting / reproducibility)
python slurm/launch.py --models fno_m8x32x16_h64,unet_d3 --variants homo --seeds '{"fno_m8x32x16_h64":"42,123","unet_d3":"42"}'

# Direct single experiment (3 chained jobs)
sbatch --export=ALL,CONFIG=configs/fno_m8x32x16_h64_homo.yml,SEED=42 slurm/train.sh

# All baselines with seed 42
bash slurm/submit_all.sh
```

### Install
```bash
pip install -r requirements.txt   # torch, neuralop, wandb, matplotlib, pyyaml
pip install -e .                  # register local packages so imports resolve
```

## Architecture

### Model adapter pattern
`training/model_adapters.py` abstracts how inputs are assembled per model type:
- **SingleTensorAdapter** (FNO, UNet3D): concatenates state + static + rate + mask into one tensor.
- **DualTensorAdapter** (LOGLO_FNO): separates state tensor from action tensor — LOGLO uses AdaLN-Zero conditioning on actions, not concatenation.

`create_adapter(model_type, heterogeneous)` is the factory. All training/inference code calls `adapter.build_model_input()` and `adapter.forward()` instead of touching models directly.

### Models
- **FNOWrapper** (`models/fno_wrapper.py`): thin wrapper around `neuralop.models.fno.FNO`.
- **UNet3D** (`models/unet3d.py`): 3D encoder-decoder with skip connections. Zero-init final layer for residual learning (output = model(x) + x[:, :out_channels]).
- **LOGLO_FNO** (`models/loglo_fno.py`): Local-Global FNO with three parallel branches per block (global spectral, local patch-based spectral, high-frequency MLP). Uses AdaLN-Zero action conditioning. Zero-init projection for identity-at-init. Patch size (8,8,8).
- **AuxHead** (`models/aux_head.py`): extracts state columns at well coordinates from y_t and y_tp1, processes through per-well MLP → linear to predict BHP + energy rate.

### Training loop (`training/loop.py`)
- **Pushforward training**: unrolls k autoregressive steps (k increases with global_step via `k_step_interval`) for temporal stability.
- **Multi-loss**: weighted MSE + H1 Sobolev + spectral (radial-binned FFT bands) + mass balance equation (MBE) + mean-field pressure + auxiliary head MSE.
- **Adaptive noise** injected to input state 80% of the time.
- **EMA** tracked for both backbone and aux head.
- **LR schedule**: linear warmup → cosine decay to `min_lr`.
- **Checkpoints**: weights-only resume checkpoint saved every `save_every` epochs (default `log_every * 5 = 100`). Optimizer state saved as separate companion file only at segment boundaries (end of each `epochs_per_run`). On resume between jobs, optimizer loads if epoch matches weights; otherwise restarts fresh.

### Data (`training/dataset.py`)
- `ARDataset.__getitem__` samples random timestep t per trajectory, returns (history, y_t, y_tp1, action, aux).
- History buffer of length `k_max` for pushforward; `valid_k` mask handles boundary.
- All fields min-max normalized; energy rate uses log1p normalization.
- Datasets expected as `.npy` files in `data.path` (see config). 400 trajectories: train [0,300), test [350,400).

### Physics losses (`training/physics.py`)
- `compute_mbe_loss`: mass balance equation residual using EOS density (thermal expansion + compressibility).
- `radial_binned_spectral_loss`: 3D FFT error binned into low/mid/high frequency bands.
- `mean_field_pressure_loss`: MSE on spatially-averaged pressure (channels 2,3).

### HPC / SLURM infrastructure (`slurm/`)
- **`train.sh`**: Parameterized SLURM template. Receives `CONFIG` and `SEED` via `--export=ALL,CONFIG=...,SEED=...`. All resource directives (1x H100-40GB, 96G RAM, ~12h wall time) live here.
- **`launch.py`**: Interactive launcher + CLI mode. Prompts for models/variants/seeds, submits 3 chained jobs per experiment. To add a new model: add entry to `MODELS` dict in `launch.py` + create its config YAML.
- **`submit_all.sh`** / **`submit_fno.sh`**: Convenience wrappers that submit all baselines with seed 42.

## Config structure

YAML configs in `configs/` follow naming: `{model}_{hyperparams}_{variant}.yml`. Hyperparams encode key architecture knobs visible at a glance:
- FNO: `fno_m{modes}x_h{hidden}_{variant}.yml` — e.g. `fno_m8x32x16_h64_homo.yml`, `fno_m4x16x8_h128_hetero.yml`
- UNet: `unet_d{depth}_{variant}.yml` — e.g. `unet_d3_homo.yml`, `unet_d4_hetero.yml`
- LOGLO: `loglo_{variant}.yml` (unchanged)

Each config has sections: `data`, `model`, `training` (with `local`/`hpc` sub-configs), `loss`, `logging`, `checkpoints`. The `model.type` field (`fno`, `loglo`, `unet3d`) selects both the model class and adapter.

### Current experiment matrix

| Model key | Modes | Layers | Hidden | Depth |
|-----------|-------|--------|--------|-------|
| `fno_m8x32x16_h64` | [8,32,16] | 4 | 64 | — |
| `fno_m4x16x8_h64` | [4,16,8] | 5 | 64 | — |
| `fno_m4x16x8_h128` | [4,16,8] | 5 | 128 | — |
| `unet_d3` | — | — | 64 | 3 |
| `unet_d4` | — | — | 64 | 4 |
| `loglo` | — | 5 blocks | 64 | — |

Each model key has both `_homo.yml` and `_hetero.yml` configs (12 total).

## Key conventions

- Heterogeneous mode adds 4 static channels; homogeneous has `in_channels=6` (4 state + rate + mask), heterogeneous has `in_channels=10` (4 state + 4 static + rate + mask). LOGLO counts differently: `in_dim=4`/`8` for state, `action_channels=2`/`6`.
- Well coordinates at layers 0-1 in action are zeroed out before model input (non-perforated layers).
- Checkpoint dict keys: `model`, `ema_model`, `aux_head`, `ema_aux`. Optimizer state in separate `_optim.pt` companion file (saved at segment boundaries only).
- Checkpoint paths auto-include seed subdirectory: `checkpoints/running/seed42/`, `checkpoints/seed42/`.
- W&B projects: `LOGLOFNO_HOMO_exp` (homogeneous) / `LOGLOFNO_HETERO_exp` (heterogeneous).
- W&B local storage is kept off the `/project` quota: `slurm/train.sh` points `WANDB_DIR`/`WANDB_CACHE_DIR`/`WANDB_DATA_DIR` at `$SLURM_TMPDIR` (node-local, auto-purged at job end, synced live in online mode). Only the **final** model is uploaded as an artifact. Per-step scalar logging is throttled by `logging.log_scalar_every` (default 10); gradient/parameter histograms (`wandb.watch`) are off unless `logging.watch_model: true`. Local checkpoints save every `save_every` epochs (default `log_every * 5`).
- Stale checkpoint cleanup: `python scripts/cleanup_checkpoints.py` (dry-run) or `--delete` to remove resume/optim files for completed experiments.