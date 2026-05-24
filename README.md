# Geothermal Reservoir Surrogate Modeling

Autoregressive neural operator surrogates for 3D dual-porosity geothermal reservoir simulation. Predicts next-timestep reservoir state (temperature + pressure, matrix + fracture) given current state and well control actions.

**Models**: FNO, LOGLO_FNO, UNet3D  
**Data variants**: homogeneous (fixed rock properties) / heterogeneous (per-realization static fields)

## Setup

```bash
pip install -r requirements.txt
```

## Training

### Local

```bash
python training/train.py --config configs/fno_homo.yml --seed 42
```

### HPC (SLURM)

**Interactive launcher** (recommended):

```bash
python slurm/launch.py
```

Prompts for model(s), variant(s), and seed(s) per model, then submits 3 chained SLURM jobs per experiment.

**CLI mode** (for scripting):

```bash
python slurm/launch.py \
  --models fno,unet3d \
  --variants homo,hetero \
  --seeds '{"fno":"42,123","unet3d":"42"}'
```

**Dry run** (preview without submitting):

```bash
python slurm/launch.py --dry-run
```

**Single experiment** (manual):

```bash
sbatch --export=ALL,CONFIG=configs/fno_homo.yml,SEED=42 slurm/train.sh
```

**All baselines** (seed 42):

```bash
bash slurm/submit_all.sh
```

## Inference

```bash
# Single trajectory
python inference/infer_hetero.py \
  --config configs/loglo_hetero.yml \
  --checkpoint checkpoints/seed42/loglo_hetero_final.pth \
  --use_ema --traj_idx 390

# Full test set
python inference/infer_hetero.py \
  --config configs/loglo_hetero.yml \
  --checkpoint checkpoints/seed42/loglo_hetero_final.pth \
  --use_ema --eval_all
```

## Project Structure

```
configs/             YAML configs: {model}_{variant}.yml
models/              Model definitions (FNO, LOGLO_FNO, UNet3D, AuxHead)
training/
  train.py           Entry point (--config, --seed, --hpc)
  loop.py            Training loop (pushforward, multi-loss, EMA)
  dataset.py         AR dataset with min-max normalization
  model_adapters.py  Adapter pattern for model-specific input assembly
  physics.py         Physics-informed losses (MBE, spectral, mean-field)
inference/           Inference scripts
slurm/
  train.sh           Parameterized SLURM template
  launch.py          Interactive/CLI job launcher
  submit_all.sh      Submit all baselines (seed 42)
  submit_fno.sh      Submit FNO baselines (seed 42)
checkpoints/         Organized by seed: checkpoints/seed{N}/
```

## Adding a New Model

1. Implement model class in `models/`
2. Add adapter in `training/model_adapters.py`
3. Create config YAML in `configs/`
4. Add entry to `MODELS` dict in `slurm/launch.py`
