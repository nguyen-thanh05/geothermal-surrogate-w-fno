# LOGLO-FNO v2 heterogeneous loss ablation

This study uses the completed `configs/loglo_v2_hetero.yml` experiment as the
full-loss control. The control was trained with seeds 5, 42, and 2026; the
ablation runs use seed 42 for paired comparison and do not retrain the control.

Each ablation preserves the control configuration and disables exactly one
training component:

| Config | Disabled component |
| --- | --- |
| `loglo_v2_hetero_no_h1.yml` | H1 loss |
| `loglo_v2_hetero_no_mbe.yml` | Mass-balance loss |
| `loglo_v2_hetero_no_spectral.yml` | Radial spectral loss |
| `loglo_v2_hetero_no_meanfield.yml` | Mean-field pressure loss |
| `loglo_v2_hetero_no_pushforward.yml` | Pushforward training |

MSE and the auxiliary BHP/energy loss remain enabled in every run. Disabled
loss weights are retained at their control values so the boolean toggle is the
only functional loss change.

All five experiments log to the W&B project
`LOGLOFNO_HETERO_loss_ablation` and use unique checkpoint filenames.

Preview the 20 submitted jobs (five experiments, four chained segments each):

```bash
python slurm/launch_loss_ablation.py --dry-run
```

Submit the study:

```bash
python slurm/launch_loss_ablation.py
```
