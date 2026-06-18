#!/usr/bin/env python3
"""Unified autoregressive evaluation for geothermal surrogate models."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import shutil
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.aux_head import AuxHead
from training.model_adapters import create_adapter


SEEDS = [5, 42, 2026]
VARIANTS = ["homo", "hetero"]
EVAL_INDICES = list(range(300, 400))
N_STEPS = 156
REPORT_STEPS = [10, 50, 100, 156]
BOOTSTRAP_RESAMPLES = 10_000
ENERGY_MAX = 2.9e12
ENERGY_LOG_DENOM = float(np.log1p(ENERGY_MAX))

WELL_COORDS = [
    [31, 15], [45, 4], [56, 15], [45, 27], [18, 27],
    [4, 15], [18, 4], [18, 15], [45, 15],
]

STATE_CHANNELS = [
    ("temp_form", "Temp Form", "deg C", 165.0),
    ("temp_frac", "Temp Frac", "deg C", 165.0),
    ("pres_form", "Pres Form", "kPa", None),
    ("pres_frac", "Pres Frac", "kPa", None),
]
BHP_WELLS = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "Inj1", "Inj2"]
ENERGY_WELLS = ["P1", "P2", "P3", "P4", "P5", "P6", "P7"]


@dataclass(frozen=True)
class NormConstants:
    temp_min: float = 20.0
    temp_max: float = 185.0
    pres_min: float = 1900.0
    pres_max: float = 68000.0
    action_min: float = 0.0
    action_max: float = 5000.0
    por_min_matrix: float = 0.0
    por_max_matrix: float = 1.0
    por_min_frac: float = 0.0
    por_max_frac: float = 1.0
    perm_min_matrix: float = 0.0
    perm_max_matrix: float = 1.0
    perm_min_frac: float = 0.0
    perm_max_frac: float = 1.0

    @property
    def temp_range(self) -> float:
        return self.temp_max - self.temp_min

    @property
    def pres_range(self) -> float:
        return self.pres_max - self.pres_min

    @property
    def action_range(self) -> float:
        return self.action_max - self.action_min

    @property
    def state_scales(self) -> np.ndarray:
        return np.array(
            [self.temp_range, self.temp_range, self.pres_range, self.pres_range],
            dtype=np.float32,
        )


HOMO_CONSTANTS = NormConstants()
HETERO_CONSTANTS = NormConstants(
    pres_min=1300.0,
    pres_max=70000.0,
    por_min_matrix=0.03,
    por_max_matrix=0.07,
    por_min_frac=0.002,
    por_max_frac=0.008,
    perm_min_matrix=0.05,
    perm_max_matrix=0.12,
    perm_min_frac=3.0,
    perm_max_frac=190.0,
)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display: str
    config_prefix: str


MODEL_SPECS = [
    ModelSpec("fno_m8x32x16_h64", "FNO m8x32x16 h64", "fno_m8x32x16_h64"),
    ModelSpec("fno_m4x16x8_h64", "FNO m4x16x8 h64", "fno_m4x16x8_h64"),
    ModelSpec("fno_m4x16x8_h128", "FNO m4x16x8 h128", "fno_m4x16x8_h128"),
    ModelSpec("unet_d3", "UNet3D d3", "unet_d3"),
    ModelSpec("unet_d4", "UNet3D d4", "unet_d4"),
    ModelSpec("loglo", "LOGLO-FNO", "loglo"),
    ModelSpec("loglo_new", "LOGLO-FNO new", "loglo_new"),
    ModelSpec("loglo_v2", "LOGLO-FNO v2", "loglo_v2"),
    ModelSpec("transolver", "Transolver", "transolver"),
    ModelSpec("transolver_h128_s64", "Transolver h128 s64", "transolver_h128_s64"),
    ModelSpec("vanilla_loglo", "Vanilla LOGLO-FNO", "vanilla_loglo"),
    ModelSpec("vanilla_loglo_v2", "Vanilla LOGLO-FNO v2", "vanilla_loglo_v2"),
]
MODEL_BY_KEY = {m.key: m for m in MODEL_SPECS}


@dataclass
class Job:
    model: str
    display: str
    variant: str
    seed: int
    config_path: Path
    checkpoint_path: Path
    data_path: Path
    metrics_path: Path


@dataclass
class MetricRecord:
    model: str
    display: str
    variant: str
    seed: int
    metadata: dict
    arrays: dict[str, np.ndarray]


@dataclass(frozen=True)
class MetricSpec:
    key: str
    title: str
    array_key: str
    stat_kind: str
    entities: tuple[str, ...]
    ylabel: str
    fmt: str


METRIC_SPECS = [
    MetricSpec(
        key="state_l2_norm",
        title="State normalized relative L2",
        array_key="state_l2_norm",
        stat_kind="mean",
        entities=tuple(ch[1] for ch in STATE_CHANNELS),
        ylabel="Normalized relative L2",
        fmt="{:.4g}",
    ),
    MetricSpec(
        key="state_rmse_phys",
        title="State native RMSE",
        array_key="state_mse_phys",
        stat_kind="rmse",
        entities=tuple(f"{ch[1]} ({ch[2]})" for ch in STATE_CHANNELS),
        ylabel="RMSE in native units",
        fmt="{:.4g}",
    ),
    MetricSpec(
        key="aux_bhp_rmse",
        title="BHP per-well RMSE (kPa)",
        array_key="aux_bhp_sqerr",
        stat_kind="rmse",
        entities=tuple(BHP_WELLS),
        ylabel="BHP RMSE (kPa)",
        fmt="{:.4g}",
    ),
    MetricSpec(
        key="aux_energy_rmse",
        title="Energy-rate per-well RMSE",
        array_key="aux_energy_sqerr",
        stat_kind="rmse",
        entities=tuple(ENERGY_WELLS),
        ylabel="Energy-rate RMSE",
        fmt="{:.4g}",
    ),
]


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "mps") and torch.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_constants(heterogeneous: bool) -> NormConstants:
    return HETERO_CONSTANTS if heterogeneous else HOMO_CONSTANTS


def resolve_repo_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    return p


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def final_checkpoint_name(cfg: dict, model_key: str, variant: str) -> str:
    final_path = cfg.get("checkpoints", {}).get("final_path")
    if final_path:
        return Path(final_path).name
    return f"{model_key}_{variant}_final.pth"


def discover_jobs(args: argparse.Namespace) -> tuple[list[Job], list[str]]:
    notes: list[str] = []
    selected_models = parse_csv_filter(args.models, [m.key for m in MODEL_SPECS])
    selected_variants = parse_csv_filter(args.variants, VARIANTS)
    selected_seeds = [int(x) for x in parse_csv_filter(args.seeds, [str(s) for s in SEEDS])]

    ckpt_root = resolve_repo_path(args.ckpt_root)
    out_dir = resolve_repo_path(args.out_dir)
    jobs: list[Job] = []

    for model_key in selected_models:
        spec = MODEL_BY_KEY[model_key]
        for variant in selected_variants:
            config_path = REPO_ROOT / "configs" / f"{spec.config_prefix}_{variant}.yml"
            if not config_path.is_file():
                notes.append(f"Missing config for {model_key}/{variant}: {config_path}")
                continue

            cfg = load_yaml(config_path)
            data_override = args.hetero_data if variant == "hetero" else args.homo_data
            data_path = resolve_repo_path(data_override or cfg["data"]["path"])
            ckpt_name = final_checkpoint_name(cfg, model_key, variant)

            for seed in selected_seeds:
                metrics_path = (
                    out_dir / "metrics" / variant / f"seed{seed}" / f"{model_key}.npz"
                )
                jobs.append(
                    Job(
                        model=model_key,
                        display=spec.display,
                        variant=variant,
                        seed=seed,
                        config_path=config_path,
                        checkpoint_path=ckpt_root / f"seed{seed}" / ckpt_name,
                        data_path=data_path,
                        metrics_path=metrics_path,
                    )
                )
    return jobs, notes


def parse_csv_filter(raw: str | None, default: list[str]) -> list[str]:
    if not raw:
        return list(default)
    values = [x.strip() for x in raw.split(",") if x.strip()]
    return values


def load_datasets(data_path: Path, heterogeneous: bool) -> dict[str, np.ndarray]:
    def load(name: str) -> np.ndarray:
        path = data_path / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing dataset file: {path}")
        return np.load(path, mmap_mode="r")

    ds = {
        "temp": load("all_temp_formation.npy"),
        "pres": load("all_pres_formation.npy"),
        "temp_frac": load("all_temp_frac.npy"),
        "pres_frac": load("all_pres_frac.npy"),
        "action": load("all_action.npy"),
        "aux": load("all_energyrate_bhp.npy"),
    }
    if heterogeneous:
        ds.update(
            {
                "por_matrix": load("all_por_matrix.npy"),
                "por_frac": load("all_por_frac.npy"),
                "perm_matrix": load("all_perm_matrix.npy"),
                "perm_frac": load("all_perm_frac.npy"),
            }
        )
    return ds


def normalize_state_batch(
    datasets: dict[str, np.ndarray],
    indices: list[int],
    nc: NormConstants,
) -> np.ndarray:
    tf = (np.asarray(datasets["temp"][indices], dtype=np.float32) - nc.temp_min) / nc.temp_range
    tfr = (
        np.asarray(datasets["temp_frac"][indices], dtype=np.float32) - nc.temp_min
    ) / nc.temp_range
    pf = (np.asarray(datasets["pres"][indices], dtype=np.float32) - nc.pres_min) / nc.pres_range
    pfr = (
        np.asarray(datasets["pres_frac"][indices], dtype=np.float32) - nc.pres_min
    ) / nc.pres_range
    return np.stack([tf, tfr, pf, pfr], axis=2)


def static_batch(
    datasets: dict[str, np.ndarray],
    indices: list[int],
    nc: NormConstants,
    device: torch.device,
) -> torch.Tensor:
    pm = (
        np.asarray(datasets["por_matrix"][indices], dtype=np.float32) - nc.por_min_matrix
    ) / (nc.por_max_matrix - nc.por_min_matrix)
    pf = (
        np.asarray(datasets["por_frac"][indices], dtype=np.float32) - nc.por_min_frac
    ) / (nc.por_max_frac - nc.por_min_frac)
    km = (
        np.asarray(datasets["perm_matrix"][indices], dtype=np.float32) - nc.perm_min_matrix
    ) / (nc.perm_max_matrix - nc.perm_min_matrix)
    kf = (
        np.asarray(datasets["perm_frac"][indices], dtype=np.float32) - nc.perm_min_frac
    ) / (nc.perm_max_frac - nc.perm_min_frac)
    static = np.stack([pm, pf, km, kf], axis=1)
    return torch.from_numpy(static).float().to(device)


def load_model_bundle(
    cfg: dict,
    checkpoint_path: Path,
    device: torch.device,
    use_ema: bool,
) -> tuple[torch.nn.Module, torch.nn.Module, object, dict]:
    model_cfg = cfg["model"]
    model_type = model_cfg["type"]
    heterogeneous = bool(cfg["data"].get("heterogeneous", False))

    model = create_eval_model(model_cfg, model_type)
    aux_head = AuxHead(
        state_channels=get_out_channels(model_cfg),
        depth=16,
        aux_channels=model_cfg.get("aux_channels", 16),
        hidden_dim=get_hidden_dim(model_cfg),
    )
    adapter = create_adapter(model_type, heterogeneous)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_key = "ema_model" if use_ema else "model"
    aux_key = "ema_aux" if use_ema else "aux_head"
    if model_key not in ckpt or aux_key not in ckpt:
        raise KeyError(f"Checkpoint lacks {model_key!r} or {aux_key!r}: {checkpoint_path}")

    model_state = ckpt[model_key]
    if isinstance(model_state, dict):
        model_state.pop("_metadata", None)
    model.load_state_dict(model_state)
    aux_head.load_state_dict(ckpt[aux_key])
    metadata = {
        "training_epoch": int(ckpt.get("epoch", -1)),
        "training_global_step": int(ckpt.get("global_step", -1)),
    }
    del ckpt

    model.to(device).eval()
    aux_head.to(device).eval()
    return model, aux_head, adapter, metadata


def create_eval_model(model_cfg: dict, model_type: str) -> torch.nn.Module:
    if model_type == "unet3d":
        from models.unet3d import UNet3D

        return UNet3D(
            in_channels=model_cfg["in_channels"],
            out_channels=model_cfg["out_channels"],
            hidden_channels=model_cfg["hidden_channels"],
            depth=model_cfg.get("depth", 3),
            channel_multipliers=model_cfg.get("channel_multipliers", None),
        )
    if model_type == "fno":
        from models.fno_wrapper import FNOWrapper

        return FNOWrapper(
            n_modes=model_cfg["n_modes"],
            in_channels=model_cfg["in_channels"],
            out_channels=model_cfg["out_channels"],
            n_layers=model_cfg["n_layers"],
            hidden_channels=model_cfg["hidden_channels"],
        )
    if model_type in ("loglo", "loglo_new"):
        from models.loglo_fno import LOGLO_FNO

        return LOGLO_FNO(
            in_dim=model_cfg["in_dim"],
            out_dim=model_cfg["out_dim"],
            lifting_dim=model_cfg["lifting_dim"],
            projection_dim=model_cfg["projection_dim"],
            hidden_dim=model_cfg["hidden_dim"],
            n_blocks=model_cfg["n_blocks"],
            action_channels=model_cfg["action_channels"],
        )
    if model_type == "loglo_v2":
        from models.loglo_fno_v2 import LOGLO_FNO

        return LOGLO_FNO(
            in_dim=model_cfg["in_dim"],
            out_dim=model_cfg["out_dim"],
            lifting_dim=model_cfg["lifting_dim"],
            projection_dim=model_cfg["projection_dim"],
            hidden_dim=model_cfg["hidden_dim"],
            n_blocks=model_cfg["n_blocks"],
            action_channels=model_cfg["action_channels"],
        )
    if model_type == "vanilla_loglo":
        from models.loglo_fno import VanillaLOGLO_FNO

        return VanillaLOGLO_FNO(
            in_dim=model_cfg["in_dim"],
            out_dim=model_cfg["out_dim"],
            lifting_dim=model_cfg["lifting_dim"],
            projection_dim=model_cfg["projection_dim"],
            hidden_dim=model_cfg["hidden_dim"],
            n_blocks=model_cfg["n_blocks"],
        )
    if model_type == "vanilla_loglo_v2":
        from models.loglo_fno_v2 import VanillaLOGLO_FNO

        return VanillaLOGLO_FNO(
            in_dim=model_cfg["in_dim"],
            out_dim=model_cfg["out_dim"],
            lifting_dim=model_cfg["lifting_dim"],
            projection_dim=model_cfg["projection_dim"],
            hidden_dim=model_cfg["hidden_dim"],
            n_blocks=model_cfg["n_blocks"],
        )
    if model_type == "transolver":
        from models.transolver3d import TransolverWrapper

        return TransolverWrapper(
            in_channels=model_cfg["in_channels"],
            out_channels=model_cfg["out_channels"],
            hidden_dim=model_cfg["hidden_dim"],
            n_layers=model_cfg["n_layers"],
            n_head=model_cfg["n_head"],
            slice_num=model_cfg.get("slice_num", 32),
            mlp_ratio=model_cfg.get("mlp_ratio", 2),
            H=model_cfg.get("H", 16),
            W=model_cfg.get("W", 64),
            D=model_cfg.get("D", 32),
            spatial_embed=model_cfg.get("spatial_embed", True),
            num_bands=model_cfg.get("num_bands", 32),
            max_freq=model_cfg.get("max_freq", 64.0),
        )
    raise ValueError(f"Unknown model type: {model_type}")


def get_out_channels(model_cfg: dict) -> int:
    return int(model_cfg.get("out_channels", model_cfg.get("out_dim", 4)))


def get_hidden_dim(model_cfg: dict) -> int:
    return int(model_cfg.get("hidden_channels", model_cfg.get("hidden_dim", 64)))


def zero_nonperforated_layers(action: torch.Tensor) -> torch.Tensor:
    for wx, wy in WELL_COORDS:
        action[:, 0:2, wx, wy] = 0.0
    return action


def relative_l2_by_channel(pred: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    diff = pred - target
    diff_norm = torch.linalg.vector_norm(diff.flatten(2), dim=2)
    target_norm = torch.linalg.vector_norm(target.flatten(2), dim=2)
    rel = diff_norm / torch.clamp(target_norm, min=1e-12)
    return rel.detach().cpu().numpy()


def denormalize_aux_predictions(
    pred_aux: np.ndarray,
    nc: NormConstants,
) -> tuple[np.ndarray, np.ndarray]:
    """Return predicted BHP and energy in physical units.

    The auxiliary head is trained in ARDataset order:
    [energy_log_norm_0..6, bhp_norm_0..8].
    """
    pred_energy = np.expm1(np.clip(pred_aux[:, 0:7] * ENERGY_LOG_DENOM, -50.0, 80.0))
    pred_bhp = pred_aux[:, 7:16] * nc.pres_range + nc.pres_min
    return pred_bhp, pred_energy


def split_raw_aux_targets(raw_aux: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return raw target BHP and energy in physical units.

    The dataset file stores [bhp_0..8, energy_0..6].
    """
    return raw_aux[:, 0:9], raw_aux[:, 9:16]


def evaluate_job(
    job: Job,
    use_ema: bool,
    batch_size: int,
    sample_limit: int | None,
) -> None:
    if not job.checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {job.checkpoint_path}")

    cfg = load_yaml(job.config_path)
    heterogeneous = bool(cfg["data"].get("heterogeneous", False))
    nc = get_constants(heterogeneous)
    device = get_device()
    datasets = load_datasets(job.data_path, heterogeneous)

    required_len = max(EVAL_INDICES) + 1
    n_available = len(datasets["temp"])
    if n_available < required_len:
        raise ValueError(
            f"Dataset has {n_available} trajectories; need at least {required_len}"
        )

    indices = list(EVAL_INDICES)
    if sample_limit is not None:
        indices = indices[:sample_limit]
    n_eval = len(indices)

    model, aux_head, adapter, ckpt_meta = load_model_bundle(
        cfg, job.checkpoint_path, device, use_ema
    )

    state_l2_norm = np.zeros((n_eval, 4, N_STEPS), dtype=np.float32)
    state_mse_phys = np.zeros((n_eval, 4, N_STEPS), dtype=np.float32)
    aux_bhp_sqerr = np.zeros((n_eval, 9, N_STEPS), dtype=np.float32)
    aux_energy_sqerr = np.zeros((n_eval, 7, N_STEPS), dtype=np.float32)
    state_scales = nc.state_scales.reshape(1, 4, 1, 1, 1)

    try:
        with torch.no_grad():
            for start in range(0, n_eval, batch_size):
                end = min(start + batch_size, n_eval)
                batch_indices = indices[start:end]
                gt_all = normalize_state_batch(datasets, batch_indices, nc)
                y_t = torch.from_numpy(gt_all[:, 0]).float().to(device)
                static = (
                    static_batch(datasets, batch_indices, nc, device)
                    if heterogeneous
                    else None
                )

                for t in range(N_STEPS):
                    action_np = np.asarray(
                        datasets["action"][batch_indices, t], dtype=np.float32
                    )
                    action_np = (action_np - nc.action_min) / nc.action_range
                    action_t = torch.from_numpy(action_np.copy()).float().to(device)
                    zero_nonperforated_layers(action_t)

                    model_input = adapter.build_model_input(y_t, action_t, static)
                    pred_y = adapter.forward(model, model_input)
                    gt_tp1 = torch.from_numpy(gt_all[:, t + 1]).float().to(device)
                    pred_aux = aux_head(y_t, pred_y)

                    state_l2_norm[start:end, :, t] = relative_l2_by_channel(pred_y, gt_tp1)

                    diff_np = (pred_y.detach().cpu().numpy() - gt_all[:, t + 1]) * state_scales
                    state_mse_phys[start:end, :, t] = np.mean(
                        diff_np ** 2, axis=(2, 3, 4)
                    )

                    pred_aux_np = pred_aux.detach().cpu().numpy()
                    gt_aux = np.asarray(datasets["aux"][batch_indices, t + 1], dtype=np.float32)
                    pred_bhp, pred_energy = denormalize_aux_predictions(pred_aux_np, nc)
                    gt_bhp, gt_energy = split_raw_aux_targets(gt_aux)
                    aux_bhp_sqerr[start:end, :, t] = (pred_bhp - gt_bhp) ** 2
                    aux_energy_sqerr[start:end, :, t] = (pred_energy - gt_energy) ** 2

                    y_t = pred_y

                print(
                    f"[{job.variant} seed{job.seed} {job.model}] "
                    f"{end}/{n_eval} trajectories"
                )
    finally:
        del model, aux_head, adapter
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    job.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        job.metrics_path,
        state_l2_norm=state_l2_norm,
        state_mse_phys=state_mse_phys,
        aux_bhp_sqerr=aux_bhp_sqerr,
        aux_energy_sqerr=aux_energy_sqerr,
    )
    metadata = {
        "model": job.model,
        "display": job.display,
        "variant": job.variant,
        "seed": job.seed,
        "config_path": str(job.config_path),
        "checkpoint_path": str(job.checkpoint_path),
        "data_path": str(job.data_path),
        "use_ema": use_ema,
        "eval_indices": indices,
        "num_samples": n_eval,
        **ckpt_meta,
    }
    write_json(job.metrics_path.with_suffix(".json"), metadata)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_error(job: Job, error: BaseException) -> None:
    write_json(
        job.metrics_path.with_suffix(".error.json"),
        {
            "model": job.model,
            "display": job.display,
            "variant": job.variant,
            "seed": job.seed,
            "config_path": str(job.config_path),
            "checkpoint_path": str(job.checkpoint_path),
            "data_path": str(job.data_path),
            "error": "".join(traceback.format_exception(error)),
        },
    )


def load_metric_records(out_dir: Path) -> dict[str, dict[str, dict[int, MetricRecord]]]:
    records: dict[str, dict[str, dict[int, MetricRecord]]] = {
        variant: {} for variant in VARIANTS
    }
    metrics_root = out_dir / "metrics"
    if not metrics_root.is_dir():
        return records

    for npz_path in metrics_root.glob("*/seed*/*.npz"):
        meta_path = npz_path.with_suffix(".json")
        if not meta_path.is_file():
            continue
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        arrays = {k: np.asarray(v) for k, v in np.load(npz_path).items()}
        rec = MetricRecord(
            model=meta["model"],
            display=meta.get("display", meta["model"]),
            variant=meta["variant"],
            seed=int(meta["seed"]),
            metadata=meta,
            arrays=arrays,
        )
        records.setdefault(rec.variant, {}).setdefault(rec.model, {})[rec.seed] = rec
    return records


def complete_models(
    records: dict[str, dict[str, dict[int, MetricRecord]]],
    variant: str,
    seeds: list[int],
    model_filter: list[str],
) -> tuple[list[str], list[str]]:
    notes: list[str] = []
    complete: list[str] = []
    for model_key in model_filter:
        seed_map = records.get(variant, {}).get(model_key, {})
        missing = [s for s in seeds if s not in seed_map]
        if missing:
            if seed_map:
                notes.append(f"{model_key}/{variant}: missing metric seeds {missing}; skipped")
            continue
        complete.append(model_key)
    return complete, notes


def bootstrap_weights(n: int, resamples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    probabilities = np.full(n, 1.0 / n, dtype=np.float64)
    return rng.multinomial(n, probabilities, size=resamples).astype(np.float64) / n


def summarize_matrix(
    matrix: np.ndarray,
    stat_kind: str,
    boot_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"Expected matrix (samples, columns), got {matrix.shape}")
    if matrix.shape[0] == 0:
        raise ValueError("Cannot summarize an empty sample matrix")

    if stat_kind == "mean":
        point = matrix.mean(axis=0)
    elif stat_kind == "rmse":
        point = np.sqrt(matrix.mean(axis=0))
    else:
        raise ValueError(f"Unknown stat kind: {stat_kind}")

    if matrix.shape[0] < 2 or boot_weights.shape[0] == 0:
        return point, point, point

    boots = boot_weights @ matrix
    if stat_kind == "rmse":
        boots = np.sqrt(boots)

    low = np.percentile(boots, 2.5, axis=0)
    high = np.percentile(boots, 97.5, axis=0)
    return point, low, high


def matrix_for_spec(records: list[MetricRecord], spec: MetricSpec) -> tuple[np.ndarray, list[str]]:
    arr = np.concatenate([r.arrays[spec.array_key] for r in records], axis=0)
    columns: list[np.ndarray] = []
    names: list[str] = []
    for entity_idx, entity_name in enumerate(spec.entities):
        for step in REPORT_STEPS:
            columns.append(arr[:, entity_idx, step - 1])
            names.append(f"{entity_name} t{step}")
    return np.column_stack(columns), names


def format_number(value: float, fmt: str) -> str:
    if not np.isfinite(value):
        return "nan"
    return fmt.format(float(value))


def format_ci_cell(mean: float, low: float, high: float, fmt: str) -> str:
    half = (high - low) / 2.0
    return f"{format_number(mean, fmt)} +/- {format_number(half, fmt)}"


def table_rows_for_scope(
    records: dict[str, dict[str, dict[int, MetricRecord]]],
    variant: str,
    models: list[str],
    seeds: list[int],
    scope: str,
    spec: MetricSpec,
    resamples: int,
) -> tuple[list[str], list[list[str]], list[dict[str, object]]]:
    header: list[str] | None = None
    md_rows: list[list[str]] = []
    csv_rows: list[dict[str, object]] = []

    for model_key in models:
        if scope.startswith("seed"):
            seed = int(scope.replace("seed", ""))
            recs = [records[variant][model_key][seed]]
        else:
            recs = [records[variant][model_key][seed] for seed in seeds]

        matrix, col_names = matrix_for_spec(recs, spec)
        if header is None:
            header = ["Model"] + col_names
        boot_weights = bootstrap_weights(
            matrix.shape[0],
            resamples,
            stable_seed(f"{variant}-{model_key}-{scope}-{spec.key}"),
        )
        means, lows, highs = summarize_matrix(matrix, spec.stat_kind, boot_weights)
        display = MODEL_BY_KEY.get(model_key, ModelSpec(model_key, model_key, model_key)).display
        md_rows.append(
            [display]
            + [
                format_ci_cell(means[i], lows[i], highs[i], spec.fmt)
                for i in range(len(col_names))
            ]
        )
        csv_row: dict[str, object] = {
            "variant": variant,
            "scope": scope,
            "model": model_key,
            "metric": spec.key,
            "num_samples": matrix.shape[0],
        }
        for i, col in enumerate(col_names):
            safe = safe_column(col)
            csv_row[f"{safe}_mean"] = float(means[i])
            csv_row[f"{safe}_ci_low"] = float(lows[i])
            csv_row[f"{safe}_ci_high"] = float(highs[i])
        csv_rows.append(csv_row)

    return header or ["Model"], md_rows, csv_rows


def seed_spread_rows(
    records: dict[str, dict[str, dict[int, MetricRecord]]],
    variant: str,
    models: list[str],
    seeds: list[int],
    spec: MetricSpec,
) -> tuple[list[str], list[list[str]], list[dict[str, object]]]:
    header: list[str] | None = None
    md_rows: list[list[str]] = []
    csv_rows: list[dict[str, object]] = []
    for model_key in models:
        per_seed_points = []
        col_names = None
        for seed in seeds:
            matrix, names = matrix_for_spec([records[variant][model_key][seed]], spec)
            if spec.stat_kind == "mean":
                point = matrix.mean(axis=0)
            else:
                point = np.sqrt(matrix.mean(axis=0))
            per_seed_points.append(point)
            col_names = names
        assert col_names is not None
        if header is None:
            header = ["Model"] + col_names
        stack = np.vstack(per_seed_points)
        means = stack.mean(axis=0)
        sds = stack.std(axis=0, ddof=1) if len(seeds) > 1 else np.zeros_like(means)
        display = MODEL_BY_KEY.get(model_key, ModelSpec(model_key, model_key, model_key)).display
        md_rows.append(
            [display]
            + [
                f"{format_number(means[i], spec.fmt)} +/- {format_number(sds[i], spec.fmt)}"
                for i in range(len(col_names))
            ]
        )
        csv_row: dict[str, object] = {
            "variant": variant,
            "scope": "seed_spread",
            "model": model_key,
            "metric": spec.key,
            "num_seeds": len(seeds),
        }
        for i, col in enumerate(col_names):
            safe = safe_column(col)
            csv_row[f"{safe}_mean"] = float(means[i])
            csv_row[f"{safe}_sd"] = float(sds[i])
        csv_rows.append(csv_row)
    return header or ["Model"], md_rows, csv_rows


def safe_column(name: str) -> str:
    return (
        name.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
    )


def stable_seed(text: str) -> int:
    value = 2166136261
    for ch in text:
        value ^= ord(ch)
        value = (value * 16777619) & 0xFFFFFFFF
    return value


def write_markdown_table(path: Path, title: str, header: list[str], rows: list[list[str]]) -> None:
    lines = [f"# {title}", ""]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv_table(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def write_latex_table(path: Path, title: str, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    colspec = "l" + "r" * (len(header) - 1)
    lines = [
        f"% {title}",
        f"\\begin{{tabular}}{{{colspec}}}",
        "\\toprule",
        " & ".join(latex_escape(h) for h in header) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(latex_escape(c) for c in row) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_table_bundle(
    out_dir: Path,
    variant: str,
    scope: str,
    spec: MetricSpec,
    header: list[str],
    md_rows: list[list[str]],
    csv_rows: list[dict[str, object]],
) -> None:
    stem = f"{variant}_{scope}_{spec.key}"
    title = f"{variant} {scope} {spec.title}"
    write_markdown_table(out_dir / "tables" / "markdown" / f"{stem}.md", title, header, md_rows)
    write_csv_table(out_dir / "tables" / "csv" / f"{stem}.csv", csv_rows)
    write_latex_table(out_dir / "tables" / "latex" / f"{stem}.tex", title, header, md_rows)


def bootstrap_curve(
    matrix: np.ndarray,
    stat_kind: str,
    resamples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix, dtype=np.float64)
    weights = bootstrap_weights(matrix.shape[0], resamples, seed)
    return summarize_matrix(matrix, stat_kind, weights)


def plot_state_curves(
    out_dir: Path,
    records: dict[str, dict[str, dict[int, MetricRecord]]],
    variant: str,
    models: list[str],
    seeds: list[int],
    spec: MetricSpec,
    resamples: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
    axes = axes.flatten()
    colors = plt.get_cmap("tab20").colors

    for ch_idx, entity in enumerate(spec.entities):
        ax = axes[ch_idx]
        for model_idx, model_key in enumerate(models):
            recs = [records[variant][model_key][seed] for seed in seeds]
            arr = np.concatenate([r.arrays[spec.array_key] for r in recs], axis=0)
            matrix = arr[:, ch_idx, :]
            mean, low, high = bootstrap_curve(
                matrix,
                spec.stat_kind,
                resamples,
                stable_seed(f"plot-{variant}-{model_key}-{spec.key}-{ch_idx}"),
            )
            x = np.arange(1, matrix.shape[1] + 1)
            color = colors[model_idx % len(colors)]
            ax.plot(x, mean, label=MODEL_BY_KEY[model_key].display, color=color, linewidth=1.7)
            ax.fill_between(x, low, high, color=color, alpha=0.14, linewidth=0)
        ax.set_title(entity)
        ax.set_ylabel(spec.ylabel)
        ax.grid(True, alpha=0.25)
    for ax in axes[2:]:
        ax.set_xlabel("Autoregressive rollout step")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    fig.suptitle(f"{variant} aggregate {spec.title}", fontsize=14)
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    path = out_dir / "plots" / "state" / f"{variant}_{spec.key}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_aux_dashboard(
    out_dir: Path,
    records: dict[str, dict[str, dict[int, MetricRecord]]],
    variant: str,
    model_key: str,
    seeds: list[int],
    spec: MetricSpec,
    resamples: int,
) -> None:
    n_entities = len(spec.entities)
    ncols = 3
    nrows = int(np.ceil(n_entities / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows), sharex=True)
    axes = np.atleast_1d(axes).flatten()
    recs = [records[variant][model_key][seed] for seed in seeds]
    arr = np.concatenate([r.arrays[spec.array_key] for r in recs], axis=0)
    x = np.arange(1, arr.shape[2] + 1)
    color = "#1f77b4" if "bhp" in spec.key else "#d62728"

    for entity_idx, entity in enumerate(spec.entities):
        ax = axes[entity_idx]
        matrix = arr[:, entity_idx, :]
        mean, low, high = bootstrap_curve(
            matrix,
            spec.stat_kind,
            resamples,
            stable_seed(f"auxplot-{variant}-{model_key}-{spec.key}-{entity_idx}"),
        )
        ax.plot(x, mean, color=color, linewidth=1.6)
        ax.fill_between(x, low, high, color=color, alpha=0.18, linewidth=0)
        ax.set_title(entity)
        ax.set_ylabel(spec.ylabel)
        ax.grid(True, alpha=0.25)
    for ax in axes[n_entities:]:
        ax.axis("off")
    for ax in axes[-ncols:]:
        ax.set_xlabel("Autoregressive rollout step")
    fig.suptitle(f"{variant} {MODEL_BY_KEY[model_key].display} {spec.title}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = out_dir / "plots" / "aux" / f"{variant}_{model_key}_{spec.key}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_final_l2_histograms(
    out_dir: Path,
    records: dict[str, dict[str, dict[int, MetricRecord]]],
    variant: str,
    model_key: str,
    seeds: list[int],
) -> None:
    recs = [records[variant][model_key][seed] for seed in seeds]
    arr = np.concatenate([r.arrays["state_l2_norm"] for r in recs], axis=0)
    final = arr[:, :, -1]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for ch_idx, (_, label, _, _) in enumerate(STATE_CHANNELS):
        ax = axes[ch_idx]
        ax.hist(final[:, ch_idx], bins=30, color="#4c78a8", alpha=0.85, edgecolor="white")
        ax.axvline(final[:, ch_idx].mean(), color="black", linewidth=1.5, label="Mean")
        ax.set_title(label)
        ax.set_xlabel("Final-step normalized relative L2")
        ax.set_ylabel("Count")
        ax.grid(True, axis="y", alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle(f"{variant} {MODEL_BY_KEY[model_key].display} final-step L2 distribution")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = out_dir / "plots" / "histograms" / f"{variant}_{model_key}_final_l2.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def aggregate_outputs(
    out_dir: Path,
    seeds: list[int],
    model_filter: list[str],
    variant_filter: list[str],
    resamples: int,
    notes: list[str],
) -> None:
    records = load_metric_records(out_dir)
    all_notes = list(notes)

    for variant in variant_filter:
        models, missing_notes = complete_models(records, variant, seeds, model_filter)
        all_notes.extend(missing_notes)
        if not models:
            all_notes.append(f"{variant}: no complete model/seed metric sets found")
            continue

        for spec in METRIC_SPECS:
            for seed in seeds:
                header, rows, csv_rows = table_rows_for_scope(
                    records, variant, models, seeds, f"seed{seed}", spec, resamples
                )
                write_table_bundle(out_dir, variant, f"seed{seed}", spec, header, rows, csv_rows)

            header, rows, csv_rows = table_rows_for_scope(
                records, variant, models, seeds, "aggregate", spec, resamples
            )
            write_table_bundle(out_dir, variant, "aggregate", spec, header, rows, csv_rows)

            header, rows, csv_rows = seed_spread_rows(records, variant, models, seeds, spec)
            write_table_bundle(out_dir, variant, "seed_spread", spec, header, rows, csv_rows)

        plot_state_curves(out_dir, records, variant, models, seeds, METRIC_SPECS[0], resamples)
        plot_state_curves(out_dir, records, variant, models, seeds, METRIC_SPECS[1], resamples)
        for model_key in models:
            plot_aux_dashboard(out_dir, records, variant, model_key, seeds, METRIC_SPECS[2], resamples)
            plot_aux_dashboard(out_dir, records, variant, model_key, seeds, METRIC_SPECS[3], resamples)
            plot_final_l2_histograms(out_dir, records, variant, model_key, seeds)

    write_notes(out_dir, all_notes)


def write_notes(out_dir: Path, notes: list[str]) -> None:
    if not notes:
        notes = ["No missing configs, checkpoints, metrics, or failed jobs were reported."]
    lines = ["# Evaluation Notes", ""]
    for note in notes:
        lines.append(f"- {note}")
    lines.append("")
    (out_dir / "notes.md").write_text("\n".join(lines), encoding="utf-8")


def clean_output_dir(out_dir: Path, ckpt_root: Path, data_paths: list[Path]) -> None:
    resolved = out_dir.resolve()
    forbidden = {REPO_ROOT.resolve(), ckpt_root.resolve()}
    forbidden.update(p.resolve() for p in data_paths if p)
    if resolved in forbidden:
        raise ValueError(f"Refusing to clean protected path: {resolved}")
    if resolved.anchor == str(resolved):
        raise ValueError(f"Refusing to clean filesystem root: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)


def run_evaluations(args: argparse.Namespace, jobs: list[Job]) -> list[str]:
    notes: list[str] = []
    existing = sum(1 for j in jobs if j.checkpoint_path.is_file())
    print(f"Discovered {len(jobs)} jobs; {existing} checkpoints currently exist")

    for i, job in enumerate(jobs, 1):
        label = f"{job.variant} seed{job.seed} {job.model}"
        if args.resume and job.metrics_path.is_file():
            print(f"[SKIP] {label}: metrics already exist")
            continue
        if not job.checkpoint_path.is_file():
            notes.append(f"{label}: missing checkpoint {job.checkpoint_path}")
            print(f"[SKIP] {label}: missing checkpoint")
            continue
        print(f"[{i}/{len(jobs)}] Evaluating {label}")
        try:
            evaluate_job(
                job,
                use_ema=not args.no_ema,
                batch_size=args.batch_size,
                sample_limit=args.sample_limit,
            )
        except Exception as exc:
            notes.append(f"{label}: failed evaluation; see error sidecar")
            write_error(job, exc)
            print(f"[ERROR] {label}: {exc}")
            traceback.print_exc()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return notes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-root", default="checkpoints")
    parser.add_argument("--out-dir", default="evaluation_results")
    parser.add_argument("--homo-data", default=None)
    parser.add_argument("--hetero-data", default=None)
    parser.add_argument("--models", default=None, help="Comma-separated model keys")
    parser.add_argument("--variants", default=None, help="Comma-separated variants")
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip existing metric caches")
    parser.add_argument("--clean", action="store_true", help="Replace generated out-dir artifacts")
    parser.add_argument("--no-ema", action="store_true", help="Use raw weights instead of EMA")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.sample_limit is not None and args.sample_limit < 1:
        raise ValueError("--sample-limit must be positive")

    out_dir = resolve_repo_path(args.out_dir)
    ckpt_root = resolve_repo_path(args.ckpt_root)
    jobs, discovery_notes = discover_jobs(args)
    data_paths = sorted({j.data_path for j in jobs})

    if args.clean and not args.skip_eval:
        clean_output_dir(out_dir, ckpt_root, data_paths)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    eval_notes = [] if args.skip_eval else run_evaluations(args, jobs)

    selected_models = parse_csv_filter(args.models, [m.key for m in MODEL_SPECS])
    selected_variants = parse_csv_filter(args.variants, VARIANTS)
    selected_seeds = [int(x) for x in parse_csv_filter(args.seeds, [str(s) for s in SEEDS])]
    aggregate_outputs(
        out_dir=out_dir,
        seeds=selected_seeds,
        model_filter=selected_models,
        variant_filter=selected_variants,
        resamples=args.bootstrap_resamples,
        notes=discovery_notes + eval_notes,
    )
    print(f"Evaluation artifacts written under {out_dir}")


if __name__ == "__main__":
    main()
