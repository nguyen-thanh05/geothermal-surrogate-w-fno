#!/usr/bin/env python3
"""Evaluate LOGLO-FNO v2 heterogeneous loss ablations vs the full-loss control.

Runs the same autoregressive rollout metrics as ``evaluation/run_eval.py`` for
each ablation checkpoint and the ``loglo_v2_hetero`` control, then writes
comparison tables and plots under the output directory.

Ablation runs are trained with seed 42 (see ``configs/loss_ablation/README.md``).
The full-loss control is evaluated at the same seed by default for paired
comparison; pass ``--seeds`` to override.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.run_eval import (  # noqa: E402
    BOOTSTRAP_RESAMPLES,
    METRIC_SPECS,
    REPORT_STEPS,
    STATE_CHANNELS,
    Job,
    MetricRecord,
    aggregate_outputs,
    bootstrap_curve,
    clean_output_dir,
    complete_models,
    final_checkpoint_name,
    format_number,
    load_metric_records,
    load_yaml,
    matrix_for_spec,
    parse_csv_filter,
    resolve_repo_path,
    run_evaluations,
    seed_spread_rows,
    stable_seed,
    table_rows_for_scope,
    write_csv_table,
    write_latex_table,
    write_markdown_table,
    write_notes,
    write_table_bundle,
)

DEFAULT_SEED = 42
VARIANT = "hetero"
ABLATION_CONFIG_DIR = REPO_ROOT / "configs" / "loss_ablation"
CONTROL_CONFIG = REPO_ROOT / "configs" / "loglo_v2_hetero.yml"

FINAL_STEP = REPORT_STEPS[-1]


@dataclass(frozen=True)
class AblationSpec:
    key: str
    display: str
    config_path: Path
    is_control: bool = False


ABLATION_SPECS: tuple[AblationSpec, ...] = (
    AblationSpec(
        key="loglo_v2_full",
        display="LOGLO-FNO v2 (full loss)",
        config_path=CONTROL_CONFIG,
        is_control=True,
    ),
    AblationSpec(
        key="loglo_v2_no_h1",
        display="w/o H1",
        config_path=ABLATION_CONFIG_DIR / "loglo_v2_hetero_no_h1.yml",
    ),
    AblationSpec(
        key="loglo_v2_no_mbe",
        display="w/o MBE",
        config_path=ABLATION_CONFIG_DIR / "loglo_v2_hetero_no_mbe.yml",
    ),
    AblationSpec(
        key="loglo_v2_no_spectral",
        display="w/o spectral",
        config_path=ABLATION_CONFIG_DIR / "loglo_v2_hetero_no_spectral.yml",
    ),
    AblationSpec(
        key="loglo_v2_no_meanfield",
        display="w/o mean-field",
        config_path=ABLATION_CONFIG_DIR / "loglo_v2_hetero_no_meanfield.yml",
    ),
    AblationSpec(
        key="loglo_v2_no_pushforward",
        display="w/o pushforward",
        config_path=ABLATION_CONFIG_DIR / "loglo_v2_hetero_no_pushforward.yml",
    ),
)
SPEC_BY_KEY = {spec.key: spec for spec in ABLATION_SPECS}


def discover_ablation_jobs(args: argparse.Namespace) -> tuple[list[Job], list[str]]:
    notes: list[str] = []
    selected_keys = parse_csv_filter(
        args.ablations,
        [spec.key for spec in ABLATION_SPECS],
    )
    selected_seeds = [
        int(x) for x in parse_csv_filter(args.seeds, [str(DEFAULT_SEED)])
    ]

    ckpt_root = resolve_repo_path(args.ckpt_root)
    out_dir = resolve_repo_path(args.out_dir)
    jobs: list[Job] = []

    for spec in ABLATION_SPECS:
        if spec.key not in selected_keys:
            continue
        if not spec.config_path.is_file():
            notes.append(f"Missing config for {spec.key}: {spec.config_path}")
            continue

        cfg = load_yaml(spec.config_path)
        data_path = resolve_repo_path(args.hetero_data or cfg["data"]["path"])
        ckpt_name = final_checkpoint_name(cfg, spec.key, VARIANT)

        for seed in selected_seeds:
            metrics_path = (
                out_dir / "metrics" / VARIANT / f"seed{seed}" / f"{spec.key}.npz"
            )
            jobs.append(
                Job(
                    model=spec.key,
                    display=spec.display,
                    variant=VARIANT,
                    seed=seed,
                    config_path=spec.config_path,
                    checkpoint_path=ckpt_root / f"seed{seed}" / ckpt_name,
                    data_path=data_path,
                    metrics_path=metrics_path,
                )
            )
    return jobs, notes


def display_name(model_key: str, records: dict[str, dict[str, dict[int, MetricRecord]]]) -> str:
    seed_map = records.get(VARIANT, {}).get(model_key, {})
    if seed_map:
        first = next(iter(seed_map.values()))
        return first.display
    return SPEC_BY_KEY.get(model_key, AblationSpec(model_key, model_key, CONTROL_CONFIG)).display


def plot_ablation_state_curves(
    out_dir: Path,
    records: dict[str, dict[str, dict[int, MetricRecord]]],
    models: list[str],
    seeds: list[int],
    resamples: int,
) -> None:
    spec = METRIC_SPECS[0]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    axes = axes.flatten()
    colors = plt.get_cmap("tab10").colors

    for ch_idx, entity in enumerate(spec.entities):
        ax = axes[ch_idx]
        for model_idx, model_key in enumerate(models):
            recs = [records[VARIANT][model_key][seed] for seed in seeds]
            arr = np.concatenate([r.arrays[spec.array_key] for r in recs], axis=0)
            matrix = arr[:, ch_idx, :]
            mean, low, high = bootstrap_curve(
                matrix,
                spec.stat_kind,
                resamples,
                stable_seed(f"ablation-plot-{model_key}-{ch_idx}"),
            )
            x = np.arange(1, matrix.shape[1] + 1)
            color = colors[model_idx % len(colors)]
            label = display_name(model_key, records)
            linewidth = 2.4 if SPEC_BY_KEY.get(model_key, AblationSpec("", "", CONTROL_CONFIG)).is_control else 1.6
            ax.plot(x, mean, label=label, color=color, linewidth=linewidth)
            ax.fill_between(x, low, high, color=color, alpha=0.12, linewidth=0)
        ax.set_title(entity)
        ax.set_ylabel(spec.ylabel)
        ax.grid(True, alpha=0.25)

    for ax in axes[2:]:
        ax.set_xlabel("Autoregressive rollout step")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    fig.suptitle("LOGLO-FNO v2 loss ablation: state normalized relative L2", fontsize=14)
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    path = out_dir / "plots" / "ablation_state_l2_norm.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_ablation_final_step_bars(
    out_dir: Path,
    records: dict[str, dict[str, dict[int, MetricRecord]]],
    models: list[str],
    seed: int,
) -> None:
    spec = METRIC_SPECS[0]
    control_key = next(
        (key for key in models if SPEC_BY_KEY.get(key) and SPEC_BY_KEY[key].is_control),
        models[0],
    )
    control_rec = records[VARIANT][control_key][seed]
    control_arr = control_rec.arrays[spec.array_key][:, :, FINAL_STEP - 1]
    control_mean = control_arr.mean(axis=0)

    ablation_models = [key for key in models if key != control_key]
    n_channels = len(spec.entities)
    x = np.arange(n_channels)
    width = 0.75 / max(len(ablation_models), 1)

    fig, ax = plt.subplots(figsize=(11, 5))
    for idx, model_key in enumerate(ablation_models):
        arr = records[VARIANT][model_key][seed].arrays[spec.array_key][:, :, FINAL_STEP - 1]
        means = arr.mean(axis=0)
        offset = (idx - (len(ablation_models) - 1) / 2.0) * width
        ax.bar(
            x + offset,
            means,
            width=width,
            label=display_name(model_key, records),
            alpha=0.9,
        )

    ax.plot(
        x,
        control_mean,
        color="black",
        marker="D",
        markersize=7,
        linewidth=0,
        label=f"{display_name(control_key, records)} (control)",
        zorder=5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(spec.entities, rotation=15, ha="right")
    ax.set_ylabel(f"{spec.ylabel} at t={FINAL_STEP}")
    ax.set_title(f"Final-step state error vs full-loss control (seed {seed})")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    path = out_dir / "plots" / f"ablation_final_step_bars_seed{seed}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_ablation_comparison_table(
    out_dir: Path,
    records: dict[str, dict[str, dict[int, MetricRecord]]],
    models: list[str],
    seed: int,
    resamples: int,
) -> None:
    spec = METRIC_SPECS[0]
    control_key = next(
        (key for key in models if SPEC_BY_KEY.get(key) and SPEC_BY_KEY[key].is_control),
        models[0],
    )
    control_matrix, _ = matrix_for_spec([records[VARIANT][control_key][seed]], spec)
    control_point = control_matrix.mean(axis=0)

    header = ["Variant"] + [f"{entity} t{FINAL_STEP}" for entity in spec.entities]
    md_rows: list[list[str]] = []
    csv_rows: list[dict[str, object]] = []

    control_display = display_name(control_key, records)
    md_rows.append(
        [control_display]
        + [format_number(control_point[i], spec.fmt) for i in range(len(spec.entities))]
    )
    csv_rows.append(
        {
            "variant": control_key,
            "display": control_display,
            "seed": seed,
            "is_control": True,
            **{
                f"{safe_column(entity)}_mean": float(control_point[i])
                for i, entity in enumerate(spec.entities)
            },
        }
    )

    for model_key in models:
        if model_key == control_key:
            continue
        matrix, _ = matrix_for_spec([records[VARIANT][model_key][seed]], spec)
        point = matrix.mean(axis=0)
        rel_pct = (point / np.clip(control_point, 1e-12, None) - 1.0) * 100.0
        label = display_name(model_key, records)
        md_rows.append(
            [label]
            + [
                f"{format_number(point[i], spec.fmt)} ({rel_pct[i]:+.1f}%)"
                for i in range(len(spec.entities))
            ]
        )
        csv_rows.append(
            {
                "variant": model_key,
                "display": label,
                "seed": seed,
                "is_control": False,
                **{
                    f"{safe_column(entity)}_mean": float(point[i])
                    for i, entity in enumerate(spec.entities)
                },
                **{
                    f"{safe_column(entity)}_pct_vs_control": float(rel_pct[i])
                    for i, entity in enumerate(spec.entities)
                },
            }
        )

    title = f"Loss ablation vs full control at t={FINAL_STEP} (seed {seed})"
    stem = f"hetero_seed{seed}_ablation_vs_control"
    write_markdown_table(out_dir / "tables" / "markdown" / f"{stem}.md", title, header, md_rows)
    write_csv_table(out_dir / "tables" / "csv" / f"{stem}.csv", csv_rows)
    write_latex_table(out_dir / "tables" / "latex" / f"{stem}.tex", title, header, md_rows)


def safe_column(name: str) -> str:
    return (
        name.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
    )


def aggregate_ablation_outputs(
    out_dir: Path,
    models: list[str],
    seeds: list[int],
    resamples: int,
    notes: list[str],
) -> None:
    records = load_metric_records(out_dir)
    all_notes = list(notes)

    complete, missing_notes = complete_models(records, VARIANT, seeds, models)
    all_notes.extend(missing_notes)
    if not complete:
        all_notes.append(f"{VARIANT}: no complete ablation metric sets found")
        write_notes(out_dir, all_notes)
        return

    for spec in METRIC_SPECS:
        for seed in seeds:
            header, rows, csv_rows = table_rows_for_scope(
                records, VARIANT, complete, seeds, f"seed{seed}", spec, resamples
            )
            for i, model_key in enumerate(complete):
                rows[i][0] = display_name(model_key, records)
            write_table_bundle(out_dir, VARIANT, f"seed{seed}", spec, header, rows, csv_rows)

        header, rows, csv_rows = table_rows_for_scope(
            records, VARIANT, complete, seeds, "aggregate", spec, resamples
        )
        for i, model_key in enumerate(complete):
            rows[i][0] = display_name(model_key, records)
        write_table_bundle(out_dir, VARIANT, "aggregate", spec, header, rows, csv_rows)

        if len(seeds) > 1:
            header, rows, csv_rows = seed_spread_rows(
                records, VARIANT, complete, seeds, spec
            )
            for i, model_key in enumerate(complete):
                rows[i][0] = display_name(model_key, records)
            write_table_bundle(out_dir, VARIANT, "seed_spread", spec, header, rows, csv_rows)

    plot_ablation_state_curves(out_dir, records, complete, seeds, resamples)
    for seed in seeds:
        if all(seed in records[VARIANT].get(model_key, {}) for model_key in complete):
            plot_ablation_final_step_bars(out_dir, records, complete, seed)
            write_ablation_comparison_table(out_dir, records, complete, seed, resamples)

    write_notes(out_dir, all_notes)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-root", default="checkpoints")
    parser.add_argument(
        "--out-dir",
        default="evaluation_results/loss_ablation",
        help="Directory for metrics, tables, and plots",
    )
    parser.add_argument("--hetero-data", default=None, help="Override heterogeneous data path")
    parser.add_argument(
        "--ablations",
        default=None,
        help="Comma-separated ablation keys (default: all variants incl. full control)",
    )
    parser.add_argument(
        "--seeds",
        default=str(DEFAULT_SEED),
        help=f"Comma-separated seeds (default: {DEFAULT_SEED})",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip existing metric caches")
    parser.add_argument("--clean", action="store_true", help="Replace generated out-dir artifacts")
    parser.add_argument("--no-ema", action="store_true", help="Use raw weights instead of EMA")
    parser.add_argument(
        "--also-run-standard-aggregate",
        action="store_true",
        help="Also emit the generic run_eval aggregate tables/plots",
    )
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
    jobs, discovery_notes = discover_ablation_jobs(args)
    data_paths = sorted({job.data_path for job in jobs})
    selected_keys = parse_csv_filter(
        args.ablations,
        [spec.key for spec in ABLATION_SPECS],
    )
    selected_seeds = [
        int(x) for x in parse_csv_filter(args.seeds, [str(DEFAULT_SEED)])
    ]

    if args.clean and not args.skip_eval:
        clean_output_dir(out_dir, ckpt_root, data_paths)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    eval_notes = [] if args.skip_eval else run_evaluations(args, jobs)

    aggregate_ablation_outputs(
        out_dir=out_dir,
        models=selected_keys,
        seeds=selected_seeds,
        resamples=args.bootstrap_resamples,
        notes=discovery_notes + eval_notes,
    )

    if args.also_run_standard_aggregate:
        aggregate_outputs(
            out_dir=out_dir,
            seeds=selected_seeds,
            model_filter=selected_keys,
            variant_filter=[VARIANT],
            resamples=args.bootstrap_resamples,
            notes=[],
        )

    print(f"Loss ablation evaluation artifacts written under {out_dir}")


if __name__ == "__main__":
    main()
