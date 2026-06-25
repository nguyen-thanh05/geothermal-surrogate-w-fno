#!/usr/bin/env python3
"""Submit the LOGLO-FNO v2 heterogeneous loss-ablation study."""

import argparse
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from slurm.launch import submit_chain


SEED = 42
JOBS_PER_CHAIN = 4
CONFIGS = (
    "configs/loss_ablation/loglo_v2_hetero_no_h1.yml",
    "configs/loss_ablation/loglo_v2_hetero_no_mbe.yml",
    "configs/loss_ablation/loglo_v2_hetero_no_spectral.yml",
    "configs/loss_ablation/loglo_v2_hetero_no_meanfield.yml",
    "configs/loss_ablation/loglo_v2_hetero_no_pushforward.yml",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print sbatch commands without submitting jobs",
    )
    args = parser.parse_args()

    os.chdir(REPO_ROOT)

    missing = [config for config in CONFIGS if not Path(config).is_file()]
    if missing:
        for config in missing:
            print(f"ERROR: missing config: {config}")
        sys.exit(1)

    total_jobs = len(CONFIGS) * JOBS_PER_CHAIN
    mode = "Dry run" if args.dry_run else "Submitting"
    print(
        f"{mode}: {len(CONFIGS)} loss ablations x "
        f"{JOBS_PER_CHAIN} segments = {total_jobs} jobs (seed {SEED})\n"
    )

    all_job_ids = []
    for config in CONFIGS:
        experiment = Path(config).stem
        job_ids = submit_chain(
            config,
            SEED,
            JOBS_PER_CHAIN,
            dry_run=args.dry_run,
        )
        all_job_ids.extend(job_ids)

        print(f"{experiment} / seed{SEED}:")
        for index, job_id in enumerate(job_ids):
            dependency = f" (depends on {job_ids[index - 1]})" if index else ""
            print(f"  Segment {index + 1}: {job_id}{dependency}")
        print()

    if not args.dry_run:
        print("Monitor: squeue -u $USER")
        print(f'Cancel all: scancel {" ".join(all_job_ids)}')


if __name__ == "__main__":
    main()
