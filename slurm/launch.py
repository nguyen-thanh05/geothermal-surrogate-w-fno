#!/usr/bin/env python3
"""Interactive HPC job submission launcher for geothermal surrogate models."""

import argparse
import json
import os
import subprocess
import sys

MODELS = {
    'loglo':  {'display': 'LOGLO_FNO', 'config_prefix': 'loglo'},
    'fno':    {'display': 'FNO',       'config_prefix': 'fno'},
    'unet3d': {'display': 'UNet3D',    'config_prefix': 'unet'},
}

VARIANTS = ['homo', 'hetero']
JOBS_PER_CHAIN = 3
SLURM_TEMPLATE = 'slurm/train.sh'


def get_config_path(model_key, variant):
    prefix = MODELS[model_key]['config_prefix']
    return f'configs/{prefix}_{variant}.yml'


def submit_chain(config_path, seed, dry_run=False):
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    job_name = f'{config_name}_s{seed}'

    job_ids = []
    for i in range(JOBS_PER_CHAIN):
        cmd = [
            'sbatch',
            f'--job-name={job_name}',
            f'--export=ALL,CONFIG={config_path},SEED={seed}',
        ]
        if job_ids:
            cmd.append(f'--dependency=afterok:{job_ids[-1]}')
        cmd.append(SLURM_TEMPLATE)

        if dry_run:
            print(f'  [dry-run] {" ".join(cmd)}')
            job_ids.append(f'DRY{i}')
            continue

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f'ERROR: sbatch failed: {result.stderr.strip()}')
            sys.exit(1)
        job_id = result.stdout.strip().split()[-1]
        job_ids.append(job_id)

    return job_ids


def pick_numbered(prompt, options, allow_all=True):
    for i, opt in enumerate(options, 1):
        print(f'  {i}) {opt}')
    hint = "comma-separated numbers, or 'all'" if allow_all else 'comma-separated numbers'
    default = 'all' if allow_all else '1'
    raw = input(f'{prompt} [{default}]: ').strip() or default
    if raw.lower() == 'all' and allow_all:
        return list(options)
    try:
        indices = [int(x.strip()) for x in raw.split(',')]
        return [options[i - 1] for i in indices if 1 <= i <= len(options)]
    except (ValueError, IndexError):
        print('Invalid selection.')
        sys.exit(1)


def interactive_mode(dry_run=False):
    print('\n=== HPC Job Launcher ===\n')

    model_keys = list(MODELS.keys())
    model_displays = [MODELS[k]['display'] for k in model_keys]
    print('Available models:')
    selected_displays = pick_numbered('Select models', model_displays)
    selected_models = [model_keys[model_displays.index(d)] for d in selected_displays]

    print('\nAvailable variants:')
    selected_variants = pick_numbered('Select variants', VARIANTS)

    seeds_per_model = {}
    print()
    for mk in selected_models:
        display = MODELS[mk]['display']
        raw = input(f'Seeds for {display} (comma-separated) [42]: ').strip() or '42'
        try:
            seeds_per_model[mk] = [int(s.strip()) for s in raw.split(',')]
        except ValueError:
            print('Seeds must be integers.')
            sys.exit(1)

    experiments = []
    for mk in selected_models:
        for var in selected_variants:
            config = get_config_path(mk, var)
            if not os.path.isfile(config):
                print(f'WARNING: {config} not found, skipping.')
                continue
            for seed in seeds_per_model[mk]:
                experiments.append((mk, var, seed, config))

    print_summary(experiments)

    confirm = input('Submit? [y/N]: ').strip().lower()
    if confirm != 'y':
        print('Aborted.')
        sys.exit(0)

    submit_experiments(experiments, dry_run)


def cli_mode(args, dry_run=False):
    selected_models = [m.strip() for m in args.models.split(',')]
    for m in selected_models:
        if m not in MODELS:
            print(f"Unknown model: '{m}'. Available: {', '.join(MODELS.keys())}")
            sys.exit(1)

    selected_variants = [v.strip() for v in args.variants.split(',')]
    for v in selected_variants:
        if v not in VARIANTS:
            print(f"Unknown variant: '{v}'. Available: {', '.join(VARIANTS)}")
            sys.exit(1)

    seeds_map = json.loads(args.seeds)
    seeds_per_model = {}
    for mk in selected_models:
        raw = seeds_map.get(mk, '42')
        seeds_per_model[mk] = [int(s.strip()) for s in str(raw).split(',')]

    experiments = []
    for mk in selected_models:
        for var in selected_variants:
            config = get_config_path(mk, var)
            if not os.path.isfile(config):
                print(f'WARNING: {config} not found, skipping.')
                continue
            for seed in seeds_per_model[mk]:
                experiments.append((mk, var, seed, config))

    print_summary(experiments)
    submit_experiments(experiments, dry_run)


def print_summary(experiments):
    total_jobs = len(experiments) * JOBS_PER_CHAIN
    print(f'\n{"Model":<12} {"Variant":<10} {"Seed":<12} {"Config"}')
    print('-' * 60)
    for mk, var, seed, config in experiments:
        print(f'{MODELS[mk]["display"]:<12} {var:<10} {seed:<12} {config}')
    print(f'\nTotal: {len(experiments)} experiments x {JOBS_PER_CHAIN} segments = {total_jobs} SLURM jobs\n')


def submit_experiments(experiments, dry_run=False):
    print('Submitting...\n')
    all_job_ids = []

    for mk, var, seed, config in experiments:
        display = MODELS[mk]['display']
        job_ids = submit_chain(config, seed, dry_run=dry_run)
        all_job_ids.extend(job_ids)

        print(f'{display} / {var} / seed{seed}:')
        for i, jid in enumerate(job_ids):
            dep = f' (depends on {job_ids[i-1]})' if i > 0 else ''
            print(f'  Segment {i+1}: {jid}{dep}')
        print()

    if not dry_run:
        print(f'Monitor: squeue -u $USER')
        print(f'Cancel all: scancel {" ".join(all_job_ids)}')


def main():
    parser = argparse.ArgumentParser(description='HPC job launcher for geothermal surrogate models')
    parser.add_argument('--models', type=str, default=None,
                        help=f'Comma-separated model keys: {",".join(MODELS.keys())}')
    parser.add_argument('--variants', type=str, default=None,
                        help=f'Comma-separated variants: {",".join(VARIANTS)}')
    parser.add_argument('--seeds', type=str, default=None,
                        help='JSON dict mapping model key to comma-separated seeds, '
                             'e.g. \'{"fno":"42,123","unet3d":"42"}\'')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print sbatch commands without submitting')
    args = parser.parse_args()

    has_cli_args = args.models is not None
    dry_run = args.dry_run

    if has_cli_args:
        if args.variants is None or args.seeds is None:
            print('CLI mode requires --models, --variants, and --seeds.')
            sys.exit(1)
        cli_mode(args, dry_run)
    else:
        interactive_mode(dry_run)


if __name__ == '__main__':
    main()
