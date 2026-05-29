import argparse
import gc
import json
import math
import os
import sys
import traceback

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.evaluate import evaluate_checkpoint
from evaluation.constants import CHANNEL_KEYS, CHANNEL_NAMES

SEEDS = [5, 42, 123, 1024, 2026]
MODELS = [
    'fno_m8x32x16_h64',
    'fno_m4x16x8_h64',
    'fno_m4x16x8_h128',
    'unet_d3',
    'unet_d4',
    'loglo',
]
VARIANTS = ['homo', 'hetero']

MODEL_DISPLAY = {
    'fno_m8x32x16_h64': 'FNO 8×32×16 h64',
    'fno_m4x16x8_h64': 'FNO 4×16×8 h64',
    'fno_m4x16x8_h128': 'FNO 4×16×8 h128',
    'unet_d3': 'UNet3D d3',
    'unet_d4': 'UNet3D d4',
    'loglo': 'LOGLO-FNO',
}


def discover_jobs(ckpt_root, homo_data, hetero_data):
    jobs = []
    for model in MODELS:
        for variant in VARIANTS:
            for seed in SEEDS:
                config_name = f'{model}_{variant}'
                ckpt_path = os.path.join(
                    ckpt_root, f'seed{seed}', f'{config_name}_resume.pt')
                config_path = os.path.join('configs', f'{config_name}.yml')
                data_path = hetero_data if variant == 'hetero' else homo_data
                out_path = os.path.join(
                    'evaluation_results', f'seed{seed}', f'{config_name}.json')

                exists = os.path.isfile(ckpt_path)
                jobs.append({
                    'model': model,
                    'variant': variant,
                    'seed': seed,
                    'config_path': config_path,
                    'checkpoint_path': ckpt_path,
                    'data_path': data_path,
                    'out_path': out_path,
                    'checkpoint_exists': exists,
                })
    return jobs


def run_evaluations(jobs, use_ema, resume):
    total = len([j for j in jobs if j['checkpoint_exists']])
    done = 0
    for job in jobs:
        label = f"{job['model']}_{job['variant']} seed={job['seed']}"

        if not job['checkpoint_exists']:
            print(f"[SKIP] {label} — checkpoint not found")
            continue

        if resume and os.path.isfile(job['out_path']):
            print(f"[SKIP] {label} — already evaluated")
            done += 1
            continue

        done += 1
        print(f"\n[{done}/{total}] Evaluating {label}")
        try:
            evaluate_checkpoint(
                job['config_path'], job['checkpoint_path'],
                job['data_path'], job['out_path'], use_ema=use_ema)
        except Exception:
            print(f"[ERROR] {label} failed:")
            traceback.print_exc()
            os.makedirs(os.path.dirname(job['out_path']), exist_ok=True)
            with open(job['out_path'], 'w') as f:
                json.dump({
                    'model_type': job['model'],
                    'variant': job['variant'],
                    'seed': job['seed'],
                    'error': traceback.format_exc(),
                }, f, indent=2)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def load_results(out_dir):
    results = {}
    for seed in SEEDS:
        seed_dir = os.path.join(out_dir, f'seed{seed}')
        if not os.path.isdir(seed_dir):
            continue
        for fname in os.listdir(seed_dir):
            if not fname.endswith('.json'):
                continue
            path = os.path.join(seed_dir, fname)
            with open(path) as f:
                data = json.load(f)
            key = fname.replace('.json', '')
            results.setdefault(key, []).append((seed, data))
    return results


def aggregate_and_report(out_dir):
    results = load_results(out_dir)
    notes = []

    for variant in VARIANTS:
        lines = []
        lines.append(f'## {variant.capitalize()} Variant\n')
        lines.append('| Model | Mean L2 | Final L2 | Drift (1e-4/step) | T RMSE (C) | P RMSE (kPa) | Amp Factor | Seeds |')
        lines.append('|-------|---------|----------|-------------------|------------|--------------|------------|-------|')

        for model in MODELS:
            key = f'{model}_{variant}'
            entries = results.get(key, [])

            valid = [(s, d) for s, d in entries if 'error' not in d]
            failed = [(s, d) for s, d in entries if 'error' in d]
            missing_seeds = set(SEEDS) - {s for s, _ in entries}

            if not valid:
                display = MODEL_DISPLAY.get(model, model)
                lines.append(f'| {display} | — | — | — | — | — | — | 0/5 |')
                if missing_seeds:
                    notes.append(f'{key}: missing seeds {sorted(missing_seeds)}')
                if failed:
                    notes.append(f'{key}: failed seeds {[s for s,_ in failed]}')
                continue

            incomplete = [(s, d) for s, d in valid
                          if not d.get('training_complete', True)]
            if incomplete:
                epochs_info = ', '.join(
                    f"seed{s}={d['training_epoch']+1}/{d['expected_epochs']}"
                    for s, d in incomplete)
                notes.append(f'{key}: incomplete training — {epochs_info}')

            n = len(valid)
            t_crit = {1: 0, 2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(n, 2.776)

            def agg(values):
                arr = np.array(values)
                m = arr.mean()
                if n < 2:
                    return m, 0.0
                ci = t_crit * arr.std(ddof=1) / math.sqrt(n)
                return m, ci

            mean_l2_vals = []
            final_l2_vals = []
            drift_vals = []
            t_rmse_vals = []
            p_rmse_vals = []
            amp_vals = []

            for _, d in valid:
                m = d['metrics']
                ch_mean = np.mean([m[k]['mean_l2_rel'] for k in CHANNEL_KEYS])
                ch_final = np.mean([m[k]['final_step_l2'] for k in CHANNEL_KEYS])
                ch_drift = np.mean([m[k]['drift_slope'] for k in CHANNEL_KEYS])
                t_rmse = np.mean([m[k]['rmse_physical']
                                  for k in ['temp_form', 'temp_frac']])
                p_rmse = np.mean([m[k]['rmse_physical']
                                  for k in ['pres_form', 'pres_frac']])
                ch_amp = np.mean([m[k]['amplification'] for k in CHANNEL_KEYS])
                mean_l2_vals.append(ch_mean)
                final_l2_vals.append(ch_final)
                drift_vals.append(ch_drift)
                t_rmse_vals.append(t_rmse)
                p_rmse_vals.append(p_rmse)
                amp_vals.append(ch_amp)

            ml, ml_ci = agg(mean_l2_vals)
            fl, fl_ci = agg(final_l2_vals)
            dr, dr_ci = agg(drift_vals)
            tr, tr_ci = agg(t_rmse_vals)
            pr, pr_ci = agg(p_rmse_vals)
            am, am_ci = agg(amp_vals)

            dr_scaled = dr * 1e4
            dr_ci_scaled = dr_ci * 1e4

            display = MODEL_DISPLAY.get(model, model)
            seed_label = f'{n}/5'
            if missing_seeds:
                notes.append(f'{key}: missing seeds {sorted(missing_seeds)}')
            if failed:
                notes.append(f'{key}: failed seeds {[s for s,_ in failed]}')

            lines.append(
                f'| {display} '
                f'| {ml:.4f}+/-{ml_ci:.4f} '
                f'| {fl:.4f}+/-{fl_ci:.4f} '
                f'| {dr_scaled:.2f}+/-{dr_ci_scaled:.2f} '
                f'| {tr:.2f}+/-{tr_ci:.2f} '
                f'| {pr:.1f}+/-{pr_ci:.1f} '
                f'| {am:.1f}+/-{am_ci:.1f} '
                f'| {seed_label} |'
            )

        lines.append('')
        report_path = os.path.join(out_dir, f'summary_{variant}.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print(f"Saved: {report_path}")

    if notes:
        notes_path = os.path.join(out_dir, 'notes.md')
        with open(notes_path, 'w', encoding='utf-8') as f:
            f.write('## Notes\n\n')
            for note in notes:
                f.write(f'- {note}\n')
        print(f"Saved: {notes_path}")

    combined = os.path.join(out_dir, 'summary_combined.md')
    with open(combined, 'w', encoding='utf-8') as f:
        f.write('# Model Evaluation Summary\n\n')
        f.write('Metrics averaged over 4 state channels, 50 test trajectories, '
                '156 autoregressive steps.\n')
        f.write('Values: mean +/- 95% CI across seeds.\n\n')
        for variant in VARIANTS:
            vpath = os.path.join(out_dir, f'summary_{variant}.md')
            if os.path.isfile(vpath):
                with open(vpath, encoding='utf-8') as vf:
                    f.write(vf.read())
                f.write('\n')
        if notes:
            f.write('## Notes\n\n')
            for note in notes:
                f.write(f'- {note}\n')
    print(f"Saved: {combined}")


def main():
    parser = argparse.ArgumentParser(description='Batch evaluation + aggregation')
    parser.add_argument('--homo-data', type=str, required=True)
    parser.add_argument('--hetero-data', type=str, required=True)
    parser.add_argument('--ckpt-root', type=str, default='checkpoints/running')
    parser.add_argument('--out-dir', type=str, default='evaluation_results')
    parser.add_argument('--use-ema', action='store_true', default=True)
    parser.add_argument('--no-ema', action='store_true')
    parser.add_argument('--resume', action='store_true',
                        help='Skip checkpoints that already have results')
    parser.add_argument('--skip-eval', action='store_true',
                        help='Only aggregate existing results')
    parser.add_argument('--models', type=str, default=None,
                        help='Comma-separated model filter')
    parser.add_argument('--variants', type=str, default=None,
                        help='Comma-separated variant filter (homo,hetero)')
    parser.add_argument('--seeds', type=str, default=None,
                        help='Comma-separated seed filter')
    args = parser.parse_args()

    use_ema = not args.no_ema

    if not args.skip_eval:
        jobs = discover_jobs(args.ckpt_root, args.homo_data, args.hetero_data)

        if args.models:
            filt = args.models.split(',')
            jobs = [j for j in jobs if j['model'] in filt]
        if args.variants:
            filt = args.variants.split(',')
            jobs = [j for j in jobs if j['variant'] in filt]
        if args.seeds:
            filt = [int(s) for s in args.seeds.split(',')]
            jobs = [j for j in jobs if j['seed'] in filt]

        print(f"Discovered {len(jobs)} jobs "
              f"({sum(j['checkpoint_exists'] for j in jobs)} with checkpoints)")
        run_evaluations(jobs, use_ema, args.resume)

    print("\n=== Aggregating results ===")
    aggregate_and_report(args.out_dir)


if __name__ == '__main__':
    main()
