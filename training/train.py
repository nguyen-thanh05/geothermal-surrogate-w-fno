import argparse
import os
import yaml
from training.loop import run_training


def _inject_seed_into_paths(cfg, seed):
    ckpt = cfg['checkpoints']
    seed_dir = f'seed{seed}'

    ckpt['running_dir'] = os.path.join(ckpt['running_dir'], seed_dir, '')

    final_dir, final_name = os.path.split(ckpt['final_path'])
    ckpt['final_path'] = os.path.join(final_dir, seed_dir, final_name)

    if 'resume_path' in ckpt:
        resume_name = os.path.basename(ckpt['resume_path'])
        ckpt['resume_path'] = os.path.join(ckpt['running_dir'], resume_name)


def main():
    parser = argparse.ArgumentParser(description='Unified training entry point')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML config file')
    parser.add_argument('--hpc', type=lambda x: x.lower() == 'true',
                        default=False, help='Set to True if running on HPC')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    _inject_seed_into_paths(cfg, args.seed)

    running_dir = cfg['checkpoints']['running_dir']
    resume_path = cfg['checkpoints'].get('resume_path',
        os.path.join(running_dir, 'resume_checkpoint.pt'))

    run_training(cfg, args, resume_path=resume_path)


if __name__ == '__main__':
    main()
