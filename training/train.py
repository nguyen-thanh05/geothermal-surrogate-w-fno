import argparse
import os
import yaml
from training.loop import run_training


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

    running_dir = cfg['checkpoints']['running_dir']
    resume_path = cfg['checkpoints'].get('resume_path',
        os.path.join(running_dir, 'resume_checkpoint.pt'))

    run_training(cfg, args, resume_path=resume_path)


if __name__ == '__main__':
    main()
