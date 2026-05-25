#!/usr/bin/env python3
"""Remove stale checkpoint files for completed experiments.

Scans checkpoints/ for resume and optimizer files whose corresponding
final checkpoint already exists, meaning training finished and the
intermediate files are no longer needed.

Dry-run by default. Pass --delete to actually remove files.
"""
import argparse
import os
import glob


def find_stale_files(ckpt_root='checkpoints'):
    running_dir = os.path.join(ckpt_root, 'running')
    if not os.path.isdir(running_dir):
        print(f"No running directory at {running_dir}")
        return []

    stale = []
    for seed_dir in sorted(glob.glob(os.path.join(running_dir, 'seed*'))):
        seed_name = os.path.basename(seed_dir)
        for f in sorted(os.listdir(seed_dir)):
            if not f.endswith('.pt'):
                continue
            base = f.replace('_resume_optim.pt', '').replace('_resume.pt', '')
            final_name = f"{base}_final.pth"
            final_path = os.path.join(ckpt_root, seed_name, final_name)
            if os.path.isfile(final_path):
                stale.append(os.path.join(seed_dir, f))

    return stale


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--ckpt-root', default='checkpoints',
                        help='Root checkpoint directory (default: checkpoints)')
    parser.add_argument('--delete', action='store_true',
                        help='Actually delete stale files (default: dry-run)')
    args = parser.parse_args()

    stale = find_stale_files(args.ckpt_root)

    if not stale:
        print("No stale checkpoint files found.")
        return

    total_bytes = sum(os.path.getsize(f) for f in stale)
    total_mb = total_bytes / (1024 * 1024)

    print(f"Found {len(stale)} stale file(s) ({total_mb:.1f} MB):")
    for f in stale:
        size_mb = os.path.getsize(f) / (1024 * 1024)
        print(f"  {f}  ({size_mb:.1f} MB)")

    if args.delete:
        for f in stale:
            os.remove(f)
        print(f"\nDeleted {len(stale)} file(s), freed {total_mb:.1f} MB.")
    else:
        print(f"\nDry run. Pass --delete to remove these files.")


if __name__ == '__main__':
    main()
