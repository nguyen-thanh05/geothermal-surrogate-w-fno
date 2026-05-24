#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=96G
#SBATCH --time=11:55:00
#SBATCH --account=def-juliana2
#SBATCH --output=/home/thanh2/projects/def-juliana2/thanh2/master-research/geothermal-surrogate-w-fno/logs/%x_%j.out
#SBATCH --mail-user=thanh2@ualberta.ca
#SBATCH --mail-type=ALL

if [ -z "$CONFIG" ] || [ -z "$SEED" ]; then
    echo "ERROR: CONFIG and SEED must be set."
    echo "Usage: sbatch --export=ALL,CONFIG=configs/fno_homo.yml,SEED=42 slurm/train.sh"
    exit 1
fi

export results=$SLURM_TMPDIR/results
export data=$SLURM_TMPDIR/data

module load python/3.11
module load cuda
source /home/thanh2/projects/def-juliana2/thanh2/.torch/bin/activate

echo "Config: $CONFIG | Seed: $SEED | Job: $SLURM_JOB_ID"

python training/train.py --config "$CONFIG" --hpc true --seed "$SEED"
