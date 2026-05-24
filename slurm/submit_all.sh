#!/bin/bash

echo "Submitting all baseline jobs (3 segments each, seed 42)..."

for config in configs/fno_homo.yml configs/fno_hetero.yml \
              configs/unet_homo.yml configs/unet_hetero.yml \
              configs/loglo_homo.yml configs/loglo_hetero.yml; do
    name=$(basename "$config" .yml)
    JOB=$(sbatch --job-name="${name}_s42" \
          --export=ALL,CONFIG=$config,SEED=42 \
          slurm/train.sh | awk '{print $4}')
    JOB=$(sbatch --job-name="${name}_s42" \
          --dependency=afterok:$JOB \
          --export=ALL,CONFIG=$config,SEED=42 \
          slurm/train.sh | awk '{print $4}')
    JOB=$(sbatch --job-name="${name}_s42" \
          --dependency=afterok:$JOB \
          --export=ALL,CONFIG=$config,SEED=42 \
          slurm/train.sh | awk '{print $4}')
    echo "$name chain submitted (last job: $JOB)"
done

echo "Done. Monitor with: squeue -u $USER"
