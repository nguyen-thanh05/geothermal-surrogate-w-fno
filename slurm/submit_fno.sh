#!/bin/bash

echo "Submitting FNO baseline jobs (3 segments each, seed 42)..."

for config in configs/fno_m8x32x16_h64_homo.yml configs/fno_m8x32x16_h64_hetero.yml \
              configs/fno_m4x16x8_h64_homo.yml configs/fno_m4x16x8_h64_hetero.yml \
              configs/fno_m4x16x8_h128_homo.yml configs/fno_m4x16x8_h128_hetero.yml; do
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
