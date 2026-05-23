#!/bin/bash

echo "Submitting FNO baseline jobs (3 segments each)..."

# FNO homo: 3 chained jobs
JOB=$(sbatch slurm/fno_homo.sh | awk '{print $4}')
JOB=$(sbatch --dependency=afterok:$JOB slurm/fno_homo.sh | awk '{print $4}')
JOB=$(sbatch --dependency=afterok:$JOB slurm/fno_homo.sh | awk '{print $4}')
echo "FNO homo chain submitted (last job: $JOB)"

# FNO hetero: 3 chained jobs
JOB=$(sbatch slurm/fno_hetero.sh | awk '{print $4}')
JOB=$(sbatch --dependency=afterok:$JOB slurm/fno_hetero.sh | awk '{print $4}')
JOB=$(sbatch --dependency=afterok:$JOB slurm/fno_hetero.sh | awk '{print $4}')
echo "FNO hetero chain submitted (last job: $JOB)"

echo "Done. Monitor with: squeue -u $USER"
