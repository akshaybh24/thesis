#!/bin/bash
#SBATCH --job-name=matmult
#SBATCH --account=vusr120677
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=120:00:00
#SBATCH --array=1-8
#SBATCH --output=matmult_%A_%a.out
#SBATCH --error=matmult_%A_%a.err

set -euo pipefail
cd /gpfs/home3/abharos/gen_prog_mm-main/rust/mat_mult

cmd=$(sed -n "${SLURM_ARRAY_TASK_ID}p" runs.txt)
if [ -z "$cmd" ]; then
  echo "No command for task ${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi

./target/release/mat_mult $cmd