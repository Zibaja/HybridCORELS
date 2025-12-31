#!/bin/bash
#SBATCH --job-name=hybrid_fairness
#SBATCH --array=0-2399
#SBATCH --time=00:02:30
#SBATCH --cpus-per-task=1
#SBATCH --mem=400M
#SBATCH --output=logs/out_%A_%a.out
#SBATCH --error=logs/err_%A_%a.err

module load python/3.10
source ~/venvs/hybrid/bin/activate

python run_experiments.py --expe_id=$SLURM_ARRAY_TASK_ID
