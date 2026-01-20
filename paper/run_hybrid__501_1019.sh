#!/bin/bash
#SBATCH --job-name=hybrid_fairness
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=730-749
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1


module load python/3.10


python run_experiments.py --expe_id=$SLURM_ARRAY_TASK_ID
