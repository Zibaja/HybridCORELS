#!/bin/bash
#SBATCH --job-name=hybridCOREL_bootstrap
#SBATCH --array=0-99
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err

# Load Python module 
module load python/3.10



# Create folders if not exist
mkdir -p logs
mkdir -p bootstrap_results

# Run script
python Run_Boots.py \
    --dataset adult \
    --model HybridCORELSPreClassifier \
    --seed 0 \
    --round_min 0 \
    --round_max 9 \
    --local_id $SLURM_ARRAY_TASK_ID