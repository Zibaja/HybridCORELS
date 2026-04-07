#!/bin/bash
#SBATCH --job-name=hybridCOREL_bootstrap
#SBATCH --array=0-49
#SBATCH --time=08:00:00
#SBATCH --mem=500G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err

# Load Python module 
module load python/3.10



# Create folders if not exist
mkdir -p logs
mkdir -p bootstrap_results

# Run your script
python Run_Boots_HybridCOREL.py \
    --dataset compas \
    --model HybridCORELSPostClassifier \
    --seed 0 \
    --round_min 0 \
    --round_max 4 \
    --local_id $SLURM_ARRAY_TASK_ID