#!/bin/bash
#SBATCH --job-name=hybridCOREL_comparisonDT
#SBATCH --array=0-749
#SBATCH --time=01:20:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err

# Load Python module 
module load python/3.10



# Create folders if not exist
mkdir -p logs

# Run script
python Run_HybridCOREL.py \
    --dataset compas \
    --model HybridCORELSPostClassifier \
    --local_id $SLURM_ARRAY_TASK_ID
