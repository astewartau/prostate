#!/usr/bin/env bash
#SBATCH --job-name=t1w_homogeneity
#SBATCH --output=t1w_homogeneity_%A_%a.out
#SBATCH --error=t1w_homogeneity_%A_%a.err
#SBATCH --array=0-25
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Load software modules
source ~/.bashrc

# Load necessary modules (adjust if needed)
conda activate prostate

# Gather T1w files from the dataset.
# This searches for files with "T1w_resliced" but NOT "homogeneity-correction".
mapfile -t T1_FILES < <(find . -type f -name "*T1w_resliced*.nii*" ! -name "*homogeneity-corrected*")

# Get the file corresponding to the array index.
FILE=${T1_FILES[$SLURM_ARRAY_TASK_ID]}
echo "Processing file: $FILE"

# Run the homogeneity correction script.
python homogeneity_correction.py "$FILE"
