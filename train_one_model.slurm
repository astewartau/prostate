#!/usr/bin/env bash
#SBATCH --partition=gpu_cuda
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --account=a_barth
#SBATCH --export=PATH,TERM,HOME,LANG
#SBATCH --output=res_%j.out
#SBATCH --error=res_%j.err

# Set the location of Miniconda
export CONDA_HOME=$HOME/miniconda3
echo "Using Miniconda from $CONDA_HOME"

# Source the conda profile script
source "$CONDA_HOME/etc/profile.d/conda.sh"
echo "Sourced $CONDA_HOME/etc/profile.d/conda.sh"

# Update the library path for the environment
export LD_LIBRARY_PATH="$CONDA_HOME/envs/prostate/lib:$LD_LIBRARY_PATH"
echo "Updated LD_LIBRARY_PATH to $LD_LIBRARY_PATH"

# Activate the conda environment
echo "Activating conda environment"
conda activate prostate39

# Check the Python path
echo "Python path is $(which python)"

# Print the Python executable and version
echo "Python version is $(python --version)"

# Test the Python installation
echo "Testing Python"
python -c "import sys; print(sys.executable, sys.version, flush=True)"

# Print a message to confirm the environment variables
echo "Starting train_one_model.py with MODEL_NAME=${MODEL_NAME} and FOLD_ID=${FOLD_ID}"

# Run the training script with command-line arguments
python train_one_model.py "$MODEL_NAME" "$FOLD_ID"
