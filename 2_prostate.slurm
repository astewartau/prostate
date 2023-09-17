#!/bin/bash
#SBATCH --partition=gpu_cuda
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --account=a_barth
#SBATCH --export=PATH,TERM,HOME,LANG
#SBATCH --output=res_%j.out
#SBATCH --error=res_%j.err


# Miniconda
export CONDA_HOME=$HOME/miniconda3

# Activate Conda
source $CONDA_HOME/etc/profile.d/conda.sh

export LD_LIBRARY_PATH=$CONDA_HOME/envs/prostate/lib:$LD_LIBRARY_PATH

# Activate your environment
conda activate prostate

python 2_prostate.py $MODEL_NAME $FOLD_ID

