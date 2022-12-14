# get miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh -O miniconda.sh
chmod +x miniconda.sh
./miniconda.sh
rm miniconda.sh

# setup conda environment
conda create --name prostate python=3.8
conda activate prostate
conda update -n base -c defaults conda
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install -c fastchan fastai
conda install ipykernel ipywidgets
#conda install -c conda-forge gxx_linux-64=11.1.0

# upate library path (add this to .bashrc)
export LD_LIBRARY_PATH="/scratch/itee/uqaste15/miniconda3/lib/:${LD_LIBRARY_PATH}"
pip install nibabel
