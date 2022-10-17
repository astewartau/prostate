wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh -O miniconda.sh
chmod +x miniconda.sh
./miniconda.sh
rm miniconda.sh
conda create --name prostate
conda activate prostate
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install -c fastchan fastai
git clone https://github.com/kbressem/faimed3d.git
cd faimed3d/
python setup.py install --user
