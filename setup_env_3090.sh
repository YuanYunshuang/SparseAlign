#!/bin/bash

set -e

# Set color codes
GREEN='\033[0;32m'
NC='\033[0m' # No Color

#conda create -n cosense3d python=3.8
#conda activate cosense3d

echo -e "${GREEN}[INFO] Install dependencies...${NC}"
conda install openblas-devel -c anaconda -y
conda install -c conda-forge libstdcxx-ng libffi -y
sudo apt install build-essential python3-dev libopenblas-dev -y

echo -e "${GREEN}[INFO] Installing pytorch essentials...${NC}"
#conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

echo - e "${GREEN}[INFO] Installing MinkowskiEngine...${NC}"
# for old version of pip
#pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps \
#    --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" \
#    --install-option="--blas=openblas"
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps \
    --global-option="--blas_include_dirs=${CONDA_PREFIX}/include" \
    --global-option="--blas=openblas"
#pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps \
#    --config-settings="--blas_include_dirs=${CONDA_PREFIX}/include" \
#    --config-settings="--blas=openblas"

echo -e "${GREEN}[INFO] Installing cuda_ops...${NC}"
cd cosense3d/ops && pip install . && cd ../..

echo -e "${GREEN}[INFO] Installing requirements...${NC}"
pip install -r requirements_cosense_3090.txt
pip install -r requirements_ui.txt

echo -e "${GREEN}[INFO] Done.${NC}"

TORCH="$(python -c "import torch; print(torch.__version__)")"
export OMP_NUM_THREADS=16
ME="$(python  -W ignore -c "import MinkowskiEngine as ME; print(ME.__version__)")"

echo -e "${GREEN}[INFO] Finished the installation!"
echo "[INFO] ========== Configurations =========="
echo "[INFO] PyTorch version: $TORCH"
echo "[INFO] MinkowskiEngine version: $ME"
echo -e "[INFO] ====================================${NC}"


#### CONDA ENV error

#  conda install -c conda-forge libstdcxx-ng
#  conda install -c conda-forge libffi

