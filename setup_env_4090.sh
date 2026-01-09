#!/bin/bash

########Base ENV
# Ubuntu = 20.04
# Cuda = 11.8
# Pytorch = 2.1.2

set -e

# ---------------- CUDA VERSION CHECK ----------------
echo "[INFO] Checking CUDA version..."

if command -v nvcc &> /dev/null; then
    CUDA_VER=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
else
    echo "[ERROR] nvcc not found — CUDA is not installed or not in PATH."
    exit 1
fi

echo "[INFO] Detected CUDA version: $CUDA_VER"

if [[ "$CUDA_VER" != "11.8" ]]; then
    echo "[ERROR] CUDA 11.8 is required, but found CUDA $CUDA_VER"
    exit 1
fi

echo -e "${GREEN}[INFO] CUDA 11.8 verified.${NC}"
# -----------------------------------------------------

# Set color codes
GREEN='\033[0;32m'
NC='\033[0m' # No Color

#echo -e "${GREEN}[INFO] Create conda environment...${NC}"
#conda create -n cosense3d python=3.8
#conda activate cosense3d
conda install openblas-devel -c anaconda -y
conda install -c conda-forge libstdcxx-ng libffi -y
sudo apt install build-essential python3-dev libopenblas-dev -y

echo -e "${GREEN}[INFO] Installing pytorch essentials...${NC}"
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

echo -e "${GREEN}[INFO] Installing requirements...${NC}"
pip install -r requirements_cosense_4090.txt
pip install -r requirements_ui.txt

echo - e "${GREEN}[INFO] Installing MinkowskiEngine...${NC}"
# for old version of pip
#pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps \
#    --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" \
#    --install-option="--blas=openblas"
pip install setuptools==68.0.0 # to avoid error with latest versions of setuptools
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps \
    --global-option="--blas_include_dirs=${CONDA_PREFIX}/include" \
    --global-option="--blas=openblas"
#pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps \
#    --config-settings="--blas_include_dirs=${CONDA_PREFIX}/include" \
#    --config-settings="--blas=openblas"

echo -e "${GREEN}[INFO] Installing cuda_ops...${NC}"
cd cosense3d/ops && pip install . && cd ../..

echo -e "${GREEN}[INFO] Done.${NC}"

TORCH="$(python -c "import torch; print(torch.__version__)")"
export OMP_NUM_THREADS=16
ME="$(python  -W ignore -c "import MinkowskiEngine as ME; print(ME.__version__)")"

echo -e "${GREEN}[INFO] Finished the installation!"
echo "[INFO] ========== Configurations =========="
echo "[INFO] PyTorch version: $TORCH"
echo "[INFO] MinkowskiEngine version: $ME"
echo -e "[INFO] ====================================${NC}"


#

