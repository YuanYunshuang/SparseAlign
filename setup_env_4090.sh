#!/bin/bash

########Base ENV
# Ubuntu = 20.04
# Cuda = 11.8
# Pytorch = 2.1.2


set -e
# ---------------- Conda VERSION CHECK ----------------
TARGET_MAJOR=24

if ! command -v conda &>/dev/null; then
    echo "❌ Conda not found."
    exit 1
fi

CUR_VER=$(conda --version | awk '{print $2}')
CUR_MAJOR=$(echo "$CUR_VER" | cut -d. -f1)

echo "Current conda version: $CUR_VER"

if (( CUR_MAJOR < TARGET_MAJOR )); then
    echo "⬆ Upgrading conda to 24.x ..."

    # Make sure base env is active
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate base

    conda install -n base -y conda=24

    echo "✅ Conda upgraded."
else
    echo "✔ Conda already 24.x or newer."
fi

echo "Final version:"
conda --version

# ---------------- g++ VERSION CHECK ----------------
MIN_VER=9
TARGET_VER=10

echo "Checking g++ version..."

if ! command -v g++ &>/dev/null; then
    echo "g++ not found. Installing g++-$TARGET_VER..."
    sudo apt update
    sudo apt install -y g++-$TARGET_VER gcc-$TARGET_VER
else
    CUR_VER=$(g++ -dumpversion | cut -d. -f1)
    echo "Current g++ version: $CUR_VER"

    if (( CUR_VER < MIN_VER )); then
        echo "g++ too old (<$MIN_VER). Upgrading to g++-$TARGET_VER..."
        sudo apt update
        sudo apt install -y g++-$TARGET_VER gcc-$TARGET_VER

        echo "Registering g++-$TARGET_VER as default..."
        sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-$TARGET_VER 100
        sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-$TARGET_VER 100
        sudo update-alternatives --set gcc /usr/bin/gcc-$TARGET_VER
        sudo update-alternatives --set g++ /usr/bin/g++-$TARGET_VER
    else
        echo "g++ is new enough ✔"
    fi
fi

echo "Final compiler:"
g++ --version | head -n1


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

echo -e "${GREEN}[INFO] Create conda environment...${NC}"
ENV_NAME=sparse-align
PY_VER=3.8

source "$(conda info --base)/etc/profile.d/conda.sh"
# Create env if it does not exist
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Creating env $ENV_NAME (python $PY_VER)..."
    conda create -y -n "$ENV_NAME" python="$PY_VER"
else
    echo "Env $ENV_NAME already exists."
fi

echo -e "${GREEN}[INFO] Installing compiling essentials...${NC}"
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
# pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps \
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

