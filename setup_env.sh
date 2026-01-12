#!/bin/bash

########Base ENV
# Ubuntu = 20.04
# Cuda = 11.8
# Pytorch = 2.1.2
# set -e
ENV_NAME=sa
PY_VER=3.8
CUDA=11.3

set -Eeuo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

trap 'echo -e "${RED}❌ FAILED at line $LINENO. Aborting.${NC}"; exit 1' ERR
run() {
    echo -e "${GREEN}[RUN] $*${NC}"
    "$@"
}

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

    conda install -y conda=24

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
        echo "✅ g++ upgraded."
    else
        echo "✔ g++ is new enough"
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

if [[ "$CUDA_VER" != $CUDA ]]; then
    echo "[ERROR] CUDA $CUDAis required, but found CUDA $CUDA_VER"
    exit 1
fi

echo -e "${GREEN}[INFO] CUDA $CUDA verified.${NC}"
# -----------------------------------------------------

echo -e "${GREEN}[INFO] Create conda environment...${NC}"

source "$(conda info --base)/etc/profile.d/conda.sh"
# Create env if it does not exist
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Creating env $ENV_NAME (python $PY_VER)..."
    conda create -y -n "$ENV_NAME" python="$PY_VER"
else
    echo "Env $ENV_NAME already exists."
    echo
    echo "Choose an action:"
    echo "  [R] Remove and recreate"
    echo "  [O] Overwrite packages inside it"
    echo "  [A] Abort"
    echo "=============================================="
    read -p "Enter R / O / A: " choice

    case "$choice" in
        R|r)
            echo "Removing environment '$ENV_NAME'..."
            conda deactivate 2>/dev/null || true
            conda remove -n "$ENV_NAME" --all -y
            echo "Recreating '$ENV_NAME'..."
            conda create -n "$ENV_NAME" python=$PY_VER -y
            ;;
        O|o)
            echo "Keeping environment, will overwrite packages."
            ;;
        A|a)
            echo "Aborted by user."
            exit 0
            ;;
        *)
            echo "Invalid choice."
            exit 1
            ;;
    esac
fi

# Activate env
conda activate "$ENV_NAME"
echo "Active env: $CONDA_DEFAULT_ENV"

######################################################################
echo -e "${GREEN}[INFO] Installing compiling essentials...${NC}"
pip install -U setuptools==68 ninja==1.11.1.1
run conda install openblas-devel -c anaconda -y
run conda install -c conda-forge libstdcxx-ng libffi -y
run sudo apt install -y build-essential python3-dev python3-distutils libopenblas-dev

echo -e "${GREEN}[INFO] Installing pytorch essentials...${NC}"
# run pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
#     --index-url https://download.pytorch.org/whl/cu118
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

echo -e "${GREEN}[INFO] Installing requirements...${NC}"
run pip install -r requirements_cosense_3090.txt
run pip install -r requirements_ui.txt

echo -e "${GREEN}[INFO] Installing MinkowskiEngine...${NC}"
# run pip install setuptools==59.8.0 # ME installation may fail with higher versions
# PIP_VERSION=$(pip --version | awk '{print $2}')
# ver() {
#   printf "%03d%03d%03d\n" ${1//./ }
# }
# echo "$(ver "$PIP_VERSION")" -ge "$(ver 23.1.0)" 
# if [ "$(ver "$PIP_VERSION")" -ge "$(ver 23.1.0)" ]; then
#     run pip install -U -v --no-deps \
#       "git+https://github.com/NVIDIA/MinkowskiEngine" \
#       --config-settings="--blas_include_dirs=${CONDA_PREFIX}/include" \
#       --config-settings="--blas=openblas"
# else
run pip install -U -v --no-deps --no-cache-dir\
    "git+https://github.com/NVIDIA/MinkowskiEngine" \
    --global-option="--blas_include_dirs=${CONDA_PREFIX}/include" \
    --global-option="--blas=openblas"
# fi

echo -e "${GREEN}[INFO] Installing cuda_ops...${NC}"
run bash -c "cd cosense3d/ops && pip install ."

echo -e "${GREEN}[INFO] Done.${NC}"

TORCH="$(python -c "import torch; print(torch.__version__)")"
export OMP_NUM_THREADS=16
ME="$(python  -W ignore -c "import MinkowskiEngine as ME; print(ME.__version__)")"

echo -e "${GREEN}[INFO] Finished the installation!"
echo "[INFO] ========== Configurations =========="
echo "[INFO] PyTorch version: $TORCH"
echo "[INFO] MinkowskiEngine version: $ME"
echo -e "[INFO] ====================================${NC}"




