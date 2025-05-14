# Installation

## Requirements
- Ubuntu LTS 20.04
- GPU: tested on *Nvidia RTX 3090 Ti* and  *Nvidia RTX 4090*
- Python: >= 3.8

## Installation options

### Via bash script
You can install the environment with our provided batch script with the following commands:
```bash
conda create -n consense3d python=3.8
conda activate cosense3d
cd OpenCosense3D 
# for Nvidia RTX 3090
bash setup_env_3090.sh
# for Nvidia RTX 4090
bash setup_env_4090.sh
```

### Step-by-step
If you confront with any errors at the script installation, please try step-by-step installation. 

1.Create conda environment and install dependencies.
```shell
conda create -n consense3d python=3.8
conda activate cosense3d
conda install openblas-devel -c anaconda -y
conda install -c conda-forge libstdcxx-ng libffi -y
sudo apt install build-essential python3-dev libopenblas-dev -y
```

2.Install pytorch and compile local Pytorch Extensions (CUDA nvcc compiler needed).
```shell
# For 3090
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 \
--extra-index-url https://download.pytorch.org/whl/cu113
# For 4090
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# Install extentions
cd cosense3d/ops
pip install . && cd ..
```

3.Install python packages.
```shell
# for 3090
pip install -r reququirements_cosense_3090.txt
# for 4090
pip install -r reququirements_cosense_4090.txt
# for Graphical Interface
pip install -r requirements_ui.txt
```

4.Install MinkovskiEngine.
```shell
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps \
    --global-option="--blas_include_dirs=${CONDA_PREFIX}/include" \
    --global-option="--blas=openblas"
export OMP_NUM_THREADS=16
```

5.Check Installation.
```shell
python -c "import torch; print(torch.__version__)" 
python  -W ignore -c "import MinkowskiEngine as ME; print(ME.__version__)"
``` 

