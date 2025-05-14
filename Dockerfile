FROM pytorch/pytorch:2.2.1-cuda11.8-cudnn8-devel
LABEL hostname="cosense3d-docker"

##############################################
# You should modify this to match your GPU compute capability
# ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6"
##############################################

ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV OMP_NUM_THREADS 16

WORKDIR /project
COPY requirements.txt /project/requirements.txt
COPY ./cosense3d/ops/ /project/ops/

# Install dependencies
RUN apt-get update
RUN apt-get install -y git ninja-build cmake build-essential libopenblas-dev \
    xterm xauth openssh-server tmux wget mate-desktop-environment-core

RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

# For faster build, use more jobs.
ENV MAX_JOBS=4
RUN git clone --recursive "https://github.com/NVIDIA/MinkowskiEngine"
RUN cd MinkowskiEngine; python setup.py install --force_cuda --blas=openblas
RUN cd ..

RUN conda update conda -y
RUN apt-get update
RUN apt install python3-dev  -y
RUN apt install libgl1-mesa-glx libglib2.0-0 -y

RUN cd ops && pip install . && cd ..

RUN pip install -r requirements.txt

#RUN conda install pybind11 -y
#RUN conda install -c conda-forge libstdcxx-ng -y

WORKDIR /workspace
