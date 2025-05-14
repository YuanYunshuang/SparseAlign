#!/bin/bash

export PYTHONPATH=.
export OMP_NUM_THREADS=16
torchrun \
--nproc_per_node=1 \
cosense3d/tools/agent_runner.py \
--config ./cosense3d/config/stream_lidar_st_v3.yaml \
--mode train \
--gpus 1 \
--batch-size 4