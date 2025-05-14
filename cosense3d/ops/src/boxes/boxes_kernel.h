#pragma once
#ifndef _boxes_KERNEL
#define _boxes_KERNEL
#include <vector>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>

#define AT at::Tensor

void points_in_boxes_cpu(AT boxes_tensor, AT pts_tensor, AT pts_indices_tensor);
void points_in_boxes_gpu(AT boxes_tensor, AT pts_tensor, AT box_idx_of_points_tensor);

#ifdef __cplusplus
extern "C" {
#endif

void points_in_boxes_launcher(int batch_size, int boxes_num, int pts_num, const float *boxes,
    const float *pts, int *box_idx_of_points);

#ifdef __cplusplus
}
#endif
#endif
