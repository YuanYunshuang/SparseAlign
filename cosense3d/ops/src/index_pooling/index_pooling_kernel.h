/*
Written by Yunshuang Yuan
*/
#pragma once
#ifndef _index_pooling_KERNEL
#define _index_pooling_KERNEL
#include <vector>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>

#define AT at::Tensor

void index_pooling_forward(
    int m, int c, AT x_tensor, AT c_indices_tensor, AT out_tensor, AT out_indices_tensor
    );
void index_pooling_backward(
    int m, int c, AT c_indices_tensor, AT out_indices_tensor, AT grad_x_tensor, AT grad_out_tensor
    );

#ifdef __cplusplus
extern "C" {
#endif

void index_pooling_forward_launcher(
    int m, int c, const float* x, const int* c_indices, float* out, const int* out_indices
    );
void index_pooling_backward_launcher(
    int m, int c, const int* c_indices, const int* out_indices, float* grad_x, const float* grad_out
    );

#ifdef __cplusplus
}
#endif
#endif