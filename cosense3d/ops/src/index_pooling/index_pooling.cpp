/*
Written by Yunshuang Yuan
*/
#include <vector>
#include <ATen/cuda/CUDAEvent.h>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include "index_pooling_kernel.h"

void index_pooling_forward(
    int m, int c, AT x_tensor, AT c_indices_tensor, AT out_tensor, AT out_indices_tensor
    )
{
    const float* x = x_tensor.data_ptr<float>();
    const int* c_indices = c_indices_tensor.data_ptr<int>();
    float* out = out_tensor.data_ptr<float>();
    const int* out_indices = out_indices_tensor.data_ptr<int>();

    index_pooling_forward_launcher(
        m, c, x, c_indices, out, out_indices
    );
}

void index_pooling_backward(
   int m, int c, AT c_indices_tensor, AT out_indices_tensor, AT grad_x_tensor, AT grad_out_tensor
    )
{
    const int* c_indices = c_indices_tensor.data_ptr<int>();
    const int* out_indices = out_indices_tensor.data_ptr<int>();

    float* grad_x = grad_x_tensor.data_ptr<float>();
    const float* grad_out = grad_out_tensor.data_ptr<float>();

    index_pooling_backward_launcher(
        m, c, c_indices, out_indices, grad_x, grad_out
    );
}