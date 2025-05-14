/*
Written by Yunshuang Yuan
*/
#include "../cuda_utils.h"
#include "index_pooling_kernel.h"


__global__ void index_pooling_forward_kernel(
    int m, int c, const float* x, const int* c_indices, float* out, const int* out_indices
)
{
    // m: # of total mappings
    // c: # of  channels
    
    int map_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (map_idx >= m) return;

    int c_idx = c_indices[map_idx];
    int out_idx_ = out_indices[map_idx];
    int out_idx = out_idx_ * c + c_idx;

//     out[out_idx] += x[map_idx];
    atomicAdd(out + out_idx, x[map_idx]); // atomic operation can avoid race condition

//     if (map_idx>6421 && map_idx<6422+250) {printf("out_after: %f, x: %f, c_idx: %d, out_idx_: %d, out_idx: %d, map_idx: %d\n",
//                        out[out_idx], x[map_idx], c_idx, out_idx_, out_idx, map_idx);}
}

__global__ void index_pooling_backward_kernel(
    int m, int c, const int* c_indices, const int* out_indices, float* grad_x, const float* grad_out
)
{
    // m: # of total mappings
    // c: # of channels
    
    int map_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (map_idx >= m) return;

    int c_idx = c_indices[map_idx];
    int out_idx_ = out_indices[map_idx];
    int out_idx = out_idx_ * c + c_idx;
//     if (map_idx < 10) {printf("grad_out:%d, %f, grad_x:%d, %f, c_idx: %d, out_idx_: %d, map_idx: %d\n",
//                         out_idx, grad_out[out_idx], map_idx, grad_x[map_idx], c_idx, out_idx_, map_idx);}
    atomicAdd(grad_x + map_idx, grad_out[out_idx]);
//     grad_x[map_idx] += grad_out[out_idx];
}

void index_pooling_forward_launcher(
    int m, int c, const float* x, const int* c_indices, float* out, const int* out_indices
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dim3 blocks(DIVUP(m, THREADS_PER_BLOCK));
    dim3 threads(THREADS_PER_BLOCK);
    index_pooling_forward_kernel<<<blocks, threads, 0, stream>>>(
        m, c, x, c_indices, out, out_indices
    );
}

void index_pooling_backward_launcher(
    int m, int c, const int* c_indices, const int* out_indices, float* grad_x, const float* grad_out
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dim3 blocks(DIVUP(m, THREADS_PER_BLOCK));
    dim3 threads(THREADS_PER_BLOCK);
    index_pooling_backward_kernel<<<blocks, threads, 0, stream>>>(
        m, c, c_indices, out_indices, grad_x, grad_out
    );
}
