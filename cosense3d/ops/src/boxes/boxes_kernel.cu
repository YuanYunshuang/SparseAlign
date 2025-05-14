/*
Written by Shaoshuai Shi
Modified by Yunshuang Yuan
*/

#include <math.h>
#include <stdio.h>
#include "boxes_kernel.h"

#define THREADS_PER_BLOCK 256
#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))


__device__ inline void lidar_to_local_coords(float shift_x, float shift_y, float rot_angle, float &local_x, float &local_y){
    float cosa = cos(-rot_angle), sina = sin(-rot_angle);
    local_x = shift_x * cosa + shift_y * (-sina);
    local_y = shift_x * sina + shift_y * cosa;
}


__device__ inline int check_pt_in_box3d(const float *pt, const float *box3d, float &local_x, float &local_y){
    // param pt: (x, y, z)
    // param box3d: [x, y, z, dx, dy, dz, heading] (x, y, z) is the box center

    const float MARGIN = 1e-5;
    float x = pt[0], y = pt[1], z = pt[2];
    float cx = box3d[0], cy = box3d[1], cz = box3d[2];
    float dx = box3d[3], dy = box3d[4], dz = box3d[5], rz = box3d[6];

    if (fabsf(z - cz) > dz / 2.0) return 0;
    lidar_to_local_coords(x - cx, y - cy, rz, local_x, local_y);
    float in_flag = (fabs(local_x) < dx / 2.0 + MARGIN) & (fabs(local_y) < dy / 2.0 + MARGIN);
    return in_flag;
}


__global__ void points_in_boxes_kernel(int batch_size, int boxes_num, int pts_num, const float *boxes,
    const float *pts, int *box_idx_of_points){
    // params boxes: (B, N, 7) [x, y, z, dx, dy, dz, heading] (x, y, z) is the box center
    // params pts: (B, npoints, 3) [x, y, z] in LiDAR coordinate
    // params boxes_idx_of_points: (B, npoints), default -1

    int bs_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= batch_size || pt_idx >= pts_num) return;

    boxes += bs_idx * boxes_num * 7;
    pts += bs_idx * pts_num * 3 + pt_idx * 3;
    box_idx_of_points += bs_idx * pts_num + pt_idx;

    float local_x = 0, local_y = 0;
    int cur_in_flag = 0;
    for (int k = 0; k < boxes_num; k++){
        cur_in_flag = check_pt_in_box3d(pts, boxes + k * 7, local_x, local_y);
        if (cur_in_flag){
            box_idx_of_points[0] = k;
            break;
        }
    }
}


void points_in_boxes_launcher(int batch_size, int boxes_num, int pts_num, const float *boxes,
    const float *pts, int *box_idx_of_points){
    // params boxes: (B, N, 7) [x, y, z, dx, dy, dz, heading] (x, y, z) is the box center
    // params pts: (B, npoints, 3) [x, y, z]
    // params boxes_idx_of_points: (B, npoints), default -1
    cudaError_t err;

    dim3 blocks(DIVUP(pts_num, THREADS_PER_BLOCK), batch_size);
    dim3 threads(THREADS_PER_BLOCK);
    points_in_boxes_kernel<<<blocks, threads>>>(batch_size, boxes_num, pts_num, boxes, pts, box_idx_of_points);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }

#ifdef DEBUG
    cudaDeviceSynchronize();  // for using printf in kernel function
#endif
}