/*
Reference paper:  https://arxiv.org/abs/1907.03670
Written by Shaoshuai Shi
Modified by Yunshuang Yuan
*/

#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <assert.h>
#include "boxes_kernel.h"


void points_in_boxes_gpu(AT boxes_tensor, AT pts_tensor, AT box_idx_of_points_tensor){
    // params boxes: (B, N, 7) [x, y, z, dx, dy, dz, heading] (x, y, z) is the box center
    // params pts: (B, npoints, 3) [x, y, z]
    // params boxes_idx_of_points: (B, npoints), default -1

//    CHECK_INPUT(boxes_tensor);
//    CHECK_INPUT(pts_tensor);
//    CHECK_INPUT(box_idx_of_points_tensor);

    int batch_size = boxes_tensor.size(0);
    int boxes_num = boxes_tensor.size(1);
    int pts_num = pts_tensor.size(1);

    const float *boxes = boxes_tensor.data<float>();
    const float *pts = pts_tensor.data<float>();
    int *box_idx_of_points = box_idx_of_points_tensor.data<int>();

    points_in_boxes_launcher(batch_size, boxes_num, pts_num, boxes, pts, box_idx_of_points);

}


inline void lidar_to_local_coords_cpu(float shift_x, float shift_y, float rot_angle, float &local_x, float &local_y){
    float cosa = cos(-rot_angle), sina = sin(-rot_angle);
    local_x = shift_x * cosa + shift_y * (-sina);
    local_y = shift_x * sina + shift_y * cosa;
}


inline int check_pt_in_box3d_cpu(const float *pt, const float *box3d, float &local_x, float &local_y){
    // param pt: (x, y, z)
    // param box3d: [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
    const float MARGIN = 1e-2;
    float x = pt[0], y = pt[1], z = pt[2];
    float cx = box3d[0], cy = box3d[1], cz = box3d[2];
    float dx = box3d[3], dy = box3d[4], dz = box3d[5], rz = box3d[6];

    if (fabsf(z - cz) > dz / 2.0) return 0;
    lidar_to_local_coords_cpu(x - cx, y - cy, rz, local_x, local_y);
    float in_flag = (fabs(local_x) < dx / 2.0 + MARGIN) & (fabs(local_y) < dy / 2.0 + MARGIN);
    return in_flag;
}


void points_in_boxes_cpu(AT boxes_tensor, AT pts_tensor, AT pts_indices_tensor){
    // params boxes: (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center, each box DO NOT overlaps
    // params pts: (num_points, 3) [x, y, z]
    // params pts_indices: (N, num_points)

    int boxes_num = boxes_tensor.size(0);
    int pts_num = pts_tensor.size(0);

    const float *boxes = boxes_tensor.data<float>();
    const float *pts = pts_tensor.data<float>();
    int *pts_indices = pts_indices_tensor.data<int>();

    float local_x = 0, local_y = 0;
    for (int i = 0; i < boxes_num; i++){
        for (int j = 0; j < pts_num; j++){
            int cur_in_flag = check_pt_in_box3d_cpu(pts + j * 3, boxes + i * 7, local_x, local_y);
            pts_indices[i * pts_num + j] = cur_in_flag;
        }
    }

}
