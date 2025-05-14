#include <torch/serialize/tensor.h>
#include <torch/extension.h>

#include "dot_product/dot_product_kernel.h"
#include "scalar_attention/scalar_attention_kernel.h"
#include "utils/boxes_kernel.h"
#include "index_pooling/index_pooling_kernel.h"
#include "iou_nms/iou3d_cpu.h"
#include "iou_nms/iou3d_nms.h"
#include "pointnet2_stack/ball_query_gpu.h"
#include "pointnet2_stack/group_points_gpu.h"
#include "pointnet2_stack/sampling_gpu.h"
#include "pointnet2_stack/interpolate_gpu.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dot_product_forward", &dot_product_forward, "dot_product_forward");
    m.def("dot_product_backward", &dot_product_backward, "dot_product_backward");
    m.def("scalar_attention_forward", &scalar_attention_forward, "scalar_attention_forward");
    m.def("scalar_attention_backward", &scalar_attention_backward, "scalar_attention_backward");
    m.def("index_pooling_forward", &index_pooling_forward, "index_pooling_forward");
    m.def("index_pooling_backward", &index_pooling_backward, "index_pooling_backward");
    m.def("points_in_boxes_gpu", &points_in_boxes_gpu, "points_in_boxes_gpu forward (CUDA)");
    m.def("points_in_boxes_cpu", &points_in_boxes_cpu, "points_in_boxes_cpu forward (CUDA)");
    m.def("boxes_overlap_bev_gpu", &boxes_overlap_bev_gpu, "oriented boxes overlap");
	m.def("boxes_iou_bev_gpu", &boxes_iou_bev_gpu, "oriented boxes iou");
	m.def("nms_gpu", &nms_gpu, "oriented nms gpu");
	m.def("nms_normal_gpu", &nms_normal_gpu, "nms gpu");
	m.def("boxes_iou_bev_cpu", &boxes_iou_bev_cpu, "oriented boxes iou");

	// pointnet2 stack
    m.def("ball_query_wrapper", &ball_query_wrapper_stack, "ball_query_wrapper_stack");
    m.def("furthest_point_sampling_wrapper", &furthest_point_sampling_wrapper, "furthest_point_sampling_wrapper");
    m.def("group_points_wrapper", &group_points_wrapper_stack, "group_points_wrapper_stack");
    m.def("group_points_grad_wrapper", &group_points_grad_wrapper_stack, "group_points_grad_wrapper_stack");
    m.def("three_nn_wrapper", &three_nn_wrapper_stack, "three_nn_wrapper_stack");
    m.def("three_interpolate_wrapper", &three_interpolate_wrapper_stack, "three_interpolate_wrapper_stack");
    m.def("three_interpolate_grad_wrapper", &three_interpolate_grad_wrapper_stack, "three_interpolate_grad_wrapper_stack");
}