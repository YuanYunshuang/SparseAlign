import copy
import open3d as o3d
import numpy as np


def register_points(source, target, init_transf=None, thr=[2.0, 0.5]):
    source_cloud = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(source.contiguous().detach().cpu().numpy()))
    target_cloud = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(target.contiguous().detach().cpu().numpy()))

    # Perform ICP registration
    if init_transf is None:
        icp_result = np.eye(4)
    else:
        icp_result = init_transf.detach().cpu().numpy()
    if not isinstance(thr, list):
        thr = [thr]

    icp_result = o3d.pipelines.registration.registration_icp(
        source_cloud, target_cloud, thr[0], icp_result,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())

    if len(thr) > 1:
        for x in thr[1:]:
            icp_result = o3d.pipelines.registration.registration_icp(
                source_cloud, target_cloud, x, icp_result.transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPoint())

    transform = copy.deepcopy(icp_result.transformation)
    source_transformed = np.array(copy.deepcopy(source_cloud).transform(transform).points)

    return transform, source_transformed