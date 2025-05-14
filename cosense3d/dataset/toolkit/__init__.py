import open3d as o3d
import copy
import numpy as np


def register_pcds(source_cloud, target_cloud, initial_transf, thr=0.2, visualize=False, title="PCD"):
    # Load point clouds
    if isinstance(source_cloud, str):
        source_cloud = o3d.io.read_point_cloud(source_cloud)
    if isinstance(target_cloud, str):
        target_cloud = o3d.io.read_point_cloud(target_cloud)

    # source_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2, max_nn=50))
    # target_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2, max_nn=50))

    # Perform ICP registration
    icp_result = initial_transf
    if not isinstance(thr, list):
        thr = [thr]

    icp_result = o3d.pipelines.registration.registration_icp(
        source_cloud, target_cloud, thr[0], initial_transf,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())

    if len(thr) > 1:
        for x in thr[1:]:
            icp_result = o3d.pipelines.registration.registration_icp(
                            source_cloud, target_cloud, x, icp_result.transformation,
                      o3d.pipelines.registration.TransformationEstimationPointToPoint())

    # Obtain the final transformation matrix
    # transformation_matrix = initial_transf
    transformation_matrix = icp_result.transformation

    if visualize:
        # Apply the final transformation to the source point cloud
        source_aligned0 = copy.deepcopy(source_cloud).transform(initial_transf)
        source_aligned = copy.deepcopy(source_cloud).transform(transformation_matrix)
    #
    #     src_pts = np.array(source_cloud.points)
    #     src_pts_aligned = np.array(source_aligned.points)
    #     tgt_pts = np.array(target_cloud.points)
    #     src_angles = (np.arctan2(src_pts[:, 1], src_pts[:, 0]) + np.pi * 3 - np.deg2rad(100)) % ( 2 * np.pi)
    #     tgt_angles = (np.arctan2(tgt_pts[:, 1], tgt_pts[:, 0]) + np.pi * 3 - np.deg2rad(255)) % ( 2 * np.pi)
    #     steps = 10
    #     res = 1 / steps
    #     pcds = []
    #     for i in range(steps):
    #         mask_src = (src_angles >= np.pi * 2 * i * res) & (src_angles < np.pi * 2 * (i + 1) * res)
    #         mask_tgt = (tgt_angles >= np.pi * 2 * i * res) & (tgt_angles < np.pi * 2 * (i + 1) * res)
    #
    #         cur_src_cloud = o3d.geometry.PointCloud()
    #         cur_tgt_cloud = o3d.geometry.PointCloud()
    #         cur_src_cloud.points = o3d.utility.Vector3dVector(src_pts[mask_src])
    #         cur_tgt_cloud.points = o3d.utility.Vector3dVector(tgt_pts[mask_tgt])
    #         cur_src_cloud.paint_uniform_color([0, 0.0 + i / steps * 1.0, 0])
    #         cur_tgt_cloud.paint_uniform_color([0, 0, 0.2 + i / steps * 0.8])
    #         pcds += [cur_src_cloud]
    #     o3d.visualization.draw_geometries(pcds)

        # Visualize the aligned point clouds
        source_aligned0.paint_uniform_color([1, 0, 0])
        source_aligned.paint_uniform_color([1, 0.706, 0])
        target_cloud.paint_uniform_color([0, 0.651, 0.929])
        o3d.visualization.draw_geometries([source_aligned0, target_cloud], window_name=title)
        o3d.visualization.draw_geometries([source_aligned, target_cloud], window_name=title)

    return copy.deepcopy(transformation_matrix)


def callback_registrations(source, target, source_points, target_points):
    """
    Callback function for point picking. Registers two point clouds using selected corresponding points.
    """
    print("Point picking callback called!")

    # Corresponding points
    correspondences = np.asarray([source_points, target_points])

    # Create Open3D point cloud from numpy arrays
    source_pc = o3d.geometry.PointCloud()
    source_pc.points = o3d.utility.Vector3dVector(source.points[source_points])
    target_pc = o3d.geometry.PointCloud()
    target_pc.points = o3d.utility.Vector3dVector(target.points[target_points])

    # Perform registration
    transformation = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_pc, target_pc, correspondences,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
    )

    # Apply the transformation to the source point cloud
    source.transform(transformation.transformation)

    # Visualize the result
    o3d.visualization.draw_geometries([source, target])
    return transformation


def click_register(source, target):
    # Visualize the two point clouds
    o3d.visualization.draw_geometries([source, target])

    # Register point clouds by picking corresponding points
    print("Pick corresponding points in both point clouds. Press 'Q' to finish picking.")
    source_points = o3d.visualization.PointCloudPickPoints()
    target_points = o3d.visualization.PointCloudPickPoints()
    transformation = o3d.visualization.draw_geometries_with_editing(
        [source, target, source_points, target_points],
                     callback=callback_registrations,
                     window_name="Pick corresponding points")
    return transformation
