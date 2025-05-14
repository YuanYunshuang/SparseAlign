import numpy as np
from torch import nn
import open3d as o3d
import g2o
import numpy as np

from cosense3d.utils.misc import load_json
from cosense3d.utils.box_utils import transform_boxes_3d
from cosense3d.utils.pclib import pose_to_transformation
from cosense3d.utils.vislib import draw_points_boxes_plt, plt


class PoseGraphOptimization2D(g2o.SparseOptimizer):
    def __init__(self, verbose=False):
        super().__init__()
        # solver = g2o.BlockSolverSE2(g2o.LinearSolverCholmodSE2())
        solver = g2o.BlockSolverSE2(g2o.LinearSolverDenseSE2())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        super().set_algorithm(solver)
        super().set_verbose(verbose)

    def optimize(self, max_iterations=1000):
        super().initialize_optimization()
        super().optimize(max_iterations)

    def add_vertex(self, id, pose, fixed=False, SE2=True):
        if SE2:
            v = g2o.VertexSE2()
        else:
            v = g2o.VertexPointXY()
        v.set_estimate(pose)
        v.set_id(id)
        v.set_fixed(fixed)
        super().add_vertex(v)

    def add_edge(self, vertices, measurement,
                 information=np.identity(3),
                 robust_kernel=None, SE2=True):
        """
        Args:
            measurement: g2o.SE2
        """
        if SE2:
            edge = g2o.EdgeSE2()
        else:
            edge = g2o.EdgeSE2PointXY()

        for i, v in enumerate(vertices):
            if isinstance(v, int) or isinstance(v, np.int64):
                v = self.vertex(v)
            edge.set_vertex(i, v)

        edge.set_measurement(measurement)  # relative pose shape [3, 1] / [2, 1]
        edge.set_information(information)  # importance of each component shape [3, 3] / [2, 2]
        if robust_kernel is not None:
            edge.set_robust_kernel(robust_kernel)
        super().add_edge(edge)

    def get_pose(self, id):
        pose = self.vertex(id).estimate()
        if not isinstance(pose, np.ndarray):
            pose = pose.vector()
        return pose

    def add_graph(self, boxes_list, pose_list):
        ego_boxes = boxes_list[0]
        self.add_vertex(id=0, pose=pose_list[0], fixed=True)



if __name__ == '__main__':
    file = "/home/yuan/data/OPV2Vt/meta/2021_08_16_22_26_54.json"
    meta = load_json(file)
    for f, fdict in meta.items():
        ego_dict = fdict['agents'][fdict['meta']['ego_id']]
        ego_boxes = np.array(ego_dict['gt_boxes'])
        mask = np.linalg.norm(ego_boxes[:, 2:4], axis=-1) < 100
        ego_boxes = ego_boxes[mask]
        ego_pose = np.array(ego_dict['lidar']['0']['pose'])
        err = np.random.randn(3)
        ego_pose_e = ego_pose + np.array([err[0], err[1], 0, 0, 0, np.deg2rad(err[2]*10)])
        ego_boxes = transform_boxes_3d(ego_boxes, pose_to_transformation(ego_pose_e), mode=11)

        boxes_list = [ego_boxes]
        pose_list = [ego_pose_e[[0, 1, -1]]]
        for ai, adict in fdict['agents'].items():
            if ai == fdict['meta']['ego_id']:
                continue
            gt_boxes = np.array(adict['gt_boxes'])
            mask = np.linalg.norm(gt_boxes[:, 2:4], axis=-1) < 100
            gt_boxes = gt_boxes[mask]
            pose = np.array(adict['lidar']['0']['pose'])
            err = np.random.randn(3)
            pose_e = pose + np.array([err[0], err[1], 0, 0, 0, np.deg2rad(err[2])*10])
            gt_boxes = transform_boxes_3d(gt_boxes, pose_to_transformation(pose_e), mode=11)
            boxes_list.append(gt_boxes)
            pose_list.append(pose_e[[0, 1, -1]])

        pgo = PoseGraphOptimization2D()
        pgo.add_graph(boxes_list, pose_list)
