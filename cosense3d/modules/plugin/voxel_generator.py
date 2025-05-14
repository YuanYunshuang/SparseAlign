import torch
from spconv.pytorch.utils import PointToVoxel


class VoxelGenerator:
    def __init__(self,
                 voxel_size,
                 lidar_range,
                 max_points_per_voxel,
                 empty_mean=True,
                 mode='train',
                 device='cuda',
                 **kwargs):
        self.voxel_size = torch.tensor(voxel_size)
        self.lidar_range = torch.tensor(lidar_range)
        self.max_points_per_voxel = max_points_per_voxel
        self.max_voxels = kwargs.get(f"max_voxels_{mode}", 50000)
        self.empty_mean = empty_mean

        self.grid_size = ((self.lidar_range[3:] - self.lidar_range[:3])
                          / self.voxel_size).round().int()
        self.voxel_generator = PointToVoxel(
            vsize_xyz=self.voxel_size.tolist(),
            coors_range_xyz=self.lidar_range.tolist(),
            max_num_points_per_voxel=self.max_points_per_voxel,
            num_point_features=4,
            max_num_voxels=self.max_voxels,
            device=torch.device(device)
        )

    def __call__(self, points_list):
        voxels_list = []
        coordinates_list = []
        num_points_list = []
        for points in points_list:
            voxels, coordinates, num_points = self.voxel_generator(
                points, empty_mean=self.empty_mean)
            voxels_list.append(voxels)
            coordinates_list.append(coordinates)
            num_points_list.append(num_points)
        return voxels_list, coordinates_list, num_points_list