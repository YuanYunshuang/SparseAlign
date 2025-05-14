import os
import shutil

import numpy as np
import torch
import torch.nn.functional as F
try:
    import open3d as o3d
except:
    o3d = None
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as R

from cosense3d.utils.misc import check_numpy_to_torch
from cosense3d.utils.pcdio import point_cloud_from_path

ply_fields = {'x': 'f4', 'y': 'f4', 'z': 'f4', 'ObjIdx': 'u4', 'ObjTag': 'u4', 'ring': 'u1', 'time': 'f4'}
np_types = {'f4': np.float32, 'u4': np.uint32, 'u1': np.uint8}


def header(points):
    return f"""\
            VERSION 0.7
            FIELDS x y z rgb
            SIZE 4 4 4 4
            TYPE F F F F
            COUNT 1 1 1 1
            WIDTH {len(points)}
            HEIGHT 1
            VIEWPOINT 0 0 0 1 0 0 0
            POINTS {len(points)}
            DATA ascii
            """


def pose_to_transformation(pose):
    """

    :param pose: list, [x, y, z, roll, pitch, yaw]

    :return:
        transformation: np.ndarray, (4, 4)
    """
    transformation = np.eye(4)
    r = R.from_euler('xyz', pose[3:]).as_matrix()
    transformation[:3, :3] = r
    transformation[:3, 3] = np.array(pose[:3])
    return transformation


def read_ply(filename):
    ply = PlyData.read(filename)
    data = ply['vertex']
    properties = [prop.name for prop in data.properties]
    property_types = [prop.val_dtype for prop in data.properties]

    return {name: np.array(data[name]) for name in properties}, property_types


def save_cosense_ply(data, output_file_name):
    data = {
        'x': data['x'].astype(np_types[ply_fields['x']]),
        'y': data['y'].astype(np_types[ply_fields['y']]),
        'z': data['z'].astype(np_types[ply_fields['z']]),
        'ObjIdx': data['ObjIdx'].astype(np_types[ply_fields['ObjIdx']]),
        'ObjTag': data['ObjTag'].astype(np_types[ply_fields['ObjTag']]),
        'ring': data['ring'].astype(np_types[ply_fields['ring']]),
        'time': data['time'].astype(np_types[ply_fields['time']])
    }
    vertex_data = list(zip(*[data[k] for k, v in ply_fields.items()]))
    vertex_type = [(k, v) for k, v in ply_fields.items()]
    vertex = np.array(vertex_data, dtype=vertex_type)
    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el]).write(output_file_name)


def lidar_ply2bin(ply_file, bin_file,
                  fields=['x', 'y', 'z', 'intensity'],
                  replace=False):
    """
    Read ply and save to the cosense3d binary format.

    :param ply_file: str, input file name
    :param bin_file: str, output file name
    :param fields: list of str, names that indicates 'x', 'y', 'z' and 'intensity'
    :param replace: replace the exisiting file if True
    """
    if not replace and os.path.exists(bin_file):
        return
    pointcloud, property_types = read_ply(ply_file)
    pcd_out = np.stack([pointcloud[k] for k in fields], axis=1)
    pcd_out.tofile(bin_file)


def lidar_bin2pcd_o3d(bin_file, out_file, replace=False):
    if not replace and os.path.exists(out_file):
        return
    bin_pcd = np.fromfile(bin_file, dtype=np.float32)

    # reshape
    points = bin_pcd.reshape(-1, 4)
    # remove nan points
    mask = np.logical_not(np.isnan(points[:, :3]).any(axis=1))
    points = points[mask]

    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(points[:, :-1])

    point_intensity = np.zeros_like(points[:, :-1])
    point_intensity[:, 0] = points[:, -1] / 255.
    o3d_pcd.colors = o3d.utility.Vector3dVector(point_intensity)

    # write to pcd file
    o3d.io.write_point_cloud(out_file,
                             pointcloud=o3d_pcd,
                             write_ascii=True)


def lidar_bin2pcd(bin_file, out_file, replace=False):
    if not replace and os.path.exists(out_file):
        return
    bin_pcd = np.fromfile(bin_file, dtype=np.float32)
    # reshape
    points = bin_pcd.reshape(-1, 4)
    points[:, 3] /= 255
    mask = np.logical_not(np.isnan(points[:, :3]).any(axis=1))
    points = points[mask]
    header_str = header(points)
    with open(out_file, 'w') as fh:
        # fh.write()
        np.savetxt(fh, points, fmt='%f', header=header_str)
    # shutil.copy(out_file.replace('pcd', 'txt'), out_file)


def lidar_bin2bin(bin_file, out_file):
    shutil.copy(bin_file, out_file)


def load_pcd(pcd_file: str, return_o3d: bool=False):
    """
    Read  pcd and return numpy array.

    :param pcd_file: The pcd file that contains the point cloud.
    :param return_o3d: Default returns numpy array, set True to return pcd as o3d PointCloud object

    :return: lidar_dict,
        xyz: (pcd_np | pcd : np.ndarray | o3d.geometry.PointCloud) the lidar xyz coordinates in numpy format, shape:(n, 3);
        intensity: (optional) np.ndarray, (n,).
        label: (optional) np.ndarray, (n,).
        time: (optional) np.ndarray, (n,).
        ray: (optional) np.ndarray, (n,).
    """
    lidar_dict = {}
    ext = os.path.splitext(pcd_file)[-1]
    if ext == '.pcd':
        if return_o3d:
            return o3d.io.read_point_cloud(pcd_file)
        else:
            pcd = point_cloud_from_path(pcd_file)
            lidar_dict['xyz'] = np.stack([pcd.pc_data[x] for x in 'xyz'], axis=-1).astype(float)
            # we save the intensity in the first channel
            if 'intensity' in pcd.fields:
                lidar_dict['intensity'] = pcd.pc_data['intensity']
            if 'timestamp' in pcd.fields:
                lidar_dict['time'] = pcd.pc_data['timestamp']

    elif ext == '.bin':
        pcd_np = np.fromfile(pcd_file, dtype=np.float32).reshape(-1, 4)
        if return_o3d:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pcd_np)
            return pcd
        lidar_dict['xyz'] = pcd_np[:, :3]
        # check attribute of last column,
        # num of unique labels for the datasets in this projects is less than 50,
        # unique intensities is normally larger then 50
        if len(np.unique(pcd_np[:, -1])) < 50:
            lidar_dict['label'] = pcd_np[:, -1]
        elif pcd_np[:, -1].max() > 1:
            lidar_dict['intensity'] = pcd_np[:, -1] / 255
        else:
            lidar_dict['intensity'] = pcd_np[:, -1]

    elif ext == '.ply':
        data = read_ply(pcd_file)[0]
        xyz = np.stack([data.pop(x) for x in 'xyz'], axis=1)
        lidar_dict['xyz'] = xyz
        lidar_dict.update(data)
    else:
        raise NotImplementedError

    return lidar_dict


def tf2pose(tf_matrix):
    euler = R.from_matrix(tf_matrix[:3, :3]).as_euler('xyz')
    translation = tf_matrix[:3, 3]
    return translation.tolist() + euler.tolist()


def pose2tf(pose):
    tf_matrix = np.eye(4)
    tf_matrix[:3, :3] = rotation_matrix(pose[3:])
    tf_matrix[:3, 3] = np.array(pose[:3])
    return tf_matrix


def rotation_matrix(euler, degrees=True):
    """
    Construct rotation matrix with the given pose.

    :param euler: list or np.ndarray
        [roll, pitch, yaw]
    :return: rot: np.ndarray, 3x3
        rotation matrix
    """
    return R.from_euler('xyz', euler, degrees=degrees).as_matrix()


def rotate3d(points, euler):
    """
    Rotate point cloud with the euler angles given in pose.

    :param points: np.ndarray, N x (3 + C)
        each point in the row has the format [x, y, z, ...]
    :param euler: list or np.ndarray
        [roll, pitch, yaw]

    :return: points: np.ndarray
        rotated point cloud
    """
    assert len(euler) == 3
    rot = rotation_matrix(euler)
    points[:, :3] = (rot @ points[:, :3].T).T
    return points


def cart2cyl(input_xyz):
    rho = np.sqrt(input_xyz[..., 0] ** 2 + input_xyz[..., 1] ** 2)
    phi = np.arctan2(input_xyz[..., 1], input_xyz[..., 0])
    return np.concatenate((rho.reshape(-1, 1), phi.reshape(-1, 1), input_xyz[..., 2:]), axis=-1)


def cyl2cart(input_xyz_polar):
    x = input_xyz_polar[..., 0] * np.cos(input_xyz_polar[..., 1])
    y = input_xyz_polar[..., 0] * np.sin(input_xyz_polar[..., 1])
    return np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), input_xyz_polar[..., 2:]), axis=-1)


def mat_yaw(cosa, sina, zeros=0, ones=1):
    return [
        cosa, -sina, zeros,
        sina, cosa, zeros,
        zeros, zeros, ones
    ]


def mat_pitch(cosa, sina, zeros=0, ones=1):
    return [
        cosa, zeros, sina,
        zeros, ones, zeros,
        -sina, zeros, cosa,
    ]


def mat_roll(cosa, sina, zeros=0, ones=1):
    return [
        ones, zeros, zeros,
        zeros, cosa, -sina,
        zeros, sina, cosa,
    ]


def rotate_points_along_z_np(points, angle):
    """
    :param points: (N, 3 + C or 2 + C)
    :param angle: float, angle along z-axis, angle increases x ==> y

    """
    cosa = np.cos(angle)
    sina = np.sin(angle)
    rot_matrix = np.array([
        [cosa,  sina, 0],
        [-sina, cosa, 0],
        [0, 0, 1]
    ]).astype(np.float)
    if points.shape[1]==2:
        points_rot = np.matmul(points, rot_matrix[:2, :2])
    elif points.shape[1]>2:
        points_rot = np.matmul(points[:, 0:3], rot_matrix)
        points_rot = np.concatenate((points_rot, points[:, 3:]), axis=-1)
    else:
        raise IOError('Input points should have the shape: (N, 3 + C or 2 + C).')
    return points_rot


def rotate_points_batch(points, angles, order='xyz'):
    """
    :param points: (B, N, 3 + C)
    :param angles: (B, 1|3), radians
        rotation = R(3)R(2)R(1) if angles shape in (B, 3)
    :return: points_rot: (B, N, 3 + C)
    """
    assert angles.shape[1] == len(order), \
        "angles should has the shape (len(points), len(order))."

    points, is_numpy = check_numpy_to_torch(points)
    angles, _ = check_numpy_to_torch(angles)

    cosas = torch.cos(angles)
    sinas = torch.sin(angles)
    zeros = angles[:, 0].new_zeros(points.shape[0])
    ones = angles[:, 0].new_ones(points.shape[0])
    rot_matrix = torch.eye(3, dtype=points.dtype, device=points.device)
    rot_matrix = rot_matrix.reshape((1, 3, 3)).repeat(angles.shape[0], 1, 1)
    for cosa, sina, ax in zip(cosas.T, sinas.T, order):
        if ax == 'z':
            rot = torch.stack(mat_yaw(
                cosa, sina, zeros, ones
            ), dim=1).view(-1, 3, 3).float()
        elif ax == 'y':
            rot = torch.stack(mat_pitch(
                cosa, sina, zeros, ones
            ), dim=1).view(-1, 3, 3).float()
        elif ax == 'x':
            rot = torch.stack(mat_roll(
                cosa, sina, zeros, ones
            ), dim=1).view(-1, 3, 3).float()
        else:
            raise NotImplementedError
        rot_matrix = torch.bmm(rot, rot_matrix)
    points_rot = torch.bmm(rot_matrix, points[:, :, 0:3].float().
                           permute(0, 2, 1)).permute(0, 2, 1)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot


def rotate_points_along_z_torch(points, angle):
    """
    :param points: (N, 2 + C) or (B, 2 + C)
    :param angle: float or tensor of shape (B), angle along z-axis, angle increases x ==> y

    """
    if len(points.shape) == 2:
        points = points.unsqueeze(0)
    if isinstance(angle, float):
        angle = torch.tensor([angle], device=points.device)
    else:
        assert isinstance(angle, torch.Tensor)
        assert points.shape[0] == 1 or angle.shape[0] == points.shape[0]
    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    rot_matrix = torch.stack([
        torch.stack([cosa,  sina], dim=-1),
        torch.stack([-sina, cosa], dim=-1)
    ], dim=1).float().to(points.device)
    if points.shape[0] == 1 and angle.shape[0] > 1:
        points = torch.tile(points, (len(rot_matrix), 1, 1))
    points_rot = torch.bmm(points[..., 0:2], rot_matrix)
    points_rot = torch.cat((points_rot, points[..., 2:]), dim=-1)
    return points_rot


def rotate_points_with_tf_np(points: np.ndarray, tf_np: np.ndarray) -> np.ndarray:
    """
    Rotate points with transformation matrix.

    :param points (np.ndarray): Nx3 points array
    :param tf_np (np.ndarray): 4x4 transformation matrix
    :return: points (np.ndarray): Nx3 points array
    """
    points_homo = np.concatenate([points, np.ones_like(points[:, :1])], axis=-1).T
    points = (tf_np @ points_homo)[:3].T
    return points


def rotate_box_corners_with_tf_np(corners: np.ndarray, tf_np: np.ndarray) -> np.ndarray:
    """
    Rotate points with transformation matrix
    :param corners: Nx8X3 points array
    :param tf_np: 4x4 transformation matrix
    :return: corners, Nx8X3 points array
    """
    points = rotate_points_with_tf_np(corners.reshape(-1, 3), tf_np)
    corners = points.reshape(corners.shape)
    return corners


def mask_values_in_range(values, min,  max):
    return np.logical_and(values>min, values<max)


def mask_points_in_box(points, pc_range):
    n_ranges = len(pc_range) // 2
    list_mask = [mask_values_in_range(points[:, i], pc_range[i],
                                       pc_range[i+n_ranges]) for i in range(n_ranges)]
    return np.array(list_mask).all(axis=0)


def mask_points_in_range(points: np.array, dist: float) -> np.array:
    """
    :rtype: np.array
    """
    return np.linalg.norm(points[:, :2], axis=1) < dist


def get_tf_matrix_torch(vectors, inv=False):
    device = vectors.device
    n, _ = vectors.shape
    xs = vectors[:, 0]
    ys = vectors[:, 1]
    angles = vectors[:, 2]
    cosa = torch.cos(angles)
    sina = torch.sin(angles)
    ones = torch.ones_like(angles)
    zeros = torch.zeros_like(angles)
    rot_matrix = torch.zeros((n, 3, 3), device=device, requires_grad=True)
    rot_matrix[:, 0, 0] = cosa
    rot_matrix[:, 0, 1] = -sina
    rot_matrix[:, 1, 0] = sina
    rot_matrix[:, 1, 1] = cosa
    shift_matrix = torch.zeros_like(rot_matrix, requires_grad=True)
    shift_matrix[:, 0, 1] = xs
    shift_matrix[:, 1, 0] = ys
    shift_matrix[:, [0, 1, 2], [0, 1, 2]] = 1.0
    if inv:
        mat = torch.einsum('...ij, ...jk', rot_matrix, shift_matrix)
    else:
        mat = torch.einsum('...ij, ...jk', shift_matrix, rot_matrix)
    return mat, rot_matrix, shift_matrix


def rotation_mat2euler_torch(mat):
    sy = torch.norm(mat[:, :2, 0], dim=1)
    singular = sy < 1e-6
    not_singular = torch.logical_not(singular)
    euler = torch.zeros_like(mat[:, 0])

    if not_singular.sum() > 0:
        euler[not_singular, 0] = torch.atan2(mat[not_singular, 2, 1], mat[not_singular, 2, 2])
        euler[not_singular, 1] = torch.atan2(-mat[not_singular, 2, 0], sy)
        euler[not_singular, 2] = torch.atan2(mat[not_singular, 1, 0], mat[not_singular, 0, 0])
    if singular.sum() > 0:
        euler[singular, 0] = torch.atan2(-mat[singular, 1, 2], mat[singular, 1, 1])
        euler[singular, 1] = torch.atan2(-mat[singular, 2, 0], sy)

    return euler


def pose_err_global2relative_torch(poses, errs):
    """
    Calculate relative pose transformation based on the errorneous global positioning
    :param poses: Nx2 or Nx3, first row is ego pose, other rows are the coop poses
    :param errs: Nx3, first row is ego pose error and other rows for coop pose errors
    :return: (N-1)x3, relative localization errors between ego and coop vehicles
    """
    if poses.shape[-1]==2:
        poses = torch.cat([poses, torch.zeros_like(poses[:, 0:1])], dim=-1)
    poses_err = poses + errs

    R01, _, _ = get_tf_matrix_torch(-poses[:1], inv=True)
    R10_hat, _, _ = get_tf_matrix_torch(poses_err[:1])
    R20, _, _ = get_tf_matrix_torch(poses[1:])
    R02_hat, _, _ = get_tf_matrix_torch(-poses_err[1:], inv=True)

    delta_R21 = torch.einsum('...ij, ...jk', R01, R20)
    delta_R21 = torch.einsum('...ij, ...jk', delta_R21, R02_hat)
    delta_R21 = torch.einsum('...ij, ...jk', delta_R21, R10_hat)

    x = delta_R21[0, 2]
    y = delta_R21[1, 2]
    theta = torch.atan2(delta_R21[1, 0], delta_R21[0, 0])
    return torch.stack([x, y, theta], dim=-1)


def project_points_by_matrix_torch(points, transformation_matrix):
    """
    Project the points to another coordinate system based on the
    transformation matrix.

    :param points: torch.Tensor, 3D points, (N, 3)
    :param transformation_matrix: torch.Tensor, Transformation matrix, (4, 4)
    :return: projected_points : torch.Tensor, The projected points, (N, 3)
    """
    points, is_numpy = \
        check_numpy_to_torch(points)
    transformation_matrix, _ = \
        check_numpy_to_torch(transformation_matrix)

    # convert to homogeneous coordinates via padding 1 at the last dimension.
    # (N, 4)
    points_homogeneous = F.pad(points, (0, 1), mode="constant", value=1)
    # (N, 4)
    projected_points = torch.einsum("ik, jk->ij", points_homogeneous,
                                    transformation_matrix)

    return projected_points[:, :3] if not is_numpy \
        else projected_points[:, :3].numpy()

if __name__=="__main__":
    for i in range(0, 300):
        frame = f"{i:06d}"
        ply_file = f"/koko/LUMPI/train/measurement5/lidar/{frame}.ply"
        bin_file = f"/media/hdd/projects/TAL/data/lumpi_m5/lidar0/{frame}.bin"
        lidar_ply2bin(ply_file, bin_file)
