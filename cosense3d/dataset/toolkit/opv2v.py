import copy
import json
import math
import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
import open3d as o3d
import os.path as osp

import cv2
from collections import OrderedDict
from torch.utils.data import Dataset


from scipy.spatial.transform import Rotation as R
from cosense3d.utils.misc import load_yaml, save_json, load_json
from cosense3d.dataset.toolkit import register_pcds
from cosense3d.dataset.toolkit.cosense import CoSenseDataConverter as cs
from cosense3d.utils.box_utils import boxes_to_corners_3d
from cosense3d.utils.pclib import load_pcd
from cosense3d.utils.vislib import draw_points_boxes_plt, draw_2d_bboxes_on_img
from cosense3d.ops.utils import points_in_boxes_cpu


def x_to_world(pose: list) -> np.ndarray:
    """
    The transformation matrix from x-coordinate system to carla world system
    Parameters

    :param pose: [x, y, z, roll, yaw, pitch]
    :return: The transformation matrix.
    """
    x, y, z, roll, yaw, pitch = pose[:]

    # used for rotation matrix
    c_y = np.cos(np.radians(yaw))
    s_y = np.sin(np.radians(yaw))
    c_r = np.cos(np.radians(roll))
    s_r = np.sin(np.radians(roll))
    c_p = np.cos(np.radians(pitch))
    s_p = np.sin(np.radians(pitch))

    matrix = np.identity(4)
    # translation matrix
    matrix[0, 3] = x
    matrix[1, 3] = y
    matrix[2, 3] = z

    # rotation matrix
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r

    return matrix


def x1_to_x2(x1, x2):
    """
    Transformation matrix from x1 to x2.

    Parameters
    ----------
    x1 : list or np.ndarray
        The pose of x1 under world coordinates or
        transformation matrix x1->world
    x2 : list or np.ndarray
        The pose of x2 under world coordinates or
         transformation matrix x2->world

    Returns
    -------
    transformation_matrix : np.ndarray
        The transformation matrix.

    """
    if isinstance(x1, list) and isinstance(x2, list):
        x1_to_world = x_to_world(x1)
        x2_to_world = x_to_world(x2)
        world_to_x2 = np.linalg.inv(x2_to_world)
        transformation_matrix = np.dot(world_to_x2, x1_to_world)

    # object pose is list while lidar pose is transformation matrix
    elif isinstance(x1, list) and not isinstance(x2, list):
        x1_to_world = x_to_world(x1)
        world_to_x2 = x2
        transformation_matrix = np.dot(world_to_x2, x1_to_world)
    # both are numpy matrix
    else:
        world_to_x2 = np.linalg.inv(x2)
        transformation_matrix = np.dot(world_to_x2, x1)

    return transformation_matrix


def create_bbx(extent):
    """
    Create bounding box with 8 corners under obstacle vehicle reference.

    Parameters
    ----------
    extent : list
        Width, height, length of the bbx.

    Returns
    -------
    bbx : np.array
        The bounding box with 8 corners, shape: (8, 3)
    """

    bbx = np.array([[extent[0], -extent[1], -extent[2]],
                    [extent[0], extent[1], -extent[2]],
                    [-extent[0], extent[1], -extent[2]],
                    [-extent[0], -extent[1], -extent[2]],
                    [extent[0], -extent[1], extent[2]],
                    [extent[0], extent[1], extent[2]],
                    [-extent[0], extent[1], extent[2]],
                    [-extent[0], -extent[1], extent[2]]])

    return bbx


def corner_to_center(corner3d, order='lwh'):
    """
    Convert 8 corners to x, y, z, dx, dy, dz, yaw.

    Parameters
    ----------
    corner3d : np.ndarray
        (N, 8, 3)

    order : str
        'lwh' or 'hwl'

    Returns
    -------
    box3d : np.ndarray
        (N, 7)
    """
    assert corner3d.ndim == 3
    batch_size = corner3d.shape[0]

    xyz = np.mean(corner3d[:, [0, 3, 5, 6], :], axis=1)
    h = abs(np.mean(corner3d[:, 4:, 2] - corner3d[:, :4, 2], axis=1,
                    keepdims=True))
    l = (np.sqrt(np.sum((corner3d[:, 0, [0, 1]] - corner3d[:, 3, [0, 1]]) ** 2,
                        axis=1, keepdims=True)) +
         np.sqrt(np.sum((corner3d[:, 2, [0, 1]] - corner3d[:, 1, [0, 1]]) ** 2,
                        axis=1, keepdims=True)) +
         np.sqrt(np.sum((corner3d[:, 4, [0, 1]] - corner3d[:, 7, [0, 1]]) ** 2,
                        axis=1, keepdims=True)) +
         np.sqrt(np.sum((corner3d[:, 5, [0, 1]] - corner3d[:, 6, [0, 1]]) ** 2,
                        axis=1, keepdims=True))) / 4

    w = (np.sqrt(
        np.sum((corner3d[:, 0, [0, 1]] - corner3d[:, 1, [0, 1]]) ** 2, axis=1,
               keepdims=True)) +
         np.sqrt(np.sum((corner3d[:, 2, [0, 1]] - corner3d[:, 3, [0, 1]]) ** 2,
                        axis=1, keepdims=True)) +
         np.sqrt(np.sum((corner3d[:, 4, [0, 1]] - corner3d[:, 5, [0, 1]]) ** 2,
                        axis=1, keepdims=True)) +
         np.sqrt(np.sum((corner3d[:, 6, [0, 1]] - corner3d[:, 7, [0, 1]]) ** 2,
                        axis=1, keepdims=True))) / 4

    theta = (np.arctan2(corner3d[:, 1, 1] - corner3d[:, 2, 1],
                        corner3d[:, 1, 0] - corner3d[:, 2, 0]) +
             np.arctan2(corner3d[:, 0, 1] - corner3d[:, 3, 1],
                        corner3d[:, 0, 0] - corner3d[:, 3, 0]) +
             np.arctan2(corner3d[:, 5, 1] - corner3d[:, 6, 1],
                        corner3d[:, 5, 0] - corner3d[:, 6, 0]) +
             np.arctan2(corner3d[:, 4, 1] - corner3d[:, 7, 1],
                        corner3d[:, 4, 0] - corner3d[:, 7, 0]))[:,
            np.newaxis] / 4

    if order == 'lwh':
        return np.concatenate([xyz, l, w, h, theta], axis=1).reshape(
            batch_size, 7)
    elif order == 'hwl':
        return np.concatenate([xyz, h, w, l, theta], axis=1).reshape(
            batch_size, 7)
    else:
        raise NotImplementedError


def project_world_objects(object_dict,
                          output_dict,
                          lidar_pose,
                          order):
    """
    Project the objects under world coordinates into another coordinate
    based on the provided extrinsic.

    Parameters
    ----------
    object_dict : dict
        The dictionary contains all objects surrounding a certain cav.

    output_dict : dict
        key: object id, value: object bbx (xyzlwhyaw).

    lidar_pose : list
        (6, ), lidar pose under world coordinate, [x, y, z, roll, yaw, pitch].

    order : str
        'lwh' or 'hwl'
    """
    for object_id, object_content in object_dict.items():
        location = object_content['location']
        rotation = object_content['angle']
        center = object_content['center']
        extent = object_content['extent']

        if 'ass_id' not in object_content or object_content['ass_id'] == -1:
            ass_id = object_id
        else:
            ass_id = object_content['ass_id']
        if 'obj_type' not in object_content:
            obj_type = 'Car'
        else:
            obj_type = object_content['obj_type']

        # todo: pedestrain is not consdered yet
        # todo: only single class now
        if obj_type == 'Pedestrian':
            continue

        object_pose = [location[0] + center[0],
                       location[1] + center[1],
                       location[2] + center[2],
                       rotation[0], rotation[1], rotation[2]]
        object2lidar = x1_to_x2(object_pose, lidar_pose)

        # shape (3, 8)
        bbx = create_bbx(extent).T
        # bounding box under ego coordinate shape (4, 8)
        bbx = np.r_[bbx, [np.ones(bbx.shape[1])]]

        # project the 8 corners to lidar coordinate
        bbx_lidar = np.dot(object2lidar, bbx).T
        bbx_lidar = np.expand_dims(bbx_lidar[:, :3], 0)
        bbx_lidar = corner_to_center(bbx_lidar, order=order)

        # get velocity
        if 'speed' in object_content:
            speed = object_content['speed']
            theta = bbx_lidar[0, -1]
            velo = np.array([speed * np.cos(theta), speed * np.sin(theta)])
        else:
            velo = None

        if bbx_lidar.shape[0] > 0:
            output_dict.update({object_id: {'coord': bbx_lidar,
                                            'ass_id': ass_id,
                                            'velo': velo}})


def update_local_boxes3d(fdict, objects_dict, ref_pose, order, data_dir, cav_id):
    output_dict = {}
    # add ground truth boxes at cav local coordinate
    project_world_objects(objects_dict,
                          output_dict,
                          ref_pose,
                          order)
    boxes_local = []
    velos = []
    for object_id, object_content in output_dict.items():
        if object_content['ass_id'] != -1:
            object_id = object_content['ass_id']
        else:
            object_id = object_id
        object_bbx = object_content['coord']
        if order == 'hwl':
            object_bbx = object_bbx[:, [0, 1, 2, 5, 4, 3, 6]]
        boxes_local.append(
            [object_id, 0, ] +
            object_bbx[0, :6].tolist() +
            [0, 0, object_bbx[0, 6]]
        )
        if 'velo' in object_content and object_content['velo'] is not None:
            velos.append(object_content['velo'].tolist())

    cs.update_agent(fdict, cav_id, gt_boxes=boxes_local)
    if len(velos) == len(boxes_local):
        cs.update_agent(fdict, cav_id, velos=velos)

    # get visibility of local boxes
    lidar = load_pcd(os.path.join(data_dir, fdict['agents'][cav_id]['lidar']['0']['filename']))['xyz']
    if len(boxes_local) > 0:
        boxes = np.array(boxes_local)[:, [2, 3, 4, 5, 6, 7, 10]]
        res = points_in_boxes_cpu(lidar, boxes)
        num_pts = res.sum(axis=1)
        cs.update_agent(fdict, cav_id, num_pts=num_pts.tolist())
    else:
        cs.update_agent(fdict, cav_id, num_pts=[])


def opv2v_pose_to_cosense(pose):
    if len(pose) == 6:
        transformation = x_to_world(pose)
    else:
        transformation = pose
    rot = R.from_matrix(transformation[:3, :3]).as_euler('xyz', degrees=False)
    tl = transformation[:3, 3]
    pose = tl.tolist() + rot.tolist()
    return pose


def update_cam_params(opv2v_params, cosense_fdict, agent_id, scenario, frame):
    for k, v in opv2v_params.items():
        if 'camera' in k:
            cam_id = int(k[-1:])
            cs.add_cam_to_fdict(
                cosense_fdict,
                agent_id,
                cam_id,
                [os.path.join(scenario, agent_id, f'{frame}_{k}.png')],
                v['intrinsic'],
                v['extrinsic'],
                pose=v['cords'],
            )
            

def project_points(points, lidar2cam, I):
    """Project 3d points to image planes"""
    points_homo = np.concatenate([points[:, :3], np.ones_like(points[:, :1])], axis=1).T
    points_homo = lidar2cam @ points_homo
    pixels = I @ points_homo[:3]
    pixels[:2] = pixels[:2] / pixels[2:]
    depths = points_homo[2]
    return pixels, depths


def boxes_3d_to_2d(boxes3d, num_pts, lidar2cam, I, img_size):
    n_box = len(boxes3d)
    box_center = boxes3d.mean(axis=1)
    box_points = boxes3d.reshape(-1, 3)
    
    box_pixels, _ = project_points(box_points, lidar2cam, I)
    center_pixels, depths = project_points(box_center, lidar2cam, I)

    box_pixels = box_pixels.T.reshape(n_box, 8, 3)
    mask = (box_pixels[:, :, 2] > 0).all(axis=1)
    box_pixels = box_pixels[mask]
    center_pixels = center_pixels[:2].T[mask]
    depths = depths[mask]
    num_pts = num_pts[mask]
    x_min = np.clip(box_pixels[..., 0].min(axis=1), a_min=0, a_max=img_size[1])
    y_min = np.clip(box_pixels[..., 1].min(axis=1), a_min=0, a_max=img_size[0])
    x_max = np.clip(box_pixels[..., 0].max(axis=1), a_min=0, a_max=img_size[1])
    y_max = np.clip(box_pixels[..., 1].max(axis=1), a_min=0, a_max=img_size[0])
    mask = (x_min < img_size[1]) & (x_max > 0) & (y_min < img_size[0]) & (y_max > 0)
    bbox_2d = np.stack([x_min[mask], y_min[mask], x_max[mask], y_max[mask]], axis=-1)
    return bbox_2d, center_pixels[mask], depths[mask], num_pts[mask]


def update_2d_bboxes(fdict, cav_id, lidar_pose, data_dir):
    local_boxes = np.array(fdict['agents'][cav_id]['gt_boxes'])
    if len(local_boxes) > 0:
        local_boxes = local_boxes[:, 2:]
        num_pts = np.array(fdict['agents'][cav_id]['num_pts'])
        boxes_corners = boxes_to_corners_3d(local_boxes)
        # lidar = load_pcd(os.path.join(data_dir, fdict['agents'][cav_id]['lidar'][0]['filename']))
        # lidar = np.concatenate([lidar['xyz'], np.ones_like(lidar['intensity'])], axis=1)
        # draw_points_boxes_plt(pc_range=100, points=lidar, filename="/home/yuan/Downloads/tmp.png")
        cam_UE2pinhole = np.array([[0, 1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
        for cam_id, cam_params in fdict['agents'][cav_id]['camera'].items():
            img = cv2.imread(os.path.join(data_dir, cam_params['filenames'][0]))[..., ::-1]
            lidar2cam_UE = x1_to_x2(lidar_pose, cam_params['pose'])
            lidar2cam_pinhole = cam_UE2pinhole @ lidar2cam_UE
            I = np.array(cam_params['intrinsic'])
            # draw_3d_points_boxes_on_img(img, lidar2cam_pinhole, I, lidar, boxes_corners)
            bboxes2d, centers2d, depths, num_pts_2d = boxes_3d_to_2d(
                boxes_corners, num_pts, lidar2cam_pinhole, I, img_size=img.shape)
            # draw_2d_bboxes_on_img(img, bboxes2d)
            cam_params['bboxes2d'] = bboxes2d.tolist()
            cam_params['centers2d'] = centers2d.tolist()
            cam_params['depths'] = depths.tolist()
            cam_params['num_pts'] = num_pts_2d.tolist()
            cam_params['lidar2cam'] = lidar2cam_pinhole.tolist()
    else:
        cam_UE2pinhole = np.array([[0, 1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
        for cam_id, cam_params in fdict['agents'][cav_id]['camera'].items():
            lidar2cam_UE = x1_to_x2(lidar_pose, cam_params['pose'])
            lidar2cam_pinhole = cam_UE2pinhole @ lidar2cam_UE
            cam_params['lidar2cam'] = lidar2cam_pinhole.tolist()
            cam_params['bboxes2d'] = []
            cam_params['centers2d'] = []
            cam_params['depths'] = []
            cam_params['num_pts'] = []


def opv2v_to_cosense(path_in, path_out, isSim=True, correct_transf=False, pcd_ext='pcd'):
    if isSim:
        order = 'lwh'
    else:
        order = 'hwl'
    flag = False
    for split in ['train', 'test']:
        scenarios = sorted(os.listdir(os.path.join(path_in, split)))
        with open(os.path.join(path_out, f'{split}.txt'), 'w') as fh:
            fh.write('\n'.join(scenarios))
        for s in scenarios:
            print(s)
            # if s == "2021_08_22_09_43_53":
            #     flag = True
            # if not flag:
            #     continue
            visualize = False
            sdict = {}
            spath = os.path.join(path_in, split, s)
            cavs = sorted([x for x in os.listdir(spath) if os.path.isdir(os.path.join(spath, x))])
            ego_id = cavs[0]
            frames = sorted([x[:-5]
                            for x in os.listdir(os.path.join(spath, ego_id)) if
                            x.endswith('.yaml') and 'sparse_gt' not in x])
            for f in tqdm.tqdm(frames):
                fdict = cs.fdict_template()
                ego_lidar_pose = None
                object_id_stack = []
                object_velo_stack = []
                object_stack = []
                for i, cav_id in enumerate(cavs):
                    yaml_file = os.path.join(spath, cav_id, f'{f}.yaml')
                    params = load_yaml(yaml_file)
                    cs.update_agent(fdict, cav_id, agent_type='cav',
                                    agent_pose=opv2v_pose_to_cosense(params['true_ego_pos']))
                    update_cam_params(params, fdict, cav_id, s, f)
                    
                    if cav_id == ego_id:
                        ego_lidar_pose = params['lidar_pose']

                    # get transformation from ego to cav, correct transformation if necessary
                    transformation = x1_to_x2(params['lidar_pose'], ego_lidar_pose)
                    if not isSim and correct_transf and cav_id != ego_id:
                        ego_lidar_file = os.path.join(path_in, split, s, ego_id, f'{f}.pcd')
                        cav_lidar_file = os.path.join(path_in, split, s, cav_id, f'{f}.pcd')
                        transformation = register_pcds(cav_lidar_file, ego_lidar_file, transformation, visualize)
                        visualize = False
                    # cav_lidar_pose2ego = opv2v_pose_to_cosense(transformation)

                    # get cav lidar pose in cosense format
                    cs.update_agent(fdict, cav_id, 'cav')
                    cs.update_agent_lidar(fdict, cav_id, '0',
                                          lidar_pose=opv2v_pose_to_cosense(params['lidar_pose']),
                                          lidar_file=os.path.join(s, cav_id, f'{f}.{pcd_ext}'))

                    objects_dict = params['vehicles']
                    output_dict = {}
                    if isSim:
                        glob_ref_pose = ego_lidar_pose
                        local_ref_pose = params['lidar_pose']
                    else:
                        glob_ref_pose = transformation
                        local_ref_pose = [0,] * 6

                    data_dir = os.path.join(path_in, split)
                    update_local_boxes3d(fdict, objects_dict, local_ref_pose, order, data_dir, cav_id)
                    if isSim:
                        # v2vreal has no camera data
                        update_2d_bboxes(fdict, cav_id, params['lidar_pose'], data_dir)

                    # add gt boxes in ego coordinates as global boxes of cosense3d format
                    project_world_objects(objects_dict,
                                          output_dict,
                                          glob_ref_pose,
                                          order)

                    for object_id, object_content in output_dict.items():
                        if object_content['ass_id'] != -1:
                            object_id_stack.append(object_content['ass_id'])
                        else:
                            object_id_stack.append(object_id + 100 * int(cav_id))
                        if object_content['velo'] is not None:
                            object_velo_stack.append(object_content['velo'])
                        object_stack.append(object_content['coord'])

                # exclude all repetitive objects
                unique_indices = \
                    [object_id_stack.index(x) for x in set(object_id_stack)]
                object_stack = np.vstack(object_stack)
                object_stack = object_stack[unique_indices]
                if len(object_velo_stack) == len(object_stack):
                    object_velo_stack = np.vstack(object_velo_stack)
                    object_velo_stack = object_velo_stack[unique_indices]
                if order == 'hwl':
                    object_stack = object_stack[:, [0, 1, 2, 5, 4, 3, 6]]

                cosense_bbx_center = np.zeros((len(object_stack), 11))
                cosense_bbx_center[:, 0] = np.array(object_id_stack)[unique_indices]
                cosense_bbx_center[:, 2:8] = object_stack[:, :6]
                cosense_bbx_center[:, 10] = object_stack[:, 6]
                cs.update_frame_bbx(fdict, cosense_bbx_center.tolist())
                if '0' not in cavs:
                    fdict['agents'].pop('0')  # remove template agent

                fdict['meta']['ego_id'] = ego_id
                fdict['meta']['ego_lidar_pose'] = opv2v_pose_to_cosense(ego_lidar_pose)
                if len(object_velo_stack) == len(object_stack):
                    fdict['meta']['bbx_velo_global'] = object_velo_stack.tolist()

                boxes_num_pts = {int(i): 0 for i in cosense_bbx_center[:, 0]}
                for adict in fdict['agents'].values():
                    for box, num_pts in zip(adict['gt_boxes'], adict['num_pts']):
                        boxes_num_pts[int(box[0])] += num_pts
                fdict['meta']['num_pts'] = [boxes_num_pts[int(i)] for i in cosense_bbx_center[:, 0]]

                sdict[f] = fdict

                # plot
                # ego_pose = pose_to_transformation(fdict['meta']['ego_lidar_pose'])
                # ax = None
                # for ai, adict in fdict['agents'].items():
                #     cav_pose = pose_to_transformation(adict['lidar'][0]['pose'])
                #     T_cav2ego = np.linalg.inv(ego_pose) @ cav_pose
                #     lidar_file = os.path.join(path_in, split, adict['lidar'][0]['filename'])
                #     points = load_pcd(lidar_file)['xyz']
                #     points = np.concatenate([points, np.ones_like(points[:, :1])], axis=-1)
                #     points = (T_cav2ego @ points.T).T
                #     color = 'g' if ai == ego_id else 'r'
                #     ax = draw_points_boxes_plt(
                #         pc_range=100,
                #         points=points[:, :3],
                #         points_c=color,
                #         ax=ax,
                #         return_ax=True
                #     )
                # plt.show()
                # plt.close()
                # pass
            save_json(sdict, os.path.join(path_out, f'{s}.json'))


def pose_to_transformation(pose):
    """

    Args:
        pose: list, [x, y, z, roll, pitch, yaw]

    Returns:
        transformation: np.ndarray, (4, 4)
    """
    transformation = np.eye(4)
    r = R.from_euler('xyz', pose[3:]).as_matrix()
    transformation[:3, :3] = r
    transformation[:3, 3] = np.array(pose[:3])
    return transformation


def update_global_bboxes_num_pts(data_dir, meta_path):
    json_files = glob(meta_path + '/*.json')
    for jf in tqdm.tqdm(json_files):
        # tmp = os.path.join(data_dir, 'train', os.path.basename(jf)[:-5])
        # data_dir_split = os.path.join(data_dir, 'train') if os.path.exists(tmp) else os.path.join(data_dir, 'test')
        with open(jf, 'r') as fh:
            meta = json.load(fh)
        for f, fdict in meta.items():
            # lidar_files = [ldict['filename'] for adict in fdict['agents'].values() for ldict in adict['lidar'].values()]
            # lidar_files = [os.path.join(data_dir_split, lf) for lf in lidar_files]
            # pcds = [load_pcd(lf)['xyz'] for lf in lidar_files]
            # pcds = np.concatenate(pcds, axis=0)
            boxes = np.array(fdict['meta']['bbx_center_global'])
            boxes_num_pts = {int(i): 0 for i in boxes[:, 0]}
            for adict in fdict['agents'].values():
                for box, num_pts in zip(adict['gt_boxes'], adict['num_pts']):
                    boxes_num_pts[int(box[0])] += num_pts
            fdict['meta']['num_pts'] = [boxes_num_pts[int(i)] for i in boxes[:, 0]]

        save_json(meta, jf.replace('opv2v', 'opv2v_full_'))


def generate_bevmaps(data_dir, meta_path):
    assets_path = f"{os.path.dirname(__file__)}/../../carla/assets"
    map_path = f"{assets_path}/maps"
    map_files = glob(os.path.join(map_path, '*.png'))
    scene_maps = load_json(os.path.join(assets_path, 'scenario_town_map.json'))
    map_bounds = load_json(os.path.join(assets_path, 'map_bounds.json'))
    bevmaps = {}
    for mf in map_files:
        town = os.path.basename(mf).split('.')[0]
        bevmap = cv2.imread(mf)
        # bevmap = np.pad(bevmap, ((pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=0)
        bevmaps[town] = bevmap

    T_corr = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 0, 1]])

    json_files = glob(meta_path + '/*.json')
    grid = np.ones((500, 500))
    inds = np.stack(np.where(grid))
    xy = inds * 0.2 - 50 + 0.1
    xy_pad = np.concatenate([xy, np.zeros_like(xy[:1]), np.ones_like(xy[:1])], axis=0)
    for jf in tqdm.tqdm(json_files):
        scene = os.path.basename(jf).split('.')[0]
        town = scene_maps[scene]
        cur_map = bevmaps[town]
        sx, sy = cur_map.shape[:2]
        meta = load_json(jf)
        for f, fdict in meta.items():
            for ai, adict in fdict['agents'].items():
                lidar_pose = adict['lidar']['0']['pose']
                transform = T_corr @ pose_to_transformation(lidar_pose)
                xy_tf = transform @ xy_pad
                # xy_tf = xy_pad
                # xy_tf[0] = xy_tf[0] - lidar_pose[0]
                # xy_tf[1] = xy_tf[1] - lidar_pose[1]
                xy_tf[0] -= map_bounds[town][0]
                xy_tf[1] -= map_bounds[town][1]
                map_inds = np.floor(xy_tf[:2] / 0.2)
                xs = np.clip(map_inds[0], 0, sx - 1).astype(int)
                ys = np.clip(map_inds[1], 0, sy - 1).astype(int)
                bevmap = cur_map[xs, ys].reshape(500, 500, 3)[::-1, ::-1]

                filename = os.path.join(data_dir, 'train', scene, ai, f'{f}_bev.png')
                if not os.path.exists(filename):
                    filename = os.path.join(data_dir, 'test', scene, ai, f'{f}_bev.png')
                gt_bev = cv2.imread(filename)

                img = np.zeros((500, 1050, 3))
                img[:, :500] = bevmap[:, ::-1]
                img[:, 550:] = gt_bev
                cv2.imwrite('/home/yuan/Downloads/tmp.png', img)
                print(filename)


def generate_roadline(map_dir, map_bounds_file):
    """
    Convert global BEV semantic maps to 2d road line points.

    :param map_dir: directory for images of BEV semantic maps
    :param map_bounds_file: json file that describe the world coordinates of the BEV map origin (image[0, 0])
    :return: Nx2 array, 2d world coordinates of road line points in meters.
    """
    bounds = load_json(map_bounds_file)
    map_files = glob(map_dir)
    for mf in map_files:
        roadmap = cv2.imread(mf)
    # TODO


def convert_bev_semantic_map_to_road_height_map(map_dir, map_bounds_file, scenario_town_map_file, meta_dir):
    import torch
    bounds = load_json(map_bounds_file)
    scenario_town_map = load_json(scenario_town_map_file)
    map_files = os.listdir(map_dir)
    bevmaps = {mf.split('.')[0]: cv2.imread(os.path.join(map_dir, mf))[..., :2] for mf in map_files}
    trajectory = {mf.split('.')[0]: [] for mf in map_files}
    meta_files = glob(os.path.join(meta_dir, "*.json"))
    for mf in meta_files:
        scenario = os.path.basename(mf).split('.')[0]
        sdict = load_json(mf)
        ego_poses = []
        for f, fdict in sdict.items():
            # gt_boxes = {f"{int(x[0]):d}": x[1:] for x in ego_dict['gt_boxes']}
            # ego_box = gt_boxes[fdict['meta']['ego_id']]
            ego_poses.append(fdict['agents'][fdict['meta']['ego_id']]['pose'][:3])
        trajectory[scenario_town_map[scenario]].extend(ego_poses)

    for town, bevmap in bevmaps.items():
        inds = np.where(bevmap[..., 1])
        coords = np.stack(inds, axis=1) * 0.2
        coords = torch.from_numpy(coords).cuda()
        bound = bounds[town]
        coords[:, 0] += bound[0]
        coords[:, 1] += bound[1]
        traj_pts = torch.tensor(trajectory[town]).cuda()

        for i in range(0, len(coords), 10000):
            i1 = i*10000
            i2 = (i+1)*10000
            dists = torch.norm(coords[i1:i2, None, :2] - traj_pts[None, :, :2], dim=-1)
            min_dist, min_idx = dists.min(dim=-1)
            heights = traj_pts[min_idx][:, -1]
    # TODO


if __name__=="__main__":
    # opv2v_to_cosense(
    #     "/home/data/v2vreal",
    #     "/home/data/v2vreal/meta",
    #     isSim=False,
    #     pcd_ext='pcd'
    # )

    # generate_bevmaps(
    #     "/home/yuan/data/OPV2Va",
    #     "/home/yuan/data/OPV2Va/meta",
    # )

    convert_bev_semantic_map_to_road_height_map(
        "/code/CoSense3d/cosense3d/carla/assets/maps",
        "/code/CoSense3d/cosense3d/carla/assets/map_bounds.json",
        "/code/CoSense3d/cosense3d/carla/assets/scenario_town_map.json",
        "/home/data/OPV2Va/meta"
    )

