import os, random, copy
import glob
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
from plyfile import PlyData
import cv2

from cosense3d.utils.pclib import pose_to_transformation
from cosense3d.utils.pcdio import point_cloud_from_path
from cosense3d.utils.misc import load_json


class LoadLidarPoints:

    def __init__(self,
                 coop_mode=True,
                 load_attributes=['xyz', 'intensity'],
                 time_offset=0):
        self.coop_mode = coop_mode
        self.load_attributes = load_attributes
        self.time_offset = time_offset

    def read_pcd(self, pts_filename):
        pcd = point_cloud_from_path(pts_filename)
        points = np.stack([pcd.pc_data[x] for x in 'xyz'], axis=-1)
        lidar_dict = {'xyz': points}
        if 'intensity' in pcd.fields:
            lidar_dict['intensity'] = pcd.pc_data['intensity']
        if 'timestamp' in pcd.fields:
            lidar_dict['time'] = pcd.pc_data['timestamp']
        return lidar_dict

    def _load_points(self, pts_filename):
        """
        Load point clouds data form file.

        Parameters
        ----------
        pcd_file : str
            The pcd file that contains the point cloud.
        return_o3d: bool
            Default returns numpy array, set True to return pcd as o3d PointCloud object

        Returns
        -------
        lidar_dict:
            xyz: pcd_np | pcd : np.ndarray | o3d.geometry.PointCloud
                    The lidar xyz coordinates in numpy format, shape:(n, 3);
            intensity: (optional) np.ndarray, (n,);
            label: (optional) np.ndarray, (n,);
            time: (optional) np.ndarray, (n,);
            ray: (optional) np.ndarray, (n,);
        """
        lidar_dict = {}
        ext = os.path.splitext(pts_filename)[-1]
        if ext == '.pcd':
            # we do not use to avoid conflict with PyQt5
            lidar_dict = self.read_pcd(pts_filename)

            # pcd = o3d.io.read_point_cloud(pts_filename)
            # xyz = np.asarray(pcd.points, dtype=np.float32)
            # lidar_dict['xyz'] = xyz
            # # we save the intensity in the first channel
            # intensity = np.expand_dims(np.asarray(pcd.colors)[:, 0], -1)
            # if len(intensity) == len(xyz):
            #     lidar_dict['intensity'] = intensity

        elif ext == '.bin':
            pcd_np = np.fromfile(pts_filename, dtype=np.float32).reshape(-1, 4)
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
            ply = PlyData.read(pts_filename)
            data = ply['vertex']
            properties = [prop.name for prop in data.properties]
            data = {name: np.array(data[name]) for name in properties}
            xyz = np.stack([data.pop(x) for x in 'xyz'], axis=1)
            lidar_dict['xyz'] = xyz
            lidar_dict.update(data)
        else:
            raise NotImplementedError
        # reshape for cat
        for k, v in lidar_dict.items():
            if v.ndim == 1:
                lidar_dict[k] = v.reshape(-1, 1)
        return lidar_dict

    def _load_single(self, pts_filename, timestamp=0):
        lidar_dict = self._load_points(pts_filename)
        if 'intensity' in self.load_attributes and 'intensity' not in lidar_dict:
            lidar_dict['intensity'] = np.ones_like(lidar_dict['xyz'][:, :1])
        if 'time' in self.load_attributes:
            if 'time' in lidar_dict:
                lidar_dict['time'] -= self.time_offset
            else:
                lidar_dict['time'] = np.zeros_like(lidar_dict['xyz'][:, :1]) + (timestamp - self.time_offset)
        if 'distance' in self.load_attributes:
            lidar_dict['distance'] = np.linalg.norm(lidar_dict['xyz'][:, :2], axis=1, keepdims=True)
        if 'cosine' in self.load_attributes:
            lidar_dict['cosine'] = np.cos(np.arctan2(lidar_dict['xyz'][:, 1:2], lidar_dict['xyz'][:, 0:1]))
        if 'sine' in self.load_attributes:
            lidar_dict['sine'] = np.sin(np.arctan2(lidar_dict['xyz'][:, 1:2], lidar_dict['xyz'][:, 0:1]))

        points = np.concatenate(
            [lidar_dict[attri] for attri in self.load_attributes], axis=-1)

        return points

    def __call__(self, data_dict):
        if self.coop_mode:
            points = []
            for ai in data_dict['valid_agent_ids']:
                adict = data_dict['sample_info']['agents'][ai]
                filename = os.path.join(data_dict['data_path'], adict['lidar']['0']['filename'])
                points.append(self._load_single(filename, adict['lidar']['0']['time']))
        else:
            ego_id = data_dict['sample_info']['meta']['ego_id']
            ego_dict = data_dict['sample_info']['agents'][ego_id]
            filename = os.path.join(data_dict['data_path'], ego_dict['lidar']['0']['filename'])
            points = self._load_single(filename)

        data_dict['points'] = points
        data_dict['points_attributes'] = self.load_attributes

        return data_dict


class LoadLidarRangeImg:

    def __init__(self,
                 coop_mode=True,
                 load_attributes=['depth', 'intensity'],
                 time_offset=0):
        self.coop_mode = coop_mode
        self.load_attributes = load_attributes
        self.time_offset = time_offset

    def read_pcd(self, pts_filename):
        pcd = point_cloud_from_path(pts_filename)
        points = np.stack([pcd.pc_data[x] for x in 'xyz'], axis=-1)
        lidar_dict = {'xyz': points}
        if 'intensity' in pcd.fields:
            lidar_dict['intensity'] = pcd.pc_data['intensity']
        if 'timestamp' in pcd.fields:
            lidar_dict['time'] = pcd.pc_data['timestamp']
        return lidar_dict

    def _load_points(self, pts_filename):
        """
        Load point clouds data form file.

        Parameters
        ----------
        pcd_file : str
            The pcd file that contains the point cloud.
        return_o3d: bool
            Default returns numpy array, set True to return pcd as o3d PointCloud object

        Returns
        -------
        lidar_dict:
            xyz: pcd_np | pcd : np.ndarray | o3d.geometry.PointCloud
                    The lidar xyz coordinates in numpy format, shape:(n, 3);
            intensity: (optional) np.ndarray, (n,);
            label: (optional) np.ndarray, (n,);
            time: (optional) np.ndarray, (n,);
            ray: (optional) np.ndarray, (n,);
        """
        lidar_dict = {}
        ext = os.path.splitext(pts_filename)[-1]
        if ext == '.pcd':
            # we do not use to avoid conflict with PyQt5
            lidar_dict = self.read_pcd(pts_filename)

            # pcd = o3d.io.read_point_cloud(pts_filename)
            # xyz = np.asarray(pcd.points, dtype=np.float32)
            # lidar_dict['xyz'] = xyz
            # # we save the intensity in the first channel
            # intensity = np.expand_dims(np.asarray(pcd.colors)[:, 0], -1)
            # if len(intensity) == len(xyz):
            #     lidar_dict['intensity'] = intensity

        elif ext == '.bin':
            pcd_np = np.fromfile(pts_filename, dtype=np.float32).reshape(-1, 4)
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
            ply = PlyData.read(pts_filename)
            data = ply['vertex']
            properties = [prop.name for prop in data.properties]
            data = {name: np.array(data[name]) for name in properties}
            xyz = np.stack([data.pop(x) for x in 'xyz'], axis=1)
            lidar_dict['xyz'] = xyz
            lidar_dict.update(data)
        else:
            raise NotImplementedError
        # reshape for cat
        for k, v in lidar_dict.items():
            if v.ndim == 1:
                lidar_dict[k] = v.reshape(-1, 1)
        return lidar_dict

    def _load_single(self, pts_filename, timestamp=0):
        lidar_dict = self._load_points(pts_filename)
        if 'intensity' in self.load_attributes and 'intensity' not in lidar_dict:
            lidar_dict['intensity'] = np.ones_like(lidar_dict['xyz'][:, :1])
        if 'time' in self.load_attributes:
            if 'time' in lidar_dict:
                lidar_dict['time'] -= self.time_offset
            else:
                lidar_dict['time'] = np.zeros_like(lidar_dict['xyz'][:, :1]) + (timestamp - self.time_offset)
        if 'distance' in self.load_attributes:
            lidar_dict['distance'] = np.linalg.norm(lidar_dict['xyz'][:, :2], axis=1, keepdims=True)
        if 'cosine' in self.load_attributes:
            lidar_dict['cosine'] = np.cos(np.arctan2(lidar_dict['xyz'][:, 1:2], lidar_dict['xyz'][:, 0:1]))
        if 'sine' in self.load_attributes:
            lidar_dict['sine'] = np.sin(np.arctan2(lidar_dict['xyz'][:, 1:2], lidar_dict['xyz'][:, 0:1]))

        points = np.concatenate(
            [lidar_dict[attri] for attri in self.load_attributes], axis=-1)

        return points

    def __call__(self, data_dict):
        if self.coop_mode:
            points = []
            for ai in data_dict['valid_agent_ids']:
                adict = data_dict['sample_info']['agents'][ai]
                filename = os.path.join(data_dict['data_path'], adict['lidar']['0']['filename'])
                points.append(self._load_single(filename, adict['lidar']['0']['time']))
        else:
            ego_id = data_dict['sample_info']['meta']['ego_id']
            ego_dict = data_dict['sample_info']['agents'][ego_id]
            filename = os.path.join(data_dict['data_path'], ego_dict['lidar']['0']['filename'])
            points = self._load_single(filename)

        data_dict['points'] = points
        data_dict['points_attributes'] = self.load_attributes

        return data_dict


class LoadMultiViewImg:
    def __init__(self, bgr2rgb=False, to_float32=False, max_num_img=None, img_filter_keys=None):
        self.bgr2rgb = bgr2rgb
        self.to_float32 = to_float32
        self.max_num_img = max_num_img
        self.img_filter_keys = img_filter_keys

    def __call__(self, data_dict):
        agents = data_dict['sample_info']['agents']
        chosen_cams = OrderedDict()

        img = []
        for ai in data_dict['valid_agent_ids']:
            if ai not in agents:
                # previous agents might not in current frame when load sequential data
                continue
            adict = agents[ai]
            chosen_cams[ai] = []
            # get image info
            num_cam = 0
            if self.max_num_img is not None and self.max_num_img < len(adict['camera']):
                selected = random.sample(list(adict['camera'].keys()), k=self.max_num_img)
                cam_dicts = {ci: adict['camera'][ci] for ci in selected}
            else:
                cam_dicts = copy.copy(adict['camera'])
            for ci, cdict in cam_dicts.items():
                # One lidar frame might have several images, only take the 1st one
                filename = cdict['filenames'][0]
                if self.img_filter_keys is not None and \
                        len([1 for k in self.img_filter_keys if k in filename]) == 0:
                    continue
                num_cam += 1
                chosen_cams[ai].append(ci)
                img_file = os.path.join(data_dict['data_path'], filename)
                I = cv2.imread(img_file)
                if self.bgr2rgb:
                    I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
                img.append(I)
        # img is of shape (h, w, c, num_views)
        img = np.stack(img, axis=0)
        if self.to_float32:
            img = img.astype(np.float32)

        data_dict['img'] = img
        data_dict['chosen_cams'] = chosen_cams
        return data_dict


class LoadAnnotations:
    def __init__(self,
                 load2d=False, load_cam_param=False,
                 load3d_local=False, load3d_global=False,
                 load_global_time=False, load3d_pred=False,
                 min_num_pts=0, with_velocity=False,
                 class_agnostic_3d=True, time_offset=0,
                 loc_err=None):
        self.load2d = load2d
        self.load_cam_param = load_cam_param
        self.load3d_local = load3d_local
        self.load3d_global = load3d_global
        self.load3d_pred = load3d_pred
        self.load_global_time = load_global_time
        self.min_num_pts = min_num_pts
        self.with_velocity = with_velocity
        self.class_agnostic_3d = class_agnostic_3d
        self.time_offset = time_offset
        self.loc_err = np.array(loc_err) if loc_err is not None else None # x, y, r

    def __call__(self, data_dict):
        self._load_essential(data_dict)
        if self.load2d:
            data_dict = self._load_anno2d(data_dict)
        elif self.load_cam_param:
            data_dict = self._load_cam_param(data_dict)

        if self.load3d_local:
            data_dict = self._load_anno3d_local(data_dict)
        if self.load3d_global:
            data_dict = self._load_anno3d_global(data_dict)
        if self.load_global_time:
            data_dict = self._load_global_time(data_dict)
        if self.load3d_pred:
            data_dict = self._load_anno3d_pred(data_dict)

        return data_dict

    def _add_loc_err(self, pose, loc_err):
        pose_ = copy.deepcopy(pose)
        if self.loc_err is not None:
            loc_err = np.random.randn(3) * self.loc_err

        pose_[0] = pose[0] + loc_err[0]
        pose_[1] = pose[1] + loc_err[1]
        pose_[5] = pose[5] + loc_err[2]
        return pose_

    def _load_essential(self, data_dict):
        lidar_poses = []
        lidar_poses_gt = []
        vehicle_poses = []
        timestampes = []
        agents = data_dict['sample_info']['agents']
        loc_err = data_dict['loc_err']
        ego_pose = agents[data_dict['sample_info']['meta']['ego_id']]['lidar']['0']['pose']
        ego_pose = self._add_loc_err(ego_pose, loc_err[0])
        ego_pose = pose_to_transformation(ego_pose)
        for i, ai in enumerate(data_dict['valid_agent_ids']):
            if ai not in agents:
                # previous agents might not in current frame when load sequential data
                continue
            adict = agents[ai]
            lidar_pose = self._add_loc_err(adict['lidar']['0']['pose'], loc_err=loc_err[i])
            lidar_pose = pose_to_transformation(lidar_pose)
            lidar_poses.append(lidar_pose)
            if self.loc_err is not None or loc_err is not None:
                lidar_poses_gt.append(pose_to_transformation(adict['lidar']['0']['pose']))
            vehicle_poses.append(adict['pose'])
            if adict['lidar']['0']['time'] is not None:
                # dairv2x
                timestampes.append(adict['lidar']['0']['time'] - self.time_offset)
            else:
                # opv2v
                # TODO update opv2v meta files with lidar timestamps
                timestampes.append(int(data_dict['frame']) * 0.1 - self.time_offset)

        data_dict.update({
            'lidar_poses': lidar_poses,
            'ego_poses': ego_pose,
            'timestamp': timestampes,
            'vehicle_poses': vehicle_poses
        })

        if len(lidar_poses) == len(lidar_poses_gt):
            data_dict['lidar_poses_gt'] = lidar_poses_gt

        return data_dict

    def _load_cam_param(self, data_dict):
        intrinsics = []
        extrinsics = []
        lidar2img = []

        agents = data_dict['sample_info']['agents']
        chosen_cams = data_dict['chosen_cams']
        for ai in data_dict['valid_agent_ids']:
            if ai not in agents:
                # previous agents might not in current frame when load sequential data
                continue
            adict = agents[ai]
            cam_ids = chosen_cams[ai]
            for ci in cam_ids:
                cdict = adict['camera'][ci]
                I4x4 = np.eye(4)
                I4x4[:3, :3] = np.array(cdict['intrinsic'])
                intrinsics.append(I4x4.astype(np.float32))
                extrinsics.append(np.array(cdict['lidar2cam']).astype(np.float32))
                lidar2img.append(self.get_lidar2img_transform(
                    cdict['lidar2cam'], cdict['intrinsic']).astype(np.float32))

        data_dict.update({
            'intrinsics': intrinsics,
            'extrinsics': extrinsics,
            'lidar2img': lidar2img,
        })
        return data_dict

    def _load_anno2d(self, data_dict):
        intrinsics = []
        extrinsics = []
        lidar2img = []
        bboxes2d = []
        centers2d = []
        depths = []
        labels = []

        agents = data_dict['sample_info']['agents']
        chosen_cams = data_dict['chosen_cams']
        for ai in data_dict['valid_agent_ids']:
            if ai not in agents:
                # previous agents might not in current frame when load sequential data
                continue
            adict = agents[ai]
            cam_ids = chosen_cams[ai]
            for ci in cam_ids:
                cdict = adict['camera'][ci]
                I4x4 = np.eye(4)
                I4x4[:3, :3] = np.array(cdict['intrinsic'])
                intrinsics.append(I4x4.astype(np.float32))
                extrinsics.append(np.array(cdict['lidar2cam']).astype(np.float32))
                lidar2img.append(self.get_lidar2img_transform(
                    cdict['lidar2cam'], cdict['intrinsic']).astype(np.float32))
                cam_info = adict['camera'][ci]
                # num_lidar_pts = np.ones(len(gt_names)).astype(int)
                # valid_flag = np.ones(len(gt_names)).astype(bool)
                mask = np.array(cam_info['num_pts']) > self.min_num_pts
                bboxes2d.append(np.array(cam_info['bboxes2d']).astype(np.float32)[mask])
                centers2d.append(np.array(cam_info['centers2d']).astype(np.float32)[mask])
                depths.append(np.array(cam_info['depths']).astype(np.float32)[mask])
                labels.append(np.zeros(mask.sum(), dtype=int))

        data_dict.update({
            'intrinsics': intrinsics,
            'extrinsics': extrinsics,
            'lidar2img': lidar2img,
            'bboxes2d': bboxes2d,
            'centers2d': centers2d,
            'depths2d': depths,
            'labels2d': labels
        })
        return data_dict

    def _load_anno3d_local(self, data_dict):
        local_bboxes_3d = []
        local_labels_3d = []
        local_boxes_3d_id = []
        local_names = []
        agents = data_dict['sample_info']['agents']
        for ai in data_dict['valid_agent_ids']:
            if ai not in agents:
                # previous agents might not in current frame when load sequential data
                continue
            adict = agents[ai]
            boxes = np.array(adict['gt_boxes']).reshape(-1, 11)
            if 'num_pts' not in adict:
                mask = np.ones_like(boxes[:, 0]).astype(bool)
            else:
                mask = np.array(adict['num_pts']) > self.min_num_pts
            if len(boxes) != len(mask):
                # TODO: update num pts in meta
                mask = np.ones_like(boxes[..., 0]).astype(bool)
            boxes = boxes[mask]
            local_boxes = boxes[:, [2, 3, 4, 5, 6, 7, 10]].astype(np.float32)
            local_boxes_id = boxes[:, 0].astype(int)
            if self.class_agnostic_3d:
                local_labels = np.zeros(len(boxes), dtype=int)
            else:
                local_labels = boxes[:, 1].astype(int)
            if self.with_velocity:
                if 'velos' in adict:
                    velos = np.array(adict['velos']).reshape(-1, 2).astype(np.float32) / 3.6
                    local_boxes = np.concatenate([local_boxes, velos[mask]], axis=-1)
                else:
                    velos = np.zeros_like(local_boxes[:, :2])
                    local_boxes = np.concatenate([local_boxes, velos], axis=-1)
            local_bboxes_3d.append(local_boxes)
            local_labels_3d.append(local_labels)
            local_boxes_3d_id.append(local_boxes_id)
            assert np.all(local_labels == 0), "Num. cls > 1 not implemented."
            local_names.append(['car' for _ in local_labels])

        data_dict.update({
            'local_bboxes_3d': local_bboxes_3d,
            'local_labels_3d': local_labels_3d,
            'local_bboxes_id': local_boxes_3d_id,
            'local_names': local_names,
        })

        return data_dict

    def _load_anno3d_global(self, data_dict):
        frame_meta = data_dict['sample_info']['meta']
        boxes = np.array(frame_meta['bbx_center_global'])
        global_bboxes_3d = boxes[:, [2, 3, 4, 5, 6, 7, 10]].astype(np.float32)
        global_bboxes_id = boxes[:, 0].astype(int)
        if self.class_agnostic_3d:
            global_labels_3d = np.zeros(len(boxes), dtype=int)
        else:
            global_labels_3d = boxes[:, 1].astype(int)

        if self.with_velocity:
            if 'bbx_velo_global' in frame_meta:
                global_velocity = np.array(frame_meta['bbx_velo_global']).astype(np.float32) / 3.6
            else:
                global_velocity = np.zeros_like(global_bboxes_3d[:, :2])
            global_bboxes_3d = np.concatenate([global_bboxes_3d, global_velocity], axis=-1)

        if 'num_pts' in frame_meta and self.min_num_pts > 0:
            global_box_num_pts = np.array(frame_meta['num_pts'])
            mask = global_box_num_pts > self.min_num_pts
            global_bboxes_3d = global_bboxes_3d[mask]
            global_labels_3d = global_labels_3d[mask]
            global_bboxes_id = global_bboxes_id[mask]

        # TODO: currently only support car
        global_names = ['car' for _ in global_labels_3d]
        data_dict.update({
            'global_bboxes_3d': global_bboxes_3d,
            'global_labels_3d': global_labels_3d,
            'global_bboxes_id': global_bboxes_id,
            'global_names': global_names,
        })
        return data_dict

    def _load_global_time(self, data_dict):
        frame_meta = data_dict['sample_info']['meta']
        if 'global_bbox_time' in frame_meta:
            data_dict['global_time'] = frame_meta['global_bbox_time'][0]
        else:
            data_dict['global_time'] = data_dict['points'][0][:, -1].max()
        return data_dict

    def _load_anno3d_pred(self, data_dict):
        frames = sorted(list(data_dict['sample_info']['meta']['boxes_pred'].keys()))
        boxes_preds = [data_dict['sample_info']['meta']['boxes_pred'][f] for f in frames]
        data_dict['bboxes_3d_pred'] = np.array(boxes_preds)
        return data_dict

    def get_lidar2img_transform(self, lidar2cam, intrinsic):
        if isinstance(lidar2cam, list):
            intrinsic = np.array(intrinsic)
        try:
            P = intrinsic @ lidar2cam[:3]
        except:
            print(intrinsic)
            print(lidar2cam)
        lidar2img = np.concatenate([P, lidar2cam[3:]], axis=0)
        return lidar2img


class LoadOPV2VBevMaps:
    def __init__(self, keys=None, use_global_map=True, ego_only=True, range=75):
        self.keys = keys
        self.use_global_map = use_global_map
        self.ego_only = ego_only
        self.range = range
        self.map_res = 0.2
        self.map_size = int(self.range * 2 / self.map_res)
        pad = int(range / self.map_res)
        if self.use_global_map:
            self.keys = ['bevmap', 'bevmap_coor']
            assets_path = f"{os.path.dirname(__file__)}/../../carla/assets"
            map_path = f"{assets_path}/maps/png"
            map_files = glob.glob(os.path.join(map_path, '*.png'))
            self.scene_maps = load_json(os.path.join(assets_path, 'scenario_town_map.json'))
            self.map_bounds = load_json(os.path.join(assets_path, 'map_bounds.json'))
            self.bevmaps = {}
            for mf in map_files:
                town = os.path.basename(mf).split('.')[0]
                bevmap = cv2.imread(mf) / 255.
                # bevmap = np.pad(bevmap, ((pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=0)
                self.bevmaps[town] = bevmap

            # grid coor template
            grid = np.ones((self.map_size, self.map_size))
            inds = np.stack(np.where(grid))
            xy = inds * 0.2 - self.range + 0.1
            self.xy_pad = np.concatenate([xy, np.zeros_like(xy[:1]), np.ones_like(xy[:1])], axis=0)
            # carla has different coor system as cosense3d, T_corr: carla -> cosense3d
            self.T_corr = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 0, 1]])
        else:
            assert keys is not None and len(keys) > 0

    def __call__(self, data_dict):
        path = os.path.join(data_dict['data_path'], data_dict['scenario'])
        ego_id = data_dict['sample_info']['meta']['ego_id']
        load_dict = {}

        agents = data_dict['valid_agent_ids']
        for ai in agents:
            if self.ego_only and ego_id != ai:
                for k in load_dict.keys():
                    load_dict[k].append(None)
                continue
            else:
                out = self.load_single(path, ai, data_dict)
                for k in out.keys():
                    if k not in load_dict:
                        load_dict[k] = []
                    load_dict[k].append(out[k])

        data_dict.update(load_dict)
        return data_dict

    def load_single(self, path, ai, data_dict):
        # map1 = self.crop_map_for_pose(data_dict, ai)[0]
        # map2 = cv2.imread(os.path.join(path, ai, f"{data_dict['frame']}_bev.png"))[..., 0]
        # map2 = np.array(map2, dtype=float) / 255.
        # map2[map2 > 0] = 1
        # map2 = np.flip(map2, 0).copy()
        # bevmap = np.zeros((500, 500, 3))
        # bevmap[..., 0] = map1
        # bevmap[..., 1] = map2
        # import matplotlib.pyplot as plt
        # plt.imshow(bevmap)
        # plt.show()
        # plt.close()

        out = {}
        if self.use_global_map:
            out['bevmap'], out['bevmap_coor'] = self.crop_map_for_pose(data_dict, ai)
        else:
            frame = data_dict['frame']
            for k in self.keys:
                filename = os.path.join(path, ai, f"{frame}_{k}.png")
                bev_map = cv2.imread(filename)[..., 0]
                # bev_map = cv2.cvtColor(bev_map, cv2.COLOR_BGR2GRAY)
                bev_map = np.array(bev_map, dtype=float) / 255.
                bev_map[bev_map > 0] = 1
                out[f'{k}map'] = np.flip(bev_map, 0).copy()
                out[f'{k}map_coor'] = [-self.range, - self.range]
        return out

    def crop_map_for_pose(self, data_dict, ai):
        scenario = data_dict['scenario']
        town = self.scene_maps[scenario]
        # lidar_pose = pose_to_transformation(data_dict['sample_info']['agents'][ai]['lidar']['0']['pose'])
        lidar_pose = data_dict['lidar_poses'][data_dict['valid_agent_ids'].index(ai)]
        cur_map = self.bevmaps[town]
        sx, sy = cur_map.shape[:2]
        bound = self.map_bounds[town]

        # transform template bev points to world coordinates
        transform = self.T_corr @ lidar_pose
        xy_tf = transform @ self.xy_pad
        # calculate map indices of bev points
        xy_tf[0] -= bound[0] + 1.0
        xy_tf[1] -= bound[1] + 1.0
        map_inds = np.floor(xy_tf[:2] / 0.2)
        xs = np.clip(map_inds[0], 0, sx - 1).astype(int)
        ys = np.clip(map_inds[1], 0, sy - 1).astype(int)
        # retrieve cropped bev map from global map
        bevmap = cur_map[xs, ys].reshape(self.map_size, self.map_size, 3) # [::-1, ::-1]

        # # bound[0] -= self.range
        # # bound[1] -= self.range
        # offset_x = int((lidar_pose[0] - self.range - bound[0]) / self.map_res)
        # offset_y = int((lidar_pose[1] - self.range - bound[1]) / self.map_res)
        #
        # xmin = max(offset_x, 0)
        # xmax = min(offset_x + size, bevmap.shape[0] - 1)
        # ymin = max(offset_y, 0)
        # ymax = min(offset_y + size, bevmap.shape[1] - 1)
        # bevmap_crop = bevmap[xmin:xmax, ymin:ymax]
        # bevmap_coor = [bound[0] + xmin * self.map_res, bound[1] + ymin * self.map_res]

        if data_dict['sample_info']['agents'][ai]['pose'][2] > 2:
            bevmap = bevmap[..., 1].astype(bool)
        else:
            bevmap = bevmap[..., :2].any(-1)

        # import matplotlib.pyplot as plt
        # points = data_dict['points'][0]
        # mask = bevmap.reshape(-1).astype(bool)
        # plt.plot(self.xy_pad[0, mask], self.xy_pad[1, mask], 'g.', markersize=1)
        # plt.plot(points[:, 0], points[:, 1], 'b.', markersize=1)
        # plt.show()
        # plt.close()

        return bevmap, [-self.range, - self.range]


class LoadCarlaRoadlineMaps:
    def __init__(self, ego_only=True, range=75):
        self.ego_only = ego_only
        self.range = range
        assets_path = f"{os.path.dirname(__file__)}/../../carla/assets"
        map_path = f"{assets_path}/maps/roadline"
        map_files = glob.glob(os.path.join(map_path, '*.bin'))
        self.scene_maps = load_json(os.path.join(assets_path, 'scenario_town_map.json'))
        self.maps = {}
        for mf in map_files:
            town = os.path.basename(mf).split('.')[0]
            rlmap = np.fromfile(mf, dtype=float).reshape(-1, 2)
            # bevmap = np.pad(bevmap, ((pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=0)
            self.maps[town] = rlmap

    def __call__(self, data_dict):
        path = os.path.join(data_dict['data_path'], data_dict['scenario'])
        ego_id = data_dict['sample_info']['meta']['ego_id']
        roadline = []

        agents = data_dict['valid_agent_ids']
        for ai in agents:
            if self.ego_only and ego_id != ai:
                out = None
            else:
                out = self.load_single(path, ai, data_dict)
            roadline.append(out)

        data_dict['roadline'] = roadline
        return data_dict

    def load_single(self, path, ai, data_dict):
        scenario = data_dict['scenario']
        town = self.scene_maps[scenario]
        # lidar_pose = pose_to_transformation(data_dict['sample_info']['agents'][ai]['lidar']['0']['pose'])
        lidar_pose = data_dict['lidar_poses'][data_dict['valid_agent_ids'].index(ai)]
        cur_map = self.maps[town]

        mask = (cur_map[:, 0] > (lidar_pose[0, 3] - self.range)) & \
               (cur_map[:, 0] < (lidar_pose[0, 3] + self.range)) & \
               (cur_map[:, 1] > (lidar_pose[1, 3] - self.range)) & \
               (cur_map[:, 1] < (lidar_pose[1, 3] + self.range))

        roadline = cur_map[mask]

        # # visualize
        # lidar_file = data_dict['sample_info']['agents'][ai]['lidar']['0']['filename']
        # lidar_file = os.path.join(os.path.dirname(path), lidar_file)
        # ply = PlyData.read(lidar_file)
        # data = ply['vertex']
        # properties = [prop.name for prop in data.properties]
        # data = {name: np.array(data[name]) for name in properties}
        # pcd = np.stack([data.pop(x) for x in 'xyz'], axis=1)
        # pcd = lidar_pose @ np.concatenate([pcd, np.ones_like(pcd[:, :1])], axis=-1).T
        #
        # import matplotlib.pyplot as plt
        # plt.plot(roadline[:, 0], roadline[:, 1], 'g.', markersize=1)
        # plt.plot(pcd[0], pcd[1], 'r.', markersize=1)
        # plt.show()
        # plt.close()

        return roadline


class LoadSparseBevTargetPoints:
    def __init__(self, num_points=3000, ego_only=False):
        self.num_points = num_points
        self.ego_only = ego_only

    def __call__(self, data_dict):
        bev_pts = []
        agents = data_dict['sample_info']['agents']
        ego_id = data_dict['sample_info']['meta']['ego_id']
        for ai in data_dict['valid_agent_ids']:
            if ai not in agents:
                # previous agents might not in current frame when load sequential data
                continue
            if self.ego_only and ai != ego_id:
                bev_pts.append(np.empty((0, 3)))
            else:
                pass

    def generate_sparse_bev_pts(self, pcd):
        pass





