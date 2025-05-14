import copy
import glob
import os
import pickle
import random

import torch
import tqdm
import yaml

import numpy as np
from cosense3d.utils.misc import load_json, save_json
from cosense3d.utils import pclib
from cosense3d.ops.utils import points_in_boxes_cpu
from cosense3d.dataset.toolkit import register_pcds


type_sustech2cosense = {
    'Car': 'vehicle.car',
    'Van': 'vehicle.van',
    'Truck': 'vehicle.truck',
    'Bus': 'vehicle.bus',
    'Tram': 'vehicle.tram',
    'Unknown': 'unknown',
    'BicycleRider': 'vehicle.cyclist',
    'Bicyclerider': 'vehicle.cyclist',
    'MotorcyleRider': 'vehicle.motorcycle',
    'Pedestrian': 'human.pedestrian',
    'HumanSitting': 'human.sitting',
    'Scooterrider': 'vehicle.scooter'
}
type_cosense2sustech = {
    v: k for k, v in type_sustech2cosense.items()
}

csColors = {
    'vehicle.car': [0, 215, 255], #0
    'vehicle.van': [246, 250, 112], #1
    'vehicle.truck': [255, 132, 0], #2
    'vehicle.bus': [0, 223, 162], #3
    'vehicle.tram': [0, 121, 255], #4
    'vehicle.motorcycle': [255, 0, 96], #5
    'vehicle.cyclist': [244, 35, 232], #6
    'vehicle.scooter': [227, 132, 255], #7
    'vehicle.other': [180, 254, 152], #8
    'human.pedestrian': [220, 20, 60], #9
    'human.wheelchair': [134, 93, 255], #10
    'human.sitting': [56, 229, 77], #11
    'static.trafficcone': [255, 0, 0], #12
    'static.barrowlist': [255, 50, 0], #13
    'vehicle.tricyclist': [255, 50, 50], #14
    'unknown': [255, 255, 255],
}


class CoSenseDataConverter:
    OBJ_LIST = [
        'vehicle.car',   #0
        'vehicle.van',   #1
        'vehicle.truck',   #2
        'vehicle.bus',   #3
        'vehicle.tram',   #4
        'vehicle.motorcycle',   #5
        'vehicle.cyclist',   #6
        'vehicle.scooter',   #7
        'vehicle.other',   #8
        'human.pedestrian',   #9
        'human.wheelchair',   #10
        'human.sitting',   #11
        'static.trafficcone',   #12
        'static.barrowlist',   #13
        'vehicle.tricyclist',   #13
        'unknown',   #14
    ]
    OBJ_ID2NAME = {i: n for i, n in enumerate(OBJ_LIST)}
    OBJ_NAME2ID = {n: i for i, n in enumerate(OBJ_LIST)}

    def __init__(self, data_path, meta_path, mode='all'):
        self.data_path = data_path
        self.meta_path = meta_path
        self.meta = self.load_meta(meta_path, mode)

    def update_from_sustech(self, sustech_path):
        for scenario, sdict in self.meta.items():
            for frame, fdict in sdict.items():
                new_label_file = os.path.join(
                    sustech_path,
                    scenario, 'label',
                    frame + '.json'
                )
                objects = self.obj_from_sustech(new_label_file)
                # TODO the transformation from local to global
                self.meta[scenario][frame]['meta']['bbx_center_global'] = objects

            save_json(sdict, os.path.join(self.meta_path, f"{scenario}.json"))

    def to_sustech(self, out_dir=None):
        # make out dirs
        out_dir = os.path.join(self.data_path, '..', 'sustech_fmt') \
            if out_dir is None else out_dir
        for s, sdict in self.meta.items():
            scenario_dir = os.path.join(out_dir, s)
            os.makedirs(os.path.join(scenario_dir, 'lidar'), exist_ok=True)
            os.makedirs(os.path.join(scenario_dir, 'label'), exist_ok=True)
            for f, fdict in tqdm.tqdm(sdict.items()):
                bbx_global_center = np.array(fdict['meta']['bbx_center_global'])
                # bbx_global_corner = boxes_to_corners_3d(bbx_global_center[:, 2:])
                lidars = []
                for a, adict in fdict['agents'].items():
                    for l, ldict in adict['lidar'].items():
                        lidar_pose = ldict['pose']
                        filename = ldict['filename'].replace('\\', '/')
                        # TODO rotate points and bbxs
                        pcd = pclib.load_pcd(os.path.join(self.data_path, filename))
                        points = np.concatenate([pcd['xyz'], pcd['intensity'].reshape(-1, 1)], axis=-1)
                        lidars.append(points.astype(np.float32))
                lidars = np.concatenate(lidars, axis=0)
                lidars.tofile(os.path.join(out_dir, scenario_dir, 'lidar', f"{f}.bin"))
                # write label file
                self.obj_to_sustech(
                    bbx_global_center,
                    os.path.join(out_dir, scenario_dir, 'label', f"{f}.json")
                )

    def to_opv2v(self, out_dir=None):
        # make out dirs
        out_dir = os.path.join(self.data_path, '..', 'opv2v_fmt') \
            if out_dir is None else out_dir
        os.makedirs(out_dir, exist_ok=True)
        for s, sdict in self.meta.items():
            scenario_dir = os.path.join(out_dir, s)
            os.makedirs(scenario_dir, exist_ok=True)
            for f, fdict in tqdm.tqdm(sdict.items()):
                bbx_global_center = np.array(fdict['meta']['bbx_center_global'])
                # bbx_global_corner = boxes_to_corners_3d(bbx_global_center[:, 2:])
                for a, adict in fdict['agents'].items():
                    agent_dir = os.path.join(scenario_dir, a)
                    if not os.path.exists(agent_dir):
                        os.makedirs(agent_dir)
                    for l, ldict in adict['lidar'].items():
                        lidar_pose = ldict['pose']
                        filename = ldict['filename'].replace('\\', '/')
                        # TODO rotate points and bbxs
                        pclib.lidar_bin2bin(
                            os.path.join(self.data_path, filename),
                            os.path.join(agent_dir, f + '.bin')
                        )
                        # write label file
                        self.obj_to_opv2v(bbx_global_center, lidar_pose,
                                          os.path.join(agent_dir, f + '.yaml'))

    def to_kitti(self, out_dir=None):
        from cosense3d.dataset.toolkit.kitti import type_cosense2kitti
        split = {
            # 'train': ['measurement4_0'],
            # 'val': ['measurement4_1'],
            'test': sorted(self.meta.keys()),
        }
        # make out dirs
        out_dir = os.path.join(self.data_path, '..', 'kitti_test') \
            if out_dir is None else out_dir
        os.makedirs(os.path.join(out_dir, 'ImageSets'), exist_ok=True)
        for dir_name in ['velodyne', 'image_2', 'label_2', 'calib']:
            os.makedirs(os.path.join(out_dir, 'training', dir_name), exist_ok=True)
            os.makedirs(os.path.join(out_dir, 'validating', dir_name), exist_ok=True)
            os.makedirs(os.path.join(out_dir, 'testing', dir_name), exist_ok=True)
        # create split files
        for sp, seqs in split.items():
            with open(os.path.join(out_dir, 'ImageSets', f"{sp}.txt"), 'w') as fh:
                frames = []
                for seq in seqs:
                    cur_frames = sorted(self.meta[seq].keys())
                    cur_frames = [seq.split('_')[0][-1] + f[1:] for f in cur_frames]
                    frames.extend(cur_frames)
                fh.write("\n".join(sorted(frames)))
            for s, sdict in self.meta.items():
                if s not in split[sp] or int(s.split('_')[1]) < 10:
                    continue
                print(sp, s)
                scenario_dir = os.path.join(out_dir, s)
                cur_split = {'train': 'training', 'val': 'validating', 'test': 'testing'}[sp]
                # os.makedirs(scenario_dir, exist_ok=True)
                # sdict = {k: sdict[k] for k in sorted(list(sdict.keys()))[:10]}
                for f, fdict in tqdm.tqdm(sdict.items()):
                    ##### save lidar ######
                    points = []
                    for ai, adict in fdict['agents'].items():
                        for li, ldict in adict['lidar'].items():
                            lidar_file = os.path.join(self.data_path, ldict['filename'])
                            points.append(
                                np.fromfile(lidar_file, np.float32).reshape(-1, 4)
                            )
                    points = np.concatenate(points, axis=0)
                    lidar_out_file = os.path.join(
                        out_dir, cur_split, 'velodyne', f"{s.split('_')[0][-1] + f[1:]}.bin"
                    )
                    points.tofile(lidar_out_file)
                    ######## save label #######
                    label = fdict['meta']['bbx_center_global']
                    label_out_file = os.path.join(
                        out_dir, cur_split, 'label_2', f"{s.split('_')[0][-1] + f[1:]}.txt"
                    )
                    with open(label_out_file, 'w') as fh:
                        for l in label:
                            # kitti label format
                            cosense_type = self.OBJ_ID2NAME[l[1]]
                            type = [type_cosense2kitti[cosense_type]]

                            trancated = ['0']
                            occluded = ['0']
                            alpha = [f"{np.arctan2(l[3], l[2]):.2f}"]
                            bbox = ['0'] * 4
                            dimensions = [f"{l[x]:.2f}" for x in [7, 6, 5]] # hwl
                            l[4] -= l[7] / 2
                            location = [f"{l[x]:.2f}" for x in [2, 3, 4]] # in cam coor
                            rotation_y = [f"{-l[10] - np.pi/2:.2f}"]
                            ls = type + trancated + occluded + alpha + bbox + dimensions +\
                                     location + rotation_y
                            line = " ".join(ls)
                            fh.write(line)
                            fh.write('\n')

    def obj_from_sustech(self, label_file):
        if not os.path.exists(label_file):
            return []
        objs = load_json(label_file)
        bboxes = []
        for obj_dict in objs:
            obj_id = obj_dict['obj_id']
            obj_type = obj_dict['obj_type']
            position = obj_dict['psr']['position']
            rotation = obj_dict['psr']['rotation']
            scale = obj_dict['psr']['scale']

            cosense_type_name = type_sustech2cosense[obj_type]
            obj_type_id = self.OBJ_NAME2ID[cosense_type_name]
            bbx_center = [
                float(obj_id),
                float(obj_type_id),
                position['x'],
                position['y'],
                position['z'],
                scale['x'],
                scale['y'],
                scale['z'],
                rotation['x'],
                rotation['y'],
                rotation['z'],
            ]
            bboxes.append(bbx_center)
        return bboxes

    def obj_to_sustech(self, cosense_objs, sustech_file):
        sustech_objs = []
        if len(cosense_objs.shape) == 0:
            save_json(sustech_objs, sustech_file)
            return
        for obj in cosense_objs:
            obj_type = type_cosense2sustech[
                self.OBJ_ID2NAME[int(obj[1])]
            ]
            sustech_objs.append(
                {
                    'obj_id': obj[0],
                    'obj_type': obj_type,
                    'psr': {
                        'position': {
                            'x': obj[2],
                            'y': obj[3],
                            'z': obj[4]
                        },
                        'scale': {
                            'x': obj[5],
                            'y': obj[6],
                            'z': obj[7]
                        },
                        'rotation': {
                            'x': obj[8],
                            'y': obj[9],
                            'z': obj[10]
                        }
                    }
                }
            )
        save_json(sustech_objs, sustech_file)

    def obj_to_opv2v(self, bbxs, pose, out_file, timestamp=None):
        vehicles = {}
        # only keep car, van, bus, truck
        bbxs = bbxs[bbxs[:, 1] < 4]
        for bbx in bbxs:
            obj_id = int(bbx[0])
            obj_type = int(bbx[1])
            # process the information to the opv2v format
            location = bbx[2:5]
            angle = bbx[[8, 10, 9]] / np.pi * 180
            angle[[0, 2]] *= -1
            extent = bbx[5:8] / 2

            vehicles[int(obj_id)] = {
                'angle': angle.tolist(),
                'center': [0.0] * 3,
                'extent': extent.tolist(),
                'location': location.tolist(),
                'speed': 0,
                'type': obj_type
            }
        if isinstance(pose, np.ndarray):
            pose = pose.tolist()
        yaml_dict = {
            'lidar_pose': pose,
            'true_ego_pos': pose,
            'ego_speed': 0,
            'vehicles': vehicles
        }
        if timestamp is not None:
            # timestamp for ouster is corrected by subtracting a systematic time offset (0.35s)
            yaml_dict['timestamp'] = float(timestamp)
        with open(out_file, 'w') as fh:
            yaml.dump(yaml_dict, fh, default_flow_style=False)


    @staticmethod
    def load_meta(meta_path, mode):
        if mode == 'all':
            scenario_meta_files = sorted(glob.glob(meta_path + "/*.json"))
        else:
            scenario_meta_files = []
            with open(os.path.join(meta_path, f'{mode}.txt'), 'r') as fh:
                for line in fh.readlines():
                    scenario_meta_files.append(os.path.join(meta_path, f'{line.strip()}.json'))

        meta_dict = {}

        for f in scenario_meta_files:
            scenario = os.path.basename(f)[:-5]
            meta_dict[scenario] = load_json(f)

        return meta_dict

    @staticmethod
    def cal_vbbx_mean_dim(meta):
        """Calculate mean dimensions of four-wheel vehicles"""
        dimensions = []
        for s, sdict in meta.items():
            for f, fdict in sdict.items():
                bbx = np.array(fdict['meta']['bbx_center_global'])
                dimensions.append(bbx[bbx[:, 5] > 2, 5:8])
        print(np.concatenate(dimensions, axis=0).mean(axis=0))

    @staticmethod
    def fdict_template():
        return {
                    'agents': {
                        '0': {
                            'type': None,
                            'pose': [0.0] * 6,
                            'time': None,  # timestamp for the current vehicle pose
                            'lidar': {
                                '0': {
                                    'pose': [0.0] * 6,
                                    'time': None,  # timestamp for the current lidar triggering round
                                    'filename': None
                                }
                            },
                            'camera': {},  # TODO API for cameras
                        }
                    },
                    # no cooperation needed, take lidar as global for each frame
                    'meta': {'bbx_center_global': []}
                }

    @staticmethod
    def add_cam_to_fdict(fdict, agent_id, cam_id, filenames, intrinsic, extrinsic, **kwargs):
        if agent_id not in fdict['agents']:
            adict = CoSenseDataConverter.fdict_template()['agents'][0]
            fdict['agents'][agent_id] = adict
        kwargs.update({
            'filenames': filenames,
            'intrinsic': intrinsic,
            'extrinsic': extrinsic
        })
        fdict['agents'][agent_id]['camera'][cam_id] = kwargs

    @staticmethod
    def update_frame_bbx(fdict, bbx):
        fdict['meta']['bbx_center_global'] = bbx

    @staticmethod
    def update_agent(fdict,
                     agent_id,
                     agent_type=None,
                     agent_pose=None,
                     agent_time=None,
                     **kwargs):
        if agent_id not in fdict['agents']:
            fdict['agents'][agent_id] = CoSenseDataConverter.fdict_template()['agents']['0']
        if agent_type is not None:
            fdict['agents'][agent_id]['type'] = agent_type
        if agent_pose is not None:
            fdict['agents'][agent_id]['pose'] = agent_pose
        if agent_time is not None:
            fdict['agents'][agent_id]['time'] = agent_time
        for k, v in kwargs.items():
            fdict['agents'][agent_id][k] = v

    @staticmethod
    def update_agent_lidar(fdict,
                           agent_id,
                           lidar_id,
                           lidar_pose=None,
                           lidar_time=None,
                           lidar_file=None):
        if agent_id not in fdict['agents']:
            fdict['agents'][agent_id] = CoSenseDataConverter.fdict_template()['agents']['0']
        if lidar_pose is not None:
            fdict['agents'][agent_id]['lidar'][lidar_id]['pose'] = lidar_pose
        if lidar_time is not None:
            fdict['agents'][agent_id]['lidar'][lidar_id]['time'] = lidar_time
        if lidar_file is not None:
            fdict['agents'][agent_id]['lidar'][lidar_id]['filename'] = lidar_file

    @staticmethod
    def update_agent_gt_boxes(fdict,
                              agent_id,
                              gt_boxes):
        if agent_id not in fdict['agents']:
            fdict['agents'][agent_id] = CoSenseDataConverter.fdict_template()['agents']['0']
        fdict['agents'][agent_id]['gt_boxes'] = gt_boxes

    @staticmethod
    def remove_lidar_info(fdict, agent_id):
        fdict['agents'][agent_id]['lidar'] = {}

    @staticmethod
    def supervison_full_to_sparse(meta_dict, out_path, lidar_range=None, det_r=None,
                                  num_box_per_frame=None, num_box_total=None, label_ratio=None):
        def select_box(bboxes, cls_idx, num):
            bboxes = np.array(bboxes)
            bboxes_car = bboxes[bboxes[:, 1] == cls_idx]
            if lidar_range is not None:
                mask = (bboxes_car[:, 2] > lidar_range[0]) & (bboxes_car[:, 2] < lidar_range[3]) & \
                       (bboxes_car[:, 3] > lidar_range[1]) & (bboxes_car[:, 3] < lidar_range[4]) & \
                       (bboxes_car[:, 4] > lidar_range[2]) & (bboxes_car[:, 4] < lidar_range[5])
            else:
                mask = np.linalg.norm(bboxes_car[:, 2:4], axis=1) < det_r
            bboxes_car = bboxes_car[mask]
            if len(bboxes_car) == 0:
                return None
            choice = np.random.choice(np.array(len(bboxes_car)), num)
            bboxes_car = bboxes_car[choice].reshape(num, 11).tolist()
            return bboxes_car

        if num_box_per_frame is not None:
            for s, sdict in meta_dict.items():
                sdict_out = copy.deepcopy(sdict)
                for f, fdict in sdict.items():
                    bboxes = fdict['meta']['bbx_center_global']
                    choice = select_box(bboxes, 0, 1)
                    if choice is None:
                        sdict_out.pop(f)
                    else:
                        sdict_out[f]['meta']['bbx_center_global'] = choice
                save_json(sdict_out, os.path.join(out_path, f'{s}.json'))
        elif num_box_total is not None:
            samples = []
            # find frames with car labels
            for s, sdict in meta_dict.items():
                for f, fdict in sdict.items():
                    bboxes = fdict['meta']['bbx_center_global']
                    classes = [int(b[1]) for b in bboxes]
                    if 0 in classes:
                        samples.append((s, f))
            # select given number of frames
            samples = random.choices(samples, k=num_box_total)
            sdict_out = {}
            for sample in samples:
                fdict = copy.deepcopy(meta_dict[sample[0]][sample[1]])
                bboxes = fdict['meta']['bbx_center_global']
                fdict['meta']['bbx_center_global'] = select_box(bboxes, 0, 1)
                sdict_out[sample[1]] = fdict
            save_json(sdict_out, os.path.join(out_path, 'train.json'))
            with open(os.path.join(out_path, 'train.txt'), 'w') as fh:
                fh.write('train')

    @staticmethod
    def global_boxes_to_local(meta_dict, data_path, meta_path):
        samples = {i: {'box': [], 'points': []} for i in CoSenseDataConverter.OBJ_ID2NAME.keys()}
        for s, sdict in meta_dict.items():
            for f, fdict in tqdm.tqdm(meta_dict[s].items()):
                global_boxes = fdict['meta']['bbx_center_global']
                global_boxes = np.array(global_boxes)
                for a, adict in fdict['agents'].items():
                    for l, ldict in adict['lidar'].items():
                        lidar = pclib.load_pcd(os.path.join(data_path, ldict['filename']))
                        box_cls = global_boxes[:, 1]
                        res = points_in_boxes_cpu(lidar['xyz'], global_boxes[:, [2, 3, 4, 5, 6, 7, 10]])
                        box_n_pts = res.sum(axis=1)
                        valid = box_n_pts > 10
                        boxes = global_boxes[valid]
                        box_cls = box_cls[valid]
                        pts_idx_of_boxes = res[valid]
                        CoSenseDataConverter.update_agent_gt_boxes(fdict, a, boxes.tolist())

                        for i, box in enumerate(boxes):
                            cls = box[1]
                            points = lidar['xyz'][pts_idx_of_boxes[i].astype(bool)]
                            intensity = lidar['intensity'][pts_idx_of_boxes[i].astype(bool)]
                            # transform box and points to box coodiante
                            points = points - box[2:5].reshape(1, 3)
                            # points will be modified during transformation, so make a copy here
                            new_points = np.copy(points)
                            st = np.sin(-box[-1])
                            ct = np.cos(-box[-1])
                            points[:, 0] = new_points[:, 0] * ct - new_points[:, 1] * st
                            points[:, 1] = new_points[:, 0] * st + new_points[:, 1] * ct
                            points = np.concatenate([points, intensity[:, None]], axis=1)
                            samples[cls]['box'].append(box[5:8])
                            samples[cls]['points'].append(points)

                            # from cosense3d.utils.vislib import draw_points_boxes_plt, plt
                            # box_vis = np.array([[0]*3 + box[5:8].tolist() + [0]])
                            # ax = plt.figure(figsize=(10, 10)).add_subplot(1, 1, 1)
                            # draw_points_boxes_plt(
                            #     ax=ax,
                            #     pc_range=5,
                            #     points=points,
                            #     boxes_gt=box_vis,
                            #     filename='/home/yuan/Downloads/tmp.png'
                            # )


            save_json(sdict, os.path.join(meta_path, f'{s}.json'))
        for sample_id, content in samples.items():
            if len(content['box']) == 0:
                continue
            sample_name = CoSenseDataConverter.OBJ_ID2NAME[sample_id]
            with open(os.path.join(meta_path, f'{sample_name}.pickle'), 'wb') as file:
                pickle.dump(content, file)

    @staticmethod
    def parse_global_bbox_velo(meta_dict, data_path, meta_path):
        for s, sdict in meta_dict.items():
            for f, fdict in sdict.items():
                cur_global_boxes = fdict['meta']['bbx_center_global']
                # cur_global_boxes = {box[0]: box[1:] for box in cur_global_boxes}
                velos = []
                next_frame = f'{int(f) + 1:06d}'
                last_frame = f'{int(f) - 1:06d}'
                next_global_boxes = {}
                prev_global_boxes = {}
                if next_frame in sdict:
                    next_global_boxes = sdict[next_frame]['meta']['bbx_center_global']
                    next_global_boxes = {box[0]: box[1:] for box in next_global_boxes}
                if last_frame in sdict:
                    prev_global_boxes = sdict[last_frame]['meta']['bbx_center_global']
                    prev_global_boxes = {box[0]: box[1:] for box in prev_global_boxes}

                for box_ in cur_global_boxes:
                    box_id = box_[0]
                    box = box_[1:]
                    if box_id in next_global_boxes:
                        velo = [(next_global_boxes[box_id][1] - box[1]) * 10,  # m/s
                                (next_global_boxes[box_id][2] - box[2]) * 10,]
                    elif box_id in prev_global_boxes:
                        velo = [(box[1] - prev_global_boxes[box_id][1]) * 10,
                                (box[2] - prev_global_boxes[box_id][2]) * 10]
                    else:
                        velo = [0., 0.]

                    velos.append(velo)
                fdict['meta']['bbx_velo_global'] = velos

            save_json(sdict, os.path.join(meta_path, f'{s}.json'))


    @staticmethod
    def draw_sample_distributions(meta_path):
        """
        Draw distribution of the number of observation points for each sample category.

        :param meta_path: path contains pickle files of object samples
        :return:
        """
        import matplotlib.pyplot as plt
        files = glob.glob(os.path.join(meta_path, '*.pickle'))
        for f in files:
            with open(f, 'rb') as file:
                samples = pickle.load(file)
            n_points = np.array([min(len(points), 500) for points in samples['points']])
            plt.hist(n_points, bins=10, density=True, alpha=0.6, label=os.path.basename(f)[:-7])
            plt.title(os.path.basename(f)[:-7])
            # plt.legend()
            plt.savefig(os.path.join(meta_path, f'{os.path.basename(f)[:-7]}.png'))
            plt.close()



if __name__=="__main__":
    cosense3d = CoSenseDataConverter(
        "/koko/LUMPI/lumpi_selected/data",
        "/koko/LUMPI/lumpi_selected/meta",
        'all'
    )
    # cosense3d.to_kitti("/koko/LUMPI/kitti_test")
    # cosense3d.to_sustech("/koko/LUMPI/lumpi_selected_sustech")
    # cosense3d.to_opv2v("/media/hdd/yuan/koko/data/LUMPI/opv2v_fmt")
    # cosense3d.update_from_sustech("/koko/LUMPI/sustech_fmt")
    # cosense.supervison_full_to_sparse(cosense.meta,
    #                                   '/koko/cosense3d/kitti-sparse-num534',
    #                                   lidar_range=[-100, -40, -3.5, 100, 40, 3],
    #                                   num_box_total=534)
    # cosense.global_boxes_to_local(cosense.meta, cosense.data_path, cosense.meta_path)
    # cosense.update_from_sustech('/koko/LUMPI/sustech_fmt')
    # cosense.parse_global_bbox_velo(cosense.meta, cosense.data_path, cosense.meta_path)
    # cosense.draw_sample_distributions(cosense.meta_path)
