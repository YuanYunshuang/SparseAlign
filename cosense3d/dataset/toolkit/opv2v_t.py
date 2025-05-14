import glob
import os.path
import random

import numpy as np
import torch
from plyfile import PlyData
from matplotlib import colormaps
from multiprocessing import Pool
import torch_scatter
from functools import partial

from cosense3d.dataset.toolkit.opv2v import *
from cosense3d.utils.vislib import o3d_draw_pcds_bbxs
from cosense3d.utils.pclib import save_cosense_ply, pose_to_transformation
from cosense3d.utils.box_utils import transform_boxes_3d
from cosense3d.utils.misc import load_json, update_dict
from cosense3d.ops.utils import points_in_boxes_gpu
from cosense3d.modules.utils.common import cat_coor_with_idx


# jet = cm.get_cmap('jet')
jet = colormaps['jet']


def read_ply(filename, properties=None):
    ply = PlyData.read(filename)
    data = ply['vertex']
    properties_from_file = [p.name for p in ply.elements[0].properties]
    if properties is None:
        properties = properties_from_file
    else:
        for p in properties:
            assert p in properties_from_file, f"Property '{p}' not found."
    data_dict = {}
    for p in properties:
        data_dict[p] = np.array(data[p])

    return data_dict


def get_local_boxes3d(objects_dict, ref_pose, order):
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
            # TODO adapt velos
        else:
            velos.append([0., 0.])

    return boxes_local, velos


def read_ply_to_dict(f):
    data = read_ply(f)
    timestamp = os.path.basename(f).split('.')[:-1]
    timestamp = int(timestamp[0]) * 0.05 + int(timestamp[1]) * 0.01
    timestamp = np.ones_like(data['x']) * timestamp
    data['time'] = timestamp.astype(np.float32)
    return data


def read_sub_frame(f):
    pcd_dict = read_ply_to_dict(f + '.ply')
    params = load_yaml(f + '_objects.yaml', cloader=True)
    # params = load_yaml(f + '.yaml')
    # update_dict(params, params_)
    gt_boxes, velos = get_local_boxes3d(params['vehicles'],
                                        params['lidar_pose'], 'lwh')
    gt_boxes = np.array(gt_boxes)
    # velos = np.array(velos)
    points = np.stack([pcd_dict[x] for x in 'xyz'], axis=-1)
    points = points[pcd_dict['ObjTag'] == 10]
    return gt_boxes, pcd_dict, points


def get_box_velo(box, speeds, frame):
    box_id = str(int(box[0]))
    try:
        speed = speeds[box_id][frame]
    except:
        if box_id not in speeds:
            speed = 0.0
        elif frame not in speeds[box_id]:
            frames = list(speeds[box_id].keys())
            nearst_frame_idx = (np.array(frames).astype(int) - int(frame)).argmax()
            speed = speeds[box_id][frames[nearst_frame_idx]]
        else:
            raise NotImplementedError
    return speed


def get_velos(boxes, speeds, frame):
    with Pool(16) as pool:
        out_speeds = pool.map(
            partial(get_box_velo, speeds=speeds, frame=frame),
            boxes
        )
    out_speeds = np.array(out_speeds)

    theta = boxes[:, -1]
    velos = np.stack([out_speeds * np.cos(theta),
                      out_speeds * np.sin(theta)], axis=-1)
    return velos


def pad_box_result(res, out_len):
    if len(res[0]) == out_len:
        return res
    box = np.zeros((out_len,) + res[0].shape[1:], dtype=res[0].dtype)
    box[:res[0].shape[0]] = res[0]
    # set id index to -1 to indicate it is padded
    box[res[0].shape[0]:, 0] = -1
    box[res[0].shape[0]:, 4] = 100
    return box, res[1], res[2]


def parse_sub_frame(f):
    pcd_dict = read_ply_to_dict(f + '.ply')
    params = load_yaml(f + '.yaml')
    gt_boxes, velos = get_local_boxes3d(params['vehicles'],
                                        params['lidar_pose'], 'lwh')
    gt_boxes = np.array(gt_boxes)
    velos = np.array(velos)
    points = np.stack([pcd_dict[x] for x in 'xyz'], axis=-1)
    pts_mask = points_in_boxes_cpu(torch.from_numpy(points),
                                   torch.from_numpy(gt_boxes)[:, [2, 3, 4, 5, 6, 7, 10]])
    num_pts = pts_mask.sum(dim=-1).numpy()
    box_mask = num_pts > 0
    gt_boxes = gt_boxes[box_mask].tolist()
    velos = velos[box_mask].tolist()
    num_pts = num_pts[box_mask].tolist()

    # update boxes dict
    # for i, box in enumerate(gt_boxes):
    #     id = int(box[0])
    #     if id not in boxes:
    #         boxes[id] = {
    #             'box': box,
    #             'velo': velos[i],
    #             'num_pts': num_pts[i]
    #         }
    #     else:
    #         if boxes[id]['num_pts'] < num_pts[i]:
    #             boxes[id] = {
    #                 'box': box,
    #                 'velo': velos[i],
    #                 'num_pts': num_pts[i] + boxes[id]['num_pts']
    #             }
    #         else:
    #             boxes[id]['num_pts'] += num_pts[i]

    return (gt_boxes, velos, num_pts, pcd_dict)


def read_frame_plys_boxes(path, frame, prev_frame=None, time_offset=0, parse_boxes=True):
    data_list = []
    files = []
    if prev_frame is not None:
        files_prev_frame = [f'{prev_frame}.{i}' for i in range(10 - time_offset, 10)]
        files.extend(files_prev_frame)
    files_cur_frame = [f'{frame}.{i}' for i in range(0, 10 - time_offset)]
    files.extend(files_cur_frame)
    files = [os.path.join(path, f) for f in files]
    boxes = {}

    with Pool(10) as pool:
        res = pool.map(read_sub_frame, files)
        max_len = max([len(x[0]) for x in res])
        res = pool.starmap(pad_box_result, zip(res, [max_len] * len(res)))

    pcd_dict = {k: np.concatenate([d[1][k] for d in res], axis=0) for k in res[0][1]}
    boxes_tensor = cat_coor_with_idx([torch.from_numpy(x[0]) for x in res]).float()
    points_tensor = cat_coor_with_idx([torch.from_numpy(x[2]) for x in res]).float()

    _, pts_idx_of_box = points_in_boxes_gpu(points_tensor.cuda(),
                                            boxes_tensor[:, [0, 3, 4, 5, 6, 7, 8, 11]].cuda(),
                                            batch_size=len(res))

    pts_idx_of_box = pts_idx_of_box[pts_idx_of_box >= 0]
    cnt = torch.ones_like(pts_idx_of_box)
    num_pts_in_box = cnt.new_zeros(len(boxes_tensor))
    torch_scatter.scatter_add(cnt, pts_idx_of_box, out=num_pts_in_box, dim=0)
    num_pts_in_box = num_pts_in_box.reshape(10, -1).cpu()
    num_pts = num_pts_in_box.sum(dim=0)
    boxes_tensor = boxes_tensor.view(10, -1, boxes_tensor.shape[-1])[..., 1:]
    max_inds = num_pts_in_box.max(dim=0).indices
    boxes_selected = boxes_tensor[max_inds, torch.arange(len(max_inds))].numpy()
    boxes_selected = boxes_selected[boxes_selected[:, 0] >= 0]

    # o3d_draw_pcds_bbxs([points_tensor[:, 1:].numpy()], [boxes_selected])

    return pcd_dict, boxes_selected, num_pts


def load_frame_data(scene_dir, cavs, frame):
    ego_id = cavs[0]
    yaml_file = os.path.join(scene_dir, ego_id, f'{frame}.5.yaml')
    meta = load_yaml(yaml_file)
    gt_boxes, velos = get_local_boxes3d(meta['vehicles'],
                                        meta['lidar_pose'], 'lwh')
    ego_pose = meta['lidar_pose']

    points_list = []
    time_list = []
    for cav in cavs:
        cav_dir = os.path.join(scene_dir, cav)
        data = read_frame_plys_boxes(cav_dir, frame, parse_boxes=False)[0]
        points = np.stack([data[k] for k in 'xyz'], axis=-1)
        times = (data['time'] - data['time'].min()) * 10
        lidar_pose = load_yaml(
            os.path.join(scene_dir, cav, f'{frame}.5.yaml'))['lidar_pose']
        transform = x1_to_x2(lidar_pose, ego_pose)
        points = (transform[:3, :3] @ points.T + transform[:3, 3].reshape(3, 1)).T
        points_list.append(points)
        time_list.append(times)
    points = np.concatenate(points_list, axis=0)
    times = np.concatenate(time_list, axis=0)
    return points, times, gt_boxes, velos


def opv2vt_to_cosense(data_dir, split, data_out_dir, meta_out_dir):
    order = 'lwh'
    time_offsets = load_json(os.path.join(data_out_dir, 'time_offsets.json'))
    split_dir = os.path.join(data_dir, split)
    scenes = sorted(os.listdir(split_dir))[:2]
    with open(os.path.join(meta_out_dir, f'{split}.txt'), 'w') as fh:
        fh.write('\n'.join(scenes))
    for s in scenes:
        print(s)
        scene_dir = os.path.join(split_dir, s)
        sdict = {}
        cavs = sorted([x for x in os.listdir(scene_dir)
                       if os.path.isdir(os.path.join(scene_dir, x))])
        if os.path.exists(os.path.join(scene_dir, 'speeds.json')):
            speeds = load_json(os.path.join(scene_dir, 'speeds.json'))
        else:
            speeds = parse_speed_from_yamls(scene_dir)
        ego_id = cavs[0]
        frames = sorted([x.split(".")[0] for x in os.listdir(
            os.path.join(scene_dir, cavs[0])) if '.0.ply' in x])
        for i, f in tqdm.tqdm(enumerate(frames[1:-1])):
            frame_mid_time = int(f) * 0.05 + 0.05
            fdict = cs.fdict_template()
            ego_lidar_pose = None
            object_id_stack = []
            object_velo_stack = []
            object_stack = []
            for j, cav_id in enumerate(cavs):
                cur_data_out_dir = os.path.join(data_out_dir, split, s, cav_id)
                os.makedirs(cur_data_out_dir, exist_ok=True)
                yaml_file = os.path.join(scene_dir, cav_id, f'{f}.5.yaml')
                params = load_yaml(yaml_file, cloader=True)
                cs.update_agent(fdict, cav_id, agent_type='cav', agent_pose=params['true_ego_pos'])
                # update_cam_params(params, fdict, cav_id, s, f)

                if cav_id == ego_id:
                    ego_lidar_pose = params['lidar_pose']

                # get cav lidar pose in cosense format
                cs.update_agent(fdict, cav_id, 'cav')
                cs.update_agent_lidar(fdict, cav_id, 0,
                                      lidar_pose=opv2v_pose_to_cosense(params['lidar_pose']),
                                      lidar_file=os.path.join(s, cav_id, f'{f}.ply'))
                # save lidar files
                data, local_boxes, num_pts = read_frame_plys_boxes(os.path.join(scene_dir, cav_id), f,
                                       prev_frame=frames[i], time_offset=time_offsets[s][cav_id])
                velos = get_velos(local_boxes, speeds, f)
                # save_cosense_ply(data, os.path.join(cur_data_out_dir, f'{f}.ply'))

                objects_dict = params.get('vehicles', {})
                output_dict = {}
                glob_ref_pose = ego_lidar_pose
                local_ref_pose = params['lidar_pose']

                 # update_local_boxes
                cs.update_agent(fdict, cav_id, gt_boxes=local_boxes.tolist())
                cs.update_agent(fdict, cav_id, velos=velos.tolist())
                cs.update_agent(fdict, cav_id, num_pts=num_pts.tolist())
                # update_2d_bboxes(fdict, cav_id, params['lidar_pose'], data_dir)

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
            if len(object_velo_stack) > 0:
                object_velo_stack = np.vstack(object_velo_stack)
                object_velo_stack = object_velo_stack[unique_indices]
            if order == 'hwl':
                object_stack = object_stack[:, [0, 1, 2, 5, 4, 3, 6]]

            cosense_bbx_center = np.zeros((len(object_stack), 11))
            cosense_bbx_center[:, 0] = np.array(object_id_stack)[unique_indices]
            cosense_bbx_center[:, 2:8] = object_stack[:, :6]
            cosense_bbx_center[:, 10] = object_stack[:, 6]
            cs.update_frame_bbx(fdict, cosense_bbx_center.tolist())
            fdict['agents'].pop(0)  # remove template agent

            fdict['meta']['ego_id'] = ego_id
            fdict['meta']['ego_lidar_pose'] = opv2v_pose_to_cosense(ego_lidar_pose)
            fdict['meta']['global_bbox_time'] = np.full(len(cosense_bbx_center), frame_mid_time).tolist()
            fdict['meta']['bbx_velo_global'] = get_velos(cosense_bbx_center, speeds, f).tolist()

            sdict[f] = fdict

        save_json(sdict, os.path.join(meta_out_dir, f'{s}.json'))
        del sdict


def vis_frame_data():
    scene_dir = "/koko/OPV2V/temporal_dump/test/2021_08_18_19_48_05"
    cavs = sorted([x for x in os.listdir(scene_dir)
                   if os.path.isdir(os.path.join(scene_dir, x))])
    frames = sorted([x.split(".")[0] for x in os.listdir(
        os.path.join(scene_dir, cavs[0])) if '.0.ply' in x])
    for f in frames[::10]:
        points, times, local_boxes, velos = load_frame_data(scene_dir, cavs, f)
        pcd = o3d.geometry.PointCloud()
        # color_inds = np.round(times).astype(int)
        colors = jet(times)[:, :3]
        o3d_draw_pcds_bbxs([points], [np.array(local_boxes)],
                           pcds_colors=[colors])


def parse_speed_from_yamls(scene_dir):
    cavs = sorted([x for x in os.listdir(scene_dir)
                   if os.path.isdir(os.path.join(scene_dir, x))])
    vehicle_dict = {}
    for cav in cavs:
        cav_dir = os.path.join(scene_dir, cav)
        yamls = sorted(glob(os.path.join(cav_dir, '*5_objects.yaml')))
        for yaml in tqdm.tqdm(yamls):
            frame = int(os.path.basename(yaml).split('.')[0])
            params = load_yaml(yaml, cloader=True)
            for k, v in params['vehicles'].items():
                if k not in vehicle_dict:
                    vehicle_dict[k] = {'frames': [], 'locations': []}
                if frame not in vehicle_dict[k]['frames']:
                    vehicle_dict[k]['frames'].append(frame)
                    vehicle_dict[k]['locations'].append(v['location'])

    # vehicle_dict = load_json(os.path.join(scene_dir, 'vehicles.json'))
    velo_dict = {}
    for veh_id, veh_info in vehicle_dict.items():
        times = np.array(veh_info['frames']) * 0.05
        sort_inds = np.argsort(times)
        times = times[sort_inds]
        locations = np.array(veh_info['locations'])
        locations = locations[sort_inds]
        time_offsets = times[1:] - times[:-1]
        interp_inds = np.where(time_offsets > 0.15)[0]
        loc_offsets = np.linalg.norm(locations[1:] - locations[:-1], axis=-1)
        speeds = loc_offsets / time_offsets

        # interpolate missed frames
        speeds_interp = []
        times_interp = []
        last_idx = 0
        for idx in interp_inds:
            speeds_interp.extend(speeds[last_idx:idx])
            times_interp.extend(times[last_idx:idx])
            steps = int(round(time_offsets[idx] * 10))
            if idx == 0:
                interp_s = [speeds[0]] * (steps - 1)
                interp_t = [times[0]] * (steps - 1)
            else:
                interp_s = np.linspace(speeds[idx-1], speeds[idx], steps + 1)[1:-1].tolist()
                interp_t = np.linspace(times[idx-1], times[idx], steps + 1)[1:-1].tolist()
            speeds_interp.extend(interp_s)
            times_interp.extend(interp_t)
            last_idx = idx
        speeds_interp.extend(speeds[last_idx:])
        times_interp.extend(times[last_idx:])

        velo_dict[veh_id] = {f'{round(t*20):06d}': speed for t, speed in zip(times_interp, speeds_interp)}
    save_json(velo_dict, os.path.join(scene_dir, 'speeds.json'))
    return velo_dict


def update_velo(scenario_meta_file):
    meta = load_json(scenario_meta_file)
    frames = sorted(list(meta.keys()))
    objects = {}

    # find all global objects
    for f in frames:
        fdict = meta[f]
        boxes = fdict['meta']['bbx_center_global']
        for box in boxes:
            box_id = int(box[0])
            if box_id not in objects:
                objects[box_id] = {'frames': [], 'box': []}
            objects[box_id]['frames'].append(int(f))
            objects[box_id]['boxes'].append(box)

    def cal_velos(cur_gt_boxes, next_gt_boxes, cur_pose, next_pose, meta_last):
        cur_gt_boxes_dict = {int(box[0]): box for box in cur_gt_boxes}
        next_gt_boxes_np = np.array(next_gt_boxes)
        cur_pose = pose_to_transformation(cur_pose)
        next_pose = pose_to_transformation(next_pose)
        transf_next_to_cur = np.linalg.inv(cur_pose) @ next_pose
        next_gt_boxes_np = transform_boxes_3d(next_gt_boxes_np, transf_next_to_cur)
        next_gt_boxes_dict = {int(box[0]): box.tolist() for box in next_gt_boxes_np}
        velos = {}
        for k, v in cur_gt_boxes_dict.items():
            if k not in next_gt_boxes_dict:
                if k in meta_last:
                    velos[k] = meta_last[k]
                else:
                    velos[k] = [0, 0]
                continue
            velo = [(next_gt_boxes_dict[k][2] - v[2]) * 10, (next_gt_boxes_dict[k][3] - v[3]) * 10]  # m/s
            velos[k] = velo
        velos = [velos[int(box[0])] for box in cur_gt_boxes]
        return velos

    for i, f in enumerate(frames[:-1]):
        fdict = meta[f]
        global_ids = sorted([int(box[0]) for box in fdict['meta']['bbx_center_global']])
        global_ids = set(global_ids)
        local_ids = []
        for a, adict in fdict['agents'].items():
            local_ids.extend([int(box[0]) for box in adict['gt_boxes']])
        local_ids = set(local_ids)
        next_fdict = meta[frames[i + 1]]
        last_fdict = meta[frames[max(i-1, 0)]]

        if i == 0:
            meta_last = {}
        else:
            meta_last = {int(box[0]): last_fdict['meta']['bbx_velo_global'][i] \
                         for i, box in enumerate(last_fdict['meta']['bbx_center_global'])}
        meta[f]['meta']['bbx_velo_global'] = cal_velos(
            fdict['meta']['bbx_center_global'],
            next_fdict['meta']['bbx_center_global'],
            fdict['meta']['ego_lidar_pose'],
            next_fdict['meta']['ego_lidar_pose'],
            meta_last
        )
        for a, adict in fdict['agents'].items():
            if i == 0:
                meta_last = {}
            else:
                meta_last = {int(box[0]): last_fdict['agents'][a]['velos'][i] \
                             for i, box in enumerate(last_fdict['agents'][a]['gt_boxes'])}
            velos = cal_velos(
                adict['gt_boxes'], next_fdict['agents'][a]['gt_boxes'],
                adict['lidar']['0']['pose'], next_fdict['agents'][a]['lidar']['0']['pose'],
                meta_last
            )
            meta[f]['agents'][a]['velos'] = velos
    save_json(meta, scenario_meta_file)


def vis_cosense_scenario(scenario_meta_file, data_dir):
    meta = load_json(scenario_meta_file)
    for f, fdict in meta.items():
        global_boxes = np.array(fdict['meta']['bbx_center_global'])
        for a, adict in fdict['agents'].items():
            lidar_file = os.path.join(data_dir, adict['lidar']['0']['filename'])
            pcd_dict = read_ply(lidar_file)
            points = np.stack([pcd_dict[x] for x in 'xyz'], axis=-1)
            boxes = np.array(adict['gt_boxes'])

            o3d_draw_pcds_bbxs([points], [boxes, global_boxes],
                               bbxs_colors=[[0, 255, 0], [255, 0, 0]])


def gen_time_offsets(data_dir):
    out_dict = {}
    for split in ['train', 'test']:
        split_dir = os.path.join(data_dir, split)
        scenes = os.listdir(split_dir)
        for s in scenes:
            out_dict[s] = {}
            scene_dir = os.path.join(split_dir, s)
            cavs = sorted([x for x in os.listdir(scene_dir)
                           if os.path.isdir(os.path.join(scene_dir, x))])
            for i, cav in enumerate(cavs):
                if i == 0:
                    out_dict[s][cav] = 0
                else:
                    out_dict[s][cav] = random.randint(0, 5)
    save_json(out_dict, os.path.join(data_dir, f'time_offsets.json'))


def load_vehicles_gframe(params):
    """Load vehicles in global coordinate system."""
    object_dict = params['vehicles']
    object_out = {}
    for object_id, object_content in object_dict.items():
        location = object_content['location']
        rotation = object_content['angle']
        center = object_content['center']
        extent = object_content['extent']

        object_pose = [location[0] + center[0],
                       location[1] + center[1],
                       location[2] + center[2],
                       rotation[0], rotation[1], rotation[2]]

        object_out[object_id] = [0,] + object_pose[:3] + extent + object_pose[3:]
    return object_out


def transform_boxes_global_to_ref(boxes, ref_pose):
    pass


def update_global_boxes(root_dir, meta_in, meta_out, split):
    split_dir = os.path.join(root_dir, split)
    scenes = os.listdir(split_dir)
    for s in scenes:
        scene_dir = os.path.join(split_dir, s)
        sdict = load_json(os.path.join(meta_in, f"{s}.json"))
        cavs = sorted([x for x in os.listdir(scene_dir) if os.path.isdir(os.path.join(scene_dir, x))])

        ego_files = sorted(glob(os.path.join(scene_dir, cavs[0], '*.0_objects.yaml')))
        sim_frames = [os.path.basename(x)[:6] for x in ego_files]
        global_objects = {x: {} for x in sim_frames}
        ego_poses = {}

        for cav in cavs[1:]:
            yaml_files = sorted(glob(os.path.join(scene_dir, cav, '*.0_objects.yaml')))
            for yf in yaml_files:
                frame = os.path.basename(yf)[:6]
                objects = load_yaml(yf)['vehicles']
                global_objects[frame].update(objects)
        for yf in ego_files:
            frame = os.path.basename(yf)[:6]
            params = load_yaml(yf)
            ego_poses[frame] = params['lidar_pose']
            global_objects[frame].update(params['vehicles'])

        frames = sorted(list(sdict.keys()))
        for f in frames[:-1]:
            lidar_pose = ego_poses[f]
            sdict[f]['meta']['boxes_pred'] = {}
            box_ids = [int(box[0]) for box in sdict[f]['meta']['bbx_center_global']]
            for i in range(1, 3):
                cur_frame = f"{int(f) + i * 2:06d}"
                boxes_global = global_objects[cur_frame]
                boxes_ref = {}
                project_world_objects(boxes_global, boxes_ref, lidar_pose, 'lwh')
                boxes_pred = []
                for box_id in box_ids:
                    if box_id in boxes_global:
                        pred = boxes_ref[box_id]['coord'].reshape(7)[[0, 1, 2, 6]].tolist()
                    else:
                        pred = [0,] * 4
                    boxes_pred.append(pred)
                sdict[f]['meta']['boxes_pred'][cur_frame] = boxes_pred
        sdict.pop(frames[-1])
        save_json(sdict, os.path.join(meta_out, f"{s}.json"))


def update_bev_map(root_dir, meta_in, meta_out, split):
    from cosense3d.dataset.const import OPV2V_TOWN_DICTIONARY
    resolution = 0.2
    pixels_per_meter = 1 / resolution
    radius = 100
    map_bounds = load_json(f'../../carla/assets/map_bounds.json')
    split_dir = os.path.join(root_dir, split)
    scenes = os.listdir(split_dir)[3:]
    x = np.linspace(- radius + 0.5 * resolution, radius,
                    int(radius * 2 / resolution) - 1)
    bev_points = np.stack(np.meshgrid(x, x), axis=0)
    bev_points = np.r_[bev_points, [np.zeros(bev_points.shape[1:]),
                                    np.ones(bev_points.shape[1:])]].reshape(4, -1)

    for s in scenes:
        town = OPV2V_TOWN_DICTIONARY[s]
        bev_map = cv2.imread(f'../../carla/assets/maps/{town}.png')
        sx, sy, _ = bev_map.shape
        map_bound = map_bounds[town]
        scene_dir = os.path.join(split_dir, s)
        sdict = load_json(os.path.join(meta_in, f"{s}.json"))
        for f, fdict in sdict.items():
            adict = fdict['agents'][fdict['meta']['ego_id']]
            lidar_pose = adict['lidar']['0']['pose']
            lidar_file = os.path.join(split_dir, adict['lidar']['0']['filename'])
            pcd = load_pcd(lidar_file)['xyz']
            transform = pose_to_transformation(lidar_pose)
            cords = np.dot(transform, bev_points).T
            xs = np.floor((cords[:, 0] - map_bound[0]) * pixels_per_meter).astype(int)
            ys = np.floor((cords[:, 1] - map_bound[1]) * pixels_per_meter).astype(int)
            xs = np.maximum(np.minimum(xs, sx - 1), 0)
            ys = np.maximum(np.minimum(ys, sy - 1), 0)
            road_mask = bev_map[xs, ys] / 255.
            mask = road_mask[:, :2].any(axis=1)

            import matplotlib.pyplot as plt
            plt.plot(bev_points[0][mask], bev_points[1][mask], '.g')
            plt.plot(pcd[:, 0], pcd[:, 1], '.r', markersize=1)
            plt.show()
            plt.close()
            break


def generate_roadline_reference_points(root_dir, meta_file):
    assets_path = f"{os.path.dirname(__file__)}/../../carla/assets"
    map_path = f"{assets_path}/maps/png"
    roadline_path = f"{assets_path}/maps/roadline"
    map_files = glob(os.path.join(map_path, '*.png'))
    map_bounds = load_json(os.path.join(assets_path, 'map_bounds.json'))

    kernel = 3
    map_res = 0.2

    for mf in map_files:
        town = os.path.basename(mf).split('.')[0]
        bound = map_bounds[town]
        bevmap = cv2.imread(mf) / 255.
        bevmap = torch.from_numpy(bevmap[..., :2]).any(dim=-1).float()
        bevmap[bevmap == 0] = -1
        filters = torch.ones(1, 1, kernel, kernel, device=bevmap.device) / (kernel ** 2 * 2)
        road = torch.conv2d(bevmap[None, None], filters).squeeze()
        mask = (road < 0.5) & (road > -0.5)
        inds = torch.where(mask)
        # scores = 1 - road[mask].abs()
        coords = torch.stack(inds).T * map_res + 0.3
        coords[:, 0] = coords[:, 0] + bound[0]
        coords[:, 1] = coords[:, 1] + bound[1]
        coords = coords.numpy().astype(float)
        coords.tofile(os.path.join(roadline_path, f'{town}.bin'))


    # sdict = load_json(meta_file)
    # scene_maps = load_json(os.path.join(assets_path, 'scenario_town_map.json'))
    # scenario = os.path.basename(meta_file).split('.')[0]
    # town = scene_maps[scenario]
    # for fi, fdict in sdict.items():
    #     if int(fi) % 10 != 1:
    #         continue
    #     for ai, adict in fdict['agents'].items():
    #         lidar_pose = adict['lidar']['0']['pose']
    #         lidar_file = os.path.join(root_dir, 'test', adict['lidar']['0']['filename'])
    #         pcd = load_pcd(lidar_file)['xyz']
    #         transform = pose_to_transformation(lidar_pose)
    #         pcd = (transform @ np.concatenate([pcd, np.ones_like(pcd[:, :1])], axis=1).T).T
    #
    #         fig = plt.figure(figsize=(16, 12))
    #         ax = fig.add_subplot()
    #         ax.plot(coords[:, 0], coords[:, 1], '.g', markersize=1)
    #         ax.scatter(pcd[:, 0], pcd[:, 1], s=1, c=np.clip(pcd[:, 2], a_min=-3, a_max=1), cmap='jet')
    #         plt.savefig("/home/yys/Downloads/tmp.jpg")
    #         plt.close()
    #         continue


def load_range_img(pcd_file, colors):
    N = 1024
    pcd_dict = read_ply(pcd_file)
    angles = ((np.arctan2(pcd_dict['y'], pcd_dict['x']) / np.pi) + 1) * (N / 2)
    angles = angles.astype(int)
    depth = np.sqrt(pcd_dict['x']**2 + pcd_dict['y']**2)
    # depth = depth / depth.max()
    times = ((pcd_dict['time'] - pcd_dict['time'].min()) * 100).astype(int)
    rings = pcd_dict['ring']
    depth_img = np.zeros((64, N))
    depth_img[rings, angles] = depth
    depth_img = cv2.Sobel(depth_img, cv2.CV_64F, 1, 0, ksize=5, scale=1, delta=0)
    depth_img = (np.abs(depth_img) > 200) * 255
    # depth_img = (depth_img - depth_img.min()) / (depth_img.max() - depth_img.min()) * 255
    depth_img = depth_img.astype(np.uint8)[..., None].repeat(3, axis=-1)
    inst_img = np.zeros((64, N, 3), dtype=np.uint8)
    for i, id in enumerate(np.unique(pcd_dict['ObjIdx'])):
        RGB = colors[id]
        mask = pcd_dict['ObjIdx'] == id
        inst_img[rings[mask], angles[mask]] = RGB.reshape(1, 3).repeat(mask.sum(), 0)
    obj_img = np.zeros((64, N, 3), dtype=np.uint8)
    for i, tag in enumerate(np.unique(pcd_dict['ObjTag'])):
        RGB = colors[tag]
        mask = pcd_dict['ObjTag'] == tag
        obj_img[rings[mask], angles[mask]] = RGB.reshape(1, 3).repeat(mask.sum(), 0)
    img = np.concatenate([depth_img, inst_img, obj_img], axis=0)
    return img


if __name__=="__main__":
    colors = np.random.randint(0, 255, (1024, 3))
    img1 = load_range_img("/home/data/OPV2V/temporal/train/2021_08_16_22_26_54/650/000071.ply", colors)
    img2 = load_range_img("/home/data/OPV2V/temporal/train/2021_08_16_22_26_54/641/000071.ply", colors)
    img3 = load_range_img("/home/data/OPV2V/temporal/train/2021_08_16_22_26_54/659/000071.ply", colors)

    cv2.imshow("depth", np.concatenate([img1, img2, img3], axis=0))
    cv2.waitKey()
    cv2.destroyAllWindows()
    # generate_roadline_reference_points(
    #     "/home/data/OPV2Va",
    #     "/home/data/OPV2Va/meta/2021_08_23_17_22_47.json"
    # )

    # gen_time_offsets("/media/yuan/luna/data/OPV2Vt")
    # parse_speed_from_yamls("/home/data/OPV2V/temporal_dump/train/2021_08_16_22_26_54")
    # opv2vt_to_cosense(
    #     "/media/yuan/luna/data/OPV2Vt/temporal_dump",
    #     "train",
    #     "/koko/OPV2V/temporal",
    #     "/koko/cosense3d/opv2v_temporal"
    # )
    # opv2vt_to_cosense(
    #     "/home/data/OPV2V/temporal_dump",
    #     "test",
    #     "/home/data/OPV2V/temporal",
    #     "/home/data/cosense3d/opv2v_temporal"
    # )
    # vis_frame_data()
    # vis_cosense_scenario(
    #     "/home/data/cosense3d/opv2v_temporal/2021_08_16_22_26_54.json",
    #     "/home/data/OPV2V/temporal/train"
    # )
    # update_velo(
    #     "/media/yuan/luna/data/OPV2Vt/meta/2021_08_16_22_26_54.json",
    # )
    # update_bev_map(
    #     "/koko/OPV2V/temporal",
    #     "/koko/cosense3d/opv2vt",
    #     "/koko/cosense3d/opv2vt_bev",
    #     "train"
    # )



