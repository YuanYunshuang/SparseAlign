import copy
import glob
import math
import os

import matplotlib.pyplot as plt
import tqdm
import numpy as np
import open3d as o3d
from scipy.optimize import linear_sum_assignment

from cosense3d.utils import pclib, vislib, box_utils
from cosense3d.utils.misc import load_json, save_json
from cosense3d.utils.box_utils import corners_to_boxes_3d, transform_boxes_3d
from cosense3d.dataset.toolkit import register_pcds
from cosense3d.dataset.toolkit.cosense import CoSenseDataConverter as cs
from cosense3d.ops.utils import points_in_boxes_cpu
from cosense3d.utils.pcdio import point_cloud_from_path
from cosense3d.utils.vislib import o3d_draw_frame_data, \
    o3d_draw_agent_data, o3d_draw_pcds_bbxs

global_time_offset = 1.62616 * 1e9


def calib_to_tf_matrix(calib_file, delta_xy=None):
    calib = load_json(calib_file)
    if 'transform' in calib:
        tf = calib['transform']
    else:
        tf = calib
    tf_matrix = np.eye(4)
    tf_matrix[:3, :3] = np.array(tf['rotation'])
    tf_matrix[:3, 3:] = np.array(tf['translation'])
    if delta_xy is not None:
        tf_matrix[0, 3] += delta_xy['delta_x']
        tf_matrix[1, 3] += delta_xy['delta_y']
    elif 'relative_error' in calib:
        tf_matrix[0, 3] += calib['relative_error']['delta_x']
        tf_matrix[1, 3] += calib['relative_error']['delta_y']
    return tf_matrix


def load_label(label_file):
    labels = load_json(label_file)
    bbxs_center = []
    bbxs_corner = []
    for l in labels:
        obj_type = {
            'car': 'vehicle.car',
            'van': 'vehicle.van',
            'truck': 'vehicle.truck',
            'bus': 'vehicle.bus',
            'pedestrian': 'human.pedestrian',
            'trafficcone': 'static.trafficcone',
            'motorcyclist': 'vehicle.motorcycle',
            'cyclist': 'vehicle.cyclist',
            'tricyclist': 'vehicle.tricyclist',
            'barrowlist': 'static.barrowlist',
        }[l.get('type', "car").lower()]
        track_id = l.get('track_id', -1)
        bbx = [
            int(track_id),
            cs.OBJ_NAME2ID[obj_type],
            l['3d_location']['x'],
            l['3d_location']['y'],
            l['3d_location']['z'],
            l['3d_dimensions']['l'],
            l['3d_dimensions']['w'],
            l['3d_dimensions']['h'],
            0,
            0,
            l['rotation']
        ]
        bbxs_center.append([float(x) for x in bbx])
        if 'world_8_points' in l:
            bbx_corner = np.array(l['world_8_points'])
            bbx_corner = [bbx.tolist() for bbx in bbx_corner]
            bbxs_corner.append(bbx_corner)
    return bbxs_center, bbxs_corner


def load_info_to_dict(info_file):
    infos = load_json(info_file)
    info_dict = {}
    for info in infos:
        frame = os.path.basename(info['pointcloud_path'][:-4])
        info_dict[frame] = info
    return info_dict


def convert_v2x_c(root_dir, meta_out_dir):
    cvi_path = "cooperative-vehicle-infrastructure"
    infra_path = "infrastructure-side"
    cav_path = "vehicle-side"
    coop_path = "cooperative"
    info_file = "data_info.json"
    inf_lidar_path = "cooperative-vehicle-infrastructure-infrastructure-side-velodyne"
    cav_lidar_path = "cooperative-vehicle-infrastructure-vehicle-side-velodyne"
    new_label_path = "DAIR-V2X-C_Complemented_Anno"
    inf_info_file = os.path.join(root_dir, cvi_path, infra_path, info_file)
    inf_info = load_info_to_dict(inf_info_file)
    veh_info_file = os.path.join(root_dir, cvi_path, cav_path, info_file)
    veh_info = load_info_to_dict(veh_info_file)
    frame_pairs = load_json(os.path.join(root_dir, cvi_path, coop_path, info_file))

    meta_dict = {}
    veh_frames = []
    inf_frames = []
    offsets = []
    for pair in frame_pairs:
        veh_frame = os.path.basename(pair['vehicle_pointcloud_path'][:-4])
        inf_frame = os.path.basename(pair['infrastructure_pointcloud_path'][:-4])
        label_frame = os.path.basename(pair['cooperative_label_path'][:-5])
        assert veh_frame == label_frame
        veh_frames.append(veh_frame)
        inf_frames.append(inf_frame)
        offsets.append(pair['system_error_offset'])

    # load all re-annotated samples
    train = load_json(os.path.join(root_dir, new_label_path, 'train.json'))
    val = load_json(os.path.join(root_dir, new_label_path, 'val.json'))
    split = {
                'train': train,
                 'test': val
             }

    for sp, frames in split.items():
        for frame in tqdm.tqdm(frames):
            cur_veh_info = veh_info[frame]
            scenario = cur_veh_info['batch_id']
            # processing vehicle meta
            tf_novatel2world = calib_to_tf_matrix(
                os.path.join(root_dir, cvi_path, cav_path,
                             cur_veh_info['calib_novatel_to_world_path'])
            )
            tf_lidar2novatel = calib_to_tf_matrix(
                os.path.join(root_dir, cvi_path, cav_path,
                             cur_veh_info['calib_lidar_to_novatel_path'])
            )
            tf_lidar2world = tf_novatel2world @ tf_lidar2novatel
            veh_lidar_pose = pclib.tf2pose(tf_lidar2world)
            veh_pose = pclib.tf2pose(tf_novatel2world)
            veh_lidar_time = float(cur_veh_info['pointcloud_timestamp']) * 1e-6
            veh_lidar_file = os.path.join(cav_lidar_path, frame + '.pcd')
            veh_bbxs_center, _ = load_label(
                os.path.join(root_dir,
                             f"{new_label_path}/new_labels/vehicle-side_label/lidar",
                             frame + '.json'
                             )
            )

            # process infra info
            cur_pair_index = veh_frames.index(frame)
            cur_delta_xy = offsets[cur_pair_index]
            cur_inf_frame = inf_frames[cur_pair_index]
            cur_inf_info = inf_info[cur_inf_frame]
            tf_virtuallidar2world = calib_to_tf_matrix(
                os.path.join(root_dir, cvi_path, infra_path,
                             cur_inf_info['calib_virtuallidar_to_world_path']),
                cur_delta_xy
            )

            inf_lidar_time = float(cur_inf_info['pointcloud_timestamp']) * 1e-6
            inf_lidar_file = os.path.join(inf_lidar_path, cur_inf_frame + ".pcd")

            # inf_lidar_pose = pclib.tf2pose(tf_infra2ego)
            inf_lidar_pose = pclib.tf2pose(tf_virtuallidar2world)
            inf_label_path = os.path.join(root_dir,
                             f"{cvi_path}/infrastructure-side/label/virtuallidar",
                             cur_inf_frame + '.json')
            inf_bbxs_center, _ = load_label(inf_label_path)

            # process global meta
            coop_label_path = os.path.join(root_dir,
                             f"{new_label_path}/new_labels/cooperative_label/label_world",
                             frame + '.json'
                             )
            world_bbxs_center, world_bbxs_corner = load_label(coop_label_path)
            coop_bbxs_corner =  pclib.rotate_box_corners_with_tf_np(
                np.array(world_bbxs_corner), np.linalg.inv(tf_lidar2world)
            )
            coop_bbxs_center = np.concatenate(
                [np.array(world_bbxs_center)[:, :2],
                 corners_to_boxes_3d(coop_bbxs_corner)],
                axis=1
            ).tolist()

            # if not os.path.exists(inf_label_path):
            #     print('infra label not found.')
            #     inf_bbxs_center = pclib.rotate_box_corners_with_tf_np(
            #         np.array(world_bbxs_corner), np.linalg.inv(tf_virtuallidar2world)
            #     )
            #     inf_bbxs_center = np.concatenate(
            #         [np.array(world_bbxs_center)[:, :2],
            #          corners_to_boxes_3d(inf_bbxs_center)],
            #         axis=1
            #     ).tolist()



            # pcd = point_cloud_from_path(os.path.join(root_dir, veh_lidar_file))
            # points = np.stack([pcd.pc_data[x] for x in 'xyz'], axis=-1)
            # o3d_draw_pcds_bbxs([points], [np.array(veh_bbxs_center)])

            # construct meta dict
            fdict = cs.fdict_template()
            # add cav lidar meta
            cs.update_agent(fdict,
                            agent_id='0',
                            agent_type='cav',
                            agent_pose=veh_pose,
                            gt_boxes=veh_bbxs_center)
            cs.update_agent_lidar(fdict,
                                  agent_id='0',
                                  lidar_id='0',
                                  lidar_pose=veh_lidar_pose,
                                  lidar_time=veh_lidar_time,
                                  lidar_file=veh_lidar_file)
            # add infra lidar meta
            cs.update_agent(fdict,
                            agent_id='1',
                            agent_type='infra',
                            agent_pose=inf_lidar_pose,
                            gt_boxes=inf_bbxs_center)
            cs.update_agent_lidar(fdict,
                                  agent_id='1',
                                  lidar_id='0',
                                  lidar_pose=inf_lidar_pose,
                                  lidar_time=inf_lidar_time,
                                  lidar_file=inf_lidar_file)
            cs.update_frame_bbx(fdict,
                                coop_bbxs_center
                                )# in global coords
            fdict['meta']['ego_id'] = '0'
            fdict['meta']['ego_lidar_pose'] = veh_lidar_pose
            if scenario not in meta_dict:
                meta_dict[scenario] = {}
            meta_dict[scenario][frame] = fdict
        # save meta
        os.makedirs(meta_out_dir, exist_ok=True)
        for scenario, meta in meta_dict.items():
            meta_file = os.path.join(meta_out_dir, f'{scenario}.json')
            save_json(meta, meta_file)
        with open(os.path.join(meta_out_dir, f'{sp}.txt'), 'w') as fh:
            fh.write('\n'.join(list(meta_dict.keys())))


def convert_v2x_seq(root_dir, meta_out_dir):
    split = "test"
    inf_info_file = os.path.join(root_dir, "infrastructure-side/data_info.json")
    inf_info = load_info_to_dict(inf_info_file)
    veh_info_file = os.path.join(root_dir, "vehicle-side/data_info.json")
    veh_info = load_info_to_dict(veh_info_file)
    frame_pairs = load_json(os.path.join(root_dir, "cooperative/data_info.json"))

    meta_dict = {}
    for pdict in frame_pairs:
        scenario = pdict['vehicle_sequence']
        #############################################################
        # processing vehicle meta
        cur_veh_info = veh_info[pdict['vehicle_frame']]
        tf_novatel2world = calib_to_tf_matrix(
            os.path.join(root_dir, "vehicle-side", cur_veh_info['calib_novatel_to_world_path'])
        )
        tf_lidar2novatel = calib_to_tf_matrix(
            os.path.join(root_dir, "vehicle-side", cur_veh_info['calib_lidar_to_novatel_path'])
        )
        tf_lidar2world = tf_novatel2world @ tf_lidar2novatel
        veh_lidar_pose = pclib.tf2pose(tf_lidar2world)
        veh_pose = pclib.tf2pose(tf_novatel2world)

        veh_lidar_time = float(cur_veh_info['pointcloud_timestamp']) * 1e-6
        veh_lidar_file = os.path.join("vehicle-side", cur_veh_info['pointcloud_path'])
        veh_bbxs_center, _ = load_label(
            os.path.join(root_dir, "vehicle-side", cur_veh_info['label_lidar_std_path'])
        )

        ###############################################################
        # process infra info
        cur_inf_info = inf_info[pdict['infrastructure_frame']]
        tf_virtuallidar2world = calib_to_tf_matrix(
            os.path.join(root_dir, "infrastructure-side", cur_inf_info['calib_virtuallidar_to_world_path'])
        )
        inf_lidar_pose = pclib.tf2pose(tf_virtuallidar2world)
        inf_lidar_time = float(cur_inf_info['pointcloud_timestamp']) * 1e-6
        inf_lidar_file = os.path.join("infrastructure-side", cur_inf_info['pointcloud_path'])
        inf_bbxs_center, _ = load_label(
            os.path.join(root_dir, "infrastructure-side", cur_inf_info['label_lidar_std_path'])
        )
        inf_bbxs_center = []

        ###############################################################
        # process global meta
        coop_bbxs_center, _ = load_label(
            os.path.join(root_dir, "cooperative", "label", f"{pdict['vehicle_frame']}.json")
        )

        ###############################################################
        # construct meta dict
        fdict = cs.fdict_template()
        # add cav lidar meta
        cs.update_agent(fdict,
                        agent_id='0',
                        agent_type='cav',
                        agent_pose=veh_pose,
                        gt_boxes=veh_bbxs_center)
        cs.update_agent_lidar(fdict,
                              agent_id='0',
                              lidar_id='0',
                              lidar_pose=veh_lidar_pose,
                              lidar_time=veh_lidar_time,
                              lidar_file=veh_lidar_file)
        # add infra lidar meta
        cs.update_agent(fdict,
                        agent_id='1',
                        agent_type='infra',
                        agent_pose=inf_lidar_pose,
                        gt_boxes=inf_bbxs_center)
        cs.update_agent_lidar(fdict,
                              agent_id='1',
                              lidar_id='0',
                              lidar_pose=inf_lidar_pose,
                              lidar_time=inf_lidar_time,
                              lidar_file=inf_lidar_file)
        cs.update_frame_bbx(fdict,
                            coop_bbxs_center
                            )# in global coords
        fdict['meta']['ego_id'] = '0'
        fdict['meta']['ego_lidar_pose'] = veh_lidar_pose
        if scenario not in meta_dict:
            meta_dict[scenario] = {}
        meta_dict[scenario][pdict['vehicle_frame']] = fdict
    # save meta
    os.makedirs(meta_out_dir, exist_ok=True)
    for scenario, meta in meta_dict.items():
        meta_file = os.path.join(meta_out_dir, f'{scenario}.json')
        save_json(meta, meta_file)
    with open(os.path.join(meta_out_dir, f'{split}.txt'), 'w') as fh:
        fh.write('\n'.join(list(meta_dict.keys())))


def parse_static_pcd(adict, root_dir):
    pose = pclib.pose_to_transformation(adict['lidar']['0']['pose'])
    pcd = o3d.io.read_point_cloud(os.path.join(root_dir, adict['lidar']['0']['filename']))
    points = np.array(pcd.points)
    boxes = np.array(adict['gt_boxes'])[:, [2, 3, 4, 5, 6, 7, 10]]
    in_box_mask = points_in_boxes_cpu(points, boxes).any(axis=0)
    pcd.points = o3d.utility.Vector3dVector(points[np.logical_not(in_box_mask)])
    return pcd, pose


def register_sequence(sdict, frames, root_dir, ignore_ids=[], vis=False):
    agents_reg = {}
    for f in tqdm.tqdm(frames):
        # print(f)
        fdict = sdict[f]
        for ai, adict in fdict['agents'].items():
            if ai in ignore_ids:
                continue
            pcd, pose = parse_static_pcd(adict, root_dir)
            if ai not in agents_reg:
                agents_reg[ai] = {
                                  'init_pose': pose,
                                  'last_pose_old': pose,
                                  'last_pose_new': pose,
                                  'last_pcd': pcd,
                                  'pcd_merged': copy.copy(pcd).transform(pose),
                                  'last_frame': f,
                                  'sequence_info': {f: {'lidar_pose': pose}}}
            else:
                source_pcd = pcd
                target_pcd = agents_reg[ai]['last_pcd']
                tf_init = np.linalg.inv(agents_reg[ai]['last_pose_old']) @ pose
                tf_out = register_pcds(source_pcd, target_pcd, tf_init, [0.2], visualize=vis)
                pose_new = agents_reg[ai]['last_pose_new'] @ tf_out
                pcd_merged = agents_reg[ai]['pcd_merged']
                pcd_transformed = copy.copy(source_pcd).transform(pose_new)
                # if vis:
                #     pcd_transformed.paint_uniform_color([1, 0.706, 0])
                #     pcd_merged.paint_uniform_color([0, 0.651, 0.929])
                #     o3d.visualization.draw_geometries([pcd_merged, pcd_transformed])
                pcd_merged = pcd_merged + pcd_transformed
                pcd_merged = pcd_merged.voxel_down_sample(voxel_size=0.1)

                agents_reg[ai]['last_pose_old'] = pose
                agents_reg[ai]['last_pose_new'] = pose_new
                agents_reg[ai]['last_pcd'] = pcd
                agents_reg[ai]['pcd_merged'] = pcd_merged
                agents_reg[ai]['sequence_info'][f] = {'lidar_pose': pose}

    return agents_reg


def register_pcds_to_blocks(seq, sdict, root_dir, idx=0):
    frames = sorted(sdict.keys())
    sub_seq = frames[:1]
    cnt = 0
    for i, f in enumerate(frames[1:]):
        if (i == len(frames) - 2 or int(f) - int(sub_seq[-1]) > 2):
            if i == len(frames) - 2:
                sub_seq.append(f)
            if len(sub_seq) >= 8:
                vis = False
                agents_reg = register_sequence(sdict, sub_seq, root_dir, ['1'], vis)
                pcd_merged = agents_reg['0']['pcd_merged']
                o3d.visualization.draw_geometries([pcd_merged])
                o3d.io.write_point_cloud(f"{root_dir}/agent0_seq{seq}_{cnt}.pcd", pcd_merged)
                info_file = f"{root_dir}/agent0_seq{seq}_{cnt}.npy"
                np.save(info_file, {k: v for k, v in agents_reg['0'].items() if 'pcd' not in k}, allow_pickle=True)
                cnt += 1
            if not i == len(frames) - 2:
                sub_seq = [f]
        else:
            sub_seq.append(f)


def optimize_trajectory(seq, sdict, root_dir, out_meta_dir, ego_agent_id, idx, sub_idx):
    """
    This function iterates over scenarios, for each scenario it does the following steps:
    1. register point clouds sequentially for each agent to get accurate trajectory of agents.
    Before registration, the points belonging to the labeled objets with high dynamics are removed.
    After registration of each sequence pair, the merged point cloud is down-sampled to save space.
    2. match the registered point clouds of different agents to get optimized relative poses.
    3. recover the relative pose to the world pose.

    Parameters
    ----------
    meta_path: directory of meta files
    root_dir: root dir of data

    Returns
    -------
    meta: meta information with updated poses of agents
    """
    info_file = f"{root_dir}/agent0_seq{seq}_{sub_idx}.npy"
    ego_info = np.load(info_file, allow_pickle=True).item()
    pcd_merged = o3d.io.read_point_cloud(f"{root_dir}/agent0_seq{seq}_{sub_idx}.pcd")
    frames = sorted(ego_info['sequence_info'].keys())
    sub_seq_dict = {}

    infra_info = sdict[frames[0]]['agents']['1']
    pcd, pose = parse_static_pcd(infra_info, root_dir)
    tf_init = pose
    # o3d.visualization.draw_geometries([pcd_merged])
    tf_out = register_pcds(pcd, pcd_merged, tf_init, [1, 0.2], visualize=True)
    pose = pclib.tf2pose(tf_out)

    for f in tqdm.tqdm(frames):
        fdict = sdict[f]
        fdict['agents']['1']['lidar']['0']['pose'] = pose
        fdict['agents']['1']['pose'] = pose

        lidar_pose_new = ego_info['sequence_info'][f]['lidar_pose']
        lidar_pose_old = pclib.pose_to_transformation(fdict['agents'][ego_agent_id]['lidar']['0']['pose'])
        # lidar_old2new = np.linalg.inv(lidar_pose_new) @ lidar_pose_old
        vpose_to_lpose = np.linalg.inv(lidar_pose_old) @ pclib.pose_to_transformation(fdict['agents'][ego_agent_id]['pose'])
        vpose_new = lidar_pose_new @ vpose_to_lpose
        fdict['agents'][ego_agent_id]['pose'] = pclib.tf2pose(vpose_new)
        fdict['agents'][ego_agent_id]['lidar']['0']['pose'] = pclib.tf2pose(lidar_pose_new)
        sub_seq_dict[f] = fdict
        if int(f) > 1002:
            vis_pcd, vis_pose = parse_static_pcd(fdict['agents'][ego_agent_id], root_dir)
            vis_pcd2, vis_pose2 = parse_static_pcd(fdict['agents']['1'], root_dir)
            vis_pcd = vis_pcd.transform(lidar_pose_new)
            vis_pcd2 = vis_pcd2.transform(vis_pose2)

            # o3d.visualization.draw_geometries([pcd_merged])
            corr = register_pcds(vis_pcd2, vis_pcd, np.eye(4), [1, 0.2], visualize=True)
            vis_pose2 = corr @ vis_pcd2
            vis_pose2 = pclib.tf2pose(vis_pose2)
            fdict['agents']['1']['lidar']['0']['pose'] = vis_pose2
            fdict['agents']['1']['pose'] = vis_pose2

            # vis_pcd.paint_uniform_color([1, 0.706, 0])
            # vis_pcd2.paint_uniform_color([0, 0.651, 0.929])
            # o3d.visualization.draw_geometries([vis_pcd, vis_pcd2.transform(corr)])

    save_json(sub_seq_dict, os.path.join(out_meta_dir, f"{seq}_{sub_idx}.json"))


def optimize_poses(meta_path):
    mfiles = glob.glob(os.path.join(meta_path, '*.json'))[3:]
    mfiles = ["/koko/cosense3d/dairv2x/45.json"]
    for idx, mf in enumerate(mfiles):
        sdict = load_json(mf)
        seq = os.path.basename(mf)[:-5]
        print('###########################', seq, len(sdict))

        # register_pcds_to_blocks(
        #     seq,
        #     sdict,
        #     "/home/data/DAIR-V2X",
        #     idx
        # )
        files = glob.glob(f"/home/data/DAIR-V2X/agent0_seq{seq}_*.npy")
        for sub_idx in range(len(files)):
            optimize_trajectory(seq, sdict,
                "/home/data/DAIR-V2X",
                "/home/data/DAIR-V2X/meta",
                '0',
                idx,
                sub_idx=sub_idx
            )


def register_step_one(mf):
    """Find vehicle that is most close to infra"""
    sdict = load_json(mf)
    seq = os.path.basename(mf)[:-5]
    frames = sorted(sdict.keys())
    min_dist = 1000
    min_dist_frame = frames[0]
    for f in frames:
        fdict = sdict[f]
        veh_pose = fdict['agents']['0']['lidar']['0']['pose']
        inf_pose = fdict['agents']['1']['lidar']['0']['pose']
        dist = np.sqrt((veh_pose[0] - inf_pose[0]) ** 2 + (inf_pose[1] - veh_pose[1]) ** 2)
        if dist < min_dist:
            min_dist = dist
            min_dist_frame = f
    print(f"Step1: registration starts from frame {min_dist_frame}")
    return min_dist_frame, min_dist


def register_step_two(start_frame, mf, meta_out_dir):
    """Register point clouds"""
    sdict = load_json(mf)
    seq = os.path.basename(mf)[:-5]
    frames = sorted(sdict.keys())
    total_frames = len(frames)
    start_idx = frames.index(start_frame)
    ref_pcd, ref_tf = parse_static_pcd(sdict[start_frame]['agents']['1'], root_dir)
    ref_pose = pclib.tf2pose(ref_tf)
    ref_pcd = ref_pcd.transform(ref_tf)
    idx_l = start_idx
    idx_r = start_idx + 1
    vis = False
    cnt = 0
    while True:
        if idx_l < 0 and idx_r >= len(frames):
            break
        if idx_l >= 0:
            cur_frame = frames[idx_l]
            pcd, tf = parse_static_pcd(sdict[cur_frame]['agents']['0'], root_dir)
            if cnt == -1:
                # tf = registration.manual_registration(pcd.transform(tf), ref_pcd)

                tf_corr = np.array([ [ 9.98532892e-01,  5.34621722e-02,  8.59413959e-03, -1.22072297e+02],
                             [-5.34946946e-02,  9.98561645e-01,  3.59984429e-03,  2.15912680e+02],
                             [-8.38932267e-03, -4.05430380e-03,  9.99956590e-01,  4.32884527e+01],
                             [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
                tf = tf_corr @ tf

            else:
                tf = register_pcds(pcd, ref_pcd, tf, [1.6, 0.5], vis, cur_frame)
            pose = pclib.tf2pose(tf)
            sdict[cur_frame]['agents']['0']['lidar']['0']['pose'] = pose
            sdict[cur_frame]['agents']['0']['pose'] = pose
            sdict[cur_frame]['agents']['1']['lidar']['0']['pose'] = ref_pose
            sdict[cur_frame]['agents']['1']['pose'] = ref_pose
            ref_pcd = ref_pcd + pcd.transform(tf)
            idx_l -= 1
            cnt += 1
        if idx_r < len(frames):
            cur_frame = frames[idx_r]
            pcd, tf = parse_static_pcd(sdict[cur_frame]['agents']['0'], root_dir)
            tf = register_pcds(pcd, ref_pcd, tf, [1.6, 0.5], vis, cur_frame)
            pose = pclib.tf2pose(tf)
            sdict[cur_frame]['agents']['0']['lidar']['0']['pose'] = pose
            sdict[cur_frame]['agents']['0']['pose'] = pose
            sdict[cur_frame]['agents']['1']['lidar']['0']['pose'] = ref_pose
            sdict[cur_frame]['agents']['1']['pose'] = ref_pose
            ref_pcd = ref_pcd + pcd.transform(tf)
            idx_r += 1
            cnt += 1

        ref_pcd = ref_pcd.voxel_down_sample(voxel_size=0.1)
        print(f"\rStep2: registered [{cnt}/{total_frames}] frames",end='',flush=True)

    save_json(sdict, os.path.join(meta_out_dir, f"{seq}.json"))
    print('\n')


def select_sub_scenes(meta_in, root_dir, meta_out, split):
    with open(os.path.join(meta_in, f"{split}.txt"), 'r') as f:
        scenes = sorted(f.read().splitlines())

    sub_scenes = []
    for s in tqdm.tqdm(scenes):
        sdict = load_json(os.path.join(meta_in, f"{s}.json"))
        frames = sorted(sdict.keys())
        sub_seq = frames[:1]
        cnt = 0
        for i, f in enumerate(frames[1:]):
            if (i == len(frames) - 2 or int(f) - int(sub_seq[-1]) > 1):
                if i == len(frames) - 2:
                    # reach the end
                    sub_seq.append(f)
                if len(sub_seq) >= 6:
                    # find one valid sub sequence
                    new_sdict = parse_global_bboxes(sdict, sub_seq, root_dir)
                    save_json(new_sdict, os.path.join(meta_out, f"{s}_{cnt}.json"))
                    sub_scenes.append(f"{s}_{cnt}")
                    cnt += 1
                if not i == len(frames) - 2:
                    # sequence breaks, add the current frame to the new seq
                    sub_seq = [f]
            else:
                sub_seq.append(f)

    with open(os.path.join(meta_out, f"{split}.txt"), 'w') as f:
        f.writelines('\n'.join(sub_scenes))


def parse_timestamped_boxes(adict, root_dir, four_wheel_only=True):
    lf = os.path.join(root_dir, adict['lidar']['0']['filename'])
    pcd = point_cloud_from_path(lf)
    boxes = np.array(adict['gt_boxes'])
    if four_wheel_only:
        boxes = boxes[boxes[:, 1] < 4]
    if 'timestamp' in pcd.fields:
        points = np.stack([pcd.pc_data[x] for x in 'xyz'], axis=-1)
        points_inds = points_in_boxes_cpu(points, boxes[:, [2, 3, 4, 5, 6, 7, 10]]).astype(bool)
        times = pcd.pc_data['timestamp']
        timestamps = []
        for i, inds in enumerate(points_inds):
            if inds.sum() == 0:
                nearst_angle_idx = np.abs(np.arctan2(boxes[i, 3], boxes[i, 2]) -
                                          np.arctan2(points[:, 1], points[:, 0])).argmin()
                timestamps.append(times[nearst_angle_idx])
            else:
                ts = times[inds]
                timestamps.append(ts.mean())
        timestamps = np.array(timestamps)
    else:
        timestamps = np.zeros_like(boxes[:, 0]) + adict['lidar']['0']['time']

    return timestamps, boxes


def parse_global_bboxes(sdict, frames, root_dir):
    """Step three"""
    new_sdict = {}
    tracklets = {}
    id_counter = 1
    last_track_ids = set()
    for fi, f in enumerate(frames):
        fdict = sdict[f]
        new_fdict = copy.deepcopy(fdict)
        matched_track_ids = set()
        matched_inds = []
        for ai, adict in fdict['agents'].items():
            timestamps, boxes = parse_timestamped_boxes(adict, root_dir)
            tf = pclib.pose_to_transformation(adict['lidar']['0']['pose'])
            boxes_global = transform_boxes_3d(boxes, tf, mode=11)
            if len(tracklets) == 0:
                for i, (t, box) in enumerate(zip(timestamps, boxes_global)):
                    tracklets[id_counter] = [[t] + box[1:].tolist()]
                    boxes[i, 0] = id_counter
                    id_counter += 1
            else:
                tracked_boxes = []
                tracked_ids = []
                for k, v in tracklets.items():
                    tracked_ids.append(k)
                    tracked_boxes.append(v[-1])
                tracked_boxes = np.array(tracked_boxes)
                tracked_ids = np.array(tracked_ids)
                dist_cost = np.linalg.norm(tracked_boxes[:, [2, 3]][:, None] - boxes_global[:, [2, 3]][None], axis=-1)
                thr = 3
                min_dist = dist_cost.min(axis=0)
                min_idx = dist_cost.argmin(axis=0)
                match_inds = []
                for i, box in enumerate(boxes_global):
                    cur_box = [timestamps[i]] + box[1:].tolist()
                    if min_dist[i] < thr:
                        tracklets[tracked_ids[min_idx[i]]].append(cur_box)
                        match_inds.append([tracked_ids[min_idx[i]], i])
                        boxes[i, 0] = tracked_ids[min_idx[i]]
                    else:
                        tracklets[id_counter] = [cur_box]
                        boxes[i, 0] = id_counter
                        id_counter += 1
                matched_inds.extend(match_inds)

            new_fdict['agents'][ai]['gt_boxes'] = boxes.tolist()
        new_sdict[f] = new_fdict

    object_size_type = {}
    for ti, tracklet in tracklets.items():
        tracklets[ti] = np.array(sorted(tracklet))
        object_size_type[ti] = {
            'size': np.median(tracklets[ti][:, 5:8], axis=0),
            'type': np.median(tracklets[ti][:, 1], axis=0),
        }

    # remove last two frames
    new_sdict.pop(frames[-1])
    new_sdict.pop(frames[-2])
    for f, fdict in new_sdict.items():
        object_ids = []
        for ai, adict in fdict['agents'].items():
            object_ids.extend([int(box[0]) for box in adict['gt_boxes']])
        object_ids = set(object_ids)
        aligned_time = math.ceil(fdict['agents']['0']['lidar']['0']['time'] * 10) / 10
        aligned_boxes = [[], [], []]
        for object_id in object_ids:
            if object_id in tracklets:
                tracklet = tracklets[object_id]
                if len(tracklet) == 0:
                    continue
                for i in range(3):
                    cur_time = aligned_time + 0.1 * i
                    time_diff = tracklet[:, 0] - cur_time
                    try:
                        prev_idx = np.where(time_diff < 0)[0].max()
                        next_idx = np.where(time_diff > 0)[0].min()
                        prev_t = tracklet[prev_idx][0]
                        next_t = tracklet[next_idx][0]
                        dxyz = tracklet[next_idx][[2, 3, 4]] - tracklet[prev_idx][[2, 3, 4]]
                        xyz = tracklet[prev_idx][[2, 3, 4]] + dxyz * (cur_time - prev_t) / (next_t - prev_t)
                        prev_rot = tracklet[next_idx][10]
                        object_param = [object_id , object_size_type[object_id]['type']] + xyz.tolist() + \
                                        object_size_type[object_id]['size'].tolist() + [0, 0, prev_rot]
                        aligned_boxes[i].append(object_param)
                    except:
                        aligned_boxes[i].append([0] * 11)
            else:
                print('d')
        aligned_boxes = np.array(aligned_boxes)
        tf = pclib.pose_to_transformation(fdict['agents']['0']['lidar']['0']['pose'])
        aligned_boxes = box_utils.transform_boxes_3d(
            aligned_boxes.reshape(-1, 11), np.linalg.inv(tf), mode=11).reshape(aligned_boxes.shape)
        fdict['meta']['bbx_center_global'] = aligned_boxes[0].tolist()
        fdict['meta']['boxes_pred'] = {f"{int(f) + i + 1:06d}": x[:, [2, 3, 4, 10]].tolist() \
                                       for i, x in enumerate(aligned_boxes[1:])}

    return new_sdict


def remove_ego_boxes(meta_in):
    mfs = glob.glob(os.path.join(meta_in, '*.json'))
    for mf in mfs:
        sdict = load_json(mf)
        for f, fdict in sdict.items():
            gt_boxes = np.array(fdict['agents']['0']['gt_boxes'])
            depth = np.linalg.norm(gt_boxes[:, 2:4], axis=-1)
            gt_boxes = gt_boxes[depth > 2]
            fdict['agents']['0']['gt_boxes'] = gt_boxes.tolist()

            global_boxes = np.array(fdict['meta']['bbx_center_global'])
            mask = np.linalg.norm(global_boxes[:, 2:4], axis=-1) > 2
            fdict['meta']['bbx_center_global'] = global_boxes[mask].tolist()
            boxes_pred = fdict['meta']['boxes_pred']
            fdict['meta']['boxes_pred'] = {k: np.array(v)[mask].tolist() for k, v in boxes_pred.items()}

        save_json(sdict, mf)


if __name__=="__main__":
    root_dir = "/home/data/DAIR-V2X"
    meta_out_dir = "/home/data/DAIR-V2X/meta-coalign"
    meta_path = "/home/data/cosense3d/dairv2x"
    # root_dir = "/home/data/DAIR-V2X-Seq/SPD-Example"
    # meta_out_dir = "/home/data/cosense3d/dairv2x_seq"
    convert_v2x_c(root_dir, meta_out_dir)
    # meta_dict = load_meta(os.path.join(meta_out_dir, 'dairv2x'))
    # o3d_play_sequence(meta_dict, root_dir)
    # optimize_poses(meta_path)

    # with open("/home/data/DAIR-V2X/meta/test.txt", 'w') as fh:
    #     files = glob.glob("/home/data/DAIR-V2X/meta/*.json")
    #     for f in files:
    #         fh.writelines(os.path.basename(f)[:-5] + '\n')

    # mfs = sorted(glob.glob("/home/yuan/data/DAIR-V2X/meta-loc-correct/*.json"))[:1]
    # # mf = "/home/data/cosense3d/dairv2x/11.json"
    # for mf in mfs:
        # if int(os.path.basename(mf)[:-5]) <= 10:
        #     continue
        # min_dist_frame, min_dist = register_step_one(mf)
        # sdict = register_step_two(min_dist_frame, mf, meta_out_dir)
        # parse_global_bboxes(mf, meta_out_dir, root_dir)

    # select_sub_scenes(
    #     "/home/yuan/data/DAIR-V2X/meta-loc-correct",
    #     "/home/yuan/data/DAIR-V2X",
    #     "/home/yuan/data/DAIR-V2X/meta-sub-scenes",
    #     "test"
    # )

    # remove_ego_boxes("/home/yuan/data/DAIR-V2X/meta_with_pred")






