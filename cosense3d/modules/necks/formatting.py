import torch
from torch import nn

from cosense3d.modules import BaseModule


class DenseToSparse(BaseModule):
    def __init__(self,
                 data_info,
                 strides=None,
                 **kwargs):
        super(DenseToSparse, self).__init__(**kwargs)
        self.lidar_range = data_info['lidar_range']
        self.voxel_size = data_info['voxel_size']
        self.strides = strides

    def forward(self, *args, **kwargs):
        input_dict = {self.gather_keys[i]: x for i, x in enumerate(args)}
        out_dict = {}
        multi_scale_bev_feat = []
        for x in input_dict['multi_scale_bev_feat']:
            tmp = {}
            for s in self.strides:
                tmp[f'p{s}'] = {
                    'ctr': self.get_centers(s, device=x[f'p{s}'].device).flatten(0, 1),
                    'feat': x[f'p{s}'].permute(1, 2, 0).flatten(0, 1)
                }
            multi_scale_bev_feat.append(tmp)
        out_dict['multi_scale_bev_feat'] = multi_scale_bev_feat

        det_local_sparse = []
        for x in input_dict['det_local_dense']:
            det_local_sparse.append({'scr': x['cls'].max(dim=0).values.flatten()})
        out_dict['det_local_sparse'] = det_local_sparse

        if 'bev_local_dense' in input_dict:
            bev_local_sparse = []
            for x in input_dict['bev_local_dense']:
                bev_local_sparse.append({'scr': x.max(dim=0).values.flatten()})
            out_dict['bev_local_sparse'] = bev_local_sparse

        # from cosense3d.utils.vislib import draw_points_boxes_plt, plt
        # draw_points_boxes_plt(
        #     pc_range=self.lidar_range,
        #     points=input_dict['points'][0][:, :3].detach().cpu().numpy(),
        #     filename="/media/yuan/luna/streamLTS/LTS_fcooper_dairv2x/points.png"
        # )
        # fig = plt.figure(figsize=(10, 5))
        # ax = fig.add_subplot()
        # pts = multi_scale_bev_feat[0]['p2']['ctr'].detach().cpu().numpy()
        # # colors = det_local_sparse[0]['scr'].sigmoid().detach().cpu().numpy()
        # colors = multi_scale_bev_feat[0]['p2']['feat'].mean(dim=1).detach().cpu().numpy()
        # ax.scatter(pts[:, 0], pts[:, 1], c=colors)
        # plt.savefig("/media/yuan/luna/streamLTS/LTS_fcooper_dairv2x/scores.png")
        return out_dict

    def get_centers(self, stride, device):
        pix_x = self.voxel_size[0] * stride
        pix_y = self.voxel_size[1] * stride
        x = torch.arange(self.lidar_range[0], self.lidar_range[3], pix_x, device=device) + pix_x * 0.5
        y = torch.arange(self.lidar_range[1], self.lidar_range[4], pix_y, device=device) + pix_y * 0.5
        centers = torch.stack(torch.meshgrid(x, y, indexing='ij'), dim=-1)
        return centers.permute(1, 0, 2)


class DetDenseToSparse(nn.Module):
    def __init__(self,
                 data_info,
                 stride,
                 **kwargs):
        super(DetDenseToSparse, self).__init__(**kwargs)
        self.lidar_range = data_info['lidar_range']
        self.voxel_size = data_info['voxel_size']
        self.stride = stride

    def forward(self, input):
        out_list = []
        for x in input:
            # select the max of two anchors at each position
            h, w = x['cls'].shape[1:]
            cls, max_inds = x['cls'].permute(0, 2, 1).max(dim=0)
            scr = cls.sigmoid()
            reg = x['reg'].view(x['cls'].shape[0], -1, h, w).permute(3, 2, 0, 1)
            ctr = self.get_centers()
            out_list.append({
                'ctr': ctr.flatten(0, 1),
                'cls': cls.flatten(0, 1),
                'reg': reg.flatten(0, 1),
                'scr': scr.flatten(0, 1)
            })

        return out_list

    def get_centers(self):
        pix_x = self.voxel_size[0] * self.stride
        pix_y = self.voxel_size[1] * self.stride
        x = torch.arange(self.lidar_range[0], self.lidar_range[3], pix_x) + pix_x * 0.5
        y = torch.arange(self.lidar_range[1], self.lidar_range[4], pix_y) + pix_y * 0.5
        centers = torch.stack(torch.meshgrid(x, y, indexing='ij'), dim=-1)
        return centers


class FPVRCNNToLTS(BaseModule):
    def __init__(self,
                 data_info,
                 strides=None,
                 **kwargs):
        super(FPVRCNNToLTS, self).__init__(**kwargs)
        self.lidar_range = data_info['lidar_range']
        self.voxel_size = data_info['voxel_size']
        self.strides = strides

    def forward(self, *args, **kwargs):
        input_dict = {self.gather_keys[i]: x for i, x in enumerate(args)}
        out_dict = {}
        multi_scale_feat = []
        roi_local = []
        for x, y in zip(input_dict['multi_scale_bev_feat'], input_dict['keypoint_feat']):
            res = {}
            if 8 in self.strides:
                res['p2'] = {
                    'ctr': y['point_coords'][:, 1:4],
                    'feat': y['point_features']
                }
            if 32 in self.strides:
                res['p8'] = {
                    'ctr': self.get_centers(32, device=x[f'p32'].device).flatten(0, 1),
                    'feat': x['p32'].permute(1, 2, 0).flatten(0, 1)
                }
            multi_scale_feat.append(res)
            roi_local.append({'scr': y['point_scores']})

        out_dict['multi_scale_feat'] = multi_scale_feat
        out_dict['roi_local'] = roi_local

        if 'bev_local_dense' in input_dict:
            bev_local_sparse = []
            for x in input_dict['bev_local_dense']:
                bev_local_sparse.append({'scr': x.max(dim=0).values.flatten()})
            out_dict['roi_global'] = bev_local_sparse

        # from cosense3d.utils.vislib import draw_points_boxes_plt, plt
        # draw_points_boxes_plt(
        #     pc_range=self.lidar_range,
        #     points=input_dict['points'][0][:, :3].detach().cpu().numpy(),
        #     filename="/media/yuan/luna/streamLTS/LTS_fcooper_dairv2x/points.png"
        # )
        # fig = plt.figure(figsize=(10, 5))
        # ax = fig.add_subplot()
        # pts = multi_scale_bev_feat[0]['p2']['ctr'].detach().cpu().numpy()
        # # colors = det_local_sparse[0]['scr'].sigmoid().detach().cpu().numpy()
        # colors = multi_scale_bev_feat[0]['p2']['feat'].mean(dim=1).detach().cpu().numpy()
        # ax.scatter(pts[:, 0], pts[:, 1], c=colors)
        # plt.savefig("/media/yuan/luna/streamLTS/LTS_fcooper_dairv2x/scores.png")
        return out_dict

    def get_centers(self, stride, device):
        pix_x = self.voxel_size[0] * stride
        pix_y = self.voxel_size[1] * stride
        x = torch.arange(self.lidar_range[0], self.lidar_range[3], pix_x, device=device) + pix_x * 0.5
        y = torch.arange(self.lidar_range[1], self.lidar_range[4], pix_y, device=device) + pix_y * 0.5
        centers = torch.stack(torch.meshgrid(x, y, indexing='ij'), dim=-1)
        return centers.permute(1, 0, 2)


class FPVRCNNToSparseAlign(BaseModule):
    def __init__(self,
                 data_info,
                 strides=None,
                 **kwargs):
        super(FPVRCNNToSparseAlign, self).__init__(**kwargs)
        self.lidar_range = data_info['lidar_range']
        self.voxel_size = data_info['voxel_size']
        self.strides = strides

    def forward(self, *args, **kwargs):
        input_dict = {self.gather_keys[i]: x for i, x in enumerate(args)}
        out_dict = {}
        multi_scale_feat = []
        roi_local = []
        for x, y in zip(input_dict['multi_scale_bev_feat'], input_dict['det_local_dense']):
            res = {}
            for s in self.strides:
                s_ = s // 4
                res[f'p{s_}'] = {
                    'ctr': self.get_centers(s, device=x[f'p{s}'].device).flatten(0, 1),
                    'feat': x[f'p{s}'].permute(1, 2, 0).flatten(0, 1)
                }
            multi_scale_feat.append(res)
            roi_local.append({'scr': y['cls'].sigmoid().max(dim=0).values.flatten()})

        out_dict['multi_scale_feat'] = multi_scale_feat
        out_dict['roi_local'] = roi_local

        return out_dict

    def get_centers(self, stride, device):
        pix_x = self.voxel_size[0] * stride
        pix_y = self.voxel_size[1] * stride
        x = torch.arange(self.lidar_range[0], self.lidar_range[3], pix_x, device=device) + pix_x * 0.5
        y = torch.arange(self.lidar_range[1], self.lidar_range[4], pix_y, device=device) + pix_y * 0.5
        centers = torch.stack(torch.meshgrid(x, y, indexing='ij'), dim=-1)
        return centers.permute(1, 0, 2)
