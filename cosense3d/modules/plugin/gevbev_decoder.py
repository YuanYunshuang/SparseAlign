import torch
import torch_scatter
from torch import nn

from cosense3d.modules.utils.misc import SELayer_Linear
from cosense3d.modules.utils.gaussian_utils import weighted_mahalanobis_dists
from cosense3d.modules.utils.me_utils import indices2metric, metric2indices, update_me_essentials


class GevBEVDecoder(nn.Module):
    def __init__(self, data_info, stride, kernel=3, var0=0.1):
        super().__init__()
        update_me_essentials(self, data_info, stride)
        self.lr = nn.Parameter(torch.tensor(self.lidar_range), requires_grad=False)
        self.vs = nn.Parameter(torch.tensor(self.voxel_size), requires_grad=False)
        self.var0 = [var0, var0]
        x = torch.arange(kernel) - kernel // 2
        self.nbrs = torch.stack(torch.meshgrid(x, x, indexing='ij'),
                                dim=-1).reshape(-1, 2)
        self.nbrs = nn.Parameter(self.nbrs, requires_grad=False)
        self.n_nbrs = len(self.nbrs)

    def coor_to_indices(self, coor):
        inds = coor.clone()
        inds[:, 1] = inds[:, 1] / self.stride - self.offset_sz_x
        inds[:, 2] = inds[:, 2] / self.stride - self.offset_sz_y
        return inds.long()

    def forward(self, ref_pts, ctr_coor, ctr_reg):
        """
        :param ref_pts: LongTensor(Q, 3) 2d coordinates in metrics(batch_idx, x, y)
        :param ctr_coor: LongTensor(V, 3) 2d coordinates in indices (batch_idx, x, y)
        :param ctr_reg: FloatTensor(V, d) bev grid center point regression result

        :return: out_evidence FloatTensor(Q, d): attended features
        """
        reg = ctr_reg.relu()
        reg_evi = reg[:, :2]
        reg_var = reg[:, 2:].view(-1, 2, 2)

        ctr_pts = indices2metric(ctr_coor, self.vs)
        ctr_inds = self.coor_to_indices(ctr_coor)
        ref_coor = metric2indices(ref_pts, self.vs)
        ref_inds = self.coor_to_indices(ref_coor)

        q_inds, v_inds, mask = self.get_nbr_mapping(ref_inds, ctr_inds)

        evidence = torch.zeros_like(ref_pts[:, :2])
        dists = ref_pts[q_inds[mask], 1:3] - ctr_pts[v_inds[mask], 1:3]
        probs_weighted = weighted_mahalanobis_dists(reg_evi[v_inds[mask]], reg_var[v_inds[mask]], dists, self.var0)
        torch_scatter.scatter(probs_weighted, q_inds[mask],
                              dim=0, out=evidence, reduce='sum')
        return evidence.reshape(len(ref_pts), self.n_nbrs, 2)

    def get_nbr_mapping(self, query_pos, value_pos):
        B = query_pos[:, 0].max() + 1
        pad_width = 2
        query_pos[:, 1:] += pad_width
        value_pos[:, 1:] += pad_width
        query_inds = torch.arange(len(query_pos), dtype=torch.long)
        value_inds = torch.arange(len(value_pos), dtype=torch.long)

        # index -1 indicates that this nbr is outside the grid range
        value_map = - torch.ones((B, self.size_x + pad_width * 2,
                                  self.size_y + pad_width * 2), dtype=torch.long)
        value_map[value_pos[:, 0],
                  value_pos[:, 1],
                  value_pos[:, 2]] = value_inds

        query_inds_nbrs = query_pos.unsqueeze(dim=1).repeat(1, self.n_nbrs, 1)
        query_inds_nbrs[..., 1:] += self.nbrs.view(1, -1, 2)
        query_inds_nbrs = query_inds_nbrs.view(-1, 3)
        mask = ((query_inds_nbrs >= 0).all(dim=-1) &
                (query_inds_nbrs[:, 1] < self.size_x + pad_width * 2) &
                (query_inds_nbrs[:, 2] < self.size_y + pad_width * 2))
        assert torch.logical_not(mask).sum() == 0
        query_inds_mapped = query_inds.unsqueeze(1).repeat(1, self.n_nbrs).view(-1)
        value_inds_mapped = value_map[query_inds_nbrs[:, 0],
                                      query_inds_nbrs[:, 1],
                                      query_inds_nbrs[:, 2]]
        mask = torch.logical_and(query_inds_mapped >= 0, value_inds_mapped >= 0)
        return query_inds_mapped, value_inds_mapped, mask
