import torch
import torch.nn.functional as F
import numpy as np
from torch import nn

from cosense3d.modules.utils.misc import SELayer_Linear
from cosense3d.modules.utils.positional_encoding import pos2posemb2d
from cosense3d.modules.utils.me_utils import indices2metric, metric2indices, update_me_essentials
from torch_scatter import scatter_min, scatter_max, scatter_add, scatter_softmax, scatter_mean


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values
    """
    def __init__(self, dim: int):
        """
        :param dim: implementation of attention
        """
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query, key, value):
        """
        :param query: (batch, q_len, d_model) tensor containing projection vector for decoder.
        :param key: (batch, k_len, d_model) tensor containing projection vector for encoder.
        :param value: (batch, v_len, d_model) tensor containing features of the encoded input sequence.
        :return: context, attn
                - **context**: tensor containing the context vector from attention mechanism.
                - **attn**: tensor containing the attention (alignment) from the encoder outputs.
        """
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim
        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context


class NeighborhoodAttention(nn.Module):
    def __init__(self, data_info, stride, emb_dim=128):
        super().__init__()
        self.stride = stride
        update_me_essentials(self, data_info, self.stride)
        self.lr = nn.Parameter(torch.tensor(self.lidar_range), requires_grad=False)
        self.vs = nn.Parameter(torch.tensor(self.voxel_size), requires_grad=False)
        # self.grid_size = (
        #     round((lr[3] - lr[0]) / vs[0] / stride),
        #     round((lr[4] - lr[1]) / vs[1] / stride),
        # )
        self.emb_dim = emb_dim
        self.num_pos_feat = emb_dim // 2
        self.sqrt_dim = np.sqrt(emb_dim)
        x = torch.arange(-1, 2)
        self.nbrs = torch.stack(torch.meshgrid(x, x, indexing='ij'),
                                dim=-1).reshape(-1, 2)
        self.nbrs = nn.Parameter(self.nbrs, requires_grad=False)
        self.n_nbrs = len(self.nbrs)

        self.query_pos_encoder = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2),
            nn.ReLU(),
            nn.Linear(emb_dim * 2, emb_dim),
        )
        self.value_pos_encoder = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2),
            nn.ReLU(),
            nn.Linear(emb_dim * 2, emb_dim),
        )
        self.featurized_pe = SELayer_Linear(emb_dim)

    def coor_to_indices(self, coor):
        inds = coor.clone()
        inds[:, 1] = inds[:, 1] / self.stride - self.offset_sz_x
        inds[:, 2] = inds[:, 2] / self.stride - self.offset_sz_y
        return inds.long()

    def forward(self, ref_pts, ctr_coor, ctr_feat):
        """

        Parameters
        ----------
        ref_pts LongTensor(Q, 3): 2d coordinates in metrics(batch_idx, x, y)
        ctr_coor LongTensor(V, 3): 2d coordinates in indices (batch_idx, x, y)
        ctr_feat FloatTensor(V, d): bev grid center point features

        Returns
        -------
            out_features FloatTensor(Q, d): attended features
        """
        Q = ref_pts.shape[0]
        V, Vd = ctr_feat.shape

        ctr_pts = indices2metric(ctr_coor, self.vs)
        ctr_inds = self.coor_to_indices(ctr_coor)
        ref_coor = metric2indices(ref_pts, self.vs)
        ref_inds = self.coor_to_indices(ref_coor)
        query_pos = (ref_pts[:, 1:] - self.lr[0:2]) / (self.lr[3:5] - self.lr[0:2])
        value_pos = (ctr_pts[:, 1:] - self.lr[0:2]) / (self.lr[3:5] - self.lr[0:2])

        qpos_emb = self.query_pos_encoder(
            pos2posemb2d(query_pos, num_pos_feats=self.num_pos_feat))
        vpos_emb = self.value_pos_encoder(
            pos2posemb2d(value_pos, num_pos_feats=self.num_pos_feat))
        vpos_emb = self.featurized_pe(vpos_emb, ctr_feat)

        q_inds, v_inds = self.get_nbr_mapping(ref_inds, ctr_inds)
        # pad pos_embs with zeros at the 1st entry
        # points outside the grid will retrieve the embedding in the 1st padded row
        qpos_emb = torch.cat([torch.zeros_like(qpos_emb[:1]), qpos_emb], dim=0)
        vpos_emb = torch.cat([torch.zeros_like(vpos_emb[:1]), vpos_emb], dim=0)
        ctr_feat = torch.cat([torch.zeros_like(ctr_feat[:1]), ctr_feat], dim=0)

        score = (qpos_emb[q_inds] * vpos_emb[v_inds]).sum(dim=-1) / self.sqrt_dim
        attn = F.softmax(score.view(-1, self.n_nbrs), dim=-1)
        context = attn.unsqueeze(-1) * ctr_feat[v_inds].view(-1, self.n_nbrs, Vd)
        return context.sum(1)

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
        # shift the overall indices by 1 step, index -1 will then become 0
        return query_inds_mapped + 1, value_inds_mapped + 1


class VerticalAttention(nn.Module):
    """
    Merge vertical features of 3D sparse features using attention mechanism.
    """
    def __init__(self, in_dim, embed_dim, dropout=0., bias=True, batch_first=False) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.batch_first = batch_first
        self.in_proj = nn.Linear(in_dim, embed_dim * 3, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.sqrt_dim = np.sqrt(embed_dim)

    def _init_weight(self):
        nn.init.xavier_uniform_(self.in_proj_weight)

        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, x):
        """
        :param x: x.C(N, 4), x,F(N, in_dim)
        :return:
            coor: (M, 3) reduced coordinates (reduced on z-coordinate)
            feat: (M, emb_dim) reduced features
        """
        coor, inverse_map = x.C[:, :3].unique(dim=0, return_inverse=True)
        mask = x.C[:, 0] == 0
        min_z = scatter_min(x.C[mask, 3], index=inverse_map[mask], dim=0)[0]
        max_z = scatter_max(x.C[mask, 3], index=inverse_map[mask], dim=0)[0]
        diff = (max_z - min_z).max().item()
        ks = diff // x.tensor_stride[2] + 1
        kernel_map = x.coordinate_manager.kernel_map(
            x.coordinate_key, x.coordinate_key, kernel_size=[1, 1, ks], stride=1
        )
        kernel_map = torch.cat([km for km in kernel_map.values()], dim=1).long()

        q, k, v = self.in_proj(x.F).split(self.embed_dim, dim=-1)
        attn = (q[kernel_map[0]] * k[kernel_map[1]]).sum(dim=-1) / self.sqrt_dim
        attn = scatter_softmax(attn, dim=0, index=kernel_map[1])

        attended_feat = scatter_add(attn.unsqueeze(1) * v[kernel_map[0]], dim=0, index=kernel_map[1])
        attended_feat = self.out_proj(attended_feat)
        mean_feat = scatter_mean(attended_feat, dim=0, index=inverse_map)
        max_feat = scatter_max(attended_feat, dim=0, index=inverse_map)[0]
        out = max_feat + mean_feat
        return coor, out
