import math
import torch
from torch import nn

from cosense3d.modules.utils.positional_encoding import pos2posemb2d


class NeighborhoodAttention(nn.Module):
    """Generate reference points and attend neighborhood features."""
    def __init__(self, emb_dim, n_nbr=16, num_pose_feat=64, **kwargs):
        super(NeighborhoodAttention, self).__init__(**kwargs)
        self.n_nbr = n_nbr
        self.emb_dim = emb_dim
        self.num_pose_feat = num_pose_feat
        self.q_pos_emb = nn.Sequential(
                    nn.Linear(num_pose_feat * 2, self.emb_dim),
                    nn.ReLU(),
                    nn.Linear(self.emb_dim, self.emb_dim),
                )
        self.kv_pos_emb = nn.Sequential(
                    nn.Linear(num_pose_feat * 2, self.emb_dim),
                    nn.ReLU(),
                    nn.Linear(self.emb_dim, self.emb_dim),
                )

    def forward(self, memory, mem_coor, q_coor, B):
        """

        Args:
            q: (S, D)
            kv: (L, D)
            q_coor: (S, 3), [idx, x, y]
            kv_coor: (L, 3)

        Returns:

        """
        query_pos = self.q_pos_emb(pos2posemb2d(q_coor[:, 1:], self.num_pose_feat))
        memory_pos = self.kv_pos_emb(pos2posemb2d(mem_coor[:, 1:], self.num_pose_feat))
        query = query_pos
        kv_pe = memory_pos + memory

        outs = []
        for b in range(B):
            qm = q_coor[:, 0] == b
            km = mem_coor[:, 0] == b
            q = query[qm]
            kv = memory[km]
            S, D = q.shape
            L = kv.shape[0]
            dists = torch.norm(q_coor[qm].unsqueeze(1) - mem_coor[km].unsqueeze(0), dim=-1)  # (B, S, L)
            topk_inds = torch.topk(-dists, k=self.n_nbr, dim=-1)  # (B, S, n_nbr)
            kv_inds = torch.cat([topk_inds[b] + b * L for b in range(B)], dim=0)  # (BS, n_nbr)
            q_inds = torch.cat([torch.arange(S) + b * S for b in range(B)], dim=0
                               ).view(-1, 1).repeat(1, self.n_nbr)  # (BS, n_nbr)
            kv_m = kv_pe[km].view(-1, D)[kv_inds.view(-1)]
            product = q.view(-1, D)[q_inds.view(-1)] * kv_m  # (BS*n_nbr, D)
            scaled_product = product / math.sqrt(D)
            attn_weights = scaled_product.softmax(dim=-1)
            out = (attn_weights * kv.view(-1, D)[kv_inds.view(-1)]).view(B, S, self.n_nbr, D)
            outs.append(out)
        return out