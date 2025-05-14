import os

import numpy as np
from torch import nn
import open3d as o3d
import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from cosense3d.utils.misc import load_json
from cosense3d.utils.box_utils import transform_boxes_3d, compute_iou
from cosense3d.utils.pclib import pose_to_transformation
from cosense3d.utils.vislib import draw_points_boxes_plt, plt
from cosense3d.utils.train_utils import is_tensor_to_cuda
from cosense3d.ops.iou3d_nms_utils import boxes_iou_bev
from cosense3d.utils.lr_scheduler import build_lr_scheduler


class GraphVertexRegistration(nn.Module):
    def __init__(self, topk=16):
        super().__init__()
        self.topk = topk
        self.in_proj = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU()
        )
        self.attn = nn.MultiheadAttention(128, 8)
        self.out_proj = nn.Linear(256, 128)

        self.triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)

        self.data = []

    def forward(self, data):
        """

        :param data: List[List[Tensor(Ni, 9)]]
        :return:
            boxes_list: List[Tensor(sum(Ni), 9)], list of boxes predicted by each agent
            feat_list: List[Tensor(sum(Ni), 128)], the corresponding neighborhood features of each box in boxes_list
            batch_cnt: List[N1, N2, ..., Ni, ...]
        """
        batch_cnt = []
        boxes_list = []
        for x in data:
            c = len(x)
            batch_cnt.append(c)
            boxes_list.extend(x)

        feat_list = []
        for boxes in boxes_list:
            x = self.get_nbrhood_features(boxes)
            feat_list.append(x)

            # ax = draw_points_boxes_plt(pc_range=100,boxes_gt=B0[:, :7].cpu().numpy(),return_ax=True)
            # ax = draw_points_boxes_plt(pc_range=100,boxes_pred=B1[:, :7].cpu().numpy(),return_ax=True, ax=ax)
            # for b0, b1 in zip(MB0, MB1):
            #     ax.plot([b0[0], b1[0]], [b0[1], b1[1]], 'b-')
            # plt.show()
            # plt.close()

        return boxes_list, feat_list, batch_cnt

    def get_nbrhood_features(self, boxes):
        nb = len(boxes)
        dist = torch.norm(boxes[:, :2][:, None] - boxes[:, :2][None], dim=-1)
        if nb < self.topk:
            topk_inds = torch.randint(0, nb, (nb, self.topk))
        else:
            _, topk_inds = torch.topk(dist, k=self.topk, largest=False)
        nbrs = boxes[topk_inds.flatten()].reshape(topk_inds.shape + (boxes.shape[-1],))
        ctr = boxes[:, [0, 1, 6]][:, None]
        nbrs_xyr = nbrs[..., [0, 1, 6]]
        nbrs_lwh = nbrs[..., 3:6] / 5
        edge_angle = torch.atan2(nbrs_xyr[..., 1] - ctr[..., 1], nbrs_xyr[..., 0] - ctr[..., 0])
        edge_angle = (edge_angle - ctr[..., -1])
        edge_angle = torch.stack([torch.sin(edge_angle), torch.cos(edge_angle)], dim=-1)
        edge_dist = torch.norm(nbrs_xyr[..., :2] - ctr[..., :2], dim=-1).unsqueeze(dim=-1) / 100
        nbr_angle = nbrs_xyr[..., -1] - ctr[..., -1]
        nbr_angle = torch.stack([torch.sin(nbr_angle), torch.cos(nbr_angle)], dim=-1)
        feat = torch.cat([edge_dist, edge_angle, nbr_angle, nbrs_lwh], dim=-1)

        x = self.in_proj(feat)
        q = k = v = x.transpose(0, 1)
        x, w = self.attn(q, k, v)
        x = torch.cat([x.mean(dim=0), x.max(dim=0).values], dim=-1)
        x = self.out_proj(x)
        return x

    def cosine_similarity(self, feat1, feat2):
        """
        :param feat1: (N1, 8, 2)
        :param feat2: (N2, 8, 2)
        :return: (N1, N2, 8, 8) cosine similarity between each pairs of features
        """
        feat1 = feat1[:, None, :, None]
        feat2 = feat2[None, :, None]
        feat1_norm = torch.norm(feat1, dim=-1)
        feat2_norm = torch.norm(feat2, dim=-1)
        prod = (feat1 * feat2).sum(dim=-1)
        sim = prod / (feat1_norm * feat2_norm)
        sim = sim.max(dim=-1)[0].mean(dim=-1)
        return sim

    def euclidean_dist(self, feat1, feat2):
        is_numpy = isinstance(feat1, np.ndarray)
        diff = feat1[:, None] - feat2[None]
        if is_numpy:
            return np.linalg.norm(diff, axis=-1)
        else:
            return torch.norm(diff, p=2, dim=-1).detach().cpu().numpy()

    def size_diff(self, boxes1, boxes2):
        diff = boxes1[:, None] - boxes2[None]
        diff = np.abs(diff).sum(axis=-1) ** 2
        return diff * 3

    def assign(self, F0, F1):
        sim = self.cosine_similarity(F0, F1).cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(1 - sim)
        match = sim[row_ind, col_ind] > 0.999
        return row_ind[match], col_ind[match]

    def match_pair(self, ego_boxes, coop_boxes):
        # self.data.append([ego_boxes.tolist(), coop_boxes.tolist()])
        ego_feat = self.get_nbrhood_features(ego_boxes)
        coop_feat = self.get_nbrhood_features(coop_boxes)
        cost_feat = self.euclidean_dist(ego_feat, coop_feat)
        cost_dist = self.euclidean_dist(ego_boxes[:, :2], coop_boxes[:, :2])
        # size_diff = self.size_diff(ego_boxes[:, 3:6], coop_boxes[:, 3:6])
        cost = cost_feat + cost_dist

        row_ind, col_ind = linear_sum_assignment(cost)
        match_cost = cost_feat[row_ind, col_ind]

        if cost_dist is not None:
            match_dist = cost_dist[row_ind, col_ind]
            match_dist_sort = np.sort(match_dist)
            diff = match_dist_sort[1:] - match_dist_sort[:-1]
            tmp = np.where(diff > 5)[0]
            tmp = len(match_dist_sort) - 1 if len(tmp) == 0 else tmp[0] + 1
            mask = match_dist < match_dist_sort[tmp]
            # mask = mask & (match_cost < 5)
            row_ind, col_ind, match_cost = row_ind[mask], col_ind[mask], match_cost[mask]
        return row_ind, col_ind, match_cost

    @torch.no_grad()
    def clustering(self, boxes_list, feat_list, batch_cnt):
        ptr = 0
        for c in batch_cnt:
            ego_boxes = boxes_list[ptr]
            ego_feat = feat_list[ptr]
            for i in range(1, c):
                coop_boxes = boxes_list[ptr+i]
                coop_feat = feat_list[ptr+i]
                row_ind, col_ind, match_cost = self.match_pair(ego_feat, coop_feat)

                # matched_ego_boxes = ego_boxes[row_ind, :7].detach().cpu().numpy()
                # matched_coop_boxes = coop_boxes[col_ind, :7].detach().cpu().numpy()

                # ax = draw_points_boxes_plt(
                #     pc_range=100,
                #     boxes_gt=matched_ego_boxes,
                #     return_ax=True
                # )
                # ax = draw_points_boxes_plt(
                #     pc_range=100,
                #     boxes_pred=matched_coop_boxes,
                #     return_ax=True,
                #     ax=ax
                # )
                #
                # for b1, b2 in zip(matched_ego_boxes, matched_coop_boxes):
                #     ax.plot([b1[0], b2[0]], [b1[1], b2[1]], color='black')
                #
                # plt.savefig("/home/yuan/Downloads/tmp.jpg")
                # plt.close()

    def loss(self, res):
        boxes_list, feat_list, batch_cnt = res
        anchor, pos, neg = [], [], []
        ptr = 0
        iou_thr_pos = 0.3
        iou_thr_neg = 0.05
        for c in batch_cnt:
            boxes = torch.cat(boxes_list[ptr:ptr+c], dim=0)[:, :7]
            feat = torch.cat(feat_list[ptr:ptr+c], dim=0)
            ptr += c

            ious = boxes_iou_bev(boxes, boxes)
            # ious = ious - 2 * torch.eye(len(ious), device=ious.device)
            pos_mask = ious > iou_thr_pos
            pos_inds = torch.where(pos_mask)
            anchor_ious = ious[pos_inds[0]]
            # the max operation below to always select the first element if all ious are 0
            # add random noisy to avoid this issue
            anchor_ious = anchor_ious + torch.rand_like(anchor_ious) * 1e-3
            anchor_ious = -1 * (anchor_ious >= iou_thr_pos).float() + anchor_ious * (anchor_ious < iou_thr_neg).float()
            _, neg_inds = anchor_ious.max(dim=-1)  # select the hardest sample

            anchor.append(feat[pos_inds[0]])
            pos.append(feat[pos_inds[1]])
            neg.append(feat[neg_inds])

        anchor = torch.cat(anchor, dim=0)
        pos = torch.cat(pos, dim=0)
        neg = torch.cat(neg, dim=0)

        loss = self.triplet_loss(anchor, pos, neg)

        return loss


class BoxDataset(Dataset):
    def __init__(self, data_dir):
        samples = os.listdir(data_dir)
        self.samples = []
        for x in samples:
            if len(x.split('.')[0]) > 5:
                continue
            self.samples.append(torch.load(os.path.join(data_dir, x))['detection_local'])
        print(f"loaded {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        # data = torch.load(os.path.join(data_dir, self.samples[item]))['detection_local']
        data = self.samples[item]
        boxes_list = []
        for ai, adict in data.items():
            if 'box' in adict:
                boxes = adict['box']
            else:
                boxes = adict['preds']['box']

            boxes_list.append(boxes)

        # ax = draw_points_boxes_plt(
        #     pc_range=100,
        #     boxes_gt=boxes_list[0][:, :7],
        #     return_ax=True
        # )
        # for boxes in boxes_list[1:]:
        #     ax = draw_points_boxes_plt(
        #         pc_range=100,
        #         boxes_pred=boxes[:, :7],
        #         return_ax=True,
        #         ax=ax
        #     )
        # plt.savefig(f"{os.environ['HOME']}/Downloads/tmp.jpg")
        # plt.close()
        return boxes_list

    @staticmethod
    def collate_batch(batch_list):
        return batch_list


class BoxInCroppedRangeDataset(Dataset):
    def __init__(self, data_dir):
        files = os.listdir(data_dir)
        self.samples = []
        for f in files:
            import json
            with open(os.path.join(data_dir, f), 'r') as fh:
                data = json.load(fh)
                self.samples.extend(data)

        print(f"loaded {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        # data = torch.load(os.path.join(data_dir, self.samples[item]))['detection_local']
        data = [torch.tensor(x) for x in self.samples[item]]

        # boxes1 = np.array(data[0])
        # boxes2 = np.array(data[1])
        # ctr = np.concatenate([boxes1, boxes2], axis=0)[:, :2].mean(axis=0)
        # boxes1[:, :2] = boxes1[:, :2] - ctr
        # boxes2[:, :2] = boxes2[:, :2] - ctr
        # ax = draw_points_boxes_plt(
        #     pc_range=100,
        #     boxes_gt=boxes1[:, :7],
        #     return_ax=True
        # )
        # ax = draw_points_boxes_plt(
        #     pc_range=100,
        #     boxes_pred=boxes2[:, :7],
        #     return_ax=True,
        #     ax=ax
        # )
        # plt.savefig(f"{os.environ['HOME']}/Downloads/tmp.jpg")
        # plt.close()
        return data

    @staticmethod
    def collate_batch(batch_list):
        return batch_list


def generate_edge_features(boxes):
    boxes = boxes[:, [2, 3, 4, 5, 6, 7, 10]]
    x = np.arange(len(boxes))
    xx = np.stack(np.meshgrid(x, x), axis=-1)
    inds = np.triu_indices(len(x), k=1)
    inds = xx[inds[0], inds[1]].T
    edge_dists = np.linalg.norm(boxes[inds[0], :2] - boxes[inds[1], :2], axis=-1) / 100
    edge_angle = np.arctan2(boxes[inds[0], 1] - boxes[inds[1], 1], boxes[inds[0], 0] - boxes[inds[1], 0])
    vertice1_angle = np.arctan2(boxes[inds[0], 1] , boxes[inds[0], 0])
    vertice2_angle = np.arctan2(boxes[inds[1], 1] , boxes[inds[1], 0])
    cs0 = np.cos(edge_angle)
    cs1 = np.cos(vertice1_angle)
    cs2 = np.cos(vertice2_angle)
    sn0 = np.sin(edge_angle)
    sn1 = np.sin(vertice1_angle)
    sn2 = np.sin(vertice2_angle)
    angle_diff1 = np.arccos(cs0 * cs1 + sn0 * sn1)
    angle_diff2 = np.arccos(cs0 * cs2 + sn0 * sn2)
    angle_diff_min = np.minimum(angle_diff1, angle_diff2) / np.pi
    angle_diff_max = np.maximum(angle_diff1, angle_diff2) / np.pi

    return np.stack([edge_dists, angle_diff_min, angle_diff_max], axis=-1), inds


def train(data_dir, n_nbr=16):
    batch_size = 8
    lr = 0.0001
    n_epochs = 800
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = BoxInCroppedRangeDataset(data_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_batch)
    model = GraphVertexRegistration(topk=n_nbr).cuda().train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    cfg = {
        'dim_embed': 256,
        'policy': 'TransformerAdaptiveScheduler',
        'warmup_steps': 1000
    }
    total_itr = len(loader)
    lr_scheduler = build_lr_scheduler(optimizer, cfg, total_itr)
    model.train()
    for epoch in range(n_epochs):
        running_loss = 0.0
        for i, data in enumerate(loader):
            data = is_tensor_to_cuda(data, device)

            # Forward pass
            outputs = model(data)
            loss = model.loss(outputs)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step_itr(i + epoch * total_itr)

            running_loss += loss.item()

        running_loss = running_loss / len(loader)
        rec_lr = lr_scheduler.get_last_lr()[0]
        print(f'Epoch [{epoch + 1}/{n_epochs}], LR: {rec_lr:.5f}, Train Loss: {running_loss:.4f}')
        if (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(), f"{os.environ.get('HOME')}/Downloads/match_nbr{n_nbr}_ep{epoch+1}.pth")


def test(data_dir):
    batch_size = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = BoxDataset(data_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_batch)
    model = GraphVertexRegistration(16).cuda()
    state_dict = torch.load(f"{os.environ.get('HOME')}/Downloads/model_nbr16.pth")
    model.load_state_dict(state_dict)

    model.eval()
    for i, data in enumerate(loader):
        data = is_tensor_to_cuda(data, device)

        # Forward pass
        outputs = model(data)
        clusters = model.clustering(*outputs)




if __name__ == '__main__':
    data_dir = "/home/yys/Downloads/pose_align"
    train(data_dir, 16)


