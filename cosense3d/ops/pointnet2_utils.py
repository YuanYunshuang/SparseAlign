from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, Variable

import cuda_ops as pointnet2


class BallQuery(Function):

    @staticmethod
    def forward(ctx, radius: float, nsample: int, xyz: torch.Tensor, xyz_batch_cnt: torch.Tensor,
                new_xyz: torch.Tensor, new_xyz_batch_cnt):
        """
        Args:
            ctx:
            radius: float, radius of the balls
            nsample: int, maximum number of features in the balls
            xyz: (N1 + N2 ..., 3) xyz coordinates of the features
            xyz_batch_cnt: (batch_size), [N1, N2, ...]
            new_xyz: (M1 + M2 ..., 3) centers of the ball query
            new_xyz_batch_cnt: (batch_size), [M1, M2, ...]

        Returns:
            idx: (M1 + M2, nsample) tensor with the indicies of the features that form the query balls
        """
        assert new_xyz.is_contiguous()
        assert new_xyz_batch_cnt.is_contiguous()
        assert xyz.is_contiguous()
        assert xyz_batch_cnt.is_contiguous()

        B = xyz_batch_cnt.shape[0]
        M = new_xyz.shape[0]
        idx = torch.cuda.IntTensor(M, nsample).zero_()

        pointnet2.ball_query_wrapper(B, M, radius, nsample, new_xyz, new_xyz_batch_cnt, xyz, xyz_batch_cnt, idx)
        empty_ball_mask = (idx[:, 0] == -1)
        idx[empty_ball_mask] = 0
        return idx, empty_ball_mask

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


ball_query = BallQuery.apply


class GroupingOperation(Function):

    @staticmethod
    def forward(ctx, features: torch.Tensor, features_batch_cnt: torch.Tensor,
                idx: torch.Tensor, idx_batch_cnt: torch.Tensor):
        """
        Args:
            ctx:
            features: (N1 + N2 ..., C) tensor of features to group
            features_batch_cnt: (batch_size) [N1 + N2 ...] tensor containing the indicies of features to group with
            idx: (M1 + M2 ..., nsample) tensor containing the indicies of features to group with
            idx_batch_cnt: (batch_size) [M1 + M2 ...] tensor containing the indicies of features to group with

        Returns:
            output: (M1 + M2, C, nsample) tensor
        """
        assert features.is_contiguous()
        assert features_batch_cnt.is_contiguous()
        assert idx.is_contiguous()
        assert idx_batch_cnt.is_contiguous()

        assert features.shape[0] == features_batch_cnt.sum(), \
            'features: %s, features_batch_cnt: %s' % (str(features.shape), str(features_batch_cnt))
        assert idx.shape[0] == idx_batch_cnt.sum(), \
            'idx: %s, idx_batch_cnt: %s' % (str(idx.shape), str(idx_batch_cnt))

        M, nsample = idx.size()
        N, C = features.size()
        B = idx_batch_cnt.shape[0]
        output = torch.cuda.FloatTensor(M, C, nsample)

        pointnet2.group_points_wrapper(B, M, C, nsample, features, features_batch_cnt, idx, idx_batch_cnt, output)

        ctx.for_backwards = (B, N, idx, features_batch_cnt, idx_batch_cnt)
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        """
        Args:
            ctx:
            grad_out: (M1 + M2 ..., C, nsample) tensor of the gradients of the output from forward

        Returns:
            grad_features: (N1 + N2 ..., C) gradient of the features
        """
        B, N, idx, features_batch_cnt, idx_batch_cnt = ctx.for_backwards

        M, C, nsample = grad_out.size()
        grad_features = Variable(torch.cuda.FloatTensor(N, C).zero_())

        grad_out_data = grad_out.data.contiguous()
        pointnet2.group_points_grad_wrapper(B, M, C, N, nsample, grad_out_data, idx,
                                            idx_batch_cnt, features_batch_cnt, grad_features.data)
        return grad_features, None, None, None


grouping_operation = GroupingOperation.apply


class QueryAndGroup(nn.Module):
    def __init__(self, radius: float, nsample: int, use_xyz: bool = True):
        """
        Args:
            radius: float, radius of ball
            nsample: int, maximum number of features to gather in the ball
            use_xyz:
        """
        super().__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz

    def forward(self, xyz: torch.Tensor, xyz_batch_cnt: torch.Tensor,
                new_xyz: torch.Tensor, new_xyz_batch_cnt: torch.Tensor,
                features: torch.Tensor = None):
        """
        Args:
            xyz: (N1 + N2 ..., 3) xyz coordinates of the features
            xyz_batch_cnt: (batch_size), [N1, N2, ...]
            new_xyz: (M1 + M2 ..., 3) centers of the ball query
            new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
            features: (N1 + N2 ..., C) tensor of features to group

        Returns:
            new_features: (M1 + M2, C, nsample) tensor
        """
        assert xyz.shape[0] == xyz_batch_cnt.sum(), 'xyz: %s, xyz_batch_cnt: %s' % (str(xyz.shape), str(new_xyz_batch_cnt))
        assert new_xyz.shape[0] == new_xyz_batch_cnt.sum(), \
            'new_xyz: %s, new_xyz_batch_cnt: %s' % (str(new_xyz.shape), str(new_xyz_batch_cnt))

        # idx: (M1 + M2 ..., nsample), empty_ball_mask: (M1 + M2 ...)
        idx, empty_ball_mask = ball_query(self.radius, self.nsample, xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt)
        grouped_xyz = grouping_operation(xyz, xyz_batch_cnt, idx, new_xyz_batch_cnt)  # (M1 + M2, 3, nsample)
        grouped_xyz -= new_xyz.unsqueeze(-1)

        grouped_xyz[empty_ball_mask] = 0

        if features is not None:
            grouped_features = grouping_operation(features, xyz_batch_cnt, idx, new_xyz_batch_cnt)  # (M1 + M2, C, nsample)
            grouped_features[empty_ball_mask] = 0
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)  # (M1 + M2 ..., C + 3, nsample)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        return new_features, idx


class FurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz: torch.Tensor, npoint: int):
        """
        Args:
            ctx:
            xyz: (B, N, 3) where N > npoint
            npoint: int, number of features in the sampled set

        Returns:
            output: (B, npoint) tensor containing the set
        """
        assert xyz.is_contiguous()

        B, N, _ = xyz.size()
        output = torch.cuda.IntTensor(B, npoint)
        temp = torch.cuda.FloatTensor(B, N).fill_(1e10)

        pointnet2.furthest_point_sampling_wrapper(B, N, npoint, xyz, temp, output)
        return output

    @staticmethod
    def backward(xyz, a=None):
        return None, None


furthest_point_sample = FurthestPointSampling.apply


class ThreeNN(Function):
    @staticmethod
    def forward(ctx, unknown, unknown_batch_cnt, known, known_batch_cnt):
        """
        Args:
            ctx:
            unknown: (N1 + N2..., 3)
            unknown_batch_cnt: (batch_size), [N1, N2, ...]
            known: (M1 + M2..., 3)
            known_batch_cnt: (batch_size), [M1, M2, ...]

        Returns:
            dist: (N1 + N2 ..., 3)  l2 distance to the three nearest neighbors
            idx: (N1 + N2 ..., 3)  index of the three nearest neighbors, range [0, M1+M2+...]
        """
        assert unknown.shape.__len__() == 2 and unknown.shape[1] == 3
        assert known.shape.__len__() == 2 and known.shape[1] == 3
        assert unknown_batch_cnt.__len__() == known_batch_cnt.__len__()

        dist2 = unknown.new_zeros(unknown.shape)
        idx = unknown_batch_cnt.new_zeros(unknown.shape).int()

        pointnet2.three_nn_wrapper(
            unknown.contiguous(), unknown_batch_cnt.contiguous(),
            known.contiguous(), known_batch_cnt.contiguous(), dist2, idx
        )
        return torch.sqrt(dist2), idx

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None


three_nn = ThreeNN.apply


class ThreeInterpolate(Function):

    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor, weight: torch.Tensor):
        """
        Args:
            ctx:
            features: (M1 + M2 ..., C)
            idx: [N1 + N2 ..., 3]
            weight: [N1 + N2 ..., 3]

        Returns:
            out_tensor: (N1 + N2 ..., C)
        """
        assert idx.shape[0] == weight.shape[0] and idx.shape[1] == weight.shape[1] == 3

        ctx.three_interpolate_for_backward = (idx, weight, features.shape[0])
        output = features.new_zeros((idx.shape[0], features.shape[1]))
        pointnet2.three_interpolate_wrapper(features.contiguous(), idx.contiguous(), weight.contiguous(), output)
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        """
        Args:
            ctx:
            grad_out: (N1 + N2 ..., C)

        Returns:
            grad_features: (M1 + M2 ..., C)
        """
        idx, weight, M = ctx.three_interpolate_for_backward
        grad_features = grad_out.new_zeros((M, grad_out.shape[1]))
        pointnet2.three_interpolate_grad_wrapper(
            grad_out.contiguous(), idx.contiguous(), weight.contiguous(), grad_features
        )
        return grad_features, None, None


three_interpolate = ThreeInterpolate.apply


class StackSAModuleMSG(nn.Module):

    def __init__(self, *, radii: List[float], nsamples: List[int], mlps: List[List[int]],
                 use_xyz: bool = True, pool_method='max_pool'):
        """
        Args:
            radii: list of float, list of radii to group with
            nsamples: list of int, number of samples in each ball query
            mlps: list of list of int, spec of the pointnet before the global pooling for each scale
            use_xyz:
            pool_method: max_pool / avg_pool
        """
        super().__init__()

        assert len(radii) == len(nsamples) == len(mlps)

        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(QueryAndGroup(radius, nsample, use_xyz=use_xyz))
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            shared_mlps = []
            for k in range(len(mlp_spec) - 1):
                shared_mlps.extend([
                    nn.Conv2d(mlp_spec[k], mlp_spec[k + 1], kernel_size=1, bias=False),
                    nn.BatchNorm2d(mlp_spec[k + 1]),
                    nn.ReLU()
                ])
            self.mlps.append(nn.Sequential(*shared_mlps))
        self.pool_method = pool_method

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, features=None, empty_voxel_set_zeros=True):
        """
        :param xyz: (N1 + N2 ..., 3) tensor of the xyz coordinates of the features
        :param xyz_batch_cnt: (batch_size), [N1, N2, ...]
        :param new_xyz: (M1 + M2 ..., 3)
        :param new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
        :param features: (N1 + N2 ..., C) tensor of the descriptors of the the features
        :return:
            new_xyz: (M1 + M2 ..., 3) tensor of the new features' xyz
            new_features: (M1 + M2 ..., \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        new_features_list = []
        for k in range(len(self.groupers)):
            new_features, ball_idxs = self.groupers[k](
                xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, features
            )  # (M1 + M2, C, nsample)
            new_features = new_features.permute(1, 0, 2).unsqueeze(dim=0)  # (1, C, M1 + M2 ..., nsample)
            new_features = self.mlps[k](new_features)  # (1, C, M1 + M2 ..., nsample)

            if self.pool_method == 'max_pool':
                new_features = F.max_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                ).squeeze(dim=-1)  # (1, C, M1 + M2 ...)
            elif self.pool_method == 'avg_pool':
                new_features = F.avg_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                ).squeeze(dim=-1)  # (1, C, M1 + M2 ...)
            else:
                raise NotImplementedError
            new_features = new_features.squeeze(dim=0).permute(1, 0)  # (M1 + M2 ..., C)
            new_features_list.append(new_features)

        new_features = torch.cat(new_features_list, dim=1)  # (M1 + M2 ..., C)

        return new_xyz, new_features


class StackPointnetFPModule(nn.Module):
    def __init__(self, *, mlp: List[int]):
        """
        Args:
            mlp: list of int
        """
        super().__init__()
        shared_mlps = []
        for k in range(len(mlp) - 1):
            shared_mlps.extend([
                nn.Conv2d(mlp[k], mlp[k + 1], kernel_size=1, bias=False),
                nn.BatchNorm2d(mlp[k + 1]),
                nn.ReLU()
            ])
        self.mlp = nn.Sequential(*shared_mlps)

    def forward(self, unknown, unknown_batch_cnt, known, known_batch_cnt, unknown_feats=None, known_feats=None):
        """
        Args:
            unknown: (N1 + N2 ..., 3)
            known: (M1 + M2 ..., 3)
            unknow_feats: (N1 + N2 ..., C1)
            known_feats: (M1 + M2 ..., C2)

        Returns:
            new_features: (N1 + N2 ..., C_out)
        """
        dist, idx = three_nn(unknown, unknown_batch_cnt, known, known_batch_cnt)
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=-1, keepdim=True)
        weight = dist_recip / norm

        interpolated_feats = three_interpolate(known_feats, idx, weight)

        if unknown_feats is not None:
            new_features = torch.cat([interpolated_feats, unknown_feats], dim=1)  # (N1 + N2 ..., C2 + C1)
        else:
            new_features = interpolated_feats
        new_features = new_features.permute(1, 0)[None, :, :, None]  # (1, C, N1 + N2 ..., 1)
        new_features = self.mlp(new_features)

        new_features = new_features.squeeze(dim=0).squeeze(dim=-1).permute(1, 0)  # (N1 + N2 ..., C)
        return new_features
