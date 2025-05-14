import torch
import torch.nn as nn

from einops import rearrange


class VanillaSegLoss(nn.Module):
    def __init__(self, d_weights, s_weights, d_coe, s_coe, l_weights=50, **kwargs):
        super(VanillaSegLoss, self).__init__()

        self.d_weights = d_weights
        self.s_weights = s_weights
        self.l_weights = l_weights

        self.d_coe = d_coe
        self.s_coe = s_coe

        self.loss_func_static = \
            nn.CrossEntropyLoss(
                weight=torch.Tensor([1., self.s_weights, self.l_weights]).cuda())
        self.loss_func_dynamic = \
            nn.CrossEntropyLoss(
                weight=torch.Tensor([1., self.d_weights]).cuda())

    def forward(self, static_pred=None, dynamic_pred=None,
                static_gt=None, dynamic_gt=None):
        """
        Perform loss function on the prediction.

        Parameters
        ----------
        output_dict : dict
            The dictionary contains the prediction.

        gt_dict : dict
            The dictionary contains the groundtruth.

        Returns
        -------
        Loss dictionary.
        """
        loss_dict = {}

        if static_pred is not None:
            # during training, only need to compute the ego vehicle's gt loss
            # static_gt = rearrange(static_gt, 'b l h w -> (b l) h w')
            # static_pred = rearrange(static_pred, 'b l c h w -> (b l) c h w')
            static_loss = self.loss_func_static(static_pred, static_gt.long())
            loss_dict['static_loss'] = self.s_coe * static_loss

        if dynamic_pred is not None:
            # dynamic_gt = rearrange(dynamic_gt, 'b l h w -> (b l) h w')
            # dynamic_pred = rearrange(dynamic_pred, 'b l c h w -> (b l) c h w')
            dynamic_loss = self.loss_func_dynamic(dynamic_pred, dynamic_gt.long())
            loss_dict['dynamic_loss'] = self.d_coe * dynamic_loss

        return loss_dict






