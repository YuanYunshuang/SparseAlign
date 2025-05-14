import torch
from .base_loss import BaseLoss


class L1Loss(BaseLoss):

    def loss(self, pred, target):
        if target.numel() == 0:
            return pred.sum() * 0

        assert pred.size() == target.size()
        loss = torch.abs(pred - target)
        return loss


class SmoothL1Loss(BaseLoss):
    def __init__(self, beta: float=1.0, **kwargs):
        """
        :param beta: The threshold in the piecewise function.
            Defaults to 1.0.
        :param kwargs:
        """
        super(SmoothL1Loss, self).__init__(**kwargs)
        assert beta > 0
        self.beta = beta

    def loss(self, pred, target):
        if target.numel() == 0:
            return pred.sum() * 0

        assert pred.size() == target.size()
        diff = torch.abs(pred - target)
        loss = torch.where(diff < self.beta,
                           0.5 * diff * diff / self.beta,
                           diff - 0.5 * self.beta)
        return loss