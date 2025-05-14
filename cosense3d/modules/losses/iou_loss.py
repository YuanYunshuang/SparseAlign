from .base_loss import BaseLoss
from cosense3d.utils.iou2d_calculator import bbox_overlaps


class IoULoss(BaseLoss):
    def __init__(self, mode: str='log', eps:float=1e-6,  **kwargs):
        """

        :param mode: Loss scaling mode, including "linear", "square", and "log".
            Default: 'log'
        :param eps: Eps to avoid log(0).
        :param kwargs:
        """
        super(IoULoss, self).__init__(**kwargs)
        assert mode in ['linear', 'square', 'log']
        self.mode = mode
        self.eps = eps

    def loss(self, pred, target):
        ious = bbox_overlaps(pred, target, is_aligned=True).clamp(min=self.eps)
        if self.mode == 'linear':
            loss = 1 - ious
        elif self.mode == 'square':
            loss = 1 - ious ** 2
        elif self.mode == 'log':
            loss = -ious.log()
        else:
            raise NotImplementedError
        return loss


class GIoULoss(BaseLoss):
    def __init__(self, eps: float=1e-7,  **kwargs):
        """

        :param eps: Eps to avoid log(0).
        :param kwargs:
        """
        super(GIoULoss, self).__init__(**kwargs)
        self.eps = eps

    def loss(self, pred, target):
        gious = bbox_overlaps(pred, target, mode='giou', is_aligned=True, eps=self.eps)
        loss = 1 - gious
        return loss

