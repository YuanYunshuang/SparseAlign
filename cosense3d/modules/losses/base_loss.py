import torch
from torch import nn


class BaseLoss(nn.Module):
    def __init__(self,
                 reduction: str = 'mean',
                 activation: str = 'none',
                 loss_weight: float = 1.0):
        """
        :param reduction: (optional) the method to reduce the loss.
        :param activation: options are "none", "mean" and "sum".
        :param loss_weight: (optional) the weight of loss.
        """
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.activation = activation

    @property
    def name(self):
        return self.__class__.__name__

    def loss(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self,
                preds: torch.Tensor,
                targets: torch.Tensor,
                weight: torch.Tensor=None,
                avg_factor: int=None,
                reduction_override: str=None,
                *args, **kwargs) -> torch.Tensor:
        """

        :param preds: prediction tensor.
        :param targets: target tensor.
        :param weight: The weight of loss for each
                prediction. Defaults to None.
        :param avg_factor: Average factor that is used to average
                the loss. Defaults to None.
        :param reduction_override: The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        :param args: additional arguments.
        :param kwargs:
        :return: weighted loss.
        """
        loss = self.loss(preds, targets, *args, **kwargs)
        # if weight is specified, apply element-wise weight
        if weight is not None:
            loss = loss * weight
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        # if avg_factor is not specified, just reduce the loss
        if avg_factor is None:
            if reduction == 'mean':
                loss = loss.mean()
            elif reduction == 'sum':
                loss = loss.sum()
        else:
            # if reduction is mean, then average the loss by avg_factor
            if reduction == 'mean':
                # Avoid causing ZeroDivisionError when avg_factor is 0.0,
                # i.e., all labels of an image belong to ignore index.
                eps = torch.finfo(torch.float32).eps
                loss = loss.sum() / (avg_factor + eps)
            # if reduction is 'none', then do nothing, otherwise raise an error
            elif reduction != 'none':
                raise ValueError('avg_factor can not be used with reduction="sum"')
        return self.loss_weight * loss