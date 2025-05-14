import torch
import torch.nn.functional as F


def weighted_smooth_l1_loss(preds, targets, sigma=3.0, weights=None):
    diff = preds - targets
    abs_diff = torch.abs(diff)
    abs_diff_lt_1 = torch.le(abs_diff, 1 / (sigma ** 2)).type_as(abs_diff)
    loss = abs_diff_lt_1 * 0.5 * torch.pow(abs_diff * sigma, 2) + \
               (abs_diff - 0.5 / (sigma ** 2)) * (1.0 - abs_diff_lt_1)
    if weights is not None:
        if len(loss.shape) > len(weights.shape):
            weights = weights.unsqueeze(dim=-1)
        loss *= weights
    return loss


def weighted_l1_loss(preds, targets, sigma=3.0, weights=None):
    diff = preds - targets
    loss = torch.abs(diff)
    if weights is not None:
        if len(loss.shape) > len(weights.shape):
            weights = weights.unsqueeze(dim=-1)
        loss *= weights
    return loss


def sigmoid_binary_cross_entropy(preds, tgts, weights=None, reduction='none'):
    """
    Parameters
    ----------
    preds: Tensor(d1, ..., dn)
    tgts: Tensor(d1, ..., dn)
    weights. Tensor(d1, ..., dn)
    reduction: str('none' | 'mean' | 'sum')
    -------
    """
    assert preds.shape == tgts.shape
    if weights is not None:
        assert weights.shape == preds.shape
    per_entry_cross_ent = F.binary_cross_entropy_with_logits(
        preds, tgts,
        weights, reduction=reduction
    )
    return per_entry_cross_ent


def weighted_sigmoid_binary_cross_entropy(preds, tgts, weights=None,
                                          class_indices=None):
    if weights is not None:
        weights = weights.unsqueeze(-1)
    if class_indices is not None:
        weights *= (
            indices_to_dense_vector(class_indices, preds.shape[2])
                .view(1, 1, -1)
                .type_as(preds)
        )
    per_entry_cross_ent = F.binary_cross_entropy_with_logits(preds, tgts, weights)
    return per_entry_cross_ent


def indices_to_dense_vector(
        indices: torch.Tensor,
        size: int,
        indices_value: float = 1.0,
        default_value: float = 0.0
) -> torch.Tensor:
    """
    Creates dense vector with indices set to specific value and rest to zeros.

    This function exists because it is unclear if it is safe to use
    tf.sparse_to_dense(indices, [size], 1, validate_indices=False)
    with indices which are not ordered.
    This function accepts a dynamic size (e.g. `tf.shape(tensor)[0]`)

    :param indices: 1d Tensor with integer indices which are to be set to indices_values.
    :param size: size of output Tensor.
    :param indices_value: values of elements specified by indices in the output vector.
    :param default_value: values of other elements in the output vector.
    :return: dense 1D Tensor of shape [size] with indices set to indices_values and the
        rest set to default_value.
    """
    dense = torch.zeros(size).fill_(default_value)
    dense[indices] = indices_value

    return dense


def cross_entroy_with_logits(preds, tgts, n_cls, weights=None, reduction='none'):
    cared = tgts >= 0
    preds = preds[cared]
    tgts = tgts[cared]
    tgt_onehot = torch.zeros((len(tgts), n_cls), device=preds.device)
    tgt_onehot[torch.arange(len(tgts), device=tgts.device), tgts.long()] = 1

    loss = F.cross_entropy(preds, tgt_onehot, weight=weights, reduction=reduction)
    return loss


def focal_loss(preds, tgts, weights=None, reduction='none',
               gamma=2.0, alpha=0.25, use_sigmoid=True):
    """

    Parameters
    ----------
    preds: FloatTensor(..., n_cls)
    tgts: FloatTensor(..., n_cls) or LongTensor(...,) or LongTensor(...,1), largest label is background
    weights: same as preds or tgts
    -------
    """
    assert len(preds.shape) == len(tgts.shape) or len(preds.shape) - 1 == len(tgts.shape)
    if use_sigmoid:
        pred_sigmoid = torch.sigmoid(preds)
    else:
        pred_sigmoid = preds

    if preds.shape[-1] != tgts.shape[-1]:
        num_classes = preds.size(1)
        one_hot_tgts = F.one_hot(tgts, num_classes=num_classes + 1)
        one_hot_tgts = one_hot_tgts[:, :num_classes]
    else:
        one_hot_tgts = tgts

    alpha_weight = one_hot_tgts * alpha + (1 - one_hot_tgts) * (1 - alpha)
    pt = one_hot_tgts * (1.0 - pred_sigmoid) + (1.0 - one_hot_tgts) * pred_sigmoid
    focal_weight = alpha_weight * torch.pow(pt, gamma)

    bce_loss = torch.clamp(preds, min=0) - preds * one_hot_tgts + \
               torch.log1p(torch.exp(-torch.abs(preds)))

    loss = focal_weight * bce_loss
    if weights is None:
        return loss
    elif weights.shape.__len__() < preds.shape.__len__():
        weights = weights.unsqueeze(-1)

    assert weights.shape.__len__() == loss.shape.__len__()

    return loss * weights