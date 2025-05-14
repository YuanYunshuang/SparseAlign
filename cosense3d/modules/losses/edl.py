import torch
import torch.nn.functional as F

from cosense3d.modules.losses import BaseLoss


def relu_evidence(y):
    return F.relu(y)


def exp_evidence(y):
    return torch.exp(torch.clamp(y, -6, 6))


def softplus_evidence(y):
    return F.softplus(y)


def kl_divergence(alpha, num_classes):
    device = alpha.device
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        # + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl


def loglikelihood_loss(y, alpha):
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
    )
    loglikelihood = loglikelihood_err + loglikelihood_var
    return loglikelihood


def mse_loss(y, alpha, epoch_num, num_classes, annealing_step):
    loglikelihood = loglikelihood_loss(y, alpha)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes)
    return loglikelihood + kl_div


def edl_mse_loss(preds, tgt, n_cls, temp, annealing_step, model_label='edl'):
    """
    Calculate evidential loss
    :param model_label: (str) a name to distinguish edl loss of different modules
    :param preds: (N, n_cls) the logits of each class
    :param tgt: (N,) labels with values from 0...(n_cls - 1) or (N, n_cls)
    :param n_cls: (int) number of classes, including background
    :param temp: current temperature for annealing of KL Divergence term of the loss
    :param annealing_step: maximum annealing step
    :return:
    """
    evidence = relu_evidence(preds)
    if len(tgt.shape) == 1:
        cared = tgt >= 0
        evidence = evidence[cared]
        tgt = tgt[cared]
        tgt_onehot = F.one_hot(tgt.long(), n_cls).float()
    elif len(tgt.shape) == 2 and tgt.shape[1] > 1:
        cared = (tgt >= 0).all(dim=-1)
        evidence = evidence[cared]
        tgt_onehot = tgt[cared]
    else:
        raise NotImplementedError
    alpha = evidence + 1
    loss = mse_loss(tgt_onehot, alpha, temp, n_cls, annealing_step).mean()

    ss = evidence.detach()
    tt = tgt_onehot.detach()
    acc = (torch.argmax(ss, dim=1) == torch.argmax(tt, dim=1)).sum() / len(tt) * 100
    loss_dict = {
        f'{model_label}_loss': loss,
        f'{model_label}_ac': acc,
    }

    # Uncomment to log recall of all classes
    # for cls in [1, 2]:
    #     loss_dict[f'acc{cls}'] = torch.logical_and(
    #         torch.argmax(ss, dim=1) == cls, tt == cls).sum() \
    #                              / max((tt == cls).sum(), 1) * 100

    return loss_dict


def evidence_to_conf_unc(evidence, edl=True):
    if edl:
    # used edl loss
        alpha = evidence + 1
        S = torch.sum(alpha, dim=-1, keepdim=True)
        conf = torch.div(alpha, S)
        K = evidence.shape[-1]
        unc = torch.div(K, S)
        # conf = torch.sqrt(conf * (1 - unc))
        unc = unc.squeeze(dim=-1)
    else:
        # use entropy as uncertainty
        entropy = -evidence * torch.log2(evidence)
        unc = entropy.sum(dim=-1)
        # conf = torch.sqrt(evidence * (1 - unc.unsqueeze(-1)))
        conf = evidence
    return conf, unc


def pred_to_conf_unc(preds, activation='relu', edl=True):
    if callable(activation):
        evidence = activation(preds)
    elif activation == 'relu':
        evidence = relu_evidence(preds)
    elif activation == 'exp':
        evidence = exp_evidence(preds)
    elif activation == 'sigmoid':
        evidence = preds.sigmoid()
    elif activation == 'softmax':
        evidence = preds.softmax(dim=-1)
    else:
        evidence = preds

    if edl:
        alpha = evidence + 1
        S = torch.sum(alpha, dim=-1, keepdim=True)
        conf = torch.div(alpha, S)
        K = evidence.shape[-1]
        unc = torch.div(K, S)
        # conf = torch.sqrt(conf * (1 - unc))
        unc = unc.squeeze(dim=-1)
    else:
        # use entropy as uncertainty
        entropy = -evidence * torch.log2(evidence)
        unc = entropy.sum(dim=-1)
        # conf = torch.sqrt(evidence * (1 - unc.unsqueeze(-1)))
        conf = evidence
    return conf, unc


class EDLLoss(BaseLoss):
    def __init__(self,
                 n_cls: int,
                 annealing_step: int,
                 **kwargs):
        """
        Evidential loss.

        :param n_cls: number of classes, including background.
        :param annealing_step: maximum temperature annealing step for KL regularization of EDL loss .
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.n_cls = n_cls
        self.annealing_step = annealing_step
        if self.activation == 'relu':
            self.activation = relu_evidence
        elif self.activation == 'exp':
            self.activation = exp_evidence
        else:
            self.activation = None

    def loss(self, preds, tgt, temp, n_cls_override=None):
        if self.activation is None:
            evidence = preds
        else:
            evidence = self.activation(preds)
        if len(tgt.shape) == 1:
            cared = tgt >= 0
            evidence = evidence[cared]
            tgt = tgt[cared]
            tgt_onehot = F.one_hot(tgt.long(), self.n_cls).float()
        elif len(tgt.shape) == 2 and tgt.shape[1] > 1:
            cared = (tgt >= 0).all(dim=-1)
            evidence = evidence[cared]
            tgt_onehot = tgt[cared]
        else:
            raise NotImplementedError
        alpha = evidence + 1
        n_cls = self.n_cls if n_cls_override is None else n_cls_override
        loss = mse_loss(tgt_onehot, alpha, temp, n_cls, self.annealing_step)

        return loss