import torch


def logit_to_edl(logits):
    """

    Parameters
    ----------
    logits: Tensor, (..., C),

    Returns
    -------

    """
    evidence = logits.relu()
    alpha = evidence + 1
    S = torch.sum(alpha, dim=-1, keepdim=True)
    conf = torch.div(alpha, S)
    K = evidence.shape[-1]
    unc = torch.div(K, S)
    # conf = torch.sqrt(conf * (1 - unc))
    unc = unc.squeeze(dim=-1)
    return conf, unc