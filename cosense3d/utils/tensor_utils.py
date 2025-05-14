import torch
import numpy as np


def pad_list_to_array_torch(data):
    """
    Pad list of numpy data to one single numpy array
    :param data: list of np.ndarray
    :return: np.ndarray
    """
    B = len(data)
    cnt = [len(d) for d in data]
    max_cnt = max(cnt)
    out = torch.zeros((B, max_cnt,) + tuple(data[0].shape[1:]),
                      device=data[0].device, dtype=data[0].dtype)
    for b in range(B):
        out[b, :cnt[b]] = data[b]
    return out


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


