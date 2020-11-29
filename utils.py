import torch
import numpy as np


def accuracy(output, target):
    # output - (batch, classes, height, width)
    # target - (batch, height, width)
    output = to_numpy(output)
    target = to_numpy(target)
    acc = target == output.argmax(axis=1)
    acc = acc.astype(np.float)
    acc = acc.mean()
    return acc


def to_numpy(x) -> np.ndarray:
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return x
