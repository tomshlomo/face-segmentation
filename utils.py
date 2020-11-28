import torch
import numpy as np


def accuracy(output, target): #, prev_acc=0, prev_weight=0):
    # output - (batch, classes, height, width)
    # target - (batch, height, width)

    output = to_numpy(output)
    target = to_numpy(target)
    acc = target == output.argmax(axis=1)
    acc = acc.astype(np.float)
    acc = acc.mean()
    # batch_size = output.shape[0]
    # acc = acc * batch_size + prev_acc * prev_weight
    # weight = prev_weight + batch_size
    # acc /= weight
    return acc


def to_numpy(x) -> np.ndarray:
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return x
