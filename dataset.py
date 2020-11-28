from collections import OrderedDict

import torch
import os
import numpy as np
import cv2

rgb2name = OrderedDict()
rgb2name[(255, 0, 0)] = 'background'
rgb2name[(127, 0, 0)] = 'hair'
rgb2name[(255, 255, 0)] = 'skin'
rgb2name[(0, 0, 255)] = 'eye'
rgb2name[(0, 255, 255)] = 'nose'
rgb2name[(0, 255, 0)] = 'mouth'


def read_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def rgb2masks(img):
    C = len(rgb2name)
    rgb_vals = np.array(list(rgb2name.keys())).T
    rgb_vals = rgb_vals.astype(np.float)
    img = img.astype(np.float)
    d = np.abs(img[:, :, :, None] - rgb_vals[None, None, :, :])
    d = np.sum(d, axis=2)
    argmin = d.argmin(axis=2)
    return argmin


class Dataset(torch.utils.data.Dataset):
    def __init__(self, rgb_folder, label_folder, transform):
        self.rgb_folder = rgb_folder
        self.label_folder = label_folder
        self.list_IDs = os.listdir(self.rgb_folder)
        self.transform = transform
        self.num_classes = len(rgb2name)
        self.num_channels = 3

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        x_path = os.path.join(self.rgb_folder, self.list_IDs[index])
        y_path = os.path.join(self.label_folder, self.list_IDs[index])

        x = read_image(x_path)
        y = read_image(y_path)
        y = rgb2masks(y)
        aug = self.transform(image=x, mask=y)
        x = np.transpose(aug["image"], [2, 0, 1])
        y = np.transpose(aug["mask"], [0, 1])
        y = y.astype(np.long)
        return x, y

    def class_weights(self):
        params = {'batch_size': 32,
                  'shuffle': False,
                  'num_workers': 0}
        C = len(rgb2name)
        w = torch.zeros(C)
        for _, target in torch.utils.data.DataLoader(self, **params):
            for c in range(C):
                w[c] += torch.sum(target == c)
        w = 1.0 / w
        w /= torch.sum(w)
        return w
