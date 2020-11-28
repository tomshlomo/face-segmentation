import torch
import os
import numpy as np
import cv2
import albumentations as albu

rgb2name = {(255, 0, 0): 'background',
            (127, 0, 0): 'hair',
            (255, 255, 0): 'skin',
            (0, 0, 255): 'eye',
            (0, 255, 255): 'nose',
            (0, 255, 0): 'mouth'}


def read_image(path):
    img = cv2.imread(path)
    # img = img[:, :, [2, 1, 0]] # BGR to RGB
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
    # m = np.zeros((argmin.shape[0], argmin.shape[1], C))
    # for c in range(C):
    #     m[:, :, c] = (argmin == c).astype(np.float)
    return argmin


class Dataset(torch.utils.data.Dataset):
    # Characterizes a dataset for PyTorch
    def __init__(self, rgb_folder, label_folder, transform):
        # Initialization
        self.rgb_folder = rgb_folder
        self.label_folder = label_folder
        self.list_IDs = os.listdir(self.rgb_folder)
        self.transform = transform

    def __len__(self):
        # Denotes the total number of samples
        return len(self.list_IDs)

    def __getitem__(self, index):
        # Generates one sample of data
        # Select sample
        x_path = os.path.join(self.rgb_folder, self.list_IDs[index])
        y_path = os.path.join(self.label_folder, self.list_IDs[index])

        x = read_image(x_path)
        y = read_image(y_path)
        y = rgb2masks(y)
        # x = read_image(x_path)
        # y = read_image(y_path)
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


mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
ratio = 1.44
size = [int(100*r) for r in [ratio, 1]]
train_transform = albu.Compose([albu.HorizontalFlip(p=0.5),
                                albu.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                                albu.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
                                albu.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                                albu.Resize(*size),
                                albu.Normalize(mean=mean, std=std)])
val_transform = albu.Compose([albu.Resize(*size),
                              albu.Normalize(mean=mean, std=std)])

params = {'batch_size': 32,
          'shuffle': True,
          'num_workers': 0}

train_set = Dataset('data/Train_RGB', 'data/Train_Labels', train_transform)
train_loader = torch.utils.data.DataLoader(train_set, **params)

val_set = Dataset('data/Test_RGB', 'data/Test_Labels', val_transform)
val_loader = torch.utils.data.DataLoader(val_set, **params)
