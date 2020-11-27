import torch
import glob
import os
import numpy as np
from PIL import Image

def read_image(path):
    x = Image.open(path)
    x = np.asarray(x)
    x = torch.from_numpy(x)
    x = x.type(torch.float)
    return x


class Dataset(torch.utils.data.Dataset):
    # Characterizes a dataset for PyTorch
    def __init__(self, rgb_folder, label_folder):
        # Initialization
        self.rgb_folder = rgb_folder
        self.label_folder = label_folder
        self.list_IDs = os.listdir(self.rgb_folder)

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

        return x, y

params = {'batch_size': 2,
          'shuffle': False,
          'num_workers': 1}
train_set = Dataset('data/Train_RGB', 'data/Train_Labels')
train_loader = torch.utils.data.DataLoader(train_set, **params)
for x, y in train_loader:
    print(x.shape)
    print(y.shape)