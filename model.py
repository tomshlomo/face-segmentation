from collections import OrderedDict

import torch.nn as nn


class Simplest(nn.Module):
    def __init__(self, n_channels=3, n_classes=6):
        super(Simplest, self).__init__()
        self.conv = nn.Conv2d(n_channels, n_classes, 3, padding=1)

    def forward(self, x):
        x = self.conv(x)
        return x


class CNN(nn.Module):
    def __init__(self, n_channels=3, n_classes=6):
        super(CNN, self).__init__()
        hidden_channels = 64
        self.seq = model = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(n_channels, hidden_channels, 3, padding=1)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)),
            ('relu2', nn.ReLU()),
            ('conv3', nn.Conv2d(hidden_channels, n_classes, 3, padding=1))
        ]))

    def forward(self, x):
        x = self.seq(x)
        return x
