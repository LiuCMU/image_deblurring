import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class Conv(nn.Module):
    def __init__(self):
        super(Conv, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(6, 3, 5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(5, stride=1, padding=2)

    def forward(self, x):
        x = self.maxpool1(self.relu1(self.conv1(x)))
        x = self.maxpool2(self.relu2(self.conv2(x)))
        return x

