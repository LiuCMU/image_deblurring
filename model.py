import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class Conv(nn.Module):
    def __init__(self, num_layers=2, conv_pad=2, hidden_channels=6, pool_pad=2):
        """
        num_layers: number of hidden sequences (conv, relu, max_pool)
        conv_pad: padding size for the hidden conv layer, kerner_size=2*conv_pad + 1
        hidden_channel: number of channels in the hidden conv layer
        pool_pad: padding size for the hidden pool layer, kerner_size=2*pool_pad + 1
        """
        super(Conv, self).__init__()
        self.conv1 = nn.Conv2d(3, hidden_channels, 5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(5, stride=1, padding=2)

        #construct hidden bolocks
        conv_kernel_size = conv_pad*2 + 1
        pool_kernel_size = pool_pad*2 + 1
        hiddens = []
        for _ in range(num_layers):
            hiddens.append(nn.Sequential(nn.Conv2d(hidden_channels, hidden_channels, conv_kernel_size, stride=1, padding=conv_pad),
                                         nn.ReLU(),
                                         nn.MaxPool2d(pool_kernel_size, stride=1, padding=pool_pad)))
        self.hiddens = nn.ModuleList(hiddens)
        self.num_layers = num_layers

        self.conv2 = nn.Conv2d(hidden_channels, 3, 5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        # self.maxpool2 = nn.MaxPool2d(5, stride=1, padding=2)

    def forward(self, x):
        x = self.maxpool1(self.relu1(self.conv1(x)))
        
        #hidden layers
        for i in range(self.num_layers):
            x = self.hiddens[i](x)

        # x = self.maxpool2(self.relu2(self.conv2(x)))
        x = self.relu2(self.conv2(x))
        return x

