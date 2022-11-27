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


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 5, stride = 1, padding = 2):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size, stride = stride, padding = padding),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size, stride = stride, padding = padding),
                        nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU()
        self.out_channels = out_channels
    
    def forward(self,x):
        input = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += input
        out = self.relu(out)
        return out

class ResConv(nn.Module):
    def __init__(self):
        super(ResConv, self).__init__()
        self.convStart = nn.Conv2d(3, 20, 5, stride=1, padding=2)
        self.ResBlock1 = ResBlock(20,20,3,1,1)
        self.ResBlock2 = ResBlock(20,20,3,1,1)
        self.ResBlock3 = ResBlock(20,20,3,1,1)
        self.ResBlock4 = ResBlock(20,20,3,1,1)
        self.ResBlock5 = ResBlock(20,20,3,1,1)
        self.convEnd = nn.Conv2d(20, 3, 5, stride=1, padding=2)

    def forward(self,x):
        out = self.convStart(x)
        out = self.ResBlock1(out)
        out = self.ResBlock2(out)
        out = self.ResBlock3(out)
        out = self.ResBlock4(out)
        out = self.ResBlock5(out)
        out = self.convEnd(out)
        return out