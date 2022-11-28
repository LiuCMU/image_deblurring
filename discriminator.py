import torch.nn as nn

class Discriminator_init(nn.Module):
    def __init__(self):
        super(Discriminator_init, self).__init__()

        self.conv_layers = nn.ModuleList([
            nn.Conv2d(3, 32, 5, stride=1, padding=2),
            nn.Conv2d(32, 32, 5, stride=2, padding=2),    # in_channels, out_channels, kernel_size, stride
            nn.Conv2d(32, 64, 5, stride=1, padding=2),   # 128
            nn.Conv2d(64, 64, 5, stride=2, padding=2),
            nn.Conv2d(64, 128, 5, stride=1, padding=2),     # 64
            nn.Conv2d(128, 128, 5, stride=4, padding=2),
            nn.Conv2d(128, 256, 5, stride=1, padding=2),     # 16
            nn.Conv2d(256, 256, 5, stride=4, padding=2),
            nn.Conv2d(256, 512, 5, stride=1, padding=2),     # 4
            nn.Conv2d(512, 512, 4, stride=4, padding=0),
        ])

        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.dense = nn.Conv2d(512, 1, 1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(15,1)
        self.sigm = nn.Sigmoid()

    def forward(self, x):

        for layer in self.conv_layers:
            x = self.act(layer(x))

        x = self.dense(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.sigm(x)

        return x # tensor: (1, 1)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv_layers = nn.ModuleList([
            nn.Conv2d(3, 32, 5, stride=1, padding=2),
            nn.Conv2d(32, 64, 5, stride=1, padding=2),   # 128
            nn.Conv2d(64, 128, 5, stride=4, padding=2),     # 64
            nn.Conv2d(128, 128, 5, stride=4, padding=2),    # 16
        ])

        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.dense = nn.Conv2d(128, 1, 1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(3600,1)
        self.sigm = nn.Sigmoid()

    def forward(self, x):

        for layer in self.conv_layers:
            x = self.act(layer(x))
            # print('shape: ', x.shape)

        x = self.dense(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.sigm(x)

        return x
