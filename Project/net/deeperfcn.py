import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def bilinear_kernel(in_channels, out_channels, kernel_size):
    """
    return a bilinear filter tensor
    """
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight)


class FCNNet_deep(nn.Module):

    def __init__(self):
        super(FCNNet_deep, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=20, padding=2)
        self.maxpool0 = nn.MaxPool2d(kernel_size=2)
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=20, padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=10, padding=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=10, padding=2)
        self.fc1 = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=1, padding=0)
        self.fc2 = nn.Conv2d(in_channels=8, out_channels=2, kernel_size=1, padding=0)
        self.unsample1 = nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=12, stride=2)
        self.unsample1.weight.data = bilinear_kernel(2, 2, 12)
        self.unsample2 = nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=24, stride=2)
        self.unsample2.weight.data = bilinear_kernel(2, 2, 24)
        self.unsample3 = nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=26, stride=2)
        self.unsample3.weight.data = bilinear_kernel(2, 2, 26)

    def forward(self, x):
        x = self.conv0(x)
        x = F.relu(self.maxpool0(x))
        x = self.conv1(x)
        x = F.relu(self.maxpool1(x))
        x = self.conv2(x)
        x2 = x
        x = F.relu(self.maxpool2(x))
        x = self.conv3(x)
        x = self.fc1(x)
        x = self.unsample1(x)
        x2 = self.fc2(x2)
        x = x + x2
        x = self.unsample2(x)
        x = self.unsample3(x)
        return F.log_softmax(x, dim=1)
