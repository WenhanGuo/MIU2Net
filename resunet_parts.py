#! -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


def conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding='same', bias=False)


def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


class ConvBlock(nn.Module):
    """"""

    def __init__(self, channels, hidden_channels, out_channels=None):
        super().__init__()
        self.conv1 = conv1x1(channels, hidden_channels)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = conv3x3(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.conv3 = conv1x1(hidden_channels, out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()
        self.channels = channels
        self.out_channels = out_channels

    def forward(self, x):
        xhat = x

        xhat = self.gelu(xhat)
        xhat = self.conv1(xhat)
        xhat = self.bn1(xhat)

        xhat = self.gelu(xhat)
        xhat = self.conv2(xhat)
        xhat = self.bn2(xhat)

        xhat = self.gelu(xhat)
        xhat = self.conv3(xhat)
        xhat = self.bn3(xhat)

        return xhat


class Down(nn.Module):
    """"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.hidden_channels = hidden_channels = int(in_channels / 4) # magic number
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.block1 = ConvBlock(in_channels, hidden_channels, in_channels)
        self.block2 = ConvBlock(in_channels, hidden_channels, in_channels)
        self.block3 = ConvBlock(in_channels, hidden_channels, out_channels)
        self.conv = conv1x1(in_channels, out_channels)

    def forward(self, x):
        x1 = self.block1(x) + x
        x2 = self.block2(x1) + x1
        x3 = self.block3(x2) + self.conv(x2)
        x4 = F.avg_pool2d(x3, 2)
        return x4


class Up(nn.Module):
    """"""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.skip_channels = skip_channels
        conv_channels = in_channels + skip_channels
        self.hidden_channels = hidden_channels = int(conv_channels / 4) # magic number
        self.block1 = ConvBlock(conv_channels, hidden_channels, in_channels)
        self.block2 = ConvBlock(conv_channels, hidden_channels, in_channels)
        self.block3 = ConvBlock(conv_channels, hidden_channels, out_channels)
        self.conv = conv1x1(in_channels, out_channels)
    
    def forward(self, x, a):
        """x is input, a is skip connect activations"""
        a = F.avg_pool2d(a, 2)
        y1 = self.block1(torch.cat([a, x], dim=1)) + x
        y2 = self.block2(torch.cat([a, y1], dim=1)) + y1
        y3 = self.block3(torch.cat([a, y2], dim=1)) + self.conv(y2)
        y4 = F.interpolate(y3, scale_factor=2, mode='nearest')
        return y4
    

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    