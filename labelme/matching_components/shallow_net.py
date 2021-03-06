import torch
import torch.nn as nn
import torch.nn.functional as functional

from labelme.matching_components.shallow_net_utils import get_coord_features
from labelme.matching_components.shallow_net_utils import gather_by_one_hot_labels, gather_by_label


class ConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1, padding=0, rate=1):
        super(ConvolutionBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel, bias=True,
            stride=stride, padding=padding, dilation=rate)
        self.bn = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        return out


class TransposeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1, padding=0):
        super(TransposeBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel, stride=stride, padding=padding)
        self.bn = nn.InstanceNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class DownBlock(nn.Module):
    def __init__(self, in_channels, features_dim, kernel=3, padding=1):
        super(DownBlock, self).__init__()
        self.features_dim = features_dim
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.downsample = nn.Sequential(
            nn.MaxPool2d(2),
            ConvolutionBlock(in_channels, features_dim, 1))
        self.conv = ConvolutionBlock(features_dim, features_dim, kernel, padding=padding)

    def forward(self, x):
        x = self.downsample(x)
        x = self.relu(x)

        out = self.conv(x)
        out = self.relu(out + x)
        return out


class Upsampling2d(nn.Module):
    def __init__(self, size=None, scale_factor=None):
        super(Upsampling2d, self).__init__()
        self.size = size
        self.scale_factor = scale_factor

    def forward(self, x):
        out = functional.interpolate(
            x, size=self.size, scale_factor=self.scale_factor,
            mode="bilinear", align_corners=True)
        return out


class UpBlock(nn.Module):
    def __init__(self, in_channels, features_dim, kernel=3, padding=1):
        super(UpBlock, self).__init__()
        self.features_dim = features_dim
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.upsample = nn.Sequential(
            Upsampling2d(scale_factor=2),
            ConvolutionBlock(in_channels, features_dim, 1))
        self.merge = ConvolutionBlock(features_dim, features_dim, 1)
        self.conv = ConvolutionBlock(features_dim, features_dim, kernel, padding=padding)

    def forward(self, x, y):
        x = self.upsample(x)
        z = x + y
        z = self.merge(z)
        z = self.relu(z)

        out = self.conv(z)
        out = self.relu(out + z)
        return out


class GlobalUp(nn.Module):
    def __init__(self, in_channels, features_dim, out_channels, scale_factor):
        super(GlobalUp, self).__init__()
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.reduce1 = ConvolutionBlock(in_channels, features_dim, 3, padding=1)
        self.conv1 = ConvolutionBlock(features_dim, features_dim, 3, padding=1)

        self.reduce2 = ConvolutionBlock(features_dim, out_channels, 3, padding=1)
        self.conv2 = ConvolutionBlock(out_channels, out_channels, 3, padding=1)
        self.conv3 = ConvolutionBlock(out_channels, out_channels, 3, padding=1)
        self.upsample = Upsampling2d(scale_factor=scale_factor)

    def forward(self, x):
        out = self.relu(self.reduce1(x))
        out = self.relu(self.conv1(out))

        x = self.relu(self.reduce2(out))
        out = self.relu(self.conv2(x))
        out = self.conv3(out)
        out = self.relu(out + x)

        out = self.upsample(out)
        return out


class UNet(nn.Module):
    def __init__(self, in_channels, dropout, get_z=False):
        super(UNet, self).__init__()
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.get_z = get_z

        self.conv1 = ConvolutionBlock(in_channels + 2, 128, 7, padding=3, stride=2)
        self.down1 = DownBlock(128, 256)
        self.down2 = DownBlock(256, 512)

        self.conv2 = ConvolutionBlock(512, 512, 3, padding=1)
        self.up3 = UpBlock(512, 256)
        self.up2 = UpBlock(256, 128)

        self.upsample = GlobalUp(512, 256, 128, 8)
        self.gamma = nn.Parameter(torch.full([1], 0.01), requires_grad=True)

        self.up1 = nn.Sequential(
            TransposeBlock(128, 128, 2, stride=2),
            ConvolutionBlock(128, 128, 3, padding=1))
        self.conv3 = nn.Conv2d(128, 128, 3, padding=1)

    def merge_with_global(self, x, y):
        y = self.upsample(y)
        out = x + self.gamma * y
        return out, x, y

    def forward(self, inputs, labels):
        inputs = torch.cat([inputs, get_coord_features(inputs)], dim=1)

        out = y1 = self.relu(self.conv1(inputs))
        out = y2 = self.down1(out)
        out = self.down2(out)

        out = self.relu(self.conv2(out))
        out = y3 = self.dropout(out)

        out = self.up3(out, y2)
        out = self.up2(out, y1)
        out = self.up1(out)

        out, x, y = self.merge_with_global(out, y3)
        out = self.conv3(out)

        out = gather_by_one_hot_labels(out, labels)
        assert not torch.isnan(out).any()

        if self.get_z:
            x = gather_by_one_hot_labels(x, labels)
            y = gather_by_one_hot_labels(y, labels)
            return out, x, y
        return out
