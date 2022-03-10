import os
import time
import numpy as np
import cv2
import torch
from torch import nn
from torch.nn import functional

from PIL import Image
from loader.pooling_loader import get_component_box, get_pool_mask
from rules.component_wrapper import ComponentWrapper, resize_mask_and_fix_components
from models.utils import count_parameters, get_coord_features


def precise_pool(features, boxes, masks):
    from pooling.roi_align import ROIAlign

    roi_func = ROIAlign([1, 1], 0.5, 3)
    components = roi_func(features, boxes, masks)
    return components


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

        self.upsample = GlobalUp(512, 256, 256, 4)
        self.gamma = nn.Parameter(torch.full([1], 0.01), requires_grad=True)

        self.up1 = nn.Sequential(
            TransposeBlock(128, 128, 2, stride=2),
            ConvolutionBlock(128, 128, 3, padding=1))
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1, stride=2)

    def merge_with_global(self, x, y):
        y = self.upsample(y)
        out = x + self.gamma * y
        return out, x, y

    def forward(self, inputs, boxes, masks):
        inputs = torch.cat([inputs, get_coord_features(inputs)], dim=1)

        out = y1 = self.relu(self.conv1(inputs))
        out = y2 = self.down1(out)
        out = self.down2(out)

        out = self.relu(self.conv2(out))
        out = y3 = self.dropout(out)

        out = self.up3(out, y2)
        out = self.up2(out, y1)
        out = self.up1(out)

        out = self.conv3(out)
        out, x, y = self.merge_with_global(out, y3)

        # 4 additional channels
        out = precise_pool(out, boxes, masks)
        out = out.view(1, out.shape[0], 260)
        assert not torch.isnan(out).any()

        if self.get_z:
            x = precise_pool(x, boxes, masks)
            x = x.view(1, x.shape[0], 260)
            y = precise_pool(y, boxes, masks)
            y = y.view(1, y.shape[0], 260)
            return out, x, y
        return out
