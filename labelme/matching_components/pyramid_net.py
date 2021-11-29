import torch
from torch import nn
from torch.nn import functional

from labelme.matching_components.resnet import resnet34
from labelme.matching_components.shallow_net_utils import get_coord_features
from labelme.matching_components.shallow_net_utils import gather_by_one_hot_labels


def scaled_gather_by_labels(features, labels):
    size = (labels.shape[1], labels.shape[2])
    features = functional.interpolate(features, size=size, mode="bilinear", align_corners=True)
    cf = gather_by_one_hot_labels(features, labels)
    return cf


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
    def __init__(self, in_planes_top, in_planes_bot, out_planes, kernel_size=3, padding=1):
        super(UpBlock, self).__init__()
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.branch_top = nn.Sequential(
            Upsampling2d(scale_factor=2),
            nn.Conv2d(in_planes_top, out_planes, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_planes))
        self.branch_bot = nn.Sequential(
            nn.Conv2d(in_planes_bot, out_planes, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_planes))
        self.merge_layer = nn.Sequential(
            nn.Conv2d(out_planes + 2, out_planes, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_planes))

    def forward(self, x, y):
        x = self.relu(self.branch_top(x))
        y = self.relu(self.branch_bot(y))

        out = x + y
        out = torch.cat([out, get_coord_features(out)], dim=1)
        out = self.relu(self.merge_layer(out))
        return out


class UNet(nn.Module):
    def __init__(self, base_weight_path, dropout_ratio):
        super(UNet, self).__init__()
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(dropout_ratio)

        self.conv_model = resnet34()
        if base_weight_path is not None:
            self.conv_model.load_state_dict(torch.load(base_weight_path), strict=False)

        self.up1 = UpBlock(512, 256, 128)
        self.up2 = UpBlock(128, 128, 128)
        self.up3 = UpBlock(128, 64, 128)
        self.up4 = UpBlock(128, 64, 128)
        self.up5 = UpBlock(128, 3, 128)

        self.cf_reduction_layer = nn.Linear(640, 256)

    def forward(self, image, labels):
        conv_features = self.conv_model(image)
        last_conv_features = self.dropout(conv_features[4])

        pf1 = self.up1(last_conv_features, conv_features[3])
        pf2 = self.up2(pf1, conv_features[2])
        pf3 = self.up3(pf2, conv_features[1])
        pf4 = self.up4(pf3, conv_features[0])
        pf5 = self.up5(pf4, image)

        cf1 = scaled_gather_by_labels(pf1, labels)
        cf2 = scaled_gather_by_labels(pf2, labels)
        cf3 = scaled_gather_by_labels(pf3, labels)
        cf4 = scaled_gather_by_labels(pf4, labels)
        cf5 = scaled_gather_by_labels(pf5, labels)

        all_cf = torch.cat([cf1, cf2, cf3, cf4, cf5], dim=2)
        out_cf = self.cf_reduction_layer(all_cf)
        return out_cf
