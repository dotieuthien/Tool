import time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as functional
from PIL import Image
from models.utils import count_parameters, get_coord_features
from models.utils import gather_by_label_matrix, gather_by_one_hot_labels
from rules.component_wrapper import ComponentWrapper


BN_MOMENTUM = 0.1


def conv3x3(in_channels, out_channels, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_channels, out_channels,
        kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1, down_sample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, channels, stride)
        self.bn1 = nn.BatchNorm2d(channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(channels, channels)
        self.bn2 = nn.BatchNorm2d(channels, momentum=BN_MOMENTUM)
        self.down_sample = down_sample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.down_sample is not None:
            residual = self.down_sample(x)

        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1, down_sample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(
            channels, channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels * self.expansion, momentum=BN_MOMENTUM)

        self.relu = nn.ReLU(inplace=True)
        self.down_sample = down_sample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.down_sample is not None:
            residual = self.down_sample(x)

        out += residual
        out = self.relu(out)
        return out


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_in_channels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(num_branches, num_blocks, num_in_channels, num_channels)

        self.num_in_channels = num_in_channels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output
        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _check_branches(self, num_branches, num_blocks, num_in_channels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = "NUM_BRANCHES({}) <> NUM_BLOCKS({})".format(num_branches, len(num_blocks))
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = "NUM_BRANCHES({}) <> NUM_CHANNELS({})".format(num_branches, len(num_channels))
            raise ValueError(error_msg)

        if num_branches != len(num_in_channels):
            error_msg = "NUM_BRANCHES({}) <> NUM_IN_CHANNELS({})".format(num_branches, len(num_in_channels))
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        down_sample = None
        if stride != 1 or \
                self.num_in_channels[branch_index] != num_channels[branch_index] * block.expansion:
            down_sample = nn.Sequential(
                nn.Conv2d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(
                    num_channels[branch_index] * block.expansion,
                    momentum=BN_MOMENTUM),
            )

        layers = list()
        layers.append(block(
            self.num_in_channels[branch_index], num_channels[branch_index],
            stride, down_sample))

        self.num_in_channels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(
                self.num_in_channels[branch_index],
                num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []
        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))
        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_in_channels = self.num_in_channels
        fuse_layers = []
        # me, adding a new hyper-param
        single_scale_output_channels = 2

        for i in range(num_branches if self.multi_scale_output else single_scale_output_channels):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(num_in_channels[j], num_in_channels[i], 1, 1, 0, bias=False),
                            nn.BatchNorm2d(num_in_channels[i], momentum=BN_MOMENTUM),
                            nn.Upsample(scale_factor=2 ** (j - i), mode="bilinear", align_corners=True),
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_out_channels_conv3x3 = num_in_channels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(num_in_channels[j], num_out_channels_conv3x3, 3, 2, 1, bias=False),
                                    nn.BatchNorm2d(num_out_channels_conv3x3, momentum=BN_MOMENTUM),
                                )
                            )
                        else:
                            num_out_channels_conv3x3 = num_in_channels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(num_in_channels[j], num_out_channels_conv3x3, 3, 2, 1, bias=False),
                                    nn.BatchNorm2d(num_out_channels_conv3x3, momentum=BN_MOMENTUM),
                                    nn.ReLU(inplace=True),
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_in_channels(self):
        return self.num_in_channels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []

        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    fused_x = self.fuse_layers[i][j](x[j])
                    y = y + fused_x
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    "BASIC": BasicBlock,
    "BOTTLENECK": Bottleneck
}


class HighResolutionNet(nn.Module):
    def __init__(self, input_nc, get_z=False, config=None):
        super(HighResolutionNet, self).__init__()
        self.in_channels = input_nc
        self.get_z = get_z
        config = get_config() if config is None else config
        self.relu = nn.ReLU(inplace=True)

        # stem net
        conv_nc = config["STAGE1"]["NUM_CHANNELS"][0]
        self.conv1 = nn.Conv2d(self.in_channels + 2, conv_nc, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(conv_nc, momentum=BN_MOMENTUM)

        # stage 1
        self.stage1_cfg = config["STAGE1"]
        num_channels = self.stage1_cfg["NUM_CHANNELS"][0]
        block = blocks_dict[self.stage1_cfg["BLOCK"]]
        num_blocks = self.stage1_cfg["NUM_BLOCKS"][0]
        self.layer1 = self._make_layer(block, conv_nc, num_channels, num_blocks)
        stage1_out_channel = block.expansion * num_channels

        # stage 2
        self.stage2_config = config["STAGE2"]
        num_channels = self.stage2_config["NUM_CHANNELS"]
        block = blocks_dict[self.stage2_config["BLOCK"]]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]

        self.transition1 = self._make_transition_layer([stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_config, num_channels)

        # stage 3
        self.stage3_config = config["STAGE3"]
        num_channels = self.stage3_config["NUM_CHANNELS"]
        block = blocks_dict[self.stage3_config["BLOCK"]]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]

        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_config, num_channels)

        # stage 4
        self.stage4_config = config["STAGE4"]
        num_channels = self.stage4_config["NUM_CHANNELS"]
        block = blocks_dict[self.stage4_config["BLOCK"]]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]

        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_config, num_channels, multi_scale_output=False)

        self.reduce_global_layer = nn.Sequential(
            nn.Conv2d(pre_stage_channels[-1], config["OUT_CHANNELS"], 1, bias=False),
            nn.BatchNorm2d(config["OUT_CHANNELS"], momentum=BN_MOMENTUM),
        )
        self.merge_layer = nn.Sequential(
            nn.Conv2d(config["OUT_CHANNELS"] * 2, config["OUT_CHANNELS"] * 2, 1, bias=False),
            nn.BatchNorm2d(config["OUT_CHANNELS"] * 2, momentum=BN_MOMENTUM),
        )

        final_input_nc = pre_stage_channels[0] + pre_stage_channels[1] + 2
        self.final_layer = nn.Conv2d(
            in_channels=final_input_nc,
            out_channels=config["OUT_CHANNELS"],
            kernel_size=config["FINAL_CONV_KERNEL"],
            stride=1,
            padding=1 if config["FINAL_CONV_KERNEL"] == 3 else 0)

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i], 3, 1, 1, bias=False),
                            nn.BatchNorm2d(num_channels_cur_layer[i]),
                            nn.ReLU(inplace=True)
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    in_channels = num_channels_pre_layer[-1]
                    out_channels = num_channels_cur_layer[i] if j == i-num_branches_pre else in_channels
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=False),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU(inplace=True)
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, in_channels, channels, blocks, stride=1):
        down_sample = None
        if stride != 1 or in_channels != channels * block.expansion:
            down_sample = nn.Sequential(
                nn.Conv2d(
                    in_channels, channels * block.expansion,
                    kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = list()
        layers.append(block(in_channels, channels, stride, down_sample))
        in_channels = channels * block.expansion
        for i in range(1, blocks):
            layers.append(block(in_channels, channels))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_in_channels,
                    multi_scale_output=True):
        num_modules = layer_config["NUM_MODULES"]
        num_branches = layer_config["NUM_BRANCHES"]
        num_blocks = layer_config["NUM_BLOCKS"]
        num_channels = layer_config["NUM_CHANNELS"]
        block = blocks_dict[layer_config["BLOCK"]]
        fuse_method = layer_config["FUSE_METHOD"]

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            reset_multi_scale_output = True
            if (not multi_scale_output) and (i == num_modules - 1):
                reset_multi_scale_output = False

            modules.append(
                HighResolutionModule(
                    num_branches, block, num_blocks, num_in_channels, num_channels,
                    fuse_method, reset_multi_scale_output))
            num_in_channels = modules[-1].get_num_in_channels()

        return nn.Sequential(*modules), num_in_channels

    def forward(self, x, label):
        x = torch.cat([x, get_coord_features(x)], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_config["NUM_BRANCHES"]):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_config["NUM_BRANCHES"]):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_config["NUM_BRANCHES"]):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        # post-process y_list
        y_list_1 = functional.interpolate(y_list[1], scale_factor=2, mode="bilinear", align_corners=True)
        final_inputs = torch.cat([y_list[0], y_list_1], dim=1)
        final_inputs = torch.cat([final_inputs, get_coord_features(final_inputs)], dim=1)
        final_out = self.final_layer(final_inputs)

        # global semantics
        global_inputs = x_list[-1]
        global_out = self.reduce_global_layer(global_inputs)
        global_out = functional.interpolate(global_out, scale_factor=8, mode="bilinear", align_corners=True)

        # merged output
        merged_inputs = torch.cat([final_out, global_out], dim=1)
        merged_out = self.merge_layer(merged_inputs)
        merged_out = gather_by_one_hot_labels(merged_out, label)

        if self.get_z:
            final_out = gather_by_one_hot_labels(final_out, label)
            global_out = gather_by_one_hot_labels(global_out, label)
            return merged_out, final_out, global_out
        return merged_out


def get_config():
    config = {
        "STAGE1": {
            "NUM_MODULES": 1,
            "NUM_BRANCHES": 1,
            "NUM_BLOCKS": [2],
            "NUM_CHANNELS": [32],
            "BLOCK": "BOTTLENECK",
            "FUSE_METHOD": "SUM",
        },
        "STAGE2": {
            "NUM_MODULES": 1,
            "NUM_BRANCHES": 2,
            "NUM_BLOCKS": [2, 2],
            "NUM_CHANNELS": [32, 64],
            "BLOCK": "BASIC",
            "FUSE_METHOD": "SUM",
        },
        "STAGE3": {
            "NUM_MODULES": 1,
            "NUM_BRANCHES": 3,
            "NUM_BLOCKS": [2, 2, 2],
            "NUM_CHANNELS": [32, 64, 128],
            "BLOCK": "BASIC",
            "FUSE_METHOD": "SUM",
        },
        "STAGE4": {
            "NUM_MODULES": 1,
            "NUM_BRANCHES": 4,
            "NUM_BLOCKS": [2, 2, 2, 2],
            "NUM_CHANNELS": [32, 64, 128, 256],
            "BLOCK": "BASIC",
            "FUSE_METHOD": "SUM",
        },
        "FINAL_CONV_KERNEL": 3,
        "OUT_CHANNELS": 64,
    }
    return config


if __name__ == "__main__":
    torch.cuda.benchmark = True

    # Read data
    paths = ["/home/tyler/work/data/GeekToys/coloring_data/simple_data/E/sketch_v3/A0001.png",
             "/home/tyler/work/data/GeekToys/coloring_data/simple_data/E/color/A0001.tga"]
    sketch_image = cv2.imread(paths[0], cv2.IMREAD_GRAYSCALE)
    color_image = cv2.cvtColor(np.array(Image.open(paths[1]).convert("RGB")), cv2.COLOR_RGB2BGR)

    color_image = np.where(
        np.stack([sketch_image == 0] * 3, axis=-1), np.zeros_like(color_image),
        color_image)
    color_image = cv2.resize(color_image, (768, 512), interpolation=cv2.INTER_NEAREST)

    # extract components
    component_wrapper = ComponentWrapper()
    input_label = component_wrapper.extract_on_color_image(color_image)[0]
    input_label = torch.tensor(input_label).long().unsqueeze(0)
    print(input_label.shape, input_label.max())

    model = HighResolutionNet(3, get_z=True)
    print(count_parameters(model))

    start = time.time()
    output = model(torch.zeros([1, 3, 512, 768]), input_label)
    print(time.time() - start)
