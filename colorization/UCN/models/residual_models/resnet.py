import time
import torch
from torch import nn
from models.utils import count_parameters


def conv_layer(in_planes, out_planes, kernel_size, stride=1, padding=0, groups=1, dilation=1):
    return nn.Conv2d(
        in_planes, out_planes,
        kernel_size=kernel_size, stride=stride, padding=padding,
        groups=groups, dilation=dilation, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes, planes,
        stride=1, downsample=None, groups=1,
        base_width=64, dilation=1, norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv_layer(in_planes, planes, 3, stride, padding=1)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv_layer(planes, planes, 3, padding=1)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self, in_planes, planes,
        stride=1, downsample=None, groups=1,
        base_width=64, dilation=1, norm_layer=None,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv_layer(in_planes, width, 1)
        self.bn1 = norm_layer(width)
        self.conv2 = conv_layer(width, width, 3, stride, dilation, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv_layer(width, planes * self.expansion, 1)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self, block, layers, groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.in_planes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        assert len(replace_stride_with_dilation) == 3

        self.groups = groups
        self.base_width = width_per_group

        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv1 = conv_layer(3, self.in_planes, 7, stride=2, padding=3)
        self.bn1 = norm_layer(self.in_planes)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2,
            dilate=replace_stride_with_dilation[0])

        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2,
            dilate=replace_stride_with_dilation[1])

        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2,
            dilate=replace_stride_with_dilation[2])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.modules():
            if isinstance(m, Bottleneck):
                nn.init.constant_(m.bn3.weight, 0)
        return

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                conv_layer(self.in_planes, planes * block.expansion, 1, stride),
                norm_layer(planes * block.expansion),
            )

        layers = [
            block(
                self.in_planes, planes, stride, downsample, self.groups,
                self.base_width, previous_dilation, norm_layer),
        ]
        self.in_planes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(
                block(
                    self.in_planes, planes, groups=self.groups,
                    base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        y1 = out = self.relu(out)
        out = self.max_pool(out)

        y2 = out = self.layer1(out)
        y3 = out = self.layer2(out)

        y4 = out = self.layer3(out)
        y5 = out = self.layer4(out)

        return y1, y2, y3, y4, y5


def resnet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def resnet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def main():
    torch.cuda.benchmark = True

    model = resnet34()
    model.eval()
    print(count_parameters(model))

    weight_path = "/home/tyler/work/data/ToeiAni/weights_house/resnet34.pth"
    model.load_state_dict(torch.load(weight_path), strict=False)

    start = time.time()
    outputs = model(torch.zeros([1, 3, 512, 768]))
    print([out.shape for out in outputs])
    print(time.time() - start)


if __name__ == "__main__":
    main()
