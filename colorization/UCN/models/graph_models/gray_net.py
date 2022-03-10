import os
import numpy as np
import cv2
import torch
from torch import nn
from PIL import Image

from loader.pooling_loader import get_component_box, get_pool_mask
from rules.component_wrapper import ComponentWrapper, build_neighbor_graph
from models.graph_models.gconv import GC
from models.utils import count_parameters, l2_normalize


def precise_pool(features, boxes, masks):
    from pooling.roi_align import ROIAlign

    roi_func = ROIAlign([3, 3], 0.125, 3)
    components = roi_func(features, boxes, masks)
    return components


class Bottleneck(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 1, stride=1, padding=0, bias=False)
        self.bn1 = nn.InstanceNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.InstanceNorm2d(hidden_channels)
        self.conv3 = nn.Conv2d(hidden_channels, out_channels, 1, stride=1, padding=0, bias=False)
        self.bn3 = nn.InstanceNorm2d(out_channels)

        if stride != 1 or (in_channels != out_channels):
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0, bias=False),
                nn.InstanceNorm2d(out_channels))
        else:
            self.downsample = None

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = out + residual
        out = self.relu(out)
        return out


class GrayNet(nn.Module):
    def __init__(self):
        super(GrayNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=4, dilation=2, bias=False)
        self.bn1 = nn.InstanceNorm2d(32)
        self.max_pool = nn.MaxPool2d(2)

        self.layer1 = self.build_layer(32, 32, 128, 1, 2)
        self.layer2 = self.build_layer(128, 64, 256, 2, 3)
        self.layer3 = self.build_layer(256, 128, 512, 2, 5)
        self.layer4 = self.build_layer(512, 256, 1024, 2, 4)
        self.layer5 = self.build_layer(1024, 256, 1024, 2, 4)

        self.conv2 = nn.Sequential(
            nn.Conv2d(1024, 256, 1, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True))

        self.fc = nn.Sequential(
            nn.Conv2d(260, 128, 3, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True))
        self.gc = GC(128, 128)

    def build_layer(self, in_channels, hidden_channels, out_channels, stride, n):
        layers = nn.ModuleList([Bottleneck(in_channels, hidden_channels, out_channels, stride=stride)])
        layers.extend([Bottleneck(out_channels, hidden_channels, out_channels) for _ in range(1, n)])
        return nn.Sequential(*layers)

    def forward(self, x, boxes, masks, adj):
        out = self.bn1(self.conv1(x))
        out = self.max_pool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.conv2(out)
        out = self.conv3(out)

        out = precise_pool(out, boxes, masks)
        out = self.fc(out)
        out = out.view(1, out.shape[0], out.shape[1])
        out = self.gc(adj, out)

        out = l2_normalize(out)
        assert not torch.isnan(out).any()
        return out


def main():
    torch.cuda.benchmark = True
    master_size = (768, 512)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    if os.name == "nt":
        paths = ["D:/Data/GeekToys/coloring_data/simple_data/E/sketch_v3/A0001.png",
                 "D:/Data/GeekToys/coloring_data/simple_data/E/color/A0001.tga"]
    else:
        paths = ["/home/tyler/projects/painting/simple_data/E/sketch_v3/A0001.png",
                 "/home/tyler/projects/painting/simple_data/E/color/A0001.tga"]

    sketch_image = cv2.imread(paths[0], cv2.IMREAD_GRAYSCALE)
    color_image = cv2.cvtColor(np.array(Image.open(paths[1]).convert("RGB")), cv2.COLOR_RGB2BGR)

    # extract components
    component_wrapper = ComponentWrapper()
    input_label, input_components = component_wrapper.process(color_image, None, ComponentWrapper.EXTRACT_COLOR)

    input_components = [input_components[137]]
    input_boxes = get_component_box(input_components, color_image, master_size)
    input_pool_masks = get_pool_mask(input_components)

    input_graph = build_neighbor_graph(input_label)[1:, 1:]
    input_graph = np.array([[1]])

    print(input_boxes)
    print(input_label.shape, input_label.min(), input_label.max())
    print(input_boxes.shape, input_pool_masks.shape)

    input_label = torch.from_numpy(input_label).long().unsqueeze(0).to(device)
    input_boxes = torch.from_numpy(input_boxes).float().to(device)
    input_pool_masks = torch.from_numpy(input_pool_masks).float().to(device)
    input_graph = torch.from_numpy(input_graph).float().unsqueeze(0).to(device)

    model = GrayNet().to(device)
    model.eval()
    print(count_parameters(model))

    image = torch.zeros([1, 1, master_size[1], master_size[0]]).to(device)
    print(input_boxes)
    print(input_pool_masks)
    print(input_graph)
    output = model(image.to(device), input_boxes, input_pool_masks, input_graph)
    print(output.size())


def try_roi_pooling():
    from pooling.roi_align import ROIAlign

    torch.cuda.benchmark = True
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    features = np.zeros([1, 1, 256, 384])
    # [75, 124], [200, 273]
    # 0.3755, 0.3106, 0.4697, 0.3884
    features[0, 0, 75, 200] = 1
    features[0, 0, 75, 273] = 1
    features[0, 0, 124, 200] = 1
    features[0, 0, 124, 273] = 1
    features = torch.tensor(features, requires_grad=True).float().to(device)

    gt = torch.zeros([1, 5, 3, 3], dtype=torch.float, requires_grad=False).to(device)
    boxes = torch.tensor([[0.0000, 399.3246, 150.1538, 546.3822, 248.0684]]).to(device)
    masks = torch.tensor([[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]).to(device)

    roi_func = ROIAlign([3, 3], 0.5, 3).to(device)
    components = roi_func(features, boxes, masks)
    print(components[0, 0, ...])

    loss = components - gt
    loss.backward([components])


if __name__ == "__main__":
    try_roi_pooling()
