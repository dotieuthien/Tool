import os
import glob
import random
import pickle
import numpy as np
import cv2
import torch
import torch.utils.data as data
from natsort import natsorted
from PIL import Image

from loader import features_utils, matching_utils
from rules.component_wrapper import ComponentWrapper, resize_mask_and_fix_components
from rules.component_wrapper import get_component_color, build_neighbor_graph


def get_image_by_index(paths, index):
    if index is None:
        return None
    path = paths[index]

    if path.endswith("tga"):
        image = cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)
    else:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return image, path


def draw_component_image(components, mask):
    image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    for component in components:
        coords = component["coords"]
        image[coords[:, 0], coords[:, 1], :] = component["color"]

    cv2.imwrite("%d.png" % len(components), image)


def loader_collate(batch):
    assert len(batch) == 1
    batch = batch[0]

    features_a = torch.tensor(batch[0]).unsqueeze(0).float()
    boxes_a = torch.tensor(batch[1]).float()
    mask_a = torch.tensor(batch[2]).float()
    graph_a = torch.tensor(batch[3]).unsqueeze(0).float()

    features_b = torch.tensor(batch[4]).unsqueeze(0).float()
    boxes_b = torch.tensor(batch[5]).float()
    mask_b = torch.tensor(batch[6]).float()
    graph_b = torch.tensor(batch[7]).unsqueeze(0).float()

    positive_pairs = torch.tensor(batch[8]).unsqueeze(0).int()
    colors_a = torch.tensor(batch[11]).unsqueeze(0).int()
    colors_b = torch.tensor(batch[12]).unsqueeze(0).int()

    output_a = (features_a, boxes_a, mask_a, graph_a, colors_a, batch[9])
    output_b = (features_b, boxes_b, mask_b, graph_b, colors_b, batch[10])
    return output_a, output_b, positive_pairs


def get_component_box(components, image, size):
    def get_box(cm):
        box = cm["bbox"]
        return [box[1], box[0], box[3], box[2]]

    # now boxes are in yx format, change to xy format
    boxes = np.array([[0.0] + get_box(c) for i, c in enumerate(components)], dtype=np.float)

    h, w = image.shape[:2]
    ratio = [size[0] / w, size[1] / h]

    boxes[:, 1] = boxes[:, 1] * ratio[0]
    boxes[:, 2] = boxes[:, 2] * ratio[1]
    boxes[:, 3] = boxes[:, 3] * ratio[0]
    boxes[:, 4] = boxes[:, 4] * ratio[1]
    return boxes


def get_pool_mask(components, sampling_size=3):
    masks = np.zeros([len(components), sampling_size, sampling_size], dtype=np.float)
    for i, component in enumerate(components):
        mask = component["image"]
        mask = cv2.resize(mask, (sampling_size, sampling_size), cv2.INTER_LINEAR)
        mask = cv2.threshold(mask, 20, 255, cv2.THRESH_BINARY)[1]

        if sampling_size == 1:
            mask[0, 0] = 255
        masks[i] = mask / 255.0
    return masks


class PairAnimeDataset(data.Dataset):
    def __init__(self, root_dir, size, mean, std):
        super(PairAnimeDataset, self).__init__()
        self.root_dir = root_dir
        self.size = size
        self.mean = mean
        self.std = std

        self.paths = {}
        self.lengths = {}
        dirs = natsorted(glob.glob(os.path.join(root_dir, "*")))

        self.component_wrapper = ComponentWrapper()

        for sub_dir in dirs:
            dir_name = os.path.basename(sub_dir)
            self.paths[dir_name] = {}

            for set_name in ["sketch_v3", "color"]:
                paths = []
                for sub_type in ["png", "jpg", "tga"]:
                    paths.extend(glob.glob(os.path.join(sub_dir, set_name, "*.%s" % sub_type)))
                self.paths[dir_name][set_name] = natsorted(paths)

            self.lengths[dir_name] = len(self.paths[dir_name]["color"])
        return

    def __len__(self):
        total = 0
        for key, count in self.lengths.items():
            total += count
        return total

    def get_component_mask(self, color_image, sketch, path, extract_prob=0.4):
        if random.random() < extract_prob:
            method = ComponentWrapper.EXTRACT_SKETCH
        else:
            method = ComponentWrapper.EXTRACT_COLOR

        name = os.path.splitext(os.path.basename(path))[0]
        save_path = os.path.join(os.path.dirname(path), "%s_%s.pkl" % (name, method))

        if not os.path.exists(save_path):
            mask, components = self.component_wrapper.process(color_image, sketch, method)
            get_component_color(components, color_image, ComponentWrapper.EXTRACT_COLOR)
            mask = resize_mask_and_fix_components(mask, components, self.size).astype(np.int32)
            save_data = {"mask": mask, "components": components}
            pickle.dump(save_data, open(save_path, "wb+"))
        else:
            save_data = pickle.load(open(save_path, "rb"))
            mask, components = save_data["mask"], save_data["components"]

        return mask, components

    def __getitem__(self, index):
        name = None
        for key, length in self.lengths.items():
            if index < length:
                name = key
                break
            index -= length

        length = len(self.paths[name]["color"])
        k = 1
        next_index = max(index - k, 0) if index == length - 1 else min(index + k, length - 1)

        # read images
        color_a, path_a = get_image_by_index(self.paths[name]["color"], index)
        color_b, path_b = get_image_by_index(self.paths[name]["color"], next_index)

        sketch_a = get_image_by_index(self.paths[name]["sketch_v3"], index)[0]
        sketch_b = get_image_by_index(self.paths[name]["sketch_v3"], next_index)[0]

        # extract components
        mask_a, components_a = self.get_component_mask(color_a, sketch_a, path_a)
        mask_b, components_b = self.get_component_mask(color_b, sketch_b, path_b)

        # component matching
        positive_pairs = matching_utils.get_pairs_three_stage(components_a, components_b)
        positive_pairs = np.array(positive_pairs)

        # component color
        colors_a = [a["color"] for a in components_a]
        colors_b = [b["color"] for b in components_b]
        colors_a, colors_b = np.array(colors_a), np.array(colors_b)

        if len(positive_pairs) == 0:
            print(name, index, next_index)
        if len(components_a) == 0 or len(components_b) == 0:
            print(name, index, next_index)
        if np.max(mask_a) == 0 or np.max(mask_b) == 0:
            print(name, index)

        # get features
        features_a = features_utils.get_moment_features(components_a, mask_a)
        features_a = (features_a - self.mean) / self.std

        features_b = features_utils.get_moment_features(components_b, mask_b)
        features_b = (features_b - self.mean) / self.std

        # get bounding boxes
        boxes_a = get_component_box(components_a, color_a, self.size)
        boxes_b = get_component_box(components_b, color_b, self.size)
        pool_mask_a = get_pool_mask(components_a, sampling_size=1)
        pool_mask_b = get_pool_mask(components_b, sampling_size=1)
        graph_a = build_neighbor_graph(mask_a)[1:, 1:]
        graph_b = build_neighbor_graph(mask_b)[1:, 1:]

        output = (
            features_a, boxes_a, pool_mask_a, graph_a,
            features_b, boxes_b, pool_mask_b, graph_b,
            positive_pairs, components_a, components_b, colors_a, colors_b,
        )
        return output
