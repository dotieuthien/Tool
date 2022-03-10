import os
import glob
import random
import json
import pickle
import numpy as np
import cv2
import torch
import torch.utils.data as data
from natsort import natsorted
from PIL import Image

from loader import matching_utils, features_utils
from loader.features_utils import convert_to_stacked_mask
from rules.component_wrapper import ComponentWrapper, resize_mask_and_fix_components
from rules.component_wrapper import get_component_color, build_neighbor_graph
from loader.data_augment import random_remove_component, add_random_noise


def get_image_by_index(paths, index):
    if index is None:
        return None
    path = paths[index]

    if path.endswith(".tga"):
        image = cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)
    else:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
    return image, path


def draw_component_image(components, mask):
    image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    for component in components:
        coords = component["coords"]
        image[coords[:, 0], coords[:, 1], :] = component["color"]

    cv2.imwrite("%d.png" % len(components), image)


def draw_matches_with_images(image_pair_name, source_image, target_image, pairs,
                             components_a, components_b, mask_a, mask_b):
    output_dir = os.path.join("/home/tyler/work/data/GeekToys/output/flow_output", image_pair_name)
    os.makedirs(output_dir, exist_ok=True)
    raw_debug_image = np.concatenate([source_image, target_image], axis=1)
    sw = source_image.shape[1]
    print(pairs.shape)

    for pair_index in range(0, pairs.shape[0]):
        comp_a = components_a[pairs[pair_index, 0]]
        comp_b = components_b[pairs[pair_index, 1]]
        pa, pb = comp_a["centroid"], comp_b["centroid"]

        debug_image = raw_debug_image.copy()
        line_color = (0, 255, 0)

        cv2.line(
            debug_image,
            (int(pa[1]), int(pa[0])),
            (int(pb[1]) + sw, int(pb[0])),
            color=line_color, thickness=3)
        cv2.circle(debug_image, (int(pa[1]), int(pa[0])), radius=6, color=(0, 0, 255), thickness=3)
        cv2.circle(debug_image, (int(pb[1]) + sw, int(pb[0])), radius=6, color=(0, 0, 255), thickness=3)

        comp_mask_a = mask_a[..., pairs[pair_index, 0]].cpu().numpy() * 255
        comp_mask_b = mask_b[..., pairs[pair_index, 1]].cpu().numpy() * 255
        comp_mask_a = np.stack([comp_mask_a] * 3, axis=-1)
        comp_mask_b = np.stack([comp_mask_b] * 3, axis=-1)
        comp_mask_a = cv2.resize(comp_mask_a, (source_image.shape[1], source_image.shape[0]))
        comp_mask_b = cv2.resize(comp_mask_b, (source_image.shape[1], source_image.shape[0]))

        debug_image = np.concatenate([debug_image, comp_mask_a, comp_mask_b], axis=1)
        cv2.imwrite(os.path.join(output_dir, "%s_%s.png" % (pairs[pair_index, 0], pairs[pair_index, 1])), debug_image)
    return


def loader_collate(batch):
    assert len(batch) == 1
    batch = batch[0]

    sketch_a = torch.tensor(batch[0]).unsqueeze(0).float()
    mask_a = torch.tensor(batch[1]).unsqueeze(0).float()
    graph_a = torch.tensor(batch[2]).unsqueeze(0).float()

    sketch_b = torch.tensor(batch[3]).unsqueeze(0).float()
    mask_b = torch.tensor(batch[4]).unsqueeze(0).float()
    graph_b = torch.tensor(batch[5]).unsqueeze(0).float()

    positive_pairs = torch.tensor(batch[6]).unsqueeze(0).long()
    colors_a = torch.tensor(batch[7]).unsqueeze(0).int()
    colors_b = torch.tensor(batch[8]).unsqueeze(0).int()

    input_a = (sketch_a, mask_a, graph_a, colors_a)
    input_b = (sketch_b, mask_b, graph_b, colors_b)
    return input_a, input_b, positive_pairs


class PairAnimeDataset(data.Dataset):
    def __init__(self, root_dir, size, config):
        super(PairAnimeDataset, self).__init__()
        self.root_dir = root_dir
        self.size = size
        self.gt_type = config.gt_type
        self.feature_type = config.feature_type

        self.paths = {}
        self.lengths = {}

        dirs = natsorted(glob.glob(os.path.join(root_dir, "*")))
        match_data_path = os.path.join(root_dir, "matches.json")
        self.match_data = json.load(open(match_data_path))

        self.component_wrapper = ComponentWrapper()
        self.all_sketch_set_names = ["sketch_v1", "sketch_v2", "sketch_v3", "sketch_v4"]
        all_set_names = self.all_sketch_set_names + ["color"]

        for sub_dir in dirs:
            dir_name = os.path.basename(sub_dir)
            self.paths[dir_name] = {}

            if not os.path.isdir(sub_dir):
                continue

            for set_name in all_set_names:
                paths = []
                for sub_type in ["png", "jpg", "tga"]:
                    sub_paths = glob.glob(os.path.join(sub_dir, set_name, "*.%s" % sub_type))
                    paths.extend(sub_paths)
                self.paths[dir_name][set_name] = natsorted(paths)

            self.lengths[dir_name] = len(self.paths[dir_name]["color"])
        return

    def __len__(self):
        total = 0
        for key, count in self.lengths.items():
            total += count
        return total

    def get_component_mask(self, color_image, sketch, path, method):
        name = os.path.splitext(os.path.basename(path))[0]
        save_path = os.path.join(os.path.dirname(path), "%s_%s.pkl" % (name, method))

        if not os.path.exists(save_path):
            mask, components = self.component_wrapper.process(color_image, sketch, method)
            get_component_color(components, color_image, method)

            save_data = {"mask": mask, "components": components}
            pickle.dump(save_data, open(save_path, "wb+"))
        else:
            save_data = pickle.load(open(save_path, "rb"))
            mask, components = save_data["mask"], save_data["components"]

        mask, components, is_removed = random_remove_component(mask, components)
        mask, components = resize_mask_and_fix_components(mask, components, self.size)
        return mask, components, is_removed

    def __getitem__(self, index):
        name = None
        for key, length in self.lengths.items():
            if index < length:
                name = key
                break
            index -= length

        length = len(self.paths[name]["color"])
        k = random.choice([1, 1, 1, 2])
        next_index = max(index - 2, 0) if index == length - 1 else min(index + k, length - 1)

        # read images
        color_a, path_a = get_image_by_index(self.paths[name]["color"], index)
        color_b, path_b = get_image_by_index(self.paths[name]["color"], next_index)

        sketch_set_name = random.choice(self.all_sketch_set_names)
        sketch_a = get_image_by_index(self.paths[name][sketch_set_name], index)[0]
        sketch_b = get_image_by_index(self.paths[name][sketch_set_name], next_index)[0]

        if random.random() < 0.2:
            component_method = ComponentWrapper.EXTRACT_SKETCH
        else:
            component_method = ComponentWrapper.EXTRACT_COLOR
        # extract components
        mask_a, components_a, is_removed_a = self.get_component_mask(color_a, sketch_a, path_a, component_method)
        mask_b, components_b, is_removed_b = self.get_component_mask(color_b, sketch_b, path_b, component_method)
        is_removed = is_removed_a or is_removed_b

        graph_a = build_neighbor_graph(mask_a)[1:, 1:]
        graph_b = build_neighbor_graph(mask_b)[1:, 1:]

        # get image pair signature
        cut_name = os.path.basename(os.path.dirname(os.path.dirname(path_a)))
        part_name = os.path.basename(os.path.dirname(path_a))
        key_name = "%s_%s" % (cut_name, part_name)
        source_name = os.path.splitext(os.path.basename(path_a))[0]
        target_name = os.path.splitext(os.path.basename(path_b))[0]
        image_pair_name = "%s_%s_%s" % (key_name, source_name, target_name)

        # component matching
        if self.gt_type == matching_utils.GT_FROM_JSON:
            if (component_method == ComponentWrapper.EXTRACT_COLOR) and\
                    (image_pair_name in self.match_data) and\
                    (not is_removed):
                positive_pairs = matching_utils.get_pairs_from_json(self.match_data, image_pair_name)
            else:
                positive_pairs = matching_utils.get_pairs_three_stage(components_a, components_b)
        elif self.gt_type == matching_utils.GT_ONLINE:
            positive_pairs = matching_utils.get_pairs_three_stage(components_a, components_b)
        else:
            raise Exception("Invalid gt_type")

        # component color
        colors_a = [a["color"] for a in components_a]
        colors_b = [b["color"] for b in components_b]
        colors_a, colors_b = np.array(colors_a), np.array(colors_b)

        if len(components_a) == 0 or len(components_b) == 0:
            print("Invalid component data:", image_pair_name)
        elif positive_pairs.shape[0] == 0:
            print("No matching pairs found:", image_pair_name)

        # get input features
        if self.feature_type == features_utils.SKETCH_FEATURES:
            sketch_a = features_utils.get_sketch_image_raw_pixels(sketch_a, self.size)
            sketch_b = features_utils.get_sketch_image_raw_pixels(sketch_b, self.size)
        elif self.feature_type == features_utils.MOMENT_FEATURES:
            sketch_a = features_utils.get_moment_features(components_a, mask_a)
            sketch_b = features_utils.get_moment_features(components_b, mask_b)
        else:
            raise Exception("Invalid feature_type")

        # augment sketch with noise
        sketch_a = add_random_noise(sketch_a, mask_a, self.feature_type)
        sketch_b = add_random_noise(sketch_b, mask_b, self.feature_type)
        # convert to stacked mask
        mask_a = convert_to_stacked_mask(mask_a)
        mask_b = convert_to_stacked_mask(mask_b)

        output = (
            sketch_a, mask_a, graph_a,
            sketch_b, mask_b, graph_b,
            positive_pairs, colors_a, colors_b,
        )
        return output
