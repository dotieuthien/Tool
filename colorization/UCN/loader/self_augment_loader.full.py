import os
import glob
import random
import pickle
import numpy as np
import cv2
import torch
import torch.utils.data as data
import torch.nn.functional as functional
from natsort import natsorted
from PIL import Image

from loader import matching_utils, features_utils
from loader.features_utils import convert_to_stacked_mask
from rules.component_wrapper import ComponentWrapper, resize_mask_and_fix_components
from rules.component_wrapper import get_component_color, build_neighbor_graph
from loader.data_augment import add_random_noise


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


def do_self_augment(image, mask, k_theta=0.03):
    theta = np.zeros(9)
    theta[0:6] = np.random.randn(6) * k_theta
    theta = theta + np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])
    theta = np.reshape(theta, (3, 3))

    theta = np.reshape(theta, -1)[0:6]
    theta = torch.tensor(theta).float()
    theta = theta.view(-1, 2, 3)

    image = image.unsqueeze(dim=0)
    grid = functional.affine_grid(theta, image.shape, align_corners=True)
    image = functional.grid_sample(image, grid, align_corners=True)
    image = image.squeeze(dim=0)

    for i in range(0, mask.shape[2]):
        temp_mask = mask[..., i].float()
        temp_mask = temp_mask.view(1, 1, temp_mask.shape[0], temp_mask.shape[1])
        temp_mask = functional.grid_sample(temp_mask, grid, align_corners=True)
        temp_mask = temp_mask.squeeze()
        mask[..., i] = (temp_mask > 0.5).float()
    return image, mask


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

    sketch_a = batch[0].unsqueeze(0).float()
    mask_a = batch[1].unsqueeze(0).float()
    graph_a = torch.tensor(batch[2]).unsqueeze(0).float()
    kpts_a = torch.tensor(batch[3]).unsqueeze(0).float()

    sketch_b = batch[4].unsqueeze(0).float()
    mask_b = batch[5].unsqueeze(0).float()
    graph_b = torch.tensor(batch[6]).unsqueeze(0).float()
    kpts_b = torch.tensor(batch[7]).unsqueeze(0).float()

    positive_pairs = torch.tensor(batch[8]).unsqueeze(0).long()
    colors_a = torch.tensor(batch[9]).unsqueeze(0).int()
    colors_b = torch.tensor(batch[10]).unsqueeze(0).int()

    input_a = (sketch_a, mask_a, graph_a, kpts_a, colors_a)
    input_b = (sketch_b, mask_b, graph_b, kpts_b, colors_b)
    return input_a, input_b, positive_pairs


class PairAnimeDataset(data.Dataset):
    def __init__(self, root_dir, size, config):
        super(PairAnimeDataset, self).__init__()
        self.root_dir = root_dir
        self.size = size

        self.paths = {}
        self.lengths = {}

        dirs = natsorted(glob.glob(os.path.join(root_dir, "*")))
        self.component_wrapper = ComponentWrapper()
        self.all_sketch_set_names = ["sketch_v1", "sketch_v2", "sketch_v3"]
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

        #TODO: changing to store only mask, color, and location
        if os.path.exists(save_path):
            try:
                save_data = pickle.load(open(save_path, "rb"))
                mask, components = save_data["mask"], save_data["components"]
                RE_EXTRACT = False
            except Exception as e:
                print(f"ERROR in load pickle for {save_path}")
                RE_EXTRACT = True                
        else:
            RE_EXTRACT=True

        if RE_EXTRACT: #not os.path.exists(save_path):
            mask, components = self.component_wrapper.process(color_image, sketch, method)
            get_component_color(components, color_image, method)
            mask, components = resize_mask_and_fix_components(mask, components, self.size)

            save_data = {"mask": mask, "components": components}
            # h, w = color_image.shape[:2]
            # color = [a["color"] for a in components]
            # centroid = [a["centroid"] for a in components]
            # kpts = np.array([np.concatenate([comp['centroid'], [len(comp['coords'])]]) for comp in components])
            # if len(kpts) > 0: 
            #     kpts[:,0] = kpts[:,0] / h
            #     kpts[:,1] = kpts[:,1] / w
            #     kpts[:,2] = kpts[:,2] / h / w

            # save_data = {
            #     "mask": mask,
            #     "color": color,
            #     "kpts": kpts,
            # }
            pickle.dump(save_data, open(save_path, "wb+"))

        h, w = color_image.shape[:2]
        kpts = np.array([np.concatenate([comp['centroid'], [len(comp['coords'])]]) for comp in components])
        if len(kpts) > 0: 
            kpts[:,0] = kpts[:,0] / h
            kpts[:,1] = kpts[:,1] / w
            kpts[:,2] = kpts[:,2] / h / w

        return mask, components, kpts

    def __getitem__(self, index):
        name = None
        for key, length in self.lengths.items():
            if index < length:
                name = key
                break
            index -= length

        length = len(self.paths[name]["color"])
        k = random.choice([0, 1, 1, 1])
        next_index = min(index + k, length - 1)

        # read images
        color_a, path_a = get_image_by_index(self.paths[name]["color"], index)
        color_b, path_b = get_image_by_index(self.paths[name]["color"], next_index)

        sketch_a = get_image_by_index(self.paths[name]["sketch_v2"], index)[0]
        sketch_b = get_image_by_index(self.paths[name]["sketch_v2"], next_index)[0]

        if index == next_index:
            if random.random() < 0.3:
                component_method_a = ComponentWrapper.EXTRACT_SKETCH
                component_method_b = ComponentWrapper.EXTRACT_SKETCH
            else:
                component_method_a = ComponentWrapper.EXTRACT_COLOR
                component_method_b = ComponentWrapper.EXTRACT_COLOR
        else:
            if random.random() < 0.3:
                component_method_a = ComponentWrapper.EXTRACT_SKETCH
            else:
                component_method_a = ComponentWrapper.EXTRACT_COLOR
            if random.random() < 0.3:
                component_method_b = ComponentWrapper.EXTRACT_SKETCH
            else:
                component_method_b = ComponentWrapper.EXTRACT_COLOR
        # extract components
        mask_a, components_a, kpts_a = self.get_component_mask(color_a, sketch_a, path_a, component_method_a)
        mask_b, components_b, kpts_b = self.get_component_mask(color_b, sketch_b, path_b, component_method_b)
        # mask_a, colors_a, kpts_a = self.get_component_mask(color_a, sketch_a, path_a, component_method_a)
        # mask_b, colors_b, kpts_b = self.get_component_mask(color_b, sketch_b, path_b, component_method_b)

        graph_a = build_neighbor_graph(mask_a)[1:, 1:]
        graph_b = build_neighbor_graph(mask_b)[1:, 1:]

        # get image pair signature
        cut_name = os.path.basename(os.path.dirname(os.path.dirname(path_a)))
        part_name = os.path.basename(os.path.dirname(path_a))
        key_name = "%s_%s" % (cut_name, part_name)

        source_cm = "sketch" if component_method_a == ComponentWrapper.EXTRACT_SKETCH else "color"
        target_cm = "sketch" if component_method_b == ComponentWrapper.EXTRACT_SKETCH else "color"
        source_name = os.path.splitext(os.path.basename(path_a))[0]
        target_name = os.path.splitext(os.path.basename(path_b))[0]
        source_name = "%s_%s" % (source_name, source_cm)
        target_name = "%s_%s" % (target_name, target_cm)

        image_pair_name = "%s_%s_%s" % (key_name, source_name, target_name)
        label_dir = os.path.join(self.root_dir, cut_name, "color", "annotations")
        label_json_path = os.path.join(label_dir, "%s_%s.json" % (source_name, target_name))

        # component matching
        if index == next_index:
            positive_pairs = [[i, i] for i in range(0, mask_a.max() - 1)]
            positive_pairs = np.array(positive_pairs)
        else:
            if os.path.exists(label_json_path):
                positive_pairs = matching_utils.get_pairs_from_tool_label(label_json_path)
            else:
                # positive_pairs = np.array([])
                positive_pairs = matching_utils.get_pairs_from_matching(
                    path_a, path_b, label_json_path,
                    components_a, components_b, mask_a, mask_b,
                    color_a, color_b)

        # component color
        colors_a = [a["color"] for a in components_a]
        colors_b = [b["color"] for b in components_b]
        colors_a, colors_b = np.array(colors_a), np.array(colors_b)

        # if len(components_a) == 0 or len(components_b) == 0:
        if mask_a.max() == 0 or mask_b.max() == 0:
            print("Invalid component data:", image_pair_name)
        elif positive_pairs.shape[0] == 0:
            print("No matching pairs found:", image_pair_name)

        # get input features
        sketch_set_name = random.choice(self.all_sketch_set_names)
        sketch_a = get_image_by_index(self.paths[name][sketch_set_name], index)[0]
        sketch_b = get_image_by_index(self.paths[name][sketch_set_name], next_index)[0]
        sketch_a = features_utils.get_sketch_image_raw_pixels(sketch_a, self.size)
        sketch_b = features_utils.get_sketch_image_raw_pixels(sketch_b, self.size)

        # augment sketch with noise
        sketch_a = add_random_noise(sketch_a)
        sketch_b = add_random_noise(sketch_b)
        # convert to stacked mask
        mask_a = convert_to_stacked_mask(mask_a)
        mask_b = convert_to_stacked_mask(mask_b)

        # do self-augmentation
        sketch_a, sketch_b = torch.tensor(sketch_a).float(), torch.tensor(sketch_b).float()
        mask_a, mask_b = torch.tensor(mask_a).float(), torch.tensor(mask_b).float()

        #TODO: adding keypoints info
        kpts_a, kpts_b = torch.tensor(kpts_a).float(), torch.tensor(kpts_b).float(), 

        #TODO: disable augmentation
        # sketch_a, mask_a = do_self_augment(sketch_a, mask_a)
        # sketch_b, mask_b = do_self_augment(sketch_b, mask_b)

        output = (
            sketch_a, mask_a, graph_a, kpts_a,
            sketch_b, mask_b, graph_b, kpts_b,
            positive_pairs, colors_a, colors_b,
        )
        return output
