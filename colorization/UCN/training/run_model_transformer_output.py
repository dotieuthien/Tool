import sys
import os
import argparse
import glob
import numpy as np
import cv2
import torch
import torch.utils.data
import torch.nn.functional
from natsort import natsorted
from PIL import Image
from scipy.optimize import linear_sum_assignment

from rules.component_wrapper import ComponentWrapper, get_component_color
from rules.component_wrapper import resize_mask_and_fix_components, build_neighbor_graph
from loader.features_utils import convert_to_stacked_mask
from loader import features_utils
from models.ucn_transformer import UCN_Transformer as UNet
from models.utils import cosine_distance


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args(sys.argv[1:])
    return args


def get_base_name(path):
    return os.path.splitext(os.path.basename(path))[0]


def read_image(path):
    if path.endswith(".tga") or path.endswith(".TGA"):
        image = cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)
    else:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
    return image


def get_input_data(path, size, reference=True):
    component_wrapper = ComponentWrapper()
    color_image = read_image(path)

    root_dir = os.path.dirname(os.path.dirname(path))
    sketch_path = os.path.join(root_dir, "sketch", os.path.basename(path))
    if not os.path.isfile(sketch_path):
        sketch_path = os.path.join(root_dir, "sketch_v3", os.path.basename(path).split(".")[0] + ".png")
    sketch_image = read_image(sketch_path)

    # extract components
    mask, components = component_wrapper.process(color_image, sketch_image, ComponentWrapper.EXTRACT_COLOR)
    # mask, components = component_wrapper.process(color_image, sketch_image, ComponentWrapper.EXTRACT_SKETCH)
    graph = build_neighbor_graph(mask)[1:, 1:]
    mask, components = resize_mask_and_fix_components(mask, components, size)
    mask = convert_to_stacked_mask(mask)
    kpts = np.array([np.concatenate([comp['centroid'], [len(comp['coords'])]]) for comp in components])
    h,w = color_image.shape[:2]
    if len(kpts) > 0: 
        kpts[:,0] = kpts[:,0] / h
        kpts[:,1] = kpts[:,1] / w
        kpts[:,2] = kpts[:,2] / h / w

    # get features
    sketch = features_utils.get_sketch_image_raw_pixels(sketch_image, size)
    sketch = torch.tensor(sketch).float().unsqueeze(0)
    mask = torch.tensor(mask).float().unsqueeze(0)
    kpts = torch.tensor(kpts).float().unsqueeze(0)

    if reference:
        get_component_color(components, color_image, ComponentWrapper.EXTRACT_COLOR)
        return sketch, mask, components, color_image, kpts

    black_values = [0, 5, 10, 15]
    frame = np.full_like(color_image, 255)
    for x in black_values:
        frame = np.where(color_image == [x, x, x], np.zeros_like(color_image), frame)
    return sketch, mask, components, color_image, frame, kpts


def draw_components(image, target, source, pairs):
    image = np.full_like(image, 255)

    for index, component in enumerate(target):
        match_index = [p[1] for p in pairs if p[0] == index]
        if len(match_index) == 0:
            continue

        match_index = match_index[0]
        match_part = source[match_index]

        match_color = np.array(match_part["color"])
        coords = component["coords"]
        image[coords[:, 0], coords[:, 1], :] = match_color
    return image


def linear_matching(output_a, output_b):
    pairs = []

    for index_b in range(0, output_b.shape[1]):
        region_b = output_b[:, index_b, :]
        region_b = region_b.unsqueeze(1).repeat([1, output_a.shape[1], 1])
        distances = cosine_distance(region_b, output_a)
        min_index = torch.argmin(distances).item()
        pairs.append([index_b, min_index])
    return pairs


def hungarian_matching(output_a, output_b):
    distance = np.zeros((output_b.shape[1], output_a.shape[1]))

    for index_b in range(0, output_b.shape[1]):
        region_b = output_b[:, index_b, :]
        region_b = region_b.unsqueeze(1).repeat([1, output_a.shape[1], 1])
        distance[index_b, :] = cosine_distance(region_b, output_a)

    pairs = linear_sum_assignment(distance)
    pairs = [[a, b] for a, b in zip(pairs[0], pairs[1])]
    return pairs


def pick_reference_sketch(character_dir):
    component_wrapper = ComponentWrapper()
    image_paths = natsorted(glob.glob(os.path.join(character_dir, "color", "*.tga")))

    candidates = []
    for image_path in image_paths:
        image = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        components = component_wrapper.process(image, None, ComponentWrapper.EXTRACT_COLOR)[1]
        candidates.append((len(components)))

    final_candidate = int(np.argmax(candidates))
    return final_candidate


def main(args):
    device = torch.device("cuda:0")
    image_size = (768, 512)

    # Initialize model
    model = UNet(None, 0.0)
    model.load_state_dict(torch.load(args.checkpoint)["model"])
    model.to(device)
    model.eval()

    character_dirs = natsorted(glob.glob(os.path.join(args.input_dir, "*")))

    for character_dir in character_dirs:
        paths = natsorted(glob.glob(os.path.join(character_dir, "color", "*.tga")))
        if len(paths) == 0:
            paths = natsorted(glob.glob(os.path.join(character_dir, "color", "*.TGA")))

        character_name = os.path.basename(character_dir)
        if not os.path.exists(os.path.join(args.output_dir, character_name)):
            os.makedirs(os.path.join(args.output_dir, character_name))
        print(character_dir)

        for index, path in enumerate(paths):
            reference_index = max(index - 1, 0)
            reference_path = paths[reference_index]

            if path == reference_path:
                continue

            image_name = get_base_name(path)
            if os.path.exists(os.path.join(args.output_dir, character_name, "%s.png" % image_name)):
                continue

            # Get data
            sketch_a, mask_a, components_a, color_a, kpts_a = get_input_data(
                reference_path, image_size, reference=True)
            sketch_b, mask_b, components_b, color_b, frame, kpts_b = get_input_data(
                path, image_size, reference=False)

            if mask_a.shape[3] == 0 or mask_b.shape[3] == 0:
                print(path, reference_path)
                continue

            print(len(components_a), len(components_b))

            # Run the model
            with torch.no_grad():
                sketch_a = sketch_a.float().to(device)
                mask_a = mask_a.to(device)
                kpts_a = kpts_a.to(device)

                sketch_b = sketch_b.float().to(device)
                mask_b = mask_b.to(device)
                kpts_b = kpts_b.to(device)

                output_a = model(sketch_a, mask_a, kpts_a)
                output_b = model(sketch_b, mask_b, kpts_b)
                pairs = linear_matching(output_a, output_b)

            output_image = draw_components(frame, components_b, components_a, pairs)
            output_image[np.where((frame == [0, 0, 0]).all(axis=-1))] = [0, 0, 0]
            cv2.imwrite(os.path.join(args.output_dir, character_name, "%s.png" % image_name), output_image)
    return


if __name__ == "__main__":
    main(parse_arguments())