import os
import glob
import shutil
import json
import numpy as np
import cv2
from natsort import natsorted


def get_all_moment_data():
    from PIL import Image
    from rules.component_wrapper import ComponentWrapper

    data_dir = "/home/tyler/work/data/GeekToys/coloring_data/complete_data"
    cut_dirs = natsorted(glob.glob(os.path.join(data_dir, "*", "color")))

    cw = ComponentWrapper()
    all_data = []

    for i, cut_dir in enumerate(cut_dirs):
        print(cut_dir, i)
        paths = natsorted(glob.glob(os.path.join(cut_dir, "*.tga")))

        for path in paths:
            image = cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)
            mask, components = cw.extract_on_color_image(image)

            for component in components:
                c_image = component["image"]
                moments = cv2.moments(c_image)
                moments = cv2.HuMoments(moments)[:, 0]
                moments = np.append(moments, component["area"])
                all_data.append(moments)

    all_data = np.array(all_data)
    print(all_data.shape)
    np.save("/home/tyler/work/data/GeekToys/coloring_data/all_data.npy", all_data)


def normalize_with_two_segments(x, min_v, mean_v, max_v):
    y1 = ((x - min_v) / (mean_v - min_v)) - 1
    y2 = ((x - mean_v) / (max_v - mean_v))
    y = np.where(x <= mean_v, y1, y2)
    return y


def compute_statistics():
    path = "/home/tyler/work/data/GeekToys/coloring_data/all_data.npy"
    data = np.load(path)
    print(data.shape)

    norm_values = [
        [0.0, 0.002, 0.057],
        [0.0, 6e-6, 0.0032],
        [0.0, 8e-9, 4.75e-5],
        [0.0, 5e-9, 2.53e-5],
        [-3e-11, 2e-14, 8.6e-10],
        [-7e-8, 6e-11, 8.5e-7],
        [-2e-10, -3.3e-15, 6.65e-13],
        [0.0, 6000.0, 1400000.0],
    ]

    for index in range(0, 8):
        row = data[:, index]
        norm_v = norm_values[index]
        row = normalize_with_two_segments(row, norm_v[0], norm_v[1], norm_v[2])
        print(np.min(row), np.quantile(row, 0.25))
        print(np.max(row), np.quantile(row, 0.75))
        print(row.mean(), row.std())
        print()
    return


def check_data():
    root_dir = "/home/tyler/work/data/GeekInt/data_dc/hor02"
    paths = natsorted(glob.glob(os.path.join(root_dir, "*", "*", "*.tga")))
    print(len(paths))

    cut_paths = natsorted(glob.glob(os.path.join(root_dir, "*")))
    for cut_path in cut_paths:
        part_paths = natsorted(glob.glob(os.path.join(cut_path, "*")))
        if len(part_paths) > 1:
            print(cut_path)

    root_dir = "/home/tyler/work/data/GeekToys/coloring_data/complete_data"
    paths = natsorted(glob.glob(os.path.join(root_dir, "*", "*", "*.tga")))
    print(len(paths))

    root_dir = "/home/tyler/work/data/GeekInt/data_dc/hard_hor02"
    paths = natsorted(glob.glob(os.path.join(root_dir, "*", "*", "*.tga")))
    print(len(paths))


def merge_two_data_sources():
    root_dir = "/home/tyler/work/data/GeekInt/data_dc/hor02"
    add_dir = "/home/tyler/work/data/GeekToys/coloring_data/complete_data"
    output_dir = "/home/tyler/work/data/GeekInt/data_dc/hard_hor02"

    root_cut_paths = natsorted(glob.glob(os.path.join(root_dir, "*")))
    root_cut_names = [os.path.basename(p) for p in root_cut_paths]
    add_cut_paths = natsorted(glob.glob(os.path.join(add_dir, "*")))

    for root_cut_path in root_cut_paths:
        cut_name = os.path.basename(root_cut_path)
        root_paths = natsorted(glob.glob(os.path.join(root_cut_path, "*", "*.tga")))
        root_image_names = [cut_name + "_" + os.path.splitext(os.path.basename(p))[0] for p in root_paths]

        add_cut_path = os.path.join(add_dir, cut_name)
        if not os.path.exists(add_cut_path):
            continue

        add_paths = natsorted(glob.glob(os.path.join(add_cut_path, "color", "*.tga")))
        add_image_names = [cut_name + "_" + os.path.splitext(os.path.basename(p))[0] for p in add_paths]

        # it seems that
        # images that can be added all have sudden changes in occlusion
        # it isn't wise to add them
        can_add_names = [p for p in add_image_names if p not in root_image_names]

    for add_cut_path in add_cut_paths:
        cut_name = os.path.basename(add_cut_path)
        if cut_name in root_cut_names:
            continue

        os.makedirs(os.path.join(output_dir, cut_name, "1"))
        add_paths = natsorted(glob.glob(os.path.join(add_cut_path, "color", "*.tga")))
        add_paths = add_paths[:2]

        for add_path in add_paths:
            shutil.copy(
                add_path,
                os.path.join(output_dir, cut_name, "1", os.path.basename(add_path)))
    return


def check_match_data():
    match_data_path = "/home/tyler/work/data/GeekToys/coloring_data/complete_data/matches.json"
    match_data = json.load(open(match_data_path))
    print(match_data["hor01_018_021_k_B_color_B0004_B0005"])


def check_data_length():
    root_dir = "/home/tyler/work/data/GeekToys/coloring_data/complete_data"
    cut_paths = natsorted(glob.glob(os.path.join(root_dir, "*", "*")))
    count = 0

    for cut_path in cut_paths:
        cut_name = os.path.basename(os.path.dirname(cut_path))
        part_name = os.path.basename(cut_path)

        if part_name not in ["color"]:
            assert "color" not in part_name
            continue

        paths = natsorted(glob.glob(os.path.join(cut_path, "*.tga")))
        count += len(paths) - 1
    print(count)

    # new iteration
    dirs = natsorted(glob.glob(os.path.join(root_dir, "*")))
    count = 0

    for sub_dir in dirs:
        if not os.path.isdir(sub_dir):
            continue
        paths = natsorted(glob.glob(os.path.join(sub_dir, "color", "*.tga")))
        count += len(paths) - 1
    print(count)


if __name__ == "__main__":
    compute_statistics()
