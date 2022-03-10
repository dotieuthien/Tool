import os
import glob
import numpy as np
import cv2
import skimage.measure
import skimage.morphology
import skimage.segmentation
from natsort import natsorted
from PIL import Image
from matplotlib import pyplot as plt
from rules.component_wrapper import ComponentWrapper, build_neighbor_graph


def read_image(path):
    if path.endswith(".tga"):
        image = cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)
    else:
        image = cv2.imread(path)
    return image


def normalize_color(image, b=64):
    image = (image // b) * b
    return image


def to_sketch(image, kernel_size=1):
    image = np.asarray(image)
    image = normalize_color(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image_lb = skimage.measure.label(image)
    result = skimage.segmentation.find_boundaries(image_lb, connectivity=8)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    result = skimage.morphology.binary_dilation(result, kernel)
    result = (1.0 - result) * 255
    return result


def get_component_coord(component):
    index = len(component["coords"]) // 2
    coord = component["coords"][index]
    return int(coord[1]), int(coord[0])


def show_graph():
    root_dir = "D:/Data/GeekToys/coloring_data/pd_to_sketch"
    cut_dirs = natsorted(glob.glob(os.path.join(root_dir, "*")))
    component_wrapper = ComponentWrapper()

    for cut_dir in cut_dirs:
        paths = glob.glob(os.path.join(cut_dir, "color", "*.tga"))

        if not os.path.exists(os.path.join(cut_dir, "graph")):
            os.makedirs(os.path.join(cut_dir, "graph"))

        for path in paths:
            name = os.path.basename(path).replace(".tga", ".png")
            image = read_image(path)
            mask, components = component_wrapper.process(image, None, ComponentWrapper.EXTRACT_COLOR)
            graph = build_neighbor_graph(mask)

            for component in components:
                k = component["label"]
                neighbors = [components[i - 1] for i in np.nonzero(graph[k])[0]]
                coord = get_component_coord(component)
                color = (0, 0, 255)
                print(len(neighbors) / len(components))

                for neighbor in neighbors[:10]:
                    neighbor_coord = get_component_coord(neighbor)
                    cv2.line(image, coord, neighbor_coord, color, 2)
                    cv2.circle(image, coord, 1, color, 2)
            cv2.imwrite(os.path.join(cut_dir, "graph", name), image)
    return


def count_folders():
    root_dir = "/home/tyler/work/data/ToeiAni/cin_labeled_data/toei_easy_training_set"
    set_name = "sketch_v3"

    cut_dirs = natsorted(glob.glob(os.path.join(root_dir, "*")))
    cut_dirs = [path for path in cut_dirs if os.path.isdir(path)]

    sketched_paths = [path for path in cut_dirs if os.path.exists(os.path.join(path, set_name))]
    print(len(sketched_paths), len(cut_dirs))


def main():
    root_dir = "/home/tyler/work/data/ToeiAni/cin_labeled_data/toei_easy_training_set"
    set_name = "sketch_v3"

    cut_dirs = natsorted(glob.glob(os.path.join(root_dir, "*")))
    cut_dirs = [path for path in cut_dirs if os.path.isdir(path)]

    for cut_dir in cut_dirs:
        print(cut_dir)
        paths = glob.glob(os.path.join(cut_dir, "color", "*.tga"))

        if not os.path.exists(os.path.join(cut_dir, set_name)):
            os.makedirs(os.path.join(cut_dir, set_name))

        for path in paths:
            name = os.path.basename(path).replace(".tga", ".png")
            image = read_image(path)
            sketch = to_sketch(image, kernel_size=4)
            plt.imsave(os.path.join(cut_dir, set_name, name), sketch, cmap=plt.cm.gray)
    return


if __name__ == "__main__":
    main()
