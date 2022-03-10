import os
import glob
import numpy as np
import cv2
import skimage.measure as measure
import skimage.feature
from PIL import Image
from natsort import natsorted
from rules.foreground_mask_utils import get_color_region


def build_neighbor_graph(mask):
    max_level = mask.max() + 1
    angles = [0, np.pi * 0.25, np.pi * 0.5, np.pi * 0.75, np.pi]

    matrix = skimage.feature.greycomatrix(mask, [1, 2, 3], angles, levels=max_level, symmetric=True)
    matrix = np.sum(matrix, axis=(2, 3))
    graph = np.zeros((max_level, max_level), dtype=np.int64)

    for i in range(1, max_level):
        for j in range(1, max_level):
            if matrix[i, j] > 0:
                graph[i, j] = 1
                graph[j, i] = 1
            if i == j:
                graph[i, j] = 1
    return graph


def get_line_art_value(image, min_area=10, min_size=1):
    line_colors = []

    # Convert rgb image to (h, w, 1)
    b, g, r = cv2.split(image)
    b, g, r = b.astype(np.uint64), g.astype(np.uint64), r.astype(np.uint64)
    processed_image = np.array(b + 300 * (g + 1) + 300 * 300 * (r + 1))
    uniques = np.unique(processed_image)

    for unique in uniques:
        rows, cols = np.where(processed_image == unique)
        image_tmp = np.zeros_like(processed_image)
        image_tmp[rows, cols] = 255

        # Get components
        labels = measure.label(image_tmp, connectivity=1, background=0)

        for region in measure.regionprops(labels, intensity_image=processed_image):
            if region.area < min_area:
                continue
            if abs(region.bbox[2] - region.bbox[0]) < min_size:
                continue
            if abs(region.bbox[3] - region.bbox[1]) < min_size:
                continue

            image_tmp_ = np.zeros_like(processed_image)
            coord = region["coords"]
            image_tmp_[coord[:, 0], coord[:, 1]] = 255

            contours = measure.find_contours(image_tmp_, 0.8)
            if len(contours) != 1:
                continue

            contour = np.array(contours[0], dtype=np.int64)
            color_values, counts = np.unique(processed_image[contour[:, 0], contour[:, 1]], return_counts=True)
            line_colors.extend(color_values)

            if len(line_colors) > 10:
                return max(set(line_colors), key=line_colors.count)

    if len(line_colors) == 0:
        return 90300
    return max(set(line_colors), key=line_colors.count)


def get_color_from_sum(sum_value):
    r = sum_value // (300 * 300) - 1
    g = (sum_value % (300 * 300)) // 300 - 1
    b = sum_value % 300
    return b, g, r


def is_sketch_line(region, fg_box, sum_value):
    size_ratio = 0.8
    area_ratio = [0.08, 0.1]
    black_sum = 80

    box = region.bbox
    h, w = float(box[2] - box[0]), float(box[3] - box[1])
    b, g, r = get_color_from_sum(sum_value)

    fg_w = max(float(fg_box[2] - fg_box[0]), 1.0)
    fg_h = max(float(fg_box[3] - fg_box[1]), 1.0)
    fg_area = fg_w * fg_h
    box_area = h * w

    if (region.area / box_area < area_ratio[0]) and (b + g + r < black_sum):
        return True
    if (h / fg_h > size_ratio) and (w / fg_w > size_ratio) and (region.area / fg_area < area_ratio[1]):
        return True
    return False


class ComponentWrapper:
    EXTRACT_COLOR = "extract_color"
    EXTRACT_SKETCH = "extract_sketch"

    def __init__(self, min_area=30, min_size=5):
        self.min_area = min_area
        self.min_size = min_size
        self.bad_values = self.get_default_line_art_values()

    def get_default_line_art_values(self):
        line_colors = [[x, x, x] for x in [0, 1, 5, 10, 15, 255]]
        line_colors += [[18, 21, 25], [19, 14, 20]]

        bad_values = [c[0] + 300 * (c[1] + 1) + 300 * 300 * (c[2] + 1) for c in line_colors]
        return bad_values

    def extract_on_color_image(self, input_image):
        b, g, r = cv2.split(input_image)
        b, g, r = b.astype(np.uint64), g.astype(np.uint64), r.astype(np.uint64)

        index = 0
        components = {}
        mask = np.zeros((input_image.shape[0], input_image.shape[1]), dtype=np.int64)

        # Get bounding box of foreground region
        _, fg_box = get_color_region(input_image)

        # Pre-processing image
        processed_image = b + 300 * (g + 1) + 300 * 300 * (r + 1)
        # Get number of colors in image
        uniques = np.unique(processed_image)

        for unique in uniques:
            # Ignore sketch (ID of background is 255)
            if (uniques.shape[0] == 2) and (23117055 in uniques) and (unique != 23117055):
                pass
            elif unique in self.bad_values:
                continue

            rows, cols = np.where(processed_image == unique)
            # Mask
            image_temp = np.zeros_like(processed_image)
            image_temp[rows, cols] = 255
            image_temp = np.array(image_temp, dtype=np.uint8)

            # Connected components
            labels = measure.label(image_temp, connectivity=1, background=0)
            regions = measure.regionprops(labels, intensity_image=processed_image)

            for region in regions:
                if region.area < self.min_area:
                    continue
                if abs(region.bbox[2] - region.bbox[0]) < self.min_size:
                    continue
                if abs(region.bbox[3] - region.bbox[1]) < self.min_size:
                    continue
                if is_sketch_line(region, fg_box, unique):
                    continue

                if unique == 23117055 and [0, 0] in region.coords:
                    continue

                components[index] = {
                    "centroid": np.array(region.centroid),
                    "area": region.area,
                    "image": region.image.astype(np.uint8) * 255,
                    "label": index + 1,
                    "coords": region.coords,
                    "bbox": region.bbox,
                    "min_intensity": region.min_intensity,
                    "mean_intensity": region.mean_intensity,
                    "max_intensity": region.max_intensity,
                }
                mask[region.coords[:, 0], region.coords[:, 1]] = index + 1
                index += 1

        components = [components[i] for i in range(0, len(components))]
        return mask, components

    def extract_on_sketch_v3(self, sketch):
        if sketch.ndim > 2:
            sketch = cv2.cvtColor(sketch, cv2.COLOR_BGR2GRAY)

        binary = cv2.threshold(sketch, 100, 255, cv2.THRESH_BINARY)[1]
        labels = measure.label(binary, connectivity=1, background=0)
        regions = measure.regionprops(labels, intensity_image=sketch)

        index = 0
        mask = np.zeros((sketch.shape[0], sketch.shape[1]), dtype=np.int64)
        components = dict()

        for region in regions[1:]:
            if region.area < self.min_area:
                continue
            if abs(region.bbox[2] - region.bbox[0]) < self.min_size:
                continue
            if abs(region.bbox[3] - region.bbox[1]) < self.min_size:
                continue

            components[index] = {
                "centroid": np.array(region.centroid),
                "area": region.area,
                "image": region.image.astype(np.uint8) * 255,
                "label": index + 1,
                "coords": region.coords,
                "bbox": region.bbox,
                "min_intensity": region.min_intensity,
                "mean_intensity": region.mean_intensity,
                "max_intensity": region.max_intensity,
            }
            mask[region.coords[:, 0], region.coords[:, 1]] = index + 1
            index += 1

        components = [components[i] for i in range(0, len(components))]
        return mask, components

    def process(self, input_image, sketch, method):
        assert len(cv2.split(input_image)) == 3, "Input image must be RGB, got binary"
        assert method in [self.EXTRACT_COLOR, self.EXTRACT_SKETCH]

        if method == self.EXTRACT_COLOR:
            mask, components = self.extract_on_color_image(input_image)
        else:
            mask, components = self.extract_on_sketch_v3(sketch)
        return mask, components


def get_component_color(components, color_image, mode=ComponentWrapper.EXTRACT_SKETCH):
    list_colors = set()

    if mode == ComponentWrapper.EXTRACT_COLOR:
        for component in components:
            index = len(component["coords"]) // 2
            coord = component["coords"][index]
            color = color_image[coord[0], coord[1]].tolist()
            component["color"] = color
            list_colors.add(tuple(color))
    elif mode == ComponentWrapper.EXTRACT_SKETCH:
        for component in components:
            coords = component["coords"]
            points = color_image[coords[:, 0], coords[:, 1]]

            unique, counts = np.unique(points, return_counts=True, axis=0)
            max_index = np.argmax(counts)
            color = unique[max_index].tolist()
            component["color"] = color
            list_colors.add(tuple(color))
    return list_colors


def rectify_mask(mask, component, ratio):
    coords = component["coords"]
    new_coords = np.array([[int(coord[0] * ratio[0]), int(coord[1] * ratio[1])] for coord in coords])
    new_coords = list(np.unique(new_coords, axis=0).tolist())

    count = 0
    mid_index = int(len(new_coords) / 2)
    new_area = {component["label"]: len(new_coords)}

    for i in range(0, mid_index + 1):
        offsets = [1] if i == 0 else [-1, 1]
        for j in offsets:
            index = mid_index + i * j
            if index >= len(new_coords):
                continue
            coord = new_coords[index]

            if mask[coord[0], coord[1]] == 0:
                mask[coord[0], coord[1]] = component["label"]
                count += 1
                continue

            label = mask[coord[0], coord[1]]
            if label not in new_area:
                new_area[label] = np.count_nonzero(mask == label)

            if new_area[label] > new_area[component["label"]] * 5:
                mask[coord[0], coord[1]] = component["label"]
                count += 1
            elif new_area[label] > 1 and count == 0:
                mask[coord[0], coord[1]] = component["label"]
                count += 1
    return mask


def resize_mask_and_fix_components(mask, components, size):
    new_mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    ratio = (size[1] / mask.shape[0], size[0] / mask.shape[1])

    # try to fix missed labels during resize
    old_labels = list(np.unique(mask).tolist())
    after_remove_labels = np.unique(new_mask).tolist()
    removed_labels = [i for i in old_labels if (i not in after_remove_labels) and (i > 0)]

    for i in removed_labels:
        component = components[i - 1]
        new_mask = rectify_mask(new_mask, component, ratio)

    # check correctness, if good then return
    is_correct = len(np.unique(mask)) == len(np.unique(new_mask))
    if is_correct:
        new_mask = new_mask.astype(np.int64)
        return new_mask, components

    # some labels cannot be fixed, remove corresponding components
    after_remove_labels = list(np.unique(new_mask).tolist())
    new_components = []
    continuous_mask = np.zeros_like(new_mask)
    num_removed = 0

    for component in components:
        old_label = component["label"]

        if old_label in after_remove_labels:
            component["label"] = old_label - num_removed
            continuous_mask[new_mask == old_label] = old_label - num_removed
            new_components.append(component)
        else:
            num_removed += 1

    # output
    continuous_mask = continuous_mask.astype(np.int64)
    return continuous_mask, new_components


def main():
    root_dir = "/home/tyler/work/data/GeekToys/coloring_data/complete_data"
    output_dir = "/home/tyler/work/data/GeekToys/output/output"

    character_dirs = natsorted(glob.glob(os.path.join(root_dir, "*")))
    component_wrapper = ComponentWrapper()

    for character_dir in character_dirs:
        character_name = os.path.basename(character_dir)
        paths = natsorted(glob.glob(os.path.join(character_dir, "color", "*.tga")))

        for path in paths:
            name = os.path.splitext(os.path.basename(path))[0]
            full_name = "%s_%s" % (character_name, name)
            print(full_name)

            color_image = cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)
            output_mask, output_components = component_wrapper.process(
                color_image, None, ComponentWrapper.EXTRACT_COLOR)
            get_component_color(output_components, color_image)

            k = int(255 / len(output_components))
            new_mask, new_components = resize_mask_and_fix_components(output_mask, output_components, (768, 512))
            print(len(output_components), len(new_components))

            output_mask = np.where(output_mask == 0, np.zeros_like(output_mask), np.full_like(output_mask, 255))
            output_mask = np.stack([output_mask] * 3, axis=-1)
            output_mask = output_mask * k
            debug_image = np.concatenate([color_image, output_mask], axis=1)

            cv2.imwrite(os.path.join(output_dir, "%s.png" % full_name), debug_image)
    return


if __name__ == "__main__":
    main()
