import os
import glob
import math
import json
import numpy as np
import cv2
from PIL import Image
from natsort import natsorted
from rules.foreground_mask_utils import get_color_region, is_small_fgb
from rules.component_wrapper import ComponentWrapper, get_component_color, resize_mask_and_fix_components
from rules.component_wrapper import build_neighbor_graph


def get_area(binary_image):
    return cv2.countNonZero(binary_image)


def get_abs_ratio(x):
    return 1.0 / x if x < 1.0 else x


def calculate_iou(comp1, comp2, img_shape, mode=1):
    h, w = img_shape
    coords1 = comp1["coords"]
    coords2 = comp2["coords"]

    coords1_flatten = coords1[:, 0] * w + coords1[:, 1]
    coords2_flatten = coords2[:, 0] * w + coords2[:, 1]
    intersect_points = np.intersect1d(coords1_flatten, coords2_flatten)

    if mode == 1:
        iou_ratio = len(intersect_points) / np.sqrt(coords2.shape[0] * coords1.shape[0])
    elif mode == 2:
        iou_ratio = len(intersect_points) / coords2_flatten.shape[0]
    else:
        iou_ratio = len(intersect_points)
    return iou_ratio


def get_component_neighbors(components, mask):
    graph = build_neighbor_graph(mask)

    for component in components:
        component_neighbor_vector = graph[component["label"]]
        neighbor_labels = np.nonzero(component_neighbor_vector)[0].tolist()
        neighbor_labels.remove(component["label"])
        component["neighbors"] = neighbor_labels
    return


def compute_anchor_of_foreground_box(
        fgb_a, fgb_b,
        ratio_threshold=5.0,
        close_distance=50, far_distance=100):

    anchor_a = [fgb_a[1], fgb_a[0]]
    anchor_b = [fgb_b[1], fgb_b[0]]

    left_distance, right_distance = abs(fgb_a[0] - fgb_b[0]), abs(fgb_a[2] - fgb_b[2])
    top_distance, bottom_distance = abs(fgb_a[1] - fgb_b[1]), abs(fgb_a[3] - fgb_b[3])

    if right_distance < close_distance and left_distance > far_distance:
        if left_distance / max(right_distance, 1) > ratio_threshold:
            anchor_a[1] = fgb_a[2]
            anchor_b[1] = fgb_b[2]

    if bottom_distance < close_distance and top_distance > far_distance:
        if top_distance / max(bottom_distance, 1) > ratio_threshold:
            anchor_a[0] = fgb_a[3]
            anchor_b[0] = fgb_b[3]

    return anchor_a, anchor_b


def remove_pairs_of_wrong_ordering(pairs, area_threshold=500):
    if len(pairs) != 2:
        return pairs

    a1, b1 = pairs[0][0], pairs[0][1]
    a2, b2 = pairs[1][0], pairs[1][1]
    all_components = [a1, b1, a2, b2]

    if not all([c["area"] < area_threshold for c in all_components]):
        return pairs

    if a1["bbox"][0] > b1["bbox"][2] and a2["bbox"][2] < b2["bbox"][0]:
        return []
    if a1["bbox"][0] < b1["bbox"][2] and a2["bbox"][2] > b2["bbox"][0]:
        return []
    if a1["bbox"][1] > b1["bbox"][3] and a2["bbox"][3] < b2["bbox"][1]:
        return []
    if a1["bbox"][1] < b1["bbox"][3] and a2["bbox"][3] > b2["bbox"][1]:
        return []
    return pairs


def filter_matches_by_neighbors(pairs, components_a, components_b):
    filtered_pairs = []
    count_threshold = 2
    shape_threshold = 0.01

    min_area = 30
    area_threshold_for_distance_checking = 100
    distance_threshold = [100, 1.25]
    far_shape_threshold = 0.1

    for pair in pairs:
        component_a = [c for c in components_a if c["label"] == pair[0]][0]
        neighbors_a = component_a["neighbors"]
        component_b = [c for c in components_b if c["label"] == pair[1]][0]
        neighbors_b = component_b["neighbors"]

        common_nb = []

        for neighbor_a in neighbors_a:
            nb_pair = [p for p in pairs if p[0] == neighbor_a]
            if len(nb_pair) == 0:
                continue

            neighbor_b = nb_pair[0][1]
            if neighbor_b in neighbors_b:
                common_nb.append([neighbor_a, neighbor_b])

        count_neighbor_matches = len(common_nb)
        half_len = int(math.ceil(min(len(neighbors_a), len(neighbors_b)) / 2))

        if component_a["area"] < min_area or component_b["area"] < min_area:
            continue

        if all([c["area"] < area_threshold_for_distance_checking for c in [component_a, component_b]]):
            distance = np.linalg.norm(component_a["centroid"] - component_b["centroid"])
            max_distance = max(component_a["area"], component_b["area"]) * distance_threshold[1]
            max_distance = max(max_distance, distance_threshold[0])

            if (distance > max_distance) and (pair[2] > far_shape_threshold):
                continue

        if len(neighbors_a) == len(neighbors_b) == 0:
            filtered_pairs.append([pair[0], pair[1], 0])
        elif count_neighbor_matches >= max(min(half_len, count_threshold), 1):
            filtered_pairs.append([pair[0], pair[1], 0])
        elif pair[2] < shape_threshold:
            filtered_pairs.append([pair[0], pair[1], 1])

    return filtered_pairs


def group_components_in_list(components, criterion):
    groups = [[components[0]]]

    for comp in components[1:]:
        if criterion(groups[-1][-1], comp):
            groups[-1].append(comp)
        else:
            groups.append([comp])
    return groups


def compute_shape_distance(x, y):
    x_image, y_image = x["image"], y["image"]
    d2 = cv2.matchShapes(x_image, y_image, cv2.CONTOURS_MATCH_I2, 1e-20)
    return d2


class ShapeMatchingWrapper:
    def __init__(self):
        self.area_thr = [1.4, 2.0]
        self.pos_thr = [40, 120]
        self.shape_thr = [0.2, 0.4]
        self.shape_max_diff = 0.04

    def match_by_area(self, c_a, c_b):
        def similar_area(x, y):
            return get_abs_ratio(y["area"] / x["area"]) < self.area_thr[0]

        pairs = dict()
        # sort by area
        c_a = sorted(c_a, key=lambda x: x["area"])
        c_b = sorted(c_b, key=lambda x: x["area"])

        # grouping by area
        g_a = group_components_in_list(c_a, similar_area)
        g_b = group_components_in_list(c_b, similar_area)

        # match between two groups
        for ia, ge_a in enumerate(g_a):
            min_area_ratio = self.area_thr[1] + 1
            min_index = 0

            for ib, ge_b in enumerate(g_b):
                if len(ge_a) == 0 or len(ge_b) == 0:
                    continue

                ff_ratio = get_abs_ratio(ge_a[0]["area"] / ge_b[0]["area"])
                ll_ratio = get_abs_ratio(ge_a[-1]["area"] / ge_b[-1]["area"])
                ff_is_close = ff_ratio < self.area_thr[1]
                ll_is_close = ll_ratio < self.area_thr[1]

                if ff_is_close and ll_is_close:
                    area_ratio = min(ff_ratio, ll_ratio)

                    if area_ratio < min_area_ratio:
                        min_area_ratio = area_ratio
                        min_index = ib

            if min_area_ratio < self.area_thr[1]:
                if min_index in pairs:
                    pairs[min_index].append([ia, min_area_ratio])
                else:
                    pairs[min_index] = [[ia, min_area_ratio]]

        processed_pairs = []
        for ib, list_ia in pairs.items():
            list_ia = np.array(list_ia)
            min_index = np.argmin(list_ia[:, 1])
            ia = int(list_ia[min_index, 0])
            processed_pairs.append([g_a[ia], g_b[ib]])
        return processed_pairs

    def match_by_position(self, c_a, fgb_anchor_a, c_b, fgb_anchor_b, small_foreground):
        pos_thr = self.pos_thr[0] if small_foreground else self.pos_thr[1]

        def distance_with_offset(x, y):
            cen_x, cen_y = x["centroid"], y["centroid"]
            cen_x = cen_x - np.array(fgb_anchor_a)
            cen_y = cen_y - np.array(fgb_anchor_b)
            return np.linalg.norm(cen_y - cen_x)

        def similar_position(x, y):
            cen_x, cen_y = x["centroid"], y["centroid"]
            diff_pos = np.linalg.norm(cen_y - cen_x)
            return diff_pos < pos_thr

        pairs = dict()
        # sort by position
        c_a = sorted(c_a, key=lambda x: tuple(x["centroid"]))
        c_b = sorted(c_b, key=lambda x: tuple(x["centroid"]))

        # grouping by position
        g_a = group_components_in_list(c_a, similar_position)
        g_b = group_components_in_list(c_b, similar_position)

        # match between two groups
        for ia, ge_a in enumerate(g_a):
            min_diff = pos_thr + 1
            min_index = 0

            for ib, ge_b in enumerate(g_b):
                if len(ge_a) == 0 or len(ge_b) == 0:
                    continue

                ff_diff = distance_with_offset(ge_a[0], ge_b[0])
                ll_diff = distance_with_offset(ge_a[0], ge_b[0])
                ff_is_close = ff_diff < pos_thr
                ll_is_close = ll_diff < pos_thr

                if ff_is_close and ll_is_close:
                    diff = min(ff_diff, ll_diff)

                    if diff < min_diff:
                        min_diff = diff
                        min_index = ib

            if min_diff < pos_thr:
                if min_index in pairs:
                    pairs[min_index].append([ia, min_diff])
                else:
                    pairs[min_index] = [[ia, min_diff]]

        processed_pairs = []
        for ib, list_ia in pairs.items():
            list_ia = np.array(list_ia)
            min_index = np.argmin(list_ia[:, 1])
            ia = int(list_ia[min_index, 0])
            processed_pairs.append([g_a[ia], g_b[ib]])
        return processed_pairs

    def match_by_shape(self, c_a, c_b):
        pairs = []

        # when there's only one component, it's clear, we just match them
        if len(c_a) == len(c_b) == 1:
            sd = compute_shape_distance(c_a[0], c_b[0])
            if sd < self.shape_thr[1]:
                pairs.append([c_a[0], c_b[0], sd])
            return pairs

        # build a table of shape distances
        match_table = np.zeros([len(c_a), len(c_b)])
        for ia, a in enumerate(c_a):
            for ib, b in enumerate(c_b):
                match_table[ia, ib] = compute_shape_distance(a, b)

        # find a table of min_index, so we can use them later
        pairs_raw = dict()
        for ia, a in enumerate(c_a):
            distances = match_table[ia, :]

            min_index_b = np.argmin(distances)
            min_distance = distances[min_index_b]

            matches = distances < min(self.shape_thr[0], min_distance + self.shape_max_diff)
            count = np.count_nonzero(matches)

            if (min_distance < self.shape_thr[0]) and (count == 1):
                if min_index_b in pairs_raw:
                    pairs_raw[min_index_b].append([ia, min_distance])
                else:
                    pairs_raw[min_index_b] = [[ia, min_distance]]

        # if one target has multiple correspondences, remove it
        for ib, list_ia in pairs_raw.items():
            if len(list_ia) == 1:
                pairs.append([c_a[list_ia[0][0]], c_b[ib], list_ia[0][1]])

        pairs = remove_pairs_of_wrong_ordering(pairs)
        return pairs

    def process(self, c_a, fgb_a, c_b, fgb_b):
        pairs = []

        small_foreground = is_small_fgb(fgb_a) and is_small_fgb(fgb_b)
        fgb_anchor_a, fgb_anchor_b = compute_anchor_of_foreground_box(fgb_a, fgb_b)
        # match components by area only, multiple candidates allowed
        pairs_by_area = self.match_by_area(c_a, c_b)

        # after matching by components, we match by position
        for c_a_area, c_b_area in pairs_by_area:
            pairs_by_pos = self.match_by_position(
                c_a_area, fgb_anchor_a, c_b_area, fgb_anchor_b, small_foreground)

            for c_a_pos, c_b_pos in pairs_by_pos:
                pairs_by_shape = self.match_by_shape(c_a_pos, c_b_pos)

                for pair in pairs_by_shape:
                    pairs.append([pair[0]["label"], pair[1]["label"], pair[2]])
        return pairs


def global_match_components(components_a, list_colors_a, fgb_a,
                            components_b, list_colors_b, fgb_b):
    matcher = ShapeMatchingWrapper()
    pairs = []
    match_colors = [color for color in list_colors_a if color in list_colors_b]

    for match_color in match_colors:
        match_color = np.array(match_color)
        c_a = [c for c in components_a if np.all(np.array(c["color"] == match_color))]
        c_b = [c for c in components_b if np.all(np.array(c["color"] == match_color))]

        if (len(c_a) == len(c_b) == 1) and (max(len(components_a), len(components_b)) < 10):
            pairs.append([c_a[0]["label"], c_b[0]["label"], 0.0])
        else:
            pairs.extend(matcher.process(c_a, fgb_a, c_b, fgb_b))
    return pairs


def manually_draw_a_pair(components_a, components_b, pairs, raw_debug_image, output_dir):
    sw = raw_debug_image.shape[1] // 2
    debug_image = raw_debug_image.copy()

    for pair in pairs:
        if pair[0] == pair[1] == 0:
            continue

        comp_a = [c for c in components_a if c["label"] == pair[0]][0]
        comp_b = [c for c in components_b if c["label"] == pair[1]][0]
        pa, pb = comp_a["centroid"], comp_b["centroid"]
        line_color = (255, 0, 0)

        cv2.line(
            debug_image,
            (int(pa[1]), int(pa[0])),
            (int(pb[1]) + sw, int(pb[0])),
            color=line_color, thickness=3)
        cv2.circle(debug_image, (int(pa[1]), int(pa[0])), radius=6, color=(0, 0, 255), thickness=3)
        cv2.circle(debug_image, (int(pb[1]) + sw, int(pb[0])), radius=6, color=(0, 0, 255), thickness=3)

    cv2.imwrite(os.path.join(output_dir, "%s_%s.png" % (pairs[0][0], pairs[0][1])), debug_image)


def process_two_component_sets(
        components_a, components_b, mask_a, mask_b,
        color_image_a, color_image_b):
    foreground_box_a = get_color_region(color_image_a)[1]
    foreground_box_b = get_color_region(color_image_b)[1]
    list_colors_a = get_component_color(components_a, color_image_a, ComponentWrapper.EXTRACT_COLOR)
    list_colors_b = get_component_color(components_b, color_image_b, ComponentWrapper.EXTRACT_COLOR)

    get_component_neighbors(components_a, mask_a)
    get_component_neighbors(components_b, mask_b)

    pairs_with_offset = global_match_components(
        components_a, list_colors_a, foreground_box_a,
        components_b, list_colors_b, foreground_box_b)
    pairs_with_offset = filter_matches_by_neighbors(pairs_with_offset, components_a, components_b)

    zero_box = [0, 0, 0, 0]
    pairs_zero_box = global_match_components(
        components_a, list_colors_a, zero_box,
        components_b, list_colors_b, zero_box)
    pairs_zero_box = filter_matches_by_neighbors(pairs_zero_box, components_a, components_b)

    pairs = pairs_zero_box if len(pairs_zero_box) > len(pairs_with_offset) else pairs_with_offset
    pairs = [[p[0], p[1]] for p in pairs]
    return pairs


def process_two_images(source_path, target_path, output_dir=None):
    source_image = cv2.cvtColor(np.array(Image.open(source_path).convert("RGB")), cv2.COLOR_RGB2BGR)
    target_image = cv2.cvtColor(np.array(Image.open(target_path).convert("RGB")), cv2.COLOR_RGB2BGR)
    cw = ComponentWrapper()
    size = (768, 512)

    foreground_box_a = get_color_region(source_image)[1]
    foreground_box_b = get_color_region(target_image)[1]
    mask_a, components_a = cw.process(source_image, None, ComponentWrapper.EXTRACT_COLOR)
    _, components_a = resize_mask_and_fix_components(mask_a, components_a, size)
    list_colors_a = get_component_color(components_a, source_image, ComponentWrapper.EXTRACT_COLOR)
    mask_b, components_b = cw.process(target_image, None, ComponentWrapper.EXTRACT_COLOR)
    _, components_b = resize_mask_and_fix_components(mask_b, components_b, size)
    list_colors_b = get_component_color(components_b, target_image, ComponentWrapper.EXTRACT_COLOR)

    if output_dir is not None:
        fgb_a, fgb_b = foreground_box_a, foreground_box_b
        output_mask_a = np.ascontiguousarray(np.stack([mask_a] * 3, axis=-1)).astype(np.uint8)
        cv2.rectangle(output_mask_a, (fgb_a[0], fgb_a[1]), (fgb_a[2], fgb_a[3]), (255, 0, 0), 2)
        output_mask_b = np.ascontiguousarray(np.stack([mask_b] * 3, axis=-1)).astype(np.uint8)
        cv2.rectangle(output_mask_b, (fgb_b[0], fgb_b[1]), (fgb_b[2], fgb_b[3]), (255, 0, 0), 2)
        cv2.imwrite(os.path.join(output_dir, "mask_a.png"), output_mask_a)
        cv2.imwrite(os.path.join(output_dir, "mask_b.png"), output_mask_b)

    get_component_neighbors(components_a, mask_a)
    get_component_neighbors(components_b, mask_b)

    pairs_with_offset = global_match_components(
        components_a, list_colors_a, foreground_box_a,
        components_b, list_colors_b, foreground_box_b)
    pairs_with_offset = filter_matches_by_neighbors(pairs_with_offset, components_a, components_b)

    zero_box = [0, 0, 0, 0]
    pairs_zero_box = global_match_components(
        components_a, list_colors_a, zero_box,
        components_b, list_colors_b, zero_box)
    pairs_zero_box = filter_matches_by_neighbors(pairs_zero_box, components_a, components_b)

    pairs = pairs_zero_box if len(pairs_zero_box) > len(pairs_with_offset) else pairs_with_offset
    raw_debug_image = np.concatenate([source_image, target_image], axis=1)
    sw = source_image.shape[1]

    if output_dir is None:
        pairs = [[p[0], p[1]] for p in pairs]
        return pairs

    for pair in pairs:
        comp_a = [c for c in components_a if c["label"] == pair[0]][0]
        comp_b = [c for c in components_b if c["label"] == pair[1]][0]
        pa, pb = comp_a["centroid"], comp_b["centroid"]

        debug_image = raw_debug_image.copy()
        line_color = (0, 255, 0) if pair[2] == 0 else (255, 0, 0)

        cv2.line(
            debug_image,
            (int(pa[1]), int(pa[0])),
            (int(pb[1]) + sw, int(pb[0])),
            color=line_color, thickness=3)
        cv2.circle(debug_image, (int(pa[1]), int(pa[0])), radius=6, color=(0, 0, 255), thickness=3)
        cv2.circle(debug_image, (int(pb[1]) + sw, int(pb[0])), radius=6, color=(0, 0, 255), thickness=3)

        comp_mask_a = np.array(mask_a == pair[0], dtype=np.uint8) * 255
        comp_mask_b = np.array(mask_b == pair[1], dtype=np.uint8) * 255
        comp_mask_a = np.stack([comp_mask_a] * 3, axis=-1)
        comp_mask_b = np.stack([comp_mask_b] * 3, axis=-1)
        comp_mask_a = cv2.resize(comp_mask_a, (source_image.shape[1], source_image.shape[0]))
        comp_mask_b = cv2.resize(comp_mask_b, (source_image.shape[1], source_image.shape[0]))

        debug_image = np.concatenate([debug_image, comp_mask_a, comp_mask_b], axis=1)

        cv2.imwrite(os.path.join(output_dir, "%s_%s.png" % (pair[0], pair[1])), debug_image)

    pairs = [[p[0], p[1]] for p in pairs]
    return pairs


def main():
    root_dir = "/home/tyler/work/data/GeekToys/coloring_data/complete_data"
    output_dir = "/home/tyler/work/data/GeekToys/output/output"
    cut_paths = natsorted(glob.glob(os.path.join(root_dir, "*", "color")))
    print(len(cut_paths))

    for cut_path in cut_paths:
        cut_name = os.path.basename(os.path.dirname(cut_path))
        part_name = os.path.basename(cut_path)
        key_name = "%s_%s" % (cut_name, part_name)
        print(key_name)

        paths = natsorted(glob.glob(os.path.join(cut_path, "*.tga")))
        paths += natsorted(glob.glob(os.path.join(cut_path, "*.png")))

        label_dir = os.path.join(cut_path, "annotations")
        os.makedirs(label_dir, exist_ok=True)

        for source_path, target_path in zip(paths[:-1], paths[1:]):
            source_name = os.path.splitext(os.path.basename(source_path))[0]
            target_name = os.path.splitext(os.path.basename(target_path))[0]
            image_pair_name = "%s_%s_%s" % (key_name, source_name, target_name)

            json_path = os.path.join(label_dir, "%s_%s.json" % (source_name, target_name))
            image_pair_output_dir = os.path.join(output_dir, image_pair_name)

            if os.path.exists(json_path):
                continue
            if (image_pair_output_dir is not None) and (not os.path.exists(image_pair_output_dir)):
                os.mkdir(image_pair_output_dir)

            json_data = {
                "imagePathRight": target_path,
                "imagePathLeft": source_path,
                "pairs": {},
            }
            pairs = process_two_images(source_path, target_path, image_pair_output_dir)
            for pair in pairs:
                source_label, target_label = pair[0], pair[1]
                key = "%d_%d" % (source_label, target_label)
                json_data["pairs"][key] = {"right": target_label, "left": source_label}
            # json.dump(json_data, open(json_path, "w+"), indent=4)
    return


if __name__ == "__main__":
    main()
