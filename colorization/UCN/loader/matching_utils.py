import os
import json
import numpy as np
from rules.old_shape_matching_wrapper import ShapeMatchingWrapper as BasicShapeMatchingWrapper
from rules.shape_matching_wrapper import process_two_component_sets


GT_ONLINE = "gt_online"
GT_FROM_JSON = "gt_from_json"


def get_pairs_three_stage(components_a, components_b, is_removed=False):
    matcher = BasicShapeMatchingWrapper()
    pairs = []

    for index_a, a in enumerate(components_a):
        matches = [(b, matcher.process(a, b)) for b in components_b]
        count_true = len([1 for match in matches if match[1][0]])
        if count_true == 0:
            continue

        distances = np.array([match[1][1] for match in matches])
        index_b = int(np.argmin(distances))
        pairs.append([index_a, index_b])

    if len(pairs) == 0:
        for index_a, a in enumerate(components_a):
            matches = [(b, matcher.process(a, b, area_filter=False, threshold=0.2))
                       for b in components_b]
            count_true = len([1 for match in matches if match[1][0]])
            if count_true == 0:
                continue

            distances = np.array([match[1][1] for match in matches])
            index_b = int(np.argmin(distances))
            pairs.append([index_a, index_b])

    if len(pairs) == 0 and (not is_removed):
        for index_a, a in enumerate(components_a):
            matches = [(b, matcher.process(a, b, area_filter=False, pos_filter=False, threshold=0.6))
                       for b in components_b]
            count_true = len([1 for match in matches if match[1][0]])
            if count_true == 0:
                continue

            distances = np.array([match[1][1] for match in matches])
            index_b = int(np.argmin(distances))
            pairs.append([index_a, index_b])

    pairs = np.array(pairs)
    return pairs


def get_pairs_from_matching(
        source_path, target_path, label_path,
        components_a, components_b, mask_a, mask_b,
        color_image_a, color_image_b):

    pairs = process_two_component_sets(
        components_a, components_b, mask_a, mask_b,
        color_image_a, color_image_b)

    label_data = {
        "imagePathRight": target_path,
        "imagePathLeft": source_path,
        "pairs": {},
    }
    for pair in pairs:
        source_label, target_label = pair[0], pair[1]
        key = "%d_%d" % (source_label, target_label)
        label_data["pairs"][key] = {"right": target_label, "left": source_label}

    if not os.path.exists(os.path.dirname(label_path)):
        os.makedirs(os.path.dirname(label_path))
    json.dump(label_data, open(label_path, "w+"), indent=4)

    pairs = [[v["left"] - 1, v["right"] - 1] for k, v in label_data["pairs"].items()]
    pairs = np.array(pairs)
    return pairs


def get_pairs_from_tool_label(label_json_path):
    label_data = json.load(open(label_json_path))

    pairs = label_data["pairs"]
    pairs = [[v["left"] - 1, v["right"] - 1] for k, v in pairs.items()]
    pairs = np.array(pairs)
    return pairs
