import copy
import os
import shutil
from math import copysign, log10

import cv2
import time
import numpy as np
import skimage.feature
import torch
import torch.nn.functional
import torch.utils.data

# from extracting_components.extract_components_utils import may_split_component
# from matching_components.match_shapes import MatchingShapes


# def gallery(array, ncols=3):
#     nindex, height, width = array.shape
#     nrows = nindex // ncols
#     assert nindex == nrows * ncols
#     result = (array.reshape(nrows, ncols, height, width)
#               .swapaxes(1, 2)
#               .reshape(height * nrows, width * ncols))
#     return result
#
#
# def make_new_folder(folder_fn):
#     """Create a new folder. If the folder already exists, delete it and create a new one."""
#     if os.path.isdir(folder_fn):
#         shutil.rmtree(folder_fn)
#     os.makedirs(folder_fn)
#
#
# def to_image(comp1, comp2):
#     list_of_coords = [comp1['coords'], comp2['coords']]
#
#     combined_coords = np.concatenate(list_of_coords, axis=0)
#     min_y, max_y = np.min(combined_coords[:, 0]), np.max(combined_coords[:, 0])
#     min_x, max_x = np.min(combined_coords[:, 1]), np.max(combined_coords[:, 1])
#
#     coords1 = comp1['coords'] - np.array([min_y, min_x]).reshape((1, 2))
#     coords2 = comp2['coords'] - np.array([min_y, min_x]).reshape((1, 2))
#
#     new_h, new_w = max_y - min_y + 1, max_x - min_x + 1
#
#     # image 1
#     tmp1 = np.zeros(shape=(new_h, new_w), dtype=np.uint8)
#     tmp1[coords1[:, 0], coords1[:, 1]] = 255
#
#     # image 2
#     tmp2 = np.zeros(shape=(new_h, new_w), dtype=np.uint8)
#     tmp2[coords2[:, 0], coords2[:, 1]] = 255
#
#     return tmp1, tmp2
#
#
# def print_component(component):
#     print(component["label"], component["bbox"], component["centroid"], component["area"])
#
#
# def match_components(
#         ref_sketch_components,
#         sketch_components,
#         ref_sketch,
#         unset_points,
#         full_points,
#         match_comp_cfg,
#         **kwargs):
#     """Match each component extracted from the sketch with the corresponding component from the reference.
#
#     Args:
#         unset_points:
#         full_points:
#         ref_sketch_components (dict): dict of all components in the reference sketch in the form: {index: numpy array}
#         sketch_components (dict): dict of all components in the sketch in the form: {index: numpy array}
#         ref_sketch (numpy array): reference sketch image
#         match_comp_cfg:
#
#     Returns:
#         dict: dictionary in the form {index number of the sketch component: index number of the reference component}
#     """
#
#     logger = kwargs['logger']
#     ucn_pairs = kwargs['ucn_pairs']
#     ucn_distances = kwargs['ucn_distances']
#     debug = kwargs.get('debug_mode', False)
#
#     paired_components = {}
#     candidate_sketch_component = {}
#
#     # Step 0: For each component in sketch_components, find a list of potential corresponding components from the
#     # reference by using rules about the areas and distances. This step can boost the speed.
#     for _id, comp in sketch_components.items():
#         comp_centroid = comp['centroid']
#         candidate_sketch_component[_id] = []
#
#         for _ref_id, ref_comp in ref_sketch_components.items():
#             ref_comp_centroid = ref_comp['centroid']
#
#             # Distance and area ratio bw target sketch and ref component
#             distance = np.linalg.norm(ref_comp_centroid - comp_centroid, ord=2)
#             area_ratio = float(comp['area']) / float(ref_comp['area'])
#
#             offset_distance = 0.0
#             offset_area = 0.0
#             area_threshold = match_comp_cfg['component_area_thres']
#
#             # scale offset
#             if comp['area'] > area_threshold and ref_comp['area'] > area_threshold:
#                 offset_distance += match_comp_cfg['offset_distance_increment']
#                 offset_area += match_comp_cfg['offset_area_increment']
#
#             lower_area_threshold = match_comp_cfg['threshold_area_lower_bound']
#             upper_area_threshold = match_comp_cfg['threshold_area_upper_bound'] + offset_area
#
#             if distance > match_comp_cfg['threshold_distance'] + offset_distance:
#                 continue
#
#             if lower_area_threshold <= area_ratio <= upper_area_threshold:
#                 candidate_sketch_component[_id] += [_ref_id]
#                 continue
#
#             int_ratio = MatchingShapes.calculate_iou(ref_comp, comp, ref_sketch.shape, mode=3)
#             if int_ratio > match_comp_cfg['threshold_intersection_ratio']:
#                 candidate_sketch_component[_id] += [_ref_id]
#
#     # Step 1: Check IOU + UCN distance
#     for comp_id in candidate_sketch_component.keys():
#         _img = sketch_components[comp_id]
#         max_iou, best_iou_ref_id, best_iou_sim = 0.0, -1, 0.0
#         max_sim, best_sim_ref_id, best_sim_iou = 0.0, -1, 0.0
#
#         for ref_comp_id in candidate_sketch_component[comp_id]:
#             ref_img = ref_sketch_components[ref_comp_id]
#
#             s_time = time.time()
#             iou = MatchingShapes.calculate_iou(ref_img, _img, ref_sketch.shape, mode=1)
#             sim = ucn_distances[_img['label'], ref_img['label']]
#             sim = 0.0 if sim == np.inf else 1.0 - sim
#
#             if debug:
#                 logger.debug(f"take time {comp_id} {ref_comp_id} {time.time() - s_time}")
#
#             # if THRESHOLD_IOU_LOWER_BOUND <= iou <= THRESHOLD_IOU_UPPER_BOUND:
#             if match_comp_cfg['threshold_iou_lower_bound'] <= iou <= match_comp_cfg['threshold_iou_upper_bound']:
#                 if iou > max_iou:
#                     max_iou, best_iou_ref_id = iou, ref_comp_id
#                     best_iou_sim = sim # similarity of the best iou
#
#             if sim >= match_comp_cfg['threshold_iou_lower_bound']:
#                 if sim > max_sim:
#                     max_sim, best_sim_ref_id = sim, ref_comp_id
#                     best_sim_iou = iou # iou of the best similarity
#
#         if best_sim_ref_id >= 0 and best_sim_ref_id != best_iou_ref_id: #TODO: check best_sim_ref_id == best_iou_ref_id
#             shape_matching_threshold = max(
#                 match_comp_cfg['threshold_shape_matching_for_iou_bound'],
#                 max_sim - match_comp_cfg['margin_shape_matching_allowed'])
#
#             iou_matching_threshold = max(
#                 match_comp_cfg['threshold_iou_lower_bound'],
#                 max_iou - match_comp_cfg['margin_iou_matching_allowed'])
#
#             if best_iou_sim < shape_matching_threshold:
#                 if best_sim_iou >= iou_matching_threshold:
#                     if max_sim > match_comp_cfg['threshold_shape_matching_maximum']:
#                         max_iou, best_iou_ref_id = best_sim_iou, best_sim_ref_id
#
#         if max_iou >= match_comp_cfg['threshold_iou_lower_bound'] and best_iou_ref_id >= 0: #TODO: check ref_id 0
#             ref_img = ref_sketch_components[best_iou_ref_id]
#             score = ucn_distances[_img['label'], ref_img['label']]
#             score = 0.0 if score == np.inf else 1.0 - score
#
#             if score >= match_comp_cfg['threshold_shape_matching_for_iou_bound']:
#                 paired_components[comp_id] = best_iou_ref_id
#                 if debug:
#                     logger.debug(
#                         "step1: ref-%d vs comp-%d, iou %.4f score %.4f" %
#                         (best_iou_ref_id, comp_id, max_iou, score))
#
#     # Step2: Use SSIM to match components based on shapes (ignore translational invariance).
#     for comp_id in candidate_sketch_component.keys():
#         if comp_id in paired_components:
#             continue
#
#         max_score = 0.0
#         best_ref_id = -1
#
#         # Target component
#         _img = sketch_components[comp_id]
#
#         # Note: Matching between target component and all reference components
#         for ref_comp_id in range(len(ref_sketch_components)):
#             # Reference component
#             ref_img = ref_sketch_components[ref_comp_id]
#
#             # _, score = MatchingShapes.compare_ssim(ref_img, _img)
#             score = ucn_distances[_img['label'], ref_img['label']]
#             score = 0.0 if ref_img['label'] not in ucn_pairs.get(_img['label'], []) else 1.0 - score
#
#             # threshold = THRESHOLD_SSIM_UPPER_BOUND
#             threshold = match_comp_cfg['threshold_ssim_upper_bound']
#             # if _img['area'] > 1000 and ref_img['area'] > 1000:
#             area_threshold = match_comp_cfg['area_thres']
#
#             if _img['area'] > area_threshold and ref_img['area'] > area_threshold:
#                 threshold += match_comp_cfg['area_thres_increment']
#
#             if score >= threshold and score > max_score:
#                 max_score = score
#                 best_ref_id = ref_comp_id
#
#         if best_ref_id >= 0:
#             paired_components[comp_id] = best_ref_id
#             if debug:
#                 logger.debug(">> ref: %d and comp: %d, pass ssim, score: %.4f" % (best_ref_id, comp_id, max_score))
#
#     # Step 3: Color the unpaired components.
#     # Some components (from the reference image) was split to 2/3 new components
#     #   > The above shape matching and pixel matching may not work well
#     #   > Current solution is use templateMatching.
#     template_match_result = {}
#     iou2_lower_bound = match_comp_cfg['threshold_iou2_lower_bound']
#     iou2_upper_bound = match_comp_cfg['threshold_iou2_upper_bound']
#
#     if True:
#         """
#         1. we want to find the ref_img (regarded as a template) in _img (full image).
#         Notes: the relation btw components is the IOU
#         So:
#             > ref_image area must smaller than _img.
#             > the same denominator, so the one with the largest intersection points will be the winner.
#         """
#         for comp_id in candidate_sketch_component.keys():
#             if comp_id in paired_components:
#                 continue
#
#             _img = sketch_components[comp_id]
#             min_dist = 0
#
#             for ref_comp_id in ref_sketch_components.keys():
#                 ref_img = ref_sketch_components[ref_comp_id]
#
#                 # The first constraint is about the area
#                 area_ratio = ref_img['area'] / _img['area']
#                 # if not THRESHOLD_AREA_TEMPLATE_LOWER_BOUND <= area_ratio <= THRESHOLD_AREA_TEMPLATE_UPPER_BOUND:
#                 lower_bound = match_comp_cfg['threshold_area_template_lower_bound']
#                 upper_bound = match_comp_cfg['threshold_area_template_upper_bound']
#
#                 if not (lower_bound <= area_ratio <= upper_bound):
#                     continue
#
#                 if ref_img['area'] < -min_dist:
#                     continue
#
#                 # The second is about iou, and final one is template matching
#                 iou = MatchingShapes.calculate_iou(_img, ref_img, ref_sketch.shape, mode=2)
#                 # iou2_lower_bound = match_comp_cfg['threshold_iou2_lower_bound']
#                 # iou2_upper_bound = match_comp_cfg['threshold_iou2_upper_bound']
#                 if not (iou2_lower_bound <= iou <= iou2_upper_bound):
#                     continue
#
#                 distance, max_val = MatchingShapes.matching_template(comp_im=_img, template=ref_img['image'])
#                 distance = -distance
#
#                 template_match_result[('ref_as_template', comp_id, ref_comp_id)] = (distance, max_val)
#                 if debug:
#                     logger.debug("[finding <ref> to <im>]. ref(template) and comp(im)",
#                                  ref_comp_id, comp_id, max_val, area_ratio)
#
#                 if distance < min_dist and max_val >= match_comp_cfg['threshold_score_template_matching']:
#                     min_dist = distance
#                     paired_components[comp_id] = ref_comp_id
#
#             """
#             2. we want to find the _img (regarded as a template) in ref_img (full image).
#             Notes: the relation btw components is the IOU
#             So:
#                 > ref_image area must smaller than _img.
#                 > the same denominator, so the one with the largest intersection points will be the winner.
#             """
#             for comp_id in candidate_sketch_component.keys():
#                 if comp_id in paired_components:
#                     continue
#
#                 min_dist = 0
#                 _img = sketch_components[comp_id]
#
#                 for ref_comp_id in ref_sketch_components.keys():
#                     ref_img = ref_sketch_components[ref_comp_id]
#
#                     # The first constraint is about the area
#                     area_ratio = _img['area'] / ref_img['area']
#                     area_temp_low_bound = match_comp_cfg['threshold_area_template_lower_bound']
#                     area_temp_up_bound = match_comp_cfg['threshold_area_template_upper_bound']
#                     # if not THRESHOLD_AREA_TEMPLATE_LOWER_BOUND <= area_ratio <= THRESHOLD_AREA_TEMPLATE_UPPER_BOUND:
#                     if not (area_temp_low_bound <= area_ratio <= area_temp_up_bound):
#                         continue
#                     if _img['area'] < -min_dist:
#                         continue
#
#                     # The second is about iou, and final one is template matching
#                     iou = MatchingShapes.calculate_iou(ref_img, _img, ref_sketch.shape, mode=2)
#
#                     if not (iou2_lower_bound <= iou <= iou2_upper_bound):
#                         continue
#
#                     # if iou2_lower_bound <= iou <= iou2_upper_bound and 0.3 <= area_ratio <= 1:
#                     area_ratio_low_bound = match_comp_cfg['area_ratio_lower_bound']
#                     area_ratio_up_bound = match_comp_cfg['area_ratio_upper_bound']
#                     if not (area_ratio_low_bound <= area_ratio <= area_ratio_up_bound):
#                         continue
#
#                     distance, max_val = MatchingShapes.matching_template(ref_img, _img['image'])
#                     distance = -distance
#
#                     template_match_result[('new_as_template', ref_comp_id, comp_id)] = (distance, max_val)
#                     if debug:
#                         logger.debug(
#                             f"[finding <im> to <ref>]. comp(template) and ref(im) {comp_id}, {ref_comp_id}, {max_val}, {area_ratio}")
#
#                     if distance < min_dist and max_val >= match_comp_cfg['threshold_score_template_matching']:
#                         min_dist = distance
#                         paired_components[comp_id] = ref_comp_id
#
#             """
#             3. Colorize the rest by using IOU...
#             """
#             for comp_id in candidate_sketch_component.keys():
#                 if comp_id in paired_components:
#                     continue
#
#                 min_dist = np.inf
#                 _img = sketch_components[comp_id]
#
#                 sorted_ref_sketch_components = {
#                     k: v for k, v in sorted(
#                         ref_sketch_components.items(),
#                         key=lambda item: abs(item[1]['area'] - _img['area']))
#                 }
#
#                 for ref_comp_id in sorted_ref_sketch_components.keys():
#                     ref_img = sorted_ref_sketch_components[ref_comp_id]
#
#                     distance = np.linalg.norm(ref_img['centroid'] - _img['centroid'], ord=2)
#                     if not (distance < min_dist):
#                         continue
#
#                     min_area = iou2_lower_bound * _img['area']
#                     if ref_img['area'] < min_area:
#                         continue
#
#                     iou = MatchingShapes.calculate_iou(ref_img, _img, ref_sketch.shape, mode=2)
#                     # if THRESHOLD_IOU2_LOWER_BOUND <= iou <= THRESHOLD_IOU2_UPPER_BOUND:
#                     if iou2_lower_bound <= iou <= iou2_upper_bound:
#                         if distance < min_dist:
#                             min_dist = distance
#                             paired_components[comp_id] = ref_comp_id
#
#     """
#     4. Maybe the un-close boundary will affect the component which is classified as background
#     So:
#         > Finding the component which is colorized as background.
#         > Run the un-close again
#         > For each new component, assign it with the most IOU one in reference component.
#     """
#     if True:
#         # assumption: the background will be the one which are is the largest
#         background_ref_ids = [
#             _id for _id, _comp in ref_sketch_components.items() if
#             _comp.get('background_predict', False)
#             # np.argmax(areas)
#         ]
#         background_comps = [
#             (_id, comp) for _id, comp in sketch_components.items() if
#             paired_components.get(_id, -1) in background_ref_ids and comp['area'] >= 10000
#             # sketch_components[background_comp_id]
#         ]
#
#         h, w = ref_sketch.shape[0], ref_sketch.shape[1]
#         new_full_points = []
#
#         for (background_comp_id, background_comp) in background_comps:
#             split_components, _new_full_points = may_split_component(background_comp, h, w, unset_points, full_points)
#             new_full_points += _new_full_points
#
#             if split_components:
#                 for new_component_id, new_component in split_components.items():
#                     min_dist = np.inf
#                     matching_ref_id = None
#
#                     for ref_comp_id in ref_sketch_components.keys():
#                         ref_img = ref_sketch_components[ref_comp_id]
#
#                         if ref_img['area'] < -min_dist:
#                             continue
#
#                         s_time = time.time()
#                         iou = MatchingShapes.calculate_iou(ref_img, new_component, ref_sketch.shape, mode=2)
#                         dist = -iou * new_component['area']
#
#                         if debug:
#                             logger.debug(
#                                 "[handle un-close boundary with background] comp: %d-%d with ref: %d, %f" %
#                                 (int(background_comp_id), new_component_id, ref_comp_id, float(dist)))
#                             logger.debug(time.time() - s_time)
#
#                         if dist < min_dist:
#                             min_dist = dist
#                             matching_ref_id = ref_comp_id
#
#                     if new_component_id == 0:
#                         paired_components[background_comp_id] = matching_ref_id
#                     else:
#                         new_id = len(sketch_components)
#                         sketch_components[new_id] = new_component
#                         paired_components[new_id] = matching_ref_id
#
#     if True:
#         centroids = [comp['centroid'] for _, comp in sketch_components.items()]
#         centroid_rows = [_[0] for _ in centroids]
#         centroid_cols = [_[1] for _ in centroids]
#
#         pair = choose_pair_by_distance(centroid_rows, centroid_cols, max_distance=match_comp_cfg['max_pair_distance'])
#
#         for comp_id in candidate_sketch_component.keys():
#             if comp_id in paired_components:
#                 continue
#
#             neighbor_id = pair.get(comp_id, -1)
#             if neighbor_id != -1 and neighbor_id in paired_components:
#                 paired_components[comp_id] = paired_components[neighbor_id]
#
#     return paired_components, new_full_points


def choose_pair_by_distance(rows, cols, max_distance, return_matrix=False):
    """
    Choose the pair for each key_point (r,c) by compare the distance between them.
    :param rows: list of row
    :param cols: list of column
    :param max_distance: max distance to be considered as pair or not
    :param return_matrix
    :return:
    """
    coords = np.array([[row, col] for (row, col) in zip(rows, cols)], dtype=np.int32)  # (n_samples,2)
    if len(coords) == 0:
        return {}

    # sorted by norm2
    distance = np.expand_dims(coords, axis=0) - np.expand_dims(coords, axis=1)  # (n_samples, n_samples, 2)
    distance = np.linalg.norm(distance, ord=2, axis=-1).T  # (n_samples, n_samples)
    distance[np.arange(len(rows)), np.arange(len(rows))] = np.inf

    # get the min distance
    min_ids = np.argmin(distance, axis=-1)  # n_samples,

    pair = {k: v for k, v in enumerate(min_ids) if distance[k, v] <= max_distance}
    if not return_matrix:
        return pair
    return pair, distance


def get_moment_features(components, mask):
    features = np.zeros([mask.shape[0], mask.shape[1], 8])

    for component in components:
        image = component["image"]
        image = cv2.resize(image, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        moments = cv2.moments(image)
        moments = cv2.HuMoments(moments)[:, 0]
        for i in range(0, 7):
            if moments[i] == 0:
                continue
            moments[i] = -1 * copysign(1.0, moments[i]) * log10(abs(moments[i]))

        moments = np.append(moments, component["area"] / 200000.0)
        coords = np.nonzero(mask == component["label"])
        features[coords[0], coords[1], :] = moments

    features = np.transpose(features, (2, 0, 1))
    return features


def build_neighbor_graph(mask):
    max_level = mask.max() + 1
    angles = [0, np.pi * 0.25, np.pi * 0.5, np.pi * 0.75, np.pi]

    matrix = skimage.feature.greycomatrix(mask, [1, 2, 3], angles, levels=max_level, symmetric=True)
    matrix = np.sum(matrix, axis=(2, 3))
    graph = np.zeros((max_level, max_level), dtype=np.int)

    for i in range(1, max_level):
        for j in range(1, max_level):
            if matrix[i, j] > 0:
                graph[i, j] = 1
                graph[j, i] = 1
            if i == j:
                graph[i, j] = 0
    return graph


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


def adapt_component_mask(mask, components, min_area=20, min_size=4):
    components = [components[i] for i in range(0, len(components))]
    new_mask = np.zeros_like(mask)
    new_components = []
    index = 1
    mapping = list()

    for component in components:
        if component["area"] < min_area:
            continue

        box = component["bbox"]
        h, w = abs(box[2] - box[0]), abs(box[3] - box[1])

        if h < min_size:
            continue
        if w < min_size:
            continue
        if h >= mask.shape[0] and w >= mask.shape[1]:
            continue

        mapping.append(component["label"])
        new_component = copy.deepcopy(component)
        new_component["label"] = index
        coords = new_component["coords"]
        new_mask[coords[:, 0], coords[:, 1]] = index
        new_components.append(new_component)
        index += 1
    return new_mask, new_components, mapping


def resize_mask(mask, components, size, mapping):
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
        new_mask = new_mask.astype(np.int32)
        return new_mask, components, mapping

    # some labels cannot be fixed, remove corresponding components
    after_remove_labels = list(np.unique(new_mask).tolist())
    new_components = []
    continuous_mask = np.zeros_like(new_mask)
    num_removed = 0
    new_mapping = []

    for component in components:
        old_label = component["label"]

        if old_label in after_remove_labels:
            component["label"] = old_label - num_removed
            new_mapping.append(mapping[old_label - 1])
            continuous_mask[new_mask == old_label] = component["label"]
            new_components.append(component)
        else:
            num_removed += 1

    # output
    continuous_mask = continuous_mask.astype(np.int32)
    return continuous_mask, new_components, new_mapping


def convert_to_stacked_mask(mask):
    num_labels = mask.max() + 1
    ravel_mask = mask.ravel()
    length = ravel_mask.shape[0]

    stacked_mask = np.zeros([length, num_labels], dtype=np.float)
    stacked_mask[np.arange(length), ravel_mask] = 1
    stacked_mask = np.reshape(stacked_mask, [mask.shape[0], mask.shape[1], num_labels])
    stacked_mask = stacked_mask[..., 1:]
    return stacked_mask


def extract_features(mask, components, sketch_image, size, mean, std):
    mask, components, mapping = adapt_component_mask(mask, components)
    mask, components, mapping = resize_mask(mask, components, size, mapping)
    graph = build_neighbor_graph(mask)
    mask = convert_to_stacked_mask(mask)

    # get features
    image = np.amin(sketch_image, axis=-1)
    image[image < 255] = 0
    image = np.stack([image] * 3, axis=-1)
    image = cv2.resize(image, size).astype(np.float)
    image = image / 255.0
    image = np.transpose(image, (2, 0, 1))

    image = torch.tensor(image).float().unsqueeze(0)
    mask = torch.tensor(mask).float().unsqueeze(0)
    return mask, components, graph, image, mapping


def get_input_data(
        ref_sketch_image, tgt_sketch_image,
        ref_mask, ref_components, tgt_mask, tgt_components,
        size, mean, std):
    # Extract features
    ref_mask, ref_components, ref_graph, ref_features, ref_mapping = extract_features(
        ref_mask, ref_components, ref_sketch_image, size, mean, std)
    tgt_mask, tgt_components, tgt_graph, tgt_features, tgt_mapping = extract_features(
        tgt_mask, tgt_components, tgt_sketch_image, size, mean, std)

    ref_list = [ref_features, ref_mask, ref_components, ref_graph, ref_mapping]
    tgt_list = [tgt_features, tgt_mask, tgt_components, tgt_graph, tgt_mapping]
    return ref_list, tgt_list


def cosine_distance(output_a, output_b):
    # range [-1, 1]
    distances = torch.nn.functional.cosine_similarity(output_a, output_b, dim=-1)
    # range [0, 2]
    distances = torch.ones_like(distances) - distances
    distances /= 2.0
    return distances


def find_neighbor_matches(anchor, components, graph):
    neighbors = []
    nei_indices = []

    for index, component in components.items():
        is_neighbor = graph[anchor["label"], component["label"]]
        if is_neighbor:
            neighbors.append(component)
            nei_indices.append(index)

    return neighbors, nei_indices


def get_top_distances_of_target_component(
        ref_output, tgt_output,
        ref_graph, tgt_graph,
        ref_components, tgt_components,
        k=4,
        area_ratio_threshold=1.6,
        shape_distance_threshold=0.3):

    k = max(min(k, ref_output.shape[1] - 2), 1)
    top_k_per_region, final_top_k_per_region = dict(), dict()
    pairs = []
    all_distances = np.zeros([tgt_output.shape[1], ref_output.shape[1]])

    # find top-k for each region
    for tgt_index in range(0, tgt_output.shape[1]):
        tgt_region = tgt_output[:, tgt_index, :]
        tgt_region = tgt_region.unsqueeze(1).repeat([1, ref_output.shape[1], 1])

        distances = cosine_distance(tgt_region, ref_output)
        all_distances[tgt_index, :] = distances.cpu().numpy()
        min_distances, min_indices = torch.topk(distances, k, largest=False, dim=-1)

        top_k_per_region[tgt_index] = []
        for min_distance, min_index in zip(min_distances.tolist()[0], min_indices.tolist()[0]):
            assert min_distance == all_distances[tgt_index, min_index]
            top_k_per_region[tgt_index].append(min_index)

        final_top_k_per_region[tgt_index] = [top_k_per_region[tgt_index][0]]

    # highly-likely matching
    for tgt_index, top_k in top_k_per_region.items():
        tgt_component = tgt_components[tgt_index]
        ref_component = ref_components[top_k[0]]

        shape_distance = all_distances[tgt_index, top_k[0]]
        if shape_distance > shape_distance_threshold:
            continue

        area_ratio = tgt_component["area"] / ref_component["area"]
        area_ratio = area_ratio if area_ratio > 1.0 else (1.0 / area_ratio)
        if area_ratio > area_ratio_threshold:
            continue

        pairs.append([tgt_index, top_k[0]])

    # neighbor matching
    while True:
        old_len = len(pairs)
        done_tgt_indices = [p[0] for p in pairs]
        done_tgt_components = {i: tgt_components[i] for i in done_tgt_indices}

        for tgt_index, top_k in top_k_per_region.items():
            if tgt_index in done_tgt_indices:
                continue

            tgt_component = tgt_components[tgt_index]
            neighbors, nei_indices = find_neighbor_matches(tgt_component, done_tgt_components, tgt_graph)
            if len(neighbors) == 0:
                continue

            neighbor_pairs = [p for p in pairs if p[0] in nei_indices]
            neighbor_ref_components = [ref_components[p[1]] for p in neighbor_pairs]

            min_index = -1
            for ref_index in top_k:
                ref_component = ref_components[ref_index]
                is_neighbor = any([
                    ref_graph[ref_component["label"], nrc["label"]]
                    for nrc in neighbor_ref_components
                ])

                if is_neighbor:
                    min_index = ref_index
                    break

            if min_index != -1:
                pairs.append([tgt_index, min_index])

        new_len = len(pairs)
        if new_len - old_len <= 0:
            break

    # merge pairs with true_top_k
    for tgt_index, src_index in pairs:
        final_top_k_per_region[tgt_index] = [src_index]

    return top_k_per_region, final_top_k_per_region, all_distances
