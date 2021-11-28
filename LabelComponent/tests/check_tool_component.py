import os
import numpy as np
import cv2
import glob
import pickle
import json
import skimage.measure as measure
from natsort import natsorted
from PIL import Image


def get_line_art_value(image):
    line_colors = []

    # Convert rgb image to (h, w, 1)
    b, r, g = cv2.split(image)
    processed_image = np.array(b + 300 * (g + 1) + 300 * 300 * (r + 1))
    uniques = np.unique(processed_image)

    for unique in uniques:
        rows, cols = np.where(processed_image == unique)
        image_tmp = np.zeros_like(processed_image)
        image_tmp[rows, cols] = 255

        # Get components
        labels = measure.label(image_tmp, connectivity=1, background=0)

        for region in measure.regionprops(labels, intensity_image=processed_image):
            if region["area"] <= 10:
                continue
            image_tmp_ = np.zeros_like(processed_image)
            coord = region["coords"]
            image_tmp_[coord[:, 0], coord[:, 1]] = 255

            contours = measure.find_contours(image_tmp_, 0.8)
            if len(contours) != 1:
                continue

            contour = np.array(contours[0], dtype=np.int)
            color_values, counts = np.unique(processed_image[contour[:, 0], contour[:, 1]], return_counts=True)
            line_colors.extend(color_values)

            if len(line_colors) > 10:
                return max(set(line_colors), key=line_colors.count)

    return max(set(line_colors), key=line_colors.count)


def extract_component(image):
    assert len(image.shape) == 3, "Require RGB image, got %d" % len(image.shape)
    h, w, _ = image.shape
    mask = np.ones((h, w)) * -1

    # Convert rgb image to (h, w, 1)
    b, r, g = cv2.split(image)
    processed_image = np.array(b + 300 * (g + 1) + 300 * 300 * (r + 1))
    uniques = np.unique(processed_image)

    index = 0
    result = {}
    bad_values = get_line_art_value(image)

    for unique in uniques:
        # Get coords by color
        if unique == bad_values:
            continue
        rows, cols = np.where(processed_image == unique)
        image_tmp = np.zeros_like(processed_image)
        image_tmp[rows, cols] = 255

        # Get components
        labels = measure.label(image_tmp, connectivity=1, background=0)

        for region in measure.regionprops(labels, intensity_image=processed_image):
            if region["area"] <= 10:
                continue

            result[index] = {
                "centroid": np.array(region.centroid),
                "area": region.area,
                "image": region.image.astype(np.uint8) * 255,
                "label": index + 1,
                "coords": region.coords,
                "bbox": region.bbox,
                "min_intensity": region.min_intensity,
                "mean_intensity": region.mean_intensity,
                "max_intensity": region.max_intensity}
            mask[region["coords"][:, 0], region["coords"][:, 1]] = index
            index += 1

    result = [result[i] for i in range(0, len(result))]
    return result, mask


def read_image(path):
    if path.endswith(".tga"):
        image = cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)
    else:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
    return image


def main():
    root_dir = "/home/hades/Downloads/labeled_data"
    output_dir = "/home/hades/Downloads/labeled_data/output"

    cut_dirs = natsorted(glob.glob(os.path.join(root_dir, "*")))
    use_image = True

    for cut_dir in cut_dirs[:1]:
        label_paths = natsorted(glob.glob(os.path.join(cut_dir, "color", "annotations", "*.json")))

        for label_path in label_paths:
            label_data = json.load(open(label_path))
            pairs = label_data["pairs"]
            pairs = [(v["left"], v["right"]) for k, v in pairs.items()]

            left_file_name = os.path.basename(label_data["imagePathLeft"].replace("\\", "/"))
            right_file_name = os.path.basename(label_data["imagePathRight"].replace("\\", "/"))

            left_path = os.path.join(cut_dir, "color", left_file_name)
            right_path = os.path.join(cut_dir, "color", right_file_name)
            left_image, right_image = read_image(left_path), read_image(right_path)
            raw_debug_image = np.concatenate([left_image, right_image], axis=1)
            sw = left_image.shape[1]

            full_name = "%s_%s_" % (os.path.basename(cut_dir), os.path.splitext(left_file_name)[0])
            file_output_dir = os.path.join(output_dir, full_name)
            os.makedirs(file_output_dir)

            if use_image:
                left_components, _ = extract_component(left_image)
                right_components, _ = extract_component(right_image)
            else:
                left_pkl_path = os.path.join(os.path.dirname(label_path), left_file_name.replace("tga", "pkl"))
                left_components = pickle.load(open(left_pkl_path, "rb"))["components"]
                left_components = [left_components[i] for i in range(0, len(left_components))]

                right_pkl_path = os.path.join(os.path.dirname(label_path), right_file_name.replace("tga", "pkl"))
                right_components = pickle.load(open(right_pkl_path, "rb"))["components"]
                right_components = [right_components[i] for i in range(0, len(right_components))]

            for pair in pairs:
                lc = [c for c in left_components if (c["label"] - 1) == pair[0]]
                rc = [c for c in right_components if (c["label"] - 1) == pair[1]]

                if len(lc) == 0 or len(rc) == 0:
                    continue
                else:
                    lc, rc = lc[0], rc[0]

                pl, pr = lc["centroid"], rc["centroid"]
                debug_image = raw_debug_image.copy()
                line_color = (0, 255, 0)

                cv2.line(
                    debug_image,
                    (int(pl[1]), int(pl[0])),
                    (int(pr[1]) + sw, int(pr[0])),
                    color=line_color, thickness=3)
                cv2.circle(debug_image, (int(pl[1]), int(pl[0])), radius=6, color=(0, 0, 255), thickness=3)
                cv2.circle(debug_image, (int(pr[1]) + sw, int(pr[0])), radius=6, color=(0, 0, 255), thickness=3)

                cv2.imwrite(os.path.join(file_output_dir, "%s_%s.png" % (pair[0], pair[1])), debug_image)
    return


if __name__ == "__main__":
    main()