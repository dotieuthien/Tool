import numpy as np
import cv2
from skimage import measure


def get_line_art_value(image):
    # TODO improvement
    assert len(image.shape) == 3, 'Require RGB image, got %d' % len(image.shape)

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
            if region['area'] <= 10:
                continue
            image_tmp_ = np.zeros_like(processed_image)
            coord = region['coords']
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


def extract_component_from_image(image):
    # TODO: check again
    assert len(image.shape) == 3, 'Require RGB image, got %d' % len(image.shape)
    h, w, _ = image.shape
    mask = np.ones((h, w)) * -1

    b, r, g = cv2.split(image)
    processed_image = np.array(b + 300 * (g + 1) + 300 * 300 * (r + 1))
    uniques = np.unique(processed_image)

    index = 0
    result = {}
    line_art_value = get_line_art_value(image)

    for unique in uniques:
        # Get coords by color
        if unique == line_art_value:
            continue
        rows, cols = np.where(processed_image == unique)
        image_tmp = np.zeros_like(processed_image)
        image_tmp[rows, cols] = 255

        # Get components
        labels = measure.label(image_tmp, connectivity=1, background=0)

        for region in measure.regionprops(labels, intensity_image=processed_image):
            if region['area'] <= 10:
                continue

            result[index] = {"centroid": np.array(region.centroid), "area": region.area,
                             "image": region.image.astype(np.uint8) * 255, "label": index + 1,
                             "coords": region.coords, "bbox": region.bbox, "min_intensity": region.min_intensity,
                             "mean_intensity": region.mean_intensity, "max_intensity": region.max_intensity}
            mask[region['coords'][:, 0], region['coords'][:, 1]] = index
            index += 1
    return result, mask


def extract_component_from_mask(mask):
    result = {}
    labels = mask
    index = 0
    for region in measure.regionprops(labels, intensity_image=mask):
        if region['area'] <= 10:
            continue

        result[index] = {"centroid": np.array(region.centroid), "area": region.area,
                         "image": region.image.astype(np.uint8) * 255, "label": index + 1,
                         "coords": region.coords, "bbox": region.bbox, "min_intensity": region.min_intensity,
                         "mean_intensity": region.mean_intensity, "max_intensity": region.max_intensity}
        mask[region['coords'][:, 0], region['coords'][:, 1]] = index
        index += 1
    return result