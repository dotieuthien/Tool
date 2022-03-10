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


def extract_component_from_sketch(sketch):
    assert len(sketch.shape) == 3, 'Require RGB image, got %d' % len(sketch.shape)
    h, w, _ = sketch.shape
    mask = np.ones((h, w)) * -1
    label_mask = np.zeros((h, w))

    # Binary image
    sketch = cv2.cvtColor(sketch, cv2.COLOR_RGB2GRAY)
    sketch = cv2.threshold(sketch, 220, 255, cv2.THRESH_BINARY)[1]
    sketch = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)

    r, g, b = cv2.split(sketch)
    r, g, b = r.astype(np.uint64), g.astype(np.uint64), b.astype(np.uint64)
    processed_image = r + 300 * (g + 1) + 300 * 300 * (b + 1)

    uniques = np.unique(processed_image)
    index = 0
    result = {}
    line_art_value = 0 + 300 + 300 * 300

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

            result[index] = {"centroid": np.array(region.centroid),
                             "area": region.area,
                             "image": region.image.astype(np.uint8) * 255,
                             "label": index + 1,
                             "coords": region.coords,
                             "bbox": region.bbox,
                             "min_intensity": region.min_intensity,
                             "mean_intensity": region.mean_intensity,
                             "max_intensity": region.max_intensity}
            mask[region['coords'][:, 0], region['coords'][:, 1]] = index
            label_mask[region['coords'][:, 0], region['coords'][:, 1]] = index + 1

            index += 1
    print('Len of components of sketch', len(result))
    return result, mask, label_mask


def extract_component_from_image(image):
    # TODO: check again
    """
    Extract components from colored image
    """
    assert len(image.shape) == 3, 'Require RGB image, got %d' % len(image.shape)
    h, w, _ = image.shape
    mask = np.ones((h, w)) * -1
    label_mask = np.zeros((h, w))

    b, r, g = cv2.split(image)
    processed_image = np.array(b + 300 * (g + 1) + 300 * 300 * (r + 1))
    uniques = np.unique(processed_image)

    index = 0
    result = {}
    # line_art_value = get_line_art_value(image)

    for unique in uniques:
        # Get coords by color
        # if unique == line_art_value:
        #     continue
        rows, cols = np.where(processed_image == unique)
        image_tmp = np.zeros_like(processed_image)
        image_tmp[rows, cols] = 255

        # Get components
        labels = measure.label(image_tmp, connectivity=1, background=0)

        for region in measure.regionprops(labels, intensity_image=processed_image):
            if region['area'] <= 10:
                continue

            result[index] = {"centroid": np.array(region.centroid),
                             "area": region.area,
                             "image": region.image.astype(np.uint8) * 255,
                             "label": index + 1,
                             "coords": region.coords,
                             "bbox": region.bbox,
                             "min_intensity": region.min_intensity,
                             "mean_intensity": region.mean_intensity,
                             "max_intensity": region.max_intensity}
            mask[region['coords'][:, 0], region['coords'][:, 1]] = index
            label_mask[region['coords'][:, 0], region['coords'][:, 1]] = index + 1
            index += 1
    print('Len of components colored image', len(result))
    return result, mask, label_mask


def extract_component_from_mask(mask):
    """
    Extract components from mask, keep the id of component as the color in the mask
    label = component id + 1
    """
    result = {}
    labels = mask

    h, w = labels.shape
    mask = np.ones((h, w)) * -1

    for region in measure.regionprops(labels, intensity_image=labels):
        index = region.max_intensity - 1
        result[index] = {"centroid": np.array(region.centroid),
                         "area": region.area,
                         "image": region.image.astype(np.uint8) * 255,
                         "label": index + 1,
                         "coords": region.coords,
                         "bbox": region.bbox,
                         "min_intensity": region.min_intensity,
                         "mean_intensity": region.mean_intensity,
                         "max_intensity": region.max_intensity}
        mask[region['coords'][:, 0], region['coords'][:, 1]] = index

    print('Len of components from mask', len(result))
    return result, mask
