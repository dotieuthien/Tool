import math
import numpy as np
import cv2


SKETCH_FEATURES = "sketch_features"
MOMENT_FEATURES = "moment_features"


def normalize_with_two_segments(x, min_v, mean_v, max_v):
    if x <= mean_v:
        y = ((x - min_v) / (mean_v - min_v)) - 1
    else:
        y = ((x - mean_v) / (max_v - mean_v))
    return y


def get_sketch_image_raw_pixels(sketch_image, size):
    sketch_image = cv2.resize(sketch_image, size).astype(np.float32)
    sketch_image = sketch_image / 255.0
    sketch_image = np.transpose(sketch_image, (2, 0, 1))
    return sketch_image


def get_moment_features(components, mask):
    mean = [2.0, 7.0, 20.0, 20.0, 10.0, 0.0, 0.0, 0.0]
    std = [0.8, 2.0, 10.0, 10.0, 30.0, 20.0, 30.0, 393216.0]
    features = np.zeros([mask.shape[0], mask.shape[1], 8])

    for component in components:
        image = component["image"]
        moments = cv2.moments(image)
        moments = cv2.HuMoments(moments)[:, 0]

        for i in range(0, 7):
            if moments[i] == 0:
                continue
            moments[i] = -1 * math.copysign(1.0, moments[i]) * math.log10(abs(moments[i]))
            moments[i] = (moments[i] - mean[i]) / std[i]

        moments = np.append(moments, component["area"] / std[7])
        coords = np.nonzero(mask == component["label"])
        features[coords[0], coords[1], :] = moments

    features = np.transpose(features, (2, 0, 1))
    return features


def convert_to_stacked_mask(mask):
    num_labels = mask.max() + 1
    ravel_mask = mask.ravel()
    length = ravel_mask.shape[0]

    stacked_mask = np.zeros([length, num_labels], dtype=np.float32)
    stacked_mask[np.arange(length), ravel_mask] = 1
    stacked_mask = np.reshape(stacked_mask, [mask.shape[0], mask.shape[1], num_labels])
    stacked_mask = stacked_mask[..., 1:]
    return stacked_mask
