import random
import numpy as np
from loader import features_utils


def random_remove_component(mask, components, max_area=300, max_removed=2, remove_prob=0.2):
    if len(components) < 10 or random.random() < 0.7:
        return mask, components, False

    index = 1
    new_components = []
    new_mask = np.zeros(mask.shape, dtype=np.int)
    removed = 0

    for component in components:
        if component["area"] > max_area or removed >= max_removed or random.random() > remove_prob:
            component["label"] = index
            new_components.append(component)
            new_mask[component["coords"][:, 0], component["coords"][:, 1]] = index
            index += 1
        else:
            removed += 1
    return new_mask, new_components, removed > 0


def add_random_noise(image):
    noise = np.random.normal(loc=0.0, scale=0.04, size=image.shape)
    image = image + noise
    image = np.clip(image, a_min=0.0, a_max=1.0)
    return image
