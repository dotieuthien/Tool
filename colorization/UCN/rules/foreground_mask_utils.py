import os
import glob
import json
import numpy as np
import cv2
from natsort import natsorted
from PIL import Image


def get_color_region(image):
    color_sum = np.sum(image.astype(np.int), axis=-1)
    mask = (color_sum < 750).astype(np.uint8)

    mask = cv2.erode(mask, np.ones([5, 5], dtype=np.uint8))
    mask = cv2.dilate(mask, np.ones([15, 15], dtype=np.uint8), iterations=3)
    box = cv2.boundingRect(mask)

    return mask, [box[0], box[1], box[0] + box[2], box[1] + box[3]]


def is_small_fgb(fgb, threshold=300):
    return fgb[2] - fgb[0] < threshold and fgb[3] - fgb[1] < threshold


def update_box(gb, box):
    if gb is None:
        return box

    gb[0] = min(gb[0], box[0])
    gb[1] = min(gb[1], box[1])
    gb[2] = max(gb[2], box[2])
    gb[3] = max(gb[3], box[3])
    return gb


def main():
    data_dir = "/home/tyler/work/data/GeekInt/real_data/test_data_for_interpolation_phase_1"
    output_dir = "/home/tyler/work/data/GeekInt/output/mask"

    cut_paths = natsorted(glob.glob(os.path.join(data_dir, "*", "*")))
    json_data = dict()

    for cut_path in cut_paths:
        cut_name = os.path.basename(os.path.dirname(cut_path))
        sub_name = os.path.basename(cut_path)
        cut_name = "%s" % sub_name
        paths = natsorted(glob.glob(os.path.join(cut_path, "*.png")))
        json_data[cut_name] = None

        for path in paths:
            image = Image.open(path).convert("RGB")
            image = np.array(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mask, box = get_color_region(image)
            json_data[cut_name] = update_box(json_data[cut_name], box)

        for path in paths:
            name = os.path.splitext(os.path.basename(path))[0]
            full_name = "%s_%s" % (cut_name, name)

            image = Image.open(path).convert("RGB")
            image = np.array(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            gb = json_data[cut_name]
            cv2.rectangle(image, (gb[0], gb[1]), (gb[2], gb[3]), (255, 0, 0))
            cv2.imwrite(os.path.join(output_dir, "%s.png" % full_name), image)

    json.dump(
        json_data,
        open(os.path.join(output_dir, "foreground_data.json"), "w+"),
        indent=4)


if __name__ == "__main__":
    main()
