import os
import glob
import shutil
import random
from natsort import natsorted


class AverageMeter:
    def __init__(self):
        self.value = 0
        self.sum = 0
        self.average = 0
        self.count = 0

    def reset(self):
        self.value = 0
        self.sum = 0
        self.average = 0
        self.count = 0

    def update(self, value, count=1):
        self.value = value
        self.sum += value * count
        self.count += count
        self.average = self.sum / self.count

    def get(self):
        return self.average


def get_full_class_name(o):
    klass = o.__class__
    module = klass.__module__
    if module == "builtins":
        return klass.__qualname__
    return module + "." + klass.__qualname__


def remove_pickle():
    root_dirs = [
        "/home/tyler/work/data/GeekToys/coloring_data/complete_data",
        "/home/tyler/work/data/ToeiAni/complete_data",
        "/mnt/ai_filestore/home/tyler/data/painting/complete_data",
        "/home/tyler/work/data/ToeiAni/cin_labeled_data/toei_easy_training_set",
        "/home/tyler/data/toei_easy_training_set",
    ]
    root_dir = root_dirs[4]
    paths = natsorted(glob.glob(os.path.join(root_dir, "*", "*", "*.pkl")))
    print(len(paths))
    print(paths)

    for path in paths:
        os.remove(path)
    return


def remove_label_data():
    root_dir = "/home/tyler/data/toei_easy_training_set"
    paths = natsorted(glob.glob(os.path.join(root_dir, "*", "color", "annotations", "*")))
    print(len(paths))

    for path in paths:
        os.remove(path)
    return


if __name__ == "__main__":
    remove_pickle()
