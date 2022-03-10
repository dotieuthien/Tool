from base64 import b64encode, b64decode
import json
import os.path
import sys


PY2 = sys.version_info[0] == 2


class LabelFileError(Exception):
    pass


class LabelFile(object):
    suffix = '.json'

    def __init__(self, filename=None):
        self.pairs = {}
        self.imagePath = None
        self.imageData = None
        if filename is not None:
            self.load(filename)

    def load(self, filename):
        try:
            with open(filename, 'rb' if PY2 else 'r') as f:
                data = json.load(f)
                imagePathRight = data['imagePathRight']
                imagePathLeft = data['imagePathLeft']
                pairs = data['pairs']
                self.pairs = pairs
                return pairs
        except Exception as e:
            raise LabelFileError(e)

    def save(self, filename, shapes, imagePath1, imagePath2, imageData, lineColor=None, fillColor=None):
        data = dict(pairs=shapes, imagePathLeft=imagePath1, imagePathRight=imagePath2)
        try:
            with open(filename, 'wb' if PY2 else 'w') as f:
                json.dump(data, f, ensure_ascii=True, indent=2)
        except Exception as e:
            raise LabelFileError(e)

    @staticmethod
    def isLabelFile(filename):
        return os.path.splitext(filename)[1].lower() == LabelFile.suffix
