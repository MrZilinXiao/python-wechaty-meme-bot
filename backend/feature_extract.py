from abc import abstractmethod

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import time
import base64
from backend.utils import Log


class FeatureExtractor(object):
    def __init__(self, img_extensions=None):
        if img_extensions is None:
            img_extensions = ['.jpg', '.png']
        self.img_extensions = img_extensions

    @abstractmethod
    def GetFeature(self, img_path: str):
        pass

    def ReadImages(self, img_path: str) -> [np.numarray, np.numarray]:
        # TODO: Rewrite using tf.data.Dataset
        img_data, img_label = [], []

        def ParseImgs(file: str):
            return tf.gfile.FastGFile(os.path.join(file), 'rb').read()

        if os.path.isdir(img_path):
            for file in os.listdir(img_path):
                if '.' + file.split('.')[-1].lower() in self.img_extensions:
                    img_data.append(ParseImgs(file))
                    img_label += [file[:-4]]  # doesn't matter since labels are useless for feature extraction
        assert os.path.isfile(img_path)
        if '.' + img_path.split('.')[-1].lower() in self.img_extensions:
            img_data.append(ParseImgs(img_path))
            img_label += [img_path.split('/')[-1][:-4]]
        return np.array(img_data), np.array(img_label)


class InceptionExtractor(FeatureExtractor):
    feature_shape = (1, 2048)
    feature_type = np.float32

    def __init__(self, model_path='./Inception/tensorflow_inception_graph.pb',
                 img_extensions=['.jpg', '.png']):
        # Official Model from here: http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
        super().__init__(img_extensions)
        self.model_path = model_path
        self.LoadInception()

    @staticmethod
    def bytes2ndarray(base64_str: str) -> np.ndarray:
        decode_b64_data = base64.urlsafe_b64decode(base64_str)
        ndarray = np.frombuffer(decode_b64_data, dtype=InceptionExtractor.feature_type)
        ndarray = ndarray.reshape(InceptionExtractor.feature_shape)
        return ndarray
    @staticmethod
    def ndarray2bytes(array: np.ndarray) -> str:
        return base64.urlsafe_b64encode(array.tobytes())

