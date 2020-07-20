from abc import abstractmethod, ABC

import tensorflow as tf
import numpy as np
import os
import time
import base64
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
from backend.utils import Log


class FeatureExtractor(object):
    def __init__(self, img_extensions=None, batch_size=1):
        if img_extensions is None:
            img_extensions = ['.jpg', '.png', '.jpeg', '.gif']
        self.img_extensions = img_extensions
        self.batch_size = batch_size

    def is_image(self, filename: str):
        return filename.lower().endswith(self.img_extensions)

    @abstractmethod
    def get_feature(self, img_mat: Variable):
        pass

    @abstractmethod
    def read_images(self):
        pass


class InceptionExtractor(FeatureExtractor, ABC):
    img_size = (299, 299)
    feature_shape = (1, 2048)
    feature_type = np.float32

    def __init__(self, img_extensions=None, batch_size=1):
        # Official Model from here: http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
        super().__init__(img_extensions, batch_size)
        self.model = InceptionExtractor.init_inception()
        self.transforms = transforms.Compose([
            transforms.Resize(InceptionExtractor.img_size),
            transforms.CenterCrop(InceptionExtractor.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def read_images(self):
        pass

    def get_feature(self, img_mat: Variable):
        """
        Extract features from certain image(s)
        :param img_mat: images variable with the shape of (N, 3, 299, 299), where N indicates batch_size
        Note that N = 1 when answering users' messages
        :return: feature matrix with the shape of (N, 2048)
        """
        return self.model(img_mat)

    @staticmethod
    def bytes2ndarray(base64_str: str) -> np.ndarray:
        decode_b64_data = base64.urlsafe_b64decode(base64_str)
        ndarray = np.frombuffer(decode_b64_data, dtype=InceptionExtractor.feature_type)
        ndarray = ndarray.reshape(InceptionExtractor.feature_shape)
        return ndarray

    @staticmethod
    def ndarray2bytes(array: np.ndarray) -> str:
        return base64.urlsafe_b64encode(array.tobytes())

    @staticmethod
    def init_inception():
        inception = models.inception_v3(pretrained=True)
        inception.fc = nn.Identity()
        inception.eval()
        return inception
