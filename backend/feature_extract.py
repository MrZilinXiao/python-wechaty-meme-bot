from abc import abstractmethod, ABC
import numpy as np
import os
import time
import base64
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from PIL import Image
from backend.utils import Log


class FeatureExtractor(object):
    def __init__(self, img_extensions=None, batch_size=1):
        if img_extensions is None:
            img_extensions = ('.jpg', '.png', '.jpeg', '.gif')
        self.img_extensions = img_extensions
        self.batch_size = batch_size

    def is_image(self, filename: str):
        return filename.lower().endswith(self.img_extensions)

    @abstractmethod
    def get_feature(self, img_mat: Variable):
        pass

    @abstractmethod
    def init_dataloader(self, **kwargs):
        pass

    @staticmethod
    def transform():
        pass


class InceptionExtractor(FeatureExtractor, ABC):
    img_size = (299, 299)
    feature_shape = (1, 2048)
    feature_type = np.float32

    def __init__(self, img_extensions=None, batch_size=1):
        super().__init__(img_extensions, batch_size)
        self.model = InceptionExtractor._init_inception()
        self.transforms = self.transform
        self.dataset = None
        self.data_loader = None
        self.use_cuda = torch.cuda.is_available()

    def init_dataloader(self, meme_path: str):
        """
        Init a torch.utils.data.Dataset instance for future use
        :param meme_path:
        :return:
        """
        from dataset import MemeDataset
        self.dataset = MemeDataset(meme_path)
        self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size,
                                      shuffle=True, num_workers=1, drop_last=False)

    def get_feature(self, img_mat: Variable):
        """
        Extract features from certain image(s)
        :param img_mat: images variable with the shape of (N, 3, 299, 299), where N indicates batch_size
        Note that N = 1 when answering users' messages
        :return: feature matrix with the shape of (N, 2048)
        """
        if self.use_cuda:
            img_mat = img_mat.cuda()
        return self.model(img_mat)

    @staticmethod
    def transform():
        return transforms.Compose([
            transforms.Resize(InceptionExtractor.img_size),
            transforms.CenterCrop(InceptionExtractor.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

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
    def _init_inception():
        inception = models.inception_v3(pretrained=True)
        if torch.cuda.is_available():
            inception = inception.cuda()
        inception.fc = nn.Identity()  # replace fully-connected layer with an Identity to get 2048 feature vector
        inception.eval()
        return inception
