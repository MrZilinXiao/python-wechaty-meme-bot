from abc import abstractmethod, ABC
import numpy as np
import base64
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from backend import config
from backend.cosine_metric_net import CosineMetricNet
import collections


class FeatureExtractor(object):
    transforms = None

    def __init__(self, img_extensions=None, batch_size=1):
        if img_extensions is None:
            img_extensions = ('.jpg', '.png', '.jpeg', '.gif')
        self.img_extensions = img_extensions
        self.batch_size = batch_size
        self.use_cuda = torch.cuda.is_available()

    def is_image(self, filename: str):
        return filename.lower().endswith(self.img_extensions)

    def init_dataloader(self, meme_path: str, num_workers: int = config.num_cores):
        """
        Init a torch.utils.data.Dataset instance for import use
        :param num_workers:
        :param meme_path:
        :return:
        """
        from backend.dataset import ImportDataset
        self.dataset = ImportDataset(meme_path, transforms=self.transforms)
        self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size,
                                      shuffle=True, num_workers=num_workers, drop_last=False)

    @abstractmethod
    def get_feature(self, img_mat: Variable):
        pass

    @staticmethod
    def bytes2ndarray(base64_str: str) -> np.ndarray:
        decode_b64_data = base64.urlsafe_b64decode(base64_str)
        ndarray = np.frombuffer(decode_b64_data, dtype=InceptionExtractor.feature_type)
        ndarray = ndarray.reshape(InceptionExtractor.feature_shape)
        return ndarray

    @staticmethod
    def ndarray2bytes(array: np.ndarray) -> str:
        return base64.urlsafe_b64encode(array.tobytes())


class InceptionExtractor(FeatureExtractor, ABC):
    """
    Since Inception is not trainable, we integrate training utils like dataloader into InceptionExtractor
    However, other extractors won't follow
    """
    img_size = (299, 299)
    feature_shape = (1, 2048)
    feature_type = np.float32
    transforms = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def __init__(self, img_extensions=None, batch_size=1):
        super().__init__(img_extensions, batch_size)
        self.model = InceptionExtractor._init_inception()
        # self.transforms = self.transform
        self.dataset = None
        self.data_loader = None

    def get_feature(self, img_mat: Variable):
        """
        Extract features from certain image(s)
        :param img_mat: images variable with the shape of (N, 3, 299, 299), where N indicates batch_size
        Note that N = 1 when answering users' messages
        :return: feature matrix with the shape of (N, 2048)
        """
        if self.use_cuda:
            img_mat = img_mat.cuda()
        out = self.model(img_mat)
        return F.normalize(out, dim=1, p=2)  # inception need L2 normalize

    @staticmethod
    def _init_inception():
        inception = models.inception_v3(pretrained=True)
        if torch.cuda.is_available():
            inception = inception.cuda()
        inception.fc = nn.Identity()  # replace fully-connected layer with an Identity to get 2048 feature vector
        inception.eval()
        return inception


class CosineMetricExtractor(FeatureExtractor, ABC):
    exclude_keys = ('weights', 'scale')  # weights & scale not needed since they were only useful for training
    img_size = (128, 64)
    feature_shape = (1, 128)
    feature_type = np.float32
    test_transforms = transforms.Compose([
        transforms.Resize((128, 64)),
        transforms.ToTensor(),
    ])

    def __init__(self, img_extensions: tuple, batch_size=1, model_path='./backend/cosine_model.pt'):
        super(CosineMetricExtractor, self).__init__(img_extensions=img_extensions, batch_size=batch_size)
        self.model = CosineMetricNet(num_classes=-1, add_logits=False)
        if self.use_cuda:
            self.model = self.model.cuda()
        # load cosine metric model
        try:
            ckpt: collections.OrderedDict = torch.load(model_path)
            ckpt['model_state_dict'] = {k: v for k, v in ckpt['model_state_dict'].items()
                                        if k not in self.exclude_keys}
            self.model.load_state_dict(ckpt['model_state_dict'], strict=False)
        except KeyError as e:
            s = "Model loaded(%s) is not compatible with the definition, please check!" % model_path
            raise KeyError(s) from e

    # def init_dataloader(self, meme_path: str, num_workers: int = config.num_cores):

    def get_feature(self, img_mat: Variable):
        if self.use_cuda:
            img_mat = img_mat.cuda()
        return self.model(img_mat)
