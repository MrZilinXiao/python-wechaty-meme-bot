from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from abc import abstractmethod, ABC


class BaseDataset(Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()
        self.transforms = None

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, item):
        pass


class CropTrainDataset(BaseDataset, ABC):
    def __init__(self, path_list, label_list):
        super(CropTrainDataset, self).__init__()
        self.transforms = transforms.Compose([
            transforms.Resize((128, 64)),
            transforms.ColorJitter(hue=.05, saturation=.05),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20, resample=Image.BILINEAR),
            transforms.ToTensor()
        ])
        self.path_list = path_list
        self.label_list = label_list
        assert len(path_list) == len(label_list), "Length of both lists should equal!"

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, item):
        img = Image.open(self.path_list[item]).convert("RGB")
        img = self.transforms(img)
        return img, self.label_list[item]


class CropTestDataset(CropTrainDataset):
    def __init__(self, path_list, label_list):
        super(CropTestDataset, self).__init__(path_list, label_list)
        self.transforms = transforms.Compose([
            transforms.Resize((128, 64)),
            transforms.ToTensor()
        ])
