import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from orm import Meme
from backend.utils import Log
from config import *
from backend.feature_extract import InceptionExtractor


class MemeDataset(Dataset):
    def __init__(self, meme_path: str):
        self.meme_list = []  # tuple like (meme_path, meme_title)
        self.get_meme_path_dict(meme_path)
        self.transforms = InceptionExtractor.transform

    def __len__(self):
        return len(self.meme_list)

    def __getitem__(self, item):
        meme_path, title = self.meme_list[item]
        img = Image.open(meme_path).convert('RGB')
        m_img = self.transforms()(img)
        return {'title': title, 'm_img': m_img, 'meme_path': meme_path}

    def get_meme_path_dict(self, meme_path: str):
        """
        Load all pictures into meme_dict, as an interface of torch.utils.data.Dataset
        :param meme_path: root path of meme pictures.
        :return:
        """
        valid_cnt, invalid_cnt = 0, 0
        if not os.path.exists(meme_path):
            raise OSError("Path " + meme_path + " Not Found")
        for root, dirs, files in os.walk(meme_path, topdown=True):
            # walk through meme_path. File layout is similar with `meme_path/meme_title/1.png`
            if not dirs:  # if os.path.join(root, name) is a file
                title = root.split(os.path.sep)[-1]  # meme_title here
                for name in files:
                    img_path = os.path.join(root, name)
                    if Meme.get_or_none(path=img_path):
                        Log.info(img_path + " already in Database! ")
                        invalid_cnt += 1
                        continue  # skip those already in database
                    if not name.lower().endswith(allow_img_extensions):
                        Log.info(name + " not supported, only " + str(
                            allow_img_extensions) + " extensions supported for now.")
                        invalid_cnt += 1
                        continue
                    self.meme_list.append((img_path, title))
        valid_cnt = len(self.meme_list)
        Log.info("Valid Meme Pictures Count: " + str(valid_cnt) + ', Invalid Count: ' + str(invalid_cnt))


if __name__ == '__main__':
    dataset = MemeDataset('./backend/meme/')
    dataloader = DataLoader(dataset, batch_size=3,
                            shuffle=True, num_workers=1, drop_last=False)
    for i, batched_data in enumerate(dataloader):
        m_img = batched_data['m_img']  # [N, 3, 299, 299]
        title = batched_data['title']
        meme_path = batched_data['meme_path']
        print(m_img, title, meme_path)
