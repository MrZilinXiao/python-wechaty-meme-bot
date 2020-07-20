import os
from torch.utils.data import Dataset
from orm import Meme


class MemeDataset(Dataset):
    def __init__(self, meme_path: str):
        pass

    @staticmethod
    def get_meme_path_dict(meme_path):
        meme_dict = {}
        for root, dirs, files in os.walk(meme_path, topdown=True):
            # walk through meme_path. File layout is similar with `meme_path/meme_title/1.png`
            if not dirs:
                title = root.split(os.path.sep)[-1]  # meme_title here
                for name in files:
                    img_path = os.path.join(root, name)
                    if Meme.get_or_none(path=img_path):
                        continue  # skip those already in database
                    # TODO: finish remaining
