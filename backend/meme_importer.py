"""
用于手动导入已收集好的表情包进入数据库及提取特征
Usage: python3 ./meme_importer.py [memepath]

[title]/*.[extension]

[title]将被作为表情的title字段,[extension]可为jpg,png,gif，
在当前gif暂时被当做只包含第一帧的静态图片；
tag将由OCR结果分词、去语气词、去重后得到；每张表情包将由InceptionV3提取特征；

Notes: under `meme` folder, there are 3 subdirs:
1. classified: meaning all folders in it indicates a similar class of meme.
Only meme in this dir will act as Cosine Metric Learning data source

2. unclassified: folders in it do not name a class, and they exist only due to
being related with the same subject

3. others: images in it are all independent, meaning no connections with each other
"""
from backend.hanlp_wrapper import HanlpWrapper
from backend.ocr_wrapper import OCRWrapper
from orm import Meme
from backend.feature_extract import InceptionExtractor, CosineMetricExtractor, NoneExtractor
from backend.utils import Log
from PIL import Image
from torch.autograd import Variable
from backend.config import *
import numpy as np
import time
import torch
import peewee
import os


class BaseImporter(object):
    def __init__(self, meme_path, ocr_type='mobilenetv2', extractor_type='inception', batch_size=16, num_workers=1):
        if not os.path.exists(meme_path):
            raise FileNotFoundError("{} doesn't exist!".format(meme_path))
        self.hanlp = HanlpWrapper()
        if ocr_type == 'mobilenetv2':
            self.ocr = OCRWrapper('mobilenetv2')
        else:
            raise NotImplementedError
        if extractor_type == 'inception':
            self.extractor = InceptionExtractor(allow_img_extensions, batch_size)
        elif extractor_type == 'cosine':
            self.extractor = CosineMetricExtractor(allow_img_extensions, batch_size)
        elif extractor_type == 'none':
            self.extractor = NoneExtractor()
        else:
            raise NotImplementedError
        self.extractor.init_dataloader(meme_path, num_workers=num_workers)

    @staticmethod
    def check_chinese(string: str) -> bool:
        """
        check if a string contains any Chinese character
        we assume such memes don't need OCR process, since their filenames tell what's in there
        :param string:
        :return: bool
        """
        for ch in string:
            if u'\u4e00' <= ch <= u'\u9fff':
                return True
        return False

    def import_meme(self):
        for k, batched_data in enumerate(self.extractor.data_loader):
            st_time = time.time()
            m_img = Variable(batched_data['m_img'])  # [batch_size, 3, 299, 299]
            title = batched_data['title']
            meme_path = batched_data['meme_path']
            if torch.cuda.is_available():
                m_img = m_img.cuda()
            v_img = self.extractor.get_feature(m_img)  # [batch_size, 2048]
            for i in range(len(title)):
                if self.check_chinese(os.path.basename(meme_path[i])):
                    text = os.path.basename(meme_path[i]).split(os.path.extsep)[0]
                else:
                    img_data = Image.open(meme_path[i]).convert('RGB')
                    img_data = np.array(img_data)
                    texts = self.ocr.text_predict(img_data)
                    text = ''.join(texts)
                tags = self.hanlp.Tokenizer(text)
                tags_text = ' '.join(tags)
                if v_img is None:
                    feature_encoded = ''
                else:
                    feature_vector = v_img[i].cpu().detach()
                    feature_encoded = self.extractor.ndarray2bytes(np.array(feature_vector))
                try:
                    Meme.create(path=meme_path[i], title=title[i], tag=tags_text, feature=feature_encoded, raw_text=text)
                except peewee.IntegrityError as e:
                    Log.info(str(e))
                    continue
                Log.info("Import Meme with title:" + title[i] + ", tag:" + tags_text + ', raw text:' + text)
            Log.info("This batch consumes %.2f seconds..." % (time.time() - st_time))


if __name__ == '__main__':
    importer = BaseImporter(meme_path='./backend/meme/', num_workers=12, extractor_type='none')
    importer.import_meme()
