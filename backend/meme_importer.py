"""
用于手动导入已收集好的表情包进入数据库及提取特征
Usage: python3 ./meme_importer.py [memepath]

[title]/*.[extension]

[title]将被作为表情的title字段,[extension]可为jpg,png,gif，
在当前gif暂时被当做只包含第一帧的静态图片；
tag将由OCR结果分词、去语气词、去重后得到；每张表情包将由InceptionV3提取特征；
"""
from backend.hanlp_wrapper import HanlpWrapper
from backend.ocr_wrapper import OCRWrapper
from orm import Meme
from backend.feature_extract import InceptionExtractor
from backend.utils import Log
from PIL import Image
from torch.autograd import Variable
from backend.config import *
import numpy as np
import time
import torch
import peewee

if __name__ == '__main__':
    # Init Modules
    hanlp = HanlpWrapper()
    ocr = OCRWrapper('mobilenetv2')
    extractor = InceptionExtractor(allow_img_extensions, 16)

    meme_path = './backend/meme/'
    extractor.init_dataloader(meme_path)
    for k, batched_data in enumerate(extractor.data_loader):
        st_time = time.time()
        m_img = Variable(batched_data['m_img'])  # [batch_size, 3, 299, 299]
        title = batched_data['title']
        meme_path = batched_data['meme_path']
        if torch.cuda.is_available():
            m_img = m_img.cuda()
        v_img = extractor.get_feature(m_img)  # [batch_size, 2048]
        for i in range(len(title)):
            img_data = Image.open(meme_path[i]).convert('RGB')
            img_data = np.array(img_data)
            texts = ocr.text_predict(img_data)
            text = ''.join(texts)
            tags = hanlp.tokenizer(text)
            tags_text = ' '.join(tags)
            feature_vector = v_img[i].cpu().detach()
            feature_encoded = extractor.ndarray2bytes(np.array(feature_vector))
            try:
                Meme.create(path=meme_path[i], title=title[i], tag=tags_text, feature=feature_encoded)
            except peewee.IntegrityError as e:
                Log.info(str(e))
                continue
            Log.info("Import Meme with title:" + title[i] + ", tag:" + tags_text + ', raw text:' + text)
        Log.info("This batch consumes %.2f seconds..." % (time.time()-st_time))


# Preprocess
# preprocess(meme_path)



# st_time = time.time()
# feature_encoded = inception.GetFeature(imgpath)
# img_data = Image.open(imgpath)
# img_data = np.array(img_data)
# texts = ocr.text_predict(img_data)
# text = ''.join(texts)
# tags = hanlp.Tokenizer(text)
# tags_text = ' '.join(tags)
# Meme.create(path=imgpath, title=title, tag=tags_text, feature=feature_encoded)
# Log.info("Import " + imgpath + "in " + str(time.time()-st_time) + ' seconds.')
