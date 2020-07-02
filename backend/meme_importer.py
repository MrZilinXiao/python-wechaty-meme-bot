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
from orm import Meme, MemeType, db
from backend.feature_extract import InceptionExtractor
from backend.preprocess import preprocess
from backend.utils import Log
from PIL import Image
import sys
import os
import numpy as np
import time

# Init Modules
hanlp = HanlpWrapper()
ocr = OCRWrapper('mobilenetv2')
inception = InceptionExtractor()

meme_path = sys.argv[1]
img_extensions = ['.jpg', '.png', '.gif', '.jpeg']

# Preprocess
# preprocess(meme_path)

for root, dirs, files in os.walk(meme_path, topdown=True):
    if not dirs:  # for each subdir
        title = root.split(os.path.sep)[-1]
        # mType = MemeType.create(title=title)
        for name in files:  # for each meme
            imgpath = os.path.join(root, name)
            if Meme.get_or_none(path=imgpath):
                continue
            st_time = time.time()
            if not '.'+name.split('.')[-1].lower() in img_extensions:
                raise ValueError("Only " + str(img_extensions) + " extensions supported for now! ")
            feature_encoded = inception.GetFeature(imgpath)
            img_data = Image.open(imgpath)
            img_data = np.array(img_data)
            texts = ocr.text_predict(img_data)
            text = ''.join(texts)
            tags = hanlp.Tokenizer(text)
            tags_text = ' '.join(tags)
            Meme.create(path=imgpath, title=title, tag=tags_text, feature=feature_encoded)
            Log.info("Import " + imgpath + "in " + str(time.time()-st_time) + ' seconds.')
