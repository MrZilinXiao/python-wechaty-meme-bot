"""
用于手动导入已收集好的表情包进入数据库及提取特征
Usage: python3 ./meme_importer.py [memepath]

[title]/*.[extension]

[title]将被作为表情的title字段,[extension]可为jpg,png,gif，
在当前gif暂时被当做只包含第一帧的静态图片；
tag将由OCR结果分词、去语气词、去重后得到；每张表情包将由InceptionV3提取特征；
"""
from backend.hanlp_wrapper import HanlpWrapper
from orm import Meme, MemeType
from backend.preprocess import preprocess
import sys
import os

# Init NLP
hanlp = HanlpWrapper()
meme_path = sys.argv[1]
img_extensions = ['.jpg', '.png', '.gif', '.jpeg']

# Preprocess
preprocess(meme_path)

for root, dirs, files in os.walk(meme_path, topdown=True):
    if not dirs:  # for each subdir
        title = root.split('/')[-1]
        mType = MemeType.create(title=title)
        for name in files:  # for each meme
            imgpath = os.path.join(root, name)
            if not '.'+name.split('.')[-1] in img_extensions:
                raise ValueError("Only " + str(img_extensions) + " extensions supported for now! ")
